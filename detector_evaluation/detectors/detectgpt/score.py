"""DetectGPT-style curvature detector implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from detectors.common.io_utils import load_dataset, save_detector_scores
from detectors.common.metrics import encode_source_labels, find_best_threshold
from detectors.detectgpt.perturb import Perturber


class DetectGPTDetector:
    def __init__(
        self,
        lm_name: str = "gpt2",
        mask_model: str = "distilroberta-base",
        device: str = "cpu",
    ) -> None:
        self.detector_name = "detectgpt_style"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(lm_name).to(self.device)
        self.model.eval()

        pipeline_device = 0 if device.startswith("cuda") else -1
        self.perturber = Perturber(mask_model=mask_model, device=pipeline_device)

    @torch.inference_mode()
    def log_prob(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        out = self.model(**enc, labels=labels)
        seq_len = int(labels.shape[1])
        return float(-out.loss.detach().cpu().item() * seq_len)

    def score_texts(self, texts: list[str], n_perturbations: int = 8) -> tuple[np.ndarray, np.ndarray]:
        raw_scores = []
        base_lps = []
        for text in tqdm(texts, desc="Scoring DetectGPT-style"):
            base = self.log_prob(text)
            perturbed_texts = self.perturber.generate(text, n_perturbations=n_perturbations)
            pert_lps = [self.log_prob(pt) for pt in perturbed_texts]
            score = base - float(np.mean(pert_lps))
            raw_scores.append(score)
            base_lps.append(base)

        raw_scores = np.array(raw_scores, dtype=float)
        ai_scores = 1.0 / (1.0 + np.exp(-(raw_scores - raw_scores.mean()) / (raw_scores.std() + 1e-9)))
        return ai_scores, np.array(base_lps, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DetectGPT-style detector")
    parser.add_argument("--input", required=True, help="Path to input csv/jsonl")
    parser.add_argument("--output", required=True, help="Path to output csv")
    parser.add_argument("--lm-name", default="gpt2")
    parser.add_argument("--mask-model", default="distilroberta-base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-perturb", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    detector = DetectGPTDetector(
        lm_name=args.lm_name,
        mask_model=args.mask_model,
        device=args.device,
    )
    ai_scores, base_lps = detector.score_texts(df["text"].tolist(), n_perturbations=args.n_perturb)

    threshold = args.threshold
    if threshold is None and "source" in df.columns and df["source"].notna().all():
        y_true = encode_source_labels(df["source"])
        threshold = find_best_threshold(y_true, ai_scores)
    if threshold is None:
        threshold = 0.5

    out = pd.DataFrame(
        {
            "id": df["id"],
            "detector_name": detector.detector_name,
            "ai_score": ai_scores,
            "predicted_label": np.where(ai_scores >= threshold, "ai", "human"),
            "threshold_used": float(threshold),
            "base_logprob": base_lps,
            "attack_type": df["attack_type"],
            "attack_owner": df["attack_owner"],
            "source": df.get("source"),
        }
    )

    save_detector_scores(out, args.output)
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
