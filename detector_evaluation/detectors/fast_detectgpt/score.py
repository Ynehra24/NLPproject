"""Fast-DetectGPT detector implementation.

This approximates curvature-based detection without expensive perturbation sampling
by comparing token-level log probabilities from scoring and reference models.
"""

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


class FastDetectGPTDetector:
    def __init__(
        self,
        scoring_model: str = "gpt2-medium",
        reference_model: str = "gpt2",
        device: str = "cpu",
    ) -> None:
        self.detector_name = "fast_detectgpt"
        self.device = torch.device(device)

        self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model)
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model).to(self.device)
        self.scoring_model.eval()

        self.reference_tokenizer = AutoTokenizer.from_pretrained(reference_model)
        if self.reference_tokenizer.pad_token is None:
            self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
        self.reference_model = AutoModelForCausalLM.from_pretrained(reference_model).to(self.device)
        self.reference_model.eval()

    @torch.inference_mode()
    def _token_log_probs(self, text: str, model, tokenizer) -> np.ndarray:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return np.array([0.0], dtype=float)

        logits = model(**enc).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        tok_lps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
        return tok_lps.detach().cpu().numpy().astype(float)

    def _discrepancy(self, text: str) -> float:
        score_lp = self._token_log_probs(text, self.scoring_model, self.scoring_tokenizer)
        ref_lp = self._token_log_probs(text, self.reference_model, self.reference_tokenizer)

        min_len = min(len(score_lp), len(ref_lp))
        if min_len < 2:
            return 0.0

        diff = score_lp[:min_len] - ref_lp[:min_len]
        return float(np.mean(diff) / (np.std(diff) + 1e-9))

    def score_texts(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        raw = []
        for text in tqdm(texts, desc="Scoring Fast-DetectGPT"):
            raw.append(self._discrepancy(text))

        raw = np.array(raw, dtype=float)
        ai_scores = 1.0 / (1.0 + np.exp(-raw))
        return ai_scores, raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fast-DetectGPT detector")
    parser.add_argument("--input", required=True, help="Path to input csv/jsonl")
    parser.add_argument("--output", required=True, help="Path to output csv")
    parser.add_argument("--scoring-model", default="gpt2-medium")
    parser.add_argument("--reference-model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    detector = FastDetectGPTDetector(
        scoring_model=args.scoring_model,
        reference_model=args.reference_model,
        device=args.device,
    )
    ai_scores, raw_scores = detector.score_texts(df["text"].tolist())

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
            "raw_discrepancy": raw_scores,
            "attack_type": df["attack_type"],
            "attack_owner": df["attack_owner"],
            "source": df.get("source"),
        }
    )

    save_detector_scores(out, args.output)
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
