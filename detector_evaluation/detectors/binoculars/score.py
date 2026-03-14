"""Binoculars detector implementation.

Computes cross-entropy ratio between observer and performer models.
Lower ratio implies more AI-like text.
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


class BinocularsDetector:
    def __init__(
        self,
        observer_model: str = "gpt2-medium",
        performer_model: str = "gpt2",
        device: str = "cpu",
    ) -> None:
        self.detector_name = "binoculars"
        self.device = torch.device(device)

        self.observer_tokenizer = AutoTokenizer.from_pretrained(observer_model)
        if self.observer_tokenizer.pad_token is None:
            self.observer_tokenizer.pad_token = self.observer_tokenizer.eos_token
        self.observer = AutoModelForCausalLM.from_pretrained(observer_model).to(self.device)
        self.observer.eval()

        self.performer_tokenizer = AutoTokenizer.from_pretrained(performer_model)
        if self.performer_tokenizer.pad_token is None:
            self.performer_tokenizer.pad_token = self.performer_tokenizer.eos_token
        self.performer = AutoModelForCausalLM.from_pretrained(performer_model).to(self.device)
        self.performer.eval()

    @torch.inference_mode()
    def _cross_entropy(self, text: str, model, tokenizer) -> float:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return 1.0

        logits = model(**enc).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return float(-token_log_probs.mean().detach().cpu().item())

    def _ratio(self, text: str) -> float:
        ce_obs = self._cross_entropy(text, self.observer, self.observer_tokenizer)
        ce_perf = self._cross_entropy(text, self.performer, self.performer_tokenizer)
        return float(ce_obs / (ce_perf + 1e-9))

    def score_texts(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        ratios = []
        for text in tqdm(texts, desc="Scoring Binoculars"):
            ratios.append(self._ratio(text))

        ratios = np.array(ratios, dtype=float)
        # Invert and scale for AI-likelihood compatibility.
        centered = -(ratios - np.median(ratios)) * 10.0
        ai_scores = 1.0 / (1.0 + np.exp(-centered))
        return ai_scores, ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Binoculars detector")
    parser.add_argument("--input", required=True, help="Path to input csv/jsonl")
    parser.add_argument("--output", required=True, help="Path to output csv")
    parser.add_argument("--observer-model", default="gpt2-medium")
    parser.add_argument("--performer-model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    detector = BinocularsDetector(
        observer_model=args.observer_model,
        performer_model=args.performer_model,
        device=args.device,
    )
    ai_scores, ratios = detector.score_texts(df["text"].tolist())

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
            "binoculars_ratio": ratios,
            "attack_type": df["attack_type"],
            "attack_owner": df["attack_owner"],
            "source": df.get("source"),
        }
    )

    save_detector_scores(out, args.output)
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
