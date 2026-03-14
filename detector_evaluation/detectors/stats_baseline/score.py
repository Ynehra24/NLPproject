"""Perplexity and rank-based detector baseline."""

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


class StatsBaselineDetector:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu") -> None:
        self.detector_name = "stats_baseline"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _ppl_and_rank(self, text: str) -> tuple[float, float]:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        labels = enc["input_ids"].clone()
        out = self.model(**enc, labels=labels)
        ppl = float(torch.exp(out.loss).detach().cpu().item())

        logits = out.logits[:, :-1, :]
        targets = labels[:, 1:]
        token_logits = logits[0]
        token_targets = targets[0]

        sorted_indices = torch.argsort(token_logits, dim=-1, descending=True)
        ranks = []
        for i in range(token_targets.shape[0]):
            target_id = token_targets[i]
            rank = (sorted_indices[i] == target_id).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

        avg_rank = float(np.mean(ranks)) if ranks else 1.0
        return ppl, avg_rank

    def score_texts(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ppls = []
        ranks = []
        for text in tqdm(texts, desc="Scoring stats baseline"):
            ppl, rank = self._ppl_and_rank(text)
            ppls.append(ppl)
            ranks.append(rank)

        ppls = np.array(ppls, dtype=float)
        ranks = np.array(ranks, dtype=float)

        ppl_z = (ppls - ppls.mean()) / (ppls.std() + 1e-9)
        rank_z = (ranks - ranks.mean()) / (ranks.std() + 1e-9)

        # Low perplexity and low token-rank usually correlate with machine text.
        ai_raw = (-0.7 * ppl_z) + (-0.3 * rank_z)
        ai_score = 1.0 / (1.0 + np.exp(-ai_raw))
        return ai_score, ppls, ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stats baseline detector")
    parser.add_argument("--input", required=True, help="Path to input csv/jsonl")
    parser.add_argument("--output", required=True, help="Path to output scores csv")
    parser.add_argument("--model-name", default="gpt2", help="Causal LM for perplexity/rank")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--threshold", type=float, default=None, help="Manual threshold if provided")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    detector = StatsBaselineDetector(model_name=args.model_name, device=args.device)
    ai_score, ppls, ranks = detector.score_texts(df["text"].tolist())

    threshold = args.threshold
    if threshold is None and "source" in df.columns and df["source"].notna().all():
        y_true = encode_source_labels(df["source"])
        threshold = find_best_threshold(y_true, ai_score)
    if threshold is None:
        threshold = 0.5

    out = pd.DataFrame(
        {
            "id": df["id"],
            "detector_name": detector.detector_name,
            "ai_score": ai_score,
            "predicted_label": np.where(ai_score >= threshold, "ai", "human"),
            "threshold_used": float(threshold),
            "ppl": ppls,
            "avg_rank": ranks,
            "attack_type": df["attack_type"],
            "attack_owner": df["attack_owner"],
            "source": df.get("source"),
        }
    )

    save_detector_scores(out, args.output)
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
