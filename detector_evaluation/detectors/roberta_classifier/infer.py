"""Run inference with a fine-tuned RoBERTa detector model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detectors.common.io_utils import load_dataset, save_detector_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoBERTa detector inference")
    parser.add_argument("--input", required=True, help="Input csv/jsonl")
    parser.add_argument("--model-dir", required=True, help="Fine-tuned model dir")
    parser.add_argument("--output", required=True, help="Output csv path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def load_threshold(model_dir: Path, fallback: float = 0.5) -> float:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return fallback
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return float(metrics.get("best_threshold", fallback))


@torch.inference_mode()
def batched_scores(texts, tokenizer, model, batch_size: int, device: torch.device):
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa inference"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        scores.extend(probs.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=float)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    model_dir = Path(args.model_dir)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    ai_scores = batched_scores(df["text"].astype(str).tolist(), tokenizer, model, args.batch_size, device)

    threshold = args.threshold if args.threshold is not None else load_threshold(model_dir)

    out = pd.DataFrame(
        {
            "id": df["id"],
            "detector_name": "roberta_classifier",
            "ai_score": ai_scores,
            "predicted_label": np.where(ai_scores >= threshold, "ai", "human"),
            "threshold_used": float(threshold),
            "attack_type": df["attack_type"],
            "attack_owner": df["attack_owner"],
            "source": df.get("source"),
        }
    )

    save_detector_scores(out, args.output)
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
