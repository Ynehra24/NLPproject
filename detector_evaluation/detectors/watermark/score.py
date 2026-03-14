"""KGW-style watermark detector with optional watermarked generation helper.

Detector uses green-token z-score under a seeded vocabulary partitioning scheme.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from detectors.common.io_utils import load_dataset, save_detector_scores
from detectors.common.metrics import encode_source_labels, find_best_threshold


class WatermarkDetector:
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        gamma: float = 0.5,
        hash_key: int = 15485863,
    ) -> None:
        self.detector_name = "kgw_watermark"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.gamma = float(gamma)
        self.hash_key = int(hash_key)

    def _green_list(self, prev_token_id: int) -> set[int]:
        seed_bytes = f"{self.hash_key}_{prev_token_id}".encode("utf-8")
        seed = int(hashlib.sha256(seed_bytes).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        n_green = int(self.vocab_size * self.gamma)
        return set(rng.choice(self.vocab_size, size=n_green, replace=False).tolist())

    def z_score(self, text: str) -> float:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 2:
            return 0.0

        green_hits = 0
        total = 0
        for i in range(1, len(token_ids)):
            prev_tid = int(token_ids[i - 1])
            curr_tid = int(token_ids[i])
            if curr_tid in self._green_list(prev_tid):
                green_hits += 1
            total += 1

        if total == 0:
            return 0.0

        expected = self.gamma * total
        std = float(np.sqrt(total * self.gamma * (1.0 - self.gamma)))
        if std < 1e-9:
            return 0.0
        return float((green_hits - expected) / std)

    def score_texts(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        z_scores = []
        for text in tqdm(texts, desc="Scoring KGW watermark"):
            z_scores.append(self.z_score(text))

        z_scores = np.array(z_scores, dtype=float)
        ai_scores = 1.0 / (1.0 + np.exp(-z_scores))
        return ai_scores, z_scores


class WatermarkGenerator:
    def __init__(
        self,
        model_name: str = "gpt2",
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.hash_key = int(hash_key)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.vocab_size = int(self.tokenizer.vocab_size)

    def _green_list(self, prev_token_id: int) -> set[int]:
        seed_bytes = f"{self.hash_key}_{prev_token_id}".encode("utf-8")
        seed = int(hashlib.sha256(seed_bytes).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        n_green = int(self.vocab_size * self.gamma)
        return set(rng.choice(self.vocab_size, size=n_green, replace=False).tolist())

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        for _ in range(max_new_tokens):
            logits = self.model(input_ids).logits[:, -1, :]
            prev_tid = int(input_ids[0, -1].item())
            greens = self._green_list(prev_tid)

            bias = torch.zeros(self.vocab_size, device=self.device)
            if greens:
                green_idx = torch.tensor(sorted(greens), dtype=torch.long, device=self.device)
                bias[green_idx] = self.delta

            step_logits = (logits + bias.unsqueeze(0)) / max(temperature, 1e-6)
            probs = torch.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            eos = self.tokenizer.eos_token_id
            if eos is not None and int(next_token.item()) == int(eos):
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KGW watermark detector")
    parser.add_argument("--input", required=True, help="Path to input csv/jsonl")
    parser.add_argument("--output", required=True, help="Path to output csv")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--hash-key", type=int, default=15485863)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    detector = WatermarkDetector(
        tokenizer_name=args.tokenizer_name,
        gamma=args.gamma,
        hash_key=args.hash_key,
    )
    ai_scores, z_scores = detector.score_texts(df["text"].tolist())

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
            "watermark_z": z_scores,
            "attack_type": df["attack_type"],
            "attack_owner": df["attack_owner"],
            "source": df.get("source"),
        }
    )

    save_detector_scores(out, args.output)
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
