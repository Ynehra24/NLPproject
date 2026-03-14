"""Text perturbation helpers for DetectGPT-style scoring."""

from __future__ import annotations

import random
import re
from typing import List

from transformers import pipeline


class Perturber:
    def __init__(self, mask_model: str = "distilroberta-base", device: int = -1, seed: int = 42) -> None:
        self.fill_mask = pipeline("fill-mask", model=mask_model, tokenizer=mask_model, device=device)
        self.mask_token = self.fill_mask.tokenizer.mask_token
        self.rng = random.Random(seed)

    def _mask_random_word(self, text: str) -> str:
        words = re.findall(r"\S+", text)
        if len(words) < 3:
            return text
        idx = self.rng.randint(0, len(words) - 1)
        words[idx] = self.mask_token
        return " ".join(words)

    def perturb_once(self, text: str) -> str:
        masked = self._mask_random_word(text)
        if self.mask_token not in masked:
            return text
        try:
            pred = self.fill_mask(masked, top_k=1)
            if isinstance(pred, list) and pred:
                return pred[0]["sequence"]
        except Exception:
            return text
        return text

    def generate(self, text: str, n_perturbations: int) -> List[str]:
        return [self.perturb_once(text) for _ in range(n_perturbations)]
