import os
import importlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

Constraint = importlib.import_module("textattack.constraints.constraint").Constraint


class CrossEncoderSemanticSimilarity(Constraint):
    """Semantic constraint backed by a Hugging Face cross-encoder.

    The default model (`cross-encoder/stsb-roberta-large`) outputs STS-B scores in
    the [0, 5] range. This class rescales scores to [0, 1] before applying the
    threshold to match TextAttack-style similarity constraints.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        model_name: str = "cross-encoder/stsb-roberta-large",
        compare_against_original: bool = True,
        window_size: Optional[int] = 50,
        skip_text_shorter_than_window: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
        stsb_max_score: float = 5.0,
        device: Optional[str] = None,
    ):
        super().__init__(compare_against_original)
        self.threshold = threshold
        self.model_name = model_name
        self.window_size = window_size if window_size else float("inf")
        self.skip_text_shorter_than_window = skip_text_shorter_than_window
        self.batch_size = batch_size
        self.max_length = max_length
        self.stsb_max_score = stsb_max_score

        requested_device = device or os.environ.get("TA_DEVICE")
        if requested_device and requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None

    def _lazy_load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _get_window_pair(self, reference_text, transformed_text) -> Tuple[str, str]:
        if self.window_size == float("inf"):
            return reference_text.text, transformed_text.text

        try:
            modified_indices = transformed_text.attack_attrs["newly_modified_indices"]
            if len(modified_indices) == 0:
                raise KeyError("`newly_modified_indices` is empty")
            sorted_indices = sorted(modified_indices)
            modified_index = sorted_indices[len(sorted_indices) // 2]
        except KeyError as exc:
            raise KeyError(
                "Cannot apply cross-encoder constraint without `newly_modified_indices`"
            ) from exc

        reference_window = reference_text.text_window_around_index(modified_index, self.window_size)
        transformed_window = transformed_text.text_window_around_index(modified_index, self.window_size)
        return reference_window, transformed_window

    def _score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> torch.Tensor:
        if not pairs:
            return torch.tensor([])

        self._lazy_load_model()

        all_scores: List[torch.Tensor] = []
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            references = [pair[0] for pair in batch_pairs]
            candidates = [pair[1] for pair in batch_pairs]

            encoded = self.tokenizer(
                references,
                candidates,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.inference_mode():
                logits = self.model(**encoded).logits

            all_scores.append(self._normalize_scores(logits).detach().cpu())

        return torch.cat(all_scores, dim=0)

    def warmup(self) -> None:
        self._lazy_load_model()

    def score_text_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        scores = self._score_pairs(pairs)
        return scores.tolist()

    def _normalize_scores(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1:
            raw_scores = logits
            return torch.clamp(raw_scores / self.stsb_max_score, 0.0, 1.0)

        if logits.shape[-1] == 1:
            raw_scores = logits.squeeze(-1)
            return torch.clamp(raw_scores / self.stsb_max_score, 0.0, 1.0)

        probs = torch.softmax(logits, dim=-1)
        return torch.clamp(probs[:, -1], 0.0, 1.0)

    def _check_constraint_many(self, transformed_texts, reference_text):
        if len(transformed_texts) == 0:
            return np.array([])

        scores = torch.ones(len(transformed_texts), dtype=torch.float32)
        pair_indices = []
        pairs = []

        for idx, transformed_text in enumerate(transformed_texts):
            if (
                self.skip_text_shorter_than_window
                and self.window_size != float("inf")
                and len(transformed_text.words) < self.window_size
            ):
                transformed_text.attack_attrs["similarity_score"] = 1.0
                continue

            pairs.append(self._get_window_pair(reference_text, transformed_text))
            pair_indices.append(idx)

        if pairs:
            pair_scores = self._score_pairs(pairs)
            for pair_idx, score in zip(pair_indices, pair_scores):
                score_item = score.item()
                transformed_texts[pair_idx].attack_attrs["similarity_score"] = score_item
                scores[pair_idx] = score

        mask = (scores >= self.threshold).cpu().numpy().nonzero()
        return np.array(transformed_texts)[mask]

    def _check_constraint(self, transformed_text, reference_text):
        if (
            self.skip_text_shorter_than_window
            and self.window_size != float("inf")
            and len(transformed_text.words) < self.window_size
        ):
            score = 1.0
        else:
            pair = self._get_window_pair(reference_text, transformed_text)
            score = self._score_pairs([pair])[0].item()

        transformed_text.attack_attrs["similarity_score"] = score
        return score >= self.threshold

    def extra_repr_keys(self):
        return [
            "model_name",
            "threshold",
            "window_size",
            "batch_size",
            "max_length",
            "stsb_max_score",
            "skip_text_shorter_than_window",
        ] + super().extra_repr_keys()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tokenizer"] = None
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.tokenizer = None
        self.model = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def build_semantic_constraint(
    threshold: float = 0.75,
    compare_against_original: bool = True,
    window_size: int = 50,
) -> CrossEncoderSemanticSimilarity:
    """Builds the Phase-1 semantic gate with optional env overrides.

    Supported environment variables:
    - HMGC_SEMANTIC_MODEL
    - HMGC_SEMANTIC_THRESHOLD
    - HMGC_SEMANTIC_WINDOW_SIZE
    - HMGC_SEMANTIC_BATCH_SIZE
    - HMGC_SEMANTIC_MAX_LENGTH
    - HMGC_SEMANTIC_STSB_MAX_SCORE
    - HMGC_SEMANTIC_SKIP_SHORT
    - HMGC_SEMANTIC_DEVICE
    """

    return CrossEncoderSemanticSimilarity(
        model_name=os.environ.get("HMGC_SEMANTIC_MODEL", "cross-encoder/stsb-roberta-large"),
        threshold=_env_float("HMGC_SEMANTIC_THRESHOLD", threshold),
        compare_against_original=compare_against_original,
        window_size=_env_int("HMGC_SEMANTIC_WINDOW_SIZE", window_size),
        skip_text_shorter_than_window=_env_bool("HMGC_SEMANTIC_SKIP_SHORT", False),
        batch_size=_env_int("HMGC_SEMANTIC_BATCH_SIZE", 16),
        max_length=_env_int("HMGC_SEMANTIC_MAX_LENGTH", 512),
        stsb_max_score=_env_float("HMGC_SEMANTIC_STSB_MAX_SCORE", 5.0),
        device=os.environ.get("HMGC_SEMANTIC_DEVICE", os.environ.get("TA_DEVICE")),
    )
