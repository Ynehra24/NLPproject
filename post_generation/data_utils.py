"""
data_utils.py
-------------
Dataset loading, preprocessing, and human-corpus baseline construction.

The human corpus is used exclusively to compute the stylometric reference
distribution P_human(sentence_length), which anchors L_style during training.
"""

from __future__ import annotations

import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence splitter (lightweight — no NLTK dependency required)
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter that avoids heavy NLP dependencies.
    Returns a list of non-empty sentence strings.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Human corpus: build sentence-length histogram
# ---------------------------------------------------------------------------

class HumanCorpusStats:
    """
    Holds the reference stylometric distribution derived from human-written text.

    The primary signal is P_human(k) — the normalised probability that a
    sentence contains exactly k tokens — used to compute the KL term in L_style.

    We also store lexical entropy and mean/std of sentence lengths for
    diagnostic purposes.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_sentence_length: int = 80,
        n_bins: int = 25,
    ):
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length
        self.n_bins = n_bins

        # Will be populated by fit()
        self.length_histogram: Optional[np.ndarray] = None  # shape (n_bins,)
        self.bin_edges: Optional[np.ndarray] = None          # shape (n_bins+1,)
        self.mean_len: float = 0.0
        self.std_len: float = 0.0
        self.lexical_entropy: float = 0.0

        # Raw token-level word-frequency dictionary for lexical entropy
        self._word_freq: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, texts: List[str], sample_size: Optional[int] = None) -> "HumanCorpusStats":
        """
        Fit the human baseline from a list of document strings.

        Args:
            texts:       List of human-authored documents.
            sample_size: If set, randomly sample this many docs before fitting.

        Returns:
            self (for chaining)
        """
        if sample_size and len(texts) > sample_size:
            texts = random.sample(texts, sample_size)

        all_lengths: List[int] = []

        for doc in texts:
            sentences = split_sentences(doc)
            for sent in sentences:
                token_ids = self.tokenizer.encode(
                    sent,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_sentence_length,
                )
                all_lengths.append(len(token_ids))

                # Word-frequency counts for lexical entropy
                words = sent.lower().split()
                for w in words:
                    self._word_freq[w] = self._word_freq.get(w, 0) + 1

        if not all_lengths:
            raise ValueError("No sentences found in human corpus — check the file path.")

        arr = np.array(all_lengths, dtype=np.float32)
        self.mean_len = float(arr.mean())
        self.std_len = float(arr.std())

        # Build normalised histogram
        self.bin_edges = np.linspace(0, self.max_sentence_length, self.n_bins + 1)
        counts, _ = np.histogram(arr, bins=self.bin_edges)
        counts = counts.astype(np.float64) + 1e-8   # Laplace smoothing
        self.length_histogram = counts / counts.sum()

        # Lexical entropy H = -sum(p log p)
        total = sum(self._word_freq.values())
        self.lexical_entropy = float(
            -sum((c / total) * np.log(c / total + 1e-12) for c in self._word_freq.values())
        )

        logger.info(
            "HumanCorpusStats fitted on %d sentences | "
            "mean_len=%.1f  std_len=%.1f  lexical_H=%.3f",
            len(all_lengths), self.mean_len, self.std_len, self.lexical_entropy,
        )
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise to JSON so we don't recompute every run."""
        data = {
            "mean_len": self.mean_len,
            "std_len": self.std_len,
            "lexical_entropy": self.lexical_entropy,
            "max_sentence_length": self.max_sentence_length,
            "n_bins": self.n_bins,
            "length_histogram": self.length_histogram.tolist(),
            "bin_edges": self.bin_edges.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("HumanCorpusStats saved to %s", path)

    @classmethod
    def load(cls, path: str, tokenizer: PreTrainedTokenizer) -> "HumanCorpusStats":
        """Load previously saved stats."""
        with open(path) as f:
            data = json.load(f)
        obj = cls(
            tokenizer=tokenizer,
            max_sentence_length=data["max_sentence_length"],
            n_bins=data["n_bins"],
        )
        obj.mean_len = data["mean_len"]
        obj.std_len = data["std_len"]
        obj.lexical_entropy = data["lexical_entropy"]
        obj.length_histogram = np.array(data["length_histogram"])
        obj.bin_edges = np.array(data["bin_edges"])
        logger.info("HumanCorpusStats loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Torch tensor helper
    # ------------------------------------------------------------------

    def get_length_dist_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Return the normalised length histogram as a float32 tensor.
        Shape: (n_bins,)
        """
        assert self.length_histogram is not None, "Call fit() first."
        return torch.tensor(self.length_histogram, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AITextDataset(Dataset):
    """
    Simple line-by-line dataset for AI-generated text.

    Each item is a dict with 'input_ids' and 'attention_mask' ready for
    the evader's encoder.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        skip_empty: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            raw = [line.strip() for line in f]

        if skip_empty:
            raw = [t for t in raw if len(t) > 20]

        self.texts = raw
        logger.info("AITextDataset loaded %d samples from %s", len(self.texts), file_path)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "raw_text": text,
        }


def build_dataloader(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Convenience wrapper that constructs a DataLoader from a text file."""
    ds = AITextDataset(file_path, tokenizer, max_length=max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Human corpus loader
# ---------------------------------------------------------------------------

def load_human_corpus(path: str) -> List[str]:
    """
    Load human-authored documents from a plain-text file (one doc per line).
    Filters out very short lines that are likely headers/metadata.
    """
    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if len(line.strip()) > 50]
    logger.info("Loaded %d human documents from %s", len(docs), path)
    return docs


def build_or_load_human_stats(
    human_corpus_path: str,
    cache_path: str,
    tokenizer: PreTrainedTokenizer,
    max_sentence_length: int = 80,
    n_bins: int = 25,
    sample_size: Optional[int] = 10_000,
    force_rebuild: bool = False,
) -> HumanCorpusStats:
    """
    Either load cached stats or re-fit from the raw human corpus.

    Args:
        human_corpus_path: Plain-text file of human documents.
        cache_path:         JSON path to save/load pre-computed stats.
        tokenizer:          Evader tokenizer (used for tokenising sentences).
        force_rebuild:      Re-fit even if cache exists.

    Returns:
        Fitted HumanCorpusStats instance.
    """
    cache = Path(cache_path)
    if cache.exists() and not force_rebuild:
        return HumanCorpusStats.load(str(cache), tokenizer)

    docs = load_human_corpus(human_corpus_path)
    stats = HumanCorpusStats(
        tokenizer, max_sentence_length=max_sentence_length, n_bins=n_bins
    )
    stats.fit(docs, sample_size=sample_size)
    stats.save(str(cache))
    return stats
