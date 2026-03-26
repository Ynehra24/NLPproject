"""
pseudo_embeddings.py
--------------------
Implements the pseudo-embedding technique from GradEscape (Meng et al., 2025).

WHY THIS IS NEEDED
==================
Gradient-based adversarial attacks are straightforward in vision (pixel space is
continuous), but NLP text is inherently discrete: you cannot differentiate through
a token sampling step.

THE SOLUTION
============
Instead of sampling tokens, we intercept the evader's softmax output — a
continuous probability matrix P ∈ R^{batch × seq_len × vocab_size} — and compute:

    pseudo_emb = P  @  W_emb

where W_emb ∈ R^{vocab_size × hidden_dim} is the detector's frozen word-embedding
weight matrix.  pseudo_emb is a weighted sum of word embeddings (still continuous)
that can be fed directly into the detector, bypassing its tokeniser lookup.

This preserves differentiability end-to-end:
    evader logits → softmax → P → pseudo_emb → frozen detector → L_adv → ∂/∂θ_evader

Reference: Section 4.2, Meng et al. (2025).  The "token re-mapping" and
"warm-started evader" extensions for cross-tokenizer scenarios are also implemented.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    RobertaForSequenceClassification,
    BertForSequenceClassification,
    GPT2ForSequenceClassification,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture-agnostic embedding extractor
# ---------------------------------------------------------------------------

def get_word_embedding_weight(model: PreTrainedModel) -> nn.Parameter:
    """
    Return the word-embedding weight matrix from any supported detector.

    Supported architectures: RoBERTa, BERT, GPT-2 (sequence classification).
    Returns a view into the model's Parameter — NOT a copy — so
    pseudo_emb = P @ weight is always using the live (frozen) weights.
    """
    if isinstance(model, RobertaForSequenceClassification):
        return model.roberta.embeddings.word_embeddings.weight
    if isinstance(model, BertForSequenceClassification):
        return model.bert.embeddings.word_embeddings.weight
    if isinstance(model, GPT2ForSequenceClassification):
        return model.transformer.wte.weight
    # Generic fallback — walk named modules looking for the first Embedding layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "position" not in name:
            logger.warning(
                "Using generic embedding extraction from module '%s'. "
                "Verify this is the word-embedding layer.", name
            )
            return module.weight
    raise ValueError(
        "Cannot locate word-embedding weight in detector. "
        "Add an explicit branch in get_word_embedding_weight()."
    )


# ---------------------------------------------------------------------------
# Probability matrix post-processing
# ---------------------------------------------------------------------------

def post_process_prob_matrix(
    prob_matrix: torch.Tensor,
    evader_vocab_size: int,
    detector_vocab_size: int,
    unk_id_in_detector: int = 3,
) -> torch.Tensor:
    """
    Align the evader's probability matrix to the detector's vocabulary.

    When evader and detector share the same tokenizer (e.g., BART + RoBERTa)
    this is effectively a no-op because their vocabularies are identical.

    For mismatched vocabularies (e.g., BART + BERT) we:
      1. Truncate or zero-pad to match detector vocab size.
      2. Redistribute probability mass of out-of-vocabulary tokens to <unk>.

    Args:
        prob_matrix:          (batch, seq_len, evader_vocab_size)
        evader_vocab_size:    int
        detector_vocab_size:  int
        unk_id_in_detector:   Token ID of <unk> in the detector vocab.

    Returns:
        Aligned prob_matrix: (batch, seq_len, detector_vocab_size)
    """
    if evader_vocab_size == detector_vocab_size:
        return prob_matrix   # fast path — same tokenizer

    B, L, V_e = prob_matrix.shape
    V_d = detector_vocab_size

    if V_e > V_d:
        # Tokens beyond detector vocab → add their mass to <unk>
        excess_mass = prob_matrix[:, :, V_d:].sum(dim=-1, keepdim=True)   # (B, L, 1)
        aligned = prob_matrix[:, :, :V_d].clone()
        aligned[:, :, unk_id_in_detector] = (
            aligned[:, :, unk_id_in_detector] + excess_mass.squeeze(-1)
        )
    else:
        # Zero-pad missing token slots
        padding = torch.zeros(
            B, L, V_d - V_e,
            dtype=prob_matrix.dtype,
            device=prob_matrix.device,
        )
        aligned = torch.cat([prob_matrix, padding], dim=-1)

    return aligned


# ---------------------------------------------------------------------------
# Main pseudo-embedding module
# ---------------------------------------------------------------------------

class PseudoEmbeddingInjector(nn.Module):
    """
    Wrapper around a FROZEN detector that accepts continuous probability
    matrices as input rather than discrete token IDs.

    Usage
    -----
    injector = PseudoEmbeddingInjector(roberta_classifier)

    # prob_matrix: softmax output of the evader (batch, seq_len, vocab_size)
    logits = injector(prob_matrix, attention_mask)
    loss   = F.cross_entropy(logits, human_labels)
    loss.backward()  # gradient flows back through prob_matrix → evader
    """

    def __init__(
        self,
        detector: PreTrainedModel,
        evader_vocab_size: Optional[int] = None,
        unk_id_in_detector: int = 3,
    ):
        super().__init__()
        self.detector = detector

        # Freeze ALL detector parameters — we never update them.
        for param in self.detector.parameters():
            param.requires_grad_(False)
        self.detector.eval()

        self.evader_vocab_size = evader_vocab_size
        self.unk_id_in_detector = unk_id_in_detector

        # Cache embedding weight reference (not a copy)
        self._emb_weight = get_word_embedding_weight(detector)
        logger.info(
            "PseudoEmbeddingInjector ready | detector=%s | emb_dim=%d",
            type(detector).__name__,
            self._emb_weight.shape[-1],
        )

    # ------------------------------------------------------------------
    # Core: P @ W_emb
    # ------------------------------------------------------------------

    def compute_pseudo_embeddings(
        self, prob_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Weighted combination of detector word-embeddings.

        Args:
            prob_matrix: (batch, seq_len, vocab_size) — softmax probabilities.

        Returns:
            pseudo_emb: (batch, seq_len, hidden_dim)
        """
        # Align vocab sizes if needed
        if self.evader_vocab_size is not None:
            det_vocab = self._emb_weight.shape[0]
            if self.evader_vocab_size != det_vocab:
                prob_matrix = post_process_prob_matrix(
                    prob_matrix,
                    self.evader_vocab_size,
                    det_vocab,
                    self.unk_id_in_detector,
                )

        # (batch, seq_len, vocab) @ (vocab, hidden) → (batch, seq_len, hidden)
        pseudo_emb = torch.matmul(prob_matrix, self._emb_weight)
        return pseudo_emb

    # ------------------------------------------------------------------
    # Forward: inject → add positional / type embeddings → classifier
    # ------------------------------------------------------------------

    def forward(
        self,
        prob_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prob_matrix:    (batch, seq_len, vocab_size) continuous probs.
            attention_mask: (batch, seq_len) mask (1 = real token, 0 = pad).

        Returns:
            logits: (batch, num_labels)  — raw classifier output.
        """
        pseudo_emb = self.compute_pseudo_embeddings(prob_matrix)   # (B, L, H)

        # HuggingFace models accept `inputs_embeds` directly, which bypasses
        # the word-embedding lookup but still runs positional & type embeddings.
        outputs = self.detector(
            inputs_embeds=pseudo_emb,
            attention_mask=attention_mask,
        )
        return outputs.logits

    # ------------------------------------------------------------------
    # Convenience: standard token-ID forward (for normal evaluation)
    # ------------------------------------------------------------------

    def forward_from_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard classification forward using discrete token IDs."""
        outputs = self.detector(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


# ---------------------------------------------------------------------------
# Special-token mask for BART-style sequences
# ---------------------------------------------------------------------------

def mask_special_token_rows(
    prob_matrix: torch.Tensor,
    special_token_ids: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    Zero out rows in prob_matrix that correspond to special tokens
    (BOS, EOS, PAD) so they don't pollute the pseudo-embedding computation.

    Args:
        prob_matrix:       (batch, seq_len, vocab_size)
        special_token_ids: (batch, seq_len) bool or 0/1 mask, 1 = special token
        fill_value:        Value to set special-token rows to (default 0).

    Returns:
        prob_matrix with special-token rows zeroed.
    """
    mask = special_token_ids.unsqueeze(-1).float()   # (B, L, 1)
    return prob_matrix * (1.0 - mask) + fill_value * mask
