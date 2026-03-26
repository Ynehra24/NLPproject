"""
stylometric_loss.py
-------------------
Implements L_style — the novel stylometric constraint that distinguishes this
work from GradEscape and HMGC.

MOTIVATION
==========
Prior evaders (GradEscape, HMGC) optimise purely for misclassification and
local token-level fluency.  They ignore *macroscopic* document structure,
which is why their adversarial text still exhibits tells like:
  • Suspiciously uniform sentence lengths (low variance / low burstiness)
  • Lower lexical diversity than typical human writing

These macroscopic signals are exactly what statistical zero-shot detectors
(DetectGPT, GLTR) and future neural detectors trained on diverse corpora
exploit.  By explicitly penalising deviations from a human-corpus baseline,
L_style pushes the adversarial text toward the interior of the human manifold
rather than just past a single classifier's decision boundary.

IMPLEMENTATION APPROACH
=======================
Computing sentence lengths exactly requires discrete segmentation (non-differentiable).
We use a two-path approach:

  Path A — Differentiable soft path (used for backpropagation):
    We treat sentence-ending token probabilities as "soft boundary" indicators.
    A running soft-counter accumulates expected token counts between boundaries.
    The resulting soft histogram is differentiable w.r.t. the evader parameters.

  Path B — Non-differentiable discrete path (used for logging / REINFORCE signal):
    Decode the top-1 sequence, split on actual punctuation, measure exact lengths.
    Used only for diagnostic reporting and optionally as a reward term.

The primary gradient signal comes from Path A.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve sentence-end token IDs
# ---------------------------------------------------------------------------

def resolve_sentence_end_ids(
    tokenizer: PreTrainedTokenizer,
    sentence_end_strings: List[str],
) -> List[int]:
    """
    Convert sentence-ending strings to token IDs in the given tokenizer vocab.
    Only includes IDs that exist as single tokens (avoids multi-token mismatches).
    """
    ids: List[int] = []
    for s in sentence_end_strings:
        encoded = tokenizer.encode(s, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
        else:
            # Try with leading space (BPE tokenizers like RoBERTa/BART)
            encoded_sp = tokenizer.encode(" " + s, add_special_tokens=False)
            for tid in encoded_sp:
                if tid not in ids:
                    ids.append(tid)
    ids = list(set(ids))
    logger.debug("Sentence-end token IDs: %s", ids)
    return ids


# ---------------------------------------------------------------------------
# Soft sentence-length histogram  (Path A — differentiable)
# ---------------------------------------------------------------------------

def soft_sentence_length_histogram(
    prob_matrix: torch.Tensor,
    sentence_end_ids: List[int],
    n_bins: int,
    max_length: int,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Construct a differentiable approximation of the sentence-length histogram.

    Algorithm
    ---------
    At each sequence position t, the "soft boundary probability" p_b[t] is the
    sum of probabilities assigned to sentence-ending tokens:

        p_b[t] = Σ_{k ∈ sentence_end_ids}  prob_matrix[b, t, k]

    We maintain a running counter `c` of expected tokens since the last boundary.
    At each step:
        • With soft probability p_b[t]: a sentence of length c ends → contribute
          to the histogram and reset c to 1.
        • Otherwise: c increments by 1.

    In expectation:
        c_{t+1} = c_t * (1 - p_b[t]) + 1

    The histogram contribution at step t is:
        h[k] += p_b[t] * soft_indicator(c_t, k)

    where soft_indicator uses a Gaussian kernel to spread mass across histogram bins.

    Args:
        prob_matrix:       (batch, seq_len, vocab_size) softmax probabilities.
        sentence_end_ids:  List of sentence-end token IDs.
        n_bins:            Number of histogram bins.
        max_length:        Maximum sentence length (bins span [0, max_length]).
        temperature:       Temperature for the Gaussian spreading kernel (higher
                           = softer boundaries, lower = sharper).
        eps:               Small constant for numerical stability.

    Returns:
        soft_hist: (batch, n_bins) normalised soft histogram (sums to ~1 per batch).
    """
    device = prob_matrix.device
    B, L, V = prob_matrix.shape

    # Sentence-end probability at each position: (B, L)
    end_ids_tensor = torch.tensor(sentence_end_ids, dtype=torch.long, device=device)
    # Clamp IDs within vocab range
    end_ids_tensor = end_ids_tensor[end_ids_tensor < V]
    if end_ids_tensor.numel() == 0:
        logger.warning("No valid sentence-end IDs found in vocabulary — L_style will be zero.")
        return torch.zeros(B, n_bins, device=device)

    p_boundary = prob_matrix[:, :, end_ids_tensor].sum(dim=-1)   # (B, L)

    # Bin centres for the histogram
    bin_centres = torch.linspace(0.0, float(max_length), n_bins, device=device)  # (n_bins,)
    sigma = float(max_length) / (n_bins * temperature) + eps

    # Running counter: expected tokens since last sentence boundary
    # Initialise at 1 (we are already inside a sentence)
    counter = torch.ones(B, device=device)    # (B,)

    # Accumulator for the soft histogram
    soft_hist = torch.zeros(B, n_bins, device=device)

    for t in range(L):
        pb = p_boundary[:, t]          # (B,) — boundary prob at position t

        # Gaussian contribution: spread the "sentence of length counter" across bins
        # gaussian_weight[b, k] ∝ exp(-(counter[b] - bin_centres[k])^2 / (2σ^2))
        diff = counter.unsqueeze(1) - bin_centres.unsqueeze(0)  # (B, n_bins)
        gaussian_w = torch.exp(-0.5 * (diff / sigma) ** 2)       # (B, n_bins)
        gaussian_w = gaussian_w / (gaussian_w.sum(dim=1, keepdim=True) + eps)

        # Contribution: pb[b] × gaussian_w[b, :]
        soft_hist = soft_hist + pb.unsqueeze(1) * gaussian_w   # (B, n_bins)

        # Update counter: reset to 1 with prob pb, else increment
        counter = counter * (1.0 - pb) + 1.0

    # Normalise to probability distribution
    soft_hist = soft_hist + eps
    soft_hist = soft_hist / soft_hist.sum(dim=1, keepdim=True)
    return soft_hist   # (B, n_bins)


# ---------------------------------------------------------------------------
# KL divergence helpers
# ---------------------------------------------------------------------------

def kl_divergence_with_reference(
    generated_dist: torch.Tensor,
    reference_dist: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric KL divergence: ½ [KL(Q ‖ P) + KL(P ‖ Q)]

    Args:
        generated_dist: (batch, n_bins) — soft histogram of generated text.
        reference_dist: (n_bins,)       — human-corpus baseline histogram.
        eps:            Smoothing constant.

    Returns:
        kl: (batch,) scalar per sample, then averaged.
    """
    # Expand reference to match batch dimension
    P = reference_dist.unsqueeze(0) + eps   # (1, n_bins)
    Q = generated_dist + eps                 # (B, n_bins)

    # KL(Q ‖ P)
    kl_qp = (Q * (Q / P).log()).sum(dim=-1)
    # KL(P ‖ Q)
    kl_pq = (P * (P / Q).log()).sum(dim=-1)

    return 0.5 * (kl_qp + kl_pq).mean()


# ---------------------------------------------------------------------------
# Lexical diversity loss (complementary to burstiness)
# ---------------------------------------------------------------------------

def soft_lexical_entropy_loss(
    prob_matrix: torch.Tensor,
    target_entropy: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalise deviations from target token-level entropy.

    AI-generated text tends to have lower token-level entropy than human text
    (it concentrates probability mass on a smaller set of predictable words).
    We encourage the soft output distribution to have entropy close to the
    human-corpus baseline.

    Args:
        prob_matrix:     (batch, seq_len, vocab_size) softmax probs.
        target_entropy:  Scalar — desired mean token-level entropy (nats).
                         Compute this from HumanCorpusStats.lexical_entropy.
        eps:             Numerical stability.

    Returns:
        Scalar loss.
    """
    # Token-level entropy: H[t] = -Σ_v p[v] log p[v]
    token_entropy = -(prob_matrix * (prob_matrix + eps).log()).sum(dim=-1)  # (B, L)
    mean_entropy = token_entropy.mean()

    # Penalise square deviation from target (smooth, symmetric)
    return (mean_entropy - target_entropy) ** 2


# ---------------------------------------------------------------------------
# Main L_style module
# ---------------------------------------------------------------------------

class StylometricLoss(nn.Module):
    """
    L_style = w_burst · KL_burstiness  +  w_lex · LexicalEntropyDeviation

    This is the novel loss term that distinguishes the Stylometry-Aware
    Differentiable Paraphraser from GradEscape and HMGC.

    Parameters
    ----------
    tokenizer           : evader tokenizer (used to resolve sentence-end IDs)
    human_stats         : HumanCorpusStats fitted on genuine human documents
    sentence_end_strings: list of string tokens that signal sentence boundaries
    n_bins              : histogram resolution
    max_sentence_length : cap for sentence length in histogram bins
    temperature         : Gaussian spreading kernel temperature
    burstiness_weight   : w_burst — relative weight of the burstiness KL term
    entropy_weight      : w_lex   — relative weight of the lexical-entropy term
    eps                 : numerical stability constant
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        human_stats,                       # HumanCorpusStats (avoid circular import)
        sentence_end_strings: Optional[List[str]] = None,
        n_bins: int = 25,
        max_sentence_length: int = 80,
        temperature: float = 1.0,
        burstiness_weight: float = 0.7,
        entropy_weight: float = 0.3,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.n_bins = n_bins
        self.max_sentence_length = max_sentence_length
        self.temperature = temperature
        self.burstiness_weight = burstiness_weight
        self.entropy_weight = entropy_weight
        self.eps = eps

        # Resolve sentence-end token IDs
        end_strings = sentence_end_strings or [".", "!", "?", "...", ".\n"]
        self.sentence_end_ids = resolve_sentence_end_ids(tokenizer, end_strings)

        # Register human baseline as a non-trainable buffer
        hist = human_stats.get_length_dist_tensor(device=torch.device("cpu"))
        self.register_buffer("human_length_dist", hist)

        # Target entropy for lexical diversity term
        self.target_entropy = human_stats.lexical_entropy
        logger.info(
            "StylometricLoss | sentence_end_ids=%s | target_entropy=%.3f | "
            "n_bins=%d | burstiness_w=%.2f | entropy_w=%.2f",
            self.sentence_end_ids, self.target_entropy,
            n_bins, burstiness_weight, entropy_weight,
        )

    # ------------------------------------------------------------------

    def forward(self, prob_matrix: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            prob_matrix: (batch, seq_len, vocab_size) — evader softmax output.

        Returns:
            l_style:     Scalar loss.
            info:        Dict of sub-component values for logging.
        """
        device = prob_matrix.device

        # Move human_length_dist buffer to same device as input
        human_dist = self.human_length_dist.to(device)

        # ----------------------------------------------------------------
        # Component 1: Burstiness KL divergence
        # ----------------------------------------------------------------
        soft_hist = soft_sentence_length_histogram(
            prob_matrix,
            self.sentence_end_ids,
            n_bins=self.n_bins,
            max_length=self.max_sentence_length,
            temperature=self.temperature,
            eps=self.eps,
        )   # (B, n_bins)

        kl_burst = kl_divergence_with_reference(soft_hist, human_dist, eps=self.eps)

        # ----------------------------------------------------------------
        # Component 2: Lexical entropy deviation
        # ----------------------------------------------------------------
        lex_loss = soft_lexical_entropy_loss(
            prob_matrix,
            self.target_entropy,
            eps=self.eps,
        )

        # ----------------------------------------------------------------
        # Weighted combination
        # ----------------------------------------------------------------
        l_style = (
            self.burstiness_weight * kl_burst
            + self.entropy_weight * lex_loss
        )

        info = {
            "l_style": l_style.item(),
            "kl_burstiness": kl_burst.item(),
            "lex_entropy_dev": lex_loss.item(),
        }
        return l_style, info


# ---------------------------------------------------------------------------
# Discrete (non-differentiable) stylometric metrics for logging
# ---------------------------------------------------------------------------

def compute_discrete_stylometrics(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """
    Compute exact (non-differentiable) stylometric features on decoded text.
    Used for logging and evaluation only — NOT for backpropagation.

    Returns a dict with:
        mean_sentence_length    : avg token count per sentence
        std_sentence_length     : std  dev (proxy for burstiness)
        burstiness_coefficient  : std / mean  (Fano-factor analogue)
        type_token_ratio        : unique tokens / total tokens (lexical diversity)
    """
    from data_utils import split_sentences   # local import to avoid circular

    all_lengths: List[int] = []
    all_tokens: List[int] = []

    for text in texts:
        sents = split_sentences(text)
        for sent in sents:
            toks = tokenizer.encode(sent, add_special_tokens=False)
            all_lengths.append(len(toks))
            all_tokens.extend(toks)

    if not all_lengths:
        return {}

    lengths = np.array(all_lengths, dtype=np.float32)
    mean_l = float(lengths.mean())
    std_l = float(lengths.std())

    return {
        "mean_sentence_length": mean_l,
        "std_sentence_length": std_l,
        "burstiness_coefficient": std_l / (mean_l + 1e-8),
        "type_token_ratio": len(set(all_tokens)) / (len(all_tokens) + 1e-8),
    }
