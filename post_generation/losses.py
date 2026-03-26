"""
losses.py
---------
Assembles the three loss components into the joint objective:

    L_total = α · L_adv  +  β · L_sem  +  γ · L_style

L_adv  — Adversarial loss (Section 4.2 GradEscape / Section 4.1–4.3 HMGC)
         Forces the frozen surrogate detector to output "human" for the
         evader's paraphrase.  Uses pseudo-embeddings for differentiability.

L_sem  — Semantic + syntactic preservation (Section 4.2 GradEscape / 4.4 HMGC)
         Two sub-terms:
           (a) Label loss  — token-level cross-entropy between the evader's
               output probability and the original input token (encourages
               the evader to copy rather than radically rewrite).
           (b) Semantic loss — MSE between sentence-encoder embeddings of
               input and output in continuous embedding space.

L_style — Stylometric naturalness (novel contribution of this work)
          KL divergence of soft sentence-length histogram vs. human baseline
          plus a lexical entropy deviation penalty.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pseudo_embeddings import PseudoEmbeddingInjector
from stylometric_loss import StylometricLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L_adv — Adversarial loss
# ---------------------------------------------------------------------------

class AdversarialLoss(nn.Module):
    """
    L_adv = CrossEntropy(detector(pseudo_emb), y_human)

    Minimising this pushes the (pseudo-embedded) evader output toward
    the detector's "human-written" class.

    Args:
        injector:        Frozen PseudoEmbeddingInjector wrapping the surrogate.
        human_label_idx: Index of the "human" class in the detector's output.
    """

    def __init__(self, injector: PseudoEmbeddingInjector, human_label_idx: int = 0):
        super().__init__()
        self.injector = injector
        self.human_label_idx = human_label_idx

    def forward(
        self,
        prob_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prob_matrix:    (batch, seq_len, vocab_size) — evader softmax output.
            attention_mask: (batch, seq_len) — 1 for real tokens.

        Returns:
            l_adv:   Scalar adversarial loss.
            logits:  (batch, num_labels) — detector logits (for logging).
        """
        # Forward through frozen detector via pseudo-embeddings
        logits = self.injector(prob_matrix, attention_mask)   # (B, num_labels)
        B = logits.size(0)

        # We want the detector to say "human" for every sample
        targets = torch.full(
            (B,), self.human_label_idx,
            dtype=torch.long,
            device=logits.device,
        )
        l_adv = F.cross_entropy(logits, targets)
        return l_adv, logits


# ---------------------------------------------------------------------------
# L_sem — Semantic + syntactic preservation
# ---------------------------------------------------------------------------

class SemanticLoss(nn.Module):
    """
    L_sem = w_label · L_label  +  w_sem · L_semantic

    L_label:
        Token-level cross-entropy between the evader output probabilities
        and the original input token IDs.  Penalises large departures from
        the source text at the token level (syntactic preservation).

        This is equivalent to training the evader as a "soft repeater" —
        it must output close to the input while also evading detection.

        Reference: Equation (8) in Meng et al. (2025).

    L_semantic:
        MSE between the sentence-encoder (e.g., sentence-transformers) embeddings
        of the pseudo-decoded output and the original input text.

        Because exact decoding is non-differentiable, we approximate the output
        embedding using the pseudo-embedding trick:
            emb_out ≈ SentEnc(P @ W_emb)

        Reference: Equation (9) in Meng et al. (2025).

    Args:
        sentence_encoder:  A frozen sentence-encoder module whose forward()
                           accepts either input_ids or inputs_embeds.
                           Pass None to disable the semantic sub-term.
        label_weight:      Weight for L_label (default 0.5).
        semantic_weight:   Weight for L_semantic (default 0.5).
        encoder_emb_weight: Embedding weight of the sentence encoder (for
                            pseudo-embedding in the semantic sub-term).
    """

    def __init__(
        self,
        sentence_encoder: Optional[nn.Module] = None,
        label_weight: float = 0.5,
        semantic_weight: float = 0.5,
        encoder_emb_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.label_weight = label_weight
        self.semantic_weight = semantic_weight
        self.encoder_emb_weight = encoder_emb_weight

        if sentence_encoder is not None:
            for param in sentence_encoder.parameters():
                param.requires_grad_(False)
            sentence_encoder.eval()

    # ------------------------------------------------------------------

    def _label_loss(
        self,
        prob_matrix: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Token-level cross-entropy: L_label = -1/N Σ log P[t, input_id[t]]

        We shift by 1 so that position t of the output is compared against
        position t of the input (standard seq2seq label alignment).
        """
        B, L, V = prob_matrix.shape
        # Clamp IDs to vocab range (handles padding with large IDs)
        target_ids = input_ids[:, :L].clamp(0, V - 1)   # (B, L)

        log_probs = torch.log(prob_matrix + 1e-10)        # (B, L, V)
        # Gather log-prob at the target token position
        gathered = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)

        if attention_mask is not None:
            mask = attention_mask[:, :L].float()
            loss = -(gathered * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = -gathered.mean()

        return loss

    def _semantic_loss(
        self,
        prob_matrix: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE between pseudo-decoded sentence embedding and source sentence embedding.

        Both embeddings are computed in continuous space via pseudo-embeddings,
        so gradients flow back to the evader.

        Args:
            prob_matrix:      (B, L, V) evader output probs.
            input_embeddings: (B, H) sentence encoder embeddings of the SOURCE text,
                              pre-computed before training (no grad required).
        """
        if self.sentence_encoder is None or self.encoder_emb_weight is None:
            return torch.tensor(0.0, device=prob_matrix.device)

        # Pseudo-embed the output for the sentence encoder
        # (B, L, V) @ (V, H_enc) → (B, L, H_enc)
        pseudo_emb = torch.matmul(prob_matrix, self.encoder_emb_weight)

        # Mean-pool over valid (non-pad) positions as a simple sentence representation
        out_sent_emb = pseudo_emb.mean(dim=1)   # (B, H_enc)

        # MSE in embedding space
        loss = F.mse_loss(out_sent_emb, input_embeddings.detach())
        return loss

    # ------------------------------------------------------------------

    def forward(
        self,
        prob_matrix: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            prob_matrix:       (B, L, V)
            input_ids:         (B, L) — original token IDs of the input text.
            attention_mask:    (B, L) — optional mask.
            input_embeddings:  (B, H) — pre-computed sentence embeddings of
                               the source text.  Required for L_semantic.

        Returns:
            l_sem:  Scalar semantic loss.
            info:   Dict with sub-term values.
        """
        l_label = self._label_loss(prob_matrix, input_ids, attention_mask)

        l_semantic = (
            self._semantic_loss(prob_matrix, input_embeddings)
            if input_embeddings is not None
            else torch.tensor(0.0, device=prob_matrix.device)
        )

        l_sem = self.label_weight * l_label + self.semantic_weight * l_semantic

        info = {
            "l_sem": l_sem.item(),
            "l_label": l_label.item(),
            "l_semantic": l_semantic.item(),
        }
        return l_sem, info


# ---------------------------------------------------------------------------
# Joint loss combinator
# ---------------------------------------------------------------------------

class JointEvaderLoss(nn.Module):
    """
    L_total = α · L_adv  +  β · L_sem  +  γ · L_style  +  δ · L_fluency

    This is the complete four-objective loss for the Stylometry-Aware
    Differentiable Paraphraser.  It wraps AdversarialLoss, SemanticLoss,
    StylometricLoss, and FluencyLoss, exposes a single forward() method, and returns
    a rich diagnostics dict for tensorboard/wandb logging.
    
    The fluency loss penalizes low-entropy (repetitive) outputs, preventing
    degenerate solutions like "BraBraBra...".
    """

    def __init__(
        self,
        adv_loss: AdversarialLoss,
        sem_loss: SemanticLoss,
        sty_loss: StylometricLoss,
        alpha: float = 0.20,
        beta: float = 0.40,
        gamma: float = 0.30,
        delta: float = 0.10,
    ):
        super().__init__()
        self.adv_loss = adv_loss
        self.sem_loss = sem_loss
        self.sty_loss = sty_loss

        # Weights now include fluency term
        total_weight = alpha + beta + gamma + delta
        assert abs(total_weight - 1.0) < 1e-4, (
            f"Loss weights must sum to 1.0, got α={alpha}, β={beta}, γ={gamma}, δ={delta} (sum={total_weight})"
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _fluency_loss(
        self,
        prob_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Penalize outputs with very low entropy (model is highly certain/repetitive).
        
        Low entropy across tokens indicates the model is outputting the same token
        repeatedly with high confidence, which prevents corruption like "BraBraBra...".
        
        Args:
            prob_matrix: (B, L, V) probability distribution over vocabulary.
            attention_mask: (B, L) optional mask for padding tokens.
        
        Returns:
            l_fluency: Scalar fluency loss.
            info: Dict with diagnostic values.
        """
        # Entropy per token: -Σ p log p
        entropy = -(prob_matrix * torch.log(prob_matrix + 1e-10)).sum(dim=-1)  # (B, L)
        
        # Penalize very low entropy (certain/repetitive predictions)
        # For a vocab of ~50k, max entropy ≈ log(50000) ≈ 10.8
        # We want to encourage diversity, so target min_entropy around 2.0
        min_entropy_threshold = 2.0
        low_entropy_penalty = torch.relu(min_entropy_threshold - entropy)  # (B, L)
        
        # Apply attention mask if available
        if attention_mask is not None:
            mask = attention_mask.float()
            loss = (low_entropy_penalty * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = low_entropy_penalty.mean()
        
        info = {
            "l_fluency": loss.item(),
            "mean_entropy": entropy.mean().item(),
        }
        return loss, info

    def forward(
        self,
        prob_matrix: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            prob_matrix:      (B, L, V) — continuous softmax output of the evader.
            input_ids:        (B, L)    — original (AI-generated) token IDs.
            attention_mask:   (B, L)    — attention mask (1 = real token).
            input_embeddings: (B, H)    — pre-computed sentence embeddings of
                              the source text.

        Returns:
            l_total: Scalar total loss (differentiable w.r.t. evader params).
            info:    Flat dict of all sub-component values for logging.
        """
        # -------- L_adv --------
        l_adv, detector_logits = self.adv_loss(prob_matrix, attention_mask)

        # -------- L_sem --------
        l_sem, sem_info = self.sem_loss(
            prob_matrix, input_ids, attention_mask, input_embeddings
        )

        # -------- L_style --------
        l_style, style_info = self.sty_loss(prob_matrix)

        # -------- L_fluency --------
        l_fluency, fluency_info = self._fluency_loss(prob_matrix, attention_mask)

        # -------- L_total --------
        l_total = (
            self.alpha * l_adv
            + self.beta * l_sem
            + self.gamma * l_style
            + self.delta * l_fluency
        )

        # Compute a "human probability" for quick monitoring
        with torch.no_grad():
            human_prob = F.softmax(detector_logits, dim=-1)[
                :, self.adv_loss.human_label_idx
            ].mean().item()

        info = {
            "l_total": l_total.item(),
            "l_adv": l_adv.item(),
            "human_prob_surrogate": human_prob,
            **sem_info,
            **style_info,
            **fluency_info,
        }
        return l_total, info
