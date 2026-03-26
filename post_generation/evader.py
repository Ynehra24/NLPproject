"""
evader.py
---------
StyleAwareEvader — the core model class.

Architecture
============
Evader backbone : BART (BartForConditionalGeneration, ~140M params)
  • BART shares its BPE tokenizer with RoBERTa, making pseudo-embedding
    multiplication P @ W_emb exact with no token remapping.
  • For BERT-family detectors, a warm-started encoder-decoder is built
    automatically (Section 4.3, Meng et al. 2025).

Forward pass (training)
========================
  1. Encoder receives input_ids (AI-generated text).
  2. Decoder runs in "teacher-forcing" mode — input_ids as decoder_input_ids.
     This gives us logits over the full sequence without beam search, which
     is required for differentiability.
  3. Logits → softmax → prob_matrix P ∈ R^{B × L × V}.
  4. P is passed to JointEvaderLoss, which computes L_total via
     pseudo-embeddings (L_adv), label/semantic constraints (L_sem),
     and soft sentence-histogram KL (L_style).
  5. Gradients flow end-to-end: L_total → prob_matrix → BART parameters.

Inference (generation)
=======================
  • Standard beam search over decoded token IDs — fast and text-quality-aware.
  • No pseudo-embeddings involved; this is the text you actually deploy.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertForSequenceClassification,
    BertConfig,
    EncoderDecoderModel,
)

from config import Config
from pseudo_embeddings import PseudoEmbeddingInjector, get_word_embedding_weight
from losses import AdversarialLoss, SemanticLoss, StylometricLoss, JointEvaderLoss
from data_utils import HumanCorpusStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warm-started encoder-decoder (for BERT-family detectors)
# ---------------------------------------------------------------------------

def build_warm_started_evader(
    detector_model: PreTrainedModel,
) -> BartForConditionalGeneration:
    """
    Construct a seq2seq model using the detector's weights for initialisation,
    following the warm-started encoder-decoder technique (Meng et al., 2025).

    This allows the evader to attack BERT-based detectors even though BERT is
    encoder-only and has no native seq2seq counterpart with a matching tokenizer.

    Steps:
      1. Initialise EncoderDecoderModel(BertEncoder, BertDecoder).
      2. Copy shared parameter blocks from the detector's BERT weights.
      3. The model is then fine-tuned as a "repeater" on a public corpus
         (WikiText, BookCorpus) before adversarial training.

    NOTE: This factory function only builds the architecture.
    Call train_repeater() in trainer.py to SFT it as a repeater.
    """
    if isinstance(detector_model, BertForSequenceClassification):
        bert = detector_model.bert
        enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=None,
            decoder_pretrained_model_name_or_path=None,
            encoder=bert,
            decoder=bert,
        )
        logger.info(
            "Built warm-started BERT encoder-decoder with %d parameters",
            sum(p.numel() for p in enc_dec.parameters()),
        )
        return enc_dec
    raise NotImplementedError(
        "Warm-started evader currently supported only for BERT-based detectors. "
        "For RoBERTa/BART (same tokenizer), use BART directly."
    )


# ---------------------------------------------------------------------------
# Main evader
# ---------------------------------------------------------------------------

class StyleAwareEvader(nn.Module):
    """
    Stylometry-Aware Differentiable Paraphraser.

    Parameters
    ----------
    config          : Top-level Config object.
    surrogate_model : Fine-tuned BINARY classifier (0=human, 1=AI).
                      Must be a HuggingFace PreTrainedModel.
    human_stats     : HumanCorpusStats fitted on genuine human text.
    sentence_encoder: Optional frozen sentence-encoder for L_semantic.
                      If None, L_semantic is skipped (L_sem = L_label only).
    """

    def __init__(
        self,
        config: Config,
        surrogate_model: PreTrainedModel,
        human_stats: HumanCorpusStats,
        sentence_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # ----------------------------------------------------------------
        # Evader backbone (BART)
        # ----------------------------------------------------------------
        logger.info("Loading evader backbone: %s", config.model.evader_model_name)
        self.evader = BartForConditionalGeneration.from_pretrained(
            config.model.evader_model_name
        )
        self.tokenizer: PreTrainedTokenizer = BartTokenizer.from_pretrained(
            config.model.evader_model_name
        )

        # ----------------------------------------------------------------
        # Pseudo-embedding injector (wraps frozen surrogate)
        # ----------------------------------------------------------------
        self.injector = PseudoEmbeddingInjector(
            detector=surrogate_model,
            evader_vocab_size=self.tokenizer.vocab_size,
        )

        # ----------------------------------------------------------------
        # Optional sentence encoder for L_semantic
        # ----------------------------------------------------------------
        self.sentence_encoder = sentence_encoder
        if sentence_encoder is not None:
            enc_emb_w = get_word_embedding_weight(sentence_encoder)
        else:
            enc_emb_w = None

        # ----------------------------------------------------------------
        # Loss modules
        # ----------------------------------------------------------------
        adv_loss = AdversarialLoss(
            injector=self.injector,
            human_label_idx=config.model.human_label_idx,
        )

        sem_loss = SemanticLoss(
            sentence_encoder=sentence_encoder,
            label_weight=config.loss.label_loss_weight,
            semantic_weight=config.loss.semantic_loss_weight,
            encoder_emb_weight=enc_emb_w,
        )

        sty_loss = StylometricLoss(
            tokenizer=self.tokenizer,
            human_stats=human_stats,
            sentence_end_strings=config.loss.sentence_end_strings,
            n_bins=config.loss.n_length_bins,
            max_sentence_length=config.loss.max_sentence_length,
            temperature=config.loss.boundary_temperature,
            eps=config.loss.kl_eps,
        )

        self.joint_loss = JointEvaderLoss(
            adv_loss=adv_loss,
            sem_loss=sem_loss,
            sty_loss=sty_loss,
            alpha=config.loss.alpha,
            beta=config.loss.beta,
            gamma=config.loss.gamma,
        )

        logger.info(
            "StyleAwareEvader ready | α=%.2f β=%.2f γ=%.2f",
            config.loss.alpha, config.loss.beta, config.loss.gamma,
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def training_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        One forward + loss computation step.

        Args:
            input_ids:        (B, L) token IDs of the AI-generated source text.
            attention_mask:   (B, L) mask.
            input_embeddings: (B, H) pre-computed sentence-encoder embeddings
                              of the source text (detached, no grad).

        Returns:
            l_total:  Scalar loss (call .backward() on this).
            info:     Dict of sub-component values for logging.
        """
        # Teacher-forcing forward: feed input as both encoder AND decoder input.
        # This gives us logits for every token position without sampling.
        outputs = self.evader(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        logits = outputs.logits   # (B, L, vocab_size)

        # Convert to probabilities
        prob_matrix = F.softmax(logits, dim=-1)   # (B, L, vocab_size)

        # Joint loss
        l_total, info = self.joint_loss(
            prob_matrix=prob_matrix,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
        )
        return l_total, info

    # ------------------------------------------------------------------
    # Inference (beam search)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate paraphrased text using beam search.

        Returns:
            generated_ids: (B, L') token IDs of the paraphrased text.
        """
        nb = num_beams or self.config.model.num_beams
        ml = max_length or self.config.model.max_length

        return self.evader.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=nb,
            max_length=ml,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    @torch.no_grad()
    def paraphrase_texts(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[str]:
        """
        High-level convenience method: take a list of raw strings,
        return paraphrased strings.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self.tokenizer(
                batch,
                max_length=self.config.model.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            gen_ids = self.generate(enc["input_ids"], enc["attention_mask"])
            decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            results.extend(decoded)
        return results

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save evader model weights only (not the frozen detector)."""
        import os
        os.makedirs(path, exist_ok=True)
        self.evader.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Evader saved to %s", path)

    @classmethod
    def load_evader_weights(cls, path: str) -> BartForConditionalGeneration:
        """Load a previously saved evader backbone."""
        return BartForConditionalGeneration.from_pretrained(path)
