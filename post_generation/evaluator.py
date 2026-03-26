"""
evaluator.py
------------
Evaluation suite for the Stylometry-Aware Differentiable Paraphraser.

Metrics computed
================

Attack utility
--------------
  ASR   — Attack Success Rate on the white-box surrogate detector.
           Fraction of paraphrased texts classified as "human".
  CPTR  — Cross-Paradigm Transferability Rate.
           ASR measured on EACH black-box detector (unseen during training).
           This is the primary metric testing the core hypothesis.

Semantic preservation
---------------------
  BLEU        — Sacre-BLEU between paraphrase and original.
  BERTScore F1 — Contextual token overlap (via bert-score library).
  ROUGE-Lsum  — Longest common subsequence recall.

Macroscopic stylometry (verifying L_style worked)
--------------------------------------------------
  mean_sentence_length     — Average sentence length in tokens.
  std_sentence_length      — Standard deviation (proxy for burstiness).
  burstiness_coefficient   — std / mean (Fano factor).
  type_token_ratio         — Vocabulary diversity.
  kl_burstiness            — Actual KL(generated ‖ human_baseline).
  lexical_entropy          — Shannon entropy over unigram distribution.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Optional heavy dependencies — wrapped in try/except
try:
    from sacrebleu.metrics import BLEU as SacreBLEU
    _SACREBLEU_OK = True
except ImportError:
    _SACREBLEU_OK = False
    logger.warning("sacrebleu not installed — BLEU scores will be skipped.")

try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_OK = True
except ImportError:
    _BERTSCORE_OK = False
    logger.warning("bert_score not installed — BERTScore will be skipped.")

try:
    from rouge_score import rouge_scorer
    _ROUGE_OK = True
except ImportError:
    _ROUGE_OK = False
    logger.warning("rouge_score not installed — ROUGE-L will be skipped.")


# ---------------------------------------------------------------------------
# Detector interface helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    human_label_idx: int = 0,
    batch_size: int = 32,
    max_length: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[float], List[int]]:
    """
    Run a text classifier over a list of strings.

    Returns:
        human_probs: List of P(human) per sample.
        predictions: List of predicted labels (0=human, 1=AI).
    """
    model.eval().to(device)
    all_probs: List[float] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits                              # (B, num_labels)
        probs = F.softmax(logits, dim=-1)[:, human_label_idx]   # (B,)
        all_probs.extend(probs.cpu().tolist())

    predictions = [0 if p >= 0.5 else 1 for p in all_probs]
    return all_probs, predictions


# ---------------------------------------------------------------------------
# Stylometric metrics on raw text
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Simple regex sentence splitter."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def compute_text_stylometrics(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    human_length_hist: Optional[np.ndarray] = None,
    bin_edges: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute macroscopic stylometric features from a list of decoded strings.

    Args:
        texts:             List of decoded paraphrase strings.
        tokenizer:         Tokenizer for sentence-length measurement.
        human_length_hist: (n_bins,) normalised histogram from HumanCorpusStats.
                           Required for kl_burstiness computation.
        bin_edges:         (n_bins+1,) bin edges from HumanCorpusStats.

    Returns:
        Dict with all stylometric features.
    """
    all_lengths: List[int] = []
    all_tokens: List[int] = []

    for text in texts:
        sents = _split_sentences(text)
        for sent in sents:
            tids = tokenizer.encode(sent, add_special_tokens=False)
            all_lengths.append(len(tids))
            all_tokens.extend(tids)

    if not all_lengths:
        return {}

    lengths = np.array(all_lengths, dtype=np.float32)
    mean_l = float(lengths.mean())
    std_l = float(lengths.std())

    # Lexical entropy
    word_freq = Counter(all_tokens)
    total = sum(word_freq.values())
    lex_entropy = float(
        -sum((c / total) * math.log(c / total + 1e-12) for c in word_freq.values())
    )

    result = {
        "mean_sentence_length": mean_l,
        "std_sentence_length": std_l,
        "burstiness_coefficient": std_l / (mean_l + 1e-8),
        "type_token_ratio": len(word_freq) / (total + 1e-8),
        "lexical_entropy": lex_entropy,
    }

    # KL divergence vs. human baseline
    if human_length_hist is not None and bin_edges is not None:
        gen_hist, _ = np.histogram(lengths, bins=bin_edges)
        gen_hist = gen_hist.astype(np.float64) + 1e-8
        gen_hist = gen_hist / gen_hist.sum()

        P = human_length_hist + 1e-8
        Q = gen_hist

        kl_qp = float(np.sum(Q * np.log(Q / P)))
        kl_pq = float(np.sum(P * np.log(P / Q)))
        result["kl_burstiness"] = 0.5 * (kl_qp + kl_pq)

    return result


# ---------------------------------------------------------------------------
# BLEU / BERTScore / ROUGE helpers
# ---------------------------------------------------------------------------

def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Compute corpus-level BLEU using sacrebleu."""
    if not _SACREBLEU_OK:
        return float("nan")
    bleu = SacreBLEU()
    result = bleu.corpus_score(hypotheses, [references])
    return float(result.score)


def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    model_type: str = "roberta-large",
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute BERTScore P / R / F1."""
    if not _BERTSCORE_OK:
        return {"P": float("nan"), "R": float("nan"), "F1": float("nan")}
    P, R, F1 = bert_score_fn(
        hypotheses, references, model_type=model_type, device=device, verbose=False
    )
    return {
        "bertscore_P": float(P.mean()),
        "bertscore_R": float(R.mean()),
        "bertscore_F1": float(F1.mean()),
    }


def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """Compute average ROUGE-Lsum."""
    if not _ROUGE_OK:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    scores = [
        scorer.score(ref, hyp)["rougeLsum"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main Evaluator class
# ---------------------------------------------------------------------------

class EvaderEvaluator:
    """
    Full evaluation harness for the Stylometry-Aware Differentiable Paraphraser.

    Computes:
      - ASR on the white-box surrogate
      - CPTR on each black-box detector
      - BLEU, BERTScore, ROUGE-L
      - Stylometric features (burstiness, lexical entropy, KL)

    Parameters
    ----------
    surrogate_model   : The detector used during training (white-box).
    surrogate_tok     : Its tokenizer.
    blackbox_models   : List of (detector_model, tokenizer) for black-box eval.
    human_stats       : HumanCorpusStats for KL computation (may be None).
    human_label_idx   : Which output index = "human" in the classifiers.
    device            : Torch device string.
    bertscore_model   : Model ID for BERTScore.
    """

    def __init__(
        self,
        surrogate_model: PreTrainedModel,
        surrogate_tok: PreTrainedTokenizer,
        blackbox_models: Optional[List[Tuple[PreTrainedModel, PreTrainedTokenizer]]] = None,
        human_stats=None,
        human_label_idx: int = 0,
        device: str = "cuda",
        bertscore_model: str = "roberta-large",
    ):
        self.surrogate = surrogate_model
        self.surrogate_tok = surrogate_tok
        self.blackbox_models = blackbox_models or []
        self.human_stats = human_stats
        self.human_label_idx = human_label_idx
        self.device = torch.device(device)
        self.bertscore_model = bertscore_model

    # ------------------------------------------------------------------

    def evaluate(
        self,
        original_texts: List[str],
        paraphrased_texts: List[str],
        baseline_asr: Optional[float] = None,
    ) -> Dict:
        """
        Run the full evaluation suite.

        Args:
            original_texts:   Raw AI-generated texts (before evasion).
            paraphrased_texts: Evader output texts.
            baseline_asr:     Optional — ASR of unmodified AI text on surrogate
                              (for ΔAcc / improvement calculation).

        Returns:
            Flat dict of all metrics.
        """
        assert len(original_texts) == len(paraphrased_texts)
        results: Dict = {}

        # ---- 1. ASR on white-box surrogate ----
        logger.info("Computing ASR on surrogate detector…")
        surr_probs, surr_preds = batch_predict(
            self.surrogate,
            self.surrogate_tok,
            paraphrased_texts,
            human_label_idx=self.human_label_idx,
            device=self.device,
        )
        asr = sum(1 for p in surr_preds if p == self.human_label_idx) / len(surr_preds)
        results["ASR_surrogate"] = asr
        results["mean_human_prob_surrogate"] = float(np.mean(surr_probs))

        if baseline_asr is not None:
            results["ASR_improvement"] = asr - baseline_asr

        # ---- 2. CPTR on black-box detectors ----
        for blackbox in self.blackbox_models:
            if hasattr(blackbox, 'detector_name'):
                # Handle detector_evaluation abstract detectors
                name = blackbox.detector_name
                logger.info("Computing CPTR on: %s", name)
                ai_scores, _ = blackbox.score_texts(paraphrased_texts)
                cptr = sum(1 for s in ai_scores if s <= 0.5) / len(ai_scores)
                results[f"CPTR_{name}"] = cptr
                
                logger.info("Computing Baseline TNR on: %s", name)
                orig_ai_scores, _ = blackbox.score_texts(original_texts)
                baseline_tnr = sum(1 for s in orig_ai_scores if s > 0.5) / len(orig_ai_scores)
                results[f"Baseline_TNR_{name}"] = baseline_tnr
            else:
                bb_model, bb_tok = blackbox
                name = getattr(bb_model.config, "_name_or_path", f"blackbox_{idx}")
                logger.info("Computing CPTR on: %s", name)
                _, bb_preds = batch_predict(
                    bb_model, bb_tok, paraphrased_texts,
                    human_label_idx=self.human_label_idx,
                    device=self.device,
                )
                cptr = sum(1 for p in bb_preds if p == self.human_label_idx) / len(bb_preds)
                results[f"CPTR_{name}"] = cptr

        # Also measure baseline detection rate of original AI text
        logger.info("Measuring baseline detection of unmodified AI text…")
        _, orig_preds = batch_predict(
            self.surrogate, self.surrogate_tok, original_texts,
            human_label_idx=self.human_label_idx,
            device=self.device,
        )
        orig_detected_as_ai = sum(1 for p in orig_preds if p != self.human_label_idx)
        results["baseline_TNR"] = orig_detected_as_ai / len(orig_preds)

        # ---- 3. Semantic preservation ----
        logger.info("Computing BLEU / ROUGE-L…")
        results["BLEU"] = compute_bleu(paraphrased_texts, original_texts)
        results["ROUGE_Lsum"] = compute_rouge_l(paraphrased_texts, original_texts)

        logger.info("Computing BERTScore…")
        bert_scores = compute_bertscore(
            paraphrased_texts, original_texts,
            model_type=self.bertscore_model,
            device=str(self.device),
        )
        results.update(bert_scores)

        # ---- 4. Stylometric features ----
        logger.info("Computing stylometric features…")
        human_hist = (
            self.human_stats.length_histogram if self.human_stats else None
        )
        bin_edges = self.human_stats.bin_edges if self.human_stats else None

        style_metrics = compute_text_stylometrics(
            paraphrased_texts,
            tokenizer=self.surrogate_tok,
            human_length_hist=human_hist,
            bin_edges=bin_edges,
        )
        results.update({f"style_{k}": v for k, v in style_metrics.items()})

        # Also compute on originals for delta comparison
        orig_style = compute_text_stylometrics(
            original_texts,
            tokenizer=self.surrogate_tok,
            human_length_hist=human_hist,
            bin_edges=bin_edges,
        )
        results.update({f"orig_style_{k}": v for k, v in orig_style.items()})

        # ---- Summary ----
        self._print_summary(results)
        return results

    # ------------------------------------------------------------------

    def ablation_compare(
        self,
        original_texts: List[str],
        outputs: Dict[str, List[str]],
    ) -> Dict[str, Dict]:
        """
        Compare multiple evaders (e.g., full model vs. ablations).

        Args:
            original_texts: Source AI-generated texts.
            outputs: Dict mapping evader name → list of paraphrased texts.

        Returns:
            Dict mapping evader name → its evaluation result dict.
        """
        comparison: Dict[str, Dict] = {}
        for name, paraphrases in outputs.items():
            logger.info("=== Ablation: %s ===", name)
            metrics = self.evaluate(original_texts, paraphrases)
            comparison[name] = metrics
        return comparison

    # ------------------------------------------------------------------

    def save_results(self, results: Dict, path: str) -> None:
        """Serialise evaluation results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy floats to native Python floats
        clean = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in results.items()
        }
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info("Evaluation results saved to %s", path)

    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(results: Dict) -> None:
        """Print a concise tabular summary to stdout."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        order = [
            ("ASR_surrogate",          "ASR (surrogate, white-box)"),
            ("baseline_TNR",           "Baseline AI detection rate"),
            ("BLEU",                   "BLEU"),
            ("ROUGE_Lsum",             "ROUGE-Lsum"),
            ("bertscore_F1",           "BERTScore F1"),
            ("style_burstiness_coefficient",   "Burstiness coefficient"),
            ("style_kl_burstiness",    "KL burstiness (↓ = more human)"),
            ("style_lexical_entropy",  "Lexical entropy"),
        ]

        for key, label in order:
            if key in results:
                val = results[key]
                if isinstance(val, float):
                    print(f"  {label:<40s}  {val:.4f}")
                else:
                    print(f"  {label:<40s}  {val}")

        # Print CPTR entries
        for k, v in results.items():
            if k.startswith("CPTR_"):
                name = k[5:]
                print(f"  CPTR [{name:<28s}]  {float(v):.4f}")

        print("=" * 60 + "\n")
