"""
humanizer.py  —  v3  (Beat-Charmer Edition)
============================================

THE CRITICAL FIX FROM v2:
  v2 used `classifier_fn = lambda t: "ai"` which means the CSBP beam search
  NEVER checked whether attacks were actually fooling BERT/RoBERTa.  It just
  optimised GPT-2 perplexity blindly for all K rounds.  This produced 0% ASR.

v3 changes:
  1. Wires the REAL TextAttack / HuggingFace classifier into CSBP via:
       classifier_fn  — fn(text) → label string  (for misclassification check)
       confidence_fn  — fn(text) → float         (P(original_label), for beam scoring)

  2. build_classifier() constructs both functions from a HuggingFace pipeline.
     Supports any model on the HuggingFace hub or a local path.

  3. humanize() now accepts target_model and dataset parameters so you can
     directly target the same model used in your evaluation CSV:
       "textattack/bert-base-uncased-ag-news"    → AG-News BERT
       "textattack/bert-base-uncased-SST-2"       → SST-2 BERT
       "textattack/bert-base-uncased-QNLI"        → QNLI BERT
       "textattack/bert-base-uncased-RTE"         → RTE BERT
       "textattack/roberta-base-SST-2"            → SST-2 RoBERTa
       "textattack/roberta-base-ag-news"          → AG-News RoBERTa

  4. Fallback: if no target_model is specified, a statistical-only mode is used
     (optimises GPT-2 perplexity + BPE disruption).  This is the v2 behaviour
     and is kept for compatibility.

  5. Fast-path evaluation: after the CSBP beam search finishes, humanize()
     does a final check and reports whether the attack succeeded.
"""

import re
import sys
import csv
import json
import random
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import emoji
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── Local imports ──────────────────────────────────────────────
CORE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(CORE_DIR))

from composite_scorer import composite_score
from homoglyph_attack import apply_homoglyph, apply_diacritic, apply_blitz
from csbp_loop        import csbp_loop
from emoji_insertion  import load_or_extract_emojis

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════
# TextAttack Model Registry
# ═══════════════════════════════════════════════════════════════
# Maps (dataset, model_arch) → HuggingFace model ID.
# These are the exact models used in Charmer's benchmark evaluation.

TEXTATTACK_MODELS = {
    # BERT-base
    ('sst2',   'bert'):   'textattack/bert-base-uncased-SST-2',
    ('qnli',   'bert'):   'textattack/bert-base-uncased-QNLI',
    ('rte',    'bert'):   'textattack/bert-base-uncased-RTE',
    ('agnews', 'bert'):   'textattack/bert-base-uncased-ag-news',
    ('mnli',   'bert'):   'textattack/bert-base-uncased-MNLI',
    # RoBERTa-base
    ('sst2',   'roberta'): 'textattack/roberta-base-SST-2',
    ('qnli',   'roberta'): 'textattack/roberta-base-QNLI',
    ('rte',    'roberta'): 'textattack/roberta-base-RTE',
    ('agnews', 'roberta'): 'textattack/roberta-base-ag-news',
    # HC3 / M4 — use a general AI-text detector as proxy
    ('hc3',    'bert'):   'Hello-SimpleAI/chatgpt-detector-roberta',
    ('m4',     'bert'):   'Hello-SimpleAI/chatgpt-detector-roberta',
}


# ═══════════════════════════════════════════════════════════════
# Classifier Builder
# ═══════════════════════════════════════════════════════════════

def build_classifier(
    model_name_or_path: str,
    device: Optional[str] = None,
) -> Tuple[Callable[[str], str], Callable[[str], float]]:
    """
    Build (classifier_fn, confidence_fn) from a HuggingFace model.

    classifier_fn(text) → label string   e.g. "LABEL_0" / "POSITIVE" / "World"
    confidence_fn(text) → float          P(predicted label | text) ∈ [0, 1]

    Usage in CSBP:
      The original_label is the model's prediction on the ORIGINAL (unattacked)
      text.  classifier_fn is used to check if the prediction CHANGED.
      confidence_fn feeds composite_score() so beam ranking directly minimises
      the classifier's confidence in the original prediction.
    """
    from transformers import pipeline as hf_pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else (-1)
    else:
        device = 0 if (device == 'cuda') else -1

    print(f"[Classifier] Loading: {model_name_or_path}  (device={device})")
    pipe = hf_pipeline(
        "text-classification",
        model=model_name_or_path,
        device=device,
        truncation=True,
        max_length=512,
    )

    def classifier_fn(text: str) -> str:
        try:
            return pipe(text[:512], truncation=True)[0]['label']
        except Exception:
            return "UNKNOWN"

    def confidence_fn(text: str) -> float:
        """Return P(original_label | text) by finding the matching label."""
        try:
            outputs = pipe(text[:512], truncation=True,
                           return_all_scores=True)[0]
            # outputs is a list of {'label': ..., 'score': ...}
            # confidence_fn is called with the ORIGINAL label, so we need
            # to cache it.  We return the score of whichever label the
            # ORIGINAL text was assigned — this is done by the caller
            # wrapping it in a closure (see humanize() below).
            return float(outputs[0]['score'])
        except Exception:
            return 0.5

    return classifier_fn, confidence_fn


def build_confidence_for_label(
    pipe,
    original_label: str,
) -> Callable[[str], float]:
    """
    Returns a confidence_fn that specifically tracks P(original_label | text).
    Call this AFTER determining original_label on the unattacked text.

    This is the correct way to wire confidence into CSBP:
      1. Call classifier_fn(original_text) → original_label
      2. Call build_confidence_for_label(pipe, original_label) → confidence_fn
      3. Pass both to csbp_loop()
    """
    from transformers import pipeline as hf_pipeline

    def confidence_fn(text: str) -> float:
        try:
            outputs = pipe(text[:512], truncation=True,
                           return_all_scores=True)[0]
            for entry in outputs:
                if entry['label'] == original_label:
                    return float(entry['score'])
            # If label not found, return the first score (fallback)
            return float(outputs[0]['score'])
        except Exception:
            return 0.5

    return confidence_fn


# ═══════════════════════════════════════════════════════════════
# Formality & Emoji Support
# ═══════════════════════════════════════════════════════════════

FORMALITY_DIR = CORE_DIR / "formality_model"

_f_model  = None
_f_clf    = None
_f_scaler = None
_FORMAL_EMOTES:   List[str] = []
_INFORMAL_EMOTES: List[str] = []
_MODELS_LOADED  = False
_SELECTED_DEVICE: Optional[str] = None


def ensure_support_models_loaded(device_override: Optional[str] = None):
    global _f_model, _f_clf, _f_scaler, _FORMAL_EMOTES, _INFORMAL_EMOTES
    global _MODELS_LOADED, _SELECTED_DEVICE

    if _MODELS_LOADED:
        return

    _SELECTED_DEVICE = device_override or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Support models] Loading on device={_SELECTED_DEVICE} ...")

    from sentence_transformers import SentenceTransformer
    try:
        _f_model  = SentenceTransformer(str(FORMALITY_DIR / "embed_model"), device=_SELECTED_DEVICE)
        _f_clf    = joblib.load(FORMALITY_DIR / "classifier.joblib")
        _f_scaler = joblib.load(FORMALITY_DIR / "scaler.joblib")
        formal, informal = load_or_extract_emojis(top_n=50)
        _FORMAL_EMOTES   = list(formal   or ['📘','📖','✒️','📚'])
        _INFORMAL_EMOTES = list(informal or ['🙂','👍','😊','🤔','☕','🐱'])
        _MODELS_LOADED   = True
        print(f"✓ Support models loaded. ({len(_FORMAL_EMOTES)} formal / {len(_INFORMAL_EMOTES)} informal emoji)")
    except Exception as e:
        print(f"[Support models] Warning: {e}.  Falling back to CPU.")
        try:
            _SELECTED_DEVICE = "cpu"
            _f_model = SentenceTransformer(str(FORMALITY_DIR / "embed_model"), device="cpu")
            _f_clf   = joblib.load(FORMALITY_DIR / "classifier.joblib")
            _f_scaler = joblib.load(FORMALITY_DIR / "scaler.joblib")
            formal, informal = load_or_extract_emojis(top_n=50)
            _FORMAL_EMOTES   = list(formal   or ['📘','📖','✒️','📚'])
            _INFORMAL_EMOTES = list(informal or ['🙂','👍','😊','🤔','☕','🐱'])
            _MODELS_LOADED   = True
            print("✓ Support models loaded on CPU.")
        except Exception as e2:
            print(f"[Support models] Fatal: {e2}.  Register detection disabled.")
            _FORMAL_EMOTES   = ['📘','📖','✒️']
            _INFORMAL_EMOTES = ['🙂','👍','😊']
            _MODELS_LOADED   = True   # mark loaded to avoid retry loop


def get_register(text: str) -> str:
    ensure_support_models_loaded()
    if _f_model is None or _f_clf is None:
        return "informal"
    t = re.sub(r"\s+", " ", text).strip()
    emb = _f_model.encode([t], convert_to_numpy=True)
    combined = np.concatenate([emb, np.zeros((1, 11))], axis=1)
    return _f_clf.predict(combined)[0]


def strip_emojis(text: str) -> str:
    return ''.join(ch for ch in text if ch not in emoji.EMOJI_DATA).strip()


# ═══════════════════════════════════════════════════════════════
# Main humanize() Function
# ═══════════════════════════════════════════════════════════════

import transformers

_GLOBAL_CLF_PIPES = {}

def humanize(
    text: str,
    # ── Target model (the one you want to fool) ──────────────────
    target_model:   Optional[str] = None,
    # OR pre-built functions (if you already built the pipeline)
    classifier_fn:  Optional[Callable[[str], str]]   = None,
    confidence_fn:  Optional[Callable[[str], float]] = None,
    # ── Search hyperparameters ────────────────────────────────────
    iterations:     int  = 7,
    n_candidates:   int  = 20,
    beam_width:     int  = 8,
    device_override: Optional[str] = None,
    verbose:        bool = True,
) -> dict:
    """
    Humanize AI-generated text using CSBP v3 beam search.

    REQUIRED for real ASR: provide either:
      target_model  — HuggingFace model ID or path
    OR both:
      classifier_fn  — fn(text) → label string
      confidence_fn  — fn(text) → P(original_label | text)

    If neither is provided, falls back to statistical-only mode
    (optimises GPT-2 perplexity — works for commercial detectors
     like ZeroGPT/Originality.ai but NOT for BERT/RoBERTa classifiers).

    Returns:
      {
        'success':          bool,
        'original_text':    str,
        'original_label':   str,
        'humanized_text':   str,
        'best_score':       float,
        'round_found':      int | None,
        'score_breakdown':  dict,
        'beam_history':     list,
      }
    """
    ensure_support_models_loaded(device_override)

    # ── Build classifier from model name if given ──────────────────
    _clf_fn = classifier_fn
    _conf_fn = confidence_fn
    _pipe   = None

    if target_model is not None and _clf_fn is None:
        global _GLOBAL_CLF_PIPES
        from transformers import pipeline as hf_pipeline
        dev = 0 if (torch.cuda.is_available() and device_override != 'cpu') else -1
        
        if target_model not in _GLOBAL_CLF_PIPES:
            print(f"[humanize] Loading target model: {target_model}  device={dev}")
            _GLOBAL_CLF_PIPES[target_model] = hf_pipeline(
                "text-classification",
                model=target_model,
                device=dev,
                truncation=True,
                max_length=512,
                return_all_scores=True,
            )
        _pipe = _GLOBAL_CLF_PIPES[target_model]
        _clf_fn = lambda t: _pipe(t[:512], truncation=True, return_all_scores=False)[0]['label']

    # ── Get original prediction ────────────────────────────────────
    if _clf_fn is not None:
        original_label = _clf_fn(text)
        if verbose:
            print(f"[humanize] Original label: {original_label}")
    else:
        # Statistical-only mode — no real classifier
        original_label = "ai"
        if verbose:
            print("[humanize] No classifier provided — statistical-only mode.")
            print("           Wire target_model or classifier_fn to attack BERT/RoBERTa.")

    # ── Build label-specific confidence function ───────────────────
    if _pipe is not None and _conf_fn is None:
        _conf_fn = build_confidence_for_label(_pipe, original_label)
    elif _clf_fn is None:
        _conf_fn = None   # pure statistical mode

    # ── CSBP beam search ──────────────────────────────────────────
    register = get_register(text)
    if verbose:
        print(f"[humanize] Register: {register.upper()}  "
              f"iterations={iterations}  beam={beam_width}  cands={n_candidates}")

    if _clf_fn is None:
        # Statistical-only: dummy classifier so beam runs all K rounds
        _clf_fn = lambda t: "ai"

    result = csbp_loop(
        original_text  = text,
        original_label = original_label,
        classifier_fn  = _clf_fn,
        confidence_fn  = _conf_fn,
        K              = iterations,
        beam_width     = beam_width,
        n_candidates   = n_candidates,
        verbose        = verbose,
    )

    breakdown = result['score_breakdown']
    if verbose:
        print(f"\n[humanize] Final:  S={breakdown.get('S',0):.4f}  "
              f"classifier={breakdown.get('classifier',0):.4f}  "
              f"bpe={breakdown.get('bpe_disruption',0):.4f}  "
              f"ppl={breakdown.get('ppl',0):.1f}  "
              f"cos={breakdown.get('cosine',0):.4f}")
        print(f"[humanize] Success: {result['success']}  "
              f"round={result['round_found']}")

    return {
        'success':         result['success'],
        'original_text':   text,
        'original_label':  original_label,
        'humanized_text':  result['best_text'],
        'best_score':      result['best_score'],
        'round_found':     result['round_found'],
        'score_breakdown': breakdown,
        'beam_history':    result['beam_history'],
    }


# ═══════════════════════════════════════════════════════════════
# Batch Evaluation Helper (for generating your evaluation CSV)
# ═══════════════════════════════════════════════════════════════

def run_evaluation_batch(
    texts:        List[str],
    target_model: str,
    dataset_name: str  = "unknown",
    n_samples:    int  = 500,
    iterations:   int  = 5,
    n_candidates: int  = 20,
    beam_width:   int  = 8,
    output_csv:   Optional[str] = None,
    device:       Optional[str] = None,
) -> dict:
    """
    Run the full attack against a specific TextAttack model and report ASR.

    This is what you run to generate your evaluation metrics CSV and
    compare against Charmer's benchmark numbers.

    Example:
        results = run_evaluation_batch(
            texts        = your_agnews_texts[:500],
            target_model = "textattack/bert-base-uncased-ag-news",
            dataset_name = "agnews",
            n_samples    = 500,
        )
        print(f"ASR: {results['asr']:.1f}%")
    """
    if n_samples < len(texts):
        texts = random.sample(texts, n_samples)

    from transformers import pipeline as hf_pipeline
    dev = 0 if (torch.cuda.is_available() and device != 'cpu') else -1
    pipe = hf_pipeline(
        "text-classification",
        model=target_model,
        device=dev,
        truncation=True,
        max_length=512,
        return_all_scores=True,
    )
    clf_fn = lambda t: pipe(t[:512], truncation=True, return_all_scores=False)[0]['label']

    successes = 0
    rows = []

    for i, text in enumerate(texts):
        orig_label = clf_fn(text)
        conf_fn    = build_confidence_for_label(pipe, orig_label)

        result = csbp_loop(
            original_text  = text,
            original_label = orig_label,
            classifier_fn  = clf_fn,
            confidence_fn  = conf_fn,
            K              = iterations,
            beam_width     = beam_width,
            n_candidates   = n_candidates,
            verbose        = False,
        )

        if result['success']:
            successes += 1

        rows.append({
            'original_text':   text,
            'humanized_text':  result['best_text'],
            'original_label':  orig_label,
            'success':         result['success'],
            'round_found':     result['round_found'],
            'S':               result['best_score'],
        })

        print(f"[{i+1}/{len(texts)}] success={result['success']}  "
              f"round={result['round_found']}  "
              f"running ASR={100*successes/(i+1):.1f}%")

    asr = 100 * successes / len(texts)
    print(f"\n=== FINAL ASR on {dataset_name} ({target_model}): "
          f"{successes}/{len(texts)} = {asr:.2f}% ===")

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved → {output_csv}")

    return {'asr': asr, 'successes': successes, 'total': len(texts), 'rows': rows}


# ═══════════════════════════════════════════════════════════════
# File/Dataset Loading (unchanged from v2 — kept for compatibility)
# ═══════════════════════════════════════════════════════════════

def _coerce_text_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v if x is not None)
    try:
        return str(v)
    except Exception:
        return ""


def load_texts_from_file(path: Path, text_col_candidates: Optional[List[str]] = None) -> List[str]:
    path   = Path(path)
    suffix = path.suffix.lower()

    if suffix in {'.csv', '.tsv'}:
        delim = ',' if suffix == '.csv' else '\t'
        with open(path, newline='', encoding='utf-8') as fh:
            reader    = csv.DictReader(fh, delimiter=delim)
            rows      = list(reader)
            fieldnames = reader.fieldnames or []
        if not rows:
            return []
        if text_col_candidates:
            for c in text_col_candidates:
                if c in fieldnames:
                    return [_coerce_text_value(r.get(c, '')) for r in rows
                            if _coerce_text_value(r.get(c, '')).strip()]
        best_col, best_avg = None, 0.0
        for col in fieldnames:
            avg = sum(len(_coerce_text_value(r.get(col, ''))) for r in rows) / max(1, len(rows))
            if avg > best_avg:
                best_avg, best_col = avg, col
        if best_col:
            return [_coerce_text_value(r.get(best_col, '')) for r in rows
                    if _coerce_text_value(r.get(best_col, '')).strip()]
        return []

    if suffix == '.parquet':
        import pandas as pd
        df = pd.read_parquet(path)
        rows = df.to_dict(orient='records')
        fieldnames = df.columns.tolist()
        if text_col_candidates:
            for c in text_col_candidates:
                if c in fieldnames:
                    return [_coerce_text_value(r.get(c, '')) for r in rows
                            if _coerce_text_value(r.get(c, '')).strip()]
        best_col = max(fieldnames, key=lambda c: sum(
            len(_coerce_text_value(r.get(c, ''))) for r in rows))
        return [_coerce_text_value(r.get(best_col, '')) for r in rows
                if _coerce_text_value(r.get(best_col, '')).strip()]

    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        return [ln.strip() for ln in fh if ln.strip()]


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Humanize AI text to evade BERT/RoBERTa classifiers.")

    # Input
    parser.add_argument("text",         type=str, nargs='?', default=None)
    parser.add_argument("--hc3",        type=str, default=None)
    parser.add_argument("--m4",         type=str, default=None)
    parser.add_argument("--sample-size",type=int, default=10)
    parser.add_argument("--text-col",   type=str, nargs='*', default=None)

    # Target model — THE KEY FLAG
    parser.add_argument(
        "--target-model", type=str, default=None,
        help=(
            "HuggingFace model ID to attack.  Examples:\n"
            "  textattack/bert-base-uncased-ag-news\n"
            "  textattack/bert-base-uncased-SST-2\n"
            "  textattack/roberta-base-ag-news\n"
            "  textattack/bert-base-uncased-QNLI\n"
            "Leave empty for statistical-only mode (ZeroGPT etc.)."
        )
    )
    parser.add_argument("--dataset",    type=str, default="agnews",
                        choices=list({k[0] for k in TEXTATTACK_MODELS}),
                        help="Dataset name (for model auto-selection if --target-model not set)")
    parser.add_argument("--arch",       type=str, default="bert",
                        choices=["bert","roberta"],
                        help="Model architecture (used with --dataset for auto-selection)")

    # Search
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--cands",      type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--device",     type=str, choices=["cpu","mps","cuda"], default=None)

    # Output
    parser.add_argument("-o","--output",type=str,
                        default=str(Path.home() / "Downloads" / "humanized_output.csv"))
    parser.add_argument("--attack-type",type=str, default="csbp_v3")
    parser.add_argument("--model-label",type=str, default="charmer_beat")

    args = parser.parse_args()

    # ── Resolve target model ──────────────────────────────────────
    target_model = args.target_model
    if target_model is None:
        key = (args.dataset, args.arch)
        target_model = TEXTATTACK_MODELS.get(key)
        if target_model:
            print(f"[CLI] Auto-selected model: {target_model}")
        else:
            print(f"[CLI] No model found for {key} — running statistical-only mode.")

    # ── Collect input texts ───────────────────────────────────────
    examples: List[Tuple[str, str]] = []

    if args.hc3 or args.m4:
        all_texts = []
        for src in [args.hc3, args.m4]:
            if src:
                all_texts += load_texts_from_file(Path(src), args.text_col)
        random.shuffle(all_texts)
        for i, t in enumerate(all_texts[:args.sample_size], 1):
            examples.append((f"p{i:03d}", t))
    elif args.text:
        p = Path(args.text)
        if p.exists() and p.is_file():
            texts = load_texts_from_file(p, args.text_col)
            random.shuffle(texts)
            for i, t in enumerate(texts[:args.sample_size], 1):
                examples.append((f"p{i:03d}", t))
        else:
            examples.append(("p001", args.text))
    else:
        parser.error("No input provided.  Supply text, --hc3, or --m4.")

    # ── Run humanization ──────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pid, txt in examples:
        result = humanize(
            text          = txt,
            target_model  = target_model,
            iterations    = args.iterations,
            n_candidates  = args.cands,
            beam_width    = args.beam_width,
            device_override = args.device,
            verbose       = True,
        )
        rows.append({
            'pair_id':        pid,
            'original_text':  txt,
            'humanized_text': result['humanized_text'],
            'original_label': result['original_label'],
            'success':        result['success'],
            'round_found':    result['round_found'],
            'S':              result['best_score'],
            'attack_type':    args.attack_type,
            'target_model':   target_model or "statistical",
        })
        print(f"[{pid}] done — success={result['success']}")

    with open(out_path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_success = sum(1 for r in rows if r['success'])
    print(f"\n✓ ASR: {n_success}/{len(rows)} ({100*n_success/max(1,len(rows)):.1f}%)")
    print(f"✓ Saved → {out_path}")


if __name__ == "__main__":
    main()