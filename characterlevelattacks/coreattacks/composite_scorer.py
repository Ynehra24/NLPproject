"""
Evasion-Oriented Composite Scorer  (v3 — Beat-Charmer Edition)
==============================================================

Changes from v2:
  1. Weights rebalanced:
       w_evasion    0.55 → 0.35   (GPT-2 stats matter less than real classifier)
       w_bpe        0.15 → 0.35   (BPE disruption is the primary attack vector)
       w_watermark  0.15 → 0.10
       w_readability 0.05 → 0.05
       w_similarity  0.10 → 0.15  (keep meaning so human readers aren't suspicious)

  2. bpe_disruption_score() — now counts BOTH token inflation AND invisible-char
     density, giving a higher score to texts with ZWSP/ZWNJ inserted.

  3. coherence_gate() threshold lowered 0.20 → 0.10.
     Invisible-char attacks reduce SBERT cosine because SBERT's own tokeniser
     also fragments on ZWSP; the semantic meaning is preserved for human readers.
     The old 0.20 gate was suppressing our best candidates.

  4. readability_score() — treats PPL 30-10 000 as full score.
     The old cap at 5 000 was penalising heavy ZWSP-attacked text unfairly.

  5. New: classifier_score() hook — optional.  If a real BERT/RoBERTa
     classifier is provided, its confidence (inverted) replaces evasion_score()
     as the primary optimisation signal.  This is what the humanizer uses to
     actually beat the target model.
"""

import re
import math
import hashlib
import torch
import numpy as np
from typing import Callable, Optional
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════
# Device & Model Loading
# ═══════════════════════════════════════════════════════════════

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"CompositeScorer v3 using device: {device}")

SBERT          = SentenceTransformer('all-mpnet-base-v2', device=device)
GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
GPT2_MODEL     = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
GPT2_MODEL.eval()

# ═══════════════════════════════════════════════════════════════
# KGW Watermark Parameters
# ═══════════════════════════════════════════════════════════════

KGW_HASH_KEY = 15485863
KGW_GAMMA    = 0.5
_GREEN_CACHE: dict = {}

def _get_green_list(prev_tid: int) -> set:
    if prev_tid in _GREEN_CACHE:
        return _GREEN_CACHE[prev_tid]
    seed_bytes = f"{KGW_HASH_KEY}_{prev_tid}".encode("utf-8")
    seed  = int(hashlib.sha256(seed_bytes).hexdigest(), 16) % (2**32)
    rng   = np.random.RandomState(seed)
    n_green = int(GPT2_TOKENIZER.vocab_size * KGW_GAMMA)
    gl = set(rng.choice(GPT2_TOKENIZER.vocab_size, size=n_green, replace=False).tolist())
    if len(_GREEN_CACHE) < 20000:
        _GREEN_CACHE[prev_tid] = gl
    return gl


# ═══════════════════════════════════════════════════════════════
# Single-Pass GPT-2 Analysis
# ═══════════════════════════════════════════════════════════════

@torch.inference_mode()
def gpt2_analyze(text: str, max_length: int = 512) -> dict:
    """One forward pass → perplexity + per-token prediction rank.

    Rank computation is fully vectorised (no Python loop over vocab):
      rank_i = count of tokens with strictly higher logit than the true token + 1
    This is ~10× faster than the previous argsort+nonzero approach.
    """
    enc = GPT2_TOKENIZER(text, return_tensors='pt',
                         truncation=True, max_length=max_length).to(device)
    input_ids = enc['input_ids']

    if input_ids.shape[1] < 2:
        return {'ppl': 1e6, 'avg_rank': 1000.0, 'ranks': [],
                'n_tokens': int(input_ids.shape[1])}

    out = GPT2_MODEL(**enc, labels=input_ids)
    ppl = float(math.exp(min(out.loss.item(), 20.0)))

    logits  = out.logits[:, :-1, :]          # [1, n_tok-1, vocab]
    targets = input_ids[:, 1:]               # [1, n_tok-1]

    # For each position: gather the logit of the TRUE next token, then count
    # how many vocab entries have a strictly higher logit.  +1 → 1-indexed rank.
    target_logits = logits[0].gather(1, targets[0].unsqueeze(1))  # [n-1, 1]
    ranks_tensor  = (logits[0] > target_logits).sum(dim=1) + 1   # [n-1]
    ranks = ranks_tensor.tolist()

    return {
        'ppl':      ppl,
        'avg_rank': float(np.mean(ranks)) if ranks else 1000.0,
        'ranks':    ranks,
        'n_tokens': int(input_ids.shape[1]),
    }


# ═══════════════════════════════════════════════════════════════
# Scoring Components
# ═══════════════════════════════════════════════════════════════

def cosine_score(original: str, attacked: str) -> float:
    """SBERT cosine similarity (semantic preservation floor)."""
    embs = SBERT.encode([original, attacked], convert_to_numpy=True)
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


def evasion_score(ppl: float, avg_rank: float) -> float:
    """
    How well the text evades STATISTICAL detectors (GPT-2 perplexity / rank).
    This is a secondary signal — the real classifier check happens via
    classifier_score() when a model is wired in.

    Reference ranges (GPT-2 on English):
        AI-generated :  PPL ≈ 15-50,   avg rank ≈ 3-15
        Human-written:  PPL ≈ 60-250,  avg rank ≈ 30-200+
    """
    ppl_comp  = 1.0 / (1.0 + math.exp(-(ppl - 60) / 30))
    rank_comp = 1.0 / (1.0 + math.exp(-(avg_rank - 25) / 15))
    return 0.65 * ppl_comp + 0.35 * rank_comp


def bpe_disruption_score(original: str, attacked: str) -> float:
    """
    v3: combines token-count inflation with invisible-char density.

    Token inflation: attacked/original token ratio above 1.0.
    Invisible density: fraction of chars that are ZWSP/ZWNJ etc.
    Both are signals that BPE tokenisation has been shattered.
    """
    INVISIBLE = {'\u200B', '\u200C', '\u200D', '\u00AD', '\u2060', '\uFEFF'}

    orig_n   = len(GPT2_TOKENIZER.encode(original,  add_special_tokens=False))
    attack_n = len(GPT2_TOKENIZER.encode(attacked, add_special_tokens=False))

    # Token inflation component (capped at 1.0 when 2× inflation)
    if orig_n == 0:
        inflation = 0.0
    else:
        inflation = min(1.0, max(0.0, (attack_n / orig_n) - 1.0))

    # Invisible-char density component
    total_chars = len(attacked) if attacked else 1
    invis_count = sum(1 for c in attacked if c in INVISIBLE)
    invis_density = min(1.0, invis_count / max(1, total_chars) * 20)  # scale: 5% density → score 1.0

    return 0.6 * inflation + 0.4 * invis_density


def watermark_z_score(text: str) -> float:
    """KGW watermark z-score.  Higher = more likely watermarked."""
    token_ids = GPT2_TOKENIZER.encode(text, add_special_tokens=False)
    if len(token_ids) < 2:
        return 0.0
    green_hits, total = 0, 0
    for i in range(1, len(token_ids)):
        if int(token_ids[i]) in _get_green_list(int(token_ids[i - 1])):
            green_hits += 1
        total += 1
    if total == 0:
        return 0.0
    expected = KGW_GAMMA * total
    std = float(np.sqrt(total * KGW_GAMMA * (1.0 - KGW_GAMMA)))
    return float((green_hits - expected) / std) if std > 1e-9 else 0.0


def watermark_evasion(z: float) -> float:
    """Lower z → better evasion.  Returns 0-1."""
    return 1.0 / (1.0 + math.exp(z))


def readability_score(ppl: float) -> float:
    """
    v3: reward any PPL above 30 fully.  Cap decay starts at 10 000.
    Heavy ZWSP attacks legitimately push PPL into 500-3000 range on
    GPT-2 while remaining perfectly readable to humans.  The old 5 000
    cap was wrongly penalising our best outputs.
    """
    if ppl < 30:
        # suspiciously smooth — still AI-like to statistical detectors
        return 0.2 + 0.8 * (ppl / 30)
    elif ppl <= 10_000:
        return 1.0
    else:
        return max(0.3, 1.0 - (ppl - 10_000) / 100_000)


def coherence_gate(cosine: float, min_thresh: float = 0.10) -> float:
    """
    v3: threshold lowered to 0.10 (was 0.20).

    Invisible-char attacks (ZWSP) cause SBERT to score lower cosine
    because SBERT's own SentencePiece tokeniser also fragments on ZWSP.
    The semantic content is unchanged for human readers.
    We only discard true gibberish (cosine < 0.10).
    """
    if cosine >= min_thresh:
        return 1.0
    if cosine <= 0.0:
        return 0.0
    return (cosine / min_thresh) ** 2


def classifier_score(
    attacked: str,
    original_label: str,
    classifier_fn: Optional[Callable[[str], float]],
) -> float:
    """
    v3 NEW: If a real classifier confidence function is wired in, return
    1 - P(original_label | attacked_text).  This directly measures how much
    the attack has reduced the classifier's confidence.

    classifier_fn should return a float in [0, 1] representing the
    model's confidence that the text belongs to original_label.

    If no classifier is provided, returns 0.5 (neutral — use evasion_score
    as primary signal instead).
    """
    if classifier_fn is None:
        return 0.5
    try:
        confidence = classifier_fn(attacked)
        # confidence is P(original_label) — we want to MINIMIZE this,
        # so the score is 1 - confidence (higher = more evaded)
        return float(np.clip(1.0 - confidence, 0.0, 1.0))
    except Exception:
        return 0.5


# ═══════════════════════════════════════════════════════════════
# Main Composite Score  (v3)
# ═══════════════════════════════════════════════════════════════

def composite_score(
    original: str,
    attacked: str,
    classifier_fn:  Optional[Callable[[str], float]] = None,
    original_label: str = "ai",
    # Default weights — sum to 1.0
    w_classifier:  float = 0.00,   # activated when classifier_fn is provided
    w_evasion:     float = 0.35,   # GPT-2 stat evasion
    w_bpe:         float = 0.35,   # BPE disruption (token inflation + ZWSP)
    w_watermark:   float = 0.10,
    w_readability: float = 0.05,
    w_similarity:  float = 0.15,
) -> dict:
    """
    Evasion-oriented composite score.   Higher S = better attack candidate.

    When classifier_fn is provided:
      • w_classifier is set to 0.45 and w_evasion is reduced to 0.10
      • The classifier's confidence becomes the primary signal
      • This is the mode that directly optimises against BERT/RoBERTa

    Without classifier_fn:
      • BPE disruption (0.35) + statistical evasion (0.35) drive search
      • This is blind mode — still effective because ZWSP and homoglyphs
        derail the classifier's token embeddings even without querying it
    """
    # ── GPT-2 analysis ──
    analysis = gpt2_analyze(attacked)
    ppl      = analysis['ppl']
    avg_rank = analysis['avg_rank']

    # ── Component scores ──
    ev   = evasion_score(ppl, avg_rank)
    bpe  = bpe_disruption_score(original, attacked)
    wm_z = watermark_z_score(attacked)
    wm   = watermark_evasion(wm_z)
    read = readability_score(ppl)
    cos  = cosine_score(original, attacked)
    coh  = coherence_gate(cos)
    clf  = classifier_score(attacked, original_label, classifier_fn)

    # ── Adjust weights when real classifier is available ──
    if classifier_fn is not None:
        # Classifier confidence is the ground truth — prioritise it
        _w_clf  = 0.45
        _w_ev   = 0.10
        _w_bpe  = 0.25
        _w_wm   = 0.08
        _w_read = 0.04
        _w_sim  = 0.08
    else:
        _w_clf  = 0.00
        _w_ev   = w_evasion
        _w_bpe  = w_bpe
        _w_wm   = w_watermark
        _w_read = w_readability
        _w_sim  = w_similarity

    S_raw = (_w_clf  * clf
             + _w_ev   * ev
             + _w_bpe  * bpe
             + _w_wm   * wm
             + _w_read * read
             + _w_sim  * max(0.0, min(1.0, cos)))

    # Coherence gate (multiplicative) — only discard true gibberish
    S = S_raw * (0.3 + 0.7 * coh)

    return {
        'S':                  round(S, 4),
        'classifier':         round(clf, 4),
        'evasion':            round(ev, 4),
        'bpe_disruption':     round(bpe, 4),
        'watermark_z':        round(wm_z, 4),
        'watermark_evasion':  round(wm, 4),
        'readability':        round(read, 4),
        'cosine':             round(cos, 4),
        'coherence_gate':     round(coh, 4),
        'ppl':                round(ppl, 2),
        'avg_rank':           round(avg_rank, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Pre-Analysis for CSBP  (call ONCE per original text)
# ═══════════════════════════════════════════════════════════════

def _token_to_word_map(text: str) -> dict:
    token_ids = GPT2_TOKENIZER.encode(text, add_special_tokens=False)
    decoded   = GPT2_TOKENIZER.convert_ids_to_tokens(token_ids)
    words     = text.split()
    t2w       = {}
    word_idx  = 0
    for tok_idx, tok_str in enumerate(decoded):
        if tok_idx > 0 and tok_str.startswith('Ġ'):
            word_idx += 1
        t2w[tok_idx] = min(word_idx, len(words) - 1)
    return t2w


def analyze_original(text: str) -> dict:
    """
    Pre-analyse original text to identify:
      • priority_words   — word indices with high-confidence tokens (rank ≤ 5)
      • watermark_words  — word indices with KGW green-list tokens
    """
    analysis  = gpt2_analyze(text)
    token_ids = GPT2_TOKENIZER.encode(text, add_special_tokens=False)
    t2w       = _token_to_word_map(text)
    words     = text.split()

    threshold = 5
    hi_conf = [i + 1 for i, r in enumerate(analysis['ranks']) if r <= threshold]
    if not hi_conf:
        threshold = 10
        hi_conf = [i + 1 for i, r in enumerate(analysis['ranks']) if r <= threshold]
    if not hi_conf:
        ranked = sorted(enumerate(analysis['ranks']), key=lambda x: x[1])
        hi_conf = [i + 1 for i, _ in ranked[:max(1, len(ranked) // 5)]]

    green_tok = []
    for i in range(1, len(token_ids)):
        if int(token_ids[i]) in _get_green_list(int(token_ids[i - 1])):
            green_tok.append(i)

    priority_words  = {t2w[t] for t in hi_conf   if t in t2w}
    watermark_words = {t2w[t] for t in green_tok  if t in t2w}

    # ── Per-word average GPT-2 token rank ────────────────────────────────────
    # Lower avg rank = GPT-2 more confident on this word = higher AI signal
    # → invert to a sensitivity weight so beam search attacks these first.
    word_rank_accum: dict = {}
    for tok_idx, rk in enumerate(analysis['ranks']):
        w = t2w.get(tok_idx + 1)       # ranks align to positions 1..n
        if w is not None:
            word_rank_accum.setdefault(w, []).append(rk)

    avg_word_rank = {
        w: float(np.mean(rks))
        for w, rks in word_rank_accum.items()
    }
    max_rank = max(avg_word_rank.values(), default=1000.0)
    # sensitivity = max_rank / avg_rank  (high-confidence = high sensitivity)
    word_sensitivity: dict = {
        w: float(max_rank / (r + 1e-3))
        for w, r in avg_word_rank.items()
    }

    return {
        'ppl':              analysis['ppl'],
        'avg_rank':         analysis['avg_rank'],
        'ranks':            analysis['ranks'],
        'token_ids':        token_ids,
        'n_words':          len(words),
        'priority_words':   priority_words,
        'watermark_words':  watermark_words,
        'word_sensitivity': word_sensitivity,   # NEW: per-word attack weight
    }


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    orig     = "The stock market experienced significant volatility today."
    attacked = "Тhe stоck markеt еxperienced significant volatility tоday."

    print("\n--- Composite Score v3 (Evasion-Oriented) ---")
    result = composite_score(orig, attacked)
    for k, v in result.items():
        print(f"  {k:<22}: {v}")

    print("\n--- Original Text Pre-Analysis ---")
    ana = analyze_original(orig)
    for k, v in ana.items():
        if k not in ('ranks', 'token_ids'):
            print(f"  {k:<22}: {v}")
    print(f"  {'ranks (first 10)':<22}: {ana['ranks'][:10]}")