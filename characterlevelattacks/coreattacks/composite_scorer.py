"""
Evasion-Oriented Composite Scorer (v2)
======================================

Scores perturbation candidates based on how effectively they EVADE
AI text detectors while maintaining readability.

KEY CHANGE FROM v1:
  v1 rewarded candidates that stayed SIMILAR to the original AI text.
  v2 rewards candidates that DEVIATE from AI statistical patterns.

Scoring dimensions:
  1. Evasion     — GPT-2 perplexity + token rank pushed into human-like range
  2. BPE disrupt — Token sequence fragmentation vs. original
  3. Watermark   — KGW green-list z-score suppression
  4. Readability — Perplexity guard (not gibberish)
  5. Coherence   — Semantic similarity floor (still means the same thing)
"""

import re
import math
import hashlib
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════
# Device & Model Loading
# ═══════════════════════════════════════════════════════════════

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"CompositeScorer v2 using device: {device}")

SBERT          = SentenceTransformer('all-mpnet-base-v2', device=device)
GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
GPT2_MODEL     = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
GPT2_MODEL.eval()

# ═══════════════════════════════════════════════════════════════
# KGW Watermark Parameters  (must match detector_evaluation/detectors/watermark/score.py)
# ═══════════════════════════════════════════════════════════════

KGW_HASH_KEY = 15485863
KGW_GAMMA    = 0.5

# Green-list cache – avoids recomputing the 25 k random draw per prev_token
_GREEN_CACHE: dict = {}

def _get_green_list(prev_tid: int) -> set:
    if prev_tid in _GREEN_CACHE:
        return _GREEN_CACHE[prev_tid]
    seed_bytes = f"{KGW_HASH_KEY}_{prev_tid}".encode("utf-8")
    seed  = int(hashlib.sha256(seed_bytes).hexdigest(), 16) % (2**32)
    rng   = np.random.RandomState(seed)
    n_green = int(GPT2_TOKENIZER.vocab_size * KGW_GAMMA)
    gl = set(rng.choice(GPT2_TOKENIZER.vocab_size, size=n_green, replace=False).tolist())
    if len(_GREEN_CACHE) < 20000:          # simple memory bound
        _GREEN_CACHE[prev_tid] = gl
    return gl


# ═══════════════════════════════════════════════════════════════
# Single-Pass GPT-2 Analysis
# ═══════════════════════════════════════════════════════════════

@torch.inference_mode()
def gpt2_analyze(text: str, max_length: int = 512) -> dict:
    """
    One forward pass through GPT-2 → perplexity + per-token prediction rank.
    Re-used by both the scorer and the pre-analysis helper.
    """
    enc = GPT2_TOKENIZER(text, return_tensors='pt',
                         truncation=True, max_length=max_length).to(device)
    input_ids = enc['input_ids']

    if input_ids.shape[1] < 2:
        return {'ppl': 1e6, 'avg_rank': 1000.0, 'ranks': [],
                'n_tokens': int(input_ids.shape[1])}

    out = GPT2_MODEL(**enc, labels=input_ids)
    ppl = float(math.exp(min(out.loss.item(), 20.0)))     # cap to avoid overflow

    logits  = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    sorted_idx = torch.argsort(logits[0], dim=-1, descending=True)

    ranks = []
    for i in range(targets.shape[1]):
        rank = (sorted_idx[i] == targets[0, i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

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
    """SBERT cosine similarity (semantic meaning preservation)."""
    embs = SBERT.encode([original, attacked], convert_to_numpy=True)
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


def evasion_score(ppl: float, avg_rank: float) -> float:
    """
    How well the text evades perplexity / rank-based detectors.

    Reference ranges (GPT-2 on English text):
        AI-generated :  PPL ≈ 15-50,   avg rank ≈ 3-15
        Human-written:  PPL ≈ 60-250,  avg rank ≈ 30-200+

    Returns 0-1.  Higher = looks more human-like to the detector.
    """
    ppl_comp  = 1.0 / (1.0 + math.exp(-(ppl - 55) / 25))
    rank_comp = 1.0 / (1.0 + math.exp(-(avg_rank - 20) / 12))
    return 0.65 * ppl_comp + 0.35 * rank_comp


def bpe_disruption_score(original: str, attacked: str) -> float:
    """
    Token-count inflation ratio.
    If attacked text tokenises into 2× the tokens → score = 1.0.
    """
    orig_n   = len(GPT2_TOKENIZER.encode(original, add_special_tokens=False))
    attack_n = len(GPT2_TOKENIZER.encode(attacked, add_special_tokens=False))
    if orig_n == 0:
        return 0.0
    return min(1.0, max(0.0, (attack_n / orig_n) - 1.0))


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
    Perplexity guard — v3 (aggressive):
      < 30   → suspiciously smooth (AI)  → low reward
      30+    → the higher the better     → full reward
      > 5000 → actual byte-soup garbage  → mild decay

    We WANT high perplexity — it means the text no longer
    looks AI-generated to statistical detectors.
    """
    if ppl < 30:
        return 0.2 + 0.8 * (ppl / 30)
    elif ppl <= 5000:
        return 1.0
    else:
        return max(0.3, 1.0 - (ppl - 5000) / 50000)


def coherence_gate(cosine: float, min_thresh: float = 0.20) -> float:
    """
    Multiplicative gate — v3 (relaxed).
    Threshold lowered to 0.20: invisible-char perturbations
    look identical to humans but can drop SBERT cosine because
    SBERT's own tokenizer also fragments on ZWSP/ZWJ.
    """
    if cosine >= min_thresh:
        return 1.0
    if cosine <= 0.0:
        return 0.0
    return (cosine / min_thresh) ** 2


# ═══════════════════════════════════════════════════════════════
# Main Composite Score
# ═══════════════════════════════════════════════════════════════

def composite_score(
    original: str,
    attacked: str,
    w_evasion:     float = 0.55,
    w_bpe:         float = 0.15,
    w_watermark:   float = 0.15,
    w_readability: float = 0.05,
    w_similarity:  float = 0.10,
) -> dict:
    """
    Evasion-oriented composite score.   Higher S = better attack candidate.

    Weights (defaults sum to 1.0):
      w_evasion     – push PPL / rank into human range       (primary)
      w_bpe         – reward tokenisation fragmentation
      w_watermark   – reward low KGW z-score
      w_readability – guard against gibberish
      w_similarity  – bonus for semantic similarity

    Coherence gate is applied multiplicatively so that even high-evasion
    candidates are rejected if they become nonsensical.
    """
    # ── Single GPT-2 pass for the attacked text ──
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

    # ── Weighted combination ──
    S_raw = (w_evasion * ev
             + w_bpe * bpe
             + w_watermark * wm
             + w_readability * read
             + w_similarity * max(0.0, min(1.0, cos)))

    # Coherence gate: multiplicative (never let gibberish win)
    S = S_raw * (0.3 + 0.7 * coh)

    return {
        'S':                  round(S, 4),
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
    """
    Map BPE token index → word index (space-split) using the Ġ prefix
    convention of GPT-2's byte-level BPE.
    """
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
    Pre-analyse the original text to identify:
      • priority_words   – word indices containing high-confidence tokens
                           (rank ≤ 5 → very predictable → strongest AI signal)
      • watermark_words  – word indices containing green-list tokens
                           (the watermark signal to suppress)

    Pass the returned dict to generate_candidates() so perturbations
    are targeted where they matter most.
    """
    analysis  = gpt2_analyze(text)
    token_ids = GPT2_TOKENIZER.encode(text, add_special_tokens=False)
    t2w       = _token_to_word_map(text)
    words     = text.split()

    # ── High-confidence tokens (rank ≤ threshold) ──
    #    ranks[i] predicts token i+1, so the "predicted" token index is i+1
    threshold = 5
    hi_conf = [i + 1 for i, r in enumerate(analysis['ranks']) if r <= threshold]
    # fallback: widen if nothing found
    if not hi_conf:
        threshold = 10
        hi_conf = [i + 1 for i, r in enumerate(analysis['ranks']) if r <= threshold]
    if not hi_conf:
        ranked = sorted(enumerate(analysis['ranks']), key=lambda x: x[1])
        hi_conf = [i + 1 for i, _ in ranked[:max(1, len(ranked) // 5)]]

    # ── Green-list tokens ──
    green_tok = []
    for i in range(1, len(token_ids)):
        if int(token_ids[i]) in _get_green_list(int(token_ids[i - 1])):
            green_tok.append(i)

    # ── Map to word indices ──
    priority_words  = {t2w[t] for t in hi_conf   if t in t2w}
    watermark_words = {t2w[t] for t in green_tok  if t in t2w}

    return {
        'ppl':              analysis['ppl'],
        'avg_rank':         analysis['avg_rank'],
        'ranks':            analysis['ranks'],
        'token_ids':        token_ids,
        'n_words':          len(words),
        'priority_words':   priority_words,
        'watermark_words':  watermark_words,
    }


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    orig     = "The stock market experienced significant volatility today."
    attacked = "Тhe stоck markеt еxperienced significant volatility tоday."

    print("\n--- Composite Score v2 (Evasion-Oriented) ---")
    result = composite_score(orig, attacked)
    for k, v in result.items():
        print(f"  {k:<22}: {v}")

    print("\n--- Original Text Pre-Analysis ---")
    ana = analyze_original(orig)
    for k, v in ana.items():
        if k not in ('ranks', 'token_ids'):
            print(f"  {k:<22}: {v}")
    print(f"  {'ranks (first 10)':<22}: {ana['ranks'][:10]}")
