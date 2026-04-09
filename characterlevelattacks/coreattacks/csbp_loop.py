"""
CSBP v2 – Evasion-Oriented Beam Search
=======================================

Character-level Search-Based Perturbation with targeted strategies:

  1. BPE-Break        – perturb at positions that maximally fragment BPE tokens
  2. High-Confidence  – target words whose tokens the model predicted with
                        near-certainty (the strongest AI signal)
  3. Whitespace       – insert invisible Unicode chars (ZWSP, ZWJ, soft-hyphen)
                        inside words to split BPE pre-tokenisation chunks
  4. Emoji Attach     – concatenate emojis directly to words (no spaces) to
                        disrupt the token-prediction chain
  5. Watermark        – target green-list tokens specifically to flip them to
                        red-list, suppressing the KGW z-score
  6. Scorched Earth   – ALL strategies applied simultaneously at maximum
                        intensity to every eligible word

The composite scorer (v2) then ranks candidates by how effectively they
EVADE detectors, not how closely they preserve the original AI text.
"""

import random
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from composite_scorer import (
    composite_score, analyze_original, GPT2_TOKENIZER,
    _get_green_list, KGW_HASH_KEY, KGW_GAMMA,
)
from homoglyph_attack import (
    HOMOGLYPH_MAP, DIACRITIC_MAP,
    apply_homoglyph, apply_diacritic, is_eligible,
)

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Invisible chars that break GPT-2's pre-tokeniser regex (?\p{L}+)
# because they are Unicode category Cf (Format), not L (Letter).
# Inserting one inside a word splits the word into separate pre-token chunks.
INVISIBLE_BREAKERS = [
    '\u200B',   # zero-width space
    '\u200C',   # zero-width non-joiner
    '\u200D',   # zero-width joiner  (from emoji ZWJ sequences)
    '\u00AD',   # soft hyphen
    '\u2060',   # word joiner
    '\uFEFF',   # zero-width no-break space (BOM)
]

# Emojis used for context-disruption attachment.
# Multi-byte sequences that shift the surrounding token IDs.
ATTACK_EMOJIS = [
    '🔥', '💡', '🚀', '📊', '🔍', '⚡', '🎯', '💻',
    '🌐', '🔑', '🔧', '📈', '🧪', '🧬', '📝', '🛡',
    '\U0001F468\u200D\U0001F4BB',      # 👨‍💻  (ZWJ sequence = long byte repr)
    '\U0001F469\u200D\U0001F52C',      # 👩‍🔬
    '\U0001F9D1\u200D\U0001F680',      # 🧑‍🚀
]

# Attack strategies — scorched_earth is weighted 3x because it's the most
# effective against statistical detectors.
STRATEGIES = ['bpe_break', 'high_conf', 'whitespace',
              'emoji_attach', 'watermark', 'combined',
              'scorched_earth', 'scorched_earth', 'scorched_earth']


# ═══════════════════════════════════════════════════════════════
# BPE Vulnerability Analysis  (cached per word)
# ═══════════════════════════════════════════════════════════════

_BPE_VULN_CACHE: dict = {}

def find_bpe_vulnerable_positions(word: str) -> List[Tuple[int, int, str, str]]:
    """
    For a given word, find character positions where a perturbation causes
    maximum BPE tokenisation disruption.

    Returns list of (position, delta_tokens, replacement_char, action)
    sorted by descending delta.
      action = 'insert'  → invisible char inserted BEFORE position
      action = 'replace' → character at position is substituted
    """
    if word in _BPE_VULN_CACHE:
        return _BPE_VULN_CACHE[word]

    orig_n  = len(GPT2_TOKENIZER.encode(word, add_special_tokens=False))
    results = []

    for i in range(len(word)):
        ch = word[i]

        # ── Try invisible-char insertion at this position ──
        for breaker in ['\u200B', '\u200C', '\u200D']:
            test = word[:i] + breaker + word[i:]
            delta = len(GPT2_TOKENIZER.encode(test, add_special_tokens=False)) - orig_n
            if delta > 0:
                results.append((i, delta, breaker, 'insert'))
                break           # one breaker per position is enough

        # ── Try homoglyph substitution ──
        lookup = ch if ch in HOMOGLYPH_MAP else ch.lower() if ch.lower() in HOMOGLYPH_MAP else None
        if lookup:
            repl = HOMOGLYPH_MAP[lookup]
            test = word[:i] + repl + word[i+1:]
            delta = len(GPT2_TOKENIZER.encode(test, add_special_tokens=False)) - orig_n
            if delta > 0:
                results.append((i, delta, repl, 'replace'))

        # ── Try diacritic substitution ──
        lookup_d = ch if ch in DIACRITIC_MAP else ch.lower() if ch.lower() in DIACRITIC_MAP else None
        if lookup_d:
            repl_d = DIACRITIC_MAP[lookup_d][0]       # deterministic for cache
            test = word[:i] + repl_d + word[i+1:]
            delta = len(GPT2_TOKENIZER.encode(test, add_special_tokens=False)) - orig_n
            if delta > 0:
                results.append((i, delta, repl_d, 'replace'))

    results.sort(key=lambda x: -x[1])       # most disruptive first

    if len(_BPE_VULN_CACHE) < 50000:
        _BPE_VULN_CACHE[word] = results
    return results


# ═══════════════════════════════════════════════════════════════
# Perturbation Helpers
# ═══════════════════════════════════════════════════════════════

def apply_bpe_targeted(word: str, max_edits: int = 3) -> str:
    """Apply the most BPE-disruptive perturbations to a word."""
    vulns = find_bpe_vulnerable_positions(word)
    if not vulns:
        fn = random.choice([apply_homoglyph, apply_diacritic])
        return fn(word, rate=0.5)

    chars      = list(word)
    insertions = []          # (pos_in_original, char)
    edits_done = 0

    for pos, _delta, repl, action in vulns:
        if edits_done >= max_edits:
            break
        if action == 'replace' and pos < len(chars):
            chars[pos] = repl
            edits_done += 1
        elif action == 'insert':
            insertions.append((pos, repl))
            edits_done += 1

    result = ''.join(chars)
    # apply insertions in reverse so indices stay valid
    for pos, ch in sorted(insertions, key=lambda x: -x[0]):
        if pos <= len(result):
            result = result[:pos] + ch + result[pos:]
    return result


def insert_whitespace_attack(word: str) -> str:
    """Insert 2-3 invisible Unicode chars at random interior positions."""
    if len(word) <= 2:
        return word
    chars    = list(word)
    n_insert = random.randint(2, min(3, len(chars) - 1))
    positions = random.sample(range(1, len(chars)), min(n_insert, len(chars) - 1))
    for pos in sorted(positions, reverse=True):
        chars.insert(pos, random.choice(INVISIBLE_BREAKERS))
    return ''.join(chars)


def insert_whitespace_attack_heavy(word: str) -> str:
    """Insert 3-5 invisible Unicode chars — used by scorched_earth.
    Targets every 2-3 character boundary to maximally shatter BPE."""
    if len(word) <= 1:
        return word
    chars = list(word)
    max_pos = len(chars) - 1
    n_insert = min(max_pos, random.randint(3, 5))
    positions = random.sample(range(1, len(chars)), min(n_insert, max_pos))
    for pos in sorted(positions, reverse=True):
        chars.insert(pos, random.choice(INVISIBLE_BREAKERS))
    return ''.join(chars)


def attach_emoji(word: str) -> str:
    """Attach an emoji directly to a word (no space)."""
    em = random.choice(ATTACK_EMOJIS)
    return (word + em) if random.random() < 0.5 else (em + word)


def scorched_earth_word(word: str) -> str:
    """
    SCORCHED EARTH: Apply ALL available attacks to a single word
    simultaneously for maximum BPE destruction.

    1. Homoglyph substitution at high rate
    2. Diacritic substitution on remaining eligible chars
    3. Heavy invisible-char insertion
    4. Optionally attach an emoji
    """
    # Step 1: heavy homoglyph pass
    result = apply_homoglyph(word, rate=0.8)
    # Step 2: diacritic pass on whatever's left
    result = apply_diacritic(result, rate=0.6)
    # Step 3: heavy invisible char insertion
    result = insert_whitespace_attack_heavy(result)
    # Step 4: attach emoji with 40% probability
    if random.random() < 0.4:
        result = attach_emoji(result)
    return result


# ═══════════════════════════════════════════════════════════════
# Candidate Generation (multi-strategy)
# ═══════════════════════════════════════════════════════════════

def generate_candidates(
    text:          str,
    original_text: str,
    analysis:      dict,
    n_candidates:  int  = 10,
) -> List[str]:
    """
    Generate n_candidates perturbations using targeted strategies.

    `analysis` comes from composite_scorer.analyze_original() and provides
    priority_words / watermark_words indices so we attack WHERE it matters.
    """
    candidates = set()
    words      = text.split()
    max_idx    = len(words) - 1

    priority_set  = {i for i in analysis.get('priority_words',  set()) if i <= max_idx}
    watermark_set = {i for i in analysis.get('watermark_words', set()) if i <= max_idx}
    eligible      = [i for i, w in enumerate(words) if is_eligible(w)]

    if not eligible:
        return [text]

    for _ in range(n_candidates * 4):        # oversample, deduplicate
        strategy  = random.choice(STRATEGIES)
        new_words = list(words)

        # ── Strategy 1: BPE-break ─────────────────────────────────
        if strategy == 'bpe_break':
            # Concentrate heavy perturbation on ≤ 20 % of words
            n_targets = max(1, len(eligible) // 5)
            targets   = random.sample(eligible, min(n_targets, len(eligible)))
            for idx in targets:
                new_words[idx] = apply_bpe_targeted(
                    words[idx], max_edits=random.randint(2, 4))

        # ── Strategy 2: High-confidence token targets ─────────────
        elif strategy == 'high_conf':
            targets = list(priority_set & set(eligible))
            if not targets:
                targets = random.sample(eligible, min(3, len(eligible)))
            random.shuffle(targets)
            for idx in targets[:max(1, len(targets) // 2)]:
                fn = random.choice([apply_bpe_targeted,
                                    lambda w: apply_homoglyph(w, rate=0.7),
                                    lambda w: apply_diacritic(w, rate=0.7)])
                new_words[idx] = fn(words[idx])

        # ── Strategy 3: Invisible whitespace insertion ────────────
        elif strategy == 'whitespace':
            n_targets = max(1, len(eligible) // 5)
            targets   = random.sample(eligible, min(n_targets, len(eligible)))
            for idx in targets:
                new_words[idx] = insert_whitespace_attack(words[idx])

        # ── Strategy 4: Emoji attachment (context disruption) ─────
        elif strategy == 'emoji_attach':
            n_emoji = random.randint(1, max(1, len(words) // 10))
            targets = random.sample(eligible, min(n_emoji, len(eligible)))
            for idx in targets:
                new_words[idx] = attach_emoji(words[idx])
                # also perturb the word itself
                if random.random() < 0.6:
                    fn = random.choice([apply_homoglyph, apply_diacritic])
                    new_words[idx] = fn(new_words[idx], rate=0.3)

        # ── Strategy 5: Watermark-targeted perturbation ───────────
        elif strategy == 'watermark':
            targets = list(watermark_set & set(eligible))
            if not targets:
                targets = random.sample(eligible, min(3, len(eligible)))
            random.shuffle(targets)
            for idx in targets[:max(1, len(targets) // 3)]:
                # try several perturbations; keep the one that changes token IDs most
                best, best_delta = words[idx], 0
                for _ in range(6):
                    fn   = random.choice([apply_homoglyph, apply_diacritic,
                                          insert_whitespace_attack])
                    if fn in (apply_homoglyph, apply_diacritic):
                        attempt = fn(words[idx], rate=random.uniform(0.4, 0.8))
                    else:
                        attempt = insert_whitespace_attack(words[idx])
                    orig_toks = GPT2_TOKENIZER.encode(words[idx], add_special_tokens=False)
                    new_toks  = GPT2_TOKENIZER.encode(attempt, add_special_tokens=False)
                    delta     = abs(len(new_toks) - len(orig_toks))
                    if delta > best_delta:
                        best, best_delta = attempt, delta
                new_words[idx] = best

        # ── Strategy 6: Combined (mix 2-3 strategies) ─────────────
        elif strategy == 'combined':
            # a) BPE-break a few words
            n_bpe     = max(1, len(eligible) // 8)
            bpe_tgts  = random.sample(eligible, min(n_bpe, len(eligible)))
            for idx in bpe_tgts:
                new_words[idx] = apply_bpe_targeted(words[idx], max_edits=2)

            # b) attach emoji to 1-2 other words
            remaining = [i for i in eligible if i not in bpe_tgts]
            if remaining:
                n_em = random.randint(0, min(2, len(remaining)))
                for idx in random.sample(remaining, min(n_em, len(remaining))):
                    new_words[idx] = attach_emoji(words[idx])

            # c) whitespace-attack one more word
            remaining2 = [i for i in eligible if i not in bpe_tgts]
            if remaining2 and random.random() < 0.5:
                idx = random.choice(remaining2)
                new_words[idx] = insert_whitespace_attack(words[idx])

        # ── Strategy 7: SCORCHED EARTH ────────────────────────────
        #    Apply ALL attacks to EVERY eligible word at maximum intensity.
        #    This is the nuclear option — maximum BPE destruction.
        elif strategy == 'scorched_earth':
            # Determine intensity: perturb 40-80% of eligible words
            intensity = random.uniform(0.4, 0.8)
            n_targets = max(1, int(len(eligible) * intensity))
            targets   = random.sample(eligible, min(n_targets, len(eligible)))
            for idx in targets:
                new_words[idx] = scorched_earth_word(words[idx])

        # ── Collect ───────────────────────────────────────────────
        cand = ' '.join(new_words)
        if cand != text:
            candidates.add(cand)
        if len(candidates) >= n_candidates:
            break

    return list(candidates)[:n_candidates]


# ═══════════════════════════════════════════════════════════════
# Beam Search Core
# ═══════════════════════════════════════════════════════════════

@dataclass(order=True)
class Beam:
    """A single candidate in the beam."""
    score:    float
    text:     str       = field(compare=False)
    round_no: int       = field(compare=False, default=0)
    history:  List[str] = field(compare=False, default_factory=list)

    def __post_init__(self):
        self.history = self.history + [self.text]


def misclassifies(
    text:           str,
    classifier_fn:  Callable[[str], str],
    original_label: str,
) -> bool:
    """True if the classifier's prediction changed (attack succeeded)."""
    return classifier_fn(text) != original_label


def csbp_loop(
    original_text:   str,
    original_label:  str,
    classifier_fn:   Callable[[str], str],     # fn(text) → label
    K:               int   = 5,                # max rounds
    beam_width:      int   = 5,                # beams kept per round
    n_candidates:    int   = 10,               # candidates per beam per round
    score_weights:   Optional[dict] = None,    # override composite_score weights
    verbose:         bool  = True,
) -> dict:
    """
    CSBP v2 beam-search loop.

    Changes from v1
    ────────────────
    • Pre-analyses the original text ONCE via `analyze_original()` to identify
      high-confidence and watermark-bearing words.
    • Passes analysis to `generate_candidates()` so perturbations are
      strategically targeted (not random).
    • composite_score() now optimises for detector EVASION, not similarity.
    • Implements scorched_earth strategy for maximum BPE destruction.
    """
    weights      = score_weights or {}
    beam_history = []

    # ── Pre-analyse original text (runs once) ──
    analysis = analyze_original(original_text)
    if verbose:
        print(f"[CSBP] Pre-analysis:  PPL={analysis['ppl']:.1f}  "
              f"avg_rank={analysis['avg_rank']:.1f}  "
              f"priority_words={len(analysis['priority_words'])}  "
              f"watermark_words={len(analysis['watermark_words'])}")

    # ── Initialise beam ──
    init_score = composite_score(original_text, original_text, **weights)['S']
    beams      = [Beam(score=init_score, text=original_text, round_no=0)]

    best_attack = None
    best_score  = -1.0

    for k in range(1, K + 1):
        if verbose:
            print(f"\n[CSBP] Round {k}/{K}  |  active beams: {len(beams)}")

        round_candidates: List[Beam] = []

        for beam in beams:
            cands = generate_candidates(
                beam.text, original_text, analysis, n_candidates)

            for cand_text in cands:
                # Score against ORIGINAL to prevent drift
                result = composite_score(original_text, cand_text, **weights)
                S      = result['S']

                # Early-stop: misclassification achieved
                if misclassifies(cand_text, classifier_fn, original_label):
                    if verbose:
                        print(f"  ✓ Attack succeeded at round {k}  S={S:.4f}")
                        print(f"    evasion={result['evasion']:.4f}  "
                              f"bpe={result['bpe_disruption']:.4f}  "
                              f"wm_z={result['watermark_z']:.4f}  "
                              f"cos={result['cosine']:.4f}")
                        print(f"    {cand_text[:120]}...")
                    return {
                        'success':         True,
                        'best_text':       cand_text,
                        'best_score':      S,
                        'round_found':     k,
                        'score_breakdown': result,
                        'beam_history':    beam_history,
                    }

                round_candidates.append(
                    Beam(score=S, text=cand_text, round_no=k,
                         history=beam.history))

        if not round_candidates:
            break

        # Prune to top-B beams
        round_candidates.sort(reverse=True)
        beams = round_candidates[:beam_width]

        top = beams[0]
        beam_history.append({
            'round': k,
            'top_score': top.score,
            'top_text': top.text,
        })

        if verbose:
            print(f"  top S={top.score:.4f}  |  {top.text[:100]}...")

        if top.score > best_score:
            best_score  = top.score
            best_attack = top.text

    # ── Loop exhausted without misclassification ──
    final = composite_score(original_text,
                            best_attack or original_text, **weights)
    return {
        'success':         False,
        'best_text':       best_attack or original_text,
        'best_score':      best_score,
        'round_found':     None,
        'score_breakdown': final,
        'beam_history':    beam_history,
    }


def run_csbp_batch(
    texts:          List[str],
    labels:         List[str],
    classifier_fn:  Callable[[str], str],
    K:              int  = 5,
    beam_width:     int  = 5,
    n_candidates:   int  = 10,
    verbose:        bool = False,
) -> List[dict]:
    """Run CSBP over a batch of (text, label) pairs."""
    results = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        print(f"\n[{i+1}/{len(texts)}] label={label}  text={text[:60]}...")
        result = csbp_loop(
            original_text  = text,
            original_label = label,
            classifier_fn  = classifier_fn,
            K              = K,
            beam_width     = beam_width,
            n_candidates   = n_candidates,
            verbose        = verbose,
        )
        result['original_text']  = text
        result['original_label'] = label
        results.append(result)

    n_success = sum(r['success'] for r in results)
    print(f"\n=== ASR: {n_success}/{len(results)} "
          f"({100*n_success/len(results):.1f}%) ===")
    return results


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def dummy_classifier(text: str) -> str:
        """Always 'positive' unless 'bad' appears."""
        return 'negative' if 'bad' in text.lower() else 'positive'

    original = "The film was genuinely bad and quite disappointing overall."
    label    = dummy_classifier(original)

    print(f"Original label : {label}")
    print(f"Original text  : {original}\n")

    result = csbp_loop(
        original_text  = original,
        original_label = label,
        classifier_fn  = dummy_classifier,
        K              = 5,
        beam_width     = 3,
        n_candidates   = 8,
        verbose        = True,
    )

    print("\n--- Final Result ---")
    print(f"Success      : {result['success']}")
    print(f"Round found  : {result['round_found']}")
    print(f"Best score S : {result['best_score']:.4f}")
    print(f"Best text    : {result['best_text']}")
    print(f"Score terms  : {result['score_breakdown']}")
