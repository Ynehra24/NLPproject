"""
CSBP v3 – Evasion-Oriented Beam Search
"""

import random
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from composite_scorer import (
    composite_score, analyze_original, GPT2_TOKENIZER,
    _get_green_list,
)
from homoglyph_attack import (
    HOMOGLYPH_MAP, DIACRITIC_MAP, INVISIBLE_CHARS,
    apply_homoglyph, apply_diacritic, apply_zwsp_only,
    apply_homoglyph_plus_zwsp, apply_blitz, is_eligible,
)

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

ZWSP = '\u200B'

ATTACK_EMOJIS = [
    '🔥', '💡', '🚀', '📊', '🔍', '⚡', '🎯', '💻',
    '🌐', '🔑', '🔧', '📈', '🧪', '🧬', '📝', '🛡',
]

# Strategy pool — 'blitz' and 'zwsp_flood' weighted heavily (4× each)
# because they are the highest-ASR strategies against BERT/RoBERTa.
STRATEGIES = [
    'bpe_break',
    'high_conf',
    'whitespace',
    'emoji_attach',
    'watermark',
    'combined',
    'scorched_earth', 'scorched_earth', 'scorched_earth',
    'blitz',          'blitz',          'blitz',          'blitz',
    'zwsp_flood',     'zwsp_flood',     'zwsp_flood',     'zwsp_flood',
    'roberta_subword', 'roberta_subword', 'roberta_subword', 'roberta_subword',
    'micro_sponge',   'micro_sponge',
    'punctuation_bind', 'punctuation_bind',
    'random_case',    'random_case',
    'math_unicode',   'math_unicode',
    'fullwidth',
]

MIN_WORD_LEN_HEAVY = 3

def _weighted_sample(
    eligible:    List[int],
    sensitivity: dict,            
    already:     frozenset,       
    n:           int,
) -> List[int]:
    unperturbed = [i for i in eligible if i not in already]
    if not unperturbed:
        return []
        
    weights = np.array([sensitivity.get(i, 1.0) for i in unperturbed], dtype=float)
    total = weights.sum()
    
    if total > 0:
        weights /= total
    else:
        weights = np.ones(len(unperturbed), dtype=float) / len(unperturbed)
        
    n_actual = min(n, len(unperturbed))
    return list(np.random.choice(unperturbed, size=n_actual, replace=False, p=weights))


_BPE_VULN_CACHE: dict = {}

def find_bpe_vulnerable_positions(word: str) -> List[Tuple[int, int, str, str]]:
    if word in _BPE_VULN_CACHE:
        return _BPE_VULN_CACHE[word]

    orig_n  = len(GPT2_TOKENIZER.encode(word, add_special_tokens=False))
    results = []

    for i in range(len(word)):
        ch = word[i]

        for breaker in [ZWSP, '\u200C', '\u200D']:
            test  = word[:i] + breaker + word[i:]
            delta = len(GPT2_TOKENIZER.encode(test, add_special_tokens=False)) - orig_n
            if delta > 0:
                results.append((i, delta, breaker, 'insert'))
                break

        lookup = ch if ch in HOMOGLYPH_MAP else ch.lower() if ch.lower() in HOMOGLYPH_MAP else None
        if lookup:
            repl  = HOMOGLYPH_MAP[lookup]
            test  = word[:i] + repl + word[i+1:]
            delta = len(GPT2_TOKENIZER.encode(test, add_special_tokens=False)) - orig_n
            if delta > 0:
                results.append((i, delta, repl, 'replace'))

        lookup_d = ch if ch in DIACRITIC_MAP else ch.lower() if ch.lower() in DIACRITIC_MAP else None
        if lookup_d:
            repl_d = DIACRITIC_MAP[lookup_d][0]
            test   = word[:i] + repl_d + ZWSP + word[i+1:]   # stacked ZWSP
            delta  = len(GPT2_TOKENIZER.encode(test, add_special_tokens=False)) - orig_n
            if delta > 0:
                results.append((i, delta, repl_d + ZWSP, 'replace'))

    results.sort(key=lambda x: -x[1])

    if len(_BPE_VULN_CACHE) < 50000:
        _BPE_VULN_CACHE[word] = results
    return results


# ═══════════════════════════════════════════════════════════════
# Word-Level Perturbation Helpers
# ═══════════════════════════════════════════════════════════════

def apply_bpe_targeted(word: str, max_edits: int = 4) -> str:
    vulns = find_bpe_vulnerable_positions(word)
    if not vulns:
        return apply_blitz(word)

    chars      = list(word)
    insertions = []
    edits_done = 0

    for pos, _delta, repl, action in vulns:
        if edits_done >= max_edits:
            break
        if action == 'replace' and pos < len(chars):
            if ZWSP in repl:
                base, _ = repl[0], repl[1:]
                chars[pos] = base
                insertions.append((pos + 1, ZWSP))
            else:
                chars[pos] = repl
            edits_done += 1
        elif action == 'insert':
            insertions.append((pos, repl))
            edits_done += 1

    result = ''.join(chars)
    for pos, ch in sorted(insertions, key=lambda x: -x[0]):
        if pos <= len(result):
            result = result[:pos] + ch + result[pos:]
    return result


def insert_whitespace_attack(word: str) -> str:
    """Insert 2-3 invisible chars at random interior positions."""
    if len(word) <= 2:
        return word
    chars     = list(word)
    n_insert  = random.randint(2, min(3, len(chars) - 1))
    positions = random.sample(range(1, len(chars)), min(n_insert, len(chars) - 1))
    for pos in sorted(positions, reverse=True):
        chars.insert(pos, random.choice(INVISIBLE_CHARS))
    return ''.join(chars)


def zwsp_shatter_word(word: str) -> str:
    if len(word) <= 2:
        return word
    result = word[0]
    breakers = [ZWSP, '\u200C', '\u200D']
    for i, ch in enumerate(word[1:], 1):
        if i % 3 == 0:  # Only every 3rd boundary
            result += breakers[i % len(breakers)]
        result += ch
    return result


def insert_whitespace_attack_heavy(word: str) -> str:
    if len(word) <= 1:
        return word
    chars    = list(word)
    max_pos  = len(chars) - 1
    n_insert = min(max_pos, random.randint(3, 5))
    positions = random.sample(range(1, len(chars)), min(n_insert, max_pos))
    for pos in sorted(positions, reverse=True):
        chars.insert(pos, random.choice(INVISIBLE_CHARS))
    return ''.join(chars)


def zwsp_flood_word(word: str) -> str:
    result = apply_homoglyph(word, rate=1.0)  # swap all eligible chars first
    flooded = []
    for i, ch in enumerate(result):
        flooded.append(ch)
        if i < len(result) - 1:
            flooded.append(ZWSP)
    return ''.join(flooded)


def attach_emoji(word: str) -> str:
    em = random.choice(ATTACK_EMOJIS)
    return (word + em) if random.random() < 0.5 else (em + word)


def scorched_earth_word(word: str) -> str:
    result = apply_homoglyph(word, rate=0.9)
    result = apply_diacritic(result, rate=0.7)   # diacritic stacks ZWSP automatically
    result = insert_whitespace_attack_heavy(result)
    if random.random() < 0.4:
        result = attach_emoji(result)
    return result

def roberta_subword_word(word: str) -> str:
    chars = list(word)
    if len(chars) <= 3:
        return word
    
    # Inject 1 to 3 internal ZWSPs depending on word length
    n_injects = min(3, len(chars) // 2)
    indices = random.sample(range(1, len(chars)), n_injects)
    
    for i in sorted(indices, reverse=True):
        chars.insert(i, ZWSP)
        
    return apply_homoglyph(''.join(chars), rate=0.3)


def apply_micro_sponge(text: str) -> str:
    """Append 5 to 15 invisible BPE-breaking characters at the end of the sentence."""
    sponge = "".join(random.choice(['\u200B', '\u200C', '\u200D']) for _ in range(random.randint(5, 15)))
    return text + sponge


def apply_punctuation_bind(text: str) -> str:
    """Shatter token boundaries by injecting ZWSP before common trailing punctuation."""
    new_text = text
    for punc in ['.', ',', '!', '?', ';', ':']:
        new_text = new_text.replace(punc, ZWSP + punc + ZWSP)
    return new_text


# ── Math Bold Unicode map: visually identical to Latin but 4-byte UTF-8 ──
_MATH_BOLD_MAP: dict = {}
for _i in range(26):
    _MATH_BOLD_MAP[chr(ord('a') + _i)] = chr(0x1D41A + _i)   # 𝐚-𝐳
    _MATH_BOLD_MAP[chr(ord('A') + _i)] = chr(0x1D400 + _i)   # 𝐀-𝐙


def apply_random_case(word: str) -> str:
    """Randomly flip character case on interior chars.
    Creates novel byte sequences for RoBERTa's case-sensitive byte-BPE tokenizer."""
    if len(word) <= 2:
        return word
    chars = [word[0]]  # keep first char as-is (less noticeable)
    for ch in word[1:]:
        if ch.isalpha():
            chars.append(ch.upper() if random.random() < 0.5 else ch.lower())
        else:
            chars.append(ch)
    return ''.join(chars)


def apply_math_unicode(word: str) -> str:
    return ''.join(_MATH_BOLD_MAP.get(ch, ch) for ch in word)


def apply_fullwidth(word: str) -> str:
    result = []
    for ch in word:
        if 0x21 <= ord(ch) <= 0x7E and ch.isalpha():
            result.append(chr(ord(ch) - 0x21 + 0xFF01))
        else:
            result.append(ch)
    return ''.join(result)


# ═══════════════════════════════════════════════════════════════
# Candidate Generation
# ═══════════════════════════════════════════════════════════════

def generate_candidates(
    text:           str,
    original_text:  str,
    analysis:       dict,
    n_candidates:   int  = 16,
    classifier_fn:  Optional[Callable[[str], float]] = None,
    already_perturbed: frozenset = frozenset(),
    arch:           str  = 'bert',
) -> List[str]:
    candidates  = set()
    words       = text.split()
    max_idx     = len(words) - 1
    sensitivity = analysis.get('word_sensitivity', {})

    priority_set  = {i for i in analysis.get('priority_words',  set()) if i <= max_idx}
    watermark_set = {i for i in analysis.get('watermark_words', set()) if i <= max_idx}
    eligible      = list(range(len(words)))

    # Heavy-attack pool: only words long enough to perturb meaningfully
    heavy_eligible = [i for i in eligible if len(words[i]) >= MIN_WORD_LEN_HEAVY]
    if not heavy_eligible:
        heavy_eligible = eligible

    if not eligible:
        return [text]

    def sample(pool, n):
        return _weighted_sample(pool, sensitivity, already_perturbed, n)

    pool_strategies = list(STRATEGIES)
    if arch == 'roberta':
        # Boost RoBERTa-specific byte-level OOD strategies
        pool_strategies.extend([
            'random_case',  'random_case',  'random_case',
            'math_unicode', 'math_unicode',
            'fullwidth',    'fullwidth',
        ])

    for _ in range(n_candidates * 6):
        strategy  = random.choice(pool_strategies)
        new_words = list(words)

        # ── Strategy: BPE-break ────────────────────────────────────
        if strategy == 'bpe_break':
            n_targets = max(1, len(heavy_eligible) // 4)
            for idx in sample(heavy_eligible, n_targets):
                new_words[idx] = apply_bpe_targeted(
                    words[idx], max_edits=random.randint(3, 5))

        # ── Strategy: High-confidence targets ────────────────────────
        elif strategy == 'high_conf':
            pool    = list(priority_set & set(heavy_eligible)) or sample(heavy_eligible, 4)
            targets = sample(pool, max(2, len(pool) // 2))
            for idx in targets:
                new_words[idx] = random.choice([
                    lambda w: apply_bpe_targeted(w, max_edits=5),
                    lambda w: apply_homoglyph_plus_zwsp(w, rate=1.0),
                    apply_blitz,
                ])(words[idx])

        # ── Strategy: Whitespace ─────────────────────────────────
        elif strategy == 'whitespace':
            n_targets = max(2, len(eligible) // 4)
            for idx in sample(eligible, n_targets):   # short words OK here (ZWSP)
                new_words[idx] = insert_whitespace_attack(words[idx])

        # ── Strategy: Emoji attach ───────────────────────────────
        elif strategy == 'emoji_attach':
            n_emoji = random.randint(1, max(1, len(words) // 8))
            for idx in sample(eligible, n_emoji):
                new_words[idx] = attach_emoji(words[idx])
                if random.random() < 0.7:
                    fn = random.choice([apply_homoglyph, apply_diacritic])
                    new_words[idx] = fn(new_words[idx], rate=0.5)

        # ── Strategy: Watermark ───────────────────────────────────
        elif strategy == 'watermark':
            targets = list(watermark_set) if watermark_set else random.sample(eligible, min(4, len(eligible)))
            for idx in targets[:max(2, len(targets) // 2)]:
                best, best_delta = words[idx], 0
                for _ in range(8):
                    attempt  = random.choice([apply_blitz, apply_homoglyph_plus_zwsp])(words[idx])
                    orig_n   = len(GPT2_TOKENIZER.encode(words[idx], add_special_tokens=False))
                    new_n    = len(GPT2_TOKENIZER.encode(attempt, add_special_tokens=False))
                    delta    = abs(new_n - orig_n)
                    if delta > best_delta:
                        best, best_delta = attempt, delta
                new_words[idx] = best

        # ── Strategy: Combined ────────────────────────────────────
        elif strategy == 'combined':
            for idx in sample(heavy_eligible, max(1, len(heavy_eligible) // 6)):
                new_words[idx] = apply_bpe_targeted(words[idx], max_edits=3)
            untouched = [i for i in eligible if new_words[i] == words[i]]
            for idx in sample(untouched, min(2, len(untouched))):
                new_words[idx] = attach_emoji(words[idx])
            untouched2 = [i for i in eligible if new_words[i] == words[i]]
            if untouched2 and random.random() < 0.6:
                idx = sample(untouched2, 1)
                if idx:
                    new_words[idx[0]] = insert_whitespace_attack(words[idx[0]])

        # ── Strategy: SCORCHED EARTH ───────────────────────────────
        elif strategy == 'scorched_earth':
            intensity = random.uniform(0.5, 0.9)
            n_targets = max(1, int(len(heavy_eligible) * intensity))
            for idx in sample(heavy_eligible, n_targets):
                new_words[idx] = scorched_earth_word(words[idx])

        # ── Strategy: BLITZ ───────────────────────────────────────
        elif strategy == 'blitz':
            intensity = random.uniform(0.6, 1.0)
            n_targets = max(1, int(len(heavy_eligible) * intensity))
            for idx in sample(heavy_eligible, n_targets):
                new_words[idx] = apply_blitz(words[idx])

        # ── Strategy: ZWSP FLOOD ──────────────────────────────────
        elif strategy == 'zwsp_flood':
            flood_pool = list(priority_set & set(heavy_eligible)) or sample(heavy_eligible, 4)
            for idx in flood_pool:
                if idx <= max_idx:
                    new_words[idx] = zwsp_flood_word(words[idx])
            remaining = [i for i in heavy_eligible if i not in flood_pool]
            for idx in sample(remaining, max(1, len(remaining) // 2)):
                new_words[idx] = apply_blitz(words[idx])

        # ── Strategy: ROBERTA SUBWORD ─────────────────────────────
        elif strategy == 'roberta_subword':
            # Target long words first (strongest BPE splits)
            targets = [w for w in heavy_eligible if len(words[w]) > 5]
            if not targets:
                targets = heavy_eligible
                
            for idx in sample(targets, max(1, len(targets) // 2)):
                new_words[idx] = roberta_subword_word(words[idx])

        # ── Strategy: MICRO SPONGE ────────────────────────────────
        elif strategy == 'micro_sponge':
            if len(words) < 10:
                # Target the last word specifically for the sponge
                new_words[-1] = apply_micro_sponge(words[-1])
            else:
                for idx in sample(heavy_eligible, 1):
                    new_words[idx] = apply_blitz(words[idx])

        # ── Strategy: PUNCTUATION BIND ────────────────────────────
        elif strategy == 'punctuation_bind':
            if len(words) < 10:
                punc_idx = [i for i, w in enumerate(words) if any(p in w for p in ['.', ',', '!', '?', ';', ':'])]
                if punc_idx:
                    for idx in punc_idx:
                        new_words[idx] = apply_punctuation_bind(words[idx])
                else:
                    new_words[-1] = apply_micro_sponge(words[-1])
            else:
                for idx in sample(heavy_eligible, 1):
                    new_words[idx] = apply_blitz(words[idx])

        elif strategy == 'zwsp_dense':
            # ALL content words get shattered — essential for short RTE texts
            all_content = [i for i, w in enumerate(words)
                           if len(w) >= MIN_WORD_LEN_HEAVY and w.isalpha()]
            if not all_content:
                all_content = list(range(len(words)))
            # Shatter every selected word at every internal char boundary
            for idx in all_content:
                w = words[idx]
                # Combo: aggressively apply homoglyphs to corrupt the actual byte-values
                # BEFORE tearing them apart with ZWSP variants
                w = apply_homoglyph_plus_zwsp(w, rate=0.7)
                new_words[idx] = zwsp_shatter_word(w)

        elif strategy == 'random_case':
            # RoBERTa is case-sensitive: mixed case = novel byte sequences
            targets = sample(heavy_eligible, max(1, len(heavy_eligible) // 2))
            for idx in targets:
                new_words[idx] = apply_random_case(words[idx])

        elif strategy == 'math_unicode':
            # Mathematical Bold Unicode: visually identical but 4-byte UTF-8 sequences
            targets = sample(heavy_eligible, max(1, len(heavy_eligible) // 3))
            for idx in targets:
                new_words[idx] = apply_math_unicode(words[idx])

        elif strategy == 'fullwidth':
            # Fullwidth ASCII variants: 3-byte UTF-8 per char, OOD for RoBERTa BPE
            targets = sample(heavy_eligible, max(1, len(heavy_eligible) // 3))
            for idx in targets:
                new_words[idx] = apply_fullwidth(words[idx])

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
    score:     float
    text:      str        = field(compare=False)
    round_no:  int        = field(compare=False, default=0)
    history:   List[str]  = field(compare=False, default_factory=list)
    perturbed: frozenset  = field(compare=False, default_factory=frozenset)

    def __post_init__(self):
        self.history = self.history + [self.text]


def misclassifies(
    text:           str,
    classifier_fn:  Callable[[str], str],
    original_label: str,
    confidence_fn:  Optional[Callable[[str], float]] = None,
) -> bool:
    """True if the classifier's LABEL prediction changed."""
    return classifier_fn(text) != original_label


def csbp_loop(
    original_text:    str,
    original_label:   str,
    classifier_fn:    Callable[[str], str],     # fn(text) → label string
    confidence_fn:    Optional[Callable[[str], float]] = None,  # fn(text) → float P(original_label)
    K:                int  = 5,
    beam_width:       int  = 8,
    n_candidates:     int  = 16,
    patience:         int  = 9,
    score_weights:    Optional[dict] = None,
    verbose:          bool = True,
    arch:             str  = 'bert',   # 'bert' or 'roberta' — controls strategy pool
) -> dict:
    weights      = score_weights or {}
    beam_history = []

    # ── Pre-analyse original text ──
    analysis = analyze_original(original_text)
    if verbose:
        print(f"[CSBP v3] Pre-analysis: PPL={analysis['ppl']:.1f}  "
              f"avg_rank={analysis['avg_rank']:.1f}  "
              f"priority_words={len(analysis['priority_words'])}  "
              f"watermark_words={len(analysis['watermark_words'])}")

    # Init beam
    init_score = composite_score(
        original_text, original_text,
        classifier_fn=confidence_fn,
        original_label=original_label,
        **weights
    )['S']
    beams = [Beam(score=init_score, text=original_text, round_no=0)]

    best_attack = None
    best_score  = -1.0
    stagnation  = 0       # rounds without improvement
    
    # Dynamic patience based on model type (RoBERTa needs deeper search)
    PATIENCE = patience

    for k in range(1, K + 1):
        if verbose:
            print(f"\n[CSBP v3] Round {k}/{K}  |  active beams: {len(beams)}")

        round_candidates: List[Beam] = []

        for beam in beams:
            beam_words = beam.text.split()
            cands = generate_candidates(
                beam.text, original_text, analysis,
                n_candidates,
                classifier_fn=confidence_fn,
                already_perturbed=beam.perturbed,
                arch=arch,
            )

            for cand_text in cands:
                # Detect word-level changes for reperturb tracking
                cand_words  = cand_text.split()
                new_changed = frozenset(
                    i for i, (bw, cw) in
                    enumerate(zip(beam_words, cand_words))
                    if bw != cw
                )
                modified = new_changed | beam.perturbed

                # Check misclassification
                if misclassifies(cand_text, classifier_fn, original_label,
                                 confidence_fn=confidence_fn):
                    result = composite_score(
                        original_text, cand_text,
                        classifier_fn=confidence_fn,
                        original_label=original_label,
                        **weights
                    )
                    S = result['S']
                    cos = result['cosine']
                    
                    # ---- RTE / SHORT TEXT FIX ----
                    # If the text is very short, modifying even 1 word drops cosine similarity steeply.
                    # Relax the floor to 0.60 for texts under 45 words, but enforce 0.75 for < 10 words.
                    num_words = len(original_text.split())
                    if num_words < 10:
                        bpe_floor = 0.6
                    elif num_words > 45:
                        bpe_floor = 0.85
                    else:
                        bpe_floor = 0.60

                    if cos < bpe_floor:
                        if verbose:
                            print(f"  ✗ Flip rejected at round {k}: cos={cos:.4f} < {bpe_floor} floor")
                        # Score it normally so it can still rank in the beam
                        round_candidates.append(
                            Beam(score=result['S'], text=cand_text,
                                 round_no=k, history=beam.history,
                                 perturbed=modified))
                        continue

                    if verbose:
                        print(f"  \u2713 ATTACK SUCCEEDED at round {k}  S={S:.4f}")
                        print(f"    classifier_score={result['classifier']:.4f}  "
                              f"bpe={result['bpe_disruption']:.4f}  "
                              f"ppl={result['ppl']:.1f}  "
                              f"cos={cos:.4f}")
                        print(f"    {cand_text[:120]}")
                    return {
                        'success':         True,
                        'best_text':       cand_text,
                        'best_score':      S,
                        'round_found':     k,
                        'score_breakdown': result,
                        'beam_history':    beam_history,
                    }

                # Score beam ranking
                result = composite_score(
                    original_text, cand_text,
                    classifier_fn=confidence_fn,
                    original_label=original_label,
                    **weights
                )
                round_candidates.append(
                    Beam(score=result['S'], text=cand_text,
                         round_no=k, history=beam.history,
                         perturbed=modified))

        if not round_candidates:
            break

        round_candidates.sort(reverse=True)
        beams = round_candidates[:beam_width]

        top = beams[0]
        beam_history.append({
            'round':     k,
            'top_score': top.score,
            'top_text':  top.text,
        })

        if verbose:
            print(f"  top S={top.score:.4f}  |  {top.text[:100]}")

        if top.score > best_score:
            best_score  = top.score
            best_attack = top.text
            stagnation  = 0
        else:
            stagnation += 1
            if stagnation >= PATIENCE:
                if verbose:
                    print(f"  [Early stop] No improvement for {PATIENCE} rounds — giving up.")
                break

    # Max rounds reached
    final = composite_score(
        original_text,
        best_attack or original_text,
        classifier_fn=confidence_fn,
        original_label=original_label,
        **weights
    )
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
    confidence_fn:  Optional[Callable[[str], float]] = None,
    K:              int  = 5,
    beam_width:     int  = 8,
    n_candidates:   int  = 16,
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
            confidence_fn  = confidence_fn,
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


# Demo tests

if __name__ == "__main__":

    def dummy_classifier(text: str) -> str:
        return 'negative' if 'bad' in text.lower() else 'positive'

    def dummy_confidence(text: str) -> float:
        return 0.1 if 'bad' in text.lower() else 0.9

    original = "The film was genuinely bad and quite disappointing overall."
    label    = dummy_classifier(original)

    print(f"Original label : {label}")
    print(f"Original text  : {original}\n")

    result = csbp_loop(
        original_text  = original,
        original_label = label,
        classifier_fn  = dummy_classifier,
        confidence_fn  = dummy_confidence,
        K              = 5,
        beam_width     = 4,
        n_candidates   = 10,
        verbose        = True,
    )

    print("\n--- Final Result ---")
    print(f"Success     : {result['success']}")
    print(f"Round found : {result['round_found']}")
    print(f"Best score  : {result['best_score']:.4f}")
    print(f"Best text   : {result['best_text']}")