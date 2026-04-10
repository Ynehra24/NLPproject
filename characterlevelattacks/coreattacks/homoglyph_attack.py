"""
homoglyph_attack.py  —  v3  (Beat-Charmer Edition)
====================================================

Key changes from v1:
  1. Expanded HOMOGLYPH_MAP  — more substitutable chars → higher coverage per word
  2. PERTURB_RATE = 0.85     — swap almost all eligible chars (was 0.3)
  3. apply_diacritic()       — NOW stacks a ZWSP after every diacritic substitution.
                               BertTokenizer strips the accent via NFD+Mn-removal,
                               but ZWSP (U+200B, category Cf) survives and shatters
                               BPE tokenisation. Net effect: diacritics finally do
                               something instead of 0 % ASR.
  4. attack_text()           — attacks ALL words including stopwords (was skipping them).
                               Stopwords like "the / is / are" are high-frequency → high
                               classifier weight → ignoring them left the model signal intact.
  5. New mode 'aggressive'   — homoglyph + ZWSP on every eligible word at rate 1.0.
  6. New mode 'blitz'        — homoglyph + diacritic + ZWSP on every word at rate 1.0.
                               This is the mode to use for beating Charmer.
"""

import re
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# Perturbation Rate
# ═══════════════════════════════════════════════════════════════

PERTURB_RATE = 0.85   # was 0.3 — swap nearly all eligible chars per word


# ═══════════════════════════════════════════════════════════════
# Homoglyph Map  (Latin → visually identical non-Latin)
# ═══════════════════════════════════════════════════════════════
# Every substitution is a different Unicode code-point, so BERT/RoBERTa
# tokenisers see a completely different token (or byte sequence for RoBERTa's
# byte-level BPE), causing misclassification.

HOMOGLYPH_MAP = {
    # ── Lowercase Cyrillic look-alikes ──────────────────────────
    'a': 'а',   # U+0430  Cyrillic small a
    'e': 'е',   # U+0435  Cyrillic small ie
    'o': 'о',   # U+043E  Cyrillic small o
    'p': 'р',   # U+0440  Cyrillic small er
    'c': 'с',   # U+0441  Cyrillic small es
    'x': 'х',   # U+0445  Cyrillic small ha
    'y': 'у',   # U+0443  Cyrillic small u
    'i': 'і',   # U+0456  Ukrainian small i
    'j': 'ј',   # U+0458  Cyrillic small je
    's': 'ѕ',   # U+0455  Cyrillic small dze  (looks like 's')
    'd': 'ԁ',   # U+0501  Cyrillic small komi de
    'g': 'ɡ',   # U+0261  Latin IPA script small g (identical glyph)
    'n': 'ո',   # U+0578  Armenian vo (close to 'n')
    'u': 'υ',   # U+03C5  Greek small upsilon
    'v': 'ν',   # U+03BD  Greek small nu
    'w': 'ѡ',   # U+0461  Cyrillic small omega
    'k': 'κ',   # U+03BA  Greek small kappa
    'l': 'ӏ',   # U+04CF  Cyrillic small palochka (looks like 'l')
    'r': 'г',   # U+0433  Cyrillic small ghe  (looks like 'r' in some fonts)
    'h': 'һ',   # U+04BB  Cyrillic small shha
    'm': 'м',   # U+043C  Cyrillic small em
    'f': 'ƒ',   # U+0192  Latin small f with hook
    't': 'τ',   # U+03C4  Greek small tau
    'b': 'ƅ',   # U+0185  Latin small tone six (looks like 'b')
    'q': 'ԛ',   # U+051B  Cyrillic small qa
    'z': 'ʐ',   # U+0290  Latin small z retroflex hook

    # ── Uppercase Cyrillic look-alikes ──────────────────────────
    'A': 'А',   # U+0410
    'B': 'В',   # U+0412
    'C': 'С',   # U+0421
    'D': 'Ꭰ',   # U+13A0  Cherokee A  (looks like 'D')
    'E': 'Е',   # U+0415
    'F': 'Ƒ',   # U+0191
    'G': 'Ԍ',   # U+050C  Cyrillic capital komi de
    'H': 'Н',   # U+041D
    'I': 'І',   # U+0406  Ukrainian capital i
    'J': 'Ј',   # U+0408  Cyrillic capital je
    'K': 'К',   # U+041A
    'L': 'Ⅼ',   # U+216C  Roman numeral 50 (identical to capital L)
    'M': 'М',   # U+041C
    'N': 'Ν',   # U+039D  Greek capital nu
    'O': 'О',   # U+041E
    'P': 'Р',   # U+0420
    'Q': 'Ԛ',   # U+051A  Cyrillic capital qa
    'R': 'Ʀ',   # U+01A6  Latin letter yr (looks like R)
    'S': 'Ѕ',   # U+0405  Cyrillic capital dze
    'T': 'Т',   # U+0422
    'U': 'Ʋ',   # U+01B2  Latin capital V with hook (close to U)
    'V': 'Ѵ',   # U+0474  Cyrillic izhitsa
    'W': 'Ԝ',   # U+051C  Cyrillic capital we
    'X': 'Х',   # U+0425
    'Y': 'Υ',   # U+03A5  Greek capital upsilon
    'Z': 'Ζ',   # U+0396  Greek capital zeta
}


# ═══════════════════════════════════════════════════════════════
# Diacritic Map  (Latin → diacritic variant)
# ═══════════════════════════════════════════════════════════════
# BertTokenizer (uncased) strips diacritics via NFD + Mn-removal.
# To KEEP the attack signal, apply_diacritic() now inserts a ZWSP
# immediately after the substituted char.  The diacritic is
# stripped BUT the ZWSP (category Cf, not Mn) survives, shattering
# BPE tokenisation.  Human readers never see ZWSP.

DIACRITIC_MAP = {
    'a': ['à', 'á', 'â', 'ã', 'ä', 'å'],
    'e': ['è', 'é', 'ê', 'ë'],
    'i': ['ì', 'í', 'î', 'ï'],
    'o': ['ò', 'ó', 'ô', 'õ', 'ö', 'ø'],
    'u': ['ù', 'ú', 'û', 'ü'],
    'n': ['ñ'],
    'c': ['ç'],
    'y': ['ý', 'ÿ'],
    's': ['š', 'ś'],
    'z': ['ž', 'ź', 'ż'],
    'r': ['ŕ', 'ř'],
    'l': ['ĺ', 'ļ', 'ľ'],
    'd': ['ď', 'đ'],
    't': ['ţ', 'ť'],
    'g': ['ĝ', 'ğ', 'ġ'],
    'h': ['ĥ', 'ħ'],
    'k': ['ķ'],
    'j': ['ĵ'],
    # Uppercase
    'A': ['À', 'Á', 'Â', 'Ã', 'Ä', 'Å'],
    'E': ['È', 'É', 'Ê', 'Ë'],
    'I': ['Ì', 'Í', 'Î', 'Ï'],
    'O': ['Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ø'],
    'U': ['Ù', 'Ú', 'Û', 'Ü'],
    'N': ['Ñ'],
    'C': ['Ç'],
    'Y': ['Ý'],
    'S': ['Š'],
    'Z': ['Ž'],
}

# ═══════════════════════════════════════════════════════════════
# Invisible BPE-breakers
# ═══════════════════════════════════════════════════════════════

ZWSP       = '\u200B'   # zero-width space         — primary BPE breaker
ZWNJ       = '\u200C'   # zero-width non-joiner
ZWJ        = '\u200D'   # zero-width joiner
SOFT_HYPHEN = '\u00AD'  # soft hyphen
WORD_JOINER = '\u2060'  # word joiner

INVISIBLE_CHARS = [ZWSP, ZWNJ, ZWJ, SOFT_HYPHEN, WORD_JOINER]


# ═══════════════════════════════════════════════════════════════
# Eligibility  (used by attack_text — now attacks ALL words)
# ═══════════════════════════════════════════════════════════════

def is_eligible(token: str) -> bool:
    """Any token containing at least one ASCII letter is eligible.
    Stopwords ARE attacked now — they carry strong classifier signal."""
    return bool(re.search(r'[a-zA-Z]', token))


# ═══════════════════════════════════════════════════════════════
# Core Perturbation Functions
# ═══════════════════════════════════════════════════════════════

def apply_homoglyph(token: str, rate: float = PERTURB_RATE) -> str:
    """Replace eligible chars with Cyrillic/Greek look-alikes."""
    chars = list(token)
    eligible = [i for i, c in enumerate(chars) if c in HOMOGLYPH_MAP]
    if not eligible:
        return token
    n_swap = max(1, int(len(eligible) * rate))
    for i in random.sample(eligible, min(n_swap, len(eligible))):
        chars[i] = HOMOGLYPH_MAP[chars[i]]
    return ''.join(chars)


def apply_diacritic(token: str, rate: float = PERTURB_RATE) -> str:
    """
    Replace eligible chars with diacritic variants AND insert ZWSP after each.

    Why ZWSP: BertTokenizer (uncased) normalises via NFD then strips combining
    marks (Unicode category Mn), reverting é → e.  But ZWSP is category Cf
    (format char), so it SURVIVES.  It shatters GPT-2/RoBERTa BPE because
    the pre-tokeniser regex `(?i:'s|'t|...)|\p{L}+|\p{N}+|...` only matches
    consecutive letter runs — a ZWSP inside a word splits it into two separate
    pre-token chunks, each tokenised independently, dramatically inflating token
    count and derailing perplexity / token-rank statistics.

    Human readers: ZWSP is invisible and zero-width.  Copy-paste preserves it,
    but screen display is identical to the original.
    """
    chars = list(token)
    eligible = [i for i, c in enumerate(chars) if c in DIACRITIC_MAP]
    if not eligible:
        # Fallback: insert ZWSP at every 2nd character position
        result = []
        for idx, ch in enumerate(chars):
            result.append(ch)
            if idx % 2 == 1 and idx < len(chars) - 1:
                result.append(ZWSP)
        return ''.join(result)

    n_swap = max(1, int(len(eligible) * rate))
    # Process in reverse so insertion offsets don't drift
    chosen = sorted(
        random.sample(eligible, min(n_swap, len(eligible))),
        reverse=True
    )
    for i in chosen:
        chars[i] = random.choice(DIACRITIC_MAP[chars[i]])
        chars.insert(i + 1, ZWSP)   # ← THE FIX: ZWSP survives tokenizer
    return ''.join(chars)


def apply_zwsp_only(token: str, rate: float = PERTURB_RATE) -> str:
    """Insert ZWSP at interior character positions — pure BPE shatter."""
    chars = list(token)
    if len(chars) <= 2:
        return ZWSP.join(chars)
    positions = list(range(1, len(chars)))
    n_insert = max(1, int(len(positions) * rate))
    chosen = sorted(random.sample(positions, min(n_insert, len(positions))),
                    reverse=True)
    for pos in chosen:
        chars.insert(pos, random.choice(INVISIBLE_CHARS))
    return ''.join(chars)


def apply_homoglyph_plus_zwsp(token: str, rate: float = PERTURB_RATE) -> str:
    """Homoglyph substitution followed by ZWSP insertion — maximum disruption."""
    step1 = apply_homoglyph(token, rate=rate)
    step2 = apply_zwsp_only(step1, rate=rate * 0.5)
    return step2


def apply_blitz(token: str) -> str:
    """
    BLITZ mode: apply ALL three perturbation layers at rate=1.0.
      1. Homoglyph substitution on every eligible char
      2. Diacritic + ZWSP on every remaining eligible char
      3. Additional ZWSP insertion at every 2nd position
    This is the nuclear option — mirrors 'scorched_earth' at word level.
    """
    # Layer 1: homoglyph at full rate
    result = apply_homoglyph(token, rate=1.0)
    # Layer 2: diacritic+ZWSP on positions that weren't homoglyph-substituted
    #           (apply to original chars that remain as ASCII letters)
    chars = list(result)
    ascii_positions = [i for i, c in enumerate(chars) if c in DIACRITIC_MAP]
    for i in sorted(ascii_positions, reverse=True):
        chars[i] = random.choice(DIACRITIC_MAP[chars[i]])
        chars.insert(i + 1, ZWSP)
    # Layer 3: insert ZWSP at every 3rd remaining position
    result2 = ''.join(chars)
    final = []
    for idx, ch in enumerate(result2):
        final.append(ch)
        if idx % 3 == 2 and idx < len(result2) - 1 and ch not in INVISIBLE_CHARS:
            final.append(random.choice(INVISIBLE_CHARS))
    return ''.join(final)


# ═══════════════════════════════════════════════════════════════
# Text-Level Attack  (used by evaluation pipeline)
# ═══════════════════════════════════════════════════════════════

def attack_text(text: str, mode: str = 'homoglyph') -> str:
    """
    Apply character-level attack to every eligible token in text.

    Modes:
      homoglyph  — Cyrillic/Greek substitution at rate 0.85
      diacritic  — diacritic + ZWSP at rate 0.85  (fixed from v1)
      mixed      — randomly chooses homoglyph or diacritic per word
      aggressive — homoglyph + ZWSP on every word at rate 1.0
      blitz      — all three layers on every word at rate 1.0  (strongest)

    CHANGE FROM v1: stopwords are NO LONGER SKIPPED.
    Skipping stopwords left the high-frequency classifier signal intact.
    All tokens with at least one ASCII letter are now attacked.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    tokens = text.split()
    attacked = []

    for token in tokens:
        if not is_eligible(token):
            attacked.append(token)
            continue

        if mode == 'homoglyph':
            attacked.append(apply_homoglyph(token, rate=PERTURB_RATE))

        elif mode == 'diacritic':
            attacked.append(apply_diacritic(token, rate=PERTURB_RATE))

        elif mode == 'mixed':
            fn = random.choice([apply_homoglyph, apply_diacritic])
            attacked.append(fn(token, rate=PERTURB_RATE))

        elif mode == 'aggressive':
            attacked.append(apply_homoglyph_plus_zwsp(token, rate=1.0))

        elif mode == 'blitz':
            attacked.append(apply_blitz(token))

        else:
            attacked.append(apply_homoglyph(token, rate=PERTURB_RATE))

    return ' '.join(attacked)


# ═══════════════════════════════════════════════════════════════
# Demo / Sanity Check
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        "The movie was absolutely fantastic and gripping.",
        "Please submit your report by Friday noon.",
        "The stock market experienced significant volatility today.",
    ]
    modes = ['homoglyph', 'diacritic', 'mixed', 'aggressive', 'blitz']

    for t in tests:
        print(f"\noriginal   : {t}")
        for m in modes:
            print(f"{m:<12}: {attack_text(t, m)}")