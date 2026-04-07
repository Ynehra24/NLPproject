import sys
import re
import json
import random
import torch
import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import emoji

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from typing import List, Callable, Optional

# Ensure local imports work
CORE_DIR = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks")
sys.path.insert(0, str(CORE_DIR.resolve()))

from composite_scorer import composite_score
from homoglyph_attack import apply_homoglyph, apply_diacritic

random.seed(42)
np.random.seed(42)

# ---------------------------
# Device & Paths
# ---------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

FORMALITY_DIR = CORE_DIR / "formality_model"
EMOJI_CACHE_PATH = FORMALITY_DIR / "extracted_emojis.json"

# ---------------------------
# Load Models & Assets
# ---------------------------
from sentence_transformers import SentenceTransformer

from emoji_insertion import load_or_extract_emojis, get_register as _emoji_get_register, extract_style_features

print("Loading models for Humanizer...")
try:
    # Formality Gate
    f_model  = SentenceTransformer(str(FORMALITY_DIR / "embed_model"), device=device)
    f_clf    = joblib.load(FORMALITY_DIR / "classifier.joblib")
    f_scaler = joblib.load(FORMALITY_DIR / "scaler.joblib")
    
    # Dynamic Emoji Sets (Scans emojibased folder if cache missing)
    FORM_EMOTES, INF_EMOTES = load_or_extract_emojis(top_n=15)
    FORMAL_EMOTES, INFORMAL_EMOTES = FORM_EMOTES, INF_EMOTES
        
    print(f"✓ All assets loaded. (Extracted {len(FORMAL_EMOTES)} formal / {len(INFORMAL_EMOTES)} informal emojis)")
except Exception as e:
    print(f"Error loading humanizer assets: {e}")
    sys.exit(1)

# ---------------------------
# Internal Helpers
# ---------------------------
def get_synonyms(word: str, pos: str) -> List[str]:
    syns = set()
    for synset in wordnet.synsets(word):
        if synset.pos() == pos:
            for lemma in synset.lemmas():
                if lemma.name().lower() != word.lower():
                    syns.add(lemma.name().replace('_', ' '))
    return list(syns)

def apply_synonym_swap(token: str, tag: str) -> str:
    # Map NLTK tags to WordNet tags
    tag_map = {'NN': 'n', 'VB': 'v', 'JJ': 'a', 'RB': 'r'}
    wn_tag = tag_map.get(tag[:2], None)
    if not wn_tag: return token
    
    syns = get_synonyms(token, wn_tag)
    return random.choice(syns) if syns else token

def apply_invisible_perturbation(text: str) -> str:
    """Insert sparse Zero-Width Spaces (U+200B) to minimally disrupt tokenization while keeping readability."""
    words = text.split()
    new_words = []
    for w in words:
        # Only perturb a subset of words (20-30%) and only sparsely between chars (~20% per char)
        if random.random() < 0.25 and len(w) > 1:
            chars = []
            for ch in w:
                chars.append(ch)
                if ch.isalnum() and random.random() < 0.20:
                    chars.append("\u200b")
            # Avoid trailing ZWSP
            new_w = "".join(chars).rstrip("\u200b")
            new_words.append(new_w)
        else:
            new_words.append(w)
    return " ".join(new_words)

def apply_case_scramble(token: str) -> str:
    """Randomly swaps character case to disrupt statistical patterns."""
    if len(token) < 2: return token
    chars = list(token)
    idx = random.randint(0, len(chars)-1)
    if chars[idx].isalpha():
        chars[idx] = chars[idx].swapcase()
    return "".join(chars)

def get_register(text: str) -> str:
    """Auto-detect formal/informal register using the trained formality model with full style features."""
    return _emoji_get_register(text)

def strip_emojis(text: str) -> str:
    """Remove all emoji characters from text for clean scoring."""
    return ''.join(ch for ch in text if ch not in emoji.EMOJI_DATA).strip()

def generate_candidates(text: str, register: str, n=20) -> List[str]:
    """
    Generate character-level perturbations with scattered emoji insertion:
     - homoglyphs or diacritics applied to existing tokens.
     - 2-4 register-appropriate emojis inserted at random positions throughout.
     - optional sparse ZWSP insertion applied to final candidate.
    """
    candidates = set()
    text_clean = text.replace('\u200b', '')
    # Strip any existing emojis from the working text to avoid emoji accumulation
    text_clean = strip_emojis(text_clean)
    tokens = text_clean.split()
    allowed_emojis = FORMAL_EMOTES if register == 'formal' else INFORMAL_EMOTES

    for _ in range(n):
        new_tokens = []
        for word in tokens:
            p = random.random()
            if p < 0.35:  # 35% tokens get character-level perturbation
                mode = random.choice(['h', 'd'])
                if mode == 'h':
                    new_word = apply_homoglyph(word, rate=0.45)
                else:
                    new_word = apply_diacritic(word, rate=0.45)
                new_tokens.append(new_word)
            else:
                new_tokens.append(word)

        # Scatter 2-4 emojis at random positions throughout the text
        n_emojis = random.randint(2, min(4, max(2, len(new_tokens) // 10)))
        # Pick random insertion points (spread across text), insert right-to-left to keep indices stable
        if len(new_tokens) > 1:
            possible_positions = list(range(1, len(new_tokens)))  # avoid position 0
            insert_positions = sorted(random.sample(possible_positions, min(n_emojis, len(possible_positions))), reverse=True)
            for idx in insert_positions:
                new_tokens.insert(idx, random.choice(allowed_emojis))

        cand_text = ' '.join(new_tokens)

        # Sparse invisible-character perturbation with controlled probability
        if random.random() < 0.30:
            cand_text = apply_invisible_perturbation(cand_text)

        candidates.add(cand_text)
    return list(candidates)

# ---------------------------
# Readability & Main Humanizer
# ---------------------------
def readability_score(text: str) -> float:
    """Lightweight readability proxy in [0,1]: fraction of known words (wordnet) and preferred avg token length."""
    t = text.replace('\u200b', ' ')
    tokens = [tok for tok in re.findall(r"\b\w[\w']*\b", t)]
    if not tokens:
        return 0.0
    known = 0
    for tok in tokens:
        if wordnet.synsets(tok.lower()):
            known += 1
    known_frac = known / len(tokens)
    avg_len = sum(len(tok) for tok in tokens) / len(tokens)
    # Prefer avg token length around 4-7; penalize overly long tokens
    length_score = max(0.0, 1 - max(0.0, (avg_len - 7) / 8))
    return 0.6 * known_frac + 0.4 * length_score

def humanize(text: str, iterations: int = 5, beam_width: int = 5) -> str:
    """
    Takes raw AI text and returns a humanized version.
    Candidate selection heavily favors composite_score 'S' (similarity) with a small readability boost.
    """
    register = get_register(text)
    print(f"Applying AGGRESSIVE {register.upper()} humanization...")
    
    current_text = text
    
    for i in range(iterations):
        cands = generate_candidates(current_text, register, n=15)
        scored_cands = []
        for c in cands:
            # Score the candidate WITHOUT emojis so emoji tokens don't tank similarity metrics
            c_clean = strip_emojis(c)
            res = composite_score(text, c_clean)
            S = res.get('S', 0.0)
            R = readability_score(c_clean)
            zwsp_count = c.count('\u200b')
            zwsp_ratio = zwsp_count / max(1, len(c))
            emoji_count = sum(1 for ch in c if ch in emoji.EMOJI_DATA)
            emoji_ratio = emoji_count / max(1, len(c.split()))
            # Score purely on perturbation quality; emojis are always added and don't affect scoring
            combined = 0.94 * S + 0.06 * R - 0.005 * zwsp_ratio
            scored_cands.append((combined, S, R, emoji_ratio, zwsp_ratio, c))
        
        # Sort by combined score descending
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        best = scored_cands[0]
        top_combined, top_S, top_R, top_emoji, top_zwsp, top_text = best
        
        print(f"  Iteration {i+1}: Combined={top_combined:.4f} | S={top_S:.4f} | R={top_R:.3f} | EMO={top_emoji:.3f} | ZWSP={top_zwsp:.4f}")
        current_text = top_text 
            
    return current_text

if __name__ == "__main__":
    import argparse
    import csv
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Humanize AI text.")
    parser.add_argument("text", type=str, help="The AI text to humanize. Can be a path to a file with multiple lines or a newline-separated string.")
    args = parser.parse_args()

    input_arg = args.text

    # Determine inputs: file path, newline-separated string, or single string
    if os.path.exists(input_arg) and os.path.isfile(input_arg):
        with open(input_arg, "r", encoding="utf-8") as fh:
            lines = [line.rstrip("\n") for line in fh if line.strip()]
    elif "\n" in input_arg:
        lines = [ln for ln in input_arg.splitlines() if ln.strip()]
    else:
        lines = [input_arg]

    if len(lines) > 1:
        rows = []
        for ln in lines:
            humanized = humanize(ln)
            rows.append((ln, humanized))

        downloads_dir = Path.home() / "Downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        out_path = downloads_dir / "humanized_texts.csv"

        with open(out_path, "w", encoding="utf-8", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["original", "humanized"])
            writer.writerows(rows)

        print(f"\nWrote {len(rows)} rows to {out_path}")
    else:
        output = humanize(lines[0])
        print("\n--- Humanized Output ---")
        print(output)