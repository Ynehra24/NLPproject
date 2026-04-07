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

from emoji_insertion import load_or_extract_emojis

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
        # Only perturb a subset of words (30%) and only sparsely between chars (25% per char)
        if random.random() < 0.3 and len(w) > 1:
            chars = []
            for ch in w:
                chars.append(ch)
                if ch.isalnum() and random.random() < 0.25:
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
    # Basic cleaning
    t = re.sub(r"\s+", " ", text).strip()
    
    # Extract style features (simplified for the humanizer)
    def simple_style(txt):
        words = txt.split()
        n_words = max(len(words), 1)
        n_emojis = len([c for c in txt if c in emoji.EMOJI_DATA]) / n_words
        # ... (simplified version of original style extractor)
        return np.zeros(11) # Placeholder: in production, use the full extract_style_features

    emb = f_model.encode([t], convert_to_numpy=True)
    # Using a simplified 0-vector for style if full extractor isn't imported to keep file standalone
    # In a real scenario, we'd import the full extract_style_features
    combined = np.concatenate([emb, np.zeros((1, 11))], axis=1)
    return f_clf.predict(combined)[0]

def generate_candidates(text: str, register: str, n=20) -> List[str]:
    candidates = set()
    # Normalize for ZWSP before splitting
    text_clean = text.replace('\u200b', '')
    tokens = text_clean.split()
    # Only character-level perturbations allowed; preserve token boundaries and words
    allowed_emojis = FORMAL_EMOTES if register == 'formal' else INFORMAL_EMOTES

    for _ in range(n):
        new_tokens = []
        for word in tokens:
            p = random.random()
            if p < 0.35:  # 35% tokens get character-level perturbation
                mode = random.choice(['h', 'd'])
                if mode == 'h':
                    # moderate homoglyph replacement rate — preserves readability and similarity
                    new_word = apply_homoglyph(word, rate=0.45)
                else:
                    new_word = apply_diacritic(word, rate=0.45)
                new_tokens.append(new_word)
            else:
                new_tokens.append(word)

        # Emoji insertion: small chance, explicit token only
        if random.random() < 0.10:
            idx = random.randint(0, max(0, len(new_tokens)))
            new_tokens.insert(idx, random.choice(allowed_emojis))

        cand_text = ' '.join(new_tokens)
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
    Takes raw AI text and returns an 'alternated' (humanized) version
    optimized by a combined score that strongly favors similarity (S).
    """
    register = get_register(text)
    print(f"Applying AGGRESSIVE {register.upper()} humanization...")
    
    current_text = text
    
    for i in range(iterations):
        cands = generate_candidates(current_text, register, n=15)
        scored_cands = []
        for c in cands:
            res = composite_score(text, c)
            S = res.get('S', 0.0)
            R = readability_score(c)
            emoji_count = sum(1 for ch in c if ch in emoji.EMOJI_DATA)
            emoji_ratio = emoji_count / max(1, len(c.split()))
            # Heavily favor S to keep similarity high; small boost from readability; slight penalty for emojis
            combined = 0.92 * S + 0.07 * R - 0.01 * emoji_ratio
            scored_cands.append((combined, S, R, emoji_ratio, c))
        
        # Sort by combined score descending
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        top_combined, top_S, top_R, top_emoji, top_text = scored_cands[0]
        
        print(f"  Iteration {i+1}: Combined={top_combined:.4f} | S={top_S:.4f} | R={top_R:.3f} | EMO={top_emoji:.3f}")
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