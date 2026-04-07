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
    """Inserts Zero-Width Spaces (U+200B) between virtually every character to incinerate BPE tokenization."""
    # Break into words, and for 50% of words, insert ZWSP between every char
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < 0.6: # 60% of words get internal ZWSPs
            new_words.append("\u200b".join(list(w)))
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
    # POS tagging for synonym swaps
    tagged_tokens = pos_tag(tokens)
    
    allowed_emojis = FORMAL_EMOTES if register == 'formal' else INFORMAL_EMOTES
    
    for _ in range(n):
        new_tokens = []
        for word, tag in tagged_tokens:
            prob = random.random()
            
            if prob < 0.20: # 20% chance to swap synonym
                new_tokens.append(apply_synonym_swap(word, tag))
            elif prob < 0.35: # 15% chance for homoglyph/diacritic
                mode = random.choice(['h', 'd'])
                new_tokens.append(apply_homoglyph(word, rate=0.6) if mode=='h' else apply_diacritic(word, rate=0.6))
            elif prob < 0.45: # 10% chance for case scramble
                new_tokens.append(apply_case_scramble(word))
            else:
                new_tokens.append(word)
        
        # Emoji and Invisible insertions
        if random.random() < 0.5:
            idx = random.randint(0, len(new_tokens))
            new_tokens.insert(idx, random.choice(allowed_emojis))
            
        cand_text = ' '.join(new_tokens)
        
        # Aggressive ZWSP
        if random.random() < 0.8: # 80% chance for ZWSP incineration
            cand_text = apply_invisible_perturbation(cand_text)
            
        candidates.add(cand_text)
    return list(candidates)

# ---------------------------
# Main Humanizer Function
# ---------------------------
def humanize(text: str, iterations: int = 5, beam_width: int = 5) -> str:
    """
    Takes raw AI text and returns an 'alternated' (humanized) version
    optimized by the composite score S.
    """
    register = get_register(text)
    print(f"Applying AGGRESSIVE {register.upper()} humanization...")
    
    current_text = text
    
    for i in range(iterations):
        cands = generate_candidates(current_text, register, n=15)
        scored_cands = []
        for c in cands:
            res = composite_score(text, c)
            # Penalize ZWSP-heavy texts slightly in S but reward them indirectly via evasion
            # S is purely for similarity/fluency here, doesn't account for detector
            scored_cands.append((res['S'], c))
        
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        top_score, top_text = scored_cands[0]
        
        print(f"  Iteration {i+1}: Best S = {top_score:.4f}")
        current_text = top_text 
            
    return current_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Humanize AI text.")
    parser.add_argument("text", type=str, help="The AI text to humanize.")
    args = parser.parse_args()
    
    output = humanize(args.text)
    print("\n--- Humanized Output ---")
    print(output)
