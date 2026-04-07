import sys
import re
import json
import random
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
from typing import Callable, List, Optional
import emoji

# Ensure imports work if running from notebook
sys.path.insert(0, str(Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks").resolve()))
from composite_scorer import composite_score

random.seed(42)
np.random.seed(42)

# ---------------------------
# Device & Paths
# ---------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Emoji Insertion using device: {device}")

FORMAL_PATH   = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/formal_augmented.parquet").resolve()
INFORMAL_PATH = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/emojibased/twittersenti/twittersenti140.csv").resolve()

FORMALITY_DIR = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/formality_model")
OUTPUT_DIR    = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/attacked_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Dynamic Emoji Extractor (No Hardcoding)
# ---------------------------
EMOJI_CACHE_PATH = FORMALITY_DIR / "extracted_emojis.json"

def scan_file_for_emojis(filepath: Path, emoji_counts: Counter):
    """Safely scans a file for emojis without loading entire datasets into memory where possible."""
    if filepath.suffix.lower() in ['.pdf', '.png', '.jpg', '.zip', '.tar']:
        return

    print(f"  Scanning {filepath.name}...")
    try:
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
            for col in df.select_dtypes(include=['object', 'string', 'category']).columns:
                for text in df[col].dropna().astype(str):
                    emoji_counts.update([res['emoji'] for res in emoji.emoji_list(text)])
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    emoji_counts.update([res['emoji'] for res in emoji.emoji_list(line)])
    except Exception as e:
        print(f"  [!] Failed to scan {filepath.name}: {e}")

def load_or_extract_emojis(top_n: int = 15):
    """Loads emojis from cache, or scans datasets to extract them dynamically."""
    if EMOJI_CACHE_PATH.exists():
        print(f"Loading dynamic emoji sets from cache: {EMOJI_CACHE_PATH.name}")
        with open(EMOJI_CACHE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['formal'], data['informal']
            
    print("Caching top emojis dynamically from datasets (this runs once)...")
    
    # 1. Load formal emojis
    print("Scanning formal dataset...")
    formal_counts = Counter()
    scan_file_for_emojis(FORMAL_PATH, formal_counts)
    formal_emojis = [e for e, c in formal_counts.most_common(top_n)]
    
    # 2. Load informal emojis (Scan the entire emojibased folder)
    print("\nScanning all emojibased datasets...")
    informal_counts = Counter()
    emojibased_dir = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/emojibased")
    target_files = [f for f in emojibased_dir.rglob("*") if f.is_file() and not f.name.startswith('.')]
    
    for fpath in target_files:
        scan_file_for_emojis(fpath, informal_counts)
        
    informal_emojis = [e for e, c in informal_counts.most_common(top_n)]
    
    # Optional fallback if formal dataset literally has 0 emojis
    if not formal_emojis:
        formal_emojis = [" ", ".", ","] # fallback to benign chars
        
    # Save to Cache
    data = {'formal': formal_emojis, 'informal': informal_emojis}
    with open(EMOJI_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Formal Top Emojis: {formal_emojis[:5]}...")
    print(f"Informal Top Emojis: {informal_emojis[:5]}...")
    return formal_emojis, informal_emojis

# Load them into global variables for the candidate generator
FORMAL_EMOTES, INFORMAL_EMOTES = load_or_extract_emojis(top_n=15)

def get_allowed_emojis(register: str) -> List[str]:
    return FORMAL_EMOTES if register == 'formal' else INFORMAL_EMOTES

# ---------------------------
# Formality Model Re-Definitions 
# ---------------------------
STYLE_WEIGHT = 5.0

def clean_text(text):
    if not text or not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_style_features(texts):
    SLANG = {"lol","lmao","omg","wtf","tbh","ngl","idk","imo","brb","gtg","smh","fyi","asap","gonna","wanna","gotta","kinda","sorta","dunno","lemme","gimme","ya","yea","yep","nope","dude","bro","sis","lit","vibe","slay","fr","rn","irl","lowkey","highkey","bet","fam","squad","thx","ty","np","btw","imho","rofl","lmfao","af","bc"}
    CONTRACTIONS = re.compile(r"\b(i'm|i've|i'd|i'll|you're|you've|you'd|you'll|he's|she's|it's|we're|we've|we'd|we'll|they're|they've|they'd|they'll|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|that's|there's|here's|what's|who's|how's|let's)\b", re.I)
    FORMAL_VOCAB = {"hereby","pursuant","accordingly","therefore","thus","hence","kindly","sincerely","regards","respectfully","dear","attached","herewith","aforementioned","notwithstanding","henceforth","request","require","submit","review","provide","ensure","deadline","extension","meeting","schedule","confirm","inform","apologize","acknowledge","appreciate","assistance","endeavor","furthermore","moreover","however","nevertheless","consequently","professor","colleague","committee","department","institution","enclosed","regarding","subject","proposal","discuss","concern"}

    features = []
    for text in texts:
        if not text or not isinstance(text, str):
            features.append([0]*11)
            continue
        words   = text.split()
        n_words = max(len(words), 1)
        lower   = text.lower()
        
        n_emojis      = len([c for c in text if c in emoji.EMOJI_DATA]) / n_words
        n_exclaim     = text.count('!') / n_words
        n_question    = text.count('?') / n_words
        n_caps_words  = sum(1 for w in words if w.isupper() and len(w)>1)/n_words
        n_contractions= len(CONTRACTIONS.findall(text)) / n_words
        word_lengths  = [len(re.sub(r'[^a-zA-Z]', '', w)) for w in words]
        avg_word_len  = np.mean(word_lengths) if word_lengths else 0
        word_tokens   = set(re.findall(r'\b\w+\b', lower))
        n_slang       = len(word_tokens & SLANG) / n_words
        n_repeated    = len(re.findall(r'(.)\1{2,}', text)) / n_words
        n_ellipsis    = text.count('...') / n_words
        formal_score  = len(word_tokens & FORMAL_VOCAB) / n_words
        has_interject = int(bool(re.search(r'\b(lol|haha|hehe|omg|wow|yay|ugh|eww|woah|whoa|yikes|wtf)\b', lower)))
        
        features.append([n_emojis, n_exclaim, n_question, n_caps_words, n_contractions, avg_word_len, n_slang, n_repeated, n_ellipsis, formal_score, has_interject])
    return np.array(features, dtype=np.float32)

# ---------------------------
# Load Register Gate Evaluator
# ---------------------------
from sentence_transformers import SentenceTransformer
try:
    f_model  = SentenceTransformer(str(FORMALITY_DIR / "embed_model"), device=device)
    f_clf    = joblib.load(FORMALITY_DIR / "classifier.joblib")
    f_scaler = joblib.load(FORMALITY_DIR / "scaler.joblib")
    print("✓ Formality model loaded successfully for Register Gate.")
except Exception as e:
    print(f"Warning: Formality model not found completely. Ensure you ran the script to save it. Error: {e}")

def get_register(text: str) -> str:
    """Returns 'formal' or 'informal'."""
    text_c   = clean_text(text)
    emb      = f_model.encode([text_c], convert_to_numpy=True)
    style    = f_scaler.transform(extract_style_features([text_c]))
    combined = np.concatenate([emb, style * STYLE_WEIGHT], axis=1)
    return f_clf.predict(combined)[0]

# ---------------------------
# Candidate Generation (Emoji)
# ---------------------------
def generate_emoji_candidates(text: str, register: str, n_candidates: int = 10) -> List[str]:
    """Generates text variations by injecting 1-2 register-appropriate emojis."""
    candidates = set()
    allowed_emojis = get_allowed_emojis(register)
    
    # If the dataset literally had 0 emojis, fallback to just returning the text
    if not allowed_emojis: 
        return [text]
        
    tokens = text.split()
    
    # Places we can safely inject an emoji: end of sentences, or end of string
    valid_insert_idx = [i for i, t in enumerate(tokens) if any(p in t for p in ['.', '!', '?', ',', ';'])]
    if not valid_insert_idx:
        valid_insert_idx = list(range(len(tokens))) # fallback to any token
        
    for _ in range(n_candidates * 3):
        n_inserts = random.choice([1, 2])
        chosen_indices = random.sample(valid_insert_idx, min(n_inserts, len(valid_insert_idx)))
        
        new_tokens = list(tokens)
        for idx in sorted(chosen_indices, reverse=True):
            emote = random.choice(allowed_emojis)
            # Insert with a space so it attaches cleanly
            new_tokens.insert(idx + 1, emote)
            
        cand_text = ' '.join(new_tokens)
        
        # Always add a candidate that just appends to the very end
        end_cand = text + " " + random.choice(allowed_emojis)
        
        candidates.add(cand_text)
        candidates.add(end_cand)
        
        if len(candidates) >= n_candidates:
            break
            
    return list(candidates)[:n_candidates]

# ---------------------------
# CSBP Loop for Emojis
# ---------------------------
@dataclass(order=True)
class Beam:
    score    : float
    text     : str = field(compare=False)
    round_no : int = field(compare=False, default=0)
    history  : List[str] = field(compare=False, default_factory=list)

    def __post_init__(self):
        self.history = self.history + [self.text]

def misclassifies(text: str, classifier_fn: Callable[[str], str], original_label: str) -> bool:
    return classifier_fn(text) != original_label

def csbp_emoji_loop(
    original_text  : str,
    original_label : str,
    classifier_fn  : Callable[[str], str],
    K              : int = 3,  # Emojis need fewer rounds (max 3 insertions)
    beam_width     : int = 3,
    n_candidates   : int = 8,
    score_weights  : Optional[dict] = None,
    verbose        : bool = True,
) -> dict:
    
    weights = score_weights or {'w_cosine': 0.35, 'w_ppl': 0.20, 'w_levenshtein': 0.15, 'w_jaccard': 0.20, 'w_stylo': 0.10}
    beam_history = []
    
    # Register Gate
    register = get_register(original_text)
    if verbose: print(f"  [Gate] Detected Register: {register.upper()}")

    init_score = composite_score(original_text, original_text, **weights)['S']
    beams = [Beam(score=init_score, text=original_text, round_no=0)]

    best_attack, best_score, round_found = None, -1.0, None

    for k in range(1, K + 1):
        if verbose: print(f"\n[CSBP Emoji] Round {k}/{K} | active beams: {len(beams)}")
        round_candidates = []

        for beam in beams:
            candidates = generate_emoji_candidates(beam.text, register, n_candidates)

            for cand_text in candidates:
                result = composite_score(original_text, cand_text, **weights)
                S = result['S']

                if misclassifies(cand_text, classifier_fn, original_label):
                    if verbose:
                        print(f"  ✓ Attack succeeded at round {k} | S={S:.4f}")
                        print(f"    {cand_text}")
                    return {
                        'success': True, 'best_text': cand_text, 'best_score': S,
                        'round_found': k, 'score_breakdown': result, 'register_used': register
                    }

                round_candidates.append(Beam(score=S, text=cand_text, round_no=k, history=beam.history))

        if not round_candidates: break

        round_candidates.sort(reverse=True)
        beams = round_candidates[:beam_width]
        top = beams[0]
        
        if verbose: print(f"  top S={top.score:.4f} | {top.text[:80]}")
        if top.score > best_score:
            best_score = top.score
            best_attack = top.text

    return {
        'success': False, 'best_text': best_attack or original_text, 'best_score': best_score,
        'round_found': None, 'score_breakdown': composite_score(original_text, best_attack or original_text, **weights),
        'register_used': register
    }

# ---------------------------
# Sanity Check
# ---------------------------
if __name__ == "__main__":
    def dummy_classifier(text: str) -> str:
        return 'negative' if 'bad' in text.lower() else 'positive'

    orig = "The film was genuinely bad and quite disappointing overall."
    label = dummy_classifier(orig)

    print(f"Original label : {label}")
    print(f"Original text  : {orig}")

    result = csbp_emoji_loop(
        original_text=orig,
        original_label=label,
        classifier_fn=dummy_classifier,
        verbose=True
    )
