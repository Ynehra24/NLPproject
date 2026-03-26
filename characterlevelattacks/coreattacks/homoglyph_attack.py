import re
import random
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

random.seed(42)

PERTURB_RATE = 0.3
OUTPUT_DIR   = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/attacked_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = set(stopwords.words('english'))

HOMOGLYPH_MAP = {
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'x': 'х', 'y': 'у', 'i': 'і',
    'A': 'А', 'B': 'В', 'E': 'Е', 'H': 'Н', 'K': 'К',
    'M': 'М', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т', 'X': 'Х',
}

DIACRITIC_MAP = {
    'a': ['à','á','â','ã','ä'], 'e': ['è','é','ê','ë'],
    'i': ['ì','í','î','ï'],     'o': ['ò','ó','ô','õ','ö'],
    'u': ['ù','ú','û','ü'],     'n': ['ñ'], 'c': ['ç'],
    'A': ['À','Á','Â','Ã','Ä'], 'E': ['È','É','Ê','Ë'],
    'I': ['Ì','Í','Î','Ï'],     'O': ['Ò','Ó','Ô','Õ','Ö'],
    'U': ['Ù','Ú','Û','Ü'],     'N': ['Ñ'], 'C': ['Ç'],
}

def is_eligible(token):
    return token.lower() not in STOPWORDS and re.search(r'[a-zA-Z]', token)

def apply_homoglyph(token, rate=PERTURB_RATE):
    chars = list(token)
    eligible = [i for i, c in enumerate(chars) if c in HOMOGLYPH_MAP]
    n_swap = max(1, int(len(eligible) * rate))
    for i in random.sample(eligible, min(n_swap, len(eligible))):
        chars[i] = HOMOGLYPH_MAP[chars[i]]
    return ''.join(chars)

def apply_diacritic(token, rate=PERTURB_RATE):
    chars = list(token)
    eligible = [i for i, c in enumerate(chars) if c in DIACRITIC_MAP]
    n_swap = max(1, int(len(eligible) * rate))
    for i in random.sample(eligible, min(n_swap, len(eligible))):
        chars[i] = random.choice(DIACRITIC_MAP[chars[i]])
    return ''.join(chars)

def attack_text(text, mode='homoglyph'):
    if not isinstance(text, str) or not text.strip():
        return text
    tokens = text.split()
    attacked = []
    for token in tokens:
        if is_eligible(token):
            if mode == 'homoglyph':
                attacked.append(apply_homoglyph(token))
            elif mode == 'diacritic':
                attacked.append(apply_diacritic(token))
            elif mode == 'mixed':
                fn = random.choice([apply_homoglyph, apply_diacritic])
                attacked.append(fn(token))
        else:
            attacked.append(token)
    return ' '.join(attacked)

def load_sst2(split='validation'):
    ds = load_dataset("glue", "sst2", split=split)
    return pd.DataFrame({'text': ds['sentence'], 'label': ds['label']})

def load_qnli(split='validation'):
    ds = load_dataset("glue", "qnli", split=split)
    return pd.DataFrame({'question': ds['question'], 'text': ds['sentence'], 'label': ds['label']})

def load_rte(split='validation'):
    ds = load_dataset("glue", "rte", split=split)
    return pd.DataFrame({'premise': ds['sentence1'], 'text': ds['sentence2'], 'label': ds['label']})

def load_agnews(split='test'):
    ds = load_dataset("ag_news", split=split)
    return pd.DataFrame({'text': ds['text'], 'label': ds['label']})

def run_attack(dataset_name, df, text_col, mode, n_samples=500):
    if n_samples:
        df = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)
    tqdm.pandas(desc=f"{dataset_name} [{mode}]")
    df[f'attacked_{text_col}'] = df[text_col].progress_apply(lambda t: attack_text(t, mode=mode))
    out_path = OUTPUT_DIR / f"{dataset_name}_{mode}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    return df

if __name__ == "__main__":
    MODES    = ['homoglyph', 'diacritic', 'mixed']
    datasets = {
        'sst2':   (load_sst2(),   'text'),
        'qnli':   (load_qnli(),   'text'),
        'rte':    (load_rte(),    'text'),
        'agnews': (load_agnews(), 'text'),
    }
    for ds_name, (df, col) in datasets.items():
        print(f"\n=== {ds_name.upper()} ===")
        for mode in MODES:
            run_attack(ds_name, df.copy(), col, mode, n_samples=500)

    # Sanity check
    tests = [
        "The movie was absolutely fantastic and gripping.",
        "Please submit your report by Friday noon.",
    ]
    print("\n--- Sanity check ---")
    for t in tests:
        print(f"original  : {t}")
        print(f"homoglyph : {attack_text(t, 'homoglyph')}")
        print(f"diacritic : {attack_text(t, 'diacritic')}")
        print(f"mixed     : {attack_text(t, 'mixed')}")
        print()
