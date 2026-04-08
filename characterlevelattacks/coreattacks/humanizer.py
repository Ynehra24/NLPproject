# ...existing code...
# python
import sys
import re
import json
import random
import torch
import joblib
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import emoji
import os
from typing import List, Optional, Tuple, Dict, Set

import nltk
from nltk.corpus import wordnet
from nltk import pos_tag

# Ensure local imports work
CORE_DIR = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks")
sys.path.insert(0, str(CORE_DIR.resolve()))

from composite_scorer import composite_score
from homoglyph_attack import apply_homoglyph, apply_diacritic

# dynamic emoji loader (uses emoji module and caches)
from emoji_insertion import load_or_extract_emojis, find_emojis_for_token, get_allowed_emojis

random.seed(42)
np.random.seed(42)

# ---------------------------
# Paths & Globals (lazy load)
# ---------------------------
FORMALITY_DIR = CORE_DIR / "formality_model"
EMOJI_CACHE_PATH = FORMALITY_DIR / "extracted_emojis.json"

_f_model = None
_f_clf = None
_f_scaler = None
_FORMAL_EMOTES: List[str] = []
_INFORMAL_EMOTES: List[str] = []
_EMOJI_KEYWORD_INDEX: Dict[str, Set[str]] = {}
_MODELS_LOADED = False
_SELECTED_DEVICE: Optional[str] = None

# ---------------------------
# (local demojize/index helpers removed)
# Using semantic search from emoji_insertion for token->emoji mapping
# ---------------------------

# ---------------------------
# Lazy loader
# ---------------------------
def ensure_models_loaded(device_override: Optional[str] = None):
    global _f_model, _f_clf, _f_scaler, _FORMAL_EMOTES, _INFORMAL_EMOTES, _MODELS_LOADED, _SELECTED_DEVICE
    if _MODELS_LOADED:
        return
    # prefer mps if available, else cpu; allow override
    if device_override:
        _SELECTED_DEVICE = device_override
    else:
        _SELECTED_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

    print(f"Loading models/emojis on device={_SELECTED_DEVICE} ...")
    from sentence_transformers import SentenceTransformer

    try:
        device = _SELECTED_DEVICE
        _f_model = SentenceTransformer(str(FORMALITY_DIR / "embed_model"), device=device)
        _f_clf = joblib.load(FORMALITY_DIR / "classifier.joblib")
        _f_scaler = joblib.load(FORMALITY_DIR / "scaler.joblib")
        # load cached/dynamically extracted emojis (uses emoji_insertion)
        form_emotes, inf_emotes = load_or_extract_emojis(top_n=50)
        _FORMAL_EMOTES = list(form_emotes or [])
        _INFORMAL_EMOTES = list(inf_emotes or [])
        # safe fallback if lists empty
        if not _FORMAL_EMOTES:
            _FORMAL_EMOTES = ['📘', '📖', '✒️', '📚']
        if not _INFORMAL_EMOTES:
            _INFORMAL_EMOTES = ['🙂', '👍', '😊', '🤔', '☕', '🐱']
        _MODELS_LOADED = True
        print(f"✓ Models and emojis loaded. ({len(_FORMAL_EMOTES)} formal / {len(_INFORMAL_EMOTES)} informal)")
    except Exception as e:
        print(f"Error loading assets on {_SELECTED_DEVICE}: {e}")
        # If we tried MPS, retry on CPU
        if _SELECTED_DEVICE == "mps":
            print("Falling back to cpu...")
            _SELECTED_DEVICE = "cpu"
            try:
                _f_model = SentenceTransformer(str(FORMALITY_DIR / "embed_model"), device="cpu")
                _f_clf = joblib.load(FORMALITY_DIR / "classifier.joblib")
                _f_scaler = joblib.load(FORMALITY_DIR / "scaler.joblib")
                form_emotes, inf_emotes = load_or_extract_emojis(top_n=50)
                _FORMAL_EMOTES = list(form_emotes or ['📘', '📖', '✒️', '📚'])
                _INFORMAL_EMOTES = list(inf_emotes or ['🙂', '👍', '😊', '🤔', '☕', '🐱'])
                _MODELS_LOADED = True
                print("✓ Models and emojis loaded on cpu.")
            except Exception as e2:
                print(f"Fatal load error on CPU: {e2}")
                raise
        else:
            raise

# ---------------------------
# Helpers
# ---------------------------
def get_register(text: str) -> str:
    ensure_models_loaded()
    t = re.sub(r"\s+", " ", text).strip()
    emb = _f_model.encode([t], convert_to_numpy=True)
    combined = np.concatenate([emb, np.zeros((1, 11))], axis=1)
    return _f_clf.predict(combined)[0]

def strip_emojis(text: str) -> str:
    return ''.join(ch for ch in text if ch not in emoji.EMOJI_DATA).strip()

def apply_invisible_perturbation(text: str) -> str:
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < 0.25 and len(w) > 1:
            chars = []
            for ch in w:
                chars.append(ch)
                if ch.isalnum() and random.random() < 0.20:
                    chars.append("\u200b")
            new_w = "".join(chars).rstrip("\u200b")
            new_words.append(new_w)
        else:
            new_words.append(w)
    return " ".join(new_words)

def readability_score(text: str) -> float:
    t = text.replace('\u200b', ' ')
    tokens = [tok for tok in re.findall(r"\b\w[\w']*\b", t)]
    if not tokens:
        return 0.0
    known = sum(1 for tok in tokens if wordnet.synsets(tok.lower()))
    known_frac = known / len(tokens)
    avg_len = sum(len(tok) for tok in tokens) / len(tokens)
    length_score = max(0.0, 1 - max(0.0, (avg_len - 7) / 8))
    return 0.6 * known_frac + 0.4 * length_score

# ---------------------------
# Candidate generation (dynamic emoji usage)
# ---------------------------
def generate_candidates(text: str, register: str, n=20) -> List[str]:
    ensure_models_loaded()
    candidates = set()
    text_clean = text.replace('\u200b', '')
    text_clean = strip_emojis(text_clean)
    tokens = re.findall(r"\S+", text_clean)

    # emoji pool
    default_pool = _FORMAL_EMOTES if register == 'formal' else _INFORMAL_EMOTES
    if not default_pool:
        default_pool = list(emoji.EMOJI_DATA.keys())[:40]

    # 🔧 REDUCED EMOJI SETTINGS
    num_force_emoji = max(1, int(n * 0.10))   # was 0.25 → now 10%
    base_insert_prob = 0.12                   # was 0.28 → lower

    for i in range(n):
        new_tokens = []
        emoji_added = False  # track per sentence

        for word in tokens:
            p = random.random()

            if p < 0.33:
                mode = random.choice(['h', 'd'])
                new_word = apply_homoglyph(word, rate=0.45) if mode == 'h' else apply_diacritic(word, rate=0.45)
                new_tokens.append(new_word)

            elif p < 0.50:
                new_tokens.append(word)

                # 🔧 MUCH LOWER emoji probability
                if not emoji_added:
                    matches = find_emojis_for_token(word)
                    if matches and random.random() < 0.25:  # was 0.75
                        new_tokens.append(random.choice(matches))
                        emoji_added = True
            else:
                new_tokens.append(word)

        # 🔧 FORCE AT MOST 1 emoji (not 1–3)
        if i < num_force_emoji and not emoji_added:
            if tokens:
                tok = random.choice(tokens)
                m = find_emojis_for_token(tok)
                em = random.choice(m) if m else random.choice(default_pool)
            else:
                em = random.choice(default_pool)

            idx = random.randint(0, max(0, len(new_tokens)))
            new_tokens.insert(idx, em)
            emoji_added = True

        else:
            # 🔧 rare random insertion
            if not emoji_added and random.random() < base_insert_prob:
                em = random.choice(default_pool)
                idx = random.randint(0, max(0, len(new_tokens)))
                new_tokens.insert(idx, em)
                emoji_added = True

        cand_text = ' '.join(new_tokens)

        if random.random() < 0.30:
            cand_text = apply_invisible_perturbation(cand_text)

        # 🔧 guarantee at most ONE emoji
        emoji_count = sum(1 for ch in cand_text if ch in emoji.EMOJI_DATA)
        if emoji_count == 0:
            if random.random() < 0.5:  # not always forced
                cand_text += " " + random.choice(default_pool)
        elif emoji_count > 1:
            # keep only first emoji
            seen = False
            filtered = []
            for ch in cand_text:
                if ch in emoji.EMOJI_DATA:
                    if not seen:
                        filtered.append(ch)
                        seen = True
                else:
                    filtered.append(ch)
            cand_text = "".join(filtered)

        candidates.add(cand_text)

    return list(candidates)
# ---------------------------
# Humanize (used by CLI and other callers)
# ---------------------------
def humanize(text: str, iterations: int = 5, n_candidates: int = 15, device_override: Optional[str] = None) -> str:
    ensure_models_loaded(device_override)
    register = get_register(text)
    print(f"Applying AGGRESSIVE {register.upper()} humanization (device={_SELECTED_DEVICE})...")
    current_text = text

    for i in range(iterations):
        cands = generate_candidates(current_text, register, n=n_candidates)
        scored = []
        for c in cands:
            c_clean = strip_emojis(c)
            res = composite_score(text, c_clean)
            S = res.get('S', 0.0)
            R = readability_score(c_clean)
            zwsp_ratio = c.count('\u200b') / max(1, len(c))
            emoji_count = sum(1 for ch in c if ch in emoji.EMOJI_DATA)
            emoji_ratio = emoji_count / max(1, len(c.split()))
            combined = 0.94 * S + 0.06 * R - 0.005 * zwsp_ratio
            scored.append((combined, S, R, emoji_ratio, zwsp_ratio, c))
        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        top_combined, top_S, top_R, top_emoji, top_zwsp, top_text = scored[0]
        print(f"  Iter {i+1}: Combined={top_combined:.4f} | S={top_S:.4f} | R={top_R:.3f} | EMO={top_emoji:.3f} | ZWSP={top_zwsp:.4f}")
        current_text = top_text
    return current_text

# ---------------------------
# Robust dataset/text loader
# ---------------------------
def _coerce_text_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (np.generic,)):
        try:
            return str(v.item())
        except Exception:
            return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        try:
            lst = v.tolist()
            if isinstance(lst, (list, tuple)):
                return " ".join(str(x) for x in lst if x is not None)
            return str(lst)
        except Exception:
            return str(v)
    if isinstance(v, (list, tuple)):
        try:
            return " ".join(str(x) for x in v if x is not None)
        except Exception:
            return str(v)
    try:
        if hasattr(v, "to_pydict"):
            return str(v.to_pydict())
        if hasattr(v, "to_py"):
            return str(v.to_py())
        if hasattr(v, "tolist"):
            return " ".join(str(x) for x in v.tolist())
    except Exception:
        pass
    try:
        return str(v)
    except Exception:
        return ""

def load_texts_from_file(path: Path, text_col_candidates: Optional[List[str]] = None) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()

    if suffix in {'.csv', '.tsv'}:
        delim = ',' if suffix == '.csv' else '\t'
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            rows = list(reader)
            if not rows:
                return []
            fieldnames = reader.fieldnames or []
            if text_col_candidates:
                for c in text_col_candidates:
                    if c in fieldnames:
                        return [_coerce_text_value(r.get(c, '')) for r in rows if _coerce_text_value(r.get(c, '')).strip()]
            best_col = None
            best_avg = 0.0
            for col in fieldnames:
                vals = [_coerce_text_value(r.get(col, '')) for r in rows]
                if not vals:
                    continue
                avg = sum(len(str(v)) for v in vals) / len(vals)
                if avg > best_avg:
                    best_avg = avg
                    best_col = col
            if best_col:
                return [_coerce_text_value(r.get(best_col, '')) for r in rows if _coerce_text_value(r.get(best_col, '')).strip()]
            first = fieldnames[0] if fieldnames else None
            if first:
                return [_coerce_text_value(r.get(first, '')) for r in rows if _coerce_text_value(r.get(first, '')).strip()]
            return []

    if suffix == '.parquet':
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            if df.empty:
                return []
            rows = df.to_dict(orient='records')
            fieldnames = df.columns.tolist()
            if text_col_candidates:
                for c in text_col_candidates:
                    if c in fieldnames:
                        return [_coerce_text_value(r.get(c, '')) for r in rows if _coerce_text_value(r.get(c, '')).strip()]
            best_col = None
            best_avg = 0.0
            for col in fieldnames:
                vals = [_coerce_text_value(r.get(col, '')) for r in rows]
                if not vals:
                    continue
                avg = sum(len(str(v)) for v in vals) / len(vals)
                if avg > best_avg:
                    best_avg = avg
                    best_col = col
            if best_col:
                return [_coerce_text_value(r.get(best_col, '')) for r in rows if _coerce_text_value(r.get(best_col, '')).strip()]
            first = fieldnames[0] if fieldnames else None
            if first:
                return [_coerce_text_value(r.get(first, '')) for r in rows if _coerce_text_value(r.get(first, '')).strip()]
            return []
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet {path}: {e}")

    if suffix in {'.arrow', '.feather'}:
        errors = []
        rows = []
        fieldnames = []
        try:
            from datasets import Dataset
            ds = Dataset.from_file(str(path))
            df = ds.to_pandas()
            if df.empty:
                return []
            fieldnames = df.columns.tolist()
            rows = df.to_dict(orient='records')
        except Exception as e_df:
            errors.append(f"datasets.from_file failed: {e_df}")
            try:
                import pyarrow as pa
                import pyarrow.ipc as ipc
                try:
                    tbl = ipc.open_file(str(path)).read_all()
                    df = tbl.to_pandas()
                    fieldnames = df.columns.tolist()
                    rows = df.to_dict(orient='records')
                except Exception as e_ipc:
                    errors.append(f"pyarrow.ipc.open_file failed: {e_ipc}")
                    try:
                        import pyarrow.feather as feather
                        tbl = feather.read_table(str(path))
                        df = tbl.to_pandas()
                        fieldnames = df.columns.tolist()
                        rows = df.to_dict(orient='records')
                    except Exception as e_feather:
                        errors.append(f"pyarrow.feather.read_table failed: {e_feather}")
                        try:
                            tbl = ipc.open_stream(str(path)).read_all()
                            df = tbl.to_pandas()
                            fieldnames = df.columns.tolist()
                            rows = df.to_dict(orient='records')
                        except Exception as e_stream:
                            errors.append(f"pyarrow.ipc.open_stream failed: {e_stream}")
                            try:
                                import pandas as pd
                                df = pd.read_feather(str(path))
                                fieldnames = df.columns.tolist()
                                rows = df.to_dict(orient='records')
                            except Exception as e_pd_feather:
                                errors.append(f"pandas.read_feather failed: {e_pd_feather}")
            except Exception as e_pa:
                errors.append(f"import pyarrow failed or operations failed: {e_pa}")

            if not rows:
                try:
                    from datasets import load_from_disk
                    cand = Path(path)
                    loaded = None
                    for ancestor in [cand.parent, cand.parent.parent, cand.parent.parent.parent]:
                        if ancestor is None:
                            continue
                        try:
                            loaded = load_from_disk(str(ancestor))
                            if loaded is not None:
                                break
                        except Exception as e_ld:
                            errors.append(f"load_from_disk({ancestor}) failed: {e_ld}")
                            loaded = None
                    if loaded is not None:
                        if hasattr(loaded, "to_pandas"):
                            df = loaded.to_pandas()
                        else:
                            if isinstance(loaded, dict) or hasattr(loaded, "keys"):
                                keys = list(loaded.keys()) if hasattr(loaded, "keys") else []
                                split = keys[0] if keys else None
                                if split is not None:
                                    df = loaded[split].to_pandas()
                                else:
                                    df = list(loaded.values())[0].to_pandas()
                            else:
                                raise RuntimeError("Loaded object from disk isn't Dataset/DatasetDict")
                        fieldnames = df.columns.tolist()
                        rows = df.to_dict(orient='records')
                    else:
                        errors.append("datasets.load_from_disk attempts returned None")
                except Exception as e_load_disk:
                    errors.append(f"datasets.load_from_disk attempts failed: {e_load_disk}")

        if not rows:
            raise RuntimeError(f"Failed to read arrow/feather file {path}. Tried methods:\n" + "\n".join(errors))

        if text_col_candidates:
            for c in text_col_candidates:
                if c in fieldnames:
                    return [_coerce_text_value(r.get(c, '')) for r in rows if _coerce_text_value(r.get(c, '')).strip()]
        best_col = None
        best_avg = 0.0
        for col in fieldnames:
            vals = [_coerce_text_value(r.get(col, '')) for r in rows]
            if not vals:
                continue
            avg = sum(len(str(v)) for v in vals) / len(vals)
            if avg > best_avg:
                best_avg = avg
                best_col = col
        if best_col:
            return [_coerce_text_value(r.get(best_col, '')) for r in rows if _coerce_text_value(r.get(best_col, '')).strip()]
        first = fieldnames[0] if fieldnames else None
        if first:
            return [_coerce_text_value(r.get(first, '')) for r in rows if _coerce_text_value(r.get(first, '')).strip()]
        return []

    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        return [ln.strip() for ln in fh if ln.strip()]

def sample_from_datasets(hc3_path: Optional[str], m4_path: Optional[str], total_samples: int, text_col_candidates: Optional[List[str]] = None) -> List[str]:
    all_texts = []
    if hc3_path:
        all_texts += load_texts_from_file(Path(hc3_path), text_col_candidates)
    if m4_path:
        all_texts += load_texts_from_file(Path(m4_path), text_col_candidates)
    if not all_texts:
        raise ValueError("No texts found in provided datasets.")
    total = min(total_samples, len(all_texts))
    random.shuffle(all_texts)
    return all_texts[:total]

# ---------------------------
# CLI
# ---------------------------
# ...existing code...
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Humanize AI text or dataset (character-level attacks with emojis/ZWSP).")
    parser.add_argument("text", type=str, nargs='?', default=None, help="Text to humanize, or path to file (CSV/TSV/parquet/arrow/feather/txt).")
    parser.add_argument("--hc3", type=str, default=None, help="Path to HC3 dataset (CSV/TSV/parquet/arrow/feather or txt).")
    parser.add_argument("--m4", type=str, default=None, help="Path to M4 dataset (CSV/TSV/parquet/arrow/feather or txt).")
    parser.add_argument("--sample-size", type=int, default=10, help="Total samples to draw across HC3+M4 (default: 10).")
    parser.add_argument("--text-col", type=str, nargs='*', default=None, help="Preferred text column names for CSVs (checked in order).")
    parser.add_argument("--attack-type", type=str, default="char", help="Attack type label for CSV.")
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2", help="Generator model label for CSV.")
    parser.add_argument("-o", "--output", type=str, default=str(Path.home() / "Downloads" / "teammate_pairs_template_3000.csv"), help="Output CSV path (default: ~/Downloads/teammate_pairs_template.csv).")
    parser.add_argument("--device", type=str, choices=["cpu","mps"], default="mps", help="Force device for model loading (default: mps).")
    parser.add_argument("--iterations", type=int, default=3, help="Humanizer iterations per example.")
    parser.add_argument("--cands", type=int, default=15, help="Candidates per iteration.")
    parser.add_argument("--text-file", type=str, default=None, help="Path to a plain text file to use as a single input (use '-' for stdin).")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device_flag = args.device

    examples: List[Tuple[str, str]] = []

    # 1) Prefer explicit text-file (safe for long / quoted content)
    if args.text_file:
        tf = args.text_file
        if tf == "-":
            content = sys.stdin.read()
        else:
            tfp = Path(tf)
            if not tfp.exists():
                parser.error(f"text-file not found: {tf}")
            with open(tfp, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        content = content.strip()
        if not content:
            parser.error("text-file is empty")
        # treat whole content as one example
        examples.append(("p001", content))

    # 2) Datasets (hc3/m4)
    elif args.hc3 or args.m4:
        texts = sample_from_datasets(args.hc3, args.m4, args.sample_size, args.text_col)
        for i, t in enumerate(texts, start=1):
            pid = f"p{i:03d}"
            examples.append((pid, t))

    # 3) Positional text argument or path
    elif args.text:
        input_arg = args.text
        input_path = Path(input_arg)
        if input_path.exists() and input_path.is_file():
            suffix = input_path.suffix.lower()
            dataset_suffixes = {'.csv', '.tsv', '.parquet', '.arrow', '.feather', '.txt', '.jsonl', '.ndjson'}
            if suffix in dataset_suffixes:
                try:
                    texts = load_texts_from_file(input_path, args.text_col)
                except Exception:
                    # fallback tolerant text read
                    with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
                        texts = [line.rstrip("\n") for line in fh if line.strip()]
                if not texts:
                    parser.error(f"No texts found in file: {input_path}")
                total = min(args.sample_size, len(texts)) if args.sample_size and args.sample_size > 0 else len(texts)
                random.shuffle(texts)
                texts = texts[:total]
                for i, t in enumerate(texts, start=1):
                    examples.append((f"p{i:03d}", t))
            else:
                try:
                    with open(input_path, "r", encoding="utf-8") as fh:
                        lines = [line.rstrip("\n") for line in fh if line.strip()]
                except UnicodeDecodeError:
                    with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
                        lines = [line.rstrip("\n") for line in fh if line.strip()]
                for i, ln in enumerate(lines, start=1):
                    examples.append((f"p{i:03d}", ln))
        elif "\n" in input_arg:
            # multi-line positional string provided (rare)
            lines = [ln for ln in input_arg.splitlines() if ln.strip()]
            for i, ln in enumerate(lines, start=1):
                examples.append((f"p{i:03d}", ln))
        else:
            # single short positional text
            examples.append(("p001", input_arg))
    else:
        parser.error("No input provided. supply --text-file, text, --hc3 or --m4")

    rows = []
    for pid, txt in examples:
        humanized = humanize(txt, iterations=args.iterations, n_candidates=args.cands, device_override=device_flag)
        rows.append((pid, txt, humanized, args.attack_type, args.model))
        print(f"[{pid}] Done.")

    with open(out_path, "w", encoding="utf-8", newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_id", "original_text", "humanized_text", "attack_type", "generator_model"])
        writer.writerows(rows)

    print(f"\n✓ Wrote {len(rows)} pairs to {out_path}")
# ...existing code...

if __name__ == "__main__":
    main()