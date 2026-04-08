# python
import json
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import emoji
import numpy as np

# lazy imports (heavy)
_EMBED_MODEL = None
_EMOJI_LIST: List[str] = []
_EMOJI_NAMES: List[str] = []
_EMOJI_EMBS: Optional[np.ndarray] = None
_EMOJI_INDEX_BUILT = False

FORMALITY_DIR = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/formality_model")
EMOJI_CACHE_PATH = FORMALITY_DIR / "extracted_emojis.json"

def _demojize_name(e: str) -> str:
    try:
        nm = emoji.demojize(e)
        nm = nm.strip(':').replace('_', ' ').lower()
        # remove skin tone modifiers or variants
        nm = re.sub(r'\s+tone\b', '', nm)
        return nm
    except Exception:
        return ""

def _uniq(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _lazy_load_embed_model(model_name: str = "all-mpnet-base-v2"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            device = "mps" if hasattr(__import__("torch").backends, "mps") and __import__("torch").backends.mps.is_available() else "cpu"
            _EMBED_MODEL = SentenceTransformer(model_name, device=device)
        except Exception as e:
            # fallback: raise to caller
            raise RuntimeError(f"Failed to load SentenceTransformer ({model_name}): {e}")
    return _EMBED_MODEL

def load_or_extract_emojis(top_n: int = 50) -> Tuple[List[str], List[str]]:
    """
    Extract heuristic formal/informal emoji lists and cache them.
    Also fills _EMOJI_LIST and _EMOJI_NAMES for runtime index building.
    """
    global _EMOJI_LIST, _EMOJI_NAMES
    # try cache
    if EMOJI_CACHE_PATH.exists():
        try:
            with open(EMOJI_CACHE_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                formal = data.get("formal", [])
                informal = data.get("informal", [])
                _EMOJI_LIST = list(dict.fromkeys(formal + informal))
                _EMOJI_NAMES = [_demojize_name(e) for e in _EMOJI_LIST]
                return formal[:top_n], informal[:top_n]
        except Exception:
            pass

    # heuristic keyword buckets
    formal_keywords = {
        "book","scroll","pen","pencil","memo","page","notebook","briefcase","file",
        "document","bookmark","calendar","date","envelope","email","letter","certificate",
        "clipboard","ledger","balance","scale","file folder","office","fountain pen","paper"
    }
    informal_keywords = {
        "face","smile","laugh","grin","cry","tear","angry","rage","heart","thumb","clap",
        "fire","star","coffee","cat","dog","party","sunglasses","wink","eyes","kiss","wave",
        "ok hand","pray","muscle","rocket","pizza","cake","beer","yum","cool","tongue","zany"
    }

    formal = []
    informal = []
    other = []

    for em in emoji.EMOJI_DATA.keys():
        name = _demojize_name(em)
        if not name:
            other.append((em, name))
            continue
        matched_formal = any(k in name for k in formal_keywords)
        matched_informal = any(k in name for k in informal_keywords)
        if matched_formal and not matched_informal:
            formal.append(em)
        elif matched_informal and not matched_formal:
            informal.append(em)
        else:
            # faces and hearts biased to informal
            if "face" in name or "smile" in name or "heart" in name or "thumb" in name or "coffee" in name:
                informal.append(em)
            else:
                other.append((em, name))

    formal = _uniq(formal)
    informal = _uniq(informal)

    # fill using other heuristics until top_n each
    for em, name in other:
        if len(formal) < top_n and any(k in name for k in ("book","pen","page","document","notebook","scroll","file","envelope")):
            formal.append(em)
        elif len(informal) < top_n and any(k in name for k in ("face","smile","heart","thumb","coffee","cat","dog","party","pizza","beer","rocket")):
            informal.append(em)
        if len(formal) >= top_n and len(informal) >= top_n:
            break

    # final fill
    others_flat = [e for (e, _) in other]
    idx = 0
    while len(formal) < top_n and idx < len(others_flat):
        formal.append(others_flat[idx]); idx += 1
    while len(informal) < top_n and idx < len(others_flat):
        informal.append(others_flat[idx]); idx += 1

    formal = formal[:top_n]
    informal = informal[:top_n]

    # cache
    try:
        EMOJI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EMOJI_CACHE_PATH, 'w', encoding='utf-8') as fh:
            json.dump({"formal": formal, "informal": informal}, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    _EMOJI_LIST = list(dict.fromkeys(formal + informal))
    _EMOJI_NAMES = [_demojize_name(e) for e in _EMOJI_LIST]
    return formal, informal

def _build_runtime_index(model_name: str = "all-mpnet-base-v2"):
    """
    Builds embeddings for the full emoji name list and populates globals.
    This is performed lazily when find_emojis_for_token is first called.
    """
    global _EMOJI_EMBS, _EMOJI_LIST, _EMOJI_NAMES, _EMOJI_INDEX_BUILT
    if _EMOJI_INDEX_BUILT:
        return
    # ensure emoji lists available
    if not _EMOJI_LIST:
        # try load cache with generous top_n
        load_or_extract_emojis(top_n=200)
    # fallback: populate from emoji.EMOJI_DATA if still empty
    if not _EMOJI_LIST:
        _EMOJI_LIST = list(emoji.EMOJI_DATA.keys())[:500]
        _EMOJI_NAMES = [_demojize_name(e) for e in _EMOJI_LIST]

    model = _lazy_load_embed_model(model_name)
    # prepare textual descriptors: emoji name plus common keywords split
    descriptors = []
    for nm in _EMOJI_NAMES:
        # use the raw demojized name plus split tokens as descriptor
        tokens = re.findall(r"[a-z0-9]+", nm)
        descriptors.append(" ".join([nm] + tokens))
    # embed
    embs = model.encode(descriptors, convert_to_numpy=True, normalize_embeddings=True)
    _EMOJI_EMBS = np.array(embs, dtype=np.float32)
    _EMOJI_INDEX_BUILT = True

def find_emojis_for_token(token: str, top_k: int = 8, model_name: str = "all-mpnet-base-v2") -> List[str]:
    """
    Return up to top_k emojis semantically matching `token`.
    Uses sentence-transformers to embed token and finds nearest emoji names.
    """
    if not token or not token.strip():
        return []
    token_clean = re.sub(r"[^\w\s]", " ", token.lower())
    token_clean = token_clean.strip()
    try:
        _build_runtime_index(model_name=model_name)
    except Exception:
        # if embedding model unavailable, fallback to simple keyword scan
        matches = []
        t = token_clean.lower()
        for em in emoji.EMOJI_DATA.keys():
            name = _demojize_name(em)
            if t == name or t in name or name in t:
                matches.append(em)
                if len(matches) >= top_k:
                    break
        if matches:
            return matches
        # last resort sample
        return random.sample(list(emoji.EMOJI_DATA.keys()), min(top_k, len(emoji.EMOJI_DATA)))

    # embed token variants
    model = _lazy_load_embed_model(model_name)
    cand_texts = [token_clean] + token_clean.split()[:3]
    q_emb = model.encode([" ".join(cand_texts)], convert_to_numpy=True, normalize_embeddings=True)
    q_emb = q_emb.astype(np.float32)
    # cosine similarity via dot (embeddings normalized)
    sims = np.dot(_EMOJI_EMBS, q_emb[0])
    best_idx = np.argsort(-sims)[:max(top_k, 1)]
    out = []
    for idx in best_idx:
        if idx < 0 or idx >= len(_EMOJI_LIST):
            continue
        out.append(_EMOJI_LIST[idx])
        if len(out) >= top_k:
            break
    # dedupe & return
    return list(dict.fromkeys(out))

def get_allowed_emojis(register: str) -> List[str]:
    """
    Return cached formal/informal lists (try cache if globals empty).
    """
    # ensure cache exists
    try:
        formal, informal = load_or_extract_emojis(top_n=50)
    except Exception:
        formal, informal = [], []
    return formal if register == "formal" else informal

# small convenience: map a whole text by tokens -> emojis for debugging
def map_tokens_to_emojis(text: str, top_k_per_token: int = 3) -> Dict[str, List[str]]:
    tokens = re.findall(r"\b\w[\w']*\b", text.lower())
    out = {}
    for t in tokens:
        out[t] = find_emojis_for_token(t, top_k=top_k_per_token)
    return out