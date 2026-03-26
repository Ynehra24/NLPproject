import re
import math
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"CompositeScorer using device: {device}")

SBERT          = SentenceTransformer('all-mpnet-base-v2', device=device)
GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
GPT2_MODEL     = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
GPT2_MODEL.eval()

def cosine_score(original: str, attacked: str) -> float:
    embs = SBERT.encode([original, attacked], convert_to_numpy=True)
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])

def perplexity_score(text: str, max_length: int = 128) -> float:
    enc = GPT2_TOKENIZER(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        loss = GPT2_MODEL(**enc, labels=enc['input_ids']).loss.item()
    ppl = math.exp(loss)
    return 1.0 / (1.0 + math.log(ppl + 1e-8))

def levenshtein_score(original: str, attacked: str) -> float:
    a, b = original, attacked
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev[j-1] + cost)
    return 1.0 - (dp[n] / max(m, n))

def jaccard_score(original: str, attacked: str) -> float:
    set_a = set(re.findall(r'\b\w+\b', original.lower()))
    set_b = set(re.findall(r'\b\w+\b', attacked.lower()))
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

def stylometric_delta(original: str, attacked: str) -> float:
    def style_vec(text):
        tokens = re.findall(r'\b\w+\b', text)
        if not tokens:
            return np.zeros(3)
        return np.array([
            np.mean([len(t) for t in tokens]),
            len(set(tokens)) / len(tokens),
            len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
        ])
    v_o = style_vec(original)
    v_a = style_vec(attacked)
    delta = np.linalg.norm(v_o - v_a) / (np.linalg.norm(v_o) + 1e-8)
    return float(1.0 - min(delta, 1.0))

def composite_score(
    original: str, attacked: str,
    w_cosine=0.35, w_ppl=0.25, w_levenshtein=0.15,
    w_jaccard=0.15, w_stylo=0.10,
) -> dict:
    c  = cosine_score(original, attacked)
    p  = perplexity_score(attacked)
    l  = levenshtein_score(original, attacked)
    j  = jaccard_score(original, attacked)
    st = stylometric_delta(original, attacked)
    S  = w_cosine*c + w_ppl*p + w_levenshtein*l + w_jaccard*j + w_stylo*st
    return {
        'S': round(S, 4), 'cosine': round(c, 4), 'perplexity': round(p, 4),
        'levenshtein': round(l, 4), 'jaccard': round(j, 4), 'stylometric': round(st, 4),
    }

if __name__ == "__main__":
    orig    = "The stock market experienced significant volatility today."
    attacked = "Тhe stоck markеt еxperienced significant volatility tоday."
    result  = composite_score(orig, attacked)
    print("\n--- Composite Score Breakdown ---")
    for k, v in result.items():
        print(f"  {k:<14}: {v}")
