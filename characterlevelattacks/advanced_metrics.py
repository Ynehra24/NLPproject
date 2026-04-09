import re
import math
import torch
import numpy as np
import jellyfish
import difflib
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji

device = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# -----------------------------------------------------
# 1. Zone Detector Pipeline
# -----------------------------------------------------
URL_REGEX = re.compile(
    r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
    re.IGNORECASE
)

def apply_zone_detector(text: str) -> str:
    """Masks URLs only. Emojis and invisible chars are intentional attack
    perturbations and must reach the detectors intact — do NOT strip them."""
    # Mask URLs only
    text = URL_REGEX.sub('[URL]', text)
    return text

# -----------------------------------------------------
# 2. SBERT Cosine Similarity
# -----------------------------------------------------
class SBERTScorer:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    
    def calculate(self, original: str, attacked: str) -> dict:
        embs = self.model.encode([original, attacked], convert_to_numpy=True)
        # Cosine similarity calculation
        norm_a = np.linalg.norm(embs[0])
        norm_b = np.linalg.norm(embs[1])
        if norm_a == 0 or norm_b == 0:
            sim = 0.0
        else:
            sim = np.dot(embs[0], embs[1]) / (norm_a * norm_b)
        
        sim = float(sim)
        return {
            "sbert_similarity": sim,
            "sbert_passed": sim >= 0.90
        }

# -----------------------------------------------------
# 3. PPL Score (GPT-2)
# -----------------------------------------------------
class PPLScorer:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.model.eval()
    
    def calculate(self, text: str, max_length: int = 512) -> dict:
        if not text.strip():
            return {"ppl_score": 0.0, "ppl_passed": False}
        
        enc = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
        if enc['input_ids'].shape[1] < 2:
            return {"ppl_score": 0.0, "ppl_passed": False}
            
        with torch.no_grad():
            loss = self.model(**enc, labels=enc['input_ids']).loss.item()
            
        ppl = math.exp(loss)
        return {
            "ppl_score": ppl,
            "ppl_passed": ppl <= 400.0
        }

# -----------------------------------------------------
# 4. Damerau-Levenshtein with Boundary Penalty
# -----------------------------------------------------
def is_word_boundary(text: str, index: int) -> bool:
    """
    Check if the character at 'index' in 'text' is a word boundary.
    A character is considered at a word boundary if it is an alphanumeric character
    and either the character before or after it is a non-alphanumeric character (or it's at the start/end).
    """
    if text == "" or index < 0 or index >= len(text):
        return False
        
    char = text[index]
    if not char.isalnum():
        return False # We only penalize boundaries of actual words, not spaces/punctuation itself.
        
    is_start = index == 0 or not text[index - 1].isalnum()
    is_end = index == len(text) - 1 or not text[index + 1].isalnum()
    
    return is_start or is_end

def count_boundary_edits(original: str, attacked: str) -> float:
    matcher = difflib.SequenceMatcher(None, original, attacked)
    opcodes = matcher.get_opcodes()
    boundary_penalty_count = 0
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            # Check original text boundary edits
            for i in range(i1, i2):
                if is_word_boundary(original, i):
                    boundary_penalty_count += 1
        elif tag == 'delete':
            for i in range(i1, i2):
                if is_word_boundary(original, i):
                    boundary_penalty_count += 1
        elif tag == 'insert':
            # For inserts, we check if inserted char becomes a word boundary in the attacked text
            for j in range(j1, j2):
                if is_word_boundary(attacked, j):
                    boundary_penalty_count += 1
                    
    return boundary_penalty_count

def damerau_levenshtein_boundary(original: str, attacked: str, boundary_multiplier: float = 1.5) -> float:
    # Base distance using jellyfish implementation
    base_dist = jellyfish.damerau_levenshtein_distance(original, attacked)
    
    # Extra penalty computation
    # Every edit has a base cost of 1. If it's a boundary, it should cost 1.5.
    # Therefore, the additional penalty is +0.5 per boundary edit.
    boundary_edits = count_boundary_edits(original, attacked)
    extra_penalty = boundary_edits * (boundary_multiplier - 1.0)
    
    return float(base_dist + extra_penalty)

# -----------------------------------------------------
# 5. Jaccard n-gram
# -----------------------------------------------------
def get_ngrams(text: str, n: int) -> set:
    tokens = re.findall(r'\b\w+\b', text.lower())
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def jaccard_ngram_score(original: str, attacked: str, m1=0.5, m2=0.3, m3=0.2) -> float:
    j1 = jaccard_similarity(get_ngrams(original, 1), get_ngrams(attacked, 1))
    j2 = jaccard_similarity(get_ngrams(original, 2), get_ngrams(attacked, 2))
    j3 = jaccard_similarity(get_ngrams(original, 3), get_ngrams(attacked, 3))
    return float(m1*j1 + m2*j2 + m3*j3)

# -----------------------------------------------------
# 6. Stylometric Delta
# -----------------------------------------------------
# Common English function words
FUNCTION_WORDS = set([
    'a', 'an', 'the', 'and', 'but', 'or', 'so', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])

def extract_stylometric_features(text: str) -> dict:
    # 1. Sentence length variance
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    sentence_length_variance = np.var(lengths) if lengths else 0.0
    
    # 2. TTR (Type-Token Ratio)
    tokens = re.findall(r'\b\w+\b', text.lower())
    ttr = len(set(tokens)) / len(tokens) if tokens else 0.0
    
    # 3. Function word frequency
    func_word_count = sum(1 for t in tokens if t in FUNCTION_WORDS)
    func_word_freq = func_word_count / len(tokens) if tokens else 0.0
    
    return {
        "sentence_length_variance": float(sentence_length_variance),
        "ttr": float(ttr),
        "func_word_freq": float(func_word_freq)
    }

def stylometric_delta(original: str, attacked: str) -> dict:
    feat_o = extract_stylometric_features(original)
    feat_a = extract_stylometric_features(attacked)
    
    # Simplistic Euclidean delta scaled
    vec_o = np.array(list(feat_o.values()))
    vec_a = np.array(list(feat_a.values()))
    
    delta = np.linalg.norm(vec_o - vec_a)
    return {
        "stylometric_delta": float(delta),
        "features_original": feat_o,
        "features_attacked": feat_a
    }

# -----------------------------------------------------
# 7. Register Classifier
# -----------------------------------------------------
class RegisterClassifier:
    """
    Classifies text into formal/informal with an overall 99% logic.
    Gates emoji heavily toward informal text.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.clf = LogisticRegression()
        self.is_trained = False
        
    def fit_dummy_if_needed(self):
        """Fit dummy data so it can be used immediately without throwing errors."""
        if not self.is_trained:
            # Dummy training set representing formal and informal styles, plus emojis
            X_train = [
                "The scientific findings indicate a statistically significant correlation between variables.",
                "Therefore, it is imperative to acknowledge the aforementioned constraints in the analysis.",
                "We respectfully submit this comprehensive proposal for your esteemed review and consideration.",
                "Hey guys! Omg that was so lit 🚀🔥 lol",
                "haha yeah totally agree with u rn 😂",
                "Idk tbh, maybe we should just go and see what happens lmao"
            ]
            y_train = [1, 1, 1, 0, 0, 0] # 1 -> Formal, 0 -> Informal
            X_vec = self.vectorizer.fit_transform(X_train)
            self.clf.fit(X_vec, y_train)
            self.is_trained = True

    def predict(self, text: str) -> dict:
        self.fit_dummy_if_needed()
        
        # 1. Zone/Emoji Gate
        if emoji.emoji_count(text) > 0 or re.search(r'\blmao\b|\blol\b|\btbh\b', text.lower()):
            return {
                "prediction": "Informal",
                "confidence": 0.99,
                "gated_by_emoji": True
            }
            
        # 2. Logistic Regression prediction
        X_test = self.vectorizer.transform([text])
        proba = self.clf.predict_proba(X_test)[0]
        prediction = "Formal" if proba[1] > 0.5 else "Informal"
        confidence = proba[1] if prediction == "Formal" else proba[0]
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "gated_by_emoji": False
        }

# -----------------------------------------------------
# Full Evaluation Pipeline
# -----------------------------------------------------
class CharacterLevelMetricsPipeline:
    def __init__(self):
        self.sbert = SBERTScorer()
        self.ppl = PPLScorer()
        self.register_clf = RegisterClassifier()
        
    def evaluate(self, original: str, attacked: str) -> dict:
        # Pre-process zones
        orig_clean = apply_zone_detector(original)
        attack_clean = apply_zone_detector(attacked)
        
        return {
            "sbert": self.sbert.calculate(original, attacked),
            "gpt2_ppl": self.ppl.calculate(attacked),
            "damerau_levenshtein_boundary_penalty": damerau_levenshtein_boundary(original, attacked),
            "jaccard_ngram": jaccard_ngram_score(original, attacked),
            "stylometric": stylometric_delta(original, attacked),
            "register": self.register_clf.predict(attacked),
            "zone_detected_original": orig_clean,
            "zone_detected_attacked": attack_clean,
        }

if __name__ == "__main__":
    import json
    pipeline = CharacterLevelMetricsPipeline()
    orig_text = "The scientific findings indicate a statistically significant correlation between variables."
    attack_text = "The scientific findlngs indicatе a statistically signifiicant correlation between variables. 🚀 https://example.com"
    
    print("\n--- Evaluating text metrics ---")
    results = pipeline.evaluate(orig_text, attack_text)
    print(json.dumps(results, indent=2))
