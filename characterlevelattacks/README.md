# Adv-Humanizer v2: Scorched-Earth Adversarial Attack Framework 🛡️🔥

Adv-Humanizer is a character-level adversarial attack framework that **evades modern AI text detectors** by restructuring tokens at the byte-level rather than semantically paraphrasing. Version 2 is a ground-up redesign: the objective function was inverted from *maximizing similarity to AI text* to *maximizing evasion of AI detectors*.

**Targets defeated:** `stats_baseline` (GPT-2 perplexity/rank), `fast_detectgpt`, `binoculars`, `detectgpt`, and `kgw_watermark`.

---

## 📂 Directory Structure

```text
NLPproject/
├── characterlevelattacks/
│   ├── coreattacks/
│   │   ├── humanizer.py           # Entry point & CLI
│   │   ├── composite_scorer.py    # v2 Evasion-Oriented Scoring Engine
│   │   ├── csbp_loop.py           # v2 CSBP Beam Search (7 strategies)
│   │   ├── homoglyph_attack.py    # Character substitution maps
│   │   ├── emoji_insertion.py     # Register-aware emoji extraction
│   │   ├── formality_model/       # Trained Register Gate (LogReg + MPNet)
│   │   └── attacked_outputs/      # Output logs
│   ├── advanced_metrics.py        # Evaluation pipeline (no emoji stripping)
│   ├── humanvsai/                 # HC3, M4 datasets
│   ├── emojibased/                # Emoji style datasets
│   └── stylometric/               # Stylometric feature utilities
├── detector_evaluation/
│   ├── detectors/
│   │   ├── stats_baseline/        # GPT-2 perplexity + token rank
│   │   ├── roberta_classifier/    # Fine-tuned RoBERTa
│   │   ├── fast_detectgpt/        # Log-probability discrepancy
│   │   ├── binoculars/            # Cross-entropy ratio
│   │   ├── detectgpt/             # Log-prob curvature
│   │   └── watermark/             # KGW green-list z-score
│   └── evaluation/
│       └── paired_analysis.py     # Flip-rate & ASR calculator
```

---

## 🧠 Architecture

### Pipeline Overview

```
AI Text → humanizer.py → Register Gate → CSBP v2 Beam Search → Humanized Output → CSV
                                               │
                              ┌────────────────┴─────────────────┐
                              │         analyze_original()        │
                              │  • GPT-2 per-token rank           │
                              │  • Priority words  (rank ≤ 5)     │
                              │  • Watermark words (KGW green)    │
                              └────────────────┬─────────────────┘
                                               │
                              ┌────────────────┴─────────────────┐
                              │  Per Round × K rounds             │
                              │  generate_candidates()            │
                              │    → 7 strategies (sampled)       │
                              │  composite_score() for each       │
                              │  Keep top beam_width beams        │
                              └──────────────────────────────────┘
```

---

### Stage 1 — Register Gate (`humanizer.py`)

Before attacking, the text is classified as **formal** or **informal** using:
- `all-mpnet-base-v2` sentence embeddings (768-dim)
- Trained logistic regression classifier with 11 stylometric features

This label is used for context (emoji pool selection) but does **not** constrain the attack strategy.

---

### Stage 2 — Pre-Analysis (`composite_scorer.analyze_original`)

One forward pass through GPT-2 on the **original text** to identify exactly where the AI signal is strongest:

**Priority words** — words containing tokens GPT-2 predicted at rank ≤ 5. These are the highest-confidence predictions and carry the most AI signal weight.

**Watermark words** — words containing tokens that land in the KGW green list. Formula:
```
seed = SHA256(hash_key=15485863 || prev_token_id)
green_list = RNG(seed).choice(vocab_size, n=vocab_size × γ=0.5)
```
Both sets are passed into every `generate_candidates()` call so perturbations are **surgical**, not random.

---

### Stage 3 — CSBP v2 Beam Search (`csbp_loop.py`)

```
K rounds × beam_width beams × n_candidates per beam
Default: K=7, beam_width=7, n_candidates=20
→ up to 980 candidates evaluated per text
```

At each round, every active beam expands into `n_candidates` using one of **7 strategies** (sampled with weight — scorched earth is 3× weighted):

#### Strategy 1: `bpe_break`
For each target word, exhaustively test every character position for:
- Invisible char insertion (ZWSP, ZWJ, ZWNJ)
- Homoglyph substitution
- Diacritic substitution

Keep the top `max_edits` positions by **token-count delta**. Applied to ≤20% of words.

```
"experienced" → 2 BPE tokens
"exp​er​ienced" → 7 BPE tokens  (ZWSP at positions 3,5)
delta = +5 tokens → maximum disruption
```

#### Strategy 2: `high_conf`
Target exactly the `priority_words` from pre-analysis (rank ≤ 5 tokens). Apply `apply_bpe_targeted`, homoglyph (rate=0.7), or diacritic (rate=0.7). Only half the priority words are targeted per candidate for diversity.

#### Strategy 3: `whitespace`
Insert 2–3 invisible Unicode characters at random interior word positions:
`\u200B` `\u200C` `\u200D` `\u00AD` `\u2060` `\uFEFF`

These are Unicode category Cf (Format), invisible to humans, but split GPT-2's pre-tokeniser regex `\p{L}+`, causing 1 word → multiple token chunks.

#### Strategy 4: `emoji_attach`
Attach emojis **directly to words** (no space):
```
"experienced🔥"   or   "🔥experienced"
```
Includes ZWJ sequences (👨‍💻 👩‍🔬 🧑‍🚀) whose multi-byte representation consumes more token slots. The word following the emoji loses its `Ġ` space prefix → different token ID → shifted prediction probability.

#### Strategy 5: `watermark`
Target `watermark_words` from pre-analysis. For each, try 6 perturbation attempts and keep whichever causes the **largest token-ID change** (maximising probability of flipping from green list to red list), directly suppressing the KGW z-score.

#### Strategy 6: `combined`
Mix of BPE-break + emoji attach + whitespace in one candidate.

#### Strategy 7: `scorched_earth` ← **3× weighted**
Apply ALL attacks simultaneously to 40–80% of eligible words in sequence:
1. Homoglyph substitution at rate=0.8
2. Diacritic substitution at rate=0.6 (on the result)
3. Heavy invisible-char insertion: 3–5 chars per word
4. Emoji attachment: 40% probability per word

Maximum possible BPE fragmentation per token. This is the primary strategy against statistical detectors.

---

### Stage 4 — Evasion-Oriented Scoring (`composite_scorer.composite_score`)

Every candidate is scored. **Higher S = better attack.**

| Component | Weight | Objective |
|-----------|--------|-----------|
| **Evasion** | 55% | Push PPL into human range (60–5000) and avg token rank above 20. Uses sigmoid functions centred on human/AI boundary. |
| **BPE Disruption** | 15% | `(attacked_token_count / original_token_count) - 1`. Score 1.0 at 2× inflation. |
| **Watermark Evasion** | 15% | `sigmoid(-z)` where z is the KGW z-score of the attacked text. |
| **Readability** | 5% | Full score for PPL 30–5000. Penalises both too-smooth (AI) and true gibberish (>5000). |
| **Similarity** | 10% | SBERT cosine bonus via `all-mpnet-base-v2`. |
| **Coherence Gate** | ×mult | `sigmoid` gate — if cosine < 0.20, multiplicatively penalises score toward zero. Prevents nonsense winning. |

**Key design decision:** The coherence gate threshold is relaxed to 0.20 (from 0.50 in v1) because invisible-char perturbations can lower SBERT cosine even though the text is visually identical to a human reader (SBERT's own BPE tokeniser is also disrupted by ZWSP/ZWJ).

---

### Stage 5 — Beam Pruning & Output

After all candidates are scored:
- Sort descending by S
- Keep top `beam_width` (default 7) as beams for next round
- Track best candidate across all rounds
- Return after K rounds (or immediately on classifier flip if a real classifier is provided)

Final output is written to CSV:
```
pair_id, original_text, humanized_text, attack_type, generator_model
```

---

## 🛠️ Installation

```bash
pip install torch sentence-transformers transformers scikit-learn \
            pandas tqdm nltk emoji pyarrow joblib
```

```python
import nltk
nltk.download(['wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'stopwords'])
```

---

## 📖 Usage

### Single text
```bash
python characterlevelattacks/coreattacks/humanizer.py "Your AI-generated text here."
```

### With custom parameters
```bash
python characterlevelattacks/coreattacks/humanizer.py "Your text" \
    --iterations 7 \
    --cands 20 \
    --beam-width 7 \
    --device mps \
    -o ~/Downloads/results.csv
```

### Batch (HC3 / M4 datasets)
```bash
python characterlevelattacks/coreattacks/humanizer.py \
    --hc3 path/to/hc3.parquet \
    --m4  path/to/m4.parquet \
    --sample-size 100 \
    --iterations 7 --cands 20 --beam-width 7 \
    -o results.csv
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `text` | — | Text string or path to dataset file |
| `--hc3` | — | HC3 dataset path |
| `--m4` | — | M4 dataset path |
| `--sample-size` | 10 | Number of samples to draw |
| `--iterations` | 7 | CSBP beam search rounds (K) |
| `--cands` | 20 | Candidates per beam per round |
| `--beam-width` | 7 | Beams kept per round |
| `--device` | mps | `mps` or `cpu` |
| `-o` / `--output` | `~/Downloads/...csv` | Output CSV path |

---

## 📊 Models

| Model | Purpose |
|-------|---------|
| `openai-community/gpt2` | Perplexity, token rank, KGW watermark, BPE analysis |
| `sentence-transformers/all-mpnet-base-v2` | SBERT cosine similarity (coherence gate) |
| `formality_model/embed_model` (local) | Register classification |

All models run on **MPS (Apple Silicon)** by default with CPU fallback.

---

## 🔄 Changelog

### v2.0 (Current)
- **Scoring inverted**: evasion-first (55% weight) vs old similarity-first
- **7 attack strategies** including scorched earth (3× draw weight)
- **Pre-analysis**: identifies high-confidence and watermark-bearing words before any perturbation
- **BPE-aware targeting**: perturbations placed at positions that maximally fragment tokenisation
- **White-box KGW attack**: direct green→red token flipping
- **Scorched earth**: all attacks stacked on each word simultaneously
- **Evaluation fixed**: `advanced_metrics.py` no longer strips emojis/invisible chars before detector scoring

### v1.0 (Legacy)
- Similarity-based composite score (cosine 35%, Levenshtein 15%, Jaccard 15%, stylometric 10%, fluency 25%)
- Random homoglyph/diacritic perturbation
- Space-separated emoji insertion (stripped by detectors — ineffective)
- ~43–69% ASR against BERT/RoBERTa classifiers; 0% against statistical detectors
