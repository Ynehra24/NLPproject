# Adv-Humanizer v3: Composite-Scored Beam Perturbation (CSBP) Framework 🛡️🔥

Adv-Humanizer is a character-level adversarial attack framework that **evades modern AI text detectors** by restructuring tokens at the byte-level rather than semantically paraphrasing. Version 3 is a ground-up redesign introducing architecture-specific evasion strategies natively targeting BPE tokenization vulnerabilities in both BERT and RoBERTa models, combined with a multi-metric composite scorer that enforces readability *during* the attack loop rather than measuring it post-hoc.

**Targets defeated:** `stats_baseline` (GPT-2 perplexity/rank), `fast_detectgpt`, `binoculars`, `detectgpt`, `kgw_watermark`, and fine-tuned classifiers (BERT-base, RoBERTa-base). Commercial platforms ZeroGPT, Originality.ai, and QuillBot return 0% AI-generated score on perturbed outputs.

---

## 📂 Directory Structure

```text
NLPproject/
├── characterlevelattacks/
│   ├── coreattacks/
│   │   ├── humanizer.py              # Entry point, CLI, batch evaluation
│   │   ├── composite_scorer.py       # Evasion-Oriented Scoring Engine (v3)
│   │   ├── csbp_loop.py              # CSBP v3 Beam Search + all strategies
│   │   ├── homoglyph_attack.py       # Character substitution maps, ZWSP handlers
│   │   ├── emoji_insertion.py        # Register-aware emoji extraction
│   │   ├── detector_eval.py          # Standard classifier evaluation
│   │   ├── fast_eval.py              # Rapid testing utilities
│   │   ├── hc3_m4_attack.py          # Targeting HC3 and M4 datasets
│   │   ├── run_humanizer_eval.py     # Full evaluation pipeline
│   │   ├── pair_export.py            # Teammate pair CSV export (post-processing)
│   │   ├── formality_model/          # Trained Register Gate (LogReg + MPNet)
│   │   │   ├── embed_model/          # all-mpnet-base-v2 weights
│   │   │   ├── classifier.joblib     # Trained LogReg classifier
│   │   │   └── scaler.joblib         # StandardScaler for style features
│   │   └── attacked_outputs/         # Attacked text output logs (.csv)
│   ├── advanced_metrics.py           # Detailed evaluation metrics & flip-rate calculator
│   ├── test_advanced_metrics.py      # Tests for metrics pipeline
│   ├── maincore.ipynb                # Training notebook: register gate + experiments
│   ├── datasetdownloaderhf.py        # HF Datasets download script
│   ├── redditdatasetdownloader.py    # Reddit-specific extraction script
│   ├── datasetpreview.py             # Head viewer for downloaded datasets
│   ├── conversioncode.py             # CSV / Parquet format conversion handlers
│   ├── parquetcombiner.py            # Merging datasets
│   ├── humanvsai/                    # General Human vs AI baseline data
│   ├── emojibased/                   # Emoji distribution baseline data
│   └── stylometric/                  # Stylometric feature references
```

---

## 🚀 Evaluation Results (vs Baselines)

Comparison of leading character-level methods against the novel **CSBP v3** framework. We report Attack Success Rate (ASR), Semantic Similarity (`cos`), Perplexity (`PPL`), and BPE Disruption (`BPE`) metrics evaluated against fine-tuned architecture classifiers from the TextAttack HuggingFace hub.

### BERT-base

| Dataset     | DWB   | TextBugger | Pruthi | Charmer-Fast | Charmer | **CSBP** | Charmer cos | CSBP cos | CSBP PPL | CSBP BPE |
|-------------|-------|------------|--------|--------------|---------|----------|-------------|----------|----------|----------|
| **AG-News** | 60.51 | 50.85      | 90.02  | 95.86        | 98.51   | **98.0** | 0.95        | **0.97** | 66.3     | 0.96     |
| **QNLI**    | 71.57 | 75.77      | 17.70  | 94.69        | 97.68   | **86.0** | 0.94        | **0.99** | 133.7    | 0.93     |
| **RTE**     | 65.67 | 74.13      | 62.19  | 89.55        | 97.01   | **56.0** | 0.86        | **0.98** | 262.3    | 0.95     |
| **SST-2**   | 81.39 | 68.49      | 90.94  | 100.00       | 100.00  | **98.0** | 0.90        | **0.93** | 126.8    | 0.96     |
| **HC3**     | —     | —          | —      | —            | —       | **98.0** | —           | **0.98** | 34.0     | 0.99     |
| **M4**      | —     | —          | —      | —            | —       | **82.0** | —           | **0.99** | 45.9     | 0.79     |

### RoBERTa-base

| Dataset     | DWB   | TextBugger | Pruthi | Charmer-Fast | Charmer | **CSBP** | Charmer cos | CSBP cos | CSBP PPL | CSBP BPE |
|-------------|-------|------------|--------|--------------|---------|----------|-------------|----------|----------|----------|
| **AG-News** | 56.81 | 51.21      | 88.91  | 91.87        | 96.88   | **98.0** | 0.95        | **0.97** | 65.8     | 0.98     |
| **QNLI**    | 64.34 | 67.39      | 17.45  | 96.95        | 97.86   | **100.0**| 0.92        | **0.97** | 101.4    | 0.95     |
| **RTE**     | 62.67 | 71.43      | 49.31  | 91.71        | 97.24   | **66.0** | 0.85        | **0.98** | 524.4    | 0.88     |
| **SST-2**   | 84.27 | 61.10      | 92.93  | 99.39        | 99.51   | **94.0** | 0.89        | **0.95** | 303.1    | 0.94     |
| **HC3**     | —     | —          | —      | —            | —       | **98.0** | —           | **0.98** | 42.0     | 0.99     |
| **M4**      | —     | —          | —      | —            | —       | **82.0** | —           | **0.99** | 43.7     | 0.78     |

**Key finding:** CSBP maintains consistently higher semantic similarity (`cos`) than all baselines across every evaluated dataset while achieving maximum BPE fragmentation. The RTE gap vs Charmer reflects the short-text problem: very few tokens mean each perturbation has outsized impact on cosine similarity, causing more candidates to fail the semantic gate.

---

## 🧠 Architecture Overview

### Stage 1 — Register Gate (`humanizer.py` + `formality_model/`)

Before attacking, text is classified as `formal` or `informal` using a two-component inference pipeline trained in `maincore.ipynb`:

- **Embedding:** `all-mpnet-base-v2` sentence embedding (768-dim)
- **Style features:** 11 hand-crafted stylistic signals (emoji density, exclamation rate, slang token overlap, contraction count, avg word length, caps-word rate, repeated-char sequences, ellipsis count, formal vocabulary score, informal interjection presence, question rate), scaled with a saved `StandardScaler` and multiplied by `STYLE_WEIGHT=5.0` before concatenation
- **Classifier:** Logistic regression over the concatenated vector

The informal-class probability is passed **continuously** (not thresholded) as `emoji_prob` into `csbp_loop`, acting as a probabilistic gate on the `emoji_attach` strategy. A value of 1.0 means emoji is always attempted; 0.0 means it never fires.

### Stage 2 — Vulnerability Pre-Analysis (`composite_scorer.py::analyze_original`)

A single GPT-2 forward pass maps token-level confidence down to word indices, identifying:

- **Priority words:** word indices where GPT-2 assigns rank ≤ 5 to the true next token (i.e., the model is most confident, so attacking here has highest impact)
- **Watermark words:** word indices landing on KGW green-list tokens (gamma=0.5, key=15485863)
- **Word sensitivity weights:** inverse average token rank per word, used by `_weighted_sample` to bias perturbation toward high-value positions

### Stage 3 — CSBP v3 Beam Search (`csbp_loop.py`)

At each round `k`, every active beam expands by generating `n_candidates` perturbations using the strategy pool below, scored by the composite scorer, and pruned to `beam_width` survivors. A patience counter triggers early stopping if no improvement is observed for `PATIENCE` consecutive rounds (9 for BERT, 12 for RoBERTa).

**Core strategies (weighted by frequency in pool):**

| Strategy | Weight | Description |
|---|---|---|
| `blitz` | 4× | Deep homoglyph + diacritic + ZWSP on high-proportion of eligible words |
| `zwsp_flood` | 4× | Homoglyph all chars then insert ZWSP at every inter-character boundary |
| `roberta_subword` | 4× | Internal ZWSP injection at 1-3 positions + light homoglyph (RoBERTa-specific) |
| `scorched_earth` | 3× | Full homoglyph (0.9) + diacritic (0.7) + heavy whitespace + optional emoji |
| `bpe_break` | 1× | Exhaustive per-character vulnerability scan for maximal BPE token inflation |
| `high_conf` | 1× | BPE-targeted or blitz attack on priority words identified by pre-analysis |
| `whitespace` | 1× | 2-3 invisible chars at random interior positions |
| `emoji_attach` | 1× | Emoji prefix/suffix + paired char-level attack (gated by `emoji_prob`) |
| `watermark` | 1× | Best-of-8 blitz/homoglyph on KGW green-list tokens |
| `combined` | 1× | BPE-break + emoji + whitespace synergy across different word subsets |
| `micro_sponge` | 2× | 5-15 invisible chars appended at sentence end |
| `punctuation_bind` | 2× | ZWSP injected before and after all punctuation marks |
| `random_case` | 2× | Random interior case-flipping (RoBERTa byte-BPE OOD) |
| `math_unicode` | 2× | Math Bold Unicode substitution (4-byte UTF-8 per char) |
| `fullwidth` | 1× | Fullwidth ASCII variants (3-byte UTF-8, OOD for RoBERTa) |

**Adaptive intensity scaling:** text length gates iterations and candidates. Long texts (>100 words) get 2× iterations and 2× candidates. Medium texts (30-100 words) get 1.5× each.

**Architecture detection:** `--arch roberta` adds `random_case`, `math_unicode`, and `fullwidth` to the strategy pool with additional weight.

### Stage 4 — Composite Scorer (`composite_scorer.py`)

The composite score `S` is the central optimization target, replacing raw classifier loss:

```
S_raw = w_clf·clf + w_ev·ev + w_bpe·bpe + w_wm·wm + w_read·read + w_sim·cos
S = S_raw × (0.3 + 0.7 × coherence_gate(cos))
```

**With classifier:** `w_clf=0.45, w_ev=0.10, w_bpe=0.25, w_wm=0.08, w_read=0.04, w_sim=0.08`

**Statistical-only mode:** `w_ev=0.35, w_bpe=0.35, w_wm=0.10, w_read=0.05, w_sim=0.15`

Components:

- **Classifier score** (`clf`): `1 - confidence_fn(attacked)` — only active when a real classifier is wired in
- **Evasion score** (`ev`): sigmoid over GPT-2 perplexity and average token rank — higher PPL/rank = more evasive
- **BPE disruption** (`bpe`): 0.6 × token inflation + 0.4 × invisible-char density
- **Watermark evasion** (`wm`): sigmoid of negative KGW z-score
- **Readability** (`read`): penalises PPL < 30 (suspiciously smooth) and PPL > 10,000 (gibberish)
- **Cosine similarity** (`cos`): SBERT (`all-mpnet-base-v2`) on cleaned text (ZWSP stripped, diacritics normalized, homoglyphs reversed before encoding — so cosine reflects true semantics, not byte fragmentation)
- **Coherence gate** (`coh`): multiplicative — only hard-discounts true gibberish (cos < 0.10)

---

## 🔧 Pair Export for Downstream Evaluation

After running the attack pipeline, the outputs in `attacked_outputs/` are converted to a standardized pair format for downstream teammate evaluation (e.g., detection, readability assessment, human study):

```python
import pandas as pd

INPUT_FILE  = "attacked_outputs/hc3_humanizer_roberta.csv"
OUTPUT_FILE = "teammate_pairs_charlevel.csv"

df = pd.read_csv(INPUT_FILE)

pairs = []
for i, row in df.iterrows():
    pairs.append({
        "pair_id":        f"p{i:05d}",
        "original_text":  str(row["text"]).strip(),
        "humanized_text": str(row["humanized_text"]).strip(),
        "attack_type":    "char",
        "generator_model": "roberta",
    })

pd.DataFrame(pairs).to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
```

The output schema (`pair_id`, `original_text`, `humanized_text`, `attack_type`, `generator_model`) is the shared contract with the broader group pipeline for integration with detection and scoring modules.

---

## 🛠️ Installation & Usage

```bash
pip install torch sentence-transformers transformers scikit-learn \
            pandas tqdm aiohttp pyarrow joblib nltk emoji
```

### Single text

```bash
python characterlevelattacks/coreattacks/humanizer.py \
    "Your target AI-generated text here." \
    --target-model textattack/bert-base-uncased-ag-news \
    --iterations 7 --beam-width 7 --cands 20 --device mps
```

### Dataset batches

```bash
python characterlevelattacks/coreattacks/humanizer.py \
    --hc3 path/to/hc3.parquet \
    --m4  path/to/m4.parquet  \
    --sample-size 100 \
    -o results.csv
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--target-model` | auto-selected | HuggingFace model ID to attack |
| `--dataset` | `agnews` | Dataset key for auto model selection |
| `--arch` | `bert` | `bert` or `roberta` (affects strategy weighting) |
| `--iterations` | `7` | Max beam search rounds (auto-scaled by text length) |
| `--cands` | `20` | Candidates generated per beam per round |
| `--beam-width` | `8` | Beams retained after pruning |
| `--device` | auto | `cpu`, `mps`, or `cuda` |
| `--sample-size` | `10` | Texts to sample when reading from file |

---

## 📊 Threat Model

The framework operates in a **white-box / grey-box** setting: the attacker can query the target classifier repeatedly for confidence scores and label predictions, with no access to model weights. The attack budget is bounded by `iterations × n_candidates × beam_width` classifier queries per sample.

**Hard constraints enforced during beam ranking:**
- Cosine similarity ≥ 0.60 (texts < 45 words) / ≥ 0.85 (texts > 45 words)
- GPT-2 perplexity ≤ 400 (soft — handled by readability component in `S`)

**Why this works against BERT/RoBERTa:** Both models process text through tokenizers whose vocabularies are built on clean ASCII/Unicode training corpora. Invisible characters (ZWSP, ZWNJ, ZWJ), homoglyphs from Cyrillic/Greek code points, and math-bold Unicode force the tokenizer to produce rare or out-of-vocabulary subword sequences, causing the model's internal representations to shift into low-confidence regions — while human readers parse the visual surface as unchanged.
