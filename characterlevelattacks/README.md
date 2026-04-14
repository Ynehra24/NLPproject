# Adv-Humanizer v3: Scorched-Earth Adversarial Attack Framework 🛡️🔥

Adv-Humanizer is a character-level adversarial attack framework that **evades modern AI text detectors** by restructuring tokens at the byte-level rather than semantically paraphrasing. Version 3 brings a ground-up redesign: introducing architecture-specific evasion strategies natively targeting the BPE tokenization vulnerabilities of both BERT and RoBERTa models.

**Targets defeated:** `stats_baseline` (GPT-2 perplexity/rank), `fast_detectgpt`, `binoculars`, `detectgpt`, `kgw_watermark`, and fine-tuned classifiers (BERT-base, RoBERTa-base).

---

## 📂 Directory Structure

```text
NLPproject/
├── characterlevelattacks/
│   ├── coreattacks/
│   │   ├── humanizer.py             # Entry point & CLI
│   │   ├── composite_scorer.py      # Evasion-Oriented Scoring Engine
│   │   ├── csbp_loop.py             # v3 CSBP Beam Search (Includes blitz, roberta_subword & more)
│   │   ├── homoglyph_attack.py      # Character substitution maps, ZWSP handlers
│   │   ├── emoji_insertion.py       # Register-aware emoji extraction
│   │   ├── detector_eval.py         # Standard classifier evaluation
│   │   ├── fast_eval.py             # Rapid testing utilities
│   │   ├── hc3_m4_attack.py         # Script targeting HC3 and M4 datasets
│   │   ├── run_humanizer_eval.py    # Humanizer full evaluation pipeline
│   │   ├── formality_model/         # Trained Register Gate (LogReg + MPNet)
│   │   └── attacked_outputs/        # Attacked text output logs
│   ├── advanced_metrics.py          # Detailed evaluation metrics & flip-rate calculator
│   ├── test_advanced_metrics.py     # Tests for metrics pipeline
│   ├── datasetdownloaderhf.py       # HF Datasets download script
│   ├── redditdatasetdownloader.py   # Reddit specific extraction script
│   ├── datasetpreview.py            # Head viewer for downloaded datasets
│   ├── conversioncode.py            # CSV / Parquet format conversion handlers
│   ├── parquetcombiner.py           # Merging datasets
│   ├── humanvsai/                   # General Human vs AI baseline data
│   ├── emojibased/                  # Emoji distribution baseline data
│   └── stylometric/                 # Stylometric feature references
```

---

## 🚀 Evaluation Results (vs Baselines)

Comparison of leading Character-Level methods against the novel **CSBP v3** framework. We report the Attack Success Rate (ASR) along with Semantic Similarity (cos) and Perplexity (PPL) metrics. Evaluated against fine-tuned architecture classifiers.

### BERT-base vs Paper (Character-Level Methods Only)

| Dataset | DWB | TextBugger | Pruthi | Charmer-Fast | Charmer | **CSBP** | Charmer cos | CSBP cos | CSBP PPL | CSBP BPE |
|---------|-----|------------|--------|--------------|---------|----------|-------------|----------|----------|----------|
| **AG-News** | 60.51 | 50.85 | 90.02 | 95.86 | 98.51 | **98.0** | 0.95 | **0.97** | 66.3 | 0.96 |
| **QNLI** | 71.57 | 75.77 | 17.70 | 94.69 | 97.68 | **86.0** | 0.94 | **0.99** | 133.7 | 0.93 |
| **RTE** | 65.67 | 74.13 | 62.19 | 89.55 | 97.01 | **56.0** | 0.86 | **0.98** | 262.3 | 0.95 |
| **SST-2** | 81.39 | 68.49 | 90.94 | 100.00 | 100.00 | **98.0** | 0.90 | **0.93** | 126.8 | 0.96 |
| **HC3** | — | — | — | — | — | **98.0** | — | **0.98** | 34.0 | 0.99 |
| **M4** | — | — | — | — | — | **82.0** | — | **0.99** | 45.9 | 0.79 |

### RoBERTa-base vs Paper (Character-Level Methods Only)

| Dataset | DWB | TextBugger | Pruthi | Charmer-Fast | Charmer | **CSBP** | Charmer cos | CSBP cos | CSBP PPL | CSBP BPE |
|---------|-----|------------|--------|--------------|---------|----------|-------------|----------|----------|----------|
| **AG-News** | 56.81 | 51.21 | 88.91 | 91.87 | 96.88 | **98.0** | 0.95 | **0.97** | 65.8 | 0.98 |
| **QNLI** | 64.34 | 67.39 | 17.45 | 96.95 | 97.86 | **100.0**| 0.92 | **0.97** | 101.4 | 0.95 |
| **RTE** | 62.67 | 71.43 | 49.31 | 91.71 | 97.24 | **66.0** | 0.85 | **0.98** | 524.4 | 0.88 |
| **SST-2** | 84.27 | 61.10 | 92.93 | 99.39 | 99.51 | **94.0** | 0.89 | **0.95** | 303.1 | 0.94 |
| **HC3** | — | — | — | — | — | **98.0** | — | **0.98** | 42.0 | 0.99 |
| **M4** | — | — | — | — | — | **82.0** | — | **0.99** | 43.7 | 0.78 |

*Note: CSBP maintains notably higher semantic similarity (`cos`) than the existing state-of-the-art across evaluated datasets, while achieving maximum fragmentation over `BPE` bounds.*

---

## 🧠 Architecture Overview

### Stage 1 — Register Gate (`humanizer.py`)
Before attacking, the text is formally classified (using `all-mpnet-base-v2` bindings and internal logical regression classifiers). The contextual label governs internal parameter flows but natively preserves the semantic baseline intent.

### Stage 2 — Vulnerability Pre-Analysis (`composite_scorer.py`)
A forward pass explicitly maps token-confidences down to byte BPE levels to identify priorities:
- **Priority Words:** Top confidence ranks.
- **Watermark Words:** Directly extracted KGW green-list spans.

### Stage 3 — CSBP v3 Model-Specific Evasion Strategies (`csbp_loop.py`)
The pipeline natively exploits the disparate vocab architectures between BERT (WordPiece/cased) and RoBERTa (case-sensitive byte-level BPE).

**Core v3 Strategies:**
1. `blitz` & `zwsp_flood` (Heavily Weighted): Deep insertion of multi-layered Invisible Unicode mappings (ZWSP, ZWNJ) to completely shatter token boundaries. Maintains visual integrity while severely penalizing sequences.
2. **RoBERTa Specific** (`roberta_subword`, `random_case`, `math_unicode`, `fullwidth`): Explicit optimizations that spoof RoBERTa's isolated BPE constraints using distinct byte length mappings.
3. `bpe_break`: Exhaustive internal boundary checking for maximal length perturbations.
4. `micro_sponge`: BPE-breaking end-of-sentence invisible sponge sequences to displace document aggregation metrics.
5. `punctuation_bind`: Encompasses punctuation delimiters (`.`, `,`) to artificially separate subsequent byte contexts.

### Stage 4 — Evasion-Oriented Objective Function
Optimized scoring focuses almost exclusively on enforcing out-of-distribution inference behavior (Evasion metrics weighted heavily at 55%). Relaxed SBERT coherence thresholds (down to 0.20-0.60) ensure visual representations are favored independently of underlying byte encoding disruptions.

---

## 🛠️ Usage & Execution

Our base libraries run out-of-the-box via pip:
```bash
pip install torch sentence-transformers transformers scikit-learn pandas tqdm aiohttp pyarrow joblib nltk emoji
```

### Direct Inference
```bash
python characterlevelattacks/coreattacks/humanizer.py "Your target AI-text here." \
    --iterations 7 \
    --beam-width 7 \
    --cands 20 \
    --device mps
```

### Dataset Batches
```bash
python characterlevelattacks/coreattacks/humanizer.py \
    --hc3 path/to/hc3.parquet \
    --m4  path/to/m4.parquet \
    --sample-size 100 \
    -o results.csv
```
