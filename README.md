# AI Text Evasion & Detection — NLP Research Project

> **Advanced NLP Course Project — Plaksha University**
>
> A full-stack research pipeline for **adversarial AI-text humanization** and **multi-paradigm detection**. Four independent attack modules generate evasion outputs; a shared detector evaluation harness stress-tests all of them simultaneously across six state-of-the-art detection paradigms.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team &amp; Roles](#team--roles)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
  - [1. Character-Level Attacks (CSBP)](#1-character-level-attacks-csbp)
  - [2. Gradient-Based Attacks](#2-gradient-based-attacks)
  - [3. Prompt-Level Evasion](#3-prompt-level-evasion)
  - [4. Post-Generation Rewriting (HMGC-V2)](#4-post-generation-rewriting-hmgc-v2)
  - [5. Detector Evaluation Toolkit](#5-detector-evaluation-toolkit)
- [Shared Data Contract](#shared-data-contract)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Key Results](#key-results)
- [Setup &amp; Installation](#setup--installation)
- [Datasets](#datasets)
- [References](#references)

---

## Project Overview

This project investigates adversarial attacks against AI-generated text detectors — a rapidly growing research area as LLM-generated content becomes ubiquitous. The central question is:

> *Can an adversary reliably make AI-generated text evade detection across multiple, fundamentally different detector paradigms?*

We approach this through four distinct evasion strategies, each attacking different layers of the text signal, unified under a shared evaluation harness:

| Attack Layer              | Strategy                                                               | Novelty                                           |
| ------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------- |
| **Byte/Character**  | BPE tokenizer disruption via homoglyphs, ZWSP, diacritics              | Composite-Scored Beam Perturbation (CSBP v3)      |
| **Gradient/RL**     | Hybrid white-box + black-box evader (LoRA + GRPO)                      | Tunable λ sweep; cross-paradigm robustness       |
| **Prompt**          | Persona-augmented prompt optimizer with KL-divergence style constraint | Stylometric persona fingerprinting                |
| **Post-generation** | Span-level rewriting with modern cross-encoder semantic gating         | HMGC-V2; USE → cross-encoder upgrade             |
| **Detection**       | 6-detector unified harness + disagreement-aware ensemble               | Inter-paradigm disagreement as adversarial signal |

---

## Team & Roles

| Member                   | Role                                                           | Module                     |
| ------------------------ | -------------------------------------------------------------- | -------------------------- |
| **Yatharth Nehra** | Character-level adversarial attacker (SilverSpeak / CSBP)      | `characterlevelattacks/` |
| **Udaiveer Singh** | Gradient + RL hybrid evader (GradEscape)                       | `gradientBasedAttacks/`  |
| **Aryan Chopra**   | Prompt-level evasion & persona optimization (SICO)             | `prompt_evasion/`        |
| **Raghav Sarna**   | Post-generation span-level rewriting (stylometric paraphraser) | `post_generation/`       |
| **Akshat Gupta**   | SOTA detector implementation & evaluation infrastructure       | `detector_evaluation/`   |

---

## Repository Structure

```text
nlp_proj/
├── characterlevelattacks/      ← Character-level CSBP attack framework
│   ├── coreattacks/            ← Attack loop, strategies, scorer
│   ├── advanced_metrics.py
│   ├── humanvsai/              ← Human vs AI baseline data
│   ├── emojibased/             ← Emoji distribution data
│   ├── stylometric/            ← Stylometric feature references
│   └── README.md
│
├── gradientBasedAttacks/       ← Gradient + RL hybrid evader
│   ├── evader/                 ← LoRA adapter, pseudo-embeddings, GRPO
│   ├── evaluation/             ← ASR, BERTScore, λ-sweep scripts
│   ├── notebooks/              ← Evaluation notebooks (Eval 1–3)
│   ├── scripts/                ← HC3 download + split preparation
│   └── README.md
│
├── prompt_evasion/             ← Persona-augmented prompt optimizer
│   ├── Prompt_Evasionn.ipynb   ← Full implementation notebook
│   ├── NLP_1.ipynb             ← Supplementary experiments
│   ├── results_ours.csv        ← Method results
│   ├── results_sico.csv        ← SICO baseline results
│   └── Readme
│
├── post_generation/            ← HMGC-V2 span-level rewriting
│   ├── app/                    ← Evader & detector training apps
│   ├── core/                   ← Runtime & logging utilities
│   ├── attack/                 ← Attack recipes and baselines
│   ├── main.py                 ← Unified CLI entrypoint
│   └── README.md
│
├── detector_evaluation/        ← Shared detector harness (6 detectors)
│   ├── detectors/              ← RoBERTa, DetectGPT, FastDetectGPT, Binoculars, KGW, Stats
│   ├── evaluation/             ← Scoring, aggregation, ensemble, plots
│   ├── data/splits/            ← train / val / test CSVs (HC3)
│   ├── results/                ← Trained model weights, thresholds.json
│   ├── requirements.txt
│   └── README.md
│
├── inputs/                     ← Raw input data shared across modules
│   ├── gradient.csv
│   ├── promptlevel_envasion.csv
│   └── promptlevel_envasion_fixed.csv
│
├── .gitignore
└── README.md
```

---

## Modules

### 1. Character-Level Attacks (CSBP)

**Location:** `characterlevelattacks/`

The **Composite-Scored Beam Perturbation (CSBP v3)** framework evades AI text detectors by restructuring tokens at the **byte level** — exploiting BPE tokenizer vulnerabilities rather than semantically paraphrasing text.

#### How It Works

```
Input Text
    │
    ▼
[Stage 1] Register Gate
  → Formal / Informal classification (MPNet + LogReg)
  → Sets emoji probability gate
    │
    ▼
[Stage 2] Vulnerability Pre-Analysis
  → GPT-2 forward pass → identifies high-confidence tokens & KGW green-list words
    │
    ▼
[Stage 3] CSBP Beam Search
  → 15 perturbation strategies (homoglyphs, ZWSP, diacritics, math-unicode…)
  → Beam width=8, patience-based early stopping
    │
    ▼
[Stage 4] Composite Scorer
  → S = w_clf·clf + w_ev·ev + w_bpe·bpe + w_wm·wm + w_read·read + w_sim·cos
  → Coherence gate: multiplies by (0.3 + 0.7 × cos_gate)
    │
    ▼
Attacked Output (CSV)
```

#### Key Results

| Dataset | Arch    | CSBP ASR       | Charmer ASR | CSBP cos-sim   |
| ------- | ------- | -------------- | ----------- | -------------- |
| HC3     | BERT    | **98%**  | —          | **0.98** |
| AG-News | BERT    | **98%**  | 98.51%      | **0.97** |
| QNLI    | RoBERTa | **100%** | 97.86%      | **0.97** |
| M4      | RoBERTa | **82%**  | —          | **0.99** |

> Commercial platforms (ZeroGPT, Originality.ai, QuillBot) return **0% AI-generated** on CSBP outputs.

**Quick Start:**

```bash
pip install torch sentence-transformers transformers scikit-learn pandas tqdm joblib nltk emoji

python characterlevelattacks/coreattacks/humanizer.py \
    "Your AI-generated text here." \
    --target-model textattack/bert-base-uncased-ag-news \
    --iterations 7 --beam-width 7 --cands 20
```

---

### 2. Gradient-Based Attacks

**Location:** `gradientBasedAttacks/`
**Author:** Udaiveer Singh (U20230017)

A **hybrid gradient + reinforcement learning** evader combining white-box and black-box signals under a tunable composite loss:

```
L = λ · L_grad  +  (1 − λ) · L_rl  +  α · L_sem
```

| Term       | Type               | Role                                                 |
| ---------- | ------------------ | ---------------------------------------------------- |
| `L_grad` | White-box          | Backprop from surrogate RoBERTa through LoRA adapter |
| `L_rl`   | Black-box (GRPO)   | Reward = 1 − detector_confidence                    |
| `L_sem`  | Quality constraint | BERTScore cosine sim ≥ 0.92                         |

A **λ sweep** (λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}) ablates the gradient-RL trade-off. **λ = 0.75 yields best performance** across all detector paradigms.

**Quick Start:**

```bash
cd gradientBasedAttacks
pip install -r requirements.txt

python scripts/download_hc3.py
python scripts/prepare_splits.py

# Train at λ=0.75 (best config)
python evader/lora_adapter/model.py \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --lambda 0.75 --alpha 0.3 \
  --epochs 3 --batch-size 8 \
  --output results/evader_outputs/lambda_0.75

# Convert for detector pipeline
python evaluation/conversion_script.py
```

---

### 3. Prompt-Level Evasion

**Location:** `prompt_evasion/`

A **Persona-Augmented Prompt Optimizer** that jointly optimizes three objectives:

```
L = λ·L_adv  +  β·L_sem  +  γ·L_persona
```

| Loss          | Goal                                             |
| ------------- | ------------------------------------------------ |
| `L_adv`     | Minimize AI-probability from surrogate RoBERTa   |
| `L_sem`     | Preserve meaning via BERTScore                   |
| `L_persona` | Align style to a human persona via KL-Divergence |

The system **extracts stylometric fingerprints** (contraction rates, hedging density, sentence variance) from real human writing and uses KL-divergence as a novel constraint to enforce persona alignment — going beyond simple perplexity manipulation.

#### 6-Metric Evaluation Protocol

| Metric        | Goal       | Description                         |
| ------------- | ---------- | ----------------------------------- |
| Evasion Rate  | ↑ High    | % samples bypassing detector        |
| Avg AI Score  | ↓ Low     | Raw AI-probability from detector    |
| BERTScore F1  | ↑ High    | Semantic similarity to original     |
| Perplexity    | Balanced   | Fluency via GPT-2                   |
| FK Grade      | Human-like | Flesch-Kincaid readability          |
| Persona Align | ↑ High    | exp(−KL) relative to human profile |

**Quick Start:**

```bash
# Open and run cells sequentially:
jupyter notebook prompt_evasion/Prompt_Evasionn.ipynb
```

Results are exported automatically to `results_ours.csv`, `results_sico.csv`, `lambda_sweep.csv`, and `evaluation_summary.csv`.

---

### 4. Post-Generation Rewriting (HMGC-V2)

**Location:** `post_generation/`

An enhanced version of the **HMGC (Humanizing Machine-Generated Content)** framework. Key upgrades over the original:

| Original HMGC                                  | HMGC-V2 (This Work)                                  |
| ---------------------------------------------- | ---------------------------------------------------- |
| Word-level synonym swapping                    | **Span-level syntactic rewriting**             |
| Universal Sentence Encoder (USE) semantic gate | **Cross-encoder / LLM-as-Judge** semantic gate |
| Greedy sequential search                       | Multi-span candidate exploration                     |

**Architecture:**

- `app/train_evader_app.py` — main evader training
- `app/train_detector_app.py` — detector training
- `attack/` — attack recipes and baselines
- `main.py` — unified CLI

**Quick Start:**

```bash
pip install -r post_generation/requirements.txt

# Train evader
python post_generation/main.py train-evader --help

# Run attack
python post_generation/main.py attack --help

# Smoke test
python post_generation/main.py smoke-phase2 --help
```

---

### 5. Detector Evaluation Toolkit

**Location:** `detector_evaluation/`
**Author:** Akshat Gupta (U20230093)

A **unified 6-detector harness** that benchmarks all attack modules simultaneously, with a novel **Disagreement-Aware Ensemble** as the core research contribution.

#### The 6 Detectors

| # | Detector               | Paradigm              | Backbone                                 |
| - | ---------------------- | --------------------- | ---------------------------------------- |
| 1 | `roberta_classifier` | Supervised neural     | Fine-tuned RoBERTa-base on HC3           |
| 2 | `detectgpt_style`    | Statistical curvature | GPT-2 perturbations + DistilRoBERTa      |
| 3 | `fast_detectgpt`     | Token discrepancy     | GPT-2-medium vs GPT-2 z-score            |
| 4 | `binoculars`         | Likelihood ratio      | Cross-entropy ratio (observer/performer) |
| 5 | `kgw_watermark`      | Proactive watermark   | Green-list token z-score (KGW)           |
| 6 | `stats_baseline`     | Statistical           | Perplexity + token-rank (GPT-2)          |

#### Novelty: Disagreement-Aware Ensemble

The key insight: when an attack selectively fools some detectors but not others, it creates **anomalously high inter-detector disagreement variance** — which itself becomes an adversarial signal.

```
D = Var(s₁, s₂, s₃, s₄)   [disagreement score across paradigms]

Feature vector → logistic regression meta-classifier:
  [s_roberta, s_detectgpt, s_fast_detectgpt, s_binoculars, D]
```

- **Human text:** low scores + low D
- **Unmodified AI text:** high scores + low D
- **Adversarially humanized text:** mixed scores + **high D** ← adversarial signature

#### Baseline Performance (HC3, Calibrated)

| Detector           | AUROC | Paper                                    |
| ------------------ | ----- | ---------------------------------------- |
| roberta_classifier | 0.961 | 0.97                                     |
| stats_baseline     | 0.743 | 0.75                                     |
| fast_detectgpt     | 0.731 | 0.72                                     |
| binoculars         | 0.841 | 0.85                                     |
| detectgpt_style    | 0.698 | 0.70                                     |
| kgw_watermark      | 0.512 | 0.50*(expected — non-watermarked data)* |

#### Phase 1 — One-Time Setup

```bash
cd detector_evaluation
pip install -r requirements.txt

# 1. Download HC3 and create splits
python -m evaluation.prepare_hc3

# 2. Train RoBERTa detector
python -m detectors.roberta_classifier.train \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --output-dir results/roberta_model \
  --epochs 3 --batch-size 8

# 3. Calibrate all thresholds
python -m evaluation.calibrate_thresholds
```

#### Phase 2 — Validate Implementation

```bash
python -m evaluation.run_baseline_test
```

#### Phase 3 — Evaluate an Attack Module

```bash
python -m evaluation.run_paired_pipeline \
  --pairs-file ../charlevel_long.csv \
  --output-dir results/attack_eval_charlevel \
  --model-dir results/roberta_model \
  --thresholds-file results/thresholds.json
```

**Output:**

```
results/attack_eval_charlevel/
├── scores/          ← per-detector score files
├── metrics.csv      ← accuracy, F1, ASR per detector
├── evasion_summary.csv
├── paired_summary.csv  ← score-drop and flip rates
├── figures/         ← ROC curves, bar charts
└── ensemble/        ← disagreement ensemble results
```

---

## Shared Data Contract

All attack modules output a **paired CSV** with this schema, consumed by the detector pipeline:

| Column              | Required | Description                                         |
| ------------------- | -------- | --------------------------------------------------- |
| `pair_id`         | ✅       | e.g.`p00001`                                      |
| `original_text`   | ✅       | Original AI-generated text                          |
| `humanized_text`  | ✅       | Post-attack text                                    |
| `attack_type`     | ✅       | `char` / `gradient` / `prompt` / `post_gen` |
| `generator_model` | Optional | e.g.`gpt-3.5-turbo`, `llama-3`                  |

The shared intermediate format for the detector pipeline is **long format** (columns: `id`, `text`, `variant`, `source`). The `paired_to_long.py` script handles conversion automatically.

---

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────┐
│                 HC3 Dataset (shared)                │
└────────────┬────────────┬────────────┬──────────────┘
             │            │            │            │
             ▼            ▼            ▼            ▼
    ┌──────────────┐ ┌──────────┐ ┌────────────┐ ┌───────────────┐
    │  Character   │ │ Gradient │ │   Prompt   │ │ Post-Gen      │
    │  Level CSBP  │ │ LoRA+GRPO│ │  Optimizer │ │ HMGC-V2       │
    └──────┬───────┘ └────┬─────┘ └─────┬──────┘ └──────┬────────┘
           │              │             │                │
           └──────────────┴─────────────┴────────────────┘
                                  │
                      Paired CSV (shared schema)
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │   Detector Evaluation Harness │
                   │  (6 detectors + ensemble)     │
                   └──────────────┬───────────────┘
                                  │
                      metrics.csv + evasion_summary.csv
                      + figures/ + ensemble/
```

---

## Key Results

### Attack Success Rates vs All 6 Detectors

| Attack Module             | Best Detector Evaded (ASR)   | Cross-Paradigm Evasion          |
| ------------------------- | ---------------------------- | ------------------------------- |
| CSBP (Character-level)    | 100% (QNLI/RoBERTa)          | Strong — targets BPE OOD       |
| Gradient Hybrid (λ=0.75) | 100% (Statistical)           | 90%+ across paradigms           |
| Prompt Persona Optimizer  | High (RoBERTa-targeted)      | Moderate — persona-constrained |
| Post-Gen HMGC-V2          | Competitive vs original HMGC | Improved fluency scores         |

### Gradient λ Sweep Summary

| λ                  | Behavior                                 |
| ------------------- | ---------------------------------------- |
| 0.0 (pure RL)       | Strong fluency-driven transformations    |
| 0.25–0.5           | Balanced evasion + quality               |
| **0.75**      | **Best overall**                   |
| 1.0 (pure gradient) | Strong adversarial shifts, lower fluency |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU supported with reduced speed)

### Per-Module Installation

Each module has its own `requirements.txt`. Install separately:

```bash
# Character-level attacks
pip install torch sentence-transformers transformers scikit-learn \
            pandas tqdm aiohttp pyarrow joblib nltk emoji

# Gradient-based attacks
cd gradientBasedAttacks && pip install -r requirements.txt

# Detector evaluation
cd detector_evaluation && pip install -r requirements.txt

# Post-generation
cd post_generation && pip install -r requirements.txt
```

### Environment (Optional — shared venv)

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

---

## Datasets

All modules use the **HC3 (Human-ChatGPT Comparison Corpus)** as the primary dataset.

| Dataset            | Description                       | Used By                     |
| ------------------ | --------------------------------- | --------------------------- |
| HC3                | ~25K paired human vs. ChatGPT QA  | All modules                 |
| M4                 | Multi-generator AI-text benchmark | CSBP evaluation             |
| AG-News            | News classification               | CSBP (BERT/RoBERTa targets) |
| QNLI / RTE / SST-2 | GLUE benchmarks                   | CSBP evaluation             |

Download HC3 automatically:

```bash
# Gradient module
python gradientBasedAttacks/scripts/download_hc3.py

# Detector module
python -m evaluation.prepare_hc3  # (from detector_evaluation/)
```

---

## References

| Paper                                                                      | Relevance                      |
| -------------------------------------------------------------------------- | ------------------------------ |
| Mitchell et al. (2023).*DetectGPT*. ICML.                                | Statistical curvature detector |
| Bao et al. (2023).*Fast-DetectGPT*. arXiv:2310.05130                     | Fast statistical detector      |
| Hans et al. (2024).*Binoculars*. ICML. arXiv:2401.12070                  | Likelihood ratio detector      |
| Kirchenbauer et al. (2023).*KGW Watermark*. ICML. arXiv:2301.10226       | Watermark detector             |
| Meng et al. (2025).*GradEscape*. USENIX Security. arXiv:2506.08188       | Gradient attack baseline       |
| David & Gervais (2025).*AuthorMist*. arXiv:2503.08716                    | RL attack baseline             |
| StealthRL (2026). arXiv:2602.08934                                         | RL evasion reference           |
| Solaiman et al. (2019).*RoBERTa detector*. OpenAI. arXiv:1908.09203      | Supervised classifier          |
| Crothers et al. (2023).*Machine Generated Text Survey*. arXiv:2210.07321 | Survey                         |

---

> For module-specific details, see the `README.md` inside each subdirectory.
