# Detector Evaluation Toolkit

A framework for training, calibrating, and stress-testing six state-of-the-art AI-text detectors against teammates' humanizer attacks.

---

## Project Goal

This module is the **detector component** of a team project. The team has members working on different attack strategies (character-level noise, paraphrasing, prompt evasion, gradient-based). This toolkit evaluates how well the 6 detectors hold up against each of those attacks.

---

## The 6 Detectors

| # | Detector | Type | What it uses |
|---|---|---|---|
| 1 | `roberta_classifier` | Supervised | Fine-tuned RoBERTa-base on HC3 data |
| 2 | `detectgpt_style` | Curvature | Log-probability curvature via perturbations (GPT-2 + DistilRoBERTa) |
| 3 | `fast_detectgpt` | Token discrepancy | Z-score difference between GPT-2-medium and GPT-2 |
| 4 | `binoculars` | Likelihood ratio | Cross-entropy ratio between observer/performer models |
| 5 | `kgw_watermark` | Watermark | Green-list token z-score (KGW scheme) |
| 6 | `stats_baseline` | Statistical | Perplexity + token-rank under GPT-2 |

---

## Environment Setup

```bash
# From the detector_evaluation/ directory
pip install -r requirements.txt
```

---

## File Structure

```
detector_evaluation/
├── data/
│   └── splits/
│       ├── train.csv       ← used to train RoBERTa
│       ├── val.csv         ← used to calibrate thresholds
│       └── test.csv        ← used to validate implementation vs paper
├── detectors/
│   ├── common/
│   │   ├── config.py
│   │   ├── interfaces.py
│   │   ├── io_utils.py
│   │   └── metrics.py
│   ├── binoculars/score.py
│   ├── detectgpt/score.py, perturb.py
│   ├── fast_detectgpt/score.py
│   ├── roberta_classifier/train.py, infer.py
│   ├── stats_baseline/score.py
│   └── watermark/score.py
├── evaluation/
│   ├── run_all.py                  ← runs all 6 detectors on one file
│   ├── aggregate_results.py        ← computes all metrics from score files
│   ├── plots.py                    ← generates bar charts and ROC curves
│   ├── paired_to_long.py           ← converts teammate paired CSV to long format
│   ├── paired_analysis.py          ← score-drop and flip-rate analysis
│   ├── run_paired_pipeline.py      ← ONE command to evaluate a teammate's attack
│   ├── disagreement_ensemble.py    ← novelty: cross-paradigm disagreement detector
│   ├── calibrate_thresholds.py     ← calibrate detector thresholds using val split
│   ├── run_baseline_test.py        ← validate implementation vs paper benchmarks
│   ├── prepare_hc3.py              ← download and split HC3 dataset
│   └── validate_schema.py          ← validate input CSV schema
├── results/
│   ├── roberta_model/              ← trained model weights
│   └── thresholds.json             ← calibrated per-detector thresholds
├── teammate_pairs_template.csv     ← template to share with teammates
└── requirements.txt
```

---

## Input Format (What Teammates Must Send You)

A CSV file with these columns:

| Column | Required | Description |
|---|---|---|
| `pair_id` | Yes | e.g. `p001` |
| `original_text` | Yes | The original AI-generated text |
| `humanized_text` | Yes | The text after their attack |
| `attack_type` | Yes | e.g. `char`, `paraphrase`, `prompt`, `gradient` |
| `generator_model` | No | e.g. `gpt-3.5-turbo`, `llama-3` |

Share `teammate_pairs_template.csv` with them as an example.

**Note**: If the file is already in long format (has `id`, `variant`, `source` columns), the pipeline auto-detects this and skips the conversion step.

---

## Phase 1 — One-Time Setup (Do Once)

### Step 1a: Prepare the dataset

If you don't already have `data/splits/` populated:

```bash
python -m evaluation.prepare_hc3
```

Downloads HC3 and creates `train.csv`, `val.csv`, `test.csv` in `data/splits/`.

### Step 1b: Train the RoBERTa detector

```bash
python -m detectors.roberta_classifier.train \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --output-dir results/roberta_model \
  --epochs 3 \
  --batch-size 8
```

Saves the trained model to `results/roberta_model/`. Takes 20–60 min on CPU.

### Step 1c: Calibrate all detector thresholds

```bash
python -m evaluation.calibrate_thresholds
```

Runs all 6 detectors on `val.csv`, finds the best-F1 threshold for each, and saves to `results/thresholds.json`. From this point on, every evaluation automatically uses these calibrated thresholds.

### Optional: Generate KGW-Watermarked HC3 Variants

KGW cannot learn anything useful from ordinary HC3 ChatGPT text because no watermark was embedded during generation. To create a KGW-positive dataset, generate local GPT-2 answers with the KGW sampler:

```bash
python -m evaluation.generate_watermarked_hc3 \
  --n-prompts 150 \
  --model-name gpt2 \
  --max-new-tokens 80 \
  --device cuda
```

This writes:

```text
data/raw/hc3_watermarked_kgw.csv
data/splits/kgw_train.csv
data/splits/kgw_val.csv
data/splits/kgw_test.csv
data/processed/hc3_watermarked_pairs.csv
```

Use `--gamma`, `--delta`, and `--hash-key` consistently between generation and detection. The default KGW detector settings match the generator defaults.

Calibrate KGW on the watermarked validation split:

```bash
python -m detectors.watermark.score \
  --input data/splits/kgw_val.csv \
  --output results/kgw_watermarked_val_scores.csv \
  --tokenizer-name gpt2 \
  --gamma 0.5 \
  --hash-key 15485863
```

The calibrated KGW threshold currently used in `results/thresholds.json` is `0.9375`. KGW is only meaningful for text generated with the KGW sampler; on ordinary non-watermarked ChatGPT/HC3 text, near-random KGW performance is expected.

---

## Phase 2 — Validate Your Implementation (Do Once)

Run all detectors on the held-out `test.csv` and compare your numbers against expected paper benchmarks:

```bash
python -m evaluation.run_baseline_test
```

This prints a comparison table:

```
============================================================
BASELINE TEST RESULTS
============================================================
Detector                   AUROC  Paper     Acc  Paper
------------------------------------------------------------
roberta_classifier         0.961   0.97   0.920   0.93
stats_baseline             0.743   0.75   0.710   0.72
detectgpt_style            0.698   0.70   0.671   0.68
fast_detectgpt             0.731   0.72   0.704   0.70
binoculars                 0.841   0.85   0.802   0.80
kgw_watermark              0.512   0.50   0.510   0.50
============================================================

Full results saved → results/baseline_test_results.csv

NOTE: KGW Watermark is not trained on this data — near-random AUROC is expected.
```

**What this tells you:**
- Your numbers should be within ~10% of the paper benchmarks
- KGW watermark will always be near-random (0.50) because the test data was not watermarked — this is expected
- Full results saved to `results/baseline_test_results.csv`

---

## Phase 3 — Evaluate a Teammate's Attack (Repeat for Each Teammate)

When a teammate gives you their paired CSV file, run one command:

```bash
python -m evaluation.run_paired_pipeline \
  --pairs-file teammate_pairs_charlevel.csv \
  --output-dir results/attack_eval_charlevel \
  --model-dir results/roberta_model \
  --device cpu \
  --thresholds-file results/thresholds.json
```

If you receive multiple files from the same teammate:

```bash
python -m evaluation.run_paired_pipeline \
  --pairs-dir path/to/folder_with_csvs/ \
  --output-dir results/attack_eval_charlevel \
  --model-dir results/roberta_model \
  --device cpu \
  --thresholds-file results/thresholds.json
```

### What happens internally (6 automatic steps):

1. **Convert** paired CSV → long format (auto-skipped if already in long format)
2. **Score** all 6 detectors on every row
3. **Aggregate** metrics + evasion rates + transferability matrix
4. **Plot** bar charts, ROC curves, confusion matrices
5. **Paired analysis** — score-drop and flip rates per pair
6. **Disagreement ensemble** — your novelty component (runs if both human + AI labels are present)

### Output files:

```
results/attack_eval_charlevel/
├── scores/
│   ├── roberta_classifier_scores.csv
│   ├── detectgpt_style_scores.csv
│   ├── fast_detectgpt_scores.csv
│   ├── binoculars_scores.csv
│   ├── kgw_watermark_scores.csv
│   └── stats_baseline_scores.csv
├── metrics.csv              ← accuracy, F1, attack success rate per detector
├── evasion_summary.csv      ← % of AI samples evading each detector + all detectors at once
├── paired_summary.csv       ← score-drop and flip rate per pair
├── figures/                 ← bar charts, ROC curves, confusion matrices
└── ensemble/                ← disagreement ensemble results (novelty)
```

### Understanding the output:

**`metrics.csv`**

| Column | Meaning |
|---|---|
| `accuracy` | Overall correct predictions |
| `f1_ai` | F1 score on the AI class |
| `attack_success_rate` | % of AI samples the attack fooled the detector on |
| `threshold_used` | The calibrated decision threshold |

**`evasion_summary.csv`**

| Column | Meaning |
|---|---|
| `evasion_rate_<detector>` | % of humanized AI samples that fooled this specific detector |
| `cross_paradigm_evasion_rate` | % that fooled ALL 6 detectors simultaneously — the hardest to achieve |

**`paired_summary.csv`**

| Column | Meaning |
|---|---|
| `mean_score_drop` | Average drop in AI score from original → humanized (higher = more effective attack) |
| `flip_rate_all_pairs` | % where predicted label changed from `ai` → `human` |

---

## Schema Validation

Before scoring, you can validate any input CSV:

```bash
python -m evaluation.validate_schema --inputs teammate_pairs.csv
```

---

## Novelty: Disagreement-Aware Ensemble

The ensemble component (`disagreement_ensemble.py`) is the novel contribution of this project. It trains a meta-classifier that uses **inter-detector disagreement** as an adversarial signal.

The key insight: when an attack successfully fools some detectors but not others, this creates high disagreement variance across detectors — which itself becomes a signal for adversarial manipulation. Even if individual detectors are fooled, the ensemble can still flag the text.

To run it standalone:

```bash
python -m evaluation.disagreement_ensemble \
  --scores-dir results/attack_eval_charlevel/scores \
  --output-dir results/attack_eval_charlevel/ensemble \
  --allow-missing-detectors
```

---

## Troubleshooting

**`UndefinedMetricWarning: No negative samples in y_true`**
Your input file only has AI samples (no human baseline). This is expected when evaluating attack-only files. AUROC will show as blank — use `attack_success_rate` and `mean_score_drop` from `paired_summary.csv` instead.

**Thresholds show 0.0 for some detectors**
Run `python -m evaluation.calibrate_thresholds` first. Without calibration, detectors default to 0.0 which makes all predictions trivially "ai".

**Ensemble step skipped**
The ensemble requires both `human` and `ai` source labels in the score files. If your teammate's file only has AI text, pass `--skip-ensemble` or ignore the skip message.

**RoBERTa model not found**
Run Phase 1b first: `python -m detectors.roberta_classifier.train ...`
