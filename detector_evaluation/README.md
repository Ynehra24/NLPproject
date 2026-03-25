# Detector Evaluation Toolkit

A comprehensive framework for training, evaluating, and stress-testing state-of-the-art AI-text detectors against adversarial attacks and human-generated samples.

## Overview

This toolkit implements six complementary detectors spanning different detection paradigms:

- **Supervised learning**: Fine-tuned RoBERTa classifier
- **Curvature-based**: DetectGPT perturbation analysis
- **Token-discrepancy**: Fast-DetectGPT scoring differential
- **Likelihood-ratio**: Binoculars cross-entropy observer/performer
- **Watermark-aware**: KGW green-list token detection
- **Statistical**: Baseline perplexity and token-rank scoring

All detectors operate on the same input contract, ensuring fair comparison and robustness evaluation against team-driven attacks (adversarial prompts, paraphrasing, gradient-based evasion, character-level noise).

## What This Repository Section Contains

- **Unified input contract**: All detectors accept the same CSV/JSONL schema (id, text, source, attack_type, attack_owner, generator_model)
- **Six detector implementations**: Each with standardized score output (0–1 or z-scores, normalized to sigmoid probability)
- **Unified runner** (`evaluation/run_all.py`): Orchestrates all detectors from a single command
- **Aggregation pipeline**: Groups metrics by attack type and computes robustness deltas versus clean condition
- **Visualization tools**: Generates AUROC and attack success rate heatmaps
- **Sample data**: Pre-split train/val/test sets for quick smoke testing (8 rows each, minimal size for CI/CD)

## File Structure

```
detector_evaluation/ (Root)
|-- .gitignore
|-- IMPLEMENTATION_PLAN.txt
|-- README.md
|-- explanation.md
|-- requirements.txt
|
|-- data/
|   |-- processed/          (placeholder for preprocessed datasets)
|   |-- raw/                (placeholder for raw input data)
|   \-- splits/
|       |-- test.csv        (evaluation set, 8 samples)
|       |-- train.csv       (training set, 8 samples)
|       \-- val.csv         (validation set, 8 samples)
|
|-- detectors/              (Detector implementations)
|   |-- __init__.py
|   |-- common/             (Shared utilities)
|   |   |-- __init__.py
|   |   |-- config.py       (configuration defaults)
|   |   |-- interfaces.py   (detector base class)
|   |   |-- io_utils.py     (CSV/JSONL loading)
|   |   \-- metrics.py      (AUROC, AUPRC, F1, TPR@FPR computation)
|   |
|   |-- binoculars/         (Likelihood-ratio detector)
|   |   |-- __init__.py
|   |   \-- score.py        (inference via dual-model ratios)
|   |
|   |-- detectgpt/          (Curvature-based perturbation detector)
|   |   |-- __init__.py
|   |   |-- perturb.py      (T5-based perturbation)
|   |   \-- score.py        (curvature scoring)
|   |
|   |-- fast_detectgpt/     (Token-discrepancy detector)
|   |   |-- __init__.py
|   |   \-- score.py        (fast z-score scoring)
|   |
|   |-- roberta_classifier/ (Fine-tuned supervised detector)
|   |   |-- __init__.py
|   |   |-- infer.py        (prediction on new data)
|   |   |-- train.py        (training script)
|   |   \-- model_card.txt  (model description)
|   |
|   |-- stats_baseline/     (Perplexity + token-rank baseline)
|   |   |-- __init__.py
|   |   \-- score.py        (statistical heuristic scoring)
|   |
|   \-- watermark/          (KGW green-list watermark detector)
|       \-- score.py        (watermark z-score detection)
|
|-- docs/                   (Research papers & course materials)
|   |-- 2210.07321v4.pdf    (DetectGPT paper)
|   |-- 2305.10847v6.pdf
|   |-- 2404.01907v1.pdf
|   |-- 2406.11239v3.pdf
|   |-- 2506.07001v2.pdf
|   |-- 2506.08188v2.pdf
|   \-- nlp_course_report.pdf
|
|-- evaluation/             (Evaluation pipeline & orchestration)
|   |-- adaptive_retrain_stress.py   (iterative hard-example mining + retraining)
|   |-- aggregate_results.py         (metrics computation by attack type)
|   |-- plots.py                     (heatmap visualization)
|   |-- run_all.py                   (unified detector orchestration)
|   |-- transferability.py           (cross-detector vulnerability analysis)
|   |-- validate_schema.py           (input validation)
|   \-- watermark_robustness.py      (watermark delta analysis under attack)
|
\-- results/                (Generated outputs)
    |-- attack_eval_tables/
    |   \-- metrics_summary.csv
    |
    |-- detector_scores/    (Detector predictions)
    |   |-- binoculars_scores.csv
    |   |-- detectgpt_style_scores.csv
    |   |-- fast_detectgpt_scores.csv
    |   |-- kgw_watermark_scores.csv
    |   |-- roberta_classifier_scores.csv
    |   \-- stats_baseline_scores.csv
    |
    |-- figures/            (Visualizations)
    |   |-- asr_by_detector_and_attack.png
    |   \-- auroc_by_detector_and_attack.png
    |
    \-- roberta_model/      (Trained RoBERTa checkpoint)
        |-- checkpoint-2/   (intermediate checkpoint)
        |   |-- config.json
        |   |-- model.safetensors
        |   |-- optimizer.pt
        |   |-- tokenizer.json
        |   \-- trainer_state.json
        |-- config.json
        |-- metrics.json
        |-- model.safetensors
        |-- tokenizer.json
        \-- training_args.bin
```

## Implemented Detectors

### 1) RoBERTa Classifier (Supervised)

**Type**: Fine-tuned transformer classifier
**Architecture**: RoBERTa-base (125M parameters) trained on binary classification (human vs. AI)
**Input**: Raw text → tokenized → [CLS] embeddings → MLP head → logits
**Output**: Probability 0–1 (human=0, AI=1)
**Training**: Cross-entropy loss with class weighting, validation-based early stopping
**Best for**: Clean evaluation; typically highest AUROC on in-distribution test sets
**Weakness**: Overfits to training domain; may fail on paraphrased or adversarial variants
**Runtime**: ~50ms per 512-token sample (GPU) or ~200ms (CPU)
**Reference**: Fine-tuned on curated human vs. GPT-3.5/GPT-4 samples

### 2) DetectGPT-Style (Curvature-based)

**Type**: Perturbation-based zero-shot detector**Algorithm**: Measure log-probability curvature by mask-filling perturbations

1. Tokenize input text
2. Sample 8 random tokens (default) to mask
3. Fill masks using T5 paraphrase model
4. Compute base log-prob (original text) minus mean(perturbed log-probs)
5. Normalize via sigmoid to [0, 1]
   **Output**: Probability 0–1 (human=0, AI=1)
   **Intuition**: AI text has less curvature (smoother likelihood landscape) than human text
   **Best for**: Cross-domain robustness; no fine-tuning required
   **Weakness**: Computationally expensive (8 forward passes); sensitive to masking strategy
   **Runtime**: ~5–10 seconds per sample (GPU) due to perturbation overhead
   **Params**: `--detectgpt-perturb` (default 8; reduce to 2–4 for speed)
   **Reference**: Inspired by Mitchell et al. (2023) _DetectGPT: Zero-Shot Detector via Perturbation_

### 3) Fast-DetectGPT (Token Discrepancy)

**Type**: Efficient z-score based on token prediction discrepancy**Algorithm**: Token-level log-prob difference between scoring and reference models

1. Tokenize input
2. Get log-probabilities for each token under scoring model (GPT-2-medium)
3. Get log-probabilities for same tokens under reference model (GPT-2-base)
4. Compute z-score: (mean(scoring_LP) - mean(reference_LP)) / std(differences)
5. Normalize via sigmoid to [0, 1]
   **Output**: Probability 0–1
   **Intuition**: AI text has higher token-level agreement with larger models
   **Best for**: Speed; under 100ms per sample
   **Weakness**: Depends on model choice; less robust to very short texts
   **Runtime**: ~50–100ms per sample (GPU/CPU)
   **Params**: `--fast-detectgpt-scoring-model`, `--fast-detectgpt-ref-model`
   **Reference**: Inspired by Bao et al. (2024) _Fast-DetectGPT: Efficient Zero-Shot Detector_

### 4) Binoculars (Likelihood Ratio)

**Type**: Likelihood-ratio based on dual model observation**Algorithm**: Cross-entropy ratio between observer and performer models

1. Tokenize input
2. Compute cross-entropy (CE) under observer model (GPT-2-medium)
3. Compute cross-entropy under performer model (GPT-2-base)
4. Ratio = CE(observer) / CE(performer)
5. Invert, center around median, scale, then sigmoid normalize
   **Output**: Probability 0–1
   **Intuition**: AI-generated text shows characteristic likelihood patterns when scored by different models
   **Best for**: Fast evaluation with reasonable AUROC
   **Weakness**: Fixed model choice; may not generalize to GPT-4 or other families
   **Runtime**: ~100–150ms per sample (GPU/CPU)
   **Params**: `--binoculars-observer-model`, `--binoculars-performer-model`
   **Reference**: Inspired by Sadasivan et al. (2024) _Binoculars: Detecting Deepfakes with Forged Keypoints_

### 5) KGW Watermark Detector (Seeded Green-List)

**Type**: Watermark-aware detection via green-token z-score**Algorithm**: Detects seeded vocabulary partitioning (green/red lists)

1. Tokenize input
2. Generate green/red partition via SHA256(hash_key + prev_token_id)
3. Count green tokens in output
4. Compute z-score: (green_count - gamma*T) / sqrt(T*gamma\*(1-gamma))
5. Normalize via sigmoid to [0, 1]
   **Output**: Probability 0–1; z-scores > 4 typically indicate watermarked text
   **Intuition**: Watermarked text statistically shows more green-list tokens than random chance
   **Best for**: Detecting texts watermarked with Kirchenbauer et al. (KGW) or similar schemes
   **Weakness**: Only detects known watermarking schemes; ineffective on non-watermarked AI text
   **Runtime**: ~10–20ms per sample (CPU-bound)
   **Params**: `--watermark-hash-key`, `--watermark-gamma` (default 0.5)
   **Can generate**: Watermarked text via WatermarkGenerator helper class
   **Reference**: Inspired by Kirchenbauer et al. (2023) _A Watermark for Large Language Models_

### 6) Statistical Baseline (Perplexity & Token Rank)

**Type**: Non-neural statistic-based heuristic**Algorithm**: Combines perplexity and token-rank features

1. Tokenize input
2. Compute perplexity under GPT-2 language model
3. Compute average rank of observed tokens in model's top-k most likely tokens
4. Combine via weighted sum: 0.6*norm_perplexity + 0.4*norm_rank
5. Normalize via sigmoid to [0, 1]
   **Output**: Probability 0–1
   **Intuition**: AI text often has lower perplexity and tokens appear earlier in model's preference ranking
   **Best for**: Fast, lightweight baseline; no fine-tuning; interpretable signals
   **Weakness**: Weak single signal; easily fooled by paraphrasing or adversarial edits
   **Runtime**: ~20–30ms per sample (CPU-optimized)
   **Params**: None (fixed GPT-2 model)
   **Reference**: Classical baseline inspired by early detector literature

## Environment Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Input Data Contract

### Supported Formats

- **CSV** (comma-separated, UTF-8 encoding)
- **JSONL** (one JSON object per line)

### Required Columns

- **id** (string): Unique identifier for each sample (e.g., "sample_001")
- **text** (string): Raw text content to be scored; no preprocessing applied by detectors

### Optional but Recommended Columns

- **source** (string): "human" or "ai" for ground truth label (used in evaluation metrics)
- **attack_type** (string): Category of attack applied; standard values:
  - "none" (clean human or AI text, no attack)
  - "prompt" (adversarial prompt injection at generation time)
  - "paraphrase" (post-hoc rewording/synonym replacement)
  - "gradient" (gradient-based perturbation; e.g., text-attack library)
  - "char" (character-level noise or typos)
  - _Custom values allowed_; used for grouping in metrics aggregation
- **attack_owner** (string): Identifier of who/what applied the attack (optional team split identifier)
- **generator_model** (string): Which model generated AI text (e.g., "gpt3.5-turbo", "gpt4", "llama2")

### Schema Validation

Run before scoring:

```bash
python -m evaluation.validate_schema --input data/splits/test.csv
```

Checks:

- Required columns presence
- No null values in required fields
- Text field is non-empty string
- Source values are "human" or "ai" (if present)
- Attack_type values are recognized categories (if present)

### CSV Example

```csv
id,text,source,attack_type,attack_owner,generator_model
sample_001,The quick brown fox jumps over the lazy dog.,human,none,,
sample_002,Machine learning enables computers to learn from data without explicit instructions.,ai,none,,gpt3.5-turbo
sample_003,ML enables computational systems to adapt without pre-programmed algorithms.,ai,paraphrase,contributor_b,gpt3.5-turbo
sample_004,The rapid auburn animal bounded across the slumbering canine.,human,gradient,contributor_a,
```

### JSONL Example

```jsonl
{"id": "sample_001", "text": "The quick brown fox jumps over the lazy dog.", "source": "human", "attack_type": "none"}
{"id": "sample_002", "text": "Machine learning enables computers to learn from data without explicit instructions.", "source": "ai", "attack_type": "none", "generator_model": "gpt3.5-turbo"}
{"id": "sample_003", "text": "ML enables computational systems to adapt without pre-programmed algorithms.", "source": "ai", "attack_type": "paraphrase", "attack_owner": "contributor_b", "generator_model": "gpt3.5-turbo"}
```

### Size Recommendations

- **Smoke/development** (~10–100 samples): Test pipeline end-to-end quickly
- **Evaluation** (1K–10K samples): Per-attack robustness analysis with confidence intervals
- **Production** (10K+ samples): High-precision decision making; deploy with ensemble voting

All detectors use the same input contract so output metrics are directly comparable.

## Standard Workflow

### 1) Validate schema (recommended)

```bash
python -m evaluation.validate_schema --input data/splits/test.csv
```

### 2) Train the RoBERTa detector

```bash
python -m detectors.roberta_classifier.train \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --output-dir results/roberta_model \
  --epochs 2 \
  --batch-size 8
```

### 3) Run detector scoring

```bash
python -m evaluation.run_all \
  --input data/splits/test.csv \
  --output-dir results/detector_scores \
  --device cpu \
  --stats-model gpt2 \
  --roberta-model-dir results/roberta_model \
  --run-detectgpt \
  --run-fast-detectgpt \
  --run-binoculars \
  --run-watermark
```

### 4) Aggregate attack-wise metrics

```bash
python -m evaluation.aggregate_results \
  --scores-dir results/detector_scores \
  --output results/attack_eval_tables/metrics_summary.csv
```

### 5) Generate visualizations

```bash
python -m evaluation.plots \
  --metrics results/attack_eval_tables/metrics_summary.csv \
  --output-dir results/figures
```

## Output Artifacts

### 1) Detector Score Files

Location: `results/detector_scores/`

Each detector produces a standardized CSV with columns:

- **id**: Sample identifier (matches input)
- **text**: Original input text (for reference)
- **[detector_name]\_score**: Raw or normalized score (typically 0–1 for sigmoid-normalized, or z-score)

Filenames:

- `roberta_classifier_scores.csv`
- `detectgpt_style_scores.csv`
- `fast_detectgpt_scores.csv`
- `binoculars_scores.csv`
- `kgw_watermark_scores.csv`
- `stats_baseline_scores.csv`

Example output:

```csv
id,text,roberta_classifier_score,detectgpt_style_score,fast_detectgpt_score,binoculars_score,kgw_watermark_score,stats_baseline_score
sample_001,The quick brown fox...,0.15,0.22,0.18,0.25,0.05,0.20
sample_002,Machine learning enables...,0.92,0.88,0.85,0.90,0.03,0.87
```

### 2) Aggregated Metrics Table

Location: `results/attack_eval_tables/metrics_summary.csv`

Columns per detector × attack_type combination:

- **detector_name**: Which detector (roberta_classifier, detectgpt_style, etc.)
- **attack_type**: Attack category (none, prompt, paraphrase, gradient, char, any custom)
- **auroc**: Area Under the ROC Curve (0–1; 0.5 = random, 1.0 = perfect)
- **auprc**: Area Under the Precision-Recall Curve (0–1; useful for imbalanced data)
- **accuracy**: (TP + TN) / (all samples); 0–1
- **f1_ai**: F1 score for AI class (harmonic mean of precision and recall)
- **tpr_at_1pct_fpr**: True Positive Rate when operating at 1% False Positive Rate
- **tpr_at_5pct_fpr**: True Positive Rate at 5% FPR (practical operating point)
- **delta_auroc_vs_clean**: AUROC_attack - AUROC_clean (negative = robust to attack)
- **delta_tpr1_vs_clean**: TPR@1%FPR_attack - TPR@1%FPR_clean
- **delta_tpr5_vs_clean**: TPR@5%FPR_attack - TPR@5%FPR_clean

Example rows:

```csv
detector_name,attack_type,auroc,auprc,accuracy,f1_ai,tpr_at_1pct_fpr,tpr_at_5pct_fpr,delta_auroc_vs_clean,delta_tpr1_vs_clean,delta_tpr5_vs_clean
roberta_classifier,none,0.95,0.94,0.90,0.91,0.75,0.85,-0.00,-0.00,-0.00
roberta_classifier,paraphrase,0.78,0.76,0.72,0.70,0.35,0.55,-0.17,-0.40,-0.30
roberta_classifier,gradient,0.62,0.58,0.55,0.50,0.10,0.22,-0.33,-0.65,-0.63
detectgpt_style,none,0.82,0.80,0.78,0.77,0.50,0.68,-0.00,-0.00,-0.00
detectgpt_style,paraphrase,0.80,0.78,0.76,0.75,0.48,0.66,-0.02,-0.02,-0.02
```

**Interpretation**:

- Delta < -0.1 = detector exhibits good robustness (only small AUROC drop)
- Delta between -0.1 and -0.3 = moderate robustness (notable but recoverable)
- Delta < -0.3 = poor robustness (attack significantly degrades detector)

### 3) Visualization Outputs

Location: `results/figures/`

Generated plots:

- **auroc_by_detector_and_attack.png**: Heatmap showing AUROC per (detector, attack) combination
- **asr_by_detector_and_attack.png**: Attack Success Rate heatmap; ASR = percentage of AI samples misclassified as human under attack

These visualizations are critical for identifying which detector-attack pairs are weak and need ensemble reinforcement.

## Detailed Metric Definitions

### AUROC (Area Under the ROC Curve)

- **Range**: 0–1 (0.5 = random classifier, 1.0 = perfect)
- **Interpretation**: Probability that the detector assigns a higher score to a random AI sample than to a random human sample
- **When to use**: Primary single-number summary of classifier performance
- **Why important**: Independent of decision threshold; reflects true discriminative power

### AUPRC (Area Under the Precision-Recall Curve)

- **Range**: 0–1
- **Interpretation**: Weighted average of precision at all recall levels
- **When to use**: When class imbalance exists (more humans than AI, or vice versa)
- **Why important**: More informative than AUROC for imbalanced datasets

### Accuracy

- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Range**: 0–1
- **Interpretation**: Overall correctness across both classes
- **Warning**: Can be misleading if classes are imbalanced (e.g., 99% humans → 99% accuracy by always predicting human)

### F1 (AI Class)

- **Formula**: 2 \* (Precision × Recall) / (Precision + Recall)
- **Range**: 0–1
- **Interpretation**: Harmonic mean of precision (of AI predictions) and recall (of actual AI samples)
- **Why important**: Balances false positives and false negatives for the AI class

### TPR@1% FPR and TPR@5% FPR

- **Definition**: Sensitivity of detector when constrained to false-positive rate of 1% or 5%
- **Range**: 0–1
- **Interpretation**: "Out of every 100 human samples, how many AI samples are correctly identified?"
- **Why important**: Practical operating points; low false-positive constraint reflects real-world deployment needs (fewer false alarms)

### Delta Metrics

- **Formula**: Metric_under_attack - Metric_clean
- **Sign interpretation**: Negative = attack reduces performance (bad); positive = attack helps (rare, suspicious)
- **Threshold for robustness**: Most detectors see delta_auroc between -0.2 and -0.4 under strong attacks
- **Aggregating robustness**: Compute mean delta across all attacks per detector

## Individual Detector Usage

### Train RoBERTa Only

```bash
python -m detectors.roberta_classifier.train \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --output-dir results/roberta_model \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --max-length 512
```

**Output**:

- `results/roberta_model/pytorch_model.bin` (trained weights)
- `results/roberta_model/config.json` (model architecture)
- `results/roberta_model/tokenizer_config.json` (tokenizer settings)
- `results/roberta_model/metrics.json` (final eval metrics)

### Score with RoBERTa Only

```bash
python -m detectors.roberta_classifier.infer \
  --input data/splits/test.csv \
  --model-dir results/roberta_model \
  --output results/detector_scores/roberta_classifier_scores.csv \
  --batch-size 32 \
  --device cuda
```

### Score with DetectGPT Only

```bash
python -m detectors.detectgpt.score \
  --input data/splits/test.csv \
  --output results/detector_scores/detectgpt_style_scores.csv \
  --num-perturbations 8 \
  --mask-model google/t5-v1_1-base \
  --device cuda
```

Tune `--num-perturbations` for speed vs. quality trade-off:

- 2 perturbations: ~1 sec/sample, lower quality
- 8 perturbations: ~5 sec/sample, standard quality
- 16+ perturbations: >10 sec/sample, marginal improvement

### Score with Fast-DetectGPT Only

```bash
python -m detectors.fast_detectgpt.score \
  --input data/splits/test.csv \
  --output results/detector_scores/fast_detectgpt_scores.csv \
  --scoring-model gpt2-medium \
  --reference-model gpt2 \
  --device cuda
```

### Score with Binoculars Only

```bash
python -m detectors.binoculars.score \
  --input data/splits/test.csv \
  --output results/detector_scores/binoculars_scores.csv \
  --observer-model gpt2-medium \
  --performer-model gpt2 \
  --device cuda
```

### Score with Watermark Only

```bash
python -m detectors.watermark.score \
  --input data/splits/test.csv \
  --output results/detector_scores/kgw_watermark_scores.csv \
  --hash-key "my_secret_key" \
  --gamma 0.5
```

### Score with Stats Baseline Only

```bash
python -m detectors.stats_baseline.score \
  --input data/splits/test.csv \
  --output results/detector_scores/stats_baseline_scores.csv \
  --model gpt2 \
  --device cuda
```

## Evaluating Team Attack Data

To evaluate attack implementations from team members (e.g., Contributor A's gradient attack, Contributor B's paraphrasing, etc.):

### Workflow

1. **Collect attack outputs**: Each team member should output a CSV/JSONL with attack-transformed text
2. **Combine**: Merge all attack outputs into a single evaluation CSV (or keep separate and score each)
3. **Validate schema**:
   ```bash
   python -m evaluation.validate_schema --input combined_attacks.csv
   ```
4. **Score under attacks**:
   ```bash
   python -m evaluation.run_all \
     --input combined_attacks.csv \
     --output-dir results/detector_scores_attacks \
     --device cuda \
     --run-detectgpt --run-fast-detectgpt --run-binoculars --run-watermark \
     --roberta-model-dir results/roberta_model
   ```
5. **Aggregate per-attack metrics**:
   ```bash
   python -m evaluation.aggregate_results \
     --scores-dir results/detector_scores_attacks \
     --output results/attack_eval_tables/attack_robustness_summary.csv
   ```
6. **Compare robustness**: See which detectors (and which attacks) cause the largest AUROC drops

### Example Attack Workflow

Suppose Contributor A generates prompt-injection attacks, Contributor B creates paraphrased variants:

```
attacks_outputs/
  attack_prompts/
    attack_samples.csv  (attack_type="prompt", attack_owner="contributor_a")
  attack_paraphrase/
    attack_samples.csv  (attack_type="paraphrase", attack_owner="contributor_b")
```

Combine these, score them all together, and inspect delta metrics per attack_owner.

## Architecture & Design Patterns

### Detector Interface

Every detector module follows a consistent structure:

```
detectors/[detector_name]/
  __init__.py
  score.py        # Main scoring function: def score(texts, **kwargs) -> List[float]
  config.py       # Constants and hyperparameters
  utils.py        # Helper functions (optional)
```

### Adding a New Detector

1. Create folder: `detectors/new_detector/`
2. Implement `score.py` with signature:
   ```python
   def score(texts: List[str], **kwargs) -> List[float]:
       """Return list of scores in [0, 1] range."""
       return scores
   ```
3. Add entry to `evaluation/run_all.py` with a new CLI flag
4. Ensure output CSV has columns: `id`, `text`, `[detector_name]_score`

### Common Utilities

- **io_utils.py**: Schema validation, CSV/JSONL loading
- **metrics.py**: AUROC, F1, TPR@FPR computation
- **config.py**: Source labels (human=0, ai=1)

## Advanced Configuration

### Multi-GPU Scoring

To parallelize across multiple GPUs, modify `evaluation/run_all.py`:

```bash
# Score simultaneously on 4 GPUs
for detector in detectgpt binoculars roberta; do
  python -m detectors.$detector.score --device "cuda:$i" &
  i=$(($i+1 % 4))
done
wait
```

### Batch Processing Large Files

For 1M+ samples, split input into chunks:

```bash
split -l 100000 large_dataset.csv chunk_
for chunk in chunk_*; do
  python -m evaluation.run_all --input $chunk --output-dir results/detector_scores_${chunk}
done
# Concatenate results at the end
```

### Ensemble Voting

Combine multiple detector scores:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load all scores
scores_df = pd.read_csv("results/detector_scores/combined.csv")
detectors = ["roberta_classifier_score", "detectgpt_style_score", "binoculars_score"]

# Ensemble: average across detectors
scores_df["ensemble_score"] = scores_df[detectors].mean(axis=1)

# Evaluate on ground truth
auroc = roc_auc_score(scores_df["source"] == "ai", scores_df["ensemble_score"])
print(f"Ensemble AUROC: {auroc:.4f}")
```

### Disagreement-Aware Ensemble (Proposed Novelty)

Run the cross-paradigm meta-detector on existing detector score files:

```bash
python -m evaluation.disagreement_ensemble \
  --scores-dir results/detector_scores \
  --output-dir results/disagreement_ensemble \
  --detectors roberta_classifier fast_detectgpt binoculars kgw_watermark \
  --calibration-attack-type none
```

Outputs:

- `ensemble_feature_table.csv`: normalized scores `s_i` with disagreement feature `disagreement_var`
- `ensemble_ablation_metrics.csv`: mean baseline, logistic base, disagreement-augmented, oracle upper-bound
- `disagreement_ks_tests.csv`: KS-test validation for disagreement distribution shifts
- `ensemble_predictions.csv`: per-sample ensemble scores by ablation variant

### Cross-Paradigm Evasion Rate (All-Detector Simultaneous Evasion)

Compute per-detector evasion and simultaneous all-detector evasion for AI samples:

```bash
python -m evaluation.cross_paradigm_evasion \
  --scores-dir results/detector_scores \
  --output-dir results/cross_paradigm_evasion \
  --detectors roberta_classifier fast_detectgpt binoculars kgw_watermark \
  --group-by attack_type_owner
```

Outputs:

- `cross_paradigm_evasion_summary.csv`: includes `cross_paradigm_evasion_rate_all_selected`
- `per_paradigm_evasion_long.csv`: per-condition per-detector evasion rates
- `cross_paradigm_ai_wide.csv`: row-level AI-only table with boolean evasion flags per detector

### Detector Latency Benchmark

Benchmark per-sample inference latency across selected detectors:

```bash
python -m evaluation.latency_benchmark \
  --input data/splits/test.csv \
  --output results/latency/latency_benchmark.csv \
  --device cpu \
  --roberta-model-dir results/roberta_model \
  --run-fast-detectgpt \
  --run-binoculars \
  --run-watermark
```

The output CSV reports total elapsed time and `latency_ms_per_sample` for each detector.

## Performance Benchmarks (on sample data)

Baseline expectations (single GPU, 512-token texts):

| Detector          | Time/Sample | Memory | AUROC (clean) | Notes                       |
| ----------------- | ----------- | ------ | ------------- | --------------------------- |
| Stats Baseline    | 20ms        | <1GB   | 0.65–0.75     | Lightweight; always fast    |
| Fast-DetectGPT    | 50ms        | 2GB    | 0.75–0.85     | Good speed/quality ratio    |
| Binoculars        | 100ms       | 3GB    | 0.75–0.85     | Stable across domains       |
| RoBERTa (trained) | 30ms        | 1GB    | 0.85–0.95     | Best on in-domain data      |
| DetectGPT-style   | 3sec        | 4GB    | 0.80–0.90     | Slow but robust; 8 perturbs |
| Watermark         | 15ms        | <1GB   | 0.95+         | Only on watermarked text    |

**Total time for 10K-sample evaluation** (on single GPU):

- If running sequentially: ~6 hours
- If running in parallel (3 detectors at a time): ~2 hours

## Troubleshooting Guide

### 1) **Import Errors**

**Symptom**: `ModuleNotFoundError: No module named 'torch'`

```bash
# Reinstall dependencies
python -m pip install --upgrade -r requirements.txt
# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

### 2) **CUDA/GPU Issues**

**Symptom**: `RuntimeError: CUDA out of memory`

```bash
# Reduce batch size or use CPU
python -m evaluation.run_all --device cpu --batch-size 8
# Or clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### 3) **Missing Required Columns**

**Symptom**: `KeyError: 'text'` during scoring

```bash
# Validate input schema first
python -m evaluation.validate_schema --input data.csv
# Check CSV header
head -1 data.csv
```

**Fix**: Ensure CSV has exactly columns: id, text (+ optional source, attack_type, etc.)

### 4) **RoBERTa Model Not Found**

**Symptom**: `FileNotFoundError: results/roberta_model/pytorch_model.bin`

```bash
# Train the model first
python -m detectors.roberta_classifier.train \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --output-dir results/roberta_model
```

### 5) **Slow or Hanging Detector**

**Symptom**: Script runs but produces no output after 10+ minutes

```bash
# Check system resources
watch -n 1 'gpu-smi' # Monitor GPU
# Or reduce perturbations (for DetectGPT)
python -m detectors.detectgpt.score --num-perturbations 2
```

### 6) **NaN or Invalid Scores**

**Symptom**: Output CSV contains NaN or inf values

```bash
# Check for empty texts in input
awk 'NF' data.csv | wc -l  # Count non-empty lines
# Inspect problematic rows
grep -n "^[^,]*,[[:space:]]*," data.csv  # Find empty text fields
```

**Fix**: Clean input data; remove or fill empty text fields.

### 7) **Metrics Summary Has Missing Rows**

**Symptom**: `metrics_summary.csv` missing entries for some (detector, attack_type) pairs
**Cause**: One or more detectors failed silently; check individual score CSVs

```bash
# List score files
ls -lh results/detector_scores/
# Check if all expected files exist
grep "detector_name,attack_type" results/attack_eval_tables/metrics_summary.csv | wc -l
```

### 8) **Low AUROC on Clean Data**

**Symptom**: AUROC < 0.6 even for clean (non-attacked) test set
**Possible causes**:

- Model undertrained; increase epochs
- Train/test domain mismatch (e.g., trained on GPT-3.5, tested on Llama)
- Insufficient training data; use larger splits

**Fix**:

```bash
# Retrain with more data and longer
python -m detectors.roberta_classifier.train \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --epochs 5 \
  --learning-rate 2e-5
```

## References & Papers

Core detector papers:

- **RoBERTa classifier**: Fine-tuned transformer baseline (supervised learning)
- **DetectGPT**: Mitchell et al. (2023) _"DetectGPT: Zero-Shot Detector via Perturbation"_
- **Fast-DetectGPT**: Bao et al. (2024) _"Fast-DetectGPT: Efficient Zero-Shot Detector"_
- **Binoculars**: Sadasivan et al. (2024) _"Binoculars: Detecting Deepfakes with Forged Keypoints"_
- **KGW Watermark**: Kirchenbauer et al. (2023) _"A Watermark for Large Language Models"_
- **Statistical baseline**: Classical perplexity-based heuristics

Attack motivation:

- SICO (2305) - Classifier evasion
- ADAT/HMGC (2404) - Detector robustness
- SilverSpeak (2406) - Paraphrasing attacks
- GradEscape (2506) - Gradient-based evasion
- Adversarial Paraphrasing (2506_07) - Semantic preservation under attack

## Advanced Evaluation Modules

### 1) Cross-Detector Transferability Analysis

Generates attack transferability tables and detector vulnerability correlations from
`*_scores.csv` files.

```bash
python -m evaluation.transferability \
  --scores-dir results/detector_scores \
  --output-dir results/transferability \
  --group-by attack_type \
  --drop-threshold 0.10
```

Outputs:

- `results/transferability/transferability_long_metrics.csv`
- `results/transferability/attack_transferability_matrix.csv`
- `results/transferability/attack_transferability_summary.csv`
- `results/transferability/detector_vulnerability_correlation.csv`

### 2) Watermark Robustness Under Attacks

Evaluates the KGW detector under attack groupings and reports delta degradation
versus clean condition.

```bash
python -m evaluation.watermark_robustness \
  --scores-dir results/detector_scores \
  --output results/attack_eval_tables/watermark_robustness.csv \
  --group-by attack_type
```

Output:

- `results/attack_eval_tables/watermark_robustness.csv`

### 3) Adaptive Retraining Stress Test (Optional)

Runs iterative rounds of:

1. Train RoBERTa on current train set
2. Evaluate on attacked set
3. Mine hard examples (false negatives + false positives)
4. Augment train set
5. Repeat

```bash
python -m evaluation.adaptive_retrain_stress \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --attack-eval data/splits/test.csv \
  --work-dir results/adaptive_retrain \
  --rounds 3 \
  --epochs 2 \
  --batch-size 8 \
  --device cpu \
  --max-hard-per-class 200
```

Outputs:

- `results/adaptive_retrain/round_*/roberta_model/` (model per round)
- `results/adaptive_retrain/round_*/attack_eval_roberta_scores.csv`
- `results/adaptive_retrain/adaptive_retrain_summary.csv`

### 4) One-Command Analysis Suite

Runs the full post-training workflow end-to-end in one command:

1. Run all detectors on train/val/test
2. Aggregate metrics and generate plots for each split
3. Build one combined all-splits table
4. Build detector mean/std summary across splits
5. Run transferability, watermark robustness, cross-paradigm evasion,
   disagreement ensemble, and latency benchmark

```bash
python -m evaluation.analysis_suite \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --test data/splits/test.csv \
  --model-dir results/roberta_model \
  --device cpu \
  --detectgpt-perturb 2 \
  --output-root results/analysis_suite
```

Main outputs:

- `results/analysis_suite/tables/metrics_train.csv`
- `results/analysis_suite/tables/metrics_val.csv`
- `results/analysis_suite/tables/metrics_test.csv`
- `results/analysis_suite/tables/metrics_all_splits.csv`
- `results/analysis_suite/tables/detector_summary_across_splits.csv`
- `results/analysis_suite/figures/train/`
- `results/analysis_suite/figures/val/`
- `results/analysis_suite/figures/test/`
- `results/analysis_suite/insights/`
- `results/analysis_suite/run_manifest.json`

Windows PowerShell wrapper (from `detector_evaluation/`):

```powershell
.\run_analysis_suite.ps1 -Device cpu -DetectGptPerturb 2
```

---

**Last updated**: March 14, 2026
