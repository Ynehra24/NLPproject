# Gradient-Based Adversarial Evasion

A hybrid gradient + reinforcement learning based adversarial evasion system evaluated across multiple AI detection paradigms.

**Author:** Udaiveer Singh (U20230017) — Advanced NLP, Plaksha University  
**Attack tag:** `gradient` (used in unified pipeline's `attack_type` column)

---

## Overview

Combines two evasion paradigms under a single composite loss:

```
L = λ · L_grad  +  (1 − λ) · L_rl  +  α · L_sem
```

| Term     | Type                | What it does                                                                |
| -------- | ------------------- | --------------------------------------------------------------------------- |
| `L_grad` | White-box gradient  | Backprop from surrogate RoBERTa through pseudo-embeddings into LoRA adapter |
| `L_rl`   | Black-box RL (GRPO) | Reward = 1 − detector_confidence; policy update without gradient access     |
| `L_sem`  | Quality constraint  | BERTScore cosine sim between original and paraphrase ≥ 0.92                 |

A tunable **λ** sweeps from pure gradient (λ=1, GradEscape replica) to pure RL
(λ=0, AuthorMist replica). The hybrid at intermediate λ is the core novelty —
testing whether combining both signals yields better cross-paradigm
transferability than either alone.

An **iterative inference loop** (K=5 rounds max) re-feeds output back through
the adapter until detector confidence drops below threshold τ=0.5.

---

## File Structure

```
gradientBasedAttacks/
|-- README.md
|-- requirements.txt
|-- IMPLEMENTATION_PLAN.txt
|
|-- notebooks/
|   |-- eval1_roberta_baseline.ipynb      ← Eval 1: RoBERTa, 500 samples
|   |-- eval2_cross_paradigm.ipynb        ← Eval 2: RoBERTa+DetectGPT, 1000 samples
|   \-- eval3_lambda_sweep.ipynb          ← Eval 3: Full ablation (planned)
|
|-- scripts/
|   |-- download_hc3.py                   ← Download and cache HC3 from HuggingFace
|   \-- prepare_splits.py                 ← Build train/val/test CSV splits
|
|-- evader/
|   |-- __init__.py
|   |-- common/
|   |   |-- __init__.py
|   |   \-- interfaces.py                 ← BaseEvader abstract class
|   |
|   |-- lora_adapter/
|   |   |-- __init__.py
|   |   \-- model.py                      ← LoRA on frozen T5-base; composite loss
|   |
|   |-- pseudo_embedding/
|   |   |-- __init__.py
|   |   \-- grad_loss.py                  ← Pseudo-embedding construction + L_grad
|   |
|   \-- rl_trainer/
|       |-- __init__.py
|       \-- grpo.py                       ← GRPO reward loop + L_rl
|
|-- evaluation/
|   |-- conversion_script.py                       ← Converts evader outputs → paired CSV for detector pipeline
|   |-- metrics.py                        ← ASR, CPTR, BERTScore, ROUGE-L, GRUEN
|   \-- lambda_sweep.py                   ← Train + eval across λ ∈ {0, 0.25, 0.5, 0.75, 1}
|
\-- data/
    |-- raw/                              ← HC3 all.jsonl (downloaded by script)
    |-- processed/                        ← Filtered + tokenized text
    \-- splits/
        |-- train.csv                     ← ~22K AI samples for evader training
        |-- val.csv                       ← ~1.5K for validation
        \-- test.csv                      ← 1K AI + 1K Human for evaluation
```

---

## Baseline Results (Completed)

### Eval 1 — RoBERTa (500 samples, HC3)

| Metric                       | Value |
| ---------------------------- | ----- |
| AI Detection Rate            | 99.4% |
| Avg AI confidence on AI text | 99.3% |
| False Positive Rate          | 0.8%  |

### Eval 2 — Cross-Paradigm (1000 samples, HC3)

| Detector   | Paradigm              | Samples | AI Det. Rate         | FPR   |
| ---------- | --------------------- | ------- | -------------------- | ----- |
| RoBERTa    | Neural classifier     | 1000    | 98.9%                | 0.5%  |
| DetectGPT  | Statistical zero-shot | 200     | 87.5%                | 10.5% |
| Binoculars | Likelihood ratio      | —       | Deferred (Falcon-7B) | —     |

**Key finding:** 11.4pp cross-paradigm gap between RoBERTa and DetectGPT —
directly motivating the hybrid evader.

---

## Eval 3 — Full λ Sweep + Detector Pipeline (Completed)

The hybrid gradient-RL evader was evaluated across the **team's unified detector pipeline**, covering multiple detection paradigms:

- Neural classifier (RoBERTa)
- Likelihood-based (Fast-DetectGPT, Binoculars)
- Statistical baseline
- Curvature-based (DetectGPT)
- Watermark detector

### Conversion to Paired Format

Evader outputs are converted before evaluation:

```bash
python evaluation/conversion_script.py
```

This generates:
`paired_clean.csv` → input for detector pipeline

### λ Sweep (Core Experiment)

Full ablation performed over: **λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}**

### Key Results

| Detector Type           | Best ASR     |
| ----------------------- | ------------ |
| Statistical baseline    | Up to 100%   |
| Likelihood-based        | Up to 95%+   |
| RoBERTa classifier      | Up to 90%    |

The evader demonstrates strong performance across multiple detectors, achieving consistent evasion across statistical, likelihood-based, and classifier-based methods within the evaluation pipeline.

### Key Observations

**1. λ as a Control Knob**

λ directly controls evasion behavior across detector types:

| λ       | Behavior                                    |
| ------- | ------------------------------------------- |
| 0.0     | Strong fluency-driven transformations       |
| 0.25–0.5 | Balanced performance                       |
| 0.75    | Best overall performance                    |
| 1.0     | Strong gradient-driven adversarial behavior |

**2. Peak Performance at λ = 0.75**

λ = 0.75 provides the best trade-off: high classifier evasion, strong likelihood-based evasion, and stable semantic quality.

**3. Complementary Hybrid Design**

- Gradient optimization → strong adversarial shifts
- RL component → improves fluency and stability
- Hybrid approach enables consistent multi-detector performance

**4. Robust Multi-Paradigm Performance**

The evader performs strongly across neural classifiers, likelihood-based detectors, and statistical methods — demonstrating effective combination of white-box and black-box signals.

### Integration Flow

```
Train Evader → evaded.csv
        ↓
conversion_script.py
        ↓
paired_clean.csv
        ↓
detector pipeline
        ↓
metrics.csv + evasion_summary.csv
```

## Environment Setup

```bash
cd gradientBasedAttacks
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Data

All humanizer modules use the **complete HC3 dataset** (~25K QA pairs).

```bash
python scripts/download_hc3.py          # downloads all.jsonl via hf_hub_download
python scripts/prepare_splits.py        # builds train/val/test.csv
```

Input schema (matches unified pipeline):

| Column            | Required    | Description                  |
| ----------------- | ----------- | ---------------------------- |
| `id`              | ✅          | Unique sample id             |
| `text`            | ✅          | Raw text                     |
| `source`          | recommended | `"human"` or `"ai"`          |
| `attack_type`     | recommended | `"gradient"` for this module |
| `generator_model` | optional    | `"gpt3.5-turbo"` etc.        |

---

## Training

```bash
# Single λ value
python evader/lora_adapter/model.py \
  --train data/splits/train.csv \
  --val   data/splits/val.csv \
  --lambda 0.5 --alpha 0.3 \
  --epochs 3 --batch-size 8 \
  --output results/evader_outputs/lambda_0.5 \
  --device cuda

# Full λ sweep (core experiment)
python evaluation/lambda_sweep.py \
  --train data/splits/train.csv \
  --test  data/splits/test.csv \
  --lambda-values 0.0 0.25 0.5 0.75 1.0 \
  --output results/metrics/lambda_sweep.csv
```

---

## Integration with Team Pipeline

Evader output CSVs (generated locally under `results/`) drop directly into the detector evaluation module:

```bash
python -m evaluation.run_all \
  --input ../gradientBasedAttacks/results/metrics/lambda_0.75/paired_clean.csv \
  --output-dir results/gradient_attack_scores \
  --run-detectgpt --run-binoculars \
  --roberta-model-dir results/roberta_model
```
> **Note:** The `results/` directory is not tracked in the repository and is generated locally during training and evaluation.
---

## References

- Meng et al. (2025). GradEscape. USENIX Security. arXiv:2506.08188
- David & Gervais (2025). AuthorMist. arXiv:2503.08716
- StealthRL (2026). arXiv:2602.08934
- Mitchell et al. (2023). DetectGPT. ICML.
- Hans et al. (2024). Binoculars. ICML.
