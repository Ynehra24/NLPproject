# Gradient-Based Adversarial Evasion

Hybrid Gradient-RL evader for AI text humanization. One of five attack modules
in the NLPproject adversarial evaluation framework.

**Author:** Udaiveer Singh (U20230017) — Advanced NLP, Plaksha University  
**Attack tag:** `gradient` (used in unified pipeline's `attack_type` column)

---

## Overview

Combines two evasion paradigms under a single composite loss:

```
L = λ · L_grad  +  (1 − λ) · L_rl  +  α · L_sem
```

| Term | Type | What it does |
|---|---|---|
| `L_grad` | White-box gradient | Backprop from surrogate RoBERTa through pseudo-embeddings into LoRA adapter |
| `L_rl` | Black-box RL (GRPO) | Reward = 1 − detector_confidence; policy update without gradient access |
| `L_sem` | Quality constraint | BERTScore cosine sim between original and paraphrase ≥ 0.92 |

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
|   |-- baseline.py                       ← Shared detector loading + inference utils
|   |-- metrics.py                        ← ASR, CPTR, BERTScore, ROUGE-L, GRUEN
|   \-- lambda_sweep.py                   ← Train + eval across λ ∈ {0, 0.25, 0.5, 0.75, 1}
|
|-- data/
|   |-- raw/                              ← HC3 all.jsonl (downloaded by script)
|   |-- processed/                        ← Filtered + tokenized text
|   \-- splits/
|       |-- train.csv                     ← ~22K AI samples for evader training
|       |-- val.csv                       ← ~1.5K for validation
|       \-- test.csv                      ← 1K AI + 1K Human for evaluation
|
\-- results/
    |-- metrics/
    |   |-- baseline_eval1.csv            ← Eval 1 results (done)
    |   |-- baseline_eval2.csv            ← Eval 2 results (done)
    \-- evader_outputs/                   ← Paraphrased CSVs per λ (Eval 3)
```

---

## Baseline Results (Completed)

### Eval 1 — RoBERTa (500 samples, HC3)

| Metric | Value |
|---|---|
| AI Detection Rate | 99.4% |
| Avg AI confidence on AI text | 99.3% |
| False Positive Rate | 0.8% |

### Eval 2 — Cross-Paradigm (1000 samples, HC3)

| Detector | Paradigm | Samples | AI Det. Rate | FPR |
|---|---|---|---|---|
| RoBERTa | Neural classifier | 1000 | 98.9% | 0.5% |
| DetectGPT | Statistical zero-shot | 200 | 87.5% | 10.5% |
| Binoculars | Likelihood ratio | — | Deferred (Falcon-7B) | — |

**Key finding:** 11.4pp cross-paradigm gap between RoBERTa and DetectGPT —
directly motivating the hybrid evader.

---

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

| Column | Required | Description |
|---|---|---|
| `id` | ✅ | Unique sample id |
| `text` | ✅ | Raw text |
| `source` | recommended | `"human"` or `"ai"` |
| `attack_type` | recommended | `"gradient"` for this module |
| `attack_owner` | recommended | `"udaiveer"` |
| `generator_model` | optional | `"gpt3.5-turbo"` etc. |

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

## Evaluation (Planned — Eval 3)

```bash
# Run iterative inference loop on test set
python evaluation/metrics.py \
  --test  data/splits/test.csv \
  --model results/evader_outputs/lambda_0.5 \
  --max-rounds 5 --threshold 0.5 \
  --output results/metrics/eval3_lambda_0.5.csv
```

Metrics tracked: ASR, CPTR, ASR@K (K=1,2,3,5), BERTScore F1, ROUGE-L, GRUEN

---

## Integration with Team Pipeline

Evader output CSVs drop directly into the detector evaluation module:

```bash
python -m evaluation.run_all \
  --input ../gradientBasedAttacks/results/evader_outputs/lambda_0.5/evaded.csv \
  --output-dir results/gradient_attack_scores \
  --run-detectgpt --run-binoculars \
  --roberta-model-dir results/roberta_model
```

---

## References

- Meng et al. (2025). GradEscape. USENIX Security. arXiv:2506.08188
- David & Gervais (2025). AuthorMist. arXiv:2503.08716
- StealthRL (2026). arXiv:2602.08934
- Mitchell et al. (2023). DetectGPT. ICML.
- Hans et al. (2024). Binoculars. ICML.
