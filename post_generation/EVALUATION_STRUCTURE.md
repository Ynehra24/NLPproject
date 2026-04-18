# Post-Generation Evaluation Structure

Organized structure for Flan-T5 evader evaluation and analysis.

## Directory Layout

```
post_generation/
├── HMGC-dataset/                    # Original HMGC dataset & trained models
├── evaluation/                      # Evaluation pipeline and results
│   ├── scripts/
│   │   └── run_evaluation.py        # Main end-to-end evaluation script
│   ├── results/
│   │   └── flan_t5_final/          # Final evaluation results
│   │       ├── evaluation_metrics.json
│   │       └── humanized_texts.csv
│   ├── metrics.py                   # Evaluation metric utilities
│   ├── eval_*.py                    # Various evaluation scripts
│   └── models/                      # Detector models for evaluation
├── analysis/
│   └── MODEL_COMPARISON_AND_DETECTOR_ANALYSIS.md  # Complete analysis document
├── attack/                          # Attack generation code
├── scripts/                         # Training scripts
├── utils/                           # Utilities
├── logs/                            # Training logs
├── README.md
├── requirements.txt
└── Paper.pdf
```

## Key Files

- **Evaluation Script:** `evaluation/scripts/run_evaluation.py`
  - End-to-end pipeline: load test data → generate humanized text → run detector → compute metrics
  - Usage: `python post_generation/evaluation/scripts/run_evaluation.py --test-data ... --evader-model ... --roberta-model ... --output-dir ...`

- **Analysis Document:** `analysis/MODEL_COMPARISON_AND_DETECTOR_ANALYSIS.md`
  - Complete comparison of Flan-T5 evader vs HMGC paper
  - Evaluation results: 100% evasion rate achieved
  - Why elaboration-based extension fools detectors

- **Results:** `evaluation/results/flan_t5_final/`
  - `evaluation_metrics.json` - Quantitative results (AUC, accuracy, evasion rate)
  - `humanized_texts.csv` - 50 generated/humanized texts with detector scores

## Model Paths

- **Evader Model:** `post_generation/HMGC-dataset/output/checkgpt/model/evader_flan_t5_base_12to15h_earlystop_v1`
- **Detector Model:** `detector_evaluation/results/` (RoBERTa, independent from HMGC)
- **Test Data:** `detector_evaluation/results/checkgpt_phase3_input_100.csv` (50 samples: 27 AI, 23 human)

## Quick Start

Run full evaluation:
```bash
cd /path/to/NLPproject
python post_generation/evaluation/scripts/run_evaluation.py \
  --test-data detector_evaluation/results/checkgpt_phase3_input_100.csv \
  --evader-model post_generation/HMGC-dataset/output/checkgpt/model/evader_flan_t5_base_12to15h_earlystop_v1 \
  --roberta-model detector_evaluation/results \
  --limit 50 \
  --device cpu \
  --output-dir post_generation/evaluation/results/flan_t5_final
```

## Results Summary

| Metric | Value |
|--------|-------|
| Test Samples | 50 (27 AI, 23 human) |
| Evasion Rate | 100% |
| AUC-ROC | 0.5000 (random baseline) |
| Detector | RoBERTa (CheckGPT) |
| vs Paper | +2.71% (100% vs 97.29%) |

See `analysis/MODEL_COMPARISON_AND_DETECTOR_ANALYSIS.md` for full details.
