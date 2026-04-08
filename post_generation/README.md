# Humanizing Machine-Generated Content

This repository contains resources of our paper:
- [Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack](https://arxiv.org/abs/2404.01907)

---

## How to reporduce our result
1. Download and unzip dataset from [Google Drive](https://drive.google.com/file/d/15rdZfNmnVeqEFKSu1A01DIvhYL30vadi)

2. Run
```
python evaluation/eval_accuracy.py \
    --detector hc3 \
    --tests ./output/hc3/**/*.jsonl \
    --output_file /tmp/hc3_evaluation.csv
```


## Do attacks on your own data
1. Distill sample labels from your target victim detector, train a surrogate model with `train_detector.py`

2. Follow `attack.multi_flint_attack` to start multi-process attacking


## Phase-1 Semantic Constraint Upgrade
The attack recipes now use a cross-encoder semantic gate instead of Universal Sentence Encoder.

- Default model: `cross-encoder/stsb-roberta-large`
- Similarity scores are normalized to `[0, 1]`
- Default threshold: `0.75`

You can tune Phase-1 behavior without changing code:

```bash
python attack/multi_flint_attack.py \
    --model_name_or_path /path/to/surrogate \
    --data_file /path/to/data.jsonl \
    --output_dir /path/to/output \
    --attacking_method dualir \
    --semantic_model_name cross-encoder/stsb-roberta-large \
    --semantic_threshold 0.78 \
    --semantic_window_size 50 \
    --semantic_batch_size 16
```

Notes:
- `--attacking_method no_use` (or `no_semantic`) disables semantic filtering for ablations.
- The semantic model is loaded lazily in each worker process.
- Phase-1 implementation only changes attack-time semantic constraints (no training changes).

Quick smoke check (loads semantic model + scores one pair, no training):

```bash
python attack/scripts/smoke_semantic_constraint.py \
    --semantic_model_name cross-encoder/stsb-roberta-large \
    --semantic_threshold 0.75
```

If `textattack` is unavailable in the active environment, the script automatically falls back to direct Transformers scoring and still validates model loading.


## Citation
If you find our paper/resources useful, please cite:
```
@inproceedings{Zhou2024_COLING,
 author = {Ying Zhou and
           Ben He and
           Le Sun},
 title = {Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack},
 booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation.},
 year = {2024},
}
```
