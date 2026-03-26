# Stylometry-Aware Differentiable Paraphraser

**Cross-Paradigm Transferable Adversarial Paraphrasing of AI-Generated Text**

---

## Overview

This repository implements the methodology described in:

> *Cross-Paradigm Transferable Adversarial Paraphrasing via Stylometric Joint Optimisation*

It builds directly upon and extends two prior works:

- **GradEscape** (Meng et al., USENIX Security 2025) — the pseudo-embedding technique and gradient-based evader.
- **HMGC** (Zhou et al., arXiv 2404.01907) — the word-importance ranking and adversarial detection attack task formulation (ADAT).

### The Core Problem

State-of-the-art evaders (GradEscape, HMGC, DIPPER) optimise purely for fooling a *single* surrogate detector. This creates **adversarial overfitting**: the paraphrased text exploits the neural classifier's specific decision boundary but ignores macroscopic document-level properties. Statistical zero-shot detectors (DetectGPT, GLTR) — which never see the adversarial text during training — remain effective because AI-generated text, even after evasion, still exhibits:

- **Suspiciously uniform sentence lengths** (low burstiness / variance)
- **Lower lexical diversity** than genuine human writing

### The Proposed Solution

We introduce a **joint tri-objective loss**:

```
L_total = α · L_adv  +  β · L_sem  +  γ · L_style
```

| Term | Purpose | Novelty |
|------|---------|---------|
| `L_adv` | Fool the surrogate neural detector | From GradEscape (pseudo-embeddings) |
| `L_sem` | Preserve token-level + semantic meaning | From GradEscape (label loss + MSE) |
| **`L_style`** | **Mimic human sentence-length distribution (KL) + lexical entropy** | **Novel contribution** |

`L_style` penalises deviations from a human-corpus baseline distribution, pushing the adversarial text toward the interior of the human manifold rather than just past the surrogate's decision boundary. This is what drives **Cross-Paradigm Transferability**.

---

## Repository Structure

```
stylometry_evader/
│
├── config.py              # All hyperparameters (dataclasses)
├── data_utils.py          # Dataset loading + HumanCorpusStats
├── pseudo_embeddings.py   # Differentiable bridge: evader → detector
├── stylometric_loss.py    # L_style: soft histogram KL + lexical entropy
├── losses.py              # L_adv, L_sem, JointEvaderLoss
├── evader.py              # StyleAwareEvader model class
├── trainer.py             # Training loop, checkpointing
├── evaluator.py           # ASR, CPTR, BLEU, BERTScore, stylometrics
├── main.py                # CLI entry point
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare data

You need three files (one document per line):

| File | Content |
|------|---------|
| `data/train_ai_text.txt` | AI-generated training texts (e.g., from LLaMA-3, GPT-4) |
| `data/eval_ai_text.txt` | AI-generated evaluation texts (held out) |
| `data/human_corpus.txt` | Genuine human-authored documents (Wikipedia, Reddit, etc.) |

### 2. Train your surrogate detector (you supply this)

The evader codebase does **not** train the detector. You must fine-tune a binary classifier (class 0 = human, class 1 = AI-generated) using any standard procedure, e.g.:

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
# ... standard HuggingFace fine-tuning on your labelled corpus ...
model.save_pretrained("my_roberta_detector/")
```

Then update `config.py`:

```python
model.surrogate_detector_name = "my_roberta_detector/"
```

### 3. Train the evader

```bash
python main.py train
```

This will:
1. Build the human-corpus stylometric baseline and cache it to `outputs/checkpoints/human_stats.json`.
2. Initialise the BART evader backbone.
3. Run the joint-loss training loop.
4. Save checkpoints to `outputs/checkpoints/`.

### 4. Evaluate

```bash
python main.py evaluate \
    --checkpoint outputs/checkpoints/best \
    --results    outputs/eval_results.json
```

To evaluate against black-box detectors, add their checkpoint paths in `config.py`:

```python
eval.blackbox_detector_paths = [
    "my_deberta_detector/",
    "some_other_detector/",
]
```

### 5. Interactive paraphrase

```bash
python main.py paraphrase --checkpoint outputs/checkpoints/best
```

---

## Architecture Details

### Pseudo-Embedding Technique

The key challenge in gradient-based NLP attacks is that token sampling is non-differentiable. We solve this using pseudo-embeddings (GradEscape, Section 4.2):

```
Evader (BART) → logits → softmax → P ∈ R^{B × L × V}
                                    ↓
                        pseudo_emb = P @ W_emb  ∈ R^{B × L × H}
                                    ↓
                        Frozen Detector → logits → L_adv
                                    ↓
                        ∂L_adv / ∂θ_evader  (end-to-end gradient)
```

BART and RoBERTa share the same BPE tokenizer, so `P @ W_emb` works with no vocabulary remapping.

### Stylometric Loss (Novel)

`L_style` is the primary novel contribution:

```
L_style = w_burst · KL_burstiness  +  w_lex · LexicalEntropyDeviation
```

**KL_burstiness**: We approximate the sentence-length distribution using *soft sentence boundaries* — the probability assigned to period/exclamation/question-mark tokens at each position. This creates a fully differentiable histogram that can be compared with the human-corpus baseline via symmetric KL divergence.

```python
# Simplified illustration:
p_boundary = prob_matrix[:, :, period_token_ids].sum(dim=-1)  # (B, L)
# Accumulate soft sentence lengths via running counter
# Build histogram via Gaussian kernel spreading
# KL(soft_hist || human_hist)
```

**LexicalEntropyDeviation**: The squared difference between the mean token-level Shannon entropy of the output and the human-corpus baseline entropy. This encourages the evader to use a similarly diverse vocabulary to human writers.

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **ASR** | Attack Success Rate on white-box surrogate | High (↑) |
| **CPTR** | Cross-Paradigm Transfer Rate on black-box detectors | High (↑) — primary metric |
| BLEU | n-gram overlap with original | ≥ 0.6 |
| ROUGE-Lsum | LCS recall | ≥ 0.90 (quality filter) |
| BERTScore F1 | Contextual semantic similarity | ≥ 0.90 |
| KL burstiness | KL div of sentence-length dist vs. human baseline | Low (↓) |
| Burstiness coeff | std / mean of sentence lengths | Higher = more human-like |
| Lexical entropy | Shannon entropy of token unigrams | Close to human baseline |

---

## Ablation Study Design

Compare four configurations:

| Config | L_adv | L_sem | L_style |
|--------|-------|-------|---------|
| **Ours (full)** | ✓ | ✓ | ✓ |
| GradEscape baseline | ✓ | ✓ | ✗ |
| HMGC baseline | mask-based | USE+POS | ✗ |
| L_style only (ablation) | ✗ | ✗ | ✓ |

The key hypothesis: while GradEscape and HMGC achieve high ASR on the surrogate, their CPTR against statistical detectors (DetectGPT) will be significantly lower than the full model, which explicitly constrains gradient updates to mimic macroscopic human distributions.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.40 | Weight of L_adv |
| `beta` | 0.30 | Weight of L_sem |
| `gamma` | 0.30 | Weight of L_style |
| `n_length_bins` | 25 | Histogram resolution for KL |
| `max_sentence_length` | 80 | Max tokens per sentence for histogram |
| `boundary_temperature` | 1.0 | Gaussian kernel spread for soft histogram |
| `burstiness_weight` | 0.7 | Relative weight within L_style |
| `entropy_weight` | 0.3 | Relative weight within L_style |

---

## References

```
@inproceedings{meng2025gradescape,
  title     = {GradEscape: A Gradient-Based Evader Against AI-Generated Text Detectors},
  author    = {Meng, Wenlong and Fan, Shuguo and Wei, Chengkun and Chen, Min and
               Li, Yuwei and Zhang, Yuanchao and Zhang, Zhikun and Chen, Wenzhi},
  booktitle = {34th USENIX Security Symposium},
  year      = {2025}
}

@article{zhou2024hmgc,
  title   = {Humanizing Machine-Generated Content: Evading AI-Text Detection
             through Adversarial Attack},
  author  = {Zhou, Ying and He, Ben and Sun, Le},
  journal = {arXiv preprint arXiv:2404.01907},
  year    = {2024}
}

@inproceedings{mitchell2023detectgpt,
  title     = {DetectGPT: Zero-Shot Machine-Generated Text Detection
               using Probability Curvature},
  author    = {Mitchell, Eric and Lee, Yoonho and Khazatsky, Alexander and
               Manning, Christopher D. and Finn, Chelsea},
  booktitle = {ICML},
  year      = {2023}
}
```
