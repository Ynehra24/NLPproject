# Adv-Humanizer v3: Scorched-Earth Adversarial Attack Framework 🛡️🔥

Adv-Humanizer is a character-level adversarial attack framework that evades modern AI text detectors by restructuring tokens at the byte-level rather than semantically paraphrasing. Version 3 brings a ground-up redesign: introducing architecture-specific evasion strategies natively targeting the BPE tokenization vulnerabilities of both BERT and RoBERTa models.

**Targets defeated:** `stats_baseline` (GPT-2 perplexity/rank), `fast_detectgpt`, `binoculars`, `detectgpt`, `kgw_watermark`, and fine-tuned classifiers (`BERT-base`, `RoBERTa-base`).

---

## 📂 Directory Structure

```
NLPproject/
├── characterlevelattacks/
│   ├── coreattacks/
│   │   ├── humanizer.py             # Entry point & CLI
│   │   ├── composite_scorer.py      # Evasion-Oriented Scoring Engine
│   │   ├── csbp_loop.py             # v3 CSBP Beam Search (blitz, roberta_subword & more)
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

## 🚀 Evaluation Results

Full results across all 12 evaluated configurations (BERT and RoBERTa, 6 datasets each). Metrics include Attack Success Rate (ASR), composite score (Avg S), evasion, BPE disruption, watermark z-score, cosine similarity, perplexity, and token rank. All runs use n=50 samples.

### Full Results Table

| # | Detector | Dataset | Mode | Total | Flips | ASR (%) | Avg S | Avg Evasion | Avg BPE Disrupt | Avg WM z-score | Avg Cosine | Avg PPL | Avg Rank |
|---|----------|---------|------|-------|-------|---------|-------|-------------|-----------------|----------------|------------|---------|----------|
| 1 | BERT | AGNEWS | humanizer_bert | 50 | 49 | **98.0** | 0.7906 | 0.6235 | 0.9603 | 0.5035 | 0.9729 | 66.33 | 287.85 |
| 2 | BERT | HC3 | humanizer_bert | 50 | 49 | **98.0** | 0.7720 | 0.5414 | 0.9900 | 0.8019 | 0.9835 | 34.03 | 167.47 |
| 3 | BERT | M4 | humanizer_bert | 50 | 41 | 82.0 | 0.7166 | 0.5775 | 0.7920 | 0.6340 | 0.9887 | 45.85 | 244.19 |
| 4 | BERT | QNLI | humanizer_bert | 50 | 43 | 86.0 | 0.8649 | 0.8344 | 0.9317 | 0.0441 | 0.9916 | 133.74 | 551.39 |
| 5 | BERT | RTE | humanizer_bert | 50 | 28 | 56.0 | 0.8956 | 0.9184 | 0.9472 | 0.1822 | 0.9776 | 262.29 | 975.74 |
| 6 | BERT | SST2 | humanizer_bert | 50 | 49 | **98.0** | 0.8273 | 0.7433 | 0.9585 | 0.3999 | 0.9281 | 126.80 | 443.90 |
| 7 | ROBERTA | AGNEWS | humanizer_roberta | 50 | 49 | **98.0** | 0.7948 | 0.6154 | 0.9848 | 0.1363 | 0.9661 | 65.75 | 329.00 |
| 8 | ROBERTA | HC3 | humanizer_roberta | 50 | 49 | **98.0** | 0.7697 | 0.5260 | 0.9947 | 0.5082 | 0.9845 | 42.04 | 195.75 |
| 9 | ROBERTA | M4 | humanizer_roberta | 50 | 41 | 82.0 | 0.7120 | 0.5887 | 0.7845 | 1.1589 | 0.9882 | 43.71 | 240.82 |
| 10 | ROBERTA | QNLI | humanizer_roberta | 50 | 50 | **100.0** | 0.8404 | 0.7777 | 0.9453 | 0.4824 | 0.9729 | 101.37 | 492.26 |
| 11 | ROBERTA | RTE | humanizer_roberta | 50 | 33 | 66.0 | 0.8532 | 0.8473 | 0.8772 | −0.5322 | 0.9809 | 524.37 | 844.91 |
| 12 | ROBERTA | SST2 | humanizer_roberta | 50 | 47 | 94.0 | 0.8186 | 0.7205 | 0.9383 | −9.6273 | 0.9491 | 303.13 | 500.37 |

### BERT-base vs Paper (Character-Level Methods Only)

Comparison of leading character-level methods against CSBP v3. ASR is the primary metric; cos and PPL reported for CSBP only.

| Dataset | DWB | TextBugger | Pruthi | Charmer-Fast | Charmer | **CSBP(Ours)** | Charmer cos | **CSBP cos** | **CSBP PPL** | **CSBP BPE** |
|---------|-----|------------|--------|--------------|---------|----------|-------------|--------------|--------------|--------------|
| AG-News | 60.51 | 50.85 | 90.02 | 95.86 | 98.51 | **98.0** | 0.95 | **0.97**  | 66.3 | 0.96 |
| QNLI | 71.57 | 75.77 | 17.70 | 94.69 | 97.68 | 86.0 | 0.94 | **0.99** | 133.7 | 0.93 |
| RTE | 65.67 | 74.13 | 62.19 | 89.55 | 97.01 | 56.0 | 0.86 | **0.98** | 262.3  | 0.95 |
| SST-2 | 81.39 | 68.49 | 90.94 | 100.00 | 100.00 | **98.0** | 0.90 | **0.93**  | 126.8 | 0.96 |
| HC3 | — | — | — | — | — | **98.0** | — | **0.98** | 34.0 | 0.99 |
| M4 | — | — | — | — | — | 82.0 | — | **0.99** | 45.9 | 0.79 |

### RoBERTa-base vs Paper (Character-Level Methods Only)

| Dataset | DWB | TextBugger | Pruthi | Charmer-Fast | Charmer | **CSBP(Ours)** | Charmer cos | **CSBP cos** | **CSBP PPL** | **CSBP BPE** |
|---------|-----|------------|--------|--------------|---------|----------|-------------|--------------|--------------|--------------|
| AG-News | 56.81 | 51.21 | 88.91 | 91.87 | 96.88 | **98.0** | 0.95 | **0.97** | 65.8 | 0.98 |
| QNLI | 64.34 | 67.39 | 17.45 | 96.95 | 97.86 | **100.0** | 0.92 | **0.97** | 101.4 | 0.95 |
| RTE | 62.67 | 71.43 | 49.31 | 91.71 | 97.24 | 66.0 | 0.85 | **0.98** | 524.4 | 0.88 |
| SST-2 | 84.27 | 61.10 | 92.93 | 99.39 | 99.51 | 94.0 | 0.89 | **0.95** | 303.1 | 0.94 |
| HC3 | — | — | — | — | — | **98.0** | — | **0.98** | 42.0 | 0.99 |
| M4 | — | — | — | — | — | 82.0 | — | **0.99** | 43.7 | 0.78 |

> **Note:** Our CSBP variant maintains a notably higher semantic similarity (cos) than existing state-of-the-art across evaluated datasets, while achieving maximum fragmentation over BPE bounds. CSBP achieves 100% ASR on RoBERTa-QNLI — the only method to do so.

---

## 🧠 Architecture Overview

### Stage 1 — Register Gate (`humanizer.py`)

Before attacking, text is formally classified using `all-mpnet-base-v2` embeddings paired with an internal logistic regression classifier. The predicted register (formal/informal) governs internal parameter flows — controlling emoji pool selection and evasion intensity — while natively preserving the semantic baseline intent.

The `humanize()` function also applies **adaptive intensity scaling** based on word count: texts over 100 words receive a 2× search budget boost; texts between 30–100 words receive a 1.5× boost. This maximises ASR on long-form datasets (HC3, M4) without relaxing semantic quality gates. RoBERTa targets additionally receive a patience extension (12 rounds vs. 9) to accommodate deeper byte-BPE search.

### Stage 2 — Vulnerability Pre-Analysis (`composite_scorer.py`)

A single GPT-2 forward pass explicitly maps token confidences down to byte BPE level to identify attack priorities:

- **Priority Words:** Token positions with rank ≤ 5 (highest classifier confidence).
- **Watermark Words:** Token positions falling within KGW green-list spans (using SHA-256-seeded deterministic sampling at γ=0.5).
- **Word Sensitivity Map:** Per-word averaged rank scores, used to weight the probabilistic sampling of attack targets in CSBP.

All analysis is performed once per original text and cached, making subsequent beam rounds fast.

### Stage 3 — CSBP v3 Model-Specific Evasion Strategies (`csbp_loop.py`)

The pipeline natively exploits the disparate vocab architectures between BERT (WordPiece/cased) and RoBERTa (case-sensitive byte-level BPE). When `arch='roberta'` is passed, the strategy pool is further weighted toward byte-level OOD techniques.

**Core v3 Strategies:**

| Strategy | Description | Weight |
|----------|-------------|--------|
| `blitz` | Deep multi-layer homoglyph + ZWSP stacking; highest ASR across both architectures | 4× |
| `zwsp_flood` | Inserts ZWSP between every character after full homoglyph substitution; maximum token fragmentation | 4× |
| `roberta_subword` | Targets long words (>5 chars) with internal ZWSP injection + 30% homoglyph rate; exploits RoBERTa's isolated byte-BPE constraints | 4× |
| `scorched_earth` | Full pipeline: homoglyph (90%) → diacritic stacking (70%) → heavy invisible insertion → optional emoji | 3× |
| `micro_sponge` | Appends 5–15 invisible BPE-breaking chars at sentence end to displace document aggregation metrics | 2× |
| `punctuation_bind` | Injects ZWSP before/after `.`, `,`, `!`, `?`, `;`, `:` to shatter inter-sentence token contexts | 2× |
| `random_case` | Random interior case flipping; creates novel byte sequences for RoBERTa's case-sensitive tokenizer | 2× |
| `math_unicode` | Substitutes Latin chars with Mathematical Bold Unicode (4-byte UTF-8); OOD for both tokenizers | 2× |
| `fullwidth` | Maps ASCII printables to fullwidth variants (3-byte UTF-8); disrupts byte-BPE without visual change | 1× |
| `bpe_break` | Exhaustive internal boundary checking for maximal token-inflation perturbations | 1× |
| `high_conf` | Directly targets priority words identified in pre-analysis using BPE-targeted or blitz attacks | 1× |
| `watermark` | Targets KGW green-list tokens; selects perturbation maximising token-count delta | 1× |
| `combined` | Multi-strategy pass: BPE-break on heavy words + emoji attach on untouched words + ZWSP fill | 1× |

Candidate selection within each strategy is **sensitivity-weighted**: words with higher pre-analysis confidence scores are sampled proportionally, ensuring attack budget concentrates on the most classifier-relevant tokens.

### Stage 4 — Evasion-Oriented Scoring Engine (`composite_scorer.py`)

The composite score `S` is a weighted combination of six components. When a real classifier is wired in, weights shift to prioritise classifier confidence (ground-truth signal):

| Component | No Classifier | With Classifier | Description |
|-----------|--------------|-----------------|-------------|
| Classifier evasion | 0.00 | 0.45 | `1 - P(original_label \| attacked)` |
| GPT-2 evasion | 0.35 | 0.10 | Sigmoid over PPL and avg rank vs. human baselines |
| BPE disruption | 0.35 | 0.25 | Token inflation (60%) + invisible char density (40%) |
| Watermark evasion | 0.10 | 0.08 | Sigmoid inverse of KGW z-score |
| Readability | 0.05 | 0.04 | Penalises suspiciously low PPL (<30) and extreme PPL (>10k) |
| Semantic similarity | 0.15 | 0.08 | SBERT cosine after stripping invisible chars and reversing homoglyphs |

A **coherence gate** (multiplicative, floor 0.10) discards true gibberish without hard-blocking borderline attacks. Cosine similarity is computed on a cleaned representation — stripping ZWSP/ZWNJ, normalising diacritics, and reversing Cyrillic/Greek homoglyphs — so the score reflects true semantic preservation rather than tokenizer fragmentation artefacts.

---

## 🛠️ Usage & Execution

### Installation

```bash
pip install torch sentence-transformers transformers scikit-learn pandas tqdm aiohttp pyarrow joblib nltk emoji
```

### Direct Inference (Single Text)

```bash
python characterlevelattacks/coreattacks/humanizer.py "Your target AI-text here." \
    --iterations 7 \
    --beam-width 7 \
    --cands 20 \
    --device mps
```

Attack a specific fine-tuned classifier:

```bash
python characterlevelattacks/coreattacks/humanizer.py "Your target AI-text here." \
    --target-model textattack/bert-base-uncased-ag-news \
    --iterations 7 \
    --beam-width 7 \
    --cands 20 \
    --device mps
```

Auto-select model by dataset and architecture:

```bash
python characterlevelattacks/coreattacks/humanizer.py "Your target AI-text here." \
    --dataset agnews \
    --arch roberta \
    --iterations 7 \
    --beam-width 7 \
    --cands 20
```

### Dataset Batches

```bash
python characterlevelattacks/coreattacks/humanizer.py \
    --hc3 path/to/hc3.parquet \
    --m4  path/to/m4.parquet \
    --sample-size 100 \
    -o results.csv
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `text` | — | Input text string or path to a text/CSV/Parquet file |
| `--target-model` | None | HuggingFace model ID to attack (required for BERT/RoBERTa ASR) |
| `--dataset` | `agnews` | Dataset name for auto model selection |
| `--arch` | `bert` | Architecture: `bert` or `roberta` |
| `--iterations` | `7` | Max beam search rounds (K) |
| `--beam-width` | `8` | Number of beams kept per round |
| `--cands` | `20` | Candidates generated per beam per round |
| `--device` | auto | `cpu`, `mps`, or `cuda` |
| `--sample-size` | `10` | Number of texts to sample from dataset files |
| `-o` / `--output` | `~/Downloads/humanized_output.csv` | Output CSV path |
| `--hc3` | — | Path to HC3 Parquet file |
| `--m4` | — | Path to M4 Parquet file |

### Supported Model Registry

The following TextAttack fine-tuned models are pre-registered for auto-selection via `--dataset` + `--arch`:

| Dataset | BERT | RoBERTa |
|---------|------|---------|
| SST-2 | `textattack/bert-base-uncased-SST-2` | `textattack/roberta-base-SST-2` |
| QNLI | `textattack/bert-base-uncased-QNLI` | `textattack/roberta-base-QNLI` |
| RTE | `textattack/bert-base-uncased-RTE` | `textattack/roberta-base-RTE` |
| AG-News | `textattack/bert-base-uncased-ag-news` | `textattack/roberta-base-ag-news` |
| HC3 / M4 | `Hello-SimpleAI/chatgpt-detector-roberta` | `Hello-SimpleAI/chatgpt-detector-roberta` |

### Programmatic API

```python
from characterlevelattacks.coreattacks.humanizer import humanize

result = humanize(
    text         = "Your AI-generated text here.",
    target_model = "textattack/bert-base-uncased-ag-news",
    iterations   = 7,
    n_candidates = 20,
    beam_width   = 8,
    verbose      = True,
)

print(result['success'])         # True if classifier label flipped
print(result['humanized_text'])  # Perturbed output
print(result['score_breakdown']) # Full metric breakdown
```

For NLI datasets (RTE, QNLI, MNLI) where the model expects a premise–hypothesis pair:

```python
result = humanize(
    text         = "The hypothesis text.",
    premise      = "The premise text.",
    target_model = "textattack/bert-base-uncased-RTE",
    iterations   = 7,
)
```

Statistical-only mode (no classifier — optimises GPT-2 perplexity for commercial detectors like ZeroGPT/Originality.ai):

```python
result = humanize(
    text       = "Your AI-generated text here.",
    iterations = 7,
)
```

---

## 📊 Metric Glossary

| Metric | Description |
|--------|-------------|
| **ASR (%)** | Attack Success Rate — % of texts for which the classifier label flipped |
| **Avg S** | Mean composite score across all samples (higher = better overall attack quality) |
| **Avg Evasion** | GPT-2 statistical evasion score (PPL + rank sigmoid) |
| **Avg BPE Disrupt** | Token inflation + invisible char density score |
| **Avg WM z-score** | KGW watermark z-score on attacked text (lower = better watermark evasion) |
| **Avg Cosine** | SBERT semantic similarity between original and attacked text (computed on cleaned representation) |
| **Avg PPL** | GPT-2 perplexity of attacked text |
| **Avg Rank** | Mean GPT-2 next-token prediction rank across attacked tokens |

---

## 🔬 CSBP v3 vs. Charmer: A Full Technical Comparison

Charmer (Abad Rocamora et al., ICML 2024) is the direct state-of-the-art baseline this framework is evaluated against, using the identical TextAttack fine-tuned BERT/RoBERTa model suite on the same five benchmark datasets. The two approaches share the same top-level goal — black-box character-level adversarial attack with high semantic preservation — but diverge fundamentally in their threat model, perturbation theory, search algorithm, scoring, and scope.

---

### 1. Threat Model and Target

**Charmer** operates within a strict Levenshtein-distance threat model. Every perturbation is a standard Unicode character from the dataset's own alphabet — no homoglyphs, no invisible characters, no out-of-alphabet insertions. The constraint `dlev(S, S') ≤ k` is the only formal bound on perturbation magnitude, and the alphabet Γ is explicitly restricted to characters already present in each evaluation dataset. The resulting adversarial examples look like plausible typos or minor punctuation insertions to a human reader.

**CSBP v3** operates under a different threat model entirely. The attack is not constrained by Levenshtein distance. Instead, perturbations are drawn from a purpose-built library of Unicode-level disruptions — Cyrillic/Greek homoglyphs, diacritic variants, zero-width spaces (ZWSP/ZWNJ/ZWJ), Mathematical Bold codepoints, and fullwidth ASCII — that are visually identical or near-identical to the original text but occupy completely different byte sequences. The threat is not "how many characters changed" but "how severely can byte-level tokenizer assumptions be violated while remaining invisible to a human reader." These are categorically different threat models: Charmer attacks through edit distance minimality; CSBP attacks through Unicode tokenizer exploitation.

---

### 2. Perturbation Space

Charmer's perturbation space is the set S_k(S, Γ) — all sentences reachable from S within k standard character insertions, deletions, or replacements drawn from the dataset alphabet. Its most common successful operations (per Appendix B.2 of the paper) are insertions of punctuation characters such as `)`, `.`, `'`, `"`, `$`, `%`, and `?`. These operations are visible, modify the rendered text slightly, and are detectable by typo-correctors if Pruthi-Jones constraints are enforced.

CSBP's perturbation space is disjoint from Charmer's. The core operations are:

- **Homoglyph substitution**: replacing `a` with Cyrillic `а` (U+0430), `e` with `е` (U+0435), etc. The substituted character is visually indistinguishable in virtually all fonts, but occupies a different Unicode code point, producing a completely different token or byte sequence for both BERT's WordPiece and RoBERTa's byte-level BPE tokenizer. The homoglyph map covers all 26 lowercase and 26 uppercase Latin letters with Cyrillic, Greek, or IPA equivalents (52 substitutions total).
- **Diacritic stacking with ZWSP injection**: replacing `e` with `é` and immediately inserting ZWSP (U+200B) after it. BERT (uncased) strips diacritics via NFD normalisation, but ZWSP is Unicode category Cf (format character), not Mn (combining mark), so it survives the normalisation pipeline. The surviving ZWSP then shatters BPE boundaries, splitting what was a single token into multiple pre-token chunks.
- **Direct ZWSP flooding**: inserting ZWSP at every inter-character boundary in a word after homoglyph substitution, producing maximal token inflation without any visible change to the rendered string.
- **Math Bold Unicode**: substituting Latin letters with Mathematical Bold variants (U+1D400–U+1D7FF). These are 4-byte UTF-8 sequences that render identically to bold Latin but are completely OOV for both tokenizers' standard vocabularies.
- **Fullwidth ASCII**: mapping printable ASCII to fullwidth variants (U+FF01–U+FF5E). These are 3-byte UTF-8 per character and equally OOV.
- **Micro-sponge**: appending 5–15 invisible format characters at sentence end to perturb document-level aggregation metrics (e.g., KGW z-score, GPT-2 avg-rank).

None of these operations increase Levenshtein distance by standard definitions because most involve Unicode characters that render as zero-width or visually identical. A typo-corrector that operates on visible characters cannot detect or correct them, because from the corrector's perspective the text is unchanged.

---

### 3. Position / Word Selection

**Charmer** selects attack positions through Algorithm 1: it replaces each character position with a fixed test character (whitespace U+0020) and evaluates the resulting loss. The top-n positions by loss differential are selected as candidates for perturbation. This is a character-position importance score, computed with O(|S|) model forward passes per iteration. After a perturbation is applied, importances are re-evaluated in the next round, meaning Charmer dynamically updates its position priorities as the attack evolves.

**CSBP v3** selects attack targets at the word level using a pre-computed sensitivity map derived from GPT-2 token rank analysis (`analyze_original()` in `composite_scorer.py`). Words are ranked by their average next-token prediction rank — high-confidence tokens (rank ≤ 5) are flagged as priority words; KGW green-list tokens are flagged as watermark words. The sensitivity map is computed once before the beam search begins and is fixed for all rounds, which amortises the pre-analysis cost across all K iterations. Within each candidate generation step, words are sampled proportionally to their sensitivity scores, ensuring attack budget concentrates on the tokens most responsible for the original classification signal. Charmer's per-round importance re-evaluation adapts to perturbation interactions; CSBP's fixed sensitivity map is cheaper per round but static.

---

### 4. Search Algorithm

**Charmer** uses a greedy single-character perturbation loop. At each iteration k, it evaluates all possible single-character modifications at the top-n selected positions (n × |Γ| candidates), picks the one with the highest Carlini-Wagner loss, and applies it. The process repeats for up to k iterations. It is deterministic and produces a single output trajectory per input — the greedily optimal path through single-edit space. Beam width is implicitly 1.

**CSBP v3** uses a multi-strategy stochastic beam search. At each of K rounds, every active beam spawns `n_candidates × 6` candidate perturbations drawn from the 13-strategy pool using weighted random sampling. Each strategy applies word-level transformations (not single characters) potentially modifying multiple words simultaneously. The top `beam_width` candidates by composite score are retained as the next beam population. This produces a diverse set of concurrent attack trajectories rather than a single greedy path, and the stochastic sampling means different runs may explore different regions of the perturbation space. Early stopping triggers if the top beam score fails to improve for `patience` rounds (9 for BERT, 12 for RoBERTa).

The key structural difference: Charmer is a character-level greedy search with importance re-evaluation; CSBP is a word-level stochastic beam search with a pre-computed sensitivity prior and multi-strategy candidate diversification.

---

### 5. Objective / Scoring Function

**Charmer** uses the Carlini-Wagner loss directly as its search objective:

```
L(f(S'), y) = max_{ŷ ≠ y} f(S')_ŷ − f(S')_y
```

A candidate is accepted as a success when this loss crosses zero (i.e., the classifier's top prediction changes). There is no secondary objective: Charmer optimises purely for misclassification, with semantic similarity enforced as a post-hoc reporting metric rather than as a term in the search objective.

**CSBP v3** optimises a six-component composite score `S` that simultaneously targets misclassification, BPE disruption, watermark evasion, readability, and semantic similarity. The weights shift depending on whether a real classifier confidence function is available. When one is wired in, classifier evasion (`1 - P(original_label | attacked)`) receives 45% weight — the largest single component, analogous to Charmer's CW loss — but BPE disruption (25%), GPT-2 statistical evasion (10%), watermark evasion (8%), and semantic similarity (8%) are jointly optimised alongside it. A multiplicative coherence gate (floor 0.10) prevents true gibberish from ranking highly regardless of evasion score. This multi-objective formulation means CSBP simultaneously optimises for evasion across multiple detector types (fine-tuned classifiers, statistical detectors, watermark detectors) in a single pass, whereas Charmer targets only the single attached classifier.

---

### 6. Semantic Similarity Handling

Both frameworks use cosine similarity of sentence embeddings as the semantic preservation metric, but handle it differently.

Charmer measures USE (Universal Sentence Encoder) cosine similarity as a reporting metric. It is not part of the search objective — Charmer simply reports how similar the final adversarial example is to the original. Charmer's high similarity scores (typically 0.86–0.95 in the paper's Table 2) arise naturally from the fact that single-character typo-style edits preserve word-level semantics well.

CSBP v3 includes SBERT cosine similarity as a component of the composite score with 8–15% weight depending on mode. Critically, similarity is computed on a **cleaned representation** of the attacked text — invisible Unicode characters (ZWSP/ZWNJ/ZWJ) are stripped, diacritics are normalised via NFKD, and Cyrillic/Greek homoglyphs are mapped back to their Latin equivalents before encoding. This ensures the similarity score reflects true semantic preservation at the meaning level rather than being inflated by the fact that the attacked string looks identical visually. Without this cleaning step, homoglyph-substituted text would score artificially low because SBERT tokenises it differently from the original. The cleaning pipeline is implemented in `cosine_score()` in `composite_scorer.py`. CSBP's cosine scores (0.93–0.99 across datasets) are consistently higher than Charmer's, despite CSBP applying far more aggressive per-character perturbations, because the invisible-character attacks preserve every visible word intact.

---

### 7. Architecture Specificity

**Charmer** is architecture-agnostic by design. The same algorithm — test character probing + greedy CW-loss maximisation — applies identically to BERT, RoBERTa, ALBERT, and LLMs (Llama 2, Vicuna). The alphabet Γ is taken directly from each evaluation dataset, so no architecture-specific knowledge is encoded in the attack.

**CSBP v3** is architecture-aware. It maintains two distinct strategy pools — one for BERT (WordPiece, cased) and one for RoBERTa (byte-level BPE, case-sensitive) — and selects between them at runtime via the `arch` parameter. When targeting RoBERTa, the strategy pool is augmented with additional weight on `random_case` (exploiting case-sensitivity of byte-BPE), `math_unicode` (4-byte UTF-8 sequences that fall outside RoBERTa's BPE vocabulary), `fullwidth` (3-byte per character, OOD for byte-level BPE), and `roberta_subword` (ZWSP injection targeting long-word BPE splits). The `humanize()` function also extends patience to 12 rounds for RoBERTa targets and applies `zwsp_shatter_word` with ZWSP at every third character boundary for short texts. This architecture specificity is a deliberate design choice: because the two tokenizer families have different fragmentation vulnerabilities (BERT's WordPiece merges on subword frequency; RoBERTa's byte-BPE merges on raw byte pairs), the most effective disruption operations differ between them.

---

### 8. Scope: What CSBP Targets Beyond Charmer

Charmer targets a single fine-tuned classifier. Its success criterion is label flip against that classifier, and its entire search objective is optimised for that one signal. It does not address watermark detectors, GPT-2 statistical detectors (ZeroGPT, Originality.ai, DetectGPT), or AI-text classifiers such as HC3 or M4.

CSBP v3 targets all of these simultaneously. The pre-analysis stage identifies KGW green-list tokens and directly targets them with the `watermark` strategy. The GPT-2 evasion component and BPE disruption component jointly push the attacked text into the high-PPL, high-rank, high-token-inflation region that defeats statistical detectors. The evaluation covers HC3 and M4 (AI-text detection datasets with `Hello-SimpleAI/chatgpt-detector-roberta` as the proxy classifier) in addition to the four TextAttack classification benchmarks, achieving 98% ASR on HC3 for both BERT and RoBERTa modes.

---

### 9. Evaluation Protocol Differences

Charmer evaluates on up to 1,000 test samples per dataset using a single A100 GPU, reporting ASR, mean Levenshtein distance, and USE cosine similarity. Beam width is implicitly 1 (greedy); k (max edits) is 10 for most datasets and 20 for AG-News.

CSBP v3 evaluates on 50 samples per dataset configuration (12 configurations total), run on Apple Silicon MPS via `detector_eval.py`. The evaluation computes ASR plus six additional metrics per example (composite score S, evasion, BPE disruption, watermark z-score, cosine, PPL, avg rank) using the full `composite_score()` pipeline on every sample — not just successful flips. Search hyperparameters: K=7 iterations, beam width=8, 20 candidates per beam per round (adaptive scaling applied for longer texts).

---

### 10. Summary Comparison

| Dimension | Charmer | CSBP v3 |
|-----------|---------|---------|
| **Threat model** | Levenshtein distance ≤ k over dataset alphabet | Invisible Unicode / BPE boundary exploitation; no edit-distance constraint |
| **Perturbation type** | Standard character insertions, deletions, replacements (typo-style) | Homoglyphs, ZWSP/ZWNJ flooding, diacritic stacking, math Unicode, fullwidth ASCII |
| **Visibility** | Slightly visible (punctuation insertions, character swaps) | Visually zero — all core operations render identically to original |
| **Position selection** | Per-round importance re-evaluation via test-character probing (O(\|S\|) queries/round) | One-time GPT-2 rank analysis; fixed per-word sensitivity map |
| **Search** | Greedy single-character, beam width = 1, deterministic | Stochastic multi-strategy beam search, beam width = 8, 13 strategy pool |
| **Objective** | Carlini-Wagner loss (misclassification only) | Six-component composite: classifier + BPE disruption + evasion + watermark + readability + similarity |
| **Semantic similarity** | USE cosine (post-hoc reporting metric) | SBERT cosine on cleaned representation (active optimisation component) |
| **Architecture specificity** | Architecture-agnostic | Dual-mode: BERT (WordPiece) vs. RoBERTa (byte-BPE) strategy pools |
| **Detector scope** | Single fine-tuned classifier | Fine-tuned classifiers + statistical detectors + KGW watermark + AI-text classifiers |
| **Datasets** | SST-2, QNLI, MNLI, RTE, AG-News | SST-2, QNLI, RTE, AG-News, HC3, M4 |
| **Defensibility** | Defeated by typo-correctors only under PJC constraints | Immune to typo-correctors (homoglyphs/ZWSP are not standard misspellings) |
