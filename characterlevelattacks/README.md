# Adv-Humanizer: A Robust Adversarial Attack Framework 🛡️✍️

Adv-Humanizer is a highly modular, character-level adversarial attack framework specifically engineered to "humanize" AI-generated text. It leverages a multi-stage mutation engine guided by semantic and stylistic optimization to bypass advanced discriminators (BERT, RoBERTa, GPTZero).

---

## 📂 Directory Structure

```text
NLPproject/
├── characterlevelattacks/         # Main Project Root
│   ├── coreattacks/               # Core Logic & Models
│   │   ├── formality_model/       # Trained Register Gate (LogReg + MPNet)
│   │   ├── attacked_outputs/      # Logs of humanized results
│   │   ├── humanizer.py           # Unified Entry Point (The Optimizer)
│   │   ├── composite_scorer.py    # Scoring Engine (SBERT + GPT2)
│   │   ├── emoji_insertion.py     # Register-aware emoji logic
│   │   ├── homoglyph_attack.py    # Character substitution logic
│   │   ├── detector_eval.py       # ASR & Evasion Evaluation suite
│   │   └── hc3_m4_attack.py       # Specialized HC3/M4 data loaders
│   ├── humanvsai/                 # Datasets (HC3, M4, Enron)
│   ├── emojibased/                # Datasets for style/emoji extraction
│   ├── stylometric/               # Stylistic features & analysis utilities
│   ├── parquetcombiner.py         # Data augmentation & merging utility
│   └── datasetpreview.py          # Parquet exploration tool
├── README.md                      # Comprehensive Documentation
```

---

## 🧠 Technical Architecture

### 1. The Register Gate (Formality Classifier)
Before humanizing, the text passes through a **formality discriminator**.
- **Model**: `all-mpnet-base-v2` embeddings + 11 stylistic features (emoji density, slang frequency, etc.).
- **Purpose**: Dynamically switches the attack strategy between `FORMAL` and `INFORMAL` modes to ensure style-congruent perturbations.

### 2. The Mutation Engine
The pipeline applies four distinct layers of adversarial noise:
- **Synonym Swaps**: Contextual replacement via NLTK/WordNet.
- **Visual Homoglyphs**: Swapping Latin characters (e.g., `a`) with Cyrillic equivalents (e.g., `а`).
- **Deep Token Incineration**: Inserting invisible Zero-Width Spaces (ZWSP) to "scramble" the BPE tokenization used by RoBERTa/BERT.
- **Structural Burstiness**: Intentionally varying sentence length and joining structures to disrupt AI-standard "flat" statistical patterns.

### 3. Optimization Logic (The S-Score)
Every candidate is evaluated against a 5-term composite score **S**:
- **Semantic Similarity (35%)**: Ensures the meaning is preserved (`all-mpnet-base-v2`).
- **Fluency/Perplexity (25%)**: Ensures the text is readable by humans (`gpt2-small`).
- **Levenshtein Distance (15%)**: Minimizes visual divergence.
- **Jaccard Index (15%)**: Monitors word-level stability.
- **Stylometric Delta (10%)**: Matches the structural "texture" of the input.

---

## 🛠️ Installation

Ensure you have Python 3.12+ and the required libraries:
```bash
pip install torch sentence-transformers transformers scikit-learn pandas tqdm nltk emoji pyarrow
```

Download NLTK resources:
```python
import nltk
nltk.download(['wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'stopwords'])
```

---

## 📖 Usage Guide

### Humanize AI Text
Generate a stealthy, humanized version of any AI prompt:
```bash
python characterlevelattacks/coreattacks/humanizer.py "Your AI Text Here"
```

### Batch Data Attack
Run the full attack suite across HC3/M4 datasets:
```bash
python characterlevelattacks/coreattacks/hc3_m4_attack.py
```

### Detector Evaluation
Run the humanized outputs through a suite of RoBERTa/BERT detectors to compute ASR:
```bash
python characterlevelattacks/coreattacks/detector_eval.py
```

---

## 📊 Models Used
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Fluency**: `openai-community/gpt2`
- **Data Loaders**: `pandas`, `pyarrow`
- **Acceleration**: Full **MPS (Metal Performance Shaders)** support for Apple Silicon.
