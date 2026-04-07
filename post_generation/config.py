"""
config.py
---------
Central configuration for the Stylometry-Aware Differentiable Paraphraser.

References:
    Meng et al. (2025) GradEscape — USENIX Security 2025
    Zhou et al. (2024) HMGC — arXiv:2404.01907
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture settings."""
    # Evader: seq2seq model that paraphrases AI text.
    # Using a BART model already fine-tuned on paraphrase generation.
    # This gives a strong starting point — the model already knows how to rephrase.
    # BART shares its BPE tokenizer with RoBERTa, so pseudo-embedding
    # multiplication P @ W_emb works without any token remapping.
    evader_model_name: str = "eugenesiow/bart-paraphrase"

    # Surrogate detector: RoBERTa fine-tuned as binary (AI vs Human) classifier.
    # This is the WHITE-BOX model used during training.
    # At eval time we also run against unseen black-box detectors.
    surrogate_detector_name: str = "Hello-SimpleAI/chatgpt-detector-roberta"

    # Maximum sequence length (tokens). Reduced to 256 to fit in MPS memory.
    max_length: int = 256

    # Beam width for inference-time text generation.
    num_beams: int = 4

    # Label index for "human-written" in the detector's output head.
    human_label_idx: int = 0


@dataclass
class LossConfig:
    """
    Weights for the joint tri-objective loss:

        L_total = α · L_adv  +  β · L_sem  +  γ · L_style  +  δ · L_fluency

    L_adv   — adversarial: fool the surrogate detector into predicting "human".
    L_sem   — semantic/syntactic: keep output close to the original AI text
               (label-level cross-entropy + sentence-encoder MSE).
    L_style — stylometric: KL divergence between the soft sentence-length
               distribution of generated text and the human-corpus baseline.
    L_fluency — fluency: penalize low-entropy (repetitive) outputs.
    
    KEY INSIGHT: Semantic preservation (beta) must heavily dominate adversarial (alpha).
    The model should mostly COPY the input, with only subtle adversarial perturbations.
    If alpha is too high, the model collapses to gibberish that fools the detector.
    """
    alpha: float = 0.10   # adversarial loss weight (very conservative)
    beta: float = 0.55    # semantic/syntactic loss weight (dominant)
    gamma: float = 0.25   # stylometric loss weight
    delta: float = 0.10   # fluency loss weight (new)

    # Semantic loss sub-weights (must sum to 1.0).
    label_loss_weight: float = 0.50   # token-level label cross-entropy
    semantic_loss_weight: float = 0.50  # sentence-encoder MSE

    # Soft sentence-boundary: tokens whose high probability signals "end of sentence".
    # These are filled in at runtime from the evader tokenizer.
    # Default strings — override in TrainingConfig if needed.
    sentence_end_strings: List[str] = field(
        default_factory=lambda: [".", "!", "?", "...", ".\n"]
    )

    # Number of histogram bins for sentence-length KL divergence.
    n_length_bins: int = 25

    # Maximum sentence length considered (tokens); longer sentences are clipped.
    max_sentence_length: int = 80

    # Temperature for converting soft boundary probabilities to a histogram.
    boundary_temperature: float = 1.0

    # Small epsilon to prevent log(0) in KL computation.
    kl_eps: float = 1e-8


@dataclass
class TrainingConfig:
    """Training-loop hyperparameters."""
    # TWO-PHASE TRAINING STRATEGY:
    #   Phase 1 (Warm-Up): Standard seq2seq copy/paraphrase training.
    #     The model learns to generate coherent text that closely matches the input.
    #   Phase 2 (Adversarial): Gentle adversarial fine-tuning on top of the
    #     strong paraphrase foundation from Phase 1.
    
    batch_size: int = 4  # MPS-safe (fits in 24GB VRAM with detector overhead)
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    seed: int = 42
    
    # ----- Phase 1: Warm-Up (Reconstruction/Paraphrase SFT) -----
    warmup_epochs: int = 2           # Epochs of pure seq2seq copy training
    warmup_learning_rate: float = 3e-5  # Standard fine-tuning LR
    
    # ----- Phase 2: Adversarial Fine-Tuning -----
    num_epochs: int = 3              # Epochs of adversarial training
    learning_rate: float = 5e-6      # Much lower LR to avoid corruption
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Path to raw AI-generated training texts (one sample per line).
    train_data_path: str = "data/train_ai_text.txt"
    eval_data_path: str = "data/eval_ai_text.txt"

    # Human-corpus file used to build the stylometric baseline.
    human_corpus_path: str = "data/human_corpus.txt"

    # How many samples to use when fitting the human baseline histogram.
    human_corpus_sample_size: int = 2000

    # Checkpoint directory.
    output_dir: str = "outputs/checkpoints"

    # Log every N steps.
    log_every: int = 25

    # Save checkpoint every N steps (0 = epoch-level only).
    save_every: int = 0
    
    # Limit training samples. Set to None for full dataset.
    # 2000 samples × 5 total epochs → ~60-90 min on M4 Pro
    max_train_samples: Optional[int] = 2000


@dataclass
class EvalConfig:
    """Evaluation and metric settings."""
    # ROUGE-Lsum threshold: samples below this are considered too modified.
    rouge_threshold: float = 0.90

    # Paths to black-box detectors used ONLY at eval time (never during training).
    # Fill these in after you have trained your own detector checkpoints.
    # Format: list of local checkpoint directories OR HuggingFace model IDs.
    blackbox_detector_paths: List[str] = field(default_factory=lambda: ["roberta-base-openai-detector"])

    # Whether to run GPT-4 annotation for semantic quality (requires API key).
    run_gpt_annotation: bool = False

    # BERTScore model.
    bertscore_model: str = "roberta-large"

    # Number of evaluation samples.
    eval_sample_size: int = 100

    # Path to save evaluation results JSON.
    results_path: str = "outputs/eval_results.json"


@dataclass
class Config:
    """Top-level configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    device: str = "mps"          # "cuda" | "cpu" | "mps"
    fp16: bool = False             # mixed-precision training
    dataloader_workers: int = 0    # MPS doesn't work well with multiprocessing


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_default_config() -> Config:
    """Return a Config with all defaults populated."""
    return Config()
