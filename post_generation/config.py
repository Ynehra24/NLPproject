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
    # BART shares its BPE tokenizer with RoBERTa, so pseudo-embedding
    # multiplication P @ W_emb works without any token remapping.
    evader_model_name: str = "facebook/bart-base"

    # Surrogate detector: RoBERTa fine-tuned as binary (AI vs Human) classifier.
    # This is the WHITE-BOX model used during training.
    # At eval time we also run against unseen black-box detectors.
    surrogate_detector_name: str = "Hello-SimpleAI/chatgpt-detector-roberta"

    # Maximum sequence length (tokens).  512 follows both papers.
    max_length: int = 512

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
    
    CRITICAL FIX: Reduced alpha (adversarial dominance), increased beta and gamma
    to allow semantic preservation and stylometric constraints to dominate.
    """
    alpha: float = 0.20   # adversarial loss weight (reduced from 0.40)
    beta: float = 0.40    # semantic/syntactic loss weight (increased from 0.30)
    gamma: float = 0.30   # stylometric loss weight (unchanged)
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
    # OPTIMIZED FOR M4 PRO (30-45 min training)
    # MPS memory constraint: batch_size=8 is max for frozen RoBERTa detector overhead
    # Note: gradient_accumulation=2 keeps effective batch size = 16
    # NOTE: Change these back for longer training:
    #   batch_size: 8 → 1
    #   gradient_accumulation_steps: 2 → 8
    #   num_epochs: 1 → 3
    #   human_corpus_sample_size: 2000 → 10000
    #   log_every: 50 → 2
    #   save_every: 0 → 500
    
    batch_size: int = 8  # MPS-safe: reduced from 16 (fits in 24GB VRAM)
    gradient_accumulation_steps: int = 2  # Re-enabled to maintain effective batch=16
    learning_rate: float = 5e-5
    num_epochs: int = 1  # Reduced from 3 (one epoch for quick test)
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42

    # Path to raw AI-generated training texts (one sample per line).
    train_data_path: str = "data/train_ai_text.txt"
    eval_data_path: str = "data/eval_ai_text.txt"

    # Human-corpus file used to build the stylometric baseline.
    # One document per line; sentence splitting is done at runtime.
    human_corpus_path: str = "data/human_corpus.txt"

    # How many samples to use when fitting the human baseline histogram.
    human_corpus_sample_size: int = 2000  # Reduced from 10000 (faster baseline computation)

    # Checkpoint directory.
    output_dir: str = "outputs/checkpoints"

    # Log every N steps.
    log_every: int = 50  # Reduced from 2 (less logging overhead)

    # Save checkpoint every N steps (0 = epoch-level only).
    save_every: int = 0  # Disabled intermediate checkpoints (save only at epoch end)
    
    # QUICK TESTING: Limit training samples for faster iteration.
    # Set to None for full training. Examples:
    #   500 samples → ~30-45 min on M4 Pro
    #   1000 samples → ~45-60 min on M4 Pro
    # Comment out or set to None for full dataset training.
    max_train_samples: Optional[int] = 500  # Limit to 500 samples for quick test


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
