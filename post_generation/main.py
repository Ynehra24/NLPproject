"""
main.py
-------
Entry point for the Stylometry-Aware Differentiable Paraphraser.

CLI modes
=========
  train    — Build human stats, build evader, run training loop.
  evaluate — Load a trained checkpoint, run full evaluation suite.
  paraphrase — Quick interactive inference.

Example usage
=============
# Training (adjust paths in config.py or override via env vars)
python main.py train

# Evaluation after training
python main.py evaluate \
    --checkpoint outputs/checkpoints/best \
    --eval_file  data/eval_ai_text.txt \
    --results    outputs/eval_results.json

# Quick paraphrase (interactive)
python main.py paraphrase --checkpoint outputs/checkpoints/best
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import Config, get_default_config
from data_utils import build_or_load_human_stats
from evader import StyleAwareEvader
from trainer import EvaderTrainer
from evaluator import EvaderEvaluator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Surrogate detector loader
# ---------------------------------------------------------------------------

def load_surrogate_detector(config: Config):
    """
    Load the white-box surrogate detector.

    This expects a fine-tuned binary classifier checkpoint at
    config.training.* or a HuggingFace model ID.

    IMPORTANT: You must fine-tune a RoBERTa (or other) classifier on your
    AI vs. Human dataset BEFORE running this.  This codebase does NOT train
    the detector — per the problem statement, that is left to the researcher.

    Checkpoint format: any HuggingFace AutoModel directory.
    """
    model_name = config.model.surrogate_detector_name
    logger.info("Loading surrogate detector: %s", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# ---------------------------------------------------------------------------
# mode: train
# ---------------------------------------------------------------------------

def run_train(config: Config) -> None:
    set_seed(config.training.seed)

    # ---- Surrogate detector (you supply this) ----
    surrogate_model, _ = load_surrogate_detector(config)

    # ---- Human corpus statistics ----
    cache_path = os.path.join(config.training.output_dir, "human_stats.json")
    # We use the EVADER's tokenizer to measure sentence lengths
    from transformers import BartTokenizer
    evader_tok = BartTokenizer.from_pretrained(config.model.evader_model_name)

    human_stats = build_or_load_human_stats(
        human_corpus_path=config.training.human_corpus_path,
        cache_path=cache_path,
        tokenizer=evader_tok,
        max_sentence_length=config.loss.max_sentence_length,
        n_bins=config.loss.n_length_bins,
        sample_size=config.training.human_corpus_sample_size,
    )

    # ---- Build evader ----
    evader = StyleAwareEvader(
        config=config,
        surrogate_model=surrogate_model,
        human_stats=human_stats,
        sentence_encoder=None,  # Set to a SentenceTransformer to enable L_semantic
    )

    # ---- Train ----
    trainer = EvaderTrainer(config=config, evader_model=evader)
    trainer.train()


# ---------------------------------------------------------------------------
# mode: evaluate
# ---------------------------------------------------------------------------

def run_evaluate(config: Config, checkpoint_path: str, results_path: str) -> None:
    set_seed(config.training.seed)

    # ---- Load evader ----
    from evader import StyleAwareEvader
    from transformers import BartTokenizer, BartForConditionalGeneration

    evader_backbone = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    evader_tok = BartTokenizer.from_pretrained(checkpoint_path)
    device = torch.device(config.device)
    evader_backbone.to(device).eval()

    # ---- Load surrogate detector ----
    surrogate_model, surrogate_tok = load_surrogate_detector(config)

    # ---- Load black-box detectors ----
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'detector_evaluation')))
    from detectors.binoculars.score import BinocularsDetector
    from detectors.fast_detectgpt.score import FastDetectGPTDetector

    blackbox_detectors = []
    
    logger.info("Loading Fast-DetectGPT detector (using gpt2 to avoid slow downloads)")
    fast_detect_gpt = FastDetectGPTDetector(scoring_model="gpt2", reference_model="gpt2", device=config.device)
    blackbox_detectors.append(fast_detect_gpt)

    logger.info("Loading Binoculars detector (using gpt2 to avoid slow downloads)")
    binoculars = BinocularsDetector(observer_model="gpt2", performer_model="distilgpt2", device=config.device)
    blackbox_detectors.append(binoculars)
    
    # Optional standard models from config
    for bb_path in config.eval.blackbox_detector_paths:
        logger.info("Loading black-box detector: %s", bb_path)
        bb_model = AutoModelForSequenceClassification.from_pretrained(bb_path)
        bb_tok = AutoTokenizer.from_pretrained(bb_path)
        blackbox_detectors.append((bb_model, bb_tok))

    # ---- Load human stats ----
    cache_path = os.path.join(config.training.output_dir, "human_stats.json")
    human_stats = None
    if os.path.exists(cache_path):
        from data_utils import HumanCorpusStats
        human_stats = HumanCorpusStats.load(cache_path, tokenizer=evader_tok)

    # ---- Load eval texts ----
    with open(config.training.eval_data_path) as f:
        eval_texts = [line.strip() for line in f if line.strip()]

    eval_texts = eval_texts[: config.eval.eval_sample_size]
    logger.info("Evaluating on %d samples", len(eval_texts))

    # ---- Paraphrase ----
    logger.info("Generating paraphrases…")
    paraphrases: list = []
    for i in range(0, len(eval_texts), 16):
        batch = eval_texts[i: i + 16]
        enc = evader_tok(
            batch,
            max_length=config.model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            gen_ids = evader_backbone.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=config.model.num_beams,
                max_length=config.model.max_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        decoded = evader_tok.batch_decode(gen_ids, skip_special_tokens=True)
        paraphrases.extend(decoded)

    # ---- Evaluate ----
    evaluator = EvaderEvaluator(
        surrogate_model=surrogate_model,
        surrogate_tok=surrogate_tok,
        blackbox_models=blackbox_detectors,
        human_stats=human_stats,
        human_label_idx=config.model.human_label_idx,
        device=config.device,
        bertscore_model=config.eval.bertscore_model,
    )
    results = evaluator.evaluate(eval_texts, paraphrases)
    evaluator.save_results(results, results_path)


# ---------------------------------------------------------------------------
# mode: paraphrase (interactive)
# ---------------------------------------------------------------------------

def run_paraphrase(config: Config, checkpoint_path: str) -> None:
    from transformers import BartTokenizer, BartForConditionalGeneration

    device = torch.device(config.device)
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path).to(device).eval()
    tok = BartTokenizer.from_pretrained(checkpoint_path)

    print("\nStyleAwareEvader — Interactive Paraphrase Mode")
    print("Type your AI-generated text and press Enter twice to paraphrase.")
    print("Type 'quit' to exit.\n")

    while True:
        lines = []
        while True:
            line = input()
            if line.lower() == "quit":
                sys.exit(0)
            if line == "":
                break
            lines.append(line)

        text = " ".join(lines).strip()
        if not text:
            continue

        enc = tok(
            text,
            max_length=config.model.max_length,
            return_tensors="pt",
            truncation=True,
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                enc["input_ids"],
                num_beams=config.model.num_beams,
                max_length=config.model.max_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        output = tok.decode(gen_ids[0], skip_special_tokens=True)
        print(f"\n[Paraphrase]\n{output}\n")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stylometry-Aware Differentiable Paraphraser"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # train
    sub.add_parser("train", help="Train the evader.")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="Evaluate a trained evader.")
    eval_p.add_argument("--checkpoint", required=True, help="Path to saved evader checkpoint.")
    eval_p.add_argument("--eval_file", help="Override eval data path.")
    eval_p.add_argument("--results", default="outputs/eval_results.json")

    # paraphrase
    para_p = sub.add_parser("paraphrase", help="Interactive paraphrase mode.")
    para_p.add_argument("--checkpoint", required=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = get_default_config()

    if args.mode == "train":
        run_train(config)

    elif args.mode == "evaluate":
        if hasattr(args, "eval_file") and args.eval_file:
            config.training.eval_data_path = args.eval_file
        run_evaluate(config, args.checkpoint, args.results)

    elif args.mode == "paraphrase":
        run_paraphrase(config, args.checkpoint)


if __name__ == "__main__":
    main()
