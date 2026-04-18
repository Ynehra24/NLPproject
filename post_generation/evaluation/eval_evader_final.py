#!/usr/bin/env python3
"""
Evaluate trained evader model against detector_evaluation detectors.

Pipeline:
  1. Load evader checkpoint model
  2. Load phase 3 input CSV (AI-generated texts)
  3. Generate rewrites using evader
  4. Run through all detectors in detector_evaluation
  5. Report human-pass rates and metrics
"""

import os
import sys
import json
import csv
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_evader_model(checkpoint_path: str, device: str = "cpu"):
    """Load fine-tuned evader model (FLAN-T5 base)."""
    print(f"Loading evader model from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_phase3_input(input_csv: str) -> pd.DataFrame:
    """Load phase 3 input CSV with AI-generated texts."""
    print(f"Loading phase 3 input from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples")
    # Filter only AI-generated texts
    ai_df = df[df['source'] == 'ai'].copy()
    print(f"Using {len(ai_df)} AI-generated samples for rewriting")
    return ai_df


def generate_rewrites(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 256,
    num_beams: int = 1,
) -> List[str]:
    """Generate rewrites for texts using evader model."""
    print(f"\nGenerating rewrites for {len(texts)} samples...")
    
    rewrites = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating rewrites"):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize input
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=256,
                num_beams=num_beams,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                no_repeat_ngram_size=2,
            )
        
        # Decode
        batch_rewrites = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        rewrites.extend(batch_rewrites)
    
    return rewrites


def prepare_detector_input(
    ai_df: pd.DataFrame,
    rewrites: List[str],
    output_csv: str,
) -> None:
    """Prepare input CSV for detector_evaluation (rephrased texts)."""
    print(f"\nPreparing detector input CSV: {output_csv}")
    
    detector_input_rows = []
    for idx, (_, row) in enumerate(ai_df.iterrows()):
        detector_input_rows.append({
            'id': row['id'],
            'text': rewrites[idx],
            'source': 'ai',  # Source is AI (even after rewriting)
            'attack_type': 'evader_rewrite',
            'generator_model': 'evader_flan_t5_base',
        })
    
    df_detector = pd.DataFrame(detector_input_rows)
    df_detector.to_csv(output_csv, index=False)
    print(f"Saved detector input to: {output_csv}")


def run_detectors(
    input_csv: str,
    output_dir: str,
    device: str = "cpu",
    roberta_model_dir: str = None,
) -> Dict[str, pd.DataFrame]:
    """Run all detectors on the evader-rephrased texts."""
    print(f"\n{'='*80}")
    print(f"Running detector_evaluation on rephrased texts")
    print(f"{'='*80}")
    
    detector_root = PROJECT_ROOT.parent / "detector_evaluation"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build run_all.py command
    cmd = [
        sys.executable,
        "-m",
        "evaluation.run_all",
        "--input",
        input_csv,
        "--output-dir",
        output_dir,
        "--device",
        device,
    ]
    
    # Add optional roberta model
    if roberta_model_dir:
        cmd.extend(["--roberta-model-dir", roberta_model_dir])
    
    # Run all detectors
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(detector_root), check=True)
    
    print(f"Detectors completed successfully")
    return output_dir


def load_detector_scores(detector_output_dir: str) -> Dict[str, pd.DataFrame]:
    """Load scores from all detector output files."""
    print(f"\n{'='*80}")
    print(f"Loading detector scores")
    print(f"{'='*80}")
    
    detector_scores = {}
    output_path = Path(detector_output_dir)
    
    score_files = {
        'stats_baseline': output_path / 'stats_baseline_scores.csv',
        'roberta_classifier': output_path / 'roberta_classifier_scores.csv',
        'detectgpt_style': output_path / 'detectgpt_style_scores.csv',
        'fast_detectgpt': output_path / 'fast_detectgpt_scores.csv',
        'binoculars': output_path / 'binoculars_scores.csv',
        'watermark_kgw': output_path / 'kgw_watermark_scores.csv',
    }
    
    for detector_name, score_file in score_files.items():
        if score_file.exists():
            df = pd.read_csv(score_file)
            detector_scores[detector_name] = df
            print(f"Loaded {detector_name}: {len(df)} scores")
        else:
            print(f"Warning: {detector_name} scores not found at {score_file}")
    
    return detector_scores


def compute_human_pass_rate(detector_scores: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Compute human-pass rates for each detector.
    A sample "passes as human" if the detector score < 0.5 (human class).
    """
    print(f"\n{'='*80}")
    print(f"Computing Human-Pass Rates")
    print(f"{'='*80}\n")
    
    human_pass_rates = {}
    
    for detector_name, scores_df in detector_scores.items():
        # Find the appropriate score column for this detector
        score_col = None
        
        # Different detectors use different column names
        if detector_name == 'stats_baseline':
            score_col = 'ai_score'  # stats_baseline uses 'ai_score'
        else:
            # For other detectors, try common column names
            for col in ['score', 'probability', 'pred_label', 'ai_score']:
                if col in scores_df.columns:
                    score_col = col
                    break
        
        if score_col is None:
            print(f"Warning: Could not find score column in {detector_name}")
            print(f"Available columns: {scores_df.columns.tolist()}")
            continue
        
        # Count samples classified as human (score < 0.5)
        # For most detectors: score < 0.5 = human, score >= 0.5 = AI
        human_count = (scores_df[score_col] < 0.5).sum()
        total_count = len(scores_df)
        human_pass_rate = human_count / total_count if total_count > 0 else 0.0
        
        human_pass_rates[detector_name] = human_pass_rate
        print(f"{detector_name:30s}: {human_pass_rate:.2%} ({human_count}/{total_count})")
    
    return human_pass_rates


def generate_report(
    human_pass_rates: Dict[str, float],
    output_report: str,
    additional_metrics: Dict = None,
) -> None:
    """Generate comprehensive evaluation report."""
    print(f"\n{'='*80}")
    print(f"Generating Evaluation Report")
    print(f"{'='*80}")
    
    report = {
        "evaluation_type": "evader_final_test",
        "human_pass_rates": human_pass_rates,
        "average_human_pass_rate": sum(human_pass_rates.values()) / len(human_pass_rates) if human_pass_rates else 0.0,
        "additional_metrics": additional_metrics or {},
    }
    
    with open(output_report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {output_report}")
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Average Human-Pass Rate: {report['average_human_pass_rate']:.2%}")
    print(f"\nDetailed Breakdown:")
    for detector, rate in human_pass_rates.items():
        print(f"  {detector:30s}: {rate:.2%}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained evader model against detectors"
    )
    parser.add_argument(
        "--evader-checkpoint",
        type=str,
        default="/Users/raghav_sarna/Desktop/Drive/Plaksha/Semester 6/AdvNLP/NLPproject/post_generation/HMGC-dataset/output/checkgpt/model/evader_flan_t5_base_12to15h_earlystop_v1/checkpoint-41298",
        help="Path to evader model checkpoint",
    )
    parser.add_argument(
        "--phase3-input",
        type=str,
        default="/Users/raghav_sarna/Desktop/Drive/Plaksha/Semester 6/AdvNLP/NLPproject/detector_evaluation/results/checkgpt_phase3_input_100.csv",
        help="Path to phase 3 input CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/raghav_sarna/Desktop/Drive/Plaksha/Semester 6/AdvNLP/NLPproject/detector_evaluation/results/evader_phase3_eval",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model inference (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--roberta-model-dir",
        type=str,
        default=None,
        help="Path to trained RoBERTa detector (optional)",
    )
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"EVADER FINAL EVALUATION PIPELINE")
    print(f"{'='*80}\n")
    
    # Step 1: Load evader model
    evader_model, tokenizer = load_evader_model(args.evader_checkpoint, device=args.device)
    
    # Step 2: Load phase 3 input
    ai_df = load_phase3_input(args.phase3_input)
    
    # Step 3: Generate rewrites
    texts_to_rewrite = ai_df['text'].tolist()
    rewrites = generate_rewrites(
        evader_model,
        tokenizer,
        texts_to_rewrite,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # Step 4: Prepare detector input
    detector_input_csv = output_path / "evader_rephrased_texts.csv"
    prepare_detector_input(ai_df, rewrites, str(detector_input_csv))
    
    # Step 5: Run detectors
    run_detectors(
        str(detector_input_csv),
        args.output_dir,
        device=args.device,
        roberta_model_dir=args.roberta_model_dir,
    )
    
    # Step 6: Load and analyze scores
    detector_scores = load_detector_scores(args.output_dir)
    
    # Step 7: Compute metrics
    human_pass_rates = compute_human_pass_rate(detector_scores)
    
    # Step 8: Generate report
    report_path = output_path / "evaluation_report.json"
    generate_report(
        human_pass_rates,
        str(report_path),
        additional_metrics={
            "num_samples": len(ai_df),
            "evader_checkpoint": args.evader_checkpoint,
            "phase3_input": args.phase3_input,
        },
    )


if __name__ == "__main__":
    main()
