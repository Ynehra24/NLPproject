"""Run the paired evaluation pipeline end-to-end.

Takes a teammate's paired CSV (original + humanized text) and runs:
  1) Convert paired → long format
  2) Run all 6 detectors
  3) Aggregate metrics + evasion + transferability
  4) Generate plots
  5) Paired analysis (score-drop, flip rates)
  6) Disagreement-aware ensemble (novelty)

Usage:
  python -m evaluation.run_paired_pipeline \
    --pairs-file teammate_pairs_charlevel.csv \
    --output-dir results/attack_eval_charlevel \
    --model-dir results/roberta_model \
    --device cpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired detector pipeline end-to-end")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs-dir", help="Directory containing teammate paired CSV files")
    group.add_argument("--pairs-file", help="Single paired CSV file")

    parser.add_argument("--output-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--model-dir", default="results/roberta_model", help="Fine-tuned RoBERTa model dir")
    parser.add_argument("--device", default="cpu", help="Device for detector scoring")
    parser.add_argument(
        "--thresholds-file",
        default="results/thresholds.json",
        help="Detector thresholds JSON passed to evaluation.run_all",
    )
    parser.add_argument("--detectgpt-perturb", type=int, default=2, help="DetectGPT perturbations")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip disagreement-ensemble step")
    return parser.parse_args()


def merge_pairs(pairs_dir: Path) -> Path:
    """Merge all CSVs in a directory into one paired file."""
    files = sorted([p for p in pairs_dir.glob("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {pairs_dir}")

    parts = [pd.read_csv(f) for f in files if not pd.read_csv(f).empty]
    if not parts:
        raise RuntimeError("All CSV files were empty")

    merged = pd.concat(parts, ignore_index=True, sort=False)
    merged_output = pairs_dir / "all_pairs.csv"
    merged.to_csv(merged_output, index=False)
    print(f"Merged {len(parts)} files → {merged_output} ({len(merged)} rows)")
    return merged_output


def main() -> None:
    args = parse_args()
    python = sys.executable

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "scores"

    # Resolve input file
    if args.pairs_file:
        paired_input = Path(args.pairs_file)
        if not paired_input.exists():
            raise FileNotFoundError(f"Pairs file not found: {paired_input}")
    else:
        paired_input = merge_pairs(Path(args.pairs_dir))

    # Check if already in long format
    df_input = pd.read_csv(paired_input, nrows=1)
    if "variant" in df_input.columns and "id" in df_input.columns:
        long_output = paired_input
        print("\n" + "=" * 60)
        print("STEP 1: Input already in long format (skipping conversion)")
        print("=" * 60)
    else:
        long_output = paired_input.with_name(f"{paired_input.stem}_long.csv")
        # ── STEP 1: Convert paired → long ────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 1: Convert paired → long format")
        print("=" * 60)
        run_cmd([
            python, "-m", "evaluation.paired_to_long",
            "--input", str(paired_input),
            "--output", str(long_output),
        ])

    # ── STEP 2: Run all 6 detectors ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Run all detectors")
    print("=" * 60)
    run_cmd([
        python, "-m", "evaluation.run_all",
        "--input", str(long_output),
        "--output-dir", str(scores_dir),
        "--device", args.device,
        "--roberta-model-dir", str(args.model_dir),
        "--thresholds-file", str(args.thresholds_file),
        "--run-detectgpt",
        "--detectgpt-perturb", str(args.detectgpt_perturb),
        "--run-fast-detectgpt",
        "--run-binoculars",
        "--run-watermark",
    ])

    # ── STEP 3: Aggregate metrics ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Aggregate metrics")
    print("=" * 60)
    run_cmd([
        python, "-m", "evaluation.aggregate_results",
        "--scores-dir", str(scores_dir),
        "--output-dir", str(output_dir),
    ])

    # ── STEP 4: Generate plots ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Generate plots")
    print("=" * 60)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        python, "-m", "evaluation.plots",
        "--metrics", str(output_dir / "metrics.csv"),
        "--output-dir", str(figures_dir),
        "--scores-dir", str(scores_dir),
    ])

    # ── STEP 5: Paired analysis ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Paired analysis (score-drop, flip rates)")
    print("=" * 60)
    run_cmd([
        python, "-m", "evaluation.paired_analysis",
        "--scores-dir", str(scores_dir),
        "--paired-long", str(long_output),
        "--output", str(output_dir / "paired_summary.csv"),
    ])

    # ── STEP 6: Disagreement ensemble (novelty) ──────────────────
    if args.skip_ensemble:
        print("\nSkipping ensemble step (--skip-ensemble)")
    else:
        # Check if we have both human and ai labels
        has_labels = False
        for f in sorted(scores_dir.glob("*_scores.csv")):
            try:
                df = pd.read_csv(f, usecols=["source"])
                labels = set(df["source"].dropna().astype(str).str.lower().tolist())
                if {"human", "ai"}.issubset(labels):
                    has_labels = True
                    break
            except Exception:
                continue

        if has_labels:
            print("\n" + "=" * 60)
            print("STEP 6: Disagreement-ensemble (novelty)")
            print("=" * 60)
            run_cmd([
                python, "-m", "evaluation.disagreement_ensemble",
                "--scores-dir", str(scores_dir),
                "--output-dir", str(output_dir / "ensemble"),
                "--allow-missing-detectors",
            ])
        else:
            print("\nSkipping ensemble: need both 'human' and 'ai' source labels")

    # ── Done ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results in: {output_dir}")
    print(f"  scores/          → raw detector scores")
    print(f"  metrics.csv      → per-detector metrics + deltas")
    print(f"  evasion_summary  → cross-paradigm evasion rates")
    print(f"  transferability  → attack × detector AUROC-drop matrix")
    print(f"  paired_summary   → score-drop + flip rates")
    print(f"  figures/         → plots")
    if not args.skip_ensemble:
        print(f"  ensemble/        → disagreement ensemble results")


if __name__ == "__main__":
    main()
