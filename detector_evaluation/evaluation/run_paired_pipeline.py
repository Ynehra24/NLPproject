"""Run the paired evaluation pipeline end-to-end.

Pipeline steps:
1) Merge teammate paired CSV files (or use an existing merged file)
2) Convert paired format to long format
3) Run attack evaluation (all detectors + aggregation + plots)
4) Run paired analysis (score-drop + flip metrics)
5) Run disagreement-aware ensemble novelty evaluation (when label coverage permits)

Example:
  python -m evaluation.run_paired_pipeline \
    --pairs-dir data/teammate_pairs \
    --output-dir results/attack_eval_paired \
    --model-dir results/roberta_model \
    --device cpu \
    --skip-report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def check_novelty_prereqs(scores_dir: Path) -> Tuple[bool, str]:
    """Check whether disagreement ensemble can be trained on current scores.

    Requires both human and ai labels in source column across score CSV files.
    """
    files = sorted(scores_dir.glob("*_scores.csv"))
    if not files:
        return False, f"No score files found in: {scores_dir}"

    labels = set()
    for file in files:
        try:
            df = pd.read_csv(file, usecols=["source"])
        except Exception:
            continue

        vals = (
            df["source"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        labels.update(vals)

    required = {"human", "ai"}
    if required.issubset(labels):
        return True, ""
    return False, f"Need both source labels {sorted(required)} but found: {sorted(labels)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired detector pipeline end-to-end")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pairs-dir",
        help="Directory containing teammate paired CSV files",
    )
    group.add_argument(
        "--pairs-file",
        help="Single merged paired CSV file",
    )

    parser.add_argument(
        "--merged-output",
        default=None,
        help="Where to save merged paired CSV (used only with --pairs-dir)",
    )
    parser.add_argument(
        "--long-output",
        default=None,
        help="Where to save long-format CSV (default derived from input)",
    )

    parser.add_argument("--output-dir", default="results/attack_eval_paired", help="Pipeline output directory")
    parser.add_argument("--model-dir", default="results/roberta_model", help="Fine-tuned RoBERTa model dir")
    parser.add_argument("--device", default="cpu", help="Device for detector scoring")
    parser.add_argument("--detectgpt-perturb", type=int, default=2, help="DetectGPT perturbations")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Gemini API env variable name")
    parser.add_argument("--skip-report", action="store_true", help="Skip Gemini report generation")
    parser.add_argument("--skip-novelty", action="store_true", help="Skip disagreement-ensemble novelty evaluation")
    parser.add_argument(
        "--novelty-output-dir",
        default=None,
        help="Output dir for disagreement ensemble artifacts (default: <output-dir>/ensemble)",
    )
    return parser.parse_args()


def merge_pairs(pairs_dir: Path, merged_output: Path) -> Path:
    if not pairs_dir.exists():
        raise FileNotFoundError(f"Pairs directory not found: {pairs_dir}")

    files = sorted([p for p in pairs_dir.glob("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {pairs_dir}")

    parts = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            print(f"Warning: {f.name} is empty and will be skipped")
            continue
        parts.append(df)

    if not parts:
        raise RuntimeError("All CSV files were empty; nothing to merge")

    merged = pd.concat(parts, ignore_index=True, sort=False)
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_output, index=False)

    print(f"Merged {len(parts)} files into: {merged_output}")
    print(f"Merged rows: {len(merged)}")
    return merged_output


def main() -> None:
    args = parse_args()
    python = sys.executable

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_file:
        paired_input = Path(args.pairs_file)
        if not paired_input.exists():
            raise FileNotFoundError(f"Pairs file not found: {paired_input}")
    else:
        pairs_dir = Path(args.pairs_dir)
        if args.merged_output:
            merged_output = Path(args.merged_output)
        else:
            merged_output = pairs_dir / "all_pairs.csv"
        paired_input = merge_pairs(pairs_dir, merged_output)

    if args.long_output:
        long_output = Path(args.long_output)
    else:
        long_output = paired_input.with_name(f"{paired_input.stem}_long.csv")

    print("\n" + "=" * 60)
    print("STEP 1: Convert paired -> long")
    print("=" * 60)
    run_cmd(
        [
            python,
            "-m",
            "evaluation.paired_to_long",
            "--input",
            str(paired_input),
            "--output",
            str(long_output),
        ]
    )

    print("\n" + "=" * 60)
    print("STEP 2: Run detector evaluation")
    print("=" * 60)
    eval_cmd = [
        python,
        "-m",
        "evaluation.evaluate_attack",
        "--input",
        str(long_output),
        "--output-dir",
        str(output_dir),
        "--model-dir",
        str(args.model_dir),
        "--device",
        str(args.device),
        "--detectgpt-perturb",
        str(args.detectgpt_perturb),
        "--api-key-env",
        str(args.api_key_env),
    ]
    if args.skip_report:
        eval_cmd.append("--skip-report")
    run_cmd(eval_cmd)

    print("\n" + "=" * 60)
    print("STEP 3: Run paired analysis")
    print("=" * 60)
    paired_summary = output_dir / "paired_summary.csv"
    run_cmd(
        [
            python,
            "-m",
            "evaluation.paired_analysis",
            "--scores-dir",
            str(output_dir / "scores"),
            "--paired-long",
            str(long_output),
            "--output",
            str(paired_summary),
        ]
    )

    novelty_dir = Path(args.novelty_output_dir) if args.novelty_output_dir else (output_dir / "ensemble")
    if args.skip_novelty:
        print("\nSkipping novelty step (--skip-novelty)")
    else:
        ok, reason = check_novelty_prereqs(output_dir / "scores")
        if not ok:
            print(f"\nSkipping novelty step: {reason}")
        else:
            print("\n" + "=" * 60)
            print("STEP 4: Run disagreement-ensemble novelty evaluation")
            print("=" * 60)
            run_cmd(
                [
                    python,
                    "-m",
                    "evaluation.disagreement_ensemble",
                    "--scores-dir",
                    str(output_dir / "scores"),
                    "--output-dir",
                    str(novelty_dir),
                    "--allow-missing-detectors",
                ]
            )

    print("\n" + "=" * 60)
    print("PAIRED PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Paired input: {paired_input}")
    print(f"Long file: {long_output}")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Paired summary: {paired_summary}")
    if not args.skip_novelty:
        print(f"Ensemble novelty outputs: {novelty_dir}")


if __name__ == "__main__":
    main()
