"""Simple one-command evaluation for attack data.

Runs all 6 detectors → aggregates metrics → generates plots → creates report.
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


def normalize_input(input_path: Path, output_dir: Path) -> Path:
    """Ensure detector input has required schema columns.

    Required by detector loader: id, text.
    For evaluation metrics, source is also needed. If source is missing and label
    exists, convert label 0/1 to source human/ai.
    """
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain 'text' column")

    if "id" not in df.columns:
        df["id"] = [f"sample_{i}" for i in range(len(df))]

    if "source" not in df.columns:
        if "label" in df.columns:
            mapped = df["label"].astype(str).str.strip().str.lower().map(
                {"0": "human", "1": "ai", "human": "human", "ai": "ai"}
            )
            if mapped.isna().any():
                bad = sorted(df.loc[mapped.isna(), "label"].astype(str).unique().tolist())
                raise ValueError(f"Invalid label values for conversion to source: {bad}")
            df["source"] = mapped

    normalized_path = output_dir / "_normalized_input.csv"
    df.to_csv(normalized_path, index=False)
    return normalized_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate attack data with all detectors")
    parser.add_argument(
        "--input",
        required=True,
        help="Attack CSV file. Requires text; id/source are auto-created when possible.",
    )
    parser.add_argument("--output-dir", default="results/attack_eval", help="Output directory")
    parser.add_argument("--model-dir", default="results/roberta_model", help="Fine-tuned RoBERTa model")
    parser.add_argument("--device", default="cpu", help="Device (cpu/gpu)")
    parser.add_argument("--detectgpt-perturb", type=int, default=2, help="DetectGPT perturbations")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Environment variable for Gemini API key")
    parser.add_argument("--skip-report", action="store_true", help="Skip Gemini report generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not Path(args.model_dir).exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "scores"
    figures_dir = output_dir / "figures"
    normalized_input = normalize_input(input_path, output_dir)
    
    python = sys.executable
    
    print("\n" + "="*60)
    print("STEP 1: Running all detectors...")
    print("="*60)
    
    run_cmd(
        [
            python,
            "-m",
            "evaluation.run_all",
            "--input",
            str(normalized_input),
            "--output-dir",
            str(scores_dir),
            "--device",
            args.device,
            "--roberta-model-dir",
            args.model_dir,
            "--run-detectgpt",
            "--detectgpt-perturb",
            str(args.detectgpt_perturb),
            "--run-fast-detectgpt",
            "--run-binoculars",
            "--run-watermark",
        ]
    )
    
    print("\n" + "="*60)
    print("STEP 2: Aggregating metrics...")
    print("="*60)
    
    metrics_csv = output_dir / "metrics.csv"
    run_cmd(
        [
            python,
            "-m",
            "evaluation.aggregate_results",
            "--scores-dir",
            str(scores_dir),
            "--output",
            str(metrics_csv),
        ]
    )
    
    print(f"\n✓ Metrics saved to: {metrics_csv}")
    print("\nMetrics summary:")
    run_cmd([python, "-c", f"import pandas; df = pandas.read_csv('{metrics_csv}'); print(df.to_string(index=False))"])
    
    print("\n" + "="*60)
    print("STEP 3: Generating plots...")
    print("="*60)
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python,
            "-m",
            "evaluation.plots",
            "--metrics",
            str(metrics_csv),
            "--output-dir",
            str(figures_dir),
        ]
    )
    
    print(f"\n✓ Plots saved to: {figures_dir}")
    
    if not args.skip_report:
        print("\n" + "="*60)
        print("STEP 4: Generating Gemini report...")
        print("="*60)
        
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        run_cmd(
            [
                python,
                "-m",
                "evaluation.gemini_report_writer",
                "--analysis-root",
                str(output_dir),
                "--output",
                str(reports_dir / "attack_report.md"),
                "--api-key-env",
                args.api_key_env,
            ]
        )
        
        print(f"\n✓ Report saved to: {reports_dir / 'attack_report.md'}")
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults in: {output_dir}")
    print(f"  - Detector scores: {scores_dir}/")
    print(f"  - Metrics table: {metrics_csv}")
    print(f"  - Plots: {figures_dir}/")
    if not args.skip_report:
        print(f"  - Report: {reports_dir}/attack_report.md")


if __name__ == "__main__":
    main()
