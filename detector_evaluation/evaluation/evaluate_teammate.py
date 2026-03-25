"""Simple one-command evaluation for teammate data.

Runs all 6 detectors → aggregates metrics → generates plots → creates report.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate teammate data with all detectors")
    parser.add_argument("--input", required=True, help="Teammate CSV file (text, label columns)")
    parser.add_argument("--output-dir", default="results/teammate_eval", help="Output directory")
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
            str(input_path),
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
            "--input",
            str(input_path),
            "--scores-dir",
            str(scores_dir),
            "--output",
            str(metrics_csv),
        ]
    )
    
    print(f"\n✓ Metrics saved to: {metrics_csv}")
    print("\nMetrics summary:")
    run_cmd(["python", "-c", f"import pandas; df = pandas.read_csv('{metrics_csv}'); print(df.to_string(index=False))"])
    
    print("\n" + "="*60)
    print("STEP 3: Generating plots...")
    print("="*60)
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python,
            "-m",
            "evaluation.plots",
            "--scores-dir",
            str(scores_dir),
            "--test-file",
            str(input_path),
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
                str(reports_dir / "teammate_report.md"),
                "--api-key-env",
                args.api_key_env,
            ]
        )
        
        print(f"\n✓ Report saved to: {reports_dir / 'teammate_report.md'}")
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults in: {output_dir}")
    print(f"  - Detector scores: {scores_dir}/")
    print(f"  - Metrics table: {metrics_csv}")
    print(f"  - Plots: {figures_dir}/")
    if not args.skip_report:
        print(f"  - Report: {reports_dir}/teammate_report.md")


if __name__ == "__main__":
    main()
