"""Baseline test: run all detectors on the held-out test split and validate implementation.

Uses data/splits/test.csv (human + AI rows) + calibrated thresholds from results/thresholds.json.
Produces a clean comparison table of your numbers vs expected paper benchmarks.

Run after calibrate_thresholds:
  python -m evaluation.run_baseline_test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.common.metrics import (
    compute_binary_metrics,
    encode_source_labels,
)

# ---------------------------------------------------------------------------
# Expected paper benchmarks on HC3 (AUROC, Accuracy).
# These are approximate ranges from the original papers on clean, balanced
# English data. Your numbers should fall within ±5-10% of these ranges.
# ---------------------------------------------------------------------------
PAPER_BENCHMARKS = {
    "roberta_classifier": {"auroc": 0.97, "accuracy": 0.93, "note": "RoBERTa fine-tuned (HC3 paper)"},
    "stats_baseline":     {"auroc": 0.75, "accuracy": 0.72, "note": "Perplexity/rank baseline"},
    "detectgpt_style":    {"auroc": 0.70, "accuracy": 0.68, "note": "DetectGPT curvature (gpt2, 2 perturb)"},
    "fast_detectgpt":     {"auroc": 0.72, "accuracy": 0.70, "note": "Fast-DetectGPT (gpt2-medium)"},
    "binoculars":         {"auroc": 0.85, "accuracy": 0.80, "note": "Binoculars (gpt2-medium/gpt2)"},
    "kgw_watermark":      {"auroc": 0.50, "accuracy": 0.50, "note": "KGW watermark (not trained on this data — expected near-random)"},
}

PASS_TOLERANCE = 0.10  # Within 10% of benchmark = PASS


def run_cmd(cmd: list[str]) -> None:
    print("  Running:", " ".join(cmd[-6:]))  # Show last 6 args only for brevity
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline test: validate detector implementations")
    parser.add_argument("--test", default="data/splits/test.csv")
    parser.add_argument("--thresholds-file", default="results/thresholds.json")
    parser.add_argument("--output", default="results/baseline_test_results.csv")
    parser.add_argument("--roberta-model-dir", default="results/roberta_model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detectgpt-perturb", type=int, default=2)
    return parser.parse_args()


def _compute_metrics_from_scores(scores_csv: Path, test_df: pd.DataFrame, threshold: float) -> dict:
    scores = pd.read_csv(scores_csv)
    merged = test_df[["id", "source"]].merge(scores[["id", "ai_score"]], on="id", how="inner")
    if merged.empty:
        return {}
    y_true = encode_source_labels(merged["source"])
    y_score = merged["ai_score"].to_numpy(dtype=float)
    m = compute_binary_metrics(y_true, y_score, threshold=threshold)
    return m


def main() -> None:
    args = parse_args()
    python = sys.executable
    test_path = Path(args.test)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}. Run prepare_hc3 first.")

    # Load calibrated thresholds
    thresholds: dict[str, float] = {}
    t_path = Path(args.thresholds_file)
    if t_path.exists():
        thresholds = json.loads(t_path.read_text())
        print(f"Loaded thresholds from {t_path}")
    else:
        print(f"⚠️  No thresholds.json found — using 0.5 for all detectors. Run calibrate_thresholds first.")

    test_df = pd.read_csv(test_path)
    n_ai = (test_df["source"] == "ai").sum()
    n_human = (test_df["source"] == "human").sum()
    print(f"\nTest split: {len(test_df)} samples ({n_ai} AI, {n_human} human)\n")

    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        detectors_to_run = [
            ("stats_baseline", [python, "-m", "detectors.stats_baseline.score",
                                "--input", str(test_path), "--device", args.device]),
            ("detectgpt_style", [python, "-m", "detectors.detectgpt.score",
                                 "--input", str(test_path), "--device", args.device,
                                 "--n-perturb", str(args.detectgpt_perturb)]),
            ("fast_detectgpt", [python, "-m", "detectors.fast_detectgpt.score",
                                "--input", str(test_path), "--device", args.device]),
            ("binoculars", [python, "-m", "detectors.binoculars.score",
                            "--input", str(test_path), "--device", args.device]),
            ("kgw_watermark", [python, "-m", "detectors.watermark.score",
                               "--input", str(test_path)]),
        ]

        roberta_dir = Path(args.roberta_model_dir)
        if roberta_dir.exists():
            detectors_to_run.insert(0, (
                "roberta_classifier",
                [python, "-m", "detectors.roberta_classifier.infer",
                 "--input", str(test_path), "--model-dir", str(roberta_dir),
                 "--device", args.device],
            ))

        for detector_name, cmd in detectors_to_run:
            print(f"── Scoring: {detector_name} ──")
            out_file = tmp_dir / f"{detector_name}_scores.csv"
            cmd += ["--output", str(out_file)]
            threshold = thresholds.get(detector_name, 0.5)
            cmd += ["--threshold", str(threshold)]

            try:
                run_cmd(cmd)
                metrics = _compute_metrics_from_scores(out_file, test_df, threshold)
                if not metrics:
                    print(f"  No overlapping IDs — skipping\n")
                    continue

                bench = PAPER_BENCHMARKS.get(detector_name, {})
                row = {
                    "detector": detector_name,
                    "your_auroc": round(metrics.get("auroc", float("nan")), 4),
                    "paper_auroc": bench.get("auroc", float("nan")),
                    "your_accuracy": round(metrics.get("accuracy", float("nan")), 4),
                    "paper_accuracy": bench.get("accuracy", float("nan")),
                    "f1_ai": round(metrics.get("f1_ai", float("nan")), 4),
                    "threshold_used": round(threshold, 4),
                    "note": bench.get("note", ""),
                }
                results.append(row)
                print(f"  AUROC: {row['your_auroc']} (paper: {row['paper_auroc']})")
                print(f"  Acc:   {row['your_accuracy']} (paper: {row['paper_accuracy']})\n")

            except subprocess.CalledProcessError as e:
                print(f"  ERROR running {detector_name}: {e}\n")

    if not results:
        print("No results produced.")
        return

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)

    # Print final summary table
    print("=" * 60)
    print("BASELINE TEST RESULTS")
    print("=" * 60)
    print(f"{'Detector':<26} {'AUROC':>7}{'Paper':>7}   {'Acc':>7}{'Paper':>7}")
    print("-" * 60)
    for _, row in df_out.iterrows():
        auroc_str = f"{row['your_auroc']:.3f}" if not np.isnan(row['your_auroc']) else "  N/A"
        acc_str = f"{row['your_accuracy']:.3f}" if not np.isnan(row['your_accuracy']) else "  N/A"
        print(
            f"{row['detector']:<26} {auroc_str:>7}{row['paper_auroc']:>7.2f}   "
            f"{acc_str:>7}{row['paper_accuracy']:>7.2f}"
        )
    print("=" * 60)

    print(f"\nFull results saved → {output_path.resolve()}\n")
    print("NOTE: KGW Watermark is not trained on this data — near-random AUROC is expected.")


if __name__ == "__main__":
    main()
