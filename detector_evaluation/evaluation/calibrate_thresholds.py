"""Calibrate detection thresholds using the validation split.

Runs each detector on data/splits/val.csv (which has both human + AI rows),
finds the optimal F1 threshold per detector, and saves a thresholds.json file.

Run ONCE after training:
  python -m evaluation.calibrate_thresholds
  python -m evaluation.calibrate_thresholds --val data/splits/val.csv --output results/thresholds.json
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

from detectors.common.metrics import encode_source_labels, find_best_threshold


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate detector thresholds on val split")
    parser.add_argument("--val", default="data/splits/val.csv", help="Validation CSV with 'source' labels")
    parser.add_argument("--output", default="results/thresholds.json", help="Where to save thresholds.json")
    parser.add_argument("--roberta-model-dir", default="results/roberta_model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detectgpt-perturb", type=int, default=2)
    return parser.parse_args()


def _find_threshold(scores_csv: Path, val_df: pd.DataFrame) -> float:
    """Load scores CSV, join with val labels, return best-F1 threshold."""
    scores = pd.read_csv(scores_csv)
    merged = val_df[["id", "source"]].merge(scores[["id", "ai_score"]], on="id", how="inner")
    if merged.empty:
        raise RuntimeError(f"No matching IDs between val split and {scores_csv.name}")
    y_true = encode_source_labels(merged["source"])
    y_score = merged["ai_score"].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        raise RuntimeError("Validation split must contain both 'human' and 'ai' rows")
    return find_best_threshold(y_true, y_score)


def main() -> None:
    args = parse_args()
    python = sys.executable
    val_path = Path(args.val)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        raise FileNotFoundError(f"Val split not found: {val_path}. Run prepare_hc3 first.")

    val_df = pd.read_csv(val_path)
    required = {"id", "source"}
    if not required.issubset(val_df.columns):
        raise ValueError(f"Val CSV must have columns: {required}")

    thresholds: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # ── Stats Baseline ────────────────────────────────────────
        print("\n── Calibrating: stats_baseline ──")
        out = tmp_dir / "stats_baseline_scores.csv"
        run_cmd([python, "-m", "detectors.stats_baseline.score",
                 "--input", str(val_path), "--output", str(out), "--device", args.device])
        thresholds["stats_baseline"] = _find_threshold(out, val_df)

        # ── RoBERTa Classifier ────────────────────────────────────
        roberta_dir = Path(args.roberta_model_dir)
        if roberta_dir.exists():
            print("\n── Calibrating: roberta_classifier ──")
            out = tmp_dir / "roberta_scores.csv"
            run_cmd([python, "-m", "detectors.roberta_classifier.infer",
                     "--input", str(val_path), "--model-dir", str(roberta_dir),
                     "--output", str(out), "--device", args.device])
            thresholds["roberta_classifier"] = _find_threshold(out, val_df)
        else:
            print(f"\nSkipping roberta_classifier: model dir not found at {roberta_dir}")

        # ── DetectGPT ─────────────────────────────────────────────
        print("\n── Calibrating: detectgpt_style ──")
        out = tmp_dir / "detectgpt_scores.csv"
        run_cmd([python, "-m", "detectors.detectgpt.score",
                 "--input", str(val_path), "--output", str(out),
                 "--device", args.device, "--n-perturb", str(args.detectgpt_perturb)])
        thresholds["detectgpt_style"] = _find_threshold(out, val_df)

        # ── Fast-DetectGPT ────────────────────────────────────────
        print("\n── Calibrating: fast_detectgpt ──")
        out = tmp_dir / "fast_detectgpt_scores.csv"
        run_cmd([python, "-m", "detectors.fast_detectgpt.score",
                 "--input", str(val_path), "--output", str(out), "--device", args.device])
        thresholds["fast_detectgpt"] = _find_threshold(out, val_df)

        # ── Binoculars ────────────────────────────────────────────
        print("\n── Calibrating: binoculars ──")
        out = tmp_dir / "binoculars_scores.csv"
        run_cmd([python, "-m", "detectors.binoculars.score",
                 "--input", str(val_path), "--output", str(out), "--device", args.device])
        thresholds["binoculars"] = _find_threshold(out, val_df)

        # ── KGW Watermark ─────────────────────────────────────────
        print("\n── Calibrating: kgw_watermark ──")
        out = tmp_dir / "watermark_scores.csv"
        run_cmd([python, "-m", "detectors.watermark.score",
                 "--input", str(val_path), "--output", str(out)])
        thresholds["kgw_watermark"] = _find_threshold(out, val_df)

    # Save
    output_path.write_text(json.dumps(thresholds, indent=2))

    print("\n" + "=" * 50)
    print("✓ Calibrated thresholds:")
    for name, t in thresholds.items():
        print(f"  {name:<25} {t:.4f}")
    print(f"\nSaved → {output_path.resolve()}")


if __name__ == "__main__":
    main()
