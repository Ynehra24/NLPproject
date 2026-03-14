"""Aggregate detector score files into comparable metrics tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.common.metrics import compute_attack_success_rate, compute_binary_metrics, encode_source_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate detector results")
    parser.add_argument("--scores-dir", required=True, help="Directory containing *_scores.csv")
    parser.add_argument("--output", required=True, help="Output aggregated csv")
    return parser.parse_args()


def evaluate_one(df: pd.DataFrame) -> dict:
    y_true = encode_source_labels(df["source"])
    y_score = df["ai_score"].to_numpy(dtype=float)
    threshold = float(df["threshold_used"].iloc[0])
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)

    y_pred = (y_score >= threshold).astype(int)
    metrics["attack_success_rate"] = compute_attack_success_rate(y_true, y_pred)
    return metrics


def main() -> None:
    args = parse_args()
    score_dir = Path(args.scores_dir)
    files = sorted(score_dir.glob("*_scores.csv"))
    if not files:
        raise FileNotFoundError(f"No *_scores.csv files found in {score_dir}")

    rows = []
    for file in files:
        df = pd.read_csv(file)
        if "source" not in df.columns or df["source"].isna().any():
            print(f"Skipping {file.name}: source labels not available")
            continue

        if "attack_type" not in df.columns:
            df["attack_type"] = "none"

        for attack_type, sub in df.groupby("attack_type"):
            row = evaluate_one(sub)
            row["detector_name"] = str(sub["detector_name"].iloc[0])
            row["attack_type"] = str(attack_type)
            row["n_samples"] = int(len(sub))
            rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("No valid rows were aggregated.")

    # Delta metrics relative to clean split per detector.
    delta_rows = []
    for detector_name, det_df in out_df.groupby("detector_name"):
        clean = det_df[det_df["attack_type"] == "none"]
        if clean.empty:
            for _, row in det_df.iterrows():
                r = row.to_dict()
                r["delta_auroc_vs_clean"] = np.nan
                r["delta_tpr1_vs_clean"] = np.nan
                r["delta_tpr5_vs_clean"] = np.nan
                delta_rows.append(r)
            continue

        clean_auroc = float(clean["auroc"].iloc[0])
        clean_tpr1 = float(clean["tpr_at_fpr_1"].iloc[0])
        clean_tpr5 = float(clean["tpr_at_fpr_5"].iloc[0])

        for _, row in det_df.iterrows():
            r = row.to_dict()
            r["delta_auroc_vs_clean"] = clean_auroc - float(row["auroc"])
            r["delta_tpr1_vs_clean"] = clean_tpr1 - float(row["tpr_at_fpr_1"])
            r["delta_tpr5_vs_clean"] = clean_tpr5 - float(row["tpr_at_fpr_5"])
            delta_rows.append(r)

    final_df = pd.DataFrame(delta_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output, index=False)
    print(f"Saved aggregated metrics: {output.resolve()}")


if __name__ == "__main__":
    main()
