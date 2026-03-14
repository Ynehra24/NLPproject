"""Cross-detector transferability analysis for attack robustness.

This script consumes detector score files ("*_scores.csv") and estimates how
well each attack transfers across different detectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.common.metrics import compute_binary_metrics, encode_source_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-detector transferability analysis")
    parser.add_argument("--scores-dir", required=True, help="Directory containing *_scores.csv")
    parser.add_argument("--output-dir", required=True, help="Where transferability CSVs are saved")
    parser.add_argument(
        "--group-by",
        choices=["attack_type", "attack_owner", "attack_type_owner"],
        default="attack_type",
        help="How to group attack conditions",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.10,
        help="AUROC drop threshold to mark a detector as affected",
    )
    return parser.parse_args()


def _condition_label(df: pd.DataFrame, group_by: str) -> pd.Series:
    attack_type = df.get("attack_type", pd.Series(["none"] * len(df))).fillna("none").astype(str)
    attack_owner = df.get("attack_owner", pd.Series(["none"] * len(df))).fillna("none").astype(str)

    if group_by == "attack_type":
        return attack_type
    if group_by == "attack_owner":
        return attack_owner
    return attack_type + "::" + attack_owner


def _evaluate_groups(score_df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    if "source" not in score_df.columns or score_df["source"].isna().any():
        raise ValueError("Score file must include non-null 'source' labels for transferability analysis")

    df = score_df.copy()
    df["condition"] = _condition_label(df, group_by)

    rows: list[dict] = []
    for condition, sub in df.groupby("condition"):
        y_true = encode_source_labels(sub["source"])
        y_score = sub["ai_score"].to_numpy(dtype=float)
        threshold = float(sub["threshold_used"].iloc[0]) if "threshold_used" in sub.columns else 0.5
        metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)

        rows.append(
            {
                "condition": str(condition),
                "n_samples": int(len(sub)),
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "accuracy": float(metrics["accuracy"]),
                "f1_ai": float(metrics["f1_ai"]),
                "tpr_at_fpr_1": float(metrics["tpr_at_fpr_1"]),
                "tpr_at_fpr_5": float(metrics["tpr_at_fpr_5"]),
            }
        )

    return pd.DataFrame(rows)


def _clean_condition_name(group_by: str) -> str:
    if group_by == "attack_owner":
        return "none"
    if group_by == "attack_type_owner":
        return "none::none"
    return "none"


def main() -> None:
    args = parse_args()
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(scores_dir.glob("*_scores.csv"))
    if not files:
        raise FileNotFoundError(f"No *_scores.csv files found in: {scores_dir}")

    detector_condition_tables: list[pd.DataFrame] = []
    clean_label = _clean_condition_name(args.group_by)

    for file in files:
        df = pd.read_csv(file)
        if "ai_score" not in df.columns:
            print(f"Skipping {file.name}: missing ai_score")
            continue

        detector_name = str(df["detector_name"].iloc[0]) if "detector_name" in df.columns else file.stem
        cond_metrics = _evaluate_groups(df, args.group_by)
        cond_metrics["detector_name"] = detector_name

        clean = cond_metrics[cond_metrics["condition"] == clean_label]
        clean_auroc = float(clean["auroc"].iloc[0]) if not clean.empty else np.nan
        clean_tpr1 = float(clean["tpr_at_fpr_1"].iloc[0]) if not clean.empty else np.nan

        cond_metrics["auroc_drop_vs_clean"] = clean_auroc - cond_metrics["auroc"]
        cond_metrics["tpr1_drop_vs_clean"] = clean_tpr1 - cond_metrics["tpr_at_fpr_1"]
        detector_condition_tables.append(cond_metrics)

    if not detector_condition_tables:
        raise RuntimeError("No valid detector score files available for transferability analysis")

    long_df = pd.concat(detector_condition_tables, ignore_index=True)
    long_path = output_dir / "transferability_long_metrics.csv"
    long_df.to_csv(long_path, index=False)

    matrix_df = (
        long_df.pivot_table(
            index="condition",
            columns="detector_name",
            values="auroc_drop_vs_clean",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    matrix_path = output_dir / "attack_transferability_matrix.csv"
    matrix_df.to_csv(matrix_path)

    summary_rows: list[dict] = []
    for condition, row in matrix_df.iterrows():
        drops = row.dropna().to_numpy(dtype=float)
        if drops.size == 0:
            continue
        affected = int((drops >= args.drop_threshold).sum())
        total = int(drops.size)
        summary_rows.append(
            {
                "condition": str(condition),
                "n_detectors": total,
                "n_affected_detectors": affected,
                "transferability_ratio": float(affected / total),
                "mean_auroc_drop": float(np.mean(drops)),
                "median_auroc_drop": float(np.median(drops)),
                "max_auroc_drop": float(np.max(drops)),
                "min_auroc_drop": float(np.min(drops)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["transferability_ratio", "mean_auroc_drop"], ascending=False
    )
    summary_path = output_dir / "attack_transferability_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    corr_df = matrix_df.corr(method="spearman", min_periods=1)
    corr_path = output_dir / "detector_vulnerability_correlation.csv"
    corr_df.to_csv(corr_path)

    print(f"Saved transferability long metrics: {long_path.resolve()}")
    print(f"Saved transferability matrix: {matrix_path.resolve()}")
    print(f"Saved transferability summary: {summary_path.resolve()}")
    print(f"Saved detector vulnerability correlation: {corr_path.resolve()}")


if __name__ == "__main__":
    main()
