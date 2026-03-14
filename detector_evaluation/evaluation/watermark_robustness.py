"""Evaluate KGW watermark detector robustness under attack conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.common.metrics import compute_binary_metrics, encode_source_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watermark robustness analysis")
    parser.add_argument("--scores-file", default=None, help="Path to kgw_watermark_scores.csv")
    parser.add_argument("--scores-dir", default=None, help="Directory containing *_scores.csv")
    parser.add_argument("--output", required=True, help="Output CSV for watermark robustness")
    parser.add_argument(
        "--group-by",
        choices=["attack_type", "attack_owner", "attack_type_owner"],
        default="attack_type",
        help="How to group attack conditions",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Optional fixed decision threshold")
    return parser.parse_args()


def _resolve_scores_path(args: argparse.Namespace) -> Path:
    if args.scores_file:
        return Path(args.scores_file)

    if not args.scores_dir:
        raise ValueError("Provide either --scores-file or --scores-dir")

    score_dir = Path(args.scores_dir)
    candidate = score_dir / "kgw_watermark_scores.csv"
    if candidate.exists():
        return candidate

    matches = sorted(score_dir.glob("*watermark*_scores.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find watermark score file in {score_dir}")
    return matches[0]


def _condition_label(df: pd.DataFrame, group_by: str) -> pd.Series:
    attack_type = df.get("attack_type", pd.Series(["none"] * len(df))).fillna("none").astype(str)
    attack_owner = df.get("attack_owner", pd.Series(["none"] * len(df))).fillna("none").astype(str)

    if group_by == "attack_type":
        return attack_type
    if group_by == "attack_owner":
        return attack_owner
    return attack_type + "::" + attack_owner


def _clean_label(group_by: str) -> str:
    if group_by == "attack_type_owner":
        return "none::none"
    return "none"


def _safe_mean(mask: np.ndarray, values: np.ndarray) -> float:
    if mask.sum() == 0:
        return np.nan
    return float(values[mask].mean())


def main() -> None:
    args = parse_args()
    score_path = _resolve_scores_path(args)
    df = pd.read_csv(score_path)

    if "ai_score" not in df.columns:
        raise ValueError("Watermark score file missing 'ai_score'")
    if "source" not in df.columns or df["source"].isna().any():
        raise ValueError("Watermark robustness requires non-null 'source' labels")

    threshold = args.threshold
    if threshold is None:
        threshold = float(df["threshold_used"].iloc[0]) if "threshold_used" in df.columns else 0.5

    df = df.copy()
    df["condition"] = _condition_label(df, args.group_by)

    rows: list[dict] = []
    for condition, sub in df.groupby("condition"):
        y_true = encode_source_labels(sub["source"])
        y_score = sub["ai_score"].to_numpy(dtype=float)
        y_pred = (y_score >= threshold).astype(int)

        metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)

        ai_mask = y_true == 1
        human_mask = y_true == 0
        z_values = sub["watermark_z"].to_numpy(dtype=float) if "watermark_z" in sub.columns else np.full(len(sub), np.nan)

        rows.append(
            {
                "condition": str(condition),
                "n_samples": int(len(sub)),
                "n_ai": int(ai_mask.sum()),
                "n_human": int(human_mask.sum()),
                "threshold": float(threshold),
                "auroc": float(metrics["auroc"]),
                "accuracy": float(metrics["accuracy"]),
                "f1_ai": float(metrics["f1_ai"]),
                "tpr_at_fpr_1": float(metrics["tpr_at_fpr_1"]),
                "tpr_at_fpr_5": float(metrics["tpr_at_fpr_5"]),
                "ai_detection_rate": _safe_mean(ai_mask, y_pred.astype(float)),
                "human_false_positive_rate": _safe_mean(human_mask, y_pred.astype(float)),
                "mean_ai_score": float(np.mean(y_score)),
                "mean_watermark_z": float(np.nanmean(z_values)),
            }
        )

    result_df = pd.DataFrame(rows)
    clean = result_df[result_df["condition"] == _clean_label(args.group_by)]

    if not clean.empty:
        clean_auroc = float(clean["auroc"].iloc[0])
        clean_tpr1 = float(clean["tpr_at_fpr_1"].iloc[0])
        clean_det = float(clean["ai_detection_rate"].iloc[0])

        result_df["delta_auroc_vs_clean"] = clean_auroc - result_df["auroc"]
        result_df["delta_tpr1_vs_clean"] = clean_tpr1 - result_df["tpr_at_fpr_1"]
        result_df["delta_ai_detection_rate_vs_clean"] = clean_det - result_df["ai_detection_rate"]
    else:
        result_df["delta_auroc_vs_clean"] = np.nan
        result_df["delta_tpr1_vs_clean"] = np.nan
        result_df["delta_ai_detection_rate_vs_clean"] = np.nan

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.sort_values("condition").to_csv(out_path, index=False)
    print(f"Saved watermark robustness table: {out_path.resolve()}")


if __name__ == "__main__":
    main()
