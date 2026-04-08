"""Compute paired robustness metrics from detector score files.

This script compares original vs humanized variants per pair_id and computes:
- mean/median score drop
- flip rate (original predicted ai, humanized predicted human)
- conditional flip rate among originally detected AI samples

Example:
  python -m evaluation.paired_analysis \
    --scores-dir results/attack_eval/scores \
    --paired-long paired_long.csv \
    --output results/attack_eval/paired_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired robustness analysis")
    parser.add_argument("--scores-dir", required=True, help="Directory containing *_scores.csv")
    parser.add_argument("--paired-long", required=True, help="Long-format paired CSV used for scoring")
    parser.add_argument("--output", required=True, help="Output paired summary CSV")
    return parser.parse_args()


def build_pair_table(scored: pd.DataFrame) -> pd.DataFrame:
    piv = scored.pivot_table(
        index="pair_id",
        columns="variant",
        values=["ai_score", "predicted_label"],
        aggfunc="first",
    )

    needed_cols = [
        ("ai_score", "original"),
        ("ai_score", "humanized"),
        ("predicted_label", "original"),
        ("predicted_label", "humanized"),
    ]
    for col in needed_cols:
        if col not in piv.columns:
            return pd.DataFrame()

    out = pd.DataFrame(
        {
            "pair_id": piv.index.astype(str),
            "orig_score": piv[("ai_score", "original")].astype(float),
            "hum_score": piv[("ai_score", "humanized")].astype(float),
            "orig_pred": piv[("predicted_label", "original")].astype(str),
            "hum_pred": piv[("predicted_label", "humanized")].astype(str),
        }
    )
    return out.dropna()


def summarize_pairs(df: pd.DataFrame) -> dict:
    orig_is_ai = df["orig_pred"].str.lower() == "ai"
    hum_is_human = df["hum_pred"].str.lower() == "human"
    flips = orig_is_ai & hum_is_human

    cond_den = int(orig_is_ai.sum())
    conditional_flip_rate = float(flips[orig_is_ai].mean()) if cond_den > 0 else np.nan

    score_drop = df["orig_score"] - df["hum_score"]
    return {
        "n_pairs": int(len(df)),
        "orig_detected_ai_rate": float(orig_is_ai.mean()),
        "humanized_pred_human_rate": float(hum_is_human.mean()),
        "flip_rate_all_pairs": float(flips.mean()),
        "flip_rate_conditional_on_orig_ai": conditional_flip_rate,
        "mean_score_drop": float(score_drop.mean()),
        "median_score_drop": float(score_drop.median()),
    }


def main() -> None:
    args = parse_args()
    scores_dir = Path(args.scores_dir)
    paired_long_path = Path(args.paired_long)

    if not scores_dir.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")
    if not paired_long_path.exists():
        raise FileNotFoundError(f"Paired long CSV not found: {paired_long_path}")

    meta = pd.read_csv(paired_long_path)
    required_meta = ["id", "pair_id", "variant"]
    missing = [c for c in required_meta if c not in meta.columns]
    if missing:
        raise ValueError(f"Paired long CSV missing columns: {missing}")

    if "attack_type" not in meta.columns:
        meta["attack_type"] = "none"

    meta = meta[["id", "pair_id", "variant", "attack_type"]].copy()
    meta["id"] = meta["id"].astype(str)
    meta["pair_id"] = meta["pair_id"].astype(str)
    meta["variant"] = meta["variant"].astype(str).str.lower()

    rows: list[dict] = []
    score_files = sorted(scores_dir.glob("*_scores.csv"))
    if not score_files:
        raise FileNotFoundError(f"No *_scores.csv files found in {scores_dir}")

    for sf in score_files:
        det_df = pd.read_csv(sf)
        required_score_cols = ["id", "detector_name", "ai_score", "predicted_label"]
        missing_score = [c for c in required_score_cols if c not in det_df.columns]
        if missing_score:
            print(f"Skipping {sf.name}: missing columns {missing_score}")
            continue

        # Keep only required score columns to avoid merge suffix collisions
        # with metadata columns like attack_type.
        det_df = det_df[required_score_cols].copy()
        det_df["id"] = det_df["id"].astype(str)
        merged = det_df.merge(meta, on="id", how="inner")
        if merged.empty:
            print(f"Skipping {sf.name}: no matching ids with paired-long file")
            continue

        detector_name = str(merged["detector_name"].iloc[0])
        for attack_type, sub in merged.groupby("attack_type"):
            pair_df = build_pair_table(sub)
            if pair_df.empty:
                continue

            summary = summarize_pairs(pair_df)
            summary["detector_name"] = detector_name
            summary["attack_type"] = str(attack_type)
            rows.append(summary)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("No paired summaries produced. Check pair mapping and score files.")

    out_df = out_df[
        [
            "detector_name",
            "attack_type",
            "n_pairs",
            "orig_detected_ai_rate",
            "humanized_pred_human_rate",
            "flip_rate_all_pairs",
            "flip_rate_conditional_on_orig_ai",
            "mean_score_drop",
            "median_score_drop",
        ]
    ].sort_values(["detector_name", "attack_type"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved paired robustness summary: {output_path.resolve()}")


if __name__ == "__main__":
    main()
