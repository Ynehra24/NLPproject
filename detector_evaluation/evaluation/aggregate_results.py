"""Aggregate detector score files into metrics, evasion rates, and transferability.

This is the single metrics script. It produces:
  1) metrics.csv             – per-detector, per-attack binary metrics + deltas
  2) evasion_summary.csv     – cross-paradigm evasion rates (fraction evading ALL detectors)
  3) transferability.csv     – attack × detector AUROC-drop matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from detectors.common.metrics import (
    compute_attack_success_rate,
    compute_binary_metrics,
    encode_source_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate detector results")
    parser.add_argument("--scores-dir", required=True, help="Directory containing *_scores.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for all metrics files")
    return parser.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_all_scores(scores_dir: Path) -> pd.DataFrame:
    """Load and concatenate all *_scores.csv files."""
    files = sorted(scores_dir.glob("*_scores.csv"))
    if not files:
        raise FileNotFoundError(f"No *_scores.csv files found in {scores_dir}")

    frames: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        required = {"id", "detector_name", "ai_score", "threshold_used"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Skipping {f.name}: missing columns {missing}")
            continue
        if "attack_type" not in df.columns:
            df["attack_type"] = "none"
        df["attack_type"] = df["attack_type"].fillna("none").astype(str)
        df["id"] = df["id"].astype(str)
        df["detector_name"] = df["detector_name"].astype(str)
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid score files found")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
# 1) Per-detector binary metrics
# ------------------------------------------------------------------

def _compute_metrics_table(all_scores: pd.DataFrame) -> pd.DataFrame:
    """Per-detector, per-attack_type binary metrics + delta vs clean."""
    rows = []
    for (detector, attack), sub in all_scores.groupby(["detector_name", "attack_type"]):
        if "source" not in sub.columns or sub["source"].isna().any():
            continue
        y_true = encode_source_labels(sub["source"])
        y_score = sub["ai_score"].to_numpy(dtype=float)
        threshold = float(sub["threshold_used"].iloc[0])

        m = compute_binary_metrics(y_true, y_score, threshold=threshold)
        y_pred = (y_score >= threshold).astype(int)
        m["attack_success_rate"] = compute_attack_success_rate(y_true, y_pred)
        m["detector_name"] = str(detector)
        m["attack_type"] = str(attack)
        m["n_samples"] = int(len(sub))
        rows.append(m)

    if not rows:
        raise RuntimeError("No valid rows were aggregated (need 'source' labels).")

    df = pd.DataFrame(rows)

    # Delta metrics relative to clean (attack_type == "none") per detector.
    delta_rows = []
    for det, det_df in df.groupby("detector_name"):
        clean = det_df[det_df["attack_type"] == "none"]
        c_auroc = float(clean["auroc"].iloc[0]) if not clean.empty else np.nan
        c_tpr1 = float(clean["tpr_at_fpr_1"].iloc[0]) if not clean.empty else np.nan
        c_tpr5 = float(clean["tpr_at_fpr_5"].iloc[0]) if not clean.empty else np.nan

        for _, row in det_df.iterrows():
            r = row.to_dict()
            r["delta_auroc_vs_clean"] = c_auroc - float(row["auroc"])
            r["delta_tpr1_vs_clean"] = c_tpr1 - float(row["tpr_at_fpr_1"])
            r["delta_tpr5_vs_clean"] = c_tpr5 - float(row["tpr_at_fpr_5"])
            delta_rows.append(r)

    return pd.DataFrame(delta_rows).dropna(axis=1, how="all")


# ------------------------------------------------------------------
# 2) Cross-paradigm evasion rates
# ------------------------------------------------------------------

def _compute_evasion_summary(all_scores: pd.DataFrame) -> pd.DataFrame:
    """For each attack_type, what fraction of AI samples evade each detector
    and what fraction evade ALL detectors simultaneously."""
    if "source" not in all_scores.columns:
        return pd.DataFrame()

    df = all_scores.copy()
    df["evaded"] = df["ai_score"] < df["threshold_used"]

    # Keep only AI-source samples.
    try:
        y = encode_source_labels(df["source"])
    except ValueError:
        return pd.DataFrame()
    df = df[y == 1].copy()

    if df.empty:
        return pd.DataFrame()

    detectors = sorted(df["detector_name"].unique().tolist())
    attack_types = sorted(df["attack_type"].unique().tolist())

    rows = []
    for attack in attack_types:
        sub = df[df["attack_type"] == attack]
        if sub.empty:
            continue

        # Build wide table: each id × detector → evaded bool
        wide = sub.pivot_table(index="id", columns="detector_name", values="evaded", aggfunc="max")
        # Drop rows with incomplete detector coverage
        wide = wide.dropna()
        if wide.empty:
            continue

        n = len(wide)
        row = {"attack_type": attack, "n_ai_samples": n}

        all_evade = np.ones(n, dtype=bool)
        for det in detectors:
            if det in wide.columns:
                rate = float(wide[det].mean())
                row[f"evasion_rate_{det}"] = rate
                all_evade &= wide[det].to_numpy(dtype=bool)

        row["cross_paradigm_evasion_rate"] = float(all_evade.mean())
        row["n_evade_all"] = int(all_evade.sum())
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 3) Transferability matrix (attack × detector AUROC drop)
# ------------------------------------------------------------------

def _compute_transferability(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot delta_auroc_vs_clean into attack_type × detector matrix."""
    if "delta_auroc_vs_clean" not in metrics_df.columns:
        return pd.DataFrame()

    matrix = metrics_df.pivot_table(
        index="attack_type",
        columns="detector_name",
        values="delta_auroc_vs_clean",
        aggfunc="mean",
    )
    return matrix.sort_index().sort_index(axis=1)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scores = _load_all_scores(scores_dir)

    # 1) Metrics table
    metrics_df = _compute_metrics_table(all_scores)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path.resolve()}")

    # 2) Evasion summary
    evasion_df = _compute_evasion_summary(all_scores)
    if not evasion_df.empty:
        evasion_path = output_dir / "evasion_summary.csv"
        evasion_df.to_csv(evasion_path, index=False)
        print(f"Saved evasion summary: {evasion_path.resolve()}")

    # 3) Transferability matrix
    transfer_df = _compute_transferability(metrics_df)
    if not transfer_df.empty:
        transfer_path = output_dir / "transferability.csv"
        transfer_df.to_csv(transfer_path)
        print(f"Saved transferability matrix: {transfer_path.resolve()}")


if __name__ == "__main__":
    main()
