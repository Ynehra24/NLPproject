"""Cross-paradigm adversarial evasion analysis.

Computes:
1) per-paradigm evasion rate (fraction of AI samples predicted as human)
2) cross-paradigm evasion rate (fraction of AI samples evading all selected detectors)

Inputs are detector score CSVs from the unified scoring pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from detectors.common.metrics import encode_source_labels


DEFAULT_DETECTORS = [
    "roberta_classifier",
    "fast_detectgpt",
    "binoculars",
    "kgw_watermark",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-paradigm evasion analysis")
    parser.add_argument("--scores-dir", required=True, help="Directory containing *_scores.csv files")
    parser.add_argument("--output-dir", required=True, help="Directory to save evasion analysis tables")
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=DEFAULT_DETECTORS,
        help="Detectors to include in cross-paradigm analysis",
    )
    parser.add_argument(
        "--allow-missing-detectors",
        action="store_true",
        help="If set, continue with available detectors when some requested detectors are absent",
    )
    parser.add_argument(
        "--group-by",
        choices=["attack_type", "attack_owner", "attack_type_owner"],
        default="attack_type_owner",
        help="Condition grouping used for evasion summaries",
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="If set, include clean condition (attack_type=none) in output tables",
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


def _load_and_validate(scores_dir: Path) -> pd.DataFrame:
    files = sorted(scores_dir.glob("*_scores.csv"))
    if not files:
        raise FileNotFoundError(f"No *_scores.csv files found in: {scores_dir}")

    frames: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file)
        required = {"id", "detector_name", "ai_score", "threshold_used", "source"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Skipping {file.name}: missing required columns {missing}")
            continue

        out = df.copy()
        if "attack_type" not in out.columns:
            out["attack_type"] = "none"
        if "attack_owner" not in out.columns:
            out["attack_owner"] = "none"

        out["id"] = out["id"].astype(str)
        out["detector_name"] = out["detector_name"].astype(str)
        out["attack_type"] = out["attack_type"].fillna("none").astype(str)
        out["attack_owner"] = out["attack_owner"].fillna("none").astype(str)
        out["source"] = out["source"].astype(str)
        out["ai_score"] = out["ai_score"].astype(float)
        out["threshold_used"] = out["threshold_used"].astype(float)

        out["evaded"] = out["ai_score"] < out["threshold_used"]
        frames.append(
            out[
                [
                    "id",
                    "detector_name",
                    "source",
                    "attack_type",
                    "attack_owner",
                    "evaded",
                ]
            ]
        )

    if not frames:
        raise RuntimeError("No valid score files found after schema validation")
    return pd.concat(frames, ignore_index=True)


def _build_wide_ai(df: pd.DataFrame, selected_detectors: List[str], group_by: str, include_clean: bool) -> pd.DataFrame:
    subset = df[df["detector_name"].isin(selected_detectors)].copy()
    if subset.empty:
        raise RuntimeError("No rows found for selected detectors")

    # Restrict to AI-source samples only for evasion rates.
    y = encode_source_labels(subset["source"])
    subset = subset[y == 1].copy()

    if not include_clean:
        subset = subset[subset["attack_type"] != "none"].copy()

    if subset.empty:
        raise RuntimeError("No AI rows available for evasion analysis under the current filters")

    subset["condition"] = _condition_label(subset, group_by)

    meta = (
        subset.sort_values(["id", "detector_name"])
        .groupby("id", as_index=False)
        .first()[["id", "attack_type", "attack_owner", "condition"]]
    )

    wide = (
        subset.pivot_table(index="id", columns="detector_name", values="evaded", aggfunc="max")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    merged = meta.merge(wide, on="id", how="inner")

    missing = [d for d in selected_detectors if d not in merged.columns]
    if missing:
        raise RuntimeError(f"Merged table missing detector columns: {missing}")

    n_drop = int(merged[selected_detectors].isna().any(axis=1).sum())
    if n_drop > 0:
        print(f"Dropping {n_drop} ids with incomplete detector coverage")
        merged = merged.dropna(subset=selected_detectors).copy()

    for detector in selected_detectors:
        merged[detector] = merged[detector].astype(bool)

    if merged.empty:
        raise RuntimeError("No rows left after enforcing complete detector coverage")

    return merged


def _summaries(wide_ai: pd.DataFrame, selected_detectors: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    for condition, sub in wide_ai.groupby("condition"):
        n = int(len(sub))
        row = {
            "condition": str(condition),
            "n_ai_samples": n,
        }

        all_evade_mask = np.ones(n, dtype=bool)
        for detector in selected_detectors:
            evade_rate = float(sub[detector].mean())
            row[f"evasion_rate_{detector}"] = evade_rate
            all_evade_mask &= sub[detector].to_numpy(dtype=bool)

        row["cross_paradigm_evasion_rate_all_selected"] = float(all_evade_mask.mean())
        row["n_evade_all_selected"] = int(all_evade_mask.sum())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        by=["cross_paradigm_evasion_rate_all_selected", "n_ai_samples"],
        ascending=[False, False],
    )

    long_rows: List[dict] = []
    for condition, sub in wide_ai.groupby("condition"):
        n = int(len(sub))
        for detector in selected_detectors:
            long_rows.append(
                {
                    "condition": str(condition),
                    "detector_name": detector,
                    "n_ai_samples": n,
                    "evasion_rate": float(sub[detector].mean()),
                }
            )

    per_paradigm_long = pd.DataFrame(long_rows).sort_values(by=["condition", "detector_name"])
    return summary, per_paradigm_long


def main() -> None:
    args = parse_args()
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df = _load_and_validate(scores_dir)

    present = sorted(long_df["detector_name"].unique().tolist())
    requested = list(dict.fromkeys(args.detectors))
    selected = [d for d in requested if d in present]

    if not selected:
        raise RuntimeError(f"Requested detectors not found. requested={requested}, present={present}")

    missing = [d for d in requested if d not in present]
    if missing and not args.allow_missing_detectors:
        raise RuntimeError(
            f"Missing requested detectors: {missing}. Use --allow-missing-detectors to proceed with available detectors."
        )
    if missing:
        print(f"Proceeding without missing detectors: {missing}")

    wide_ai = _build_wide_ai(
        df=long_df,
        selected_detectors=selected,
        group_by=args.group_by,
        include_clean=args.include_clean,
    )

    summary, per_paradigm_long = _summaries(wide_ai, selected)

    wide_path = output_dir / "cross_paradigm_ai_wide.csv"
    summary_path = output_dir / "cross_paradigm_evasion_summary.csv"
    long_path = output_dir / "per_paradigm_evasion_long.csv"

    wide_ai.to_csv(wide_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_paradigm_long.to_csv(long_path, index=False)

    print(f"Selected detectors: {selected}")
    print(f"Saved AI-wide table: {wide_path.resolve()}")
    print(f"Saved cross-paradigm summary: {summary_path.resolve()}")
    print(f"Saved per-paradigm long table: {long_path.resolve()}")


if __name__ == "__main__":
    main()
