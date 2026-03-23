"""Cross-paradigm disagreement-aware ensemble evaluation.

This module builds a unified feature table from detector score CSVs,
normalizes detector outputs with held-out calibration stats, computes
inter-detector disagreement D, and evaluates ablation variants:
1) mean-of-scores baseline
2) logistic meta-classifier on normalized detector scores
3) logistic meta-classifier on normalized detector scores + D (novelty)
4) oracle upper-bound (train/test on full data; optimistic)

It also validates the disagreement hypothesis using two-sample KS tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from detectors.common.metrics import compute_binary_metrics, encode_source_labels, find_best_threshold


DEFAULT_DETECTORS = [
    "roberta_classifier",
    "fast_detectgpt",
    "binoculars",
    "kgw_watermark",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disagreement-aware ensemble evaluation")
    parser.add_argument("--scores-dir", required=True, help="Directory containing *_scores.csv files")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs are written")
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=DEFAULT_DETECTORS,
        help="Detector names to fuse (must match detector_name values in score CSVs)",
    )
    parser.add_argument(
        "--allow-missing-detectors",
        action="store_true",
        help="If set, silently drop requested detectors not found in scores-dir",
    )
    parser.add_argument(
        "--calibration-attack-type",
        default="none",
        help="Calibration subset attack_type label for min-max normalization",
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-calibration-samples", type=int, default=20)
    return parser.parse_args()


def _load_score_files(scores_dir: Path) -> pd.DataFrame:
    files = sorted(scores_dir.glob("*_scores.csv"))
    if not files:
        raise FileNotFoundError(f"No *_scores.csv files found in: {scores_dir}")

    frames: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file)
        required = {"id", "detector_name", "ai_score"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Skipping {file.name}: missing required columns {missing}")
            continue

        slim = df.copy()
        if "source" not in slim.columns:
            slim["source"] = np.nan
        if "attack_type" not in slim.columns:
            slim["attack_type"] = "none"
        if "attack_owner" not in slim.columns:
            slim["attack_owner"] = "none"

        slim["id"] = slim["id"].astype(str)
        slim["detector_name"] = slim["detector_name"].astype(str)
        slim["attack_type"] = slim["attack_type"].fillna("none").astype(str)
        slim["attack_owner"] = slim["attack_owner"].fillna("none").astype(str)
        frames.append(
            slim[
                [
                    "id",
                    "detector_name",
                    "ai_score",
                    "source",
                    "attack_type",
                    "attack_owner",
                ]
            ]
        )

    if not frames:
        raise RuntimeError("No valid score files available after schema checks")
    return pd.concat(frames, ignore_index=True)


def _build_feature_table(long_df: pd.DataFrame, detector_names: List[str]) -> pd.DataFrame:
    subset = long_df[long_df["detector_name"].isin(detector_names)].copy()
    if subset.empty:
        raise RuntimeError("None of the requested detectors are present in provided score files")

    # Keep one metadata row per id. Existing pipeline already uses one row per text id.
    meta = (
        subset.sort_values(["id", "detector_name"])
        .groupby("id", as_index=False)
        .first()[["id", "source", "attack_type", "attack_owner"]]
    )

    wide = (
        subset.pivot_table(index="id", columns="detector_name", values="ai_score", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    merged = meta.merge(wide, on="id", how="inner")
    missing_rows = merged[detector_names].isna().any(axis=1).sum()
    if missing_rows > 0:
        print(f"Dropping {missing_rows} rows with incomplete detector coverage")
        merged = merged.dropna(subset=detector_names).copy()

    if merged.empty:
        raise RuntimeError("No rows left after requiring complete detector scores")
    return merged


def _fit_minmax(calibration_df: pd.DataFrame, detector_names: List[str]) -> Dict[str, tuple[float, float]]:
    params: Dict[str, tuple[float, float]] = {}
    for detector in detector_names:
        col = calibration_df[detector].to_numpy(dtype=float)
        lo = float(np.min(col))
        hi = float(np.max(col))
        params[detector] = (lo, hi)
    return params


def _apply_minmax(df: pd.DataFrame, detector_names: List[str], params: Dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for detector in detector_names:
        lo, hi = params[detector]
        denom = hi - lo
        if denom <= 1e-12:
            out[f"s_{detector}"] = 0.5
        else:
            out[f"s_{detector}"] = ((out[detector] - lo) / denom).clip(0.0, 1.0)
    norm_cols = [f"s_{d}" for d in detector_names]
    out["disagreement_var"] = out[norm_cols].var(axis=1, ddof=0)
    out["mean_score"] = out[norm_cols].mean(axis=1)
    return out


def _metrics_with_train_threshold(
    y_train: np.ndarray,
    score_train: np.ndarray,
    y_eval: np.ndarray,
    score_eval: np.ndarray,
) -> tuple[dict, float]:
    threshold = float(find_best_threshold(y_train, score_train))
    return compute_binary_metrics(y_eval, score_eval, threshold=threshold), threshold


def _evaluate_ablation(feature_df: pd.DataFrame, detector_names: List[str], test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_df["source"].isna().any():
        raise ValueError("'source' labels are required for ensemble training/evaluation")

    y = encode_source_labels(feature_df["source"])
    norm_cols = [f"s_{d}" for d in detector_names]

    X_base = feature_df[norm_cols].to_numpy(dtype=float)
    X_full = feature_df[norm_cols + ["disagreement_var"]].to_numpy(dtype=float)

    idx = np.arange(len(feature_df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    Xb_train = X_base[train_idx]
    Xb_test = X_base[test_idx]
    Xf_train = X_full[train_idx]
    Xf_test = X_full[test_idx]

    pred_rows: List[pd.DataFrame] = []
    metric_rows: List[dict] = []

    # Mean-of-scores baseline.
    mean_train = Xb_train.mean(axis=1)
    mean_test = Xb_test.mean(axis=1)
    mean_metrics, mean_threshold = _metrics_with_train_threshold(y_train, mean_train, y_test, mean_test)
    metric_rows.append({"variant": "mean_of_scores", "split": "test", **mean_metrics, "threshold": mean_threshold})

    pred_rows.append(
        pd.DataFrame(
            {
                "id": feature_df.iloc[test_idx]["id"].to_numpy(),
                "source": feature_df.iloc[test_idx]["source"].to_numpy(),
                "attack_type": feature_df.iloc[test_idx]["attack_type"].to_numpy(),
                "attack_owner": feature_df.iloc[test_idx]["attack_owner"].to_numpy(),
                "variant": "mean_of_scores",
                "score": mean_test,
            }
        )
    )

    # Logistic on normalized detector scores (no disagreement).
    lr_base = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    lr_base.fit(Xb_train, y_train)
    base_train = lr_base.predict_proba(Xb_train)[:, 1]
    base_test = lr_base.predict_proba(Xb_test)[:, 1]
    base_metrics, base_threshold = _metrics_with_train_threshold(y_train, base_train, y_test, base_test)
    metric_rows.append({"variant": "logreg_base", "split": "test", **base_metrics, "threshold": base_threshold})

    pred_rows.append(
        pd.DataFrame(
            {
                "id": feature_df.iloc[test_idx]["id"].to_numpy(),
                "source": feature_df.iloc[test_idx]["source"].to_numpy(),
                "attack_type": feature_df.iloc[test_idx]["attack_type"].to_numpy(),
                "attack_owner": feature_df.iloc[test_idx]["attack_owner"].to_numpy(),
                "variant": "logreg_base",
                "score": base_test,
            }
        )
    )

    # Logistic on normalized detector scores + disagreement variance (novelty).
    lr_full = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    lr_full.fit(Xf_train, y_train)
    full_train = lr_full.predict_proba(Xf_train)[:, 1]
    full_test = lr_full.predict_proba(Xf_test)[:, 1]
    full_metrics, full_threshold = _metrics_with_train_threshold(y_train, full_train, y_test, full_test)
    metric_rows.append(
        {
            "variant": "logreg_disagreement_augmented",
            "split": "test",
            **full_metrics,
            "threshold": full_threshold,
        }
    )

    pred_rows.append(
        pd.DataFrame(
            {
                "id": feature_df.iloc[test_idx]["id"].to_numpy(),
                "source": feature_df.iloc[test_idx]["source"].to_numpy(),
                "attack_type": feature_df.iloc[test_idx]["attack_type"].to_numpy(),
                "attack_owner": feature_df.iloc[test_idx]["attack_owner"].to_numpy(),
                "variant": "logreg_disagreement_augmented",
                "score": full_test,
            }
        )
    )

    # Oracle optimistic upper-bound.
    oracle = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    oracle.fit(Xf_train, y_train)
    oracle_all = oracle.predict_proba(X_full)[:, 1]
    oracle_threshold = float(find_best_threshold(y, oracle_all))
    oracle_metrics = compute_binary_metrics(y, oracle_all, threshold=oracle_threshold)
    metric_rows.append(
        {
            "variant": "oracle_upper_bound",
            "split": "all_data_self_eval",
            **oracle_metrics,
            "threshold": oracle_threshold,
        }
    )

    pred_rows.append(
        pd.DataFrame(
            {
                "id": feature_df["id"].to_numpy(),
                "source": feature_df["source"].to_numpy(),
                "attack_type": feature_df["attack_type"].to_numpy(),
                "attack_owner": feature_df["attack_owner"].to_numpy(),
                "variant": "oracle_upper_bound",
                "score": oracle_all,
            }
        )
    )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(pred_rows, ignore_index=True)
    return metrics_df, predictions_df


def _ks_disagreement(feature_df: pd.DataFrame) -> pd.DataFrame:
    clean_human = feature_df[(feature_df["source"].astype(str).str.lower() == "human") & (feature_df["attack_type"] == "none")]
    clean_ai = feature_df[(feature_df["source"].astype(str).str.lower() == "ai") & (feature_df["attack_type"] == "none")]
    adversarial = feature_df[(feature_df["source"].astype(str).str.lower() == "ai") & (feature_df["attack_type"] != "none")]

    pairs = [
        ("adversarial_vs_clean_human", adversarial, clean_human),
        ("adversarial_vs_clean_ai", adversarial, clean_ai),
    ]

    rows: List[dict] = []
    for name, left, right in pairs:
        n_left = int(len(left))
        n_right = int(len(right))
        if n_left < 2 or n_right < 2:
            rows.append(
                {
                    "comparison": name,
                    "n_left": n_left,
                    "n_right": n_right,
                    "ks_statistic": np.nan,
                    "p_value": np.nan,
                    "significant_p_lt_0_05": False,
                    "note": "Insufficient samples for KS test",
                }
            )
            continue

        result = ks_2samp(left["disagreement_var"].to_numpy(dtype=float), right["disagreement_var"].to_numpy(dtype=float))
        rows.append(
            {
                "comparison": name,
                "n_left": n_left,
                "n_right": n_right,
                "ks_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "significant_p_lt_0_05": bool(result.pvalue < 0.05),
                "note": "",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df = _load_score_files(scores_dir)

    present = sorted(long_df["detector_name"].unique().tolist())
    requested = list(dict.fromkeys(args.detectors))
    selected = [d for d in requested if d in present]

    if not selected:
        raise RuntimeError(f"Requested detectors not found. requested={requested}, present={present}")

    missing = [d for d in requested if d not in present]
    if missing and not args.allow_missing_detectors:
        raise RuntimeError(
            f"Missing requested detectors: {missing}. Use --allow-missing-detectors to continue with available detectors."
        )

    if missing:
        print(f"Proceeding without missing detectors: {missing}")

    feature_df = _build_feature_table(long_df, selected)

    calibration_df = feature_df[feature_df["attack_type"] == args.calibration_attack_type].copy()
    if len(calibration_df) < args.min_calibration_samples:
        print(
            "Calibration subset too small. Falling back to full feature table for min-max normalization. "
            f"subset={len(calibration_df)}"
        )
        calibration_df = feature_df

    params = _fit_minmax(calibration_df, selected)
    feature_df = _apply_minmax(feature_df, selected, params)

    metrics_df, predictions_df = _evaluate_ablation(
        feature_df=feature_df,
        detector_names=selected,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    ks_df = _ks_disagreement(feature_df)

    params_rows = [
        {
            "detector_name": detector,
            "calibration_min": params[detector][0],
            "calibration_max": params[detector][1],
        }
        for detector in selected
    ]
    params_df = pd.DataFrame(params_rows)

    feature_path = output_dir / "ensemble_feature_table.csv"
    metrics_path = output_dir / "ensemble_ablation_metrics.csv"
    preds_path = output_dir / "ensemble_predictions.csv"
    ks_path = output_dir / "disagreement_ks_tests.csv"
    norm_path = output_dir / "normalization_params.csv"

    feature_df.to_csv(feature_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(preds_path, index=False)
    ks_df.to_csv(ks_path, index=False)
    params_df.to_csv(norm_path, index=False)

    print(f"Selected detectors: {selected}")
    print(f"Saved feature table: {feature_path.resolve()}")
    print(f"Saved ablation metrics: {metrics_path.resolve()}")
    print(f"Saved predictions: {preds_path.resolve()}")
    print(f"Saved KS tests: {ks_path.resolve()}")
    print(f"Saved normalization params: {norm_path.resolve()}")


if __name__ == "__main__":
    main()
