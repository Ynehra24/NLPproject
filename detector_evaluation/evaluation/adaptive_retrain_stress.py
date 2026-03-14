"""Adaptive retraining stress test for RoBERTa detector robustness.

The loop performs iterative train -> evaluate on attack set -> mine hard examples
-> augment train set -> retrain for the next round.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.common.metrics import compute_binary_metrics, encode_source_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive retraining stress test")
    parser.add_argument("--train", required=True, help="Initial train split (.csv/.jsonl)")
    parser.add_argument("--val", required=True, help="Validation split (.csv/.jsonl)")
    parser.add_argument("--attack-eval", required=True, help="Attacked evaluation split with source labels")
    parser.add_argument("--work-dir", required=True, help="Output working directory")
    parser.add_argument("--rounds", type=int, default=3, help="Number of adaptive rounds")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-hard-per-class", type=int, default=200)
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_as_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".csv":
        raise ValueError("Adaptive retraining internal files must be .csv")
    df.to_csv(path, index=False)


def _evaluate_predictions(pred_df: pd.DataFrame) -> dict:
    y_true = encode_source_labels(pred_df["source"])
    y_score = pred_df["ai_score"].to_numpy(dtype=float)
    threshold = float(pred_df["threshold_used"].iloc[0]) if "threshold_used" in pred_df.columns else 0.5
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)

    y_pred = (y_score >= threshold).astype(int)
    ai_mask = y_true == 1
    hard_fn = int(((y_pred == 0) & ai_mask).sum())
    hard_fp = int(((y_pred == 1) & (y_true == 0)).sum())

    return {
        "threshold": threshold,
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "accuracy": float(metrics["accuracy"]),
        "f1_ai": float(metrics["f1_ai"]),
        "tpr_at_fpr_1": float(metrics["tpr_at_fpr_1"]),
        "tpr_at_fpr_5": float(metrics["tpr_at_fpr_5"]),
        "hard_false_negatives_ai": hard_fn,
        "hard_false_positives_human": hard_fp,
    }


def _mine_hard_examples(pred_df: pd.DataFrame, max_hard_per_class: int) -> pd.DataFrame:
    threshold = float(pred_df["threshold_used"].iloc[0]) if "threshold_used" in pred_df.columns else 0.5
    y_true = encode_source_labels(pred_df["source"])
    y_score = pred_df["ai_score"].to_numpy(dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    df = pred_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["confidence_gap"] = np.abs(y_score - threshold)

    # Hard false negatives (AI predicted as human) and false positives.
    hard_fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].sort_values("confidence_gap", ascending=False)
    hard_fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].sort_values("confidence_gap", ascending=False)

    selected = pd.concat(
        [hard_fn.head(max_hard_per_class), hard_fp.head(max_hard_per_class)],
        ignore_index=True,
    )

    if selected.empty:
        return selected

    keep_cols = [c for c in ["id", "text", "source", "attack_type", "attack_owner", "generator_model"] if c in selected.columns]
    mined = selected[keep_cols].copy()
    mined["id"] = mined["id"].astype(str) + "__adaptive"
    return mined


def main() -> None:
    args = parse_args()
    python = sys.executable

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train)
    val_path = Path(args.val)
    attack_eval_path = Path(args.attack_eval)

    current_train_df = _load_as_df(train_path)
    if "source" not in current_train_df.columns or current_train_df["source"].isna().any():
        raise ValueError("Train data must contain non-null 'source' labels")

    if "text" not in current_train_df.columns:
        raise ValueError("Train data must contain 'text' column")

    round_rows: list[dict] = []
    current_train_csv = work_dir / "round0_train.csv"
    _save_df(current_train_df, current_train_csv)

    for round_idx in range(1, args.rounds + 1):
        round_dir = work_dir / f"round_{round_idx}"
        model_dir = round_dir / "roberta_model"
        pred_csv = round_dir / "attack_eval_roberta_scores.csv"
        round_dir.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [
                python,
                "-m",
                "detectors.roberta_classifier.train",
                "--train",
                str(current_train_csv),
                "--val",
                str(val_path),
                "--output-dir",
                str(model_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
            ]
        )

        run_cmd(
            [
                python,
                "-m",
                "detectors.roberta_classifier.infer",
                "--input",
                str(attack_eval_path),
                "--model-dir",
                str(model_dir),
                "--output",
                str(pred_csv),
                "--batch-size",
                str(args.batch_size),
                "--device",
                str(args.device),
            ]
        )

        pred_df = pd.read_csv(pred_csv)
        if "source" not in pred_df.columns or pred_df["source"].isna().any():
            raise ValueError("Attack eval file must include non-null source labels for adaptive retraining")

        metrics = _evaluate_predictions(pred_df)
        metrics["round"] = round_idx
        metrics["train_size_before_mining"] = int(len(current_train_df))

        hard_examples = _mine_hard_examples(pred_df, max_hard_per_class=args.max_hard_per_class)
        metrics["mined_hard_examples"] = int(len(hard_examples))

        if not hard_examples.empty:
            next_train = pd.concat([current_train_df, hard_examples], ignore_index=True)
            next_train = next_train.drop_duplicates(subset=["id", "text"], keep="first")
        else:
            next_train = current_train_df.copy()

        metrics["train_size_after_mining"] = int(len(next_train))
        round_rows.append(metrics)

        next_train_csv = work_dir / f"round{round_idx}_train.csv"
        _save_df(next_train, next_train_csv)
        current_train_df = next_train
        current_train_csv = next_train_csv

    summary_df = pd.DataFrame(round_rows).sort_values("round")
    summary_csv = work_dir / "adaptive_retrain_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved adaptive retraining summary: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
