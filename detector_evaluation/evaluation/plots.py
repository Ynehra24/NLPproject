"""Generate summary plots from aggregated detector metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create evaluation plots")
    parser.add_argument("--metrics", required=True, help="Aggregated metrics csv")
    parser.add_argument("--output-dir", required=True, help="Directory for figures")
    parser.add_argument(
        "--scores-dir",
        default=None,
        help="Optional directory containing *_scores.csv for ROC/confusion plots",
    )
    return parser.parse_args()


def _load_scores(scores_dir: Path) -> pd.DataFrame:
    files = sorted(scores_dir.glob("*_scores.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping unreadable score file {file.name}: {exc}")
            continue

        required = {"detector_name", "ai_score"}
        if not required.issubset(df.columns):
            print(f"Skipping {file.name}: missing required columns {sorted(required - set(df.columns))}")
            continue

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _to_binary_source(series: pd.Series) -> np.ndarray:
    src = series.astype(str).str.strip().str.lower()
    mapped = src.map({"human": 0, "ai": 1})
    return mapped.to_numpy(dtype=float)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    if "auroc" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="detector_name", y="auroc", hue="attack_type")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_dir / "auroc_by_detector_and_attack.png", dpi=200)
        plt.close()

    if "attack_success_rate" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="detector_name", y="attack_success_rate", hue="attack_type")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_dir / "asr_by_detector_and_attack.png", dpi=200)
        plt.close()

    if args.scores_dir:
        scores_dir = Path(args.scores_dir)
        long_df = _load_scores(scores_dir)
        if not long_df.empty and "source" in long_df.columns:
            valid = long_df.copy()
            valid["y_true"] = _to_binary_source(valid["source"])
            valid = valid[np.isfinite(valid["y_true"])].copy()
            valid["y_true"] = valid["y_true"].astype(int)

            # ROC curves by detector.
            plt.figure(figsize=(8, 6))
            plotted = False
            for detector_name, sub in valid.groupby("detector_name"):
                if sub["y_true"].nunique() < 2:
                    continue
                fpr, tpr, _ = roc_curve(sub["y_true"], sub["ai_score"].astype(float))
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2, label=f"{detector_name} (AUC={roc_auc:.3f})")
                plotted = True

            if plotted:
                plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curves by Detector")
                plt.legend(loc="lower right", fontsize=8)
                plt.tight_layout()
                plt.savefig(out_dir / "roc_curves_by_detector.png", dpi=200)
                plt.close()
            else:
                plt.close()

            # Confusion matrix per detector (aggregated across attack types).
            detectors = sorted(valid["detector_name"].astype(str).unique().tolist())
            if detectors:
                cols = min(3, len(detectors))
                rows = math.ceil(len(detectors) / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                if not isinstance(axes, np.ndarray):
                    axes = np.array([axes])
                axes = axes.flatten()

                for idx, detector_name in enumerate(detectors):
                    ax = axes[idx]
                    sub = valid[valid["detector_name"] == detector_name].copy()
                    if sub.empty:
                        ax.set_visible(False)
                        continue

                    if "predicted_label" in sub.columns:
                        pred = sub["predicted_label"].astype(str).str.strip().str.lower().map({"human": 0, "ai": 1})
                        if pred.isna().any():
                            threshold = float(sub["threshold_used"].median()) if "threshold_used" in sub.columns else 0.5
                            y_pred = (sub["ai_score"].astype(float) >= threshold).astype(int)
                        else:
                            y_pred = pred.astype(int)
                    else:
                        threshold = float(sub["threshold_used"].median()) if "threshold_used" in sub.columns else 0.5
                        y_pred = (sub["ai_score"].astype(float) >= threshold).astype(int)

                    cm = confusion_matrix(sub["y_true"], y_pred, labels=[0, 1])
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        cbar=False,
                        ax=ax,
                        xticklabels=["pred_human", "pred_ai"],
                        yticklabels=["true_human", "true_ai"],
                    )
                    ax.set_title(str(detector_name))
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")

                for j in range(len(detectors), len(axes)):
                    axes[j].set_visible(False)

                fig.suptitle("Confusion Matrices by Detector", y=1.02)
                fig.tight_layout()
                fig.savefig(out_dir / "confusion_matrices_by_detector.png", dpi=200, bbox_inches="tight")
                plt.close(fig)

    print(f"Plots written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
