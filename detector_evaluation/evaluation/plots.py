"""Generate summary plots from aggregated detector metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create evaluation plots")
    parser.add_argument("--metrics", required=True, help="Aggregated metrics csv")
    parser.add_argument("--output-dir", required=True, help="Directory for figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

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

    print(f"Plots written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
