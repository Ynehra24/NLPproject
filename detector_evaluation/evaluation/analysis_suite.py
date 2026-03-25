"""One-command analysis suite.

This script automates the full post-training analysis workflow:
1) run all detectors on train/val/test
2) aggregate metrics and generate plots per split
3) combine split metrics into one table
4) compute detector mean/std summary across splits
5) run extra analyses (transferability, watermark robustness,
   cross-paradigm evasion, disagreement ensemble, latency benchmark)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full post-training detector analysis suite")
    parser.add_argument("--train", default="data/splits/train.csv", help="Train split path")
    parser.add_argument("--val", default="data/splits/val.csv", help="Validation split path")
    parser.add_argument("--test", default="data/splits/test.csv", help="Test split path")

    parser.add_argument("--model-dir", default="results/roberta_model", help="Fine-tuned RoBERTa model directory")
    parser.add_argument("--device", default="cpu", help="Device for detector scoring runs (cpu/cuda)")
    parser.add_argument("--detectgpt-perturb", type=int, default=2, help="DetectGPT perturbations for speed/quality")

    parser.add_argument(
        "--output-root",
        default="results/analysis_suite",
        help="Root output directory for all generated artifacts",
    )

    parser.add_argument(
        "--skip-extra-insights",
        action="store_true",
        help="Skip transferability/watermark/cross-paradigm/disagreement/latency analyses",
    )
    parser.add_argument("--latency-device", default="cpu", help="Device for latency benchmark runs")
    parser.add_argument("--cross-group-by", default="attack_type_owner", choices=["attack_type", "attack_owner", "attack_type_owner"])
    parser.add_argument("--transfer-group-by", default="attack_type", choices=["attack_type", "attack_owner", "attack_type_owner"])
    parser.add_argument("--watermark-group-by", default="attack_type", choices=["attack_type", "attack_owner", "attack_type_owner"])

    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_full_detector_pass(
    split_name: str,
    split_path: str,
    model_dir: str,
    device: str,
    detectgpt_perturb: int,
    output_root: Path,
) -> Dict[str, Path]:
    python = sys.executable

    scores_dir = output_root / "scores" / split_name
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures" / split_name

    metrics_csv = tables_dir / f"metrics_{split_name}.csv"

    scores_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            python,
            "-m",
            "evaluation.run_all",
            "--input",
            split_path,
            "--output-dir",
            str(scores_dir),
            "--device",
            device,
            "--roberta-model-dir",
            model_dir,
            "--run-detectgpt",
            "--detectgpt-perturb",
            str(detectgpt_perturb),
            "--run-fast-detectgpt",
            "--run-binoculars",
            "--run-watermark",
        ]
    )

    run_cmd(
        [
            python,
            "-m",
            "evaluation.aggregate_results",
            "--scores-dir",
            str(scores_dir),
            "--output",
            str(metrics_csv),
        ]
    )

    run_cmd(
        [
            python,
            "-m",
            "evaluation.plots",
            "--metrics",
            str(metrics_csv),
            "--output-dir",
            str(figures_dir),
        ]
    )

    return {
        "scores_dir": scores_dir,
        "metrics_csv": metrics_csv,
        "figures_dir": figures_dir,
    }


def build_combined_tables(metrics_by_split: Dict[str, Path], output_root: Path) -> Dict[str, Path]:
    rows = []
    for split_name, metrics_path in metrics_by_split.items():
        df = pd.read_csv(metrics_path)
        df["split"] = split_name
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True)

    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    combined_path = tables_dir / "metrics_all_splits.csv"
    combined.to_csv(combined_path, index=False)

    clean = combined[combined["attack_type"].astype(str) == "none"].copy()
    summary = (
        clean.groupby("detector_name", as_index=False)
        .agg(
            splits_covered=("split", "nunique"),
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1_ai=("f1_ai", "mean"),
            std_f1_ai=("f1_ai", "std"),
            mean_attack_success_rate=("attack_success_rate", "mean"),
            std_attack_success_rate=("attack_success_rate", "std"),
        )
        .sort_values(["mean_auroc", "mean_accuracy"], ascending=False)
    )

    summary_path = tables_dir / "detector_summary_across_splits.csv"
    summary.to_csv(summary_path, index=False)

    return {
        "combined": combined_path,
        "summary": summary_path,
    }


def run_extra_insights(args: argparse.Namespace, test_scores_dir: Path, output_root: Path) -> Dict[str, Path]:
    python = sys.executable

    out = {}
    insights_dir = output_root / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)

    transfer_dir = insights_dir / "transferability"
    run_cmd(
        [
            python,
            "-m",
            "evaluation.transferability",
            "--scores-dir",
            str(test_scores_dir),
            "--output-dir",
            str(transfer_dir),
            "--group-by",
            args.transfer_group_by,
        ]
    )
    out["transferability_dir"] = transfer_dir

    watermark_path = insights_dir / "watermark_robustness.csv"
    run_cmd(
        [
            python,
            "-m",
            "evaluation.watermark_robustness",
            "--scores-dir",
            str(test_scores_dir),
            "--output",
            str(watermark_path),
            "--group-by",
            args.watermark_group_by,
        ]
    )
    out["watermark_robustness_csv"] = watermark_path

    cross_dir = insights_dir / "cross_paradigm_evasion"
    run_cmd(
        [
            python,
            "-m",
            "evaluation.cross_paradigm_evasion",
            "--scores-dir",
            str(test_scores_dir),
            "--output-dir",
            str(cross_dir),
            "--group-by",
            args.cross_group_by,
            "--include-clean",
            "--allow-missing-detectors",
        ]
    )
    out["cross_paradigm_dir"] = cross_dir

    disagreement_dir = insights_dir / "disagreement_ensemble"
    run_cmd(
        [
            python,
            "-m",
            "evaluation.disagreement_ensemble",
            "--scores-dir",
            str(test_scores_dir),
            "--output-dir",
            str(disagreement_dir),
            "--allow-missing-detectors",
        ]
    )
    out["disagreement_dir"] = disagreement_dir

    latency_csv = insights_dir / "latency_benchmark.csv"
    run_cmd(
        [
            python,
            "-m",
            "evaluation.latency_benchmark",
            "--input",
            args.test,
            "--output",
            str(latency_csv),
            "--device",
            args.latency_device,
            "--roberta-model-dir",
            args.model_dir,
            "--run-detectgpt",
            "--detectgpt-perturb",
            str(args.detectgpt_perturb),
            "--run-fast-detectgpt",
            "--run-binoculars",
            "--run-watermark",
        ]
    )
    out["latency_csv"] = latency_csv

    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }

    split_outputs: Dict[str, Dict[str, Path]] = {}
    for split_name, split_path in split_paths.items():
        split_outputs[split_name] = run_full_detector_pass(
            split_name=split_name,
            split_path=split_path,
            model_dir=args.model_dir,
            device=args.device,
            detectgpt_perturb=args.detectgpt_perturb,
            output_root=output_root,
        )

    metrics_by_split = {k: v["metrics_csv"] for k, v in split_outputs.items()}
    table_outputs = build_combined_tables(metrics_by_split, output_root)

    extra_outputs: Dict[str, Path] = {}
    if not args.skip_extra_insights:
        extra_outputs = run_extra_insights(
            args=args,
            test_scores_dir=split_outputs["test"]["scores_dir"],
            output_root=output_root,
        )

    manifest = {
        "config": {
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "model_dir": args.model_dir,
            "device": args.device,
            "detectgpt_perturb": args.detectgpt_perturb,
            "skip_extra_insights": args.skip_extra_insights,
            "latency_device": args.latency_device,
        },
        "split_outputs": {
            split: {
                "scores_dir": str(paths["scores_dir"]),
                "metrics_csv": str(paths["metrics_csv"]),
                "figures_dir": str(paths["figures_dir"]),
            }
            for split, paths in split_outputs.items()
        },
        "tables": {k: str(v) for k, v in table_outputs.items()},
        "extra_outputs": {k: str(v) for k, v in extra_outputs.items()},
    }

    manifest_path = output_root / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Analysis suite complete.")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Combined metrics: {table_outputs['combined'].resolve()}")
    print(f"Summary table: {table_outputs['summary'].resolve()}")


if __name__ == "__main__":
    main()
