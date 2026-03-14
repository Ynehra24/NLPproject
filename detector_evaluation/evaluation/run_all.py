"""Run all implemented detectors on the same input file."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified detector runner")
    parser.add_argument("--input", required=True, help="Input dataset (.csv/.jsonl)")
    parser.add_argument("--output-dir", required=True, help="Where score files are saved")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stats-model", default="gpt2")
    parser.add_argument("--run-detectgpt", action="store_true")
    parser.add_argument("--detectgpt-lm", default="gpt2")
    parser.add_argument("--detectgpt-mask", default="distilroberta-base")
    parser.add_argument("--detectgpt-perturb", type=int, default=8)
    parser.add_argument("--run-fast-detectgpt", action="store_true")
    parser.add_argument("--fast-scoring-model", default="gpt2-medium")
    parser.add_argument("--fast-reference-model", default="gpt2")
    parser.add_argument("--run-binoculars", action="store_true")
    parser.add_argument("--binoculars-observer-model", default="gpt2-medium")
    parser.add_argument("--binoculars-performer-model", default="gpt2")
    parser.add_argument("--run-watermark", action="store_true")
    parser.add_argument("--watermark-tokenizer", default="gpt2")
    parser.add_argument("--watermark-gamma", type=float, default=0.5)
    parser.add_argument("--watermark-hash-key", type=int, default=15485863)
    parser.add_argument("--roberta-model-dir", default=None, help="Run only if fine-tuned dir is provided")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    run_cmd(
        [
            python,
            "-m",
            "detectors.stats_baseline.score",
            "--input",
            args.input,
            "--output",
            str(output_dir / "stats_baseline_scores.csv"),
            "--model-name",
            args.stats_model,
            "--device",
            args.device,
        ]
    )

    if args.roberta_model_dir:
        run_cmd(
            [
                python,
                "-m",
                "detectors.roberta_classifier.infer",
                "--input",
                args.input,
                "--model-dir",
                args.roberta_model_dir,
                "--output",
                str(output_dir / "roberta_classifier_scores.csv"),
                "--device",
                args.device,
            ]
        )

    if args.run_detectgpt:
        run_cmd(
            [
                python,
                "-m",
                "detectors.detectgpt.score",
                "--input",
                args.input,
                "--output",
                str(output_dir / "detectgpt_style_scores.csv"),
                "--lm-name",
                args.detectgpt_lm,
                "--mask-model",
                args.detectgpt_mask,
                "--device",
                args.device,
                "--n-perturb",
                str(args.detectgpt_perturb),
            ]
        )

    if args.run_fast_detectgpt:
        run_cmd(
            [
                python,
                "-m",
                "detectors.fast_detectgpt.score",
                "--input",
                args.input,
                "--output",
                str(output_dir / "fast_detectgpt_scores.csv"),
                "--scoring-model",
                args.fast_scoring_model,
                "--reference-model",
                args.fast_reference_model,
                "--device",
                args.device,
            ]
        )

    if args.run_binoculars:
        run_cmd(
            [
                python,
                "-m",
                "detectors.binoculars.score",
                "--input",
                args.input,
                "--output",
                str(output_dir / "binoculars_scores.csv"),
                "--observer-model",
                args.binoculars_observer_model,
                "--performer-model",
                args.binoculars_performer_model,
                "--device",
                args.device,
            ]
        )

    if args.run_watermark:
        run_cmd(
            [
                python,
                "-m",
                "detectors.watermark.score",
                "--input",
                args.input,
                "--output",
                str(output_dir / "kgw_watermark_scores.csv"),
                "--tokenizer-name",
                args.watermark_tokenizer,
                "--gamma",
                str(args.watermark_gamma),
                "--hash-key",
                str(args.watermark_hash_key),
            ]
        )

    print(f"All requested detector runs complete. Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
