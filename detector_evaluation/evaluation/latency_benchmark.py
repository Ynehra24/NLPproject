"""Latency benchmark for detector scoring scripts.

This script times end-to-end detector execution via subprocess calls and reports
wall-clock latency per sample on a shared input file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from detectors.common.io_utils import load_dataset


@dataclass
class BenchmarkTask:
    name: str
    module: str
    args: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detector runtime latency")
    parser.add_argument("--input", required=True, help="Input csv/jsonl")
    parser.add_argument("--output", required=True, help="Output CSV path")
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

    parser.add_argument("--roberta-model-dir", default=None)
    return parser.parse_args()


def _run_task(task: BenchmarkTask, input_path: str, output_path: Path) -> tuple[bool, float, str]:
    cmd = [
        sys.executable,
        "-m",
        task.module,
        "--input",
        input_path,
        "--output",
        str(output_path),
        *task.args,
    ]

    start = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        return True, elapsed, ""
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - start
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        err = stderr if stderr else stdout
        return False, elapsed, err[:1000]


def _build_tasks(args: argparse.Namespace) -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []

    tasks.append(
        BenchmarkTask(
            name="stats_baseline",
            module="detectors.stats_baseline.score",
            args=["--model-name", args.stats_model, "--device", args.device],
        )
    )

    if args.roberta_model_dir:
        tasks.append(
            BenchmarkTask(
                name="roberta_classifier",
                module="detectors.roberta_classifier.infer",
                args=["--model-dir", args.roberta_model_dir, "--device", args.device],
            )
        )

    if args.run_detectgpt:
        tasks.append(
            BenchmarkTask(
                name="detectgpt_style",
                module="detectors.detectgpt.score",
                args=[
                    "--lm-name",
                    args.detectgpt_lm,
                    "--mask-model",
                    args.detectgpt_mask,
                    "--n-perturb",
                    str(args.detectgpt_perturb),
                    "--device",
                    args.device,
                ],
            )
        )

    if args.run_fast_detectgpt:
        tasks.append(
            BenchmarkTask(
                name="fast_detectgpt",
                module="detectors.fast_detectgpt.score",
                args=[
                    "--scoring-model",
                    args.fast_scoring_model,
                    "--reference-model",
                    args.fast_reference_model,
                    "--device",
                    args.device,
                ],
            )
        )

    if args.run_binoculars:
        tasks.append(
            BenchmarkTask(
                name="binoculars",
                module="detectors.binoculars.score",
                args=[
                    "--observer-model",
                    args.binoculars_observer_model,
                    "--performer-model",
                    args.binoculars_performer_model,
                    "--device",
                    args.device,
                ],
            )
        )

    if args.run_watermark:
        tasks.append(
            BenchmarkTask(
                name="kgw_watermark",
                module="detectors.watermark.score",
                args=[
                    "--tokenizer-name",
                    args.watermark_tokenizer,
                    "--gamma",
                    str(args.watermark_gamma),
                    "--hash-key",
                    str(args.watermark_hash_key),
                ],
            )
        )

    return tasks


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    n_samples = int(len(df))
    if n_samples <= 0:
        raise ValueError("Input dataset is empty")

    tasks = _build_tasks(args)
    if not tasks:
        raise RuntimeError("No benchmark tasks selected")

    temp_dir = output_path.parent / "_latency_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in tasks:
        temp_output = temp_dir / f"{task.name}_scores.csv"
        ok, elapsed_s, error_msg = _run_task(task, args.input, temp_output)
        per_sample_ms = (elapsed_s * 1000.0) / n_samples

        rows.append(
            {
                "detector_name": task.name,
                "success": bool(ok),
                "n_samples": n_samples,
                "elapsed_seconds": float(elapsed_s),
                "latency_ms_per_sample": float(per_sample_ms),
                "error": error_msg,
            }
        )
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {task.name}: {elapsed_s:.3f}s total ({per_sample_ms:.3f} ms/sample)")

    out_df = pd.DataFrame(rows).sort_values("latency_ms_per_sample")
    out_df.to_csv(output_path, index=False)
    print(f"Saved latency benchmark: {output_path.resolve()}")


if __name__ == "__main__":
    main()
