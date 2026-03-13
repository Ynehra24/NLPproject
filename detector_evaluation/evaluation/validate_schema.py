"""Validate dataset schema for detector/evaluation pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from detectors.common.io_utils import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate detector input schema")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more csv/jsonl files to validate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = []

    for file_path in args.inputs:
        path = Path(file_path)
        try:
            df = load_dataset(path)
            print(f"[OK] {path} rows={len(df)}")
        except Exception as exc:  # noqa: BLE001
            failures.append((str(path), str(exc)))
            print(f"[FAIL] {path}: {exc}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
