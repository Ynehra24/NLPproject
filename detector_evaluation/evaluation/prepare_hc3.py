"""Prepare HC3 dataset into detector-compatible CSV splits.

Usage:
  python -m evaluation.prepare_hc3 --as-default-splits
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HC3 dataset for detector pipeline")
    parser.add_argument("--dataset", default="Hello-SimpleAI/HC3")
    parser.add_argument("--config", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--out-raw", default="data/raw/hc3_balanced.csv")
    parser.add_argument("--out-train", default="data/splits/hc3_train.csv")
    parser.add_argument("--out-val", default="data/splits/hc3_val.csv")
    parser.add_argument("--out-test", default="data/splits/hc3_test.csv")
    parser.add_argument(
        "--as-default-splits",
        action="store_true",
        help="Also write outputs to data/splits/train.csv, val.csv, test.csv",
    )
    return parser.parse_args()


def _iter_answers(value) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
        return out
    return []


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _load_hc3_records(dataset_name: str, config_name: str):
    try:
        ds = load_dataset(dataset_name, config_name)
        if "train" in ds:
            return ds["train"]
        first = list(ds.keys())[0]
        return ds[first]
    except RuntimeError as exc:
        # Newer datasets versions may block script-based datasets.
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        filename = "all.jsonl" if config_name == "all" else f"{config_name}.jsonl"
        local_path = hf_hub_download(repo_id=dataset_name, filename=filename, repo_type="dataset")
        return pd.read_json(local_path, lines=True).to_dict(orient="records")


def main() -> None:
    args = parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    split = _load_hc3_records(args.dataset, args.config)

    human_rows: List[dict] = []
    ai_rows: List[dict] = []

    for i, ex in enumerate(split):
        source_domain = str(ex.get("source", "hc3"))

        for j, text in enumerate(_iter_answers(ex.get("human_answers"))):
            text = _normalize_text(text)
            if len(text) < args.min_chars:
                continue
            human_rows.append(
                {
                    "id": f"hc3_h_{i}_{j}",
                    "text": text,
                    "source": "human",
                    "attack_type": "none",
                    "attack_owner": "none",
                    "generator_model": source_domain,
                }
            )

        for j, text in enumerate(_iter_answers(ex.get("chatgpt_answers"))):
            text = _normalize_text(text)
            if len(text) < args.min_chars:
                continue
            ai_rows.append(
                {
                    "id": f"hc3_ai_{i}_{j}",
                    "text": text,
                    "source": "ai",
                    "attack_type": "none",
                    "attack_owner": "none",
                    "generator_model": "chatgpt",
                }
            )

    if not human_rows or not ai_rows:
        raise RuntimeError("No usable samples extracted from HC3")

    max_per_class = args.max_per_class
    if max_per_class and max_per_class > 0:
        human_df = pd.DataFrame(human_rows).sample(min(max_per_class, len(human_rows)), random_state=args.seed)
        ai_df = pd.DataFrame(ai_rows).sample(min(max_per_class, len(ai_rows)), random_state=args.seed)
    else:
        human_df = pd.DataFrame(human_rows)
        ai_df = pd.DataFrame(ai_rows)

    merged = pd.concat([human_df, ai_df], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    y = merged["source"]
    train_df, temp_df = train_test_split(
        merged,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=y,
    )

    temp_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=temp_test_ratio,
        random_state=args.seed,
        stratify=temp_df["source"],
    )

    for p in [args.out_raw, args.out_train, args.out_val, args.out_test]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(args.out_raw, index=False)
    train_df.to_csv(args.out_train, index=False)
    val_df.to_csv(args.out_val, index=False)
    test_df.to_csv(args.out_test, index=False)

    if args.as_default_splits:
        default_train = Path("data/splits/train.csv")
        default_val = Path("data/splits/val.csv")
        default_test = Path("data/splits/test.csv")
        train_df.to_csv(default_train, index=False)
        val_df.to_csv(default_val, index=False)
        test_df.to_csv(default_test, index=False)

    print(f"Extracted human samples: {len(human_rows)}")
    print(f"Extracted ai samples: {len(ai_rows)}")
    print(f"Balanced merged rows: {len(merged)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Saved raw: {Path(args.out_raw).resolve()}")
    print(f"Saved train: {Path(args.out_train).resolve()}")
    print(f"Saved val: {Path(args.out_val).resolve()}")
    print(f"Saved test: {Path(args.out_test).resolve()}")
    if args.as_default_splits:
        print("Also updated default splits: data/splits/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
