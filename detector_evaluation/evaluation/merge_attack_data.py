"""Merge attack CSVs into one dataset with attack metadata.

Mode A (explicit mapping):
- Provide --mapping CSV with columns file,attack_type

Mode B (auto-discovery):
- Provide --input-dir containing attack CSV files
- Attack type is inferred from filename tokens

Input attack CSV requirements:
- Must contain: text
- Must contain one of:
    - source with values human/ai, or
    - label with values 0/1 (0 -> human, 1 -> ai)

Output columns are aligned to the project schema:
id, text, source, attack_type, generator_model
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_MAPPING_COLUMNS = {"file", "attack_type"}
OUTPUT_COLUMNS = ["id", "text", "source", "attack_type", "generator_model"]
DEFAULT_ATTACK_ALIASES = {
    "none": "none",
    "clean": "none",
    "prompt": "prompt",
    "paraphrase": "paraphrase",
    "gradient": "gradient",
    "char": "char",
    "character": "char",
    "noise": "char",
    "typo": "char",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge attack CSVs with attack mapping")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mapping", help="CSV mapping file -> attack_type")
    group.add_argument("--input-dir", help="Folder containing attack CSV files to auto-merge")
    parser.add_argument("--output", required=True, help="Output merged CSV path")
    parser.add_argument(
        "--id-prefix",
        default="sample",
        help="Prefix used for generated IDs when input files do not contain an id column",
    )
    parser.add_argument(
        "--keep-extra-columns",
        action="store_true",
        help="Keep all extra input columns in output (default keeps only standard schema columns)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --input-dir for CSV files",
    )
    parser.add_argument(
        "--default-attack-type",
        default="none",
        help="Fallback attack type when filename has no recognized attack token",
    )
    return parser.parse_args()


def to_source(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapped = values.map({"0": "human", "1": "ai", "human": "human", "ai": "ai"})
    if mapped.isna().any():
        bad = sorted(values[mapped.isna()].unique().tolist())
        raise ValueError(f"Invalid label/source values found: {bad}")
    return mapped


def resolve_input_path(path_value: str, mapping_path: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (mapping_path.parent / p).resolve()


def load_mapping(mapping_path: Path) -> pd.DataFrame:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    mapping_df = pd.read_csv(mapping_path)
    missing = REQUIRED_MAPPING_COLUMNS - set(mapping_df.columns)
    if missing:
        raise ValueError(
            "Mapping CSV missing required columns: " + ", ".join(sorted(missing))
        )

    if mapping_df.empty:
        raise ValueError("Mapping CSV is empty")

    return mapping_df


def infer_attack_type(file_path: Path, default_attack_type: str) -> str:
    stem = file_path.stem.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]
    for token in tokens:
        if token in DEFAULT_ATTACK_ALIASES:
            return DEFAULT_ATTACK_ALIASES[token]
    return default_attack_type


def build_mapping_from_input_dir(input_dir: Path, recursive: bool, default_attack_type: str) -> pd.DataFrame:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    globber = input_dir.rglob if recursive else input_dir.glob
    files = sorted([p for p in globber("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    rows = []
    for file_path in files:
        attack_type = infer_attack_type(file_path, default_attack_type)
        rows.append(
            {
                "file": str(file_path.resolve()),
                "attack_type": attack_type,
            }
        )

    return pd.DataFrame(rows)


def normalize_input_df(df: pd.DataFrame, attack_type: str, id_prefix: str, row_offset: int) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Input CSV missing required column: text")

    if "source" in df.columns:
        df["source"] = to_source(df["source"])
    elif "label" in df.columns:
        df["source"] = to_source(df["label"])
    else:
        raise ValueError("Input CSV must contain either source or label column")

    if "id" not in df.columns:
        df["id"] = [f"{id_prefix}_{row_offset + i}" for i in range(len(df))]

    df["attack_type"] = str(attack_type)

    if "generator_model" not in df.columns:
        df["generator_model"] = ""

    return df


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()

    if args.mapping:
        mapping_path = Path(args.mapping).resolve()
        mapping_df = load_mapping(mapping_path)
    else:
        input_dir = Path(args.input_dir).resolve()
        mapping_df = build_mapping_from_input_dir(
            input_dir=input_dir,
            recursive=args.recursive,
            default_attack_type=args.default_attack_type,
        )

    merged_parts: List[pd.DataFrame] = []
    total_rows = 0

    for idx, row in mapping_df.iterrows():
        attack_type = str(row["attack_type"]).strip()

        if args.mapping:
            mapping_path = Path(args.mapping).resolve()
            file_path = resolve_input_path(str(row["file"]), mapping_path)
        else:
            file_path = Path(str(row["file"])).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found for row {idx}: {file_path}")

        input_df = pd.read_csv(file_path)
        normalized = normalize_input_df(
            input_df,
            attack_type=attack_type,
            id_prefix=args.id_prefix,
            row_offset=total_rows,
        )

        normalized["_source_file"] = str(file_path)

        merged_parts.append(normalized)
        total_rows += len(normalized)

    merged = pd.concat(merged_parts, ignore_index=True)

    if args.keep_extra_columns:
        ordered_existing = [c for c in OUTPUT_COLUMNS if c in merged.columns]
        extra = [c for c in merged.columns if c not in ordered_existing]
        merged = merged[ordered_existing + extra]
    else:
        merged = merged[OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Saved merged CSV: {output_path}")
    print(f"Rows: {len(merged)}")
    print("Attack type counts:")
    print(merged["attack_type"].value_counts(dropna=False).to_string())

    print("Resolved mapping (file -> attack_type):")
    for _, row in mapping_df.iterrows():
        print(f"- {row['file']} -> {row['attack_type']}")


if __name__ == "__main__":
    main()
