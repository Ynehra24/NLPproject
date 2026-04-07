"""Convert paired attack CSV into detector-ready long format.

Input (paired): one row per pair with original and humanized text columns.
Output (long): two rows per pair (original/humanized) with standard detector schema.

Example:
  python -m evaluation.paired_to_long \
    --input teammate_pairs.csv \
    --output paired_long.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert paired CSV to long detector format")
    parser.add_argument("--input", required=True, help="Input paired CSV")
    parser.add_argument("--output", required=True, help="Output long CSV")

    parser.add_argument("--pair-id-col", default="pair_id")
    parser.add_argument("--original-text-col", default="original_text")
    parser.add_argument("--humanized-text-col", default="humanized_text")

    parser.add_argument("--attack-type-col", default="attack_type")
    parser.add_argument("--generator-model-col", default="generator_model")

    parser.add_argument(
        "--source-original-col",
        default=None,
        help="Optional source column name for original rows. If absent, uses default.",
    )
    parser.add_argument(
        "--source-humanized-col",
        default=None,
        help="Optional source column name for humanized rows. If absent, uses default.",
    )
    parser.add_argument("--default-source-original", default="ai")
    parser.add_argument("--default-source-humanized", default="ai")

    parser.add_argument("--default-attack-type", default="none")
    parser.add_argument("--default-generator-model", default="")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input paired CSV missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input paired CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    require_columns(df, [args.pair_id_col, args.original_text_col, args.humanized_text_col])

    long_rows: list[dict] = []
    for _, row in df.iterrows():
        pair_id = str(row[args.pair_id_col])
        attack_type = str(row.get(args.attack_type_col, args.default_attack_type))
        generator_model = str(row.get(args.generator_model_col, args.default_generator_model))

        if args.source_original_col and args.source_original_col in df.columns:
            source_original = str(row[args.source_original_col])
        else:
            source_original = str(args.default_source_original)

        if args.source_humanized_col and args.source_humanized_col in df.columns:
            source_humanized = str(row[args.source_humanized_col])
        else:
            source_humanized = str(args.default_source_humanized)

        long_rows.append(
            {
                "id": f"{pair_id}__orig",
                "pair_id": pair_id,
                "variant": "original",
                "text": str(row[args.original_text_col]),
                "source": source_original,
                "attack_type": attack_type,
                "generator_model": generator_model,
            }
        )
        long_rows.append(
            {
                "id": f"{pair_id}__hum",
                "pair_id": pair_id,
                "variant": "humanized",
                "text": str(row[args.humanized_text_col]),
                "source": source_humanized,
                "attack_type": attack_type,
                "generator_model": generator_model,
            }
        )

    out_df = pd.DataFrame(long_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved long-format paired dataset: {output_path.resolve()}")
    print(f"Pairs: {len(df)}")
    print(f"Rows: {len(out_df)}")
    print("Attack type counts:")
    print(out_df["attack_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
