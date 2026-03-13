"""Schema-aware IO and result persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .config import DEFAULT_ATTACK_OWNER, DEFAULT_ATTACK_TYPE, OPTIONAL_COLUMNS, REQUIRED_COLUMNS


def _read_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".jsonl":
        df = _read_jsonl(p)
    else:
        raise ValueError("Supported dataset formats are .csv and .jsonl")

    validate_input_schema(df)
    return normalize_optional_columns(df)


def validate_input_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def normalize_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "attack_type" not in out.columns:
        out["attack_type"] = DEFAULT_ATTACK_TYPE
    out["attack_type"] = out["attack_type"].fillna(DEFAULT_ATTACK_TYPE)

    if "attack_owner" not in out.columns:
        out["attack_owner"] = DEFAULT_ATTACK_OWNER
    out["attack_owner"] = out["attack_owner"].fillna(DEFAULT_ATTACK_OWNER)

    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out["id"] = out["id"].astype(str)
    out["text"] = out["text"].astype(str)
    return out


def save_detector_scores(df: pd.DataFrame, output_path: str | Path) -> None:
    required = ["id", "detector_name", "ai_score", "predicted_label", "threshold_used"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Score dataframe missing columns: {missing}")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
