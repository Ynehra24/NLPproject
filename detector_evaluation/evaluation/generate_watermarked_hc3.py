"""Generate KGW-watermarked HC3 variants.

This creates a detector-compatible dataset with paired HC3 human answers and
locally generated KGW-watermarked AI answers. KGW needs this because the
watermark signal only exists when text is generated with the KGW sampler.

Example:
  python -m evaluation.generate_watermarked_hc3 --n-prompts 500 --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from detectors.watermark.score import WatermarkDetector, WatermarkGenerator
from evaluation.prepare_hc3 import _iter_answers, _load_hc3_records, _normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KGW-watermarked HC3 variants")
    parser.add_argument("--dataset", default="Hello-SimpleAI/HC3")
    parser.add_argument("--config", default="all")
    parser.add_argument("--model-name", default="gpt2", help="Local causal LM used for KGW generation")
    parser.add_argument("--n-prompts", type=int, default=500, help="Number of HC3 prompts to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.5, help="Green-list fraction; must match KGW detector")
    parser.add_argument("--delta", type=float, default=2.0, help="Logit boost for green-list tokens")
    parser.add_argument("--hash-key", type=int, default=15485863)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or a torch device like 'cuda'")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--out-raw", default="data/raw/hc3_watermarked_kgw.csv")
    parser.add_argument("--out-train", default="data/splits/kgw_train.csv")
    parser.add_argument("--out-val", default="data/splits/kgw_val.csv")
    parser.add_argument("--out-test", default="data/splits/kgw_test.csv")
    parser.add_argument(
        "--paired-output",
        default="data/processed/hc3_watermarked_pairs.csv",
        help="Optional paired CSV for attack-style evaluation",
    )
    parser.add_argument(
        "--as-default-splits",
        action="store_true",
        help="Also write watermarked splits to data/splits/train.csv, val.csv, test.csv",
    )
    parser.add_argument(
        "--summary-score-limit",
        type=int,
        default=50,
        help="Rows per class to KGW-score for the summary JSON. Use 0 to skip summary scoring.",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prompt_text(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def _candidate_examples(split: Iterable[dict], min_chars: int) -> list[dict]:
    examples: list[dict] = []
    for i, ex in enumerate(split):
        question = _normalize_text(str(ex.get("question", "")))
        if not question:
            continue

        human_answers = [_normalize_text(t) for t in _iter_answers(ex.get("human_answers"))]
        human_answers = [t for t in human_answers if len(t) >= min_chars]
        if not human_answers:
            continue

        chatgpt_answers = [_normalize_text(t) for t in _iter_answers(ex.get("chatgpt_answers"))]
        chatgpt_answers = [t for t in chatgpt_answers if len(t) >= min_chars]

        examples.append(
            {
                "prompt_id": f"hc3_prompt_{i}",
                "question": question,
                "human_text": human_answers[0],
                "chatgpt_text": chatgpt_answers[0] if chatgpt_answers else "",
                "domain": str(ex.get("source", "hc3")),
            }
        )
    return examples


def _split_by_prompt(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    prompt_ids = pd.Series(sorted(df["prompt_id"].unique()))
    if len(prompt_ids) < 3:
        raise ValueError("Need at least 3 prompts to create train/val/test splits")

    n_prompts = len(prompt_ids)
    n_val = max(1, int(round(n_prompts * val_ratio)))
    n_test = max(1, int(round(n_prompts * test_ratio)))
    n_train = n_prompts - n_val - n_test
    while n_train < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("Need at least 3 prompts to create train/val/test splits")
        n_train = n_prompts - n_val - n_test

    shuffled = prompt_ids.sample(frac=1.0, random_state=seed).tolist()
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    train_df = df[df["prompt_id"].isin(train_ids)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = df[df["prompt_id"].isin(val_ids)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = df[df["prompt_id"].isin(test_ids)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def _write_csv(df: pd.DataFrame, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _score_watermarked_outputs(
    df: pd.DataFrame,
    model_name: str,
    gamma: float,
    hash_key: int,
    score_limit: int,
    seed: int,
) -> dict[str, float]:
    if score_limit == 0:
        return {}

    detector = WatermarkDetector(tokenizer_name=model_name, gamma=gamma, hash_key=hash_key)
    wm_df = df[df["variant"] == "watermarked_ai"]
    human_df = df[df["variant"] == "human"]
    if score_limit > 0:
        wm_df = wm_df.sample(min(score_limit, len(wm_df)), random_state=seed)
        human_df = human_df.sample(min(score_limit, len(human_df)), random_state=seed)

    wm_texts = wm_df["text"].tolist()
    human_texts = human_df["text"].tolist()
    wm_scores, wm_z = detector.score_texts(wm_texts)
    human_scores, human_z = detector.score_texts(human_texts)
    return {
        "summary_scored_per_class": int(score_limit) if score_limit > 0 else "all",
        "watermarked_mean_ai_score": float(np.mean(wm_scores)) if len(wm_scores) else float("nan"),
        "watermarked_mean_z": float(np.mean(wm_z)) if len(wm_z) else float("nan"),
        "human_mean_ai_score": float(np.mean(human_scores)) if len(human_scores) else float("nan"),
        "human_mean_z": float(np.mean(human_z)) if len(human_z) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    print(f"Loading HC3: {args.dataset} / {args.config}")
    split = _load_hc3_records(args.dataset, args.config)
    examples = _candidate_examples(split, args.min_chars)
    if not examples:
        raise RuntimeError("No usable HC3 prompt/human-answer examples found")

    rng = np.random.RandomState(args.seed)
    selected = rng.choice(examples, size=min(args.n_prompts, len(examples)), replace=False).tolist()
    print(f"Selected prompts: {len(selected)}")

    generator = WatermarkGenerator(
        model_name=args.model_name,
        gamma=args.gamma,
        delta=args.delta,
        hash_key=args.hash_key,
        device=device,
    )

    rows: list[dict] = []
    pair_rows: list[dict] = []

    for ex in tqdm(selected, desc="Generating KGW-watermarked HC3"):
        prompt = _prompt_text(ex["question"])
        watermarked_text = _normalize_text(
            generator.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                return_full_text=False,
            )
        )

        if len(watermarked_text) < args.min_chars:
            continue

        prompt_id = ex["prompt_id"]
        rows.append(
            {
                "id": f"{prompt_id}_human",
                "text": ex["human_text"],
                "source": "human",
                "attack_type": "none",
                "attack_owner": "none",
                "generator_model": ex["domain"],
                "variant": "human",
                "prompt_id": prompt_id,
                "question": ex["question"],
            }
        )
        rows.append(
            {
                "id": f"{prompt_id}_kgw",
                "text": watermarked_text,
                "source": "ai",
                "attack_type": "watermarked",
                "attack_owner": "kgw",
                "generator_model": f"{args.model_name}_kgw",
                "variant": "watermarked_ai",
                "prompt_id": prompt_id,
                "question": ex["question"],
                "watermark_gamma": args.gamma,
                "watermark_delta": args.delta,
                "watermark_hash_key": args.hash_key,
            }
        )
        pair_rows.append(
            {
                "pair_id": prompt_id,
                "original_text": ex["chatgpt_text"] or ex["human_text"],
                "humanized_text": watermarked_text,
                "attack_type": "kgw_watermarked",
                "generator_model": f"{args.model_name}_kgw",
                "question": ex["question"],
            }
        )

    if not rows:
        raise RuntimeError("Generation produced no usable watermarked samples")

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_df, val_df, test_df = _split_by_prompt(df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    _write_csv(df, args.out_raw)
    _write_csv(train_df, args.out_train)
    _write_csv(val_df, args.out_val)
    _write_csv(test_df, args.out_test)
    _write_csv(pd.DataFrame(pair_rows), args.paired_output)

    if args.as_default_splits:
        _write_csv(train_df, "data/splits/train.csv")
        _write_csv(val_df, "data/splits/val.csv")
        _write_csv(test_df, "data/splits/test.csv")

    generator._green_list.cache_clear()
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "model_name": args.model_name,
        "device": device,
        "requested_prompts": args.n_prompts,
        "generated_prompts": int((df["variant"] == "watermarked_ai").sum()),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "gamma": args.gamma,
        "delta": args.delta,
        "hash_key": args.hash_key,
        **_score_watermarked_outputs(
            df,
            args.model_name,
            args.gamma,
            args.hash_key,
            args.summary_score_limit,
            args.seed,
        ),
    }
    summary_path = Path(args.out_raw).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nGenerated KGW-watermarked HC3 dataset")
    print(f"  Raw:    {Path(args.out_raw).resolve()}")
    print(f"  Train:  {Path(args.out_train).resolve()}")
    print(f"  Val:    {Path(args.out_val).resolve()}")
    print(f"  Test:   {Path(args.out_test).resolve()}")
    print(f"  Pairs:  {Path(args.paired_output).resolve()}")
    print(f"  Summary:{summary_path.resolve()}")
    if "watermarked_mean_z" in summary:
        print(
            "  KGW z means: "
            f"watermarked={summary['watermarked_mean_z']:.3f}, "
            f"human={summary['human_mean_z']:.3f}"
        )


if __name__ == "__main__":
    main()
