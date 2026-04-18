#!/usr/bin/env python
"""Train a seq2seq post-generation evader with safe checkpoint resume support.

This script fine-tunes an instruction-tuned seq2seq model on paired
origin->humanized text samples (for example HMGC JSONL outputs with
`origin_text` and `attacked_text`).

Key capabilities:
- Periodic checkpoints + checkpoint pruning
- Automatic resume from latest checkpoint
- Graceful pause on Ctrl+C (SIGINT) with save-on-stop behavior
- Persistent training logs (TensorBoard + JSONL)
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


LOGGER = logging.getLogger("train_evader")


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a post-generation evader model")

    parser.add_argument(
        "--train_files",
        nargs="+",
        required=True,
        help="One or more JSONL/CSV files or glob patterns for train pairs",
    )
    parser.add_argument(
        "--eval_files",
        nargs="+",
        default=None,
        help="Optional JSONL/CSV files or glob patterns for eval pairs",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to store checkpoints, logs, and final model",
    )

    parser.add_argument(
        "--model_name_or_path",
        default="google/flan-t5-base",
        help="Base seq2seq model to fine-tune",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="auto | none | path/to/checkpoint",
    )

    parser.add_argument("--source_field", default="origin_text", help="Source text field")
    parser.add_argument("--target_field", default="attacked_text", help="Target text field")
    parser.add_argument("--label_field", default="label", help="Label field name")
    parser.add_argument(
        "--label_filter",
        default="gpt",
        help="Only use rows where label_field == label_filter. Use empty string to disable.",
    )

    parser.add_argument(
        "--instruction_prefix",
        default=(
            "Rewrite the following AI-generated text so it reads naturally like human writing, "
            "while preserving the original meaning."
        ),
        help="Instruction prefix prepended to each source text",
    )
    parser.add_argument(
        "--paraphrase_mode",
        action="store_true",
        help=(
            "Enable strict paraphrase mode to reduce elaboration and keep rewrites length-faithful."
        ),
    )
    parser.add_argument(
        "--paraphrase_min_ratio",
        type=float,
        default=0.75,
        help="Minimum allowed target/source token-length ratio in paraphrase mode.",
    )
    parser.add_argument(
        "--paraphrase_max_ratio",
        type=float,
        default=1.12,
        help="Maximum allowed target/source token-length ratio in paraphrase mode.",
    )
    parser.add_argument(
        "--paraphrase_min_jaccard",
        type=float,
        default=0.18,
        help="Minimum token-set Jaccard overlap in paraphrase mode.",
    )
    parser.add_argument(
        "--paraphrase_max_jaccard",
        type=float,
        default=0.92,
        help="Maximum token-set Jaccard overlap in paraphrase mode.",
    )
    parser.add_argument(
        "--disable_semantic_filter",
        action="store_true",
        help="Disable Phase-1 semantic gate filtering during paraphrase training.",
    )
    parser.add_argument(
        "--semantic_filter_model_name",
        default="cross-encoder/stsb-roberta-base",
        help="Cross-encoder model for Phase-1 semantic gate filtering.",
    )
    parser.add_argument(
        "--semantic_filter_threshold",
        type=float,
        default=0.78,
        help="Minimum semantic similarity score [0,1] required to keep a pair.",
    )
    parser.add_argument(
        "--semantic_filter_batch_size",
        type=int,
        default=16,
        help="Batch size for semantic gate scoring.",
    )
    parser.add_argument(
        "--semantic_filter_max_length",
        type=int,
        default=256,
        help="Max tokenized length for semantic gate scoring.",
    )
    parser.add_argument(
        "--semantic_filter_device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device used for semantic gate scoring.",
    )
    parser.add_argument(
        "--phase2_span_guidance",
        action="store_true",
        help="Enable Phase-2 span-guided paraphrase prompting.",
    )
    parser.add_argument(
        "--phase2_min_span_len",
        type=int,
        default=3,
        help="Minimum span length (words) for Phase-2 guidance.",
    )
    parser.add_argument(
        "--phase2_max_span_len",
        type=int,
        default=6,
        help="Maximum span length (words) for Phase-2 guidance.",
    )

    parser.add_argument("--max_source_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=192)
    parser.add_argument("--eval_split_ratio", type=float, default=0.1)

    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", default="cosine")

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=25)

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help=(
            "Number of consecutive eval calls with no meaningful eval_loss improvement "
            "before training is stopped. Set to 0 to disable early stopping."
        ),
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=5e-4,
        help="Minimum eval_loss improvement required to reset early stopping patience.",
    )

    parser.add_argument(
        "--disable_hourly_evasion_eval",
        action="store_true",
        help="Disable periodic detector-side evasion evaluation during training.",
    )
    parser.add_argument(
        "--evasion_eval_interval_seconds",
        type=int,
        default=3600,
        help="Wall-clock interval between evasion evaluations during training.",
    )
    parser.add_argument(
        "--evasion_eval_detector_model_path",
        default="HMGC-dataset/output/checkgpt/model/surrogate_distilroberta_base_fast",
        help="Detector model path used to score generated text as gpt/human.",
    )
    parser.add_argument(
        "--evasion_eval_probe_files",
        nargs="+",
        default=None,
        help="Optional JSONL/CSV files used as probe set for periodic evasion evaluation.",
    )
    parser.add_argument(
        "--evasion_eval_samples",
        type=int,
        default=256,
        help="Number of probe samples used for each periodic evasion evaluation.",
    )
    parser.add_argument(
        "--evasion_eval_batch_size",
        type=int,
        default=16,
        help="Batch size for generation and detector inference during evasion evaluation.",
    )
    parser.add_argument(
        "--evasion_eval_generation_max_new_tokens",
        type=int,
        default=160,
        help="Max new tokens to generate per sample for periodic evasion evaluation.",
    )
    parser.add_argument(
        "--evasion_eval_detector_max_length",
        type=int,
        default=256,
        help="Max tokenized length for detector scoring during evasion evaluation.",
    )
    parser.add_argument(
        "--evasion_eval_detector_device",
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for detector inference during periodic evasion evaluation.",
    )

    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--dataloader_pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use_mps_device",
        action="store_true",
        help="Use Apple Metal backend if available",
    )
    parser.add_argument(
        "--max_train_minutes",
        type=float,
        default=0.0,
        help="Hard wall-clock budget in minutes. Set <=0 to disable.",
    )

    return parser.parse_args()


def expand_paths(path_patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for entry in path_patterns:
        p = Path(entry)
        if p.exists():
            paths.append(p)
            continue

        # Allow glob patterns for convenience.
        matches = sorted(Path(m) for m in glob.glob(entry, recursive=True))
        paths.extend(matches)

    deduped = sorted({p.resolve() for p in paths if p.exists()})
    return [Path(p) for p in deduped]


def iter_records(path: Path) -> Iterable[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    else:
        raise ValueError(f"Unsupported file format for {path}. Use .jsonl or .csv")


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def word_count(text: str) -> int:
    return len(text.split())


def token_jaccard_overlap(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b) / len(union))


def is_valid_paraphrase_pair(
    source: str,
    target: str,
    min_ratio: float,
    max_ratio: float,
    min_jaccard: float,
    max_jaccard: float,
) -> bool:
    src_len = max(1, word_count(source))
    tgt_len = max(1, word_count(target))
    ratio = float(tgt_len / src_len)
    if ratio < min_ratio or ratio > max_ratio:
        return False

    overlap = token_jaccard_overlap(source, target)
    if overlap < min_jaccard or overlap > max_jaccard:
        return False

    return True


def load_pairs(
    files: Sequence[Path],
    source_field: str,
    target_field: str,
    label_field: str,
    label_filter: str,
    paraphrase_mode: bool,
    paraphrase_min_ratio: float,
    paraphrase_max_ratio: float,
    paraphrase_min_jaccard: float,
    paraphrase_max_jaccard: float,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    dropped_identical = 0
    dropped_paraphrase = 0

    for path in files:
        for rec in iter_records(path):
            if label_filter:
                if str(rec.get(label_field, "")).strip() != label_filter:
                    continue

            source = clean_text(rec.get(source_field, ""))
            target = clean_text(rec.get(target_field, ""))
            if not source or not target:
                continue
            if source == target:
                dropped_identical += 1
                continue

            if paraphrase_mode and not is_valid_paraphrase_pair(
                source=source,
                target=target,
                min_ratio=paraphrase_min_ratio,
                max_ratio=paraphrase_max_ratio,
                min_jaccard=paraphrase_min_jaccard,
                max_jaccard=paraphrase_max_jaccard,
            ):
                dropped_paraphrase += 1
                continue

            pair = (source, target)
            if pair in seen:
                continue
            seen.add(pair)

            rows.append(
                {
                    "source_text": source,
                    "target_text": target,
                    "source_file": str(path),
                }
            )

    if dropped_identical > 0:
        LOGGER.info("Dropped %d identical source-target rows", dropped_identical)
    if paraphrase_mode and dropped_paraphrase > 0:
        LOGGER.info(
            "Dropped %d rows outside paraphrase constraints (ratio/jaccard)",
            dropped_paraphrase,
        )

    return rows


def resolve_resume_checkpoint(output_dir: Path, resume_from_checkpoint: str) -> Optional[str]:
    mode = (resume_from_checkpoint or "").strip().lower()
    if mode in {"", "none", "false", "no"}:
        return None
    if mode == "auto":
        if not output_dir.exists():
            return None
        return get_last_checkpoint(str(output_dir))

    explicit = Path(resume_from_checkpoint)
    if explicit.exists():
        return str(explicit)

    raise FileNotFoundError(f"Requested resume checkpoint does not exist: {explicit}")


@dataclass
class RuntimeState:
    stop_requested: bool = False
    stop_signal: int = 0


class GracefulStopCallback(TrainerCallback):
    """Stop at next step boundary and trigger a checkpoint save on SIGINT/SIGTERM."""

    def __init__(self, state: RuntimeState):
        self.state = state
        self._old_handlers: Dict[int, object] = {}

    def _signal_handler(self, signum, _frame) -> None:  # type: ignore[no-untyped-def]
        if not self.state.stop_requested:
            self.state.stop_requested = True
            self.state.stop_signal = int(signum)
            LOGGER.warning(
                "Received signal %s. Will save a checkpoint and stop safely at the next step.",
                signum,
            )
        else:
            LOGGER.warning("Second interrupt received. Attempting immediate shutdown.")
            raise KeyboardInterrupt

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal_handler)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if self.state.stop_requested:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        for sig, old_handler in self._old_handlers.items():
            signal.signal(sig, old_handler)
        return control


class JsonlLoggerCallback(TrainerCallback):
    """Persist trainer logs as JSONL so progress survives terminal interruptions."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if not logs:
            return control
        payload = {
            "time": time.time(),
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **logs,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return control


class WallClockLimitCallback(TrainerCallback):
    """Stop training after a hard wall-clock budget."""

    def __init__(self, max_minutes: float):
        self.max_seconds = max(1.0, float(max_minutes) * 60.0)
        self.start_unix = 0.0
        self.notified = False

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        self.start_unix = time.time()
        LOGGER.info("Wall-clock limit enabled: %.1f minutes", self.max_seconds / 60.0)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if self.start_unix <= 0.0:
            self.start_unix = time.time()

        elapsed = time.time() - self.start_unix
        if elapsed >= self.max_seconds:
            if not self.notified:
                LOGGER.warning(
                    "Reached wall-clock budget (%.1f min). Saving checkpoint and stopping.",
                    self.max_seconds / 60.0,
                )
                self.notified = True
            control.should_save = True
            control.should_training_stop = True
        return control


def select_focus_span(text: str, min_span_len: int, max_span_len: int) -> str:
    words = text.split()
    if not words:
        return ""

    min_len = max(1, int(min_span_len))
    max_len = max(min_len, int(max_span_len))
    if len(words) <= min_len:
        return " ".join(words)

    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    span_len = min(len(words), rng.randint(min_len, max_len))
    start = rng.randint(0, max(0, len(words) - span_len))
    return " ".join(words[start : start + span_len])


def build_prompt(
    prefix: str,
    source_text: str,
    paraphrase_mode: bool,
    paraphrase_min_ratio: float,
    paraphrase_max_ratio: float,
    focus_span: str,
) -> str:
    if not paraphrase_mode:
        return f"{prefix}\n\nInput:\n{source_text}\n\nRewrite:"

    min_pct = int(round(paraphrase_min_ratio * 100))
    max_pct = int(round(paraphrase_max_ratio * 100))
    span_line = ""
    if focus_span:
        span_line = f"- Keep this key span semantically intact while rephrasing: {focus_span}\n"

    return (
        f"{prefix}\n\n"
        "Rules:\n"
        "- Paraphrase only. Do not elaborate.\n"
        "- Do not add new facts, claims, examples, or citations.\n"
        "- Keep technical meaning unchanged.\n"
        f"- Keep output length roughly {min_pct}% to {max_pct}% of input length.\n"
        f"{span_line}"
        "- Prefer sentence-level and clause-level rewrites over simple synonym swaps.\n\n"
        f"Input:\n{source_text}\n\n"
        "Paraphrased humanized output:"
    )


def resolve_report_to() -> List[str]:
    has_tensorboard = importlib.util.find_spec("tensorboard") is not None
    has_tensorboardx = importlib.util.find_spec("tensorboardX") is not None
    if has_tensorboard or has_tensorboardx:
        return ["tensorboard"]
    LOGGER.warning(
        "TensorBoard is not installed. Falling back to JSONL logging only (training_log.jsonl)."
    )
    return []


def resolve_torch_device(device_name: str) -> torch.device:
    name = (device_name or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested evasion eval detector device 'cuda' but CUDA is unavailable.")
        return torch.device("cuda")
    if name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("Requested evasion eval detector device 'mps' but MPS is unavailable.")
        return torch.device("mps")
    return torch.device("cpu")


def normalize_semantic_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)

    if logits.shape[-1] == 1:
        raw = logits.squeeze(-1)
        as_sts = (raw / 5.0).clamp(0.0, 1.0)
        as_signed = ((raw + 1.0) / 2.0).clamp(0.0, 1.0)
        return torch.where(raw > 1.0, as_sts, as_signed)

    probs = torch.softmax(logits, dim=-1)
    if probs.shape[-1] == 2:
        return probs[:, 1]
    return probs.max(dim=-1).values


def lexical_semantic_score(source: str, target: str) -> float:
    overlap = token_jaccard_overlap(source, target)
    src_len = max(1, word_count(source))
    tgt_len = max(1, word_count(target))
    ratio = float(tgt_len / src_len)
    length_score = max(0.0, 1.0 - abs(1.0 - ratio))
    return float(0.65 * overlap + 0.35 * length_score)


def lexical_semantic_filter_rows(rows: List[Dict[str, str]], threshold: float) -> List[Dict[str, str]]:
    kept_rows: List[Dict[str, str]] = []
    dropped = 0
    score_sum = 0.0
    score_count = 0

    for row in rows:
        score = lexical_semantic_score(row["source_text"], row["target_text"])
        score_sum += score
        score_count += 1
        if score >= threshold:
            kept_rows.append(row)
        else:
            dropped += 1

    avg_score = (score_sum / score_count) if score_count > 0 else 0.0
    LOGGER.info(
        "Lexical semantic gate kept %d/%d pairs (dropped=%d, avg_score=%.4f)",
        len(kept_rows),
        len(rows),
        dropped,
        avg_score,
    )
    return kept_rows


def semantic_filter_rows(
    rows: List[Dict[str, str]],
    model_name: str,
    threshold: float,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> List[Dict[str, str]]:
    if not rows:
        return rows

    LOGGER.info(
        "Applying semantic gate with %s on %d pairs (threshold=%.3f, device=%s)",
        model_name,
        len(rows),
        threshold,
        str(device),
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
        )
        model = model.to(device)
        model.eval()
    except Exception as exc:  # pylint: disable=broad-except
        fallback_threshold = min(0.85, max(0.25, threshold * 0.65))
        LOGGER.warning(
            "Could not load semantic gate model '%s' from local cache (%s). "
            "Falling back to lexical semantic gate with threshold=%.3f.",
            model_name,
            exc,
            fallback_threshold,
        )
        return lexical_semantic_filter_rows(rows, threshold=fallback_threshold)

    kept_rows: List[Dict[str, str]] = []
    dropped = 0
    score_sum = 0.0
    score_count = 0

    with torch.no_grad():
        for start in range(0, len(rows), max(1, batch_size)):
            batch = rows[start : start + max(1, batch_size)]
            if start > 0 and (start // max(1, batch_size)) % 100 == 0:
                LOGGER.info("Semantic gate progress: %d/%d pairs", start, len(rows))
            sources = [r["source_text"] for r in batch]
            targets = [r["target_text"] for r in batch]
            enc = tokenizer(
                text=sources,
                text_pair=targets,
                truncation=True,
                max_length=max(8, int(max_length)),
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            scores = normalize_semantic_logits(logits).detach().cpu().tolist()

            for row, score in zip(batch, scores):
                score_f = float(score)
                score_sum += score_f
                score_count += 1
                if score_f >= threshold:
                    kept_rows.append(row)
                else:
                    dropped += 1

    avg_score = (score_sum / score_count) if score_count > 0 else 0.0
    LOGGER.info(
        "Semantic gate kept %d/%d pairs (dropped=%d, avg_score=%.4f)",
        len(kept_rows),
        len(rows),
        dropped,
        avg_score,
    )
    return kept_rows


def select_probe_sources(sources: Sequence[str], max_samples: int, seed: int) -> List[str]:
    cleaned = [s for s in sources if isinstance(s, str) and s.strip()]
    # Preserve order while deduplicating repeated source texts.
    deduped = list(dict.fromkeys(cleaned))
    if max_samples <= 0 or len(deduped) <= max_samples:
        return deduped

    indices = list(range(len(deduped)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = sorted(indices[:max_samples])
    return [deduped[i] for i in selected]


class HourlyEvasionEvalCallback(TrainerCallback):
    """Periodically evaluate detector-side evasion on generated rewrites."""

    def __init__(
        self,
        output_dir: Path,
        training_log_path: Path,
        generator_tokenizer,
        instruction_prefix: str,
        paraphrase_mode: bool,
        paraphrase_min_ratio: float,
        paraphrase_max_ratio: float,
        span_guidance: bool,
        span_min_len: int,
        span_max_len: int,
        probe_sources: Sequence[str],
        detector_model_path: str,
        detector_device: torch.device,
        interval_seconds: int,
        batch_size: int,
        max_source_length: int,
        max_new_tokens: int,
        detector_max_length: int,
    ):
        self.output_dir = output_dir
        self.training_log_path = training_log_path
        self.generator_tokenizer = generator_tokenizer
        self.instruction_prefix = instruction_prefix
        self.paraphrase_mode = paraphrase_mode
        self.paraphrase_min_ratio = paraphrase_min_ratio
        self.paraphrase_max_ratio = paraphrase_max_ratio
        self.span_guidance = span_guidance
        self.span_min_len = span_min_len
        self.span_max_len = span_max_len
        self.probe_sources = list(probe_sources)
        self.detector_model_path = detector_model_path
        self.detector_device = detector_device
        self.interval_seconds = max(1, int(interval_seconds))
        self.batch_size = max(1, int(batch_size))
        self.max_source_length = max(8, int(max_source_length))
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.detector_max_length = max(8, int(detector_max_length))
        self.last_eval_unix = time.time()
        self._running = False

        self.hourly_eval_log_path = self.output_dir / "hourly_evasion_eval.jsonl"
        self.latest_eval_path = self.output_dir / "hourly_evasion_latest.json"
        self.status_path = self.output_dir / "hourly_evasion_status.json"

        self.detector_tokenizer = AutoTokenizer.from_pretrained(detector_model_path)
        self.detector_model = AutoModelForSequenceClassification.from_pretrained(detector_model_path)
        self.detector_model = self.detector_model.to(self.detector_device)
        self.detector_model.eval()

        detector_label2id = getattr(self.detector_model.config, "label2id", None) or {}
        norm_label2id = {str(k).lower(): int(v) for k, v in detector_label2id.items()}
        self.human_label_id = norm_label2id.get("human", 1)

        self._write_status_payload(last_eval=None)

    def _append_jsonl(self, path: Path, payload: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_status_payload(self, last_eval: Optional[Dict[str, object]]) -> None:
        status = {
            "enabled": True,
            "interval_seconds": int(self.interval_seconds),
            "probe_samples": len(self.probe_sources),
            "detector_model_path": self.detector_model_path,
            "detector_device": str(self.detector_device),
            "last_eval": last_eval,
            "next_due_unix": float(self.last_eval_unix + self.interval_seconds),
        }
        self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def _build_prompts(self) -> List[str]:
        prompts: List[str] = []
        for source_text in self.probe_sources:
            focus_span = ""
            if self.paraphrase_mode and self.span_guidance:
                focus_span = select_focus_span(
                    text=source_text,
                    min_span_len=self.span_min_len,
                    max_span_len=self.span_max_len,
                )
            prompts.append(
                build_prompt(
                    prefix=self.instruction_prefix,
                    source_text=source_text,
                    paraphrase_mode=self.paraphrase_mode,
                    paraphrase_min_ratio=self.paraphrase_min_ratio,
                    paraphrase_max_ratio=self.paraphrase_max_ratio,
                    focus_span=focus_span,
                )
            )
        return prompts

    def _generate_rewrites(self, model) -> List[str]:  # type: ignore[no-untyped-def]
        prompts = self._build_prompts()
        rewrites: List[str] = []

        model_was_training = bool(model.training)
        model.eval()
        try:
            for start in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[start : start + self.batch_size]
                batch_sources = self.probe_sources[start : start + self.batch_size]

                model_inputs = self.generator_tokenizer(
                    batch_prompts,
                    max_length=self.max_source_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )

                decoded = self.generator_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for idx, text in enumerate(decoded):
                    cleaned = " ".join(text.strip().split())
                    if not cleaned:
                        cleaned = batch_sources[idx]
                    rewrites.append(cleaned)
        finally:
            if model_was_training:
                model.train()

        return rewrites

    def _detector_human_rate(self, texts: Sequence[str]) -> float:
        human_hits = 0
        total = 0
        for start in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[start : start + self.batch_size])
            det_inputs = self.detector_tokenizer(
                batch_texts,
                max_length=self.detector_max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            det_inputs = {k: v.to(self.detector_device) for k, v in det_inputs.items()}
            with torch.no_grad():
                logits = self.detector_model(**det_inputs).logits
            preds = logits.argmax(dim=-1)
            human_hits += int((preds == self.human_label_id).sum().item())
            total += int(preds.numel())

        if total == 0:
            return 0.0
        return float(human_hits / total)

    def _run_hourly_eval(self, model, state) -> Dict[str, object]:  # type: ignore[no-untyped-def]
        rewrites = self._generate_rewrites(model)
        human_rate = self._detector_human_rate(rewrites)

        payload = {
            "event": "hourly_evasion_eval",
            "time": time.time(),
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "probe_samples": len(rewrites),
            "detector_model_path": self.detector_model_path,
            "detector_device": str(self.detector_device),
            "ai_evasion_accuracy": human_rate,
            "detector_human_rate": human_rate,
            "detector_gpt_rate": float(1.0 - human_rate),
        }
        return payload

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[no-untyped-def]
        if self._running:
            return control

        now = time.time()
        if now - self.last_eval_unix < self.interval_seconds:
            return control

        if model is None or state.global_step <= 0:
            return control

        self._running = True
        try:
            payload = self._run_hourly_eval(model=model, state=state)
            self.last_eval_unix = float(payload["time"])

            self._append_jsonl(self.hourly_eval_log_path, payload)
            self._append_jsonl(self.training_log_path, payload)
            self.latest_eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._write_status_payload(last_eval=payload)

            LOGGER.info(
                "Hourly evasion eval @ step %d: ai_evasion_accuracy=%.4f (%d samples)",
                payload["step"],
                payload["ai_evasion_accuracy"],
                payload["probe_samples"],
            )
        except Exception as exc:  # pylint: disable=broad-except
            error_payload = {
                "event": "hourly_evasion_eval_error",
                "time": time.time(),
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "error": str(exc),
            }
            self.last_eval_unix = float(error_payload["time"])
            self._append_jsonl(self.hourly_eval_log_path, error_payload)
            self._append_jsonl(self.training_log_path, error_payload)
            self._write_status_payload(last_eval=error_payload)
            LOGGER.exception("Hourly evasion evaluation failed")
        finally:
            self._running = False

        return control


def main() -> None:
    setup_logging()
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    train_files = expand_paths(args.train_files)
    if not train_files:
        raise ValueError("No training files were found. Check --train_files paths/patterns.")

    eval_files = expand_paths(args.eval_files or [])

    train_rows = load_pairs(
        files=train_files,
        source_field=args.source_field,
        target_field=args.target_field,
        label_field=args.label_field,
        label_filter=args.label_filter,
        paraphrase_mode=args.paraphrase_mode,
        paraphrase_min_ratio=args.paraphrase_min_ratio,
        paraphrase_max_ratio=args.paraphrase_max_ratio,
        paraphrase_min_jaccard=args.paraphrase_min_jaccard,
        paraphrase_max_jaccard=args.paraphrase_max_jaccard,
    )
    if not train_rows:
        raise ValueError("No train pairs were loaded. Verify source/target/label fields.")

    if args.paraphrase_mode and not args.disable_semantic_filter:
        semantic_device = resolve_torch_device(args.semantic_filter_device)
        train_rows = semantic_filter_rows(
            rows=train_rows,
            model_name=args.semantic_filter_model_name,
            threshold=args.semantic_filter_threshold,
            batch_size=args.semantic_filter_batch_size,
            max_length=args.semantic_filter_max_length,
            device=semantic_device,
        )
        if not train_rows:
            raise ValueError(
                "All train pairs were filtered out by semantic gate. "
                "Lower --semantic_filter_threshold."
            )

    if eval_files:
        eval_rows = load_pairs(
            files=eval_files,
            source_field=args.source_field,
            target_field=args.target_field,
            label_field=args.label_field,
            label_filter=args.label_filter,
            paraphrase_mode=args.paraphrase_mode,
            paraphrase_min_ratio=args.paraphrase_min_ratio,
            paraphrase_max_ratio=args.paraphrase_max_ratio,
            paraphrase_min_jaccard=args.paraphrase_min_jaccard,
            paraphrase_max_jaccard=args.paraphrase_max_jaccard,
        )
        if not eval_rows:
            raise ValueError("No eval pairs were loaded from --eval_files.")
        if args.paraphrase_mode and not args.disable_semantic_filter:
            semantic_device = resolve_torch_device(args.semantic_filter_device)
            eval_rows = semantic_filter_rows(
                rows=eval_rows,
                model_name=args.semantic_filter_model_name,
                threshold=args.semantic_filter_threshold,
                batch_size=args.semantic_filter_batch_size,
                max_length=args.semantic_filter_max_length,
                device=semantic_device,
            )
            if not eval_rows:
                raise ValueError(
                    "All eval pairs were filtered out by semantic gate. "
                    "Lower --semantic_filter_threshold."
                )
        train_ds = Dataset.from_list(train_rows)
        eval_ds = Dataset.from_list(eval_rows)
    else:
        split = Dataset.from_list(train_rows).train_test_split(
            test_size=args.eval_split_ratio,
            seed=args.seed,
        )
        train_ds = split["train"]
        eval_ds = split["test"]

    LOGGER.info("Loaded train pairs: %d", len(train_ds))
    LOGGER.info("Loaded eval pairs : %d", len(eval_ds))
    LOGGER.info("Using model       : %s", args.model_name_or_path)

    probe_sources: List[str] = []
    probe_files: List[Path] = []
    if not args.disable_hourly_evasion_eval and args.evasion_eval_interval_seconds > 0:
        if args.evasion_eval_probe_files:
            probe_files = expand_paths(args.evasion_eval_probe_files)
            if not probe_files:
                raise ValueError("No probe files were found for --evasion_eval_probe_files.")
            probe_rows = load_pairs(
                files=probe_files,
                source_field=args.source_field,
                target_field=args.target_field,
                label_field=args.label_field,
                label_filter=args.label_filter,
                paraphrase_mode=False,
                paraphrase_min_ratio=args.paraphrase_min_ratio,
                paraphrase_max_ratio=args.paraphrase_max_ratio,
                paraphrase_min_jaccard=args.paraphrase_min_jaccard,
                paraphrase_max_jaccard=args.paraphrase_max_jaccard,
            )
            probe_sources = [row["source_text"] for row in probe_rows]
        else:
            probe_sources = [row["source_text"] for row in eval_ds]

        probe_sources = select_probe_sources(
            sources=probe_sources,
            max_samples=args.evasion_eval_samples,
            seed=args.seed,
        )
        LOGGER.info("Prepared evasion probe samples: %d", len(probe_sources))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    def build_training_prompt(source_text: str) -> str:
        focus_span = ""
        if args.paraphrase_mode and args.phase2_span_guidance:
            focus_span = select_focus_span(
                text=source_text,
                min_span_len=args.phase2_min_span_len,
                max_span_len=args.phase2_max_span_len,
            )
        return build_prompt(
            prefix=args.instruction_prefix,
            source_text=source_text,
            paraphrase_mode=args.paraphrase_mode,
            paraphrase_min_ratio=args.paraphrase_min_ratio,
            paraphrase_max_ratio=args.paraphrase_max_ratio,
            focus_span=focus_span,
        )

    def preprocess_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        prompts = [build_training_prompt(s) for s in batch["source_text"]]
        targets = batch["target_text"]

        model_inputs = tokenizer(
            prompts,
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train pairs",
    )
    eval_tok = eval_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval pairs",
    )

    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save_steps must be a multiple of --eval_steps for stable best-model selection.")

    log_jsonl_path = output_dir / "training_log.jsonl"
    tensorboard_dir = output_dir / "logs" / "tensorboard"
    report_to = resolve_report_to()
    dataloader_workers = args.dataloader_num_workers
    if args.use_mps_device and dataloader_workers > 0:
        LOGGER.warning(
            "MPS run detected: forcing dataloader_num_workers=0 to avoid PyTorch shared-memory worker crashes."
        )
        dataloader_workers = 0

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_dir=str(tensorboard_dir),
        report_to=report_to,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        seed=args.seed,
        use_mps_device=args.use_mps_device,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    runtime_state = RuntimeState()
    callbacks: List[TrainerCallback] = [
        GracefulStopCallback(runtime_state),
        JsonlLoggerCallback(log_jsonl_path),
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
        LOGGER.info(
            "Early stopping enabled: patience=%d, threshold=%.6f",
            args.early_stopping_patience,
            args.early_stopping_threshold,
        )
    else:
        LOGGER.info("Early stopping disabled")

    if args.max_train_minutes and args.max_train_minutes > 0:
        callbacks.append(WallClockLimitCallback(max_minutes=args.max_train_minutes))

    hourly_evasion_enabled = False
    detector_model_path_for_config: Optional[str] = None
    if args.disable_hourly_evasion_eval:
        LOGGER.info("Hourly evasion eval disabled by --disable_hourly_evasion_eval")
    elif args.evasion_eval_interval_seconds <= 0:
        LOGGER.info("Hourly evasion eval disabled because --evasion_eval_interval_seconds <= 0")
    elif not probe_sources:
        LOGGER.warning("Hourly evasion eval disabled because probe set is empty")
    else:
        detector_path_obj = Path(args.evasion_eval_detector_model_path)
        if detector_path_obj.exists():
            detector_model_path_for_config = str(detector_path_obj.resolve())
        else:
            detector_model_path_for_config = args.evasion_eval_detector_model_path

        detector_device = resolve_torch_device(args.evasion_eval_detector_device)
        callbacks.append(
            HourlyEvasionEvalCallback(
                output_dir=output_dir,
                training_log_path=log_jsonl_path,
                generator_tokenizer=tokenizer,
                instruction_prefix=args.instruction_prefix,
                paraphrase_mode=args.paraphrase_mode,
                paraphrase_min_ratio=args.paraphrase_min_ratio,
                paraphrase_max_ratio=args.paraphrase_max_ratio,
                span_guidance=args.phase2_span_guidance,
                span_min_len=args.phase2_min_span_len,
                span_max_len=args.phase2_max_span_len,
                probe_sources=probe_sources,
                detector_model_path=detector_model_path_for_config,
                detector_device=detector_device,
                interval_seconds=args.evasion_eval_interval_seconds,
                batch_size=args.evasion_eval_batch_size,
                max_source_length=args.max_source_length,
                max_new_tokens=args.evasion_eval_generation_max_new_tokens,
                detector_max_length=args.evasion_eval_detector_max_length,
            )
        )
        hourly_evasion_enabled = True
        LOGGER.info(
            "Hourly evasion eval enabled: interval=%ds, probe_samples=%d, detector=%s, device=%s",
            args.evasion_eval_interval_seconds,
            len(probe_sources),
            detector_model_path_for_config,
            args.evasion_eval_detector_device,
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        callbacks=callbacks,
    )

    resume_checkpoint = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    if resume_checkpoint:
        LOGGER.info("Resuming from checkpoint: %s", resume_checkpoint)
    else:
        LOGGER.info("Starting a fresh training run")

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        train_metrics = dict(train_result.metrics)
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted abruptly; saving emergency state.")
        emergency_dir = output_dir / f"checkpoint-interrupt-step-{trainer.state.global_step}"
        emergency_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(emergency_dir))
        trainer.save_state()
        train_metrics = {
            "interrupted": True,
            "global_step": int(trainer.state.global_step),
            "epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
        }

    train_metrics["train_pairs"] = len(train_tok)
    trainer.save_model()  # also saves tokenizer
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate(metric_key_prefix="eval")
    eval_metrics["eval_pairs"] = len(eval_tok)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    run_config = {
        "train_files": [str(p) for p in train_files],
        "eval_files": [str(p) for p in eval_files],
        "evasion_probe_files": [str(p) for p in probe_files],
        "train_pairs": len(train_tok),
        "eval_pairs": len(eval_tok),
        "model_name_or_path": args.model_name_or_path,
        "paraphrase_mode": args.paraphrase_mode,
        "paraphrase_min_ratio": args.paraphrase_min_ratio,
        "paraphrase_max_ratio": args.paraphrase_max_ratio,
        "paraphrase_min_jaccard": args.paraphrase_min_jaccard,
        "paraphrase_max_jaccard": args.paraphrase_max_jaccard,
        "phase2_span_guidance": args.phase2_span_guidance,
        "phase2_min_span_len": args.phase2_min_span_len,
        "phase2_max_span_len": args.phase2_max_span_len,
        "semantic_filter_enabled": args.paraphrase_mode and (not args.disable_semantic_filter),
        "semantic_filter_model_name": args.semantic_filter_model_name,
        "semantic_filter_threshold": args.semantic_filter_threshold,
        "semantic_filter_batch_size": args.semantic_filter_batch_size,
        "semantic_filter_max_length": args.semantic_filter_max_length,
        "semantic_filter_device": args.semantic_filter_device,
        "max_train_minutes": args.max_train_minutes,
        "resume_from_checkpoint": resume_checkpoint,
        "stop_requested": runtime_state.stop_requested,
        "stop_signal": runtime_state.stop_signal,
        "hourly_evasion_eval_enabled": hourly_evasion_enabled,
        "evasion_eval_interval_seconds": args.evasion_eval_interval_seconds,
        "evasion_eval_probe_samples": len(probe_sources),
        "evasion_eval_detector_model_path": detector_model_path_for_config,
        "evasion_eval_detector_device": args.evasion_eval_detector_device,
        "evasion_eval_log_file": str(output_dir / "hourly_evasion_eval.jsonl"),
        "evasion_eval_latest_file": str(output_dir / "hourly_evasion_latest.json"),
        "evasion_eval_status_file": str(output_dir / "hourly_evasion_status.json"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    LOGGER.info("Training complete. Model and logs saved to %s", output_dir)


if __name__ == "__main__":
    main()