import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke check for Phase-1 cross-encoder semantic constraint"
    )
    parser.add_argument(
        "--reference_text",
        type=str,
        default="The committee approved the proposal after a short discussion.",
        help="Reference text",
    )
    parser.add_argument(
        "--candidate_text",
        type=str,
        default="After a brief discussion, the committee approved the proposal.",
        help="Candidate text",
    )
    parser.add_argument(
        "--semantic_model_name",
        type=str,
        default="cross-encoder/stsb-roberta-large",
        help="Hugging Face cross-encoder model name",
    )
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.75,
        help="Threshold in [0, 1] after normalization",
    )
    parser.add_argument(
        "--semantic_window_size",
        type=int,
        default=50,
        help="Window size used by the constraint",
    )
    parser.add_argument(
        "--semantic_batch_size",
        type=int,
        default=16,
        help="Batch size for scoring",
    )
    parser.add_argument(
        "--semantic_max_length",
        type=int,
        default=512,
        help="Tokenizer max length",
    )
    parser.add_argument(
        "--semantic_stsb_max_score",
        type=float,
        default=5.0,
        help="Raw STS-B max score for normalization",
    )
    parser.add_argument(
        "--semantic_device",
        type=str,
        default=None,
        help="Device override (for example: cpu or cuda:0)",
    )
    parser.add_argument(
        "--fail_on_below_threshold",
        action="store_true",
        help="Exit with code 2 when score is below threshold",
    )
    return parser.parse_args()


def _set_env_if_value(key: str, value):
    if value is None:
        return
    os.environ[key] = str(value)


def _normalize_logits(logits: torch.Tensor, stsb_max_score: float) -> float:
    if logits.ndim == 1:
        return float(torch.clamp(logits[0] / stsb_max_score, 0.0, 1.0).item())

    if logits.shape[-1] == 1:
        return float(torch.clamp(logits.squeeze(-1)[0] / stsb_max_score, 0.0, 1.0).item())

    probs = torch.softmax(logits, dim=-1)
    return float(torch.clamp(probs[:, -1][0], 0.0, 1.0).item())


def _fallback_score_without_textattack(args) -> dict:
    requested_device = args.semantic_device
    if requested_device and requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.semantic_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.semantic_model_name)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        [args.reference_text],
        [args.candidate_text],
        padding=True,
        truncation=True,
        max_length=args.semantic_max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits

    score = _normalize_logits(logits, args.semantic_stsb_max_score)
    return {
        "mode": "fallback_without_textattack",
        "model_name": args.semantic_model_name,
        "device": device,
        "threshold": args.semantic_threshold,
        "score": score,
        "passes_threshold": score >= args.semantic_threshold,
        "reference_text": args.reference_text,
        "candidate_text": args.candidate_text,
    }


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    _set_env_if_value("HMGC_SEMANTIC_MODEL", args.semantic_model_name)
    _set_env_if_value("HMGC_SEMANTIC_THRESHOLD", args.semantic_threshold)
    _set_env_if_value("HMGC_SEMANTIC_WINDOW_SIZE", args.semantic_window_size)
    _set_env_if_value("HMGC_SEMANTIC_BATCH_SIZE", args.semantic_batch_size)
    _set_env_if_value("HMGC_SEMANTIC_MAX_LENGTH", args.semantic_max_length)
    _set_env_if_value("HMGC_SEMANTIC_STSB_MAX_SCORE", args.semantic_stsb_max_score)
    _set_env_if_value("HMGC_SEMANTIC_DEVICE", args.semantic_device)

    try:
        from attack.methods.constraints import build_semantic_constraint

        constraint = build_semantic_constraint(
            threshold=args.semantic_threshold,
            compare_against_original=True,
            window_size=args.semantic_window_size,
        )

        constraint.warmup()
        score = constraint.score_text_pairs([(args.reference_text, args.candidate_text)])[0]
        result = {
            "mode": "constraint_api",
            "model_name": constraint.model_name,
            "device": constraint.device,
            "threshold": constraint.threshold,
            "score": score,
            "passes_threshold": score >= constraint.threshold,
            "reference_text": args.reference_text,
            "candidate_text": args.candidate_text,
        }
    except ModuleNotFoundError as exc:
        if exc.name != "textattack":
            raise
        result = _fallback_score_without_textattack(args)
        result["warning"] = "textattack not installed; used transformer-only fallback"

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.fail_on_below_threshold and not result["passes_threshold"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
