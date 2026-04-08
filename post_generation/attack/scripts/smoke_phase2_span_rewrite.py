import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke check for Phase-2 span-level seq2seq rewriting"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Large language models can generate polished text quickly, but robust "
            "detectors often rely on stylistic and syntactic artifacts."
        ),
        help="Input text to rewrite",
    )
    parser.add_argument(
        "--span_start",
        type=int,
        default=None,
        help="Inclusive start word index of span to rewrite",
    )
    parser.add_argument(
        "--span_end",
        type=int,
        default=None,
        help="Inclusive end word index of span to rewrite",
    )
    parser.add_argument(
        "--rewrite_model_name",
        type=str,
        default="google/flan-t5-small",
        help="Seq2seq rewrite model name",
    )
    parser.add_argument(
        "--num_candidate",
        type=int,
        default=4,
        help="Number of rewrite candidates to sample",
    )
    parser.add_argument(
        "--min_span_len",
        type=int,
        default=3,
        help="Minimum span length",
    )
    parser.add_argument(
        "--max_span_len",
        type=int,
        default=6,
        help="Maximum span length",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=48,
        help="Maximum generated tokens for rewritten span",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, for example cpu or cuda:0",
    )
    parser.add_argument(
        "--print_top_k",
        type=int,
        default=3,
        help="Maximum number of generated candidates to print",
    )
    return parser.parse_args()


def _pick_auto_span(num_words, min_span_len, max_span_len):
    if num_words <= 0:
        return 0, -1

    span_len = min(max_span_len, num_words)
    if span_len < min_span_len:
        span_len = min_span_len
    span_len = min(span_len, num_words)

    start = max(0, (num_words - span_len) // 2)
    end = start + span_len - 1
    return start, end


def _validate_and_get_span(args, attacked_text):
    if attacked_text.num_words <= 0:
        raise ValueError("Input text has no tokenizable words")

    if args.span_start is None and args.span_end is None:
        return _pick_auto_span(attacked_text.num_words, args.min_span_len, args.max_span_len)

    if args.span_start is None or args.span_end is None:
        raise ValueError("Provide both --span_start and --span_end, or neither")

    start = args.span_start
    end = args.span_end
    if start < 0 or end < 0:
        raise ValueError("Span indices must be non-negative")
    if start > end:
        raise ValueError("span_start must be <= span_end")
    if end >= attacked_text.num_words:
        raise ValueError(
            f"span_end={end} is out of range for num_words={attacked_text.num_words}"
        )

    return start, end


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    if args.device is not None:
        os.environ["TA_DEVICE"] = args.device

    from textattack.shared import AttackedText
    from attack.methods.transformations.span_seq2seq_rewrite import SpanSeq2SeqRewrite

    attacked_text = AttackedText(args.text)
    start_idx, end_idx = _validate_and_get_span(args, attacked_text)

    transformation = SpanSeq2SeqRewrite(
        rewrite_model_name=args.rewrite_model_name,
        num_candidate=args.num_candidate,
        min_span_len=args.min_span_len,
        max_span_len=args.max_span_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        device=args.device,
    )

    selected_indices = list(range(start_idx, end_idx + 1))
    transformed = transformation(
        attacked_text,
        pre_transformation_constraints=[],
        indices_to_modify=selected_indices,
    )

    source_span = " ".join(attacked_text.words[start_idx : end_idx + 1])
    top_k = max(1, args.print_top_k)
    candidate_texts = [candidate.text for candidate in transformed[:top_k]]

    result = {
        "model_name": args.rewrite_model_name,
        "device": transformation.device,
        "num_words": attacked_text.num_words,
        "span_start": start_idx,
        "span_end": end_idx,
        "source_span": source_span,
        "num_candidates_generated": len(transformed),
        "preview_candidates": candidate_texts,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
