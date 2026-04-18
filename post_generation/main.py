#!/usr/bin/env python
"""Unified CLI for post_generation workflows."""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import List, Optional, Tuple

COMMAND_MODULES = {
    "train-evader": "post_generation.app.train_evader_app",
    "train-detector": "post_generation.app.train_detector_app",
    "attack": "post_generation.attack.multi_flint_attack",
    "smoke-phase2": "post_generation.attack.scripts.smoke_phase2_span_rewrite",
    "smoke-semantic": "post_generation.attack.scripts.smoke_semantic_constraint",
}

FALLBACK_MODULES = {
    "train-evader": "app.train_evader_app",
    "train-detector": "app.train_detector_app",
    "attack": "attack.multi_flint_attack",
    "smoke-phase2": "attack.scripts.smoke_phase2_span_rewrite",
    "smoke-semantic": "attack.scripts.smoke_semantic_constraint",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run post_generation commands from one entrypoint",
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMAND_MODULES.keys()),
        help="Command to execute",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    parser = _build_parser()

    if not raw_args or raw_args[0] in {"-h", "--help"}:
        parser.print_help()
        raise SystemExit(0)

    parsed = parser.parse_args([raw_args[0]])
    return parsed.command, raw_args[1:]


def _run_module(module_name: str, passthrough: List[str]) -> bool:
    try:
        sys.argv = [module_name, *passthrough]
        runpy.run_module(module_name, run_name="__main__")
        return True
    except ModuleNotFoundError:
        return False
    except ImportError as exc:
        if "Error while finding module specification" in str(exc):
            return False
        raise


def main(argv: Optional[List[str]] = None) -> None:
    command, passthrough = parse_args(argv)
    preferred = COMMAND_MODULES[command]
    fallback = FALLBACK_MODULES[command]

    if _run_module(preferred, passthrough):
        return
    if _run_module(fallback, passthrough):
        return

    raise ModuleNotFoundError(
        f"Unable to resolve command module for '{command}'. Tried {preferred} and {fallback}."
    )


if __name__ == "__main__":
    main()
