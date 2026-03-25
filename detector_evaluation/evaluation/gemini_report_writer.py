"""Generate a structured Markdown analysis report using Gemini API.

This script reads outputs from the local analysis pipeline (tables, insights,
manifest), builds a rich context bundle, sends it to Gemini, and saves a
report markdown file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TEXT_EXTENSIONS = {".csv", ".json", ".md", ".txt", ".log"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis report markdown via Gemini API")
    parser.add_argument("--analysis-root", default="results/analysis_suite", help="Root folder with analysis outputs")
    parser.add_argument(
        "--prompt-file",
        default="prompts/gemini_analysis_report_prompt.md",
        help="Prompt template markdown file",
    )
    parser.add_argument("--output", default="reports/analysis_report_gemini.md", help="Output markdown path")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Environment variable for Gemini API key")
    parser.add_argument("--max-context-chars", type=int, default=140000, help="Max chars included from analysis files")
    parser.add_argument("--max-file-chars", type=int, default=15000, help="Max chars included per file")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    return parser.parse_args()


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - 200]
    tail = text[-120:]
    return f"{head}\n\n...[TRUNCATED]...\n\n{tail}"


def collect_analysis_files(analysis_root: Path) -> List[Path]:
    preferred_patterns = [
        "run_manifest.json",
        "tables/*.csv",
        "insights/**/*.csv",
        "figures/**/*.md",
    ]

    files: List[Path] = []
    seen = set()

    for pattern in preferred_patterns:
        for p in sorted(analysis_root.glob(pattern)):
            if p.is_file() and p not in seen and _is_text_file(p):
                files.append(p)
                seen.add(p)

    for p in sorted(analysis_root.rglob("*")):
        if p.is_file() and p not in seen and _is_text_file(p):
            files.append(p)
            seen.add(p)

    return files


def build_context(analysis_root: Path, max_context_chars: int, max_file_chars: int) -> Tuple[str, List[Path], int]:
    files = collect_analysis_files(analysis_root)
    if not files:
        raise FileNotFoundError(f"No text analysis files found under {analysis_root}")

    sections: List[str] = []
    used: List[Path] = []
    omitted = 0
    total = 0

    for file_path in files:
        rel = file_path.as_posix()
        text = file_path.read_text(encoding="utf-8", errors="replace")
        text = _truncate(text, max_file_chars)

        block = f"\n## SOURCE FILE: {rel}\n\n```text\n{text}\n```\n"
        if total + len(block) > max_context_chars:
            omitted += 1
            continue

        sections.append(block)
        used.append(file_path)
        total += len(block)

    context = "\n".join(sections)
    return context, used, omitted


def compose_prompt(prompt_template: str, context: str, used_files: List[Path], omitted_count: int) -> str:
    file_manifest = "\n".join(f"- {p.as_posix()}" for p in used_files)
    suffix = (
        "\n\n# INPUT FILE MANIFEST\n"
        f"Included files ({len(used_files)}):\n{file_manifest}\n\n"
        f"Omitted files due to context budget: {omitted_count}\n"
        "If information appears incomplete, state that clearly in the report.\n"
    )

    payload_context = f"\n\n# ANALYSIS CONTEXT\n\n{context}\n"

    if "{{ANALYSIS_CONTEXT}}" in prompt_template:
        prompt = prompt_template.replace("{{ANALYSIS_CONTEXT}}", payload_context)
    else:
        prompt = prompt_template + payload_context

    return prompt + suffix


def call_gemini(prompt: str, model: str, api_key: str, temperature: float) -> str:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }

    data = json.dumps(body).encode("utf-8")
    req = Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urlopen(req, timeout=180) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Gemini API connection error: {exc}") from exc

    payload = json.loads(raw)
    candidates = payload.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {json.dumps(payload)[:1000]}")

    parts = candidates[0].get("content", {}).get("parts", [])
    chunks = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")

    return text


def main() -> None:
    args = parse_args()

    analysis_root = Path(args.analysis_root)
    if not analysis_root.exists():
        raise FileNotFoundError(f"Analysis root does not exist: {analysis_root}")

    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing Gemini API key in environment variable: {args.api_key_env}")

    context, used_files, omitted_count = build_context(
        analysis_root=analysis_root,
        max_context_chars=args.max_context_chars,
        max_file_chars=args.max_file_chars,
    )

    prompt_template = prompt_file.read_text(encoding="utf-8")
    prompt = compose_prompt(prompt_template, context, used_files, omitted_count)

    report_md = call_gemini(
        prompt=prompt,
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    print(f"Saved Gemini report: {out_path.resolve()}")
    print(f"Used files: {len(used_files)} | Omitted files: {omitted_count}")


if __name__ == "__main__":
    main()
