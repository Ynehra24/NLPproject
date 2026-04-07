"""
Download HC3 dataset from HuggingFace and cache locally.
Usage: python scripts/download_hc3.py
"""
import json
from huggingface_hub import hf_hub_download
from pathlib import Path

OUTPUT = Path("data/raw/all.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    print(f"Already exists: {OUTPUT}")
else:
    print("Downloading HC3 all.jsonl...")
    filepath = hf_hub_download(
        repo_id="Hello-SimpleAI/HC3",
        filename="all.jsonl",
        repo_type="dataset"
    )
    import shutil
    shutil.copy(filepath, OUTPUT)
    print(f"Saved to {OUTPUT}")

# Count records
with open(OUTPUT) as f:
    lines = f.readlines()
print(f"Total QA pairs: {len(lines)}")
