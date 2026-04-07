"""
Build train/val/test CSV splits from HC3 all.jsonl.
All humanizer modules use the same splits for consistency.

Usage: python scripts/prepare_splits.py
Output: data/splits/train.csv, val.csv, test.csv

Schema: id, text, source, attack_type, attack_owner, generator_model
"""
import json
import random
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/all.jsonl")
SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

ai_texts, human_texts = [], []
with open(RAW) as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        for ans in item.get("chatgpt_answers", []):
            if ans and len(ans.strip()) > 50:
                ai_texts.append({
                    "id": f"hc3_ai_{len(ai_texts):05d}",
                    "text": ans.strip(),
                    "source": "ai",
                    "attack_type": "none",
                    "attack_owner": "",
                    "generator_model": "gpt3.5-turbo"
                })
        for ans in item.get("human_answers", []):
            if ans and len(ans.strip()) > 50:
                human_texts.append({
                    "id": f"hc3_human_{len(human_texts):05d}",
                    "text": ans.strip(),
                    "source": "human",
                    "attack_type": "none",
                    "attack_owner": "",
                    "generator_model": ""
                })

random.shuffle(ai_texts)
random.shuffle(human_texts)

print(f"Total AI texts: {len(ai_texts)}")
print(f"Total Human texts: {len(human_texts)}")

# Splits: test=1000 AI + 1000 Human, val=500+500, rest=train
test_ai,  rest_ai  = ai_texts[:1000],  ai_texts[1000:]
test_hu,  rest_hu  = human_texts[:1000], human_texts[1000:]
val_ai,   train_ai = rest_ai[:500],  rest_ai[500:]
val_hu,   train_hu = rest_hu[:500],  rest_hu[500:]

def save(records, path):
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"Saved {len(records)} rows → {path}")

save(train_ai + train_hu, SPLITS_DIR / "train.csv")
save(val_ai   + val_hu,   SPLITS_DIR / "val.csv")
save(test_ai  + test_hu,  SPLITS_DIR / "test.csv")
