import sys
from pathlib import Path
import traceback

sys.path.insert(0, str(Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks").resolve()))
from humanizer import build_confidence_for_label
from transformers import pipeline

pipe = pipeline('text-classification', model='textattack/bert-base-uncased-ag-news', device='mps')

print("Pipe created.")
conf_fn = build_confidence_for_label(pipe, "LABEL_1")

text = "Fan v Fan: Manchester City-Tottenham Hotspur This weekend Manchester City entertain Spurs"
try:
    print("Calling conf_fn...")
    score = conf_fn(text)
    print("Score:", score)
    
    # Try it without try/except!
    outputs = pipe(text, truncation=True, top_k=None)[0]
    for entry in outputs:
        if entry['label'] == "LABEL_1":
            print("Found natively:", float(entry['score']))
except Exception as e:
    traceback.print_exc()

