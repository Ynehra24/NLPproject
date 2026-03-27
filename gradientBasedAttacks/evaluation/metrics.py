"""
Evaluation Metrics for Gradient-Based Evader
----------------------------------------------
Usage:
    python evaluation/metrics.py \
      --test  data/splits/test.csv \
      --evaded results/evader_outputs/lambda_0.5/evaded.csv \
      --output results/metrics/eval3_lambda_0.5.csv
"""

import argparse
import pandas as pd
import torch
import numpy as np
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def compute_asr(evaded_texts, detector, det_tokenizer, device, batch_size=32):
    """Attack Success Rate — % evaded texts classified as human."""
    detector.eval()
    preds = []
    for i in range(0, len(evaded_texts), batch_size):
        batch = evaded_texts[i:i+batch_size]
        enc = det_tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = detector(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    # label 0 = human → success if predicted human
    return sum(p == 0 for p in preds) / len(preds) * 100


def compute_bertscore(originals, paraphrases, device):
    _, _, F = bert_score(paraphrases, originals, lang="en", device=device, verbose=False)
    return F.mean().item()


def compute_rouge_l(originals, paraphrases):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(o, p)["rougeL"].fmeasure for o, p in zip(originals, paraphrases)]
    return np.mean(scores)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_df   = pd.read_csv(args.test)
    evade_df  = pd.read_csv(args.evaded)

    ai_orig   = test_df[test_df["source"] == "ai"]["text"].tolist()
    ai_evaded = evade_df["text"].tolist()

    print("Loading detector...")
    det_tok  = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
    detector = AutoModelForSequenceClassification.from_pretrained(
        "Hello-SimpleAI/chatgpt-detector-roberta"
    ).to(device)

    print("Computing ASR...")
    asr = compute_asr(ai_evaded, detector, det_tok, device)

    print("Computing BERTScore...")
    bs = compute_bertscore(ai_orig[:len(ai_evaded)], ai_evaded, device)

    print("Computing ROUGE-L...")
    rl = compute_rouge_l(ai_orig[:len(ai_evaded)], ai_evaded)

    results = {
        "asr":         round(asr, 2),
        "bertscore_f1": round(bs, 4),
        "rouge_l":     round(rl, 4),
    }

    print("\n── RESULTS ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<20} {v}")

    pd.DataFrame([results]).to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test",    default="data/splits/test.csv")
    p.add_argument("--evaded",  required=True)
    p.add_argument("--output",  required=True)
    run(p.parse_args())
