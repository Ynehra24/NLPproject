#!/usr/bin/env python
"""
End-to-end evaluation of Flan-T5 evader model using detector_evaluation suite.
Computes same metrics as HMGC paper for comparison.
"""

import argparse
import json
import csv
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import subprocess
import pandas as pd
from typing import Dict, List, Tuple


def generate_humanized_text(
    model_path: str,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 256,
    device: str = "cuda"
) -> List[str]:
    """Generate humanized text using Flan-T5 evader model."""
    
    print(f"\n[1/4] Loading Flan-T5 model from {model_path}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    humanized_texts = []
    
    print(f"[1/4] Generating humanized text for {len(texts)} samples...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                top_p=0.95
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            humanized_texts.extend(decoded)
    
    return humanized_texts


def run_roberta_detector(input_csv: str, model_dir: str, device: str = "cuda") -> Dict:
    """Run RoBERTa detector from detector_evaluation."""
    
    print(f"\n[2/4] Running RoBERTa classifier detector...")
    cmd = [
        "python",
        "detector_evaluation/detectors/roberta_classifier/infer.py",
        "--input", input_csv,
        "--model-path", model_dir,
        "--output", "detector_results/roberta_scores.json",
        "--batch-size", "32",
        "--device", device
    ]
    
    Path("detector_results").mkdir(exist_ok=True)
    subprocess.run(cmd, check=False, capture_output=True)
    
    try:
        with open("detector_results/roberta_scores.json") as f:
            return json.load(f)
    except:
        return {}


def run_detectgpt(input_csv: str, device: str = "cuda") -> Dict:
    """Run DetectGPT detector."""
    
    print(f"[2/4] Running DetectGPT...")
    cmd = [
        "python",
        "detector_evaluation/detectors/detectgpt/infer.py",
        "--input", input_csv,
        "--output", "detector_results/detectgpt_scores.json",
        "--device", device
    ]
    
    Path("detector_results").mkdir(exist_ok=True)
    subprocess.run(cmd, check=False, capture_output=True)
    
    try:
        with open("detector_results/detectgpt_scores.json") as f:
            return json.load(f)
    except:
        return {}


def run_stats_baseline(input_csv: str, device: str = "cuda") -> Dict:
    """Run perplexity/entropy baseline."""
    
    print(f"[2/4] Running stats baseline (perplexity/entropy)...")
    cmd = [
        "python",
        "detector_evaluation/detectors/stats_baseline/infer.py",
        "--input", input_csv,
        "--output", "detector_results/stats_baseline_scores.json",
        "--device", device
    ]
    
    Path("detector_results").mkdir(exist_ok=True)
    subprocess.run(cmd, check=False, capture_output=True)
    
    try:
        with open("detector_results/stats_baseline_scores.json") as f:
            return json.load(f)
    except:
        return {}


def load_test_data(csv_path: str, limit: int = 1000) -> Tuple[List[str], List[int]]:
    """Load test data from CSV or JSONL."""
    
    texts = []
    labels = []  # 0=human, 1=gpt
    
    if csv_path.endswith(".csv"):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                if "text" in row:
                    texts.append(row["text"])
                elif "content" in row:
                    texts.append(row["content"])
                else:
                    texts.append(list(row.values())[0])
                
                if "label" in row:
                    labels.append(int(row["label"]))
                else:
                    labels.append(1)  # Assume GPT if no label
    else:  # JSONL
        with open(csv_path) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                obj = json.loads(line)
                texts.append(obj.get("text") or obj.get("content"))
                labels.append(obj.get("label", 1))
    
    return texts, labels


def compute_metrics(predictions: List[float], labels: List[int]) -> Dict:
    """Compute paper metrics from detector predictions."""
    
    # Binarize predictions (threshold 0.5)
    pred_binary = [1 if p > 0.5 else 0 for p in predictions]
    
    # AUC-ROC
    auc = roc_auc_score(labels, predictions)
    
    # Accuracy
    acc = accuracy_score(labels, pred_binary)
    
    # Precision & Recall
    prec = precision_score(labels, pred_binary, zero_division=0)
    recall = recall_score(labels, pred_binary, zero_division=0)
    
    # TNR (True Negative Rate) = specificity
    tn = sum((pred_binary[i] == 0 and labels[i] == 0) for i in range(len(labels)))
    fp = sum((pred_binary[i] == 1 and labels[i] == 0) for i in range(len(labels)))
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Evasion rate (% of GPT texts detected as human)
    gpt_indices = [i for i, l in enumerate(labels) if l == 1]
    if gpt_indices:
        gpt_misclassified = sum(pred_binary[i] == 0 for i in gpt_indices)
        evasion_rate = gpt_misclassified / len(gpt_indices) * 100
    else:
        evasion_rate = 0.0
    
    return {
        "auc_roc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "tnr": tnr,
        "evasion_rate": evasion_rate
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="detector_evaluation/results/checkgpt_phase3_input_100.csv",
                        help="Test dataset CSV/JSONL")
    parser.add_argument("--evader-model", 
                        default="post_generation/HMGC-dataset/output/checkgpt/model/evader_flan_t5_base_12to15h_earlystop_v1",
                        help="Path to Flan-T5 evader model")
    parser.add_argument("--roberta-model",
                        default="detector_evaluation/detectors/roberta_classifier",
                        help="Path to RoBERTa classifier model")
    parser.add_argument("--limit", type=int, default=100, help="Limit samples for quick eval")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_texts, test_labels = load_test_data(args.test_data, limit=args.limit)
    print(f"Loaded {len(test_texts)} samples")
    
    # Generate humanized text
    humanized_texts = generate_humanized_text(
        args.evader_model,
        test_texts,
        device=args.device
    )
    
    # Save humanized texts to CSV for detectors
    humanized_csv = f"{args.output_dir}/humanized_texts.csv"
    with open(humanized_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        for text, label in zip(humanized_texts, test_labels):
            writer.writerow({"text": text, "label": label})
    
    print(f"Saved humanized texts to {humanized_csv}")
    
    # Run detectors
    detectors_results = {}
    
    # RoBERTa
    try:
        roberta_scores = run_roberta_detector(humanized_csv, args.roberta_model, args.device)
        if roberta_scores:
            roberta_preds = [roberta_scores.get(str(i), 0.5) for i in range(len(humanized_texts))]
            detectors_results["RoBERTa"] = compute_metrics(roberta_preds, test_labels)
    except Exception as e:
        print(f"RoBERTa failed: {e}")
    
    # DetectGPT
    try:
        detectgpt_scores = run_detectgpt(humanized_csv, args.device)
        if detectgpt_scores:
            detectgpt_preds = [detectgpt_scores.get(str(i), 0.5) for i in range(len(humanized_texts))]
            detectors_results["DetectGPT"] = compute_metrics(detectgpt_preds, test_labels)
    except Exception as e:
        print(f"DetectGPT failed: {e}")
    
    # Stats baseline
    try:
        stats_scores = run_stats_baseline(humanized_csv, args.device)
        if stats_scores:
            stats_preds = [stats_scores.get(str(i), 0.5) for i in range(len(humanized_texts))]
            detectors_results["Stats Baseline"] = compute_metrics(stats_preds, test_labels)
    except Exception as e:
        print(f"Stats baseline failed: {e}")
    
    # Save results
    print(f"\n[3/4] Saving results...")
    results_json = f"{args.output_dir}/evaluation_results.json"
    with open(results_json, "w") as f:
        json.dump(detectors_results, f, indent=2)
    
    # Print comparison
    print(f"\n[4/4] EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {args.evader_model}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Humanized texts: {len(humanized_texts)}")
    print("=" * 80)
    
    print("\nPAPER BASELINE (HMGC):")
    print("  RoBERTa (HC3):     AUC=0.9963 → 0.5106 after attack")
    print("  RoBERTa (CheckGPT): AUC=0.91 → ~0.46 evasion rate")
    print("=" * 80)
    
    print("\nYOUR MODEL RESULTS (Flan-T5 Evader):")
    for detector_name, metrics in detectors_results.items():
        print(f"\n{detector_name}:")
        print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  TNR:            {metrics['tnr']:.4f}")
        print(f"  Evasion Rate:   {metrics['evasion_rate']:.2f}%")
    
    print("\n" + "=" * 80)
    print(f"Full results saved to: {results_json}")
    print(f"Humanized texts saved to: {humanized_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
