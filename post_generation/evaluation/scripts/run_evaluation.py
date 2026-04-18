#!/usr/bin/env python
"""
Simple end-to-end evaluation of Flan-T5 evader against detector_evaluation suite.
Reproduces paper metrics.
"""

import argparse
import json
import csv
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import sys


def generate_humanized_text(model_path, texts, batch_size=16, device="cuda"):
    """Generate humanized text using Flan-T5 evader model."""
    
    print(f"\n{'='*80}")
    print(f"[STEP 1/4] Loading Flan-T5 Evader Model")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.to(device)
    
    model.eval()
    
    print(f"Generating humanized text for {len(texts)} samples...")
    humanized_texts = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                temperature=0.8,
                top_p=0.9,
                length_penalty=1.0
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            humanized_texts.extend(decoded)
    
    print(f"✓ Generated {len(humanized_texts)} humanized texts")
    return humanized_texts


def run_detector_evaluation(humanized_csv, output_dir, roberta_model, device="cuda"):
    """Run detector_evaluation pipeline on humanized texts."""
    
    print(f"\n{'='*80}")
    print(f"[STEP 2/4] Running Detectors from detector_evaluation")
    print(f"{'='*80}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    df = pd.read_csv(humanized_csv)
    if "id" not in df.columns:
        df["id"] = range(len(df))
    if "attack_type" not in df.columns:
        df["attack_type"] = "flan_t5_evader"
    if "source" not in df.columns:
        df["source"] = "generated"
    df.to_csv(humanized_csv, index=False)
    
    detectors = {}
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    
    # RoBERTa Classifier - Direct integration
    print("\nRunning RoBERTa Classifier...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_path = Path(roberta_model)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(device_obj)
        model.eval()
        
        texts = df["text"].astype(str).tolist()
        ai_scores = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), 16), desc="RoBERTa scoring"):
                batch_texts = texts[i:i+16]
                enc = tokenizer(
                    batch_texts, 
                    truncation=True, 
                    max_length=512, 
                    padding=True, 
                    return_tensors="pt"
                )
                enc = {k: v.to(device_obj) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                ai_scores.extend(probs.cpu().numpy().tolist())
        
        detectors["RoBERTa"] = {i: score for i, score in enumerate(ai_scores)}
        print(f"✓ RoBERTa completed ({len(ai_scores)} scores)")
    except Exception as e:
        print(f"✗ RoBERTa error: {e}")
        import traceback
        traceback.print_exc()
    
    return detectors


def load_test_data(csv_path, limit=None):
    """Load test data from CSV."""
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    if limit:
        df = df.head(limit)
    
    texts = df["text"].astype(str).tolist()
    
    # Parse labels: "ai" or "human" from 'source' column
    labels = []
    for source in df.get("source", ["human"] * len(df)):
        labels.append(1 if source.lower() in ["ai", "gpt"] else 0)
    
    return texts, labels, df


def compute_paper_metrics(predictions, labels):
    """Compute same metrics as HMGC paper."""
    
    # Binarize at 0.5 threshold
    pred_binary = (np.array(predictions) >= 0.5).astype(int)
    labels = np.array(labels)
    
    # Avoid issues with all zeros or all ones
    if len(np.unique(pred_binary)) < 2 or len(np.unique(labels)) < 2:
        return {
            "auc_roc": 0.5,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tnr": 0.0,
            "evasion_rate": 0.0,
            "error": "Insufficient class variance"
        }
    
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.5
    
    acc = accuracy_score(labels, pred_binary)
    prec = precision_score(labels, pred_binary, zero_division=0)
    recall = recall_score(labels, pred_binary, zero_division=0)
    
    # TNR = True Negative Rate = TN / (TN + FP)
    tn = np.sum((pred_binary == 0) & (labels == 0))
    fp = np.sum((pred_binary == 1) & (labels == 0))
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Evasion rate: % of AI-generated texts misclassified as human
    ai_mask = labels == 1
    if np.sum(ai_mask) > 0:
        ai_as_human = np.sum((pred_binary == 0) & ai_mask)
        evasion_rate = (ai_as_human / np.sum(ai_mask)) * 100.0
    else:
        evasion_rate = 0.0
    
    return {
        "auc_roc": float(auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(recall),
        "tnr": float(tnr),
        "evasion_rate": float(evasion_rate)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flan-T5 Evader Model")
    parser.add_argument("--test-data", 
                        default="detector_evaluation/results/checkgpt_phase3_input_100.csv",
                        help="Test dataset CSV")
    parser.add_argument("--evader-model",
                        default="post_generation/HMGC-dataset/output/checkgpt/model/evader_flan_t5_base_12to15h_earlystop_v1",
                        help="Flan-T5 evader model path")
    parser.add_argument("--roberta-model",
                        default="detector_evaluation/results",
                        help="RoBERTa detector model path")
    parser.add_argument("--limit", type=int, default=100, help="Limit samples")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", default="flan_t5_evaluation", help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print(f"\n{'='*80}")
    print(f"[STEP 0/4] Loading Test Data")
    print(f"{'='*80}")
    print(f"Dataset: {args.test_data}")
    
    test_texts, test_labels, df_orig = load_test_data(args.test_data, limit=args.limit)
    print(f"✓ Loaded {len(test_texts)} samples ({sum(test_labels)} AI, {len(test_labels)-sum(test_labels)} human)")
    
    # Generate humanized text
    humanized_texts = generate_humanized_text(
        args.evader_model,
        test_texts,
        batch_size=16,
        device=args.device
    )
    
    # Save to CSV for detectors
    humanized_csv = f"{args.output_dir}/humanized_texts.csv"
    with open(humanized_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "source", "attack_type"])
        writer.writeheader()
        for i, (text, label) in enumerate(zip(humanized_texts, test_labels)):
            writer.writerow({
                "id": i,
                "text": text,
                "source": "ai" if label == 1 else "human",
                "attack_type": "flan_t5_evader"
            })
    
    print(f"✓ Saved to {humanized_csv}")
    
    # Run detectors
    detector_results = run_detector_evaluation(
        humanized_csv,
        args.output_dir,
        args.roberta_model,
        device=args.device
    )
    
    # Compute metrics
    print(f"\n{'='*80}")
    print(f"[STEP 3/4] Computing Metrics")
    print(f"{'='*80}")
    
    metrics_results = {}
    for detector_name, scores_dict in detector_results.items():
        # Convert dict scores to array in order
        predictions = [scores_dict.get(i, 0.5) for i in range(len(test_labels))]
        metrics = compute_paper_metrics(predictions, test_labels)
        metrics_results[detector_name] = metrics
        print(f"✓ Computed metrics for {detector_name}")
    
    # Print results
    print(f"\n{'='*80}")
    print(f"[STEP 4/4] RESULTS COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nTEST SETUP:")
    print(f"  Evader Model: Flan-T5 Base (12L encoder/decoder, 768 hidden)")
    print(f"  Test Samples: {len(test_texts)}")
    print(f"  AI-generated: {sum(test_labels)}")
    print(f"  Human-written: {len(test_labels) - sum(test_labels)}")
    
    print(f"\nPAPER BASELINE (HMGC Framework):")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ HC3 Detector (White-box Attack):            │")
    print(f"  │   Before:  AUC = 0.9963                     │")
    print(f"  │   After:   AUC = 0.5106  (Δ = -0.4857)     │")
    print(f"  │   Evasion: 97.29%                           │")
    print(f"  │                                             │")
    print(f"  │ CheckGPT Detector (Black-box Attack):       │")
    print(f"  │   Before:  AUC ≈ 0.91                       │")
    print(f"  │   Evasion: ~46%                             │")
    print(f"  └─────────────────────────────────────────────┘")
    
    print(f"\nYOUR MODEL RESULTS (Flan-T5 Evader):")
    for detector_name, metrics in metrics_results.items():
        print(f"\n  {detector_name} Detector:")
        print(f"    AUC-ROC:      {metrics['auc_roc']:.4f}")
        print(f"    Accuracy:     {metrics['accuracy']:.4f}")
        print(f"    Precision:    {metrics['precision']:.4f}")
        print(f"    Recall:       {metrics['recall']:.4f}")
        print(f"    TNR (Spec):   {metrics['tnr']:.4f}")
        print(f"    Evasion Rate: {metrics['evasion_rate']:.2f}%")
        
        if metrics.get("error"):
            print(f"    ⚠️  {metrics['error']}")
    
    # Save results
    results_file = Path(args.output_dir) / "evaluation_metrics.json"
    with open(results_file, "w") as f:
        json.dump({
            "model": "flan_t5_base_evader",
            "test_samples": len(test_texts),
            "ai_samples": sum(test_labels),
            "human_samples": len(test_labels) - sum(test_labels),
            "detectors": metrics_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {results_file}")
    print(f"✓ Humanized texts: {humanized_csv}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
