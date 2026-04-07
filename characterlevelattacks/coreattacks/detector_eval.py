import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure we can import the composite scorer
sys.path.insert(0, str(Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks").resolve()))
from composite_scorer import composite_score

# ---------------------------
# Setup & Paths
# ---------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"[Detector Eval] Using device: {device}")

INPUT_DIR = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/attacked_outputs")

# Dynamically find all datasets that have been attacked (sst2, hc3, etc.)
DATASETS = sorted(list(set([f.stem.split('_')[0] for f in INPUT_DIR.glob('*.csv') if '_' in f.name and 'metrics' not in f.name])))
ATTACK_MODES = ["homoglyph", "diacritic", "mixed", "emoji"]

# ---------------------------
# HuggingFace Models
# ---------------------------
# Replace these with the exact TextAttack repo IDs you used for HC3/M4 if different.
# e.g., using ag-news or imdb models as placeholders for BERT/RoBERTa baselines.
MODELS = {
    "BERT-base": "textattack/bert-base-uncased-imdb",
    "RoBERTa-base": "textattack/roberta-base-imdb"
}

# ---------------------------
# Batch Inference Helper
# ---------------------------
def predict_batch(texts: list, tokenizer, model, batch_size: int = 64) -> list:
    """Runs fast batched inference on MPS/GPU."""
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Predicting", leave=False):
        batch = texts[i : i + batch_size]
        # TextAttack models sometimes struggle with pure empty strings
        batch = [t if isinstance(t, str) and t.strip() else " " for t in batch]
        
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).tolist())
    return preds

# ---------------------------
# Evaluation Routine
# ---------------------------
def evaluate_all():
    summary_results = []

    for model_name, model_id in MODELS.items():
        print(f"\n========================================")
        print(f" Loading Detector: {model_name}")
        print(f"========================================")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        model.eval()

        for dataset in DATASETS:
            for mode in ATTACK_MODES:
                # Expecting files like hc3_homoglyph.csv
                csv_path = INPUT_DIR / f"{dataset}_{mode}.csv"
                if not csv_path.exists():
                    continue
                    
                print(f"\nEvaluating [ {dataset.upper()} | {mode} ]")
                df = pd.read_csv(csv_path)
                
                # We expect the CSV to have the original text in, e.g., 'text'
                # and the perturbed humanized text in 'attacked_text'.
                # Adjust 'text' if your dataset uses 'premise', 'sentence', etc.
                orig_col = 'text' if 'text' in df.columns else df.columns[0]
                attk_col = f'attacked_{orig_col}'
                
                if attk_col not in df.columns:
                    print(f"  [!] Missing {attk_col} in {csv_path.name}")
                    continue

                orig_texts = df[orig_col].tolist()
                attk_texts = df[attk_col].tolist()
                
                # 1. Get predictions for original texts
                orig_preds = predict_batch(orig_texts, tokenizer, model)
                
                # 2. Get predictions for humanized texts
                attk_preds = predict_batch(attk_texts, tokenizer, model)
                
                # 3. Compute Attack Success Rate (ASR)
                # ASR = original was classified correctly (or as AI), but attacked swapped the label.
                # Assuming binary classification where we just care if the label flipped:
                successes = []
                s_scores = []
                
                for i in range(len(df)):
                    flipped = orig_preds[i] != attk_preds[i]
                    successes.append(1 if flipped else 0)
                    
                    if flipped:
                        # Only compute Composite S for successful evasions
                        s_val = composite_score(orig_texts[i], attk_texts[i])['S']
                        s_scores.append(s_val)

                total_evaluated = len(df)
                total_success   = sum(successes)
                asr             = (total_success / total_evaluated) * 100 if total_evaluated > 0 else 0
                avg_s           = np.mean(s_scores) if s_scores else 0.0

                print(f"  -> ASR          : {asr:.2f}%  ({total_success}/{total_evaluated} flips)")
                if s_scores:
                    print(f"  -> Avg S Score  : {avg_s:.4f}")
                else:
                    print(f"  -> Avg S Score  : N/A (0 successes)")

                summary_results.append({
                    "Detector": model_name,
                    "Dataset": dataset.upper(),
                    "Mode": mode,
                    "Total": total_evaluated,
                    "Flips": total_success,
                    "ASR (%)": round(asr, 2),
                    "Avg Composite S": round(avg_s, 4)
                })

        # Free up VRAM before loading the next detector
        del model
        del tokenizer
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Print clean summary table at the very end
    print("\n\n" + "="*60)
    print("                 FINAL EVALUATION SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))
    
    # Save the summary to disk
    out_path = INPUT_DIR / "final_evaluation_metrics.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nMetrics saved to: {out_path}")

if __name__ == "__main__":
    evaluate_all()
