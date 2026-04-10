import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Import all updated v2 modules ─────────────────────────────────────────────
sys.path.insert(0, str(Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks").resolve()))

from composite_scorer import composite_score, analyze_original, gpt2_analyze
from csbp_loop import csbp_loop, generate_candidates, scorched_earth_word
from homoglyph_attack import apply_homoglyph, apply_diacritic, is_eligible
from humanizer import humanize, get_register, ensure_support_models_loaded

# ---------------------------
# Setup & Paths
# ---------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"[Detector Eval] Using device: {device}")

INPUT_DIR = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/attacked_outputs")

# Dynamically find all datasets that have been attacked (sst2, hc3, etc.)
DATASETS = sorted(list(set([f.stem.split('_')[0] for f in INPUT_DIR.glob('*.csv') if '_' in f.name and 'metrics' not in f.name])))
ATTACK_MODES = ["homoglyph", "diacritic", "mixed", "emoji", "humanizer"]

# ---------------------------
# HuggingFace Models
# ---------------------------
MODELS = {
    "BERT-base":    "textattack/bert-base-uncased-imdb",
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
        print(f"\n{'='*40}")
        print(f" Loading Detector: {model_name}")
        print(f"{'='*40}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        model.eval()

        for dataset in DATASETS:
            for mode in ATTACK_MODES:
                csv_path = INPUT_DIR / f"{dataset}_{mode}.csv"
                if not csv_path.exists():
                    continue

                print(f"\nEvaluating [ {dataset.upper()} | {mode} ]")
                df = pd.read_csv(csv_path)

                orig_col = 'original_text' if 'original_text' in df.columns else ('text' if 'text' in df.columns else df.columns[0])

                if 'humanized_text' in df.columns:
                    attk_col = 'humanized_text'
                else:
                    attk_col = f'attacked_{orig_col}'

                if attk_col not in df.columns:
                    print(f"  [!] Missing {attk_col} in {csv_path.name}")
                    continue

                orig_texts = df[orig_col].tolist()
                raw_attk_texts = df[attk_col].tolist()

                attk_texts = []
                import ast
                for raw in raw_attk_texts:
                    if isinstance(raw, str) and raw.strip().startswith('{'):
                        try:
                            # It's a dictionary string from humanizer output
                            parsed = ast.literal_eval(raw)
                            attk_texts.append(parsed.get('humanized_text') or parsed.get('best_text') or str(raw))
                        except Exception:
                            attk_texts.append(str(raw))
                    else:
                        attk_texts.append(str(raw))

                # 1. Predictions on original texts
                orig_preds = predict_batch(orig_texts, tokenizer, model)

                # 2. Predictions on humanized texts
                attk_preds = predict_batch(attk_texts, tokenizer, model)

                # 3. Compute ASR + v2 composite score breakdown
                successes = []
                s_scores = []
                evasion_scores = []
                bpe_scores = []
                wm_z_scores = []
                cos_scores = []
                ppl_scores = []
                rank_scores = []

                for i in range(len(df)):
                    flipped = orig_preds[i] != attk_preds[i]
                    successes.append(1 if flipped else 0)

                    # Run v2 composite score for every example (not just flips)
                    result = composite_score(str(orig_texts[i]), str(attk_texts[i]))
                    s_scores.append(result['S'])
                    evasion_scores.append(result['evasion'])
                    bpe_scores.append(result['bpe_disruption'])
                    wm_z_scores.append(result['watermark_z'])
                    cos_scores.append(result['cosine'])
                    ppl_scores.append(result['ppl'])
                    rank_scores.append(result['avg_rank'])

                total_evaluated = len(df)
                total_success   = sum(successes)
                asr             = (total_success / total_evaluated) * 100 if total_evaluated > 0 else 0
                avg_S           = float(np.mean(s_scores))
                avg_evasion     = float(np.mean(evasion_scores))
                avg_bpe         = float(np.mean(bpe_scores))
                avg_wm_z        = float(np.mean(wm_z_scores))
                avg_cos         = float(np.mean(cos_scores))
                avg_ppl         = float(np.mean(ppl_scores))
                avg_rank        = float(np.mean(rank_scores))

                print(f"  -> ASR          : {asr:.2f}%  ({total_success}/{total_evaluated} flips)")
                print(f"  -> Avg S        : {avg_S:.4f}")
                print(f"  -> Avg Evasion  : {avg_evasion:.4f}")
                print(f"  -> Avg BPE Disr : {avg_bpe:.4f}")
                print(f"  -> Avg WM z     : {avg_wm_z:.4f}")
                print(f"  -> Avg Cosine   : {avg_cos:.4f}")
                print(f"  -> Avg PPL      : {avg_ppl:.1f}")
                print(f"  -> Avg Rank     : {avg_rank:.1f}")

                summary_results.append({
                    "Detector":        model_name,
                    "Dataset":         dataset.upper(),
                    "Mode":            mode,
                    "Total":           total_evaluated,
                    "Flips":           total_success,
                    "ASR (%)":         round(asr, 2),
                    "Avg S":           round(avg_S, 4),
                    "Avg Evasion":     round(avg_evasion, 4),
                    "Avg BPE Disrupt": round(avg_bpe, 4),
                    "Avg WM z-score":  round(avg_wm_z, 4),
                    "Avg Cosine":      round(avg_cos, 4),
                    "Avg PPL":         round(avg_ppl, 2),
                    "Avg Rank":        round(avg_rank, 2),
                })

        # Free up memory before next detector
        del model
        del tokenizer
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Print & save summary
    print("\n\n" + "="*60)
    print("                 FINAL EVALUATION SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))

    out_path = INPUT_DIR / "final_evaluation_metrics.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    evaluate_all()
