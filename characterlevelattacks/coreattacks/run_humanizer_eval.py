import time
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks").resolve()))
from humanizer import humanize

INPUT_DIR = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/attacked_outputs")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50) 
    parser.add_argument("--dataset", type=str, default="")
    args = parser.parse_args()

    # Find all base datasets we already ran tests for
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(set([f.stem.split('_')[0] for f in INPUT_DIR.glob('*.csv') if '_' in f.name and 'metrics' not in f.name]))
    
    for ds in sorted(datasets):
        # We can read the original text from the homoglyph baseline file
        input_csv = INPUT_DIR / f"{ds}_homoglyph.csv"
        if not input_csv.exists():
            continue
            
        print(f"\n=== Processing {ds} with CSBP Humanizer ===")
        df = pd.read_csv(input_csv)
        if args.samples:
            df = df.head(args.samples)
        
        orig_col = 'text' if 'text' in df.columns else df.columns[0]
        
        keep_cols = [c for c in df.columns if not c.startswith('attacked_')]
        out_df = df[keep_cols].copy()
        
        tqdm.pandas(desc=f"Humanizing {ds}")
        out_df['humanized_text'] = out_df[orig_col].progress_apply(
            lambda t: humanize(
                str(t), 
                iterations=15,       
                n_candidates=10, 
                beam_width=5, 
                device_override="mps",
                target_model="textattack/bert-base-uncased-imdb"
            ).get('humanized_text', str(t))
        )
        
        out_path = INPUT_DIR / f"{ds}_humanizer.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved test run -> {out_path}")

    print("\n\n" + "="*50)
    print(" ALL DATASETS GENERATED! LAUNCHING EVALUATION SCRIPT...")
    print("="*50 + "\n")
    
    # We strip the args so detector_eval.py doesn't get confused by --samples
    sys.argv = ['detector_eval.py']
    
    try:
        from detector_eval import evaluate_all
        evaluate_all()
    except Exception as e:
        print(f"Generation successful, but evaluation script failed: {e}")

if __name__ == "__main__":
    run()
