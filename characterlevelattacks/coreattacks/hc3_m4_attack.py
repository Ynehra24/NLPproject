import sys
import os
import random
import pyarrow as pa
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ensure local imports work
sys.path.insert(0, str(Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks").resolve()))
from homoglyph_attack import attack_text

random.seed(42)

# ---------------------------
# Paths
# ---------------------------
HC3_PATH = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/humanvsai/hc3_dataset/test/data-00000-of-00001.arrow")
M4_PATH = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/humanvsai/m4dataset/m4_merged.parquet")
OUTPUT_DIR = Path("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/attacked_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Loaders
# ---------------------------
def load_hc3(n_samples=500):
    """Loads HC3 from Arrow IPC stream and extracts ChatGPT answers."""
    print(f"Loading HC3 from {HC3_PATH}...")
    with pa.ipc.open_stream(str(HC3_PATH)) as reader:
        # We only need the first batch for 500 samples usually, but let's be safe
        batches = []
        count = 0
        for batch in reader:
            batches.append(batch)
            count += batch.num_rows
            if count >= n_samples:
                break
        
        table = pa.Table.from_batches(batches)
        df = table.to_pandas().head(n_samples)
        
        # HC3 'chatgpt_answers' is an array per row. We'll take the first one.
        df['text'] = df['chatgpt_answers'].apply(lambda x: x[0] if len(x) > 0 else "")
        return df[['text']]

def load_m4(n_samples=500):
    """Loads M4 from Parquet and uses the 'Output' column."""
    print(f"Loading M4 from {M4_PATH}...")
    # Since it's huge, we use row group filtering or just read the head
    # For now, reading head is fine since we only need 500 samples
    df = pd.read_parquet(M4_PATH).head(n_samples)
    df = df.rename(columns={'Output': 'text'})
    return df[['text']]

# ---------------------------
# Attack Runner
# ---------------------------
def run_dataset_attacks(name, df, modes=['homoglyph', 'diacritic', 'mixed']):
    print(f"\n--- Attacking {name} ({len(df)} rows) ---")
    for mode in modes:
        tqdm.pandas(desc=f"  {name} [{mode}]")
        attk_df = df.copy()
        attk_df['attacked_text'] = attk_df['text'].progress_apply(lambda t: attack_text(t, mode=mode))
        
        out_path = OUTPUT_DIR / f"{name}_{mode}.csv"
        attk_df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path.name}")

if __name__ == "__main__":
    # HC3
    try:
        hc3_df = load_hc3(500)
        run_dataset_attacks("hc3", hc3_df)
    except Exception as e:
        print(f"Error processing HC3: {e}")

    # M4
    try:
        m4_df = load_m4(500)
        run_dataset_attacks("m4", m4_df)
    except Exception as e:
        print(f"Error processing M4: {e}")

    print("\nHC3 and M4 attacks completed.")
