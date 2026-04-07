"""
λ-Sweep Ablation — Core Experiment
-------------------------------------
Trains one LoRA adapter per λ value and evaluates ASR + CPTR.

Usage:
    python evaluation/lambda_sweep.py \
      --train data/splits/train.csv \
      --test  data/splits/test.csv \
      --lambda-values 0.0 0.25 0.5 0.75 1.0 \
      --epochs 3 --batch-size 8 \
      --output results/metrics/lambda_sweep.csv \
      --device cuda
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path


def run(args):
    results = []
    for lam in args.lambda_values:
        lam_str = str(lam).replace(".", "_")
        out_dir = f"results/evader_outputs/lambda_{lam_str}"

        print(f"\n{'='*50}")
        print(f"Training λ = {lam}")
        print(f"{'='*50}")

        # Train
        subprocess.run([
            "python", "evader/lora_adapter/model.py",
            "--train",      args.train,
            "--val",        args.val if hasattr(args, 'val') else "data/splits/val.csv",
            "--lam",        str(lam),
            "--alpha",      "0.3",
            "--epochs",     str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--output",     out_dir,
            "--device",     args.device,
        ], check=True)

        # Evaluate
        evaded_path = f"{out_dir}/evaded.csv"
        metrics_path = f"results/metrics/eval3_lambda_{lam_str}.csv"

        subprocess.run([
            "python", "evaluation/metrics.py",
            "--test",   args.test,
            "--evaded", evaded_path,
            "--output", metrics_path,
        ], check=True)

        m = pd.read_csv(metrics_path).iloc[0].to_dict()
        m["lambda"] = lam
        results.append(m)
        print(f"λ={lam} → ASR: {m.get('asr')}%  BERTScore: {m.get('bertscore_f1')}")

    df = pd.DataFrame(results)[["lambda", "asr", "bertscore_f1", "rouge_l"]]
    df.to_csv(args.output, index=False)
    print(f"\nSweep complete. Results → {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",         default="data/splits/train.csv")
    p.add_argument("--test",          default="data/splits/test.csv")
    p.add_argument("--lambda-values", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--epochs",        type=int, default=3)
    p.add_argument("--batch-size",    type=int, default=8)
    p.add_argument("--output",        default="results/metrics/lambda_sweep.csv")
    p.add_argument("--device",        default="cuda")
    run(p.parse_args())
