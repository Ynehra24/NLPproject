import pandas as pd

lambdas = ["0_0", "1_0"]

base_path = "gradientBasedAttacks/results/metrics"

for lam in lambdas:
    df = pd.read_csv(f"{base_path}/lambda_{lam}/evaded.csv")

    paired = pd.DataFrame({
        "pair_id": df["id"],
        "original_text": df["original_text"],
        "humanized_text": df["text"],
        "attack_type": "gradient",
        "generator_model": df.get("generator_model", "unknown")
    })

    # 🔥 FILTER BAD SAMPLES
    paired = paired[
        (paired["humanized_text"].str.len() > 50) &
        (~paired["humanized_text"].str.contains("Too many requests", na=False))
    ]

    paired.to_csv(f"{base_path}/lambda_{lam}/paired_clean.csv", index=False)

    print(f"✅ Cleaned λ={lam}: {len(paired)} samples")