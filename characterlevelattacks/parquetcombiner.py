import pandas as pd

files = [
    "/Users/yatharthnehva/NLPproject/characterlevelattacks/humanvsai/m4dataset/combined_datasets_batch_1.parquet",
    "/Users/yatharthnehva/NLPproject/characterlevelattacks/humanvsai/m4dataset/combined_datasets_batch_2.parquet",
    "/Users/yatharthnehva/NLPproject/characterlevelattacks/humanvsai/m4dataset/combined_datasets_batch_3.parquet"
]

df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

# Save RIGHT HERE (current folder)
df.to_parquet("m4_merged.parquet", index=False)

print("Saved as m4_merged.parquet in current directory")