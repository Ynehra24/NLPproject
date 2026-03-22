from datasets import load_dataset
import os

# Path to Downloads (Mac/Linux)
download_path = os.path.expanduser("~/Downloads/hc3_dataset")

# Load (this downloads automatically)
dataset = load_dataset("json", data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl")["train"]

# Create train/test split (80% train, 20% test)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Save to your Downloads folder
dataset.save_to_disk(download_path)

print(f"Saved to: {download_path}")