import os
import pandas as pd
import shutil


# ---------- Helpers ----------

def ensure_sample_dir(base_path):
    sample_dir = os.path.join(base_path, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    return sample_dir


def get_preview_name(file_name):
    name, ext = os.path.splitext(file_name)
    return f"{name}_preview{ext}"


def safe_read_csv(file_path, n):
    try:
        return pd.read_csv(file_path, encoding="utf-8").head(n)
    except:
        return pd.read_csv(
            file_path,
            encoding="latin1",
            on_bad_lines="skip"
        ).head(n)


# ---------- Core Processing ----------

def process_file(file_path, sample_dir, n=100):
    file_name = os.path.basename(file_path)

    # ❌ Skip already processed preview files
    if "_preview" in file_name:
        return

    out_name = get_preview_name(file_name)
    out_path = os.path.join(sample_dir, out_name)

    ext = file_name.split('.')[-1].lower()

    try:
        if ext == "csv":
            df = safe_read_csv(file_path, n)
            df.to_csv(out_path, index=False)

        elif ext == "parquet":
            df = pd.read_parquet(file_path)
            df.head(n).to_parquet(out_path, index=False)

        elif ext == "json":
            df = pd.read_json(file_path)
            df.head(n).to_json(out_path, orient="records", lines=True)

        elif ext in ["txt", "log"]:
            with open(file_path, errors="ignore") as f:
                lines = []
                for _ in range(n):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)

            with open(out_path, "w") as f:
                f.writelines(lines)

        else:
            # fallback → copy small file
            shutil.copy(file_path, out_path)

        print(f"✅ Saved preview: {out_path}")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")


# ---------- Main Function ----------

def create_sample_dataset(path, n=100):
    if not os.path.exists(path):
        print("❌ Invalid path")
        return

    base_path = os.path.dirname(path) if os.path.isfile(path) else path
    sample_dir = ensure_sample_dir(base_path)

    # ---------- Case 1: Single file ----------
    if os.path.isfile(path):
        process_file(path, sample_dir, n)

    # ---------- Case 2: Folder ----------
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):

            # ❌ Skip sample folder completely
            if "sample" in root:
                continue

            for file in files:
                file_path = os.path.join(root, file)
                process_file(file_path, sample_dir, n)


# ---------- Run ----------

if __name__ == "__main__":
    # 🔁 Change this path
    path = "/Users/yatharthnehva/NLPproject/characterlevelattacks/stylometric/SPGC/gutenberg_en_20k.parquet"

    create_sample_dataset(path, n=100)