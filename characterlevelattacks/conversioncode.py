import sys
import argparse
import os
import pandas as pd

def read_magic(path, n=16):
    with open(path, "rb") as f:
        return f.read(n)

def convert_to_csv(path, out):
    # Debug header
    try:
        magic = read_magic(path, 64)
        print("File magic (first 64 bytes):", magic[:64])
    except Exception as e:
        print("Could not read header:", e)

    # 1. HuggingFace datasets
    try:
        from datasets import Dataset
        print("Trying datasets.Dataset.from_file...")
        ds = Dataset.from_file(path)
        df = ds.to_pandas()
        df.to_csv(out, index=False)
        print("WROTE (datasets):", out)
        return
    except Exception as e:
        print("datasets failed:", e)

    # 2. Arrow IPC file
    try:
        import pyarrow.ipc as ipc
        print("Trying pyarrow.ipc.open_file...")
        tbl = ipc.open_file(path).read_all()
        df = tbl.to_pandas()
        df.to_csv(out, index=False)
        print("WROTE (ipc file):", out)
        return
    except Exception as e:
        print("ipc.open_file failed:", e)

    # 3. Arrow stream
    try:
        import pyarrow.ipc as ipc
        print("Trying pyarrow.ipc.open_stream...")
        tbl = ipc.open_stream(path).read_all()
        df = tbl.to_pandas()
        df.to_csv(out, index=False)
        print("WROTE (ipc stream):", out)
        return
    except Exception as e:
        print("ipc.open_stream failed:", e)

    # 4. Feather
    try:
        import pyarrow.feather as feather
        print("Trying feather.read_table...")
        tbl = feather.read_table(path)
        df = tbl.to_pandas()
        df.to_csv(out, index=False)
        print("WROTE (feather):", out)
        return
    except Exception as e:
        print("feather failed:", e)

    # 5. Pandas feather
    try:
        print("Trying pandas.read_feather...")
        df = pd.read_feather(path)
        df.to_csv(out, index=False)
        print("WROTE (pandas feather):", out)
        return
    except Exception as e:
        print("pandas feather failed:", e)

    # 6. Parquet
    try:
        print("Trying pandas.read_parquet...")
        df = pd.read_parquet(path)
        df.to_csv(out, index=False)
        print("WROTE (parquet):", out)
        return
    except Exception as e:
        print("parquet failed:", e)

    # 7. CSV (in case it's already CSV)
    try:
        print("Trying pandas.read_csv...")
        df = pd.read_csv(path)
        df.to_csv(out, index=False)
        print("WROTE (csv passthrough):", out)
        return
    except Exception as e:
        print("csv read failed:", e)

    # 8. JSON / NDJSON fallback
    try:
        print("Trying JSON/NDJSON fallback...")
        import json
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    rows = []
                    break

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out, index=False)
            print("WROTE (json/ndjson):", out)
            return
    except Exception as e:
        print("json fallback failed:", e)

    # 9. Plain text fallback
    try:
        print("Trying plain text fallback...")
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        df = pd.DataFrame({"text": lines})
        df.to_csv(out, index=False)
        print("WROTE (text fallback):", out)
        return
    except Exception as e:
        print("text fallback failed:", e)

    raise RuntimeError(f"Failed to convert file: {path}")

def main():
    parser = argparse.ArgumentParser(description="Universal file → CSV converter")
    parser.add_argument("input_path", help="Input file (arrow, parquet, csv, json, txt, etc.)")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")

    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    output = args.output or os.path.splitext(input_path)[0] + ".csv"

    convert_to_csv(input_path, output)

    print("\n✅ Conversion complete:")
    print("→", output)

if __name__ == "__main__":
    main()