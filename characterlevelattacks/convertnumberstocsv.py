# python
import argparse
import subprocess
from pathlib import Path
import shlex
import sys

def convert_numbers_to_csv(src: str, dest_dir: str = None, output_name: str = None, timeout: int = 30) -> Path:
    """
    Convert a .numbers file to CSV using Numbers.app (macOS) via AppleScript (osascript).
    - src: path to .numbers file
    - dest_dir: directory to save CSV (defaults to ~/Downloads)
    - output_name: optional output filename (without extension). If omitted, use source stem.
    - timeout: seconds to wait for osascript
    Returns Path to written CSV on success, raises RuntimeError on failure.
    """
    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")
    if src_path.suffix.lower() != ".numbers":
        raise ValueError("Source file does not have a .numbers extension")

    dest_dir_path = Path(dest_dir).expanduser().resolve() if dest_dir else (Path.home() / "Downloads")
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    base = output_name or src_path.stem
    dest_path = dest_dir_path / f"{base}.csv"

    # AppleScript to open Numbers document and export to CSV
    # Use POSIX paths to interop with osascript
    applescript = f'''
    tell application "Numbers"
      activate
      set theDoc to open POSIX file "{src_path.as_posix()}"
      delay 0.6
      try
        export theDoc to POSIX file "{dest_path.as_posix()}" as CSV
        close theDoc saving no
        return "OK"
      on error errMsg number errNum
        try
          close theDoc saving no
        end try
        return "ERROR: " & errNum & " - " & errMsg
      end try
    end tell
    '''

    # Call osascript
    try:
        proc = subprocess.run(
            ["osascript", "-e", applescript],
            text=True,
            capture_output=True,
            timeout=timeout
        )
    except FileNotFoundError:
        raise RuntimeError("osascript not found: this script must run on macOS with osascript available.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("osascript timed out while converting; increase timeout or try manually.")

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if proc.returncode != 0 or out.startswith("ERROR") or "error" in out.lower() or err:
        msg = out or err or f"osascript failed with returncode {proc.returncode}"
        raise RuntimeError(f"Failed to convert .numbers -> .csv: {msg}")

    if not dest_path.exists():
        raise RuntimeError("Conversion reported success but output file not found: " + str(dest_path))

    return dest_path

def main():
    p = argparse.ArgumentParser(description="Convert Apple .numbers to CSV and save to Downloads (macOS).")
    p.add_argument("source", help=".numbers file to convert")
    p.add_argument("--dest-dir", help="Destination directory (default: ~/Downloads)", default=str(Path.home() / "Downloads"))
    p.add_argument("--name", help="Output filename (without .csv). Defaults to source stem", default=None)
    p.add_argument("--timeout", type=int, help="osascript timeout seconds", default=30)
    args = p.parse_args()

    try:
        out = convert_numbers_to_csv(args.source, dest_dir=args.dest_dir, output_name=args.name, timeout=args.timeout)
        print(f"Saved CSV to: {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()