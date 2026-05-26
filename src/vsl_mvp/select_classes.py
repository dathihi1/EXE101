from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .utils import write_json


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Select MVP classes from a manifest by sample count.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--num-classes", default=15, type=int)
    parser.add_argument("--min-samples", default=20, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    counts = df["gloss"].dropna().value_counts()
    selected = counts[counts >= args.min_samples].head(args.num_classes)
    payload = {
        "classes": selected.index.tolist(),
        "counts": {str(k): int(v) for k, v in selected.items()},
        "selection_rule": f"top {args.num_classes} classes with at least {args.min_samples} samples",
    }
    write_json(args.out, payload)
    print(f"Selected {len(selected)} classes -> {args.out}")
    print(selected.to_string())


if __name__ == "__main__":
    main()
