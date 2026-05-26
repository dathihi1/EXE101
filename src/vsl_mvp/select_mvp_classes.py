from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .utils import write_json


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Select clean MVP classes with sample and signer coverage metadata.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--num-classes", default=30, type=int)
    parser.add_argument("--min-samples", default=18, type=int)
    parser.add_argument("--min-signers", default=2, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["gloss"].notna()].copy()
    df["signer_id"] = df["signer_id"].fillna("").astype(str)
    grouped = (
        df.groupby("gloss")
        .agg(
            samples=("video_path", "count"),
            signers=("signer_id", lambda values: len(set(v for v in values if v))),
            first_clip=("clip_id", "first"),
        )
        .reset_index()
    )
    grouped = grouped[(grouped["samples"] >= args.min_samples) & (grouped["signers"] >= args.min_signers)]
    grouped = grouped.sort_values(["samples", "signers", "gloss"], ascending=[False, False, True]).head(args.num_classes)

    payload = {
        "classes": grouped["gloss"].tolist(),
        "stats": {
            row.gloss: {
                "samples": int(row.samples),
                "signers": int(row.signers),
                "first_clip": str(row.first_clip),
            }
            for row in grouped.itertuples(index=False)
        },
        "selection_rule": (
            f"top {args.num_classes} classes with at least {args.min_samples} samples "
            f"and at least {args.min_signers} signers"
        ),
        "notes": [
            "This is a count/signer based first pass.",
            "Re-rank after V2 extraction using missing hand/pose/face quality metrics.",
        ],
    }
    write_json(args.out, payload)
    print(f"Selected {len(grouped)} classes -> {args.out}")
    if not grouped.empty:
        print(grouped[["gloss", "samples", "signers"]].to_string(index=False))


if __name__ == "__main__":
    main()
