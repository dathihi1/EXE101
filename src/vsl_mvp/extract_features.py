from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import FeatureConfig
from .landmarks import LandmarkExtractor
from .utils import ensure_parent, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmark features from VSL videos.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=Path, help="JSON from select_classes.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sequence-length", default=64, type=int)
    parser.add_argument("--limit-per-class", default=0, type=int)
    parser.add_argument(
        "--sample-frames",
        default=0,
        type=int,
        help="If > 0, run MediaPipe on this many evenly spaced source frames per video.",
    )
    args = parser.parse_args()

    selected = set(read_json(args.classes)["classes"])
    df = pd.read_csv(args.manifest)
    df = df[df["gloss"].isin(selected)].copy()
    if args.limit_per_class > 0:
        df = df.groupby("gloss", group_keys=False).head(args.limit_per_class)
    df = df.reset_index(drop=True)

    labels = sorted(df["gloss"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    config = FeatureConfig(sequence_length=args.sequence_length)
    extractor = LandmarkExtractor(config)

    features = []
    y = []
    paths = []
    signers = []
    statuses = []
    valid_frames = []
    try:
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            result = extractor.extract_video(row.video_path, sample_frames=args.sample_frames)
            features.append(result.features)
            y.append(label_to_id[row.gloss])
            paths.append(row.video_path)
            signers.append("" if pd.isna(row.signer_id) else str(row.signer_id))
            statuses.append(result.status)
            valid_frames.append(result.valid_frames)
    finally:
        extractor.close()

    ensure_parent(args.out)
    np.savez_compressed(
        args.out,
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        paths=np.asarray(paths),
        signers=np.asarray(signers),
        statuses=np.asarray(statuses),
        valid_frames=np.asarray(valid_frames, dtype=np.int32),
        labels=np.asarray(labels),
        feature_dim=np.asarray([config.feature_dim], dtype=np.int32),
        sequence_length=np.asarray([config.sequence_length], dtype=np.int32),
    )
    print(f"Wrote {len(features)} samples to {args.out}")
    print(pd.Series(statuses).value_counts().to_string())


if __name__ == "__main__":
    main()
