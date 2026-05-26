from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import FeatureConfigV2
from .landmark_schema import schema_metadata
from .landmarks_v2 import HolisticLandmarkExtractor
from .utils import ensure_parent, read_json, write_json


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic V2 landmark features from VSL videos.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=Path, help="JSON with a classes list.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sequence-length", default=64, type=int)
    parser.add_argument("--min-valid-frames", default=8, type=int)
    parser.add_argument("--min-hand-frame-ratio", default=0.35, type=float)
    parser.add_argument("--trim-motion-threshold", default=0.015, type=float)
    parser.add_argument("--limit-per-class", default=0, type=int)
    parser.add_argument(
        "--sample-frames",
        default=0,
        type=int,
        help="If > 0, run MediaPipe on this many evenly spaced source frames per video.",
    )
    parser.add_argument("--report", default=None, type=Path, help="Optional JSON quality report path.")
    args = parser.parse_args()

    selected = set(read_json(args.classes)["classes"])
    df = pd.read_csv(args.manifest)
    df = df[df["gloss"].isin(selected)].copy()
    if args.limit_per_class > 0:
        df = df.groupby("gloss", group_keys=False).head(args.limit_per_class)
    df = df.reset_index(drop=True)

    labels = sorted(df["gloss"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    config = FeatureConfigV2(
        sequence_length=args.sequence_length,
        min_valid_frames=args.min_valid_frames,
        min_hand_frame_ratio=args.min_hand_frame_ratio,
        trim_motion_threshold=args.trim_motion_threshold,
    )
    extractor = HolisticLandmarkExtractor(config)

    features = []
    y = []
    paths = []
    signers = []
    statuses = []
    valid_frames = []
    quality_rows = []
    try:
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            result = extractor.extract_video(row.video_path, sample_frames=args.sample_frames)
            features.append(result.features)
            y.append(label_to_id[row.gloss])
            paths.append(row.video_path)
            signers.append("" if pd.isna(row.signer_id) else str(row.signer_id))
            statuses.append(result.status)
            valid_frames.append(result.valid_frames)
            quality_rows.append(result.quality)
    finally:
        extractor.close()

    ensure_parent(args.out)
    quality_keys = sorted({key for row in quality_rows for key in row})
    quality_matrix = np.asarray([[row.get(key, 0.0) for key in quality_keys] for row in quality_rows], dtype=np.float32)
    metadata = schema_metadata(config)
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
        schema_version=np.asarray([config.schema_version]),
        quality_keys=np.asarray(quality_keys),
        quality=np.asarray(quality_matrix, dtype=np.float32),
        schema_metadata=np.asarray([metadata], dtype=object),
    )

    status_counts = pd.Series(statuses).value_counts()
    report = {
        "out": str(args.out),
        "samples": int(len(features)),
        "classes": int(len(labels)),
        "labels": labels,
        "feature_dim": int(config.feature_dim),
        "sequence_length": int(config.sequence_length),
        "schema": metadata,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "quality_means": {
            key: float(quality_matrix[:, idx].mean()) if len(quality_matrix) else 0.0
            for idx, key in enumerate(quality_keys)
        },
    }
    if args.report:
        write_json(args.report, report)

    print(f"Wrote {len(features)} samples to {args.out}")
    print(f"Feature dim: {config.feature_dim}")
    print(status_counts.to_string())
    if quality_keys:
        print(pd.Series(report["quality_means"]).round(3).to_string())


if __name__ == "__main__":
    main()
