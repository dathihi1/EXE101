from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils import ensure_parent


ABLATION_SLICES = {
    "hands": ("left_hand", "right_hand", "quality"),
    "hands_pose": ("left_hand", "right_hand", "pose", "quality"),
    "hands_pose_face": ("left_hand", "right_hand", "pose", "face", "quality"),
    "full": ("left_hand", "right_hand", "pose", "face", "motion", "geometry", "quality"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create V2 feature ablation NPZ files.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=["hands", "hands_pose", "hands_pose_face", "full"])
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["X"].astype(np.float32)
    try:
        metadata = data["schema_metadata"][0].item()
    except AttributeError:
        metadata = data["schema_metadata"][0]
    slices = metadata["slices"]

    for variant in args.variants:
        if variant not in ABLATION_SLICES:
            raise ValueError(f"Unknown ablation variant: {variant}")
        cols = []
        for name in ABLATION_SLICES[variant]:
            start, stop = slices[name]
            cols.extend(range(start, stop))
        cols_np = np.asarray(cols, dtype=np.int64)
        out_path = args.out_dir / f"{Path(args.features).stem}_{variant}.npz"
        ensure_parent(out_path)
        payload = {name: data[name] for name in data.files if name not in {"X", "feature_dim", "schema_metadata"}}
        ablation_metadata = dict(metadata)
        ablation_metadata["ablation_variant"] = variant
        ablation_metadata["selected_groups"] = list(ABLATION_SLICES[variant])
        ablation_metadata["source_feature_dim"] = int(X.shape[-1])
        ablation_metadata["feature_dim"] = int(len(cols_np))
        np.savez_compressed(
            out_path,
            X=X[:, :, cols_np],
            **payload,
            feature_dim=np.asarray([len(cols_np)], dtype=np.int32),
            schema_metadata=np.asarray([ablation_metadata], dtype=object),
        )
        print(f"{variant}: {X.shape[-1]} -> {len(cols_np)} dims -> {out_path}")


if __name__ == "__main__":
    main()
