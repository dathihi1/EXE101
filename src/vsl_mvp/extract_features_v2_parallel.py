from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FeatureConfigV2
from .landmark_schema import schema_metadata
from .landmarks_v2 import ExtractResultV2, HolisticLandmarkExtractor
from .utils import ensure_parent, write_json

# Global extractor instance per worker process
_WORKER_EXTRACTOR: HolisticLandmarkExtractor | None = None


def _init_worker(config: FeatureConfigV2) -> None:
    global _WORKER_EXTRACTOR
    _WORKER_EXTRACTOR = HolisticLandmarkExtractor(config)


def _process_video_task(args: tuple[str, str, str, int]) -> dict[str, Any]:
    global _WORKER_EXTRACTOR
    video_path, video_id, gloss, signer_id = args
    assert _WORKER_EXTRACTOR is not None
    try:
        res = _WORKER_EXTRACTOR.extract_video(video_path)
        return {
            "video_id": video_id,
            "video_path": video_path,
            "gloss": gloss,
            "signer_id": signer_id,
            "features": res.features,
            "status": res.status,
            "valid_frames": res.valid_frames,
            "quality": res.quality,
            "error": None,
        }
    except Exception as e:
        empty = _WORKER_EXTRACTOR._empty("extraction_error")
        return {
            "video_id": video_id,
            "video_path": video_path,
            "gloss": gloss,
            "signer_id": signer_id,
            "features": empty.features,
            "status": "error",
            "valid_frames": 0,
            "quality": {},
            "error": str(e),
        }


def save_shard(shard_path: Path, items: list[dict[str, Any]], config: FeatureConfigV2, labels: list[str]) -> None:
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    features = [it["features"] for it in items]
    y = [label_to_id[it["gloss"]] for it in items]
    paths = [it["video_path"] for it in items]
    video_ids = [it["video_id"] for it in items]
    signers = [it["signer_id"] for it in items]
    statuses = [it["status"] for it in items]
    valid_frames = [it["valid_frames"] for it in items]
    quality_rows = [it["quality"] for it in items]

    quality_keys = sorted({k for row in quality_rows for k in row})
    quality_matrix = np.asarray([[row.get(k, 0.0) for k in quality_keys] for row in quality_rows], dtype=np.float32)

    ensure_parent(shard_path)
    np.savez_compressed(
        shard_path,
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        video_ids=np.asarray(video_ids),
        paths=np.asarray(paths),
        signers=np.asarray(signers),
        statuses=np.asarray(statuses),
        valid_frames=np.asarray(valid_frames, dtype=np.int32),
        labels=np.asarray(labels),
        feature_dim=np.asarray([config.feature_dim], dtype=np.int32),
        sequence_length=np.asarray([config.sequence_length], dtype=np.int32),
        schema_version=np.asarray([config.schema_version]),
        quality_keys=np.asarray(quality_keys),
        quality=quality_matrix,
    )


def merge_all_shards(shard_files: list[Path], out_path: Path, config: FeatureConfigV2, labels: list[str], report_path: Path | None = None) -> None:
    print(f"\nMerging {len(shard_files)} shards into {out_path}...", flush=True)
    all_X = []
    all_y = []
    all_paths = []
    all_vids = []
    all_signers = []
    all_statuses = []
    all_valid_frames = []
    all_quality = []
    quality_keys = None

    for sf in sorted(shard_files):
        data = np.load(sf, allow_pickle=True)
        all_X.append(data["X"])
        all_y.append(data["y"])
        all_paths.append(data["paths"])
        all_vids.append(data["video_ids"])
        all_signers.append(data["signers"])
        all_statuses.append(data["statuses"])
        all_valid_frames.append(data["valid_frames"])
        all_quality.append(data["quality"])
        if quality_keys is None and "quality_keys" in data:
            quality_keys = data["quality_keys"]

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    paths = np.concatenate(all_paths, axis=0)
    video_ids = np.concatenate(all_vids, axis=0)
    signers = np.concatenate(all_signers, axis=0)
    statuses = np.concatenate(all_statuses, axis=0)
    valid_frames = np.concatenate(all_valid_frames, axis=0)
    quality = np.concatenate(all_quality, axis=0) if all_quality else np.empty((len(X), 0), dtype=np.float32)

    metadata = schema_metadata(config)
    ensure_parent(out_path)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        video_ids=video_ids,
        paths=paths,
        signers=signers,
        statuses=statuses,
        valid_frames=valid_frames,
        labels=np.asarray(labels),
        feature_dim=np.asarray([config.feature_dim], dtype=np.int32),
        sequence_length=np.asarray([config.sequence_length], dtype=np.int32),
        schema_version=np.asarray([config.schema_version]),
        quality_keys=np.asarray(quality_keys if quality_keys is not None else []),
        quality=quality,
        schema_metadata=np.asarray([metadata], dtype=object),
    )

    status_counts = pd.Series(statuses).value_counts()
    print(f"Merge completed! Final dataset shape: X={X.shape}, y={y.shape}", flush=True)
    print("Status counts:\n" + status_counts.to_string(), flush=True)

    if report_path:
        q_means = {}
        if quality_keys is not None and len(quality):
            for idx, k in enumerate(quality_keys):
                q_means[str(k)] = float(quality[:, idx].mean())
        report = {
            "out": str(out_path),
            "samples": int(len(X)),
            "classes": int(len(labels)),
            "feature_dim": int(config.feature_dim),
            "sequence_length": int(config.sequence_length),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "quality_means": q_means,
        }
        write_json(report_path, report)
        print(f"Report written to {report_path}", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Parallel MediaPipe Holistic V2 feature extraction.")
    parser.add_argument("--json-metadata", default=Path("data/raw/VSL400/front_view.json"), type=Path)
    parser.add_argument("--video-dir", default=Path("data/raw/VSL400/front_view"), type=Path)
    parser.add_argument("--shard-dir", default=Path("data/processed/shards_v2"), type=Path)
    parser.add_argument("--out", default=Path("data/processed/features_vsl400_v2_sf8.npz"), type=Path)
    parser.add_argument("--report", default=Path("data/processed/features_vsl400_v2_sf8_report.json"), type=Path)
    parser.add_argument("--workers", default=8, type=int, help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--chunk-size", default=500, type=int, help="Number of videos per shard file (default: 500)")
    parser.add_argument("--limit", default=0, type=int, help="Limit total videos for testing (0 = no limit)")
    args = parser.parse_args()

    # 1. Load metadata
    print(f"Loading metadata from {args.json_metadata}...", flush=True)
    with open(args.json_metadata, "r", encoding="utf-8") as f:
        meta_items = json.load(f)

    labels = sorted(list(set(x["gloss"] for x in meta_items)))
    print(f"Found {len(meta_items)} entries across {len(labels)} classes.", flush=True)

    # 2. Filter existing videos
    tasks = []
    for it in meta_items:
        vid = it["video_id"]
        vpath = args.video_dir / f"{vid}.mp4"
        if vpath.exists():
            tasks.append((str(vpath), vid, it["gloss"], str(it.get("signer_id", ""))))

    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"Testing mode enabled: limited to {len(tasks)} videos.", flush=True)
    else:
        print(f"Found {len(tasks)} videos ready on disk to process.", flush=True)

    args.shard_dir.mkdir(parents=True, exist_ok=True)

    # 3. Check existing shards to allow resuming
    existing_shard_files = sorted(args.shard_dir.glob("shard_*.npz"))
    already_processed_vids = set()
    for sf in existing_shard_files:
        try:
            d = np.load(sf, allow_pickle=True)
            if "video_ids" in d:
                already_processed_vids.update(d["video_ids"].tolist())
        except Exception:
            pass

    if already_processed_vids:
        print(f"Found {len(already_processed_vids)} videos already processed in existing shards. Resuming...", flush=True)
        tasks = [t for t in tasks if t[1] not in already_processed_vids]
        print(f"Remaining videos to process: {len(tasks)}", flush=True)

    config = FeatureConfigV2()

    if tasks:
        print(f"\nStarting parallel extraction with {args.workers} workers...", flush=True)
        start_time = time.time()
        processed_count = 0
        total_tasks = len(tasks)

        # Process in chunks and write shards
        num_chunks = (total_tasks + args.chunk_size - 1) // args.chunk_size
        shard_counter = len(existing_shard_files)

        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(config,)) as executor:
            for chunk_idx in range(num_chunks):
                chunk_tasks = tasks[chunk_idx * args.chunk_size : (chunk_idx + 1) * args.chunk_size]
                chunk_start = time.time()
                chunk_results = []

                for res in executor.map(_process_video_task, chunk_tasks):
                    chunk_results.append(res)
                    processed_count += 1
                    if processed_count % 50 == 0 or processed_count == total_tasks:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta_sec = (total_tasks - processed_count) / rate if rate > 0 else 0
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
                        print(
                            f"[{processed_count}/{total_tasks}] ({processed_count/total_tasks*100:.1f}%) | "
                            f"Rate: {rate:.2f} vid/s | ETA: {eta_str} | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
                            flush=True,
                        )

                # Save shard
                shard_path = args.shard_dir / f"shard_{shard_counter:04d}.npz"
                save_shard(shard_path, chunk_results, config, labels)
                shard_counter += 1
                chunk_elapsed = time.time() - chunk_start
                print(f"Saved checkpoint shard {shard_path.name} ({len(chunk_results)} videos in {chunk_elapsed:.1f}s)", flush=True)

        total_elapsed = time.time() - start_time
        print(f"\nAll tasks finished in {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}!", flush=True)

    # 4. Merge all shards
    all_shards = sorted(args.shard_dir.glob("shard_*.npz"))
    if all_shards:
        merge_all_shards(all_shards, args.out, config, labels, report_path=args.report)
    else:
        print("No shards found to merge.", flush=True)


if __name__ == "__main__":
    main()
