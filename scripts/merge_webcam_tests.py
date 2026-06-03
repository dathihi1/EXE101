from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_HIDDEN_TEST_DIR = ROOT / "runs/webcam_tests/hidden_data"
LEGACY_TEST_DIR = ROOT / "runs/webcam_tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vsl_mvp.config import FeatureConfigV2
from vsl_mvp.landmarks_v2 import HolisticLandmarkExtractor


def fix_mojibake(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def load_logs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def label_lookup(labels: list[str]) -> dict[str, str]:
    lookup = {}
    for label in labels:
        lookup[label] = label
        lookup[fix_mojibake(label)] = label
    return lookup


def selected_label_range(
    labels: list[str],
    from_label: str | None,
    to_label: str | None,
    from_index: int | None,
    to_index: int | None,
) -> set[str] | None:
    if from_index is not None or to_index is not None:
        start = 0 if from_index is None else int(from_index)
        end = len(labels) - 1 if to_index is None else int(to_index)
        if start > end:
            start, end = end, start
        return set(labels[start : end + 1])
    if not from_label and not to_label:
        return None
    lookup = label_lookup(labels)
    start_label = lookup.get(from_label or "") or lookup.get(fix_mojibake(from_label or ""))
    end_label = lookup.get(to_label or "") or lookup.get(fix_mojibake(to_label or ""))
    if start_label is None:
        raise ValueError(f"from-label not found in labels: {from_label}")
    if end_label is None:
        raise ValueError(f"to-label not found in labels: {to_label}")
    start = labels.index(start_label)
    end = labels.index(end_label)
    if start > end:
        start, end = end, start
    return set(labels[start : end + 1])


def iter_unique_tests(
    rows: list[dict],
    labels: list[str],
    include_failed: bool,
    allowed_labels: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    lookup = label_lookup(labels)
    seen = set()
    tests = []
    for row in rows:
        video = str(row.get("test_video") or "")
        target = str(row.get("target_label") or "")
        if not video or not target:
            continue
        if not include_failed and row.get("result_status") in {"too_few_valid_frames", "low_hand_ratio", "low_face_ratio", "low_motion"}:
            continue

        canonical = lookup.get(target) or lookup.get(fix_mojibake(target))
        if canonical is None:
            print(f"[skip] target label not in original labels: {target}")
            continue
        if allowed_labels is not None and canonical not in allowed_labels:
            continue

        video_path = ROOT / video if not Path(video).is_absolute() else Path(video)
        if not video_path.exists():
            print(f"[skip] missing test video: {video_path}")
            continue
        key = str(video_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        tests.append((video_path, canonical, str(row.get("result_status") or "")))
    return tests


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Merge webcam test videos into a VSL feature dataset.")
    parser.add_argument("--base-features", default=ROOT / "data/processed/features_vsl_mvp30_v2_sf8.npz", type=Path)
    parser.add_argument("--log", default=DEFAULT_HIDDEN_TEST_DIR / "test_log.jsonl", type=Path)
    parser.add_argument("--out", default=ROOT / "data/processed/features_vsl_mvp30_v2_sf8_webcam_tests.npz", type=Path)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--legacy-log-fallback", action="store_true")
    parser.add_argument("--from-label", default=None)
    parser.add_argument("--to-label", default=None)
    parser.add_argument("--from-index", default=None, type=int)
    parser.add_argument("--to-index", default=None, type=int)
    parser.add_argument("--sample-frames", default=0, type=int)
    args = parser.parse_args()
    if args.legacy_log_fallback and not args.log.exists():
        legacy_log = LEGACY_TEST_DIR / "test_log.jsonl"
        if legacy_log.exists():
            print(f"Using legacy webcam test log: {legacy_log}")
            args.log = legacy_log

    base = np.load(args.base_features, allow_pickle=True)
    labels = [str(x) for x in base["labels"].tolist()]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    rows = load_logs(args.log)
    allowed_labels = selected_label_range(labels, args.from_label, args.to_label, args.from_index, args.to_index)
    tests = iter_unique_tests(rows, labels, include_failed=args.include_failed, allowed_labels=allowed_labels)

    if not tests:
        print("No webcam test videos to merge.")
        return

    sequence_length = int(base["sequence_length"][0]) if "sequence_length" in base.files else 64
    config = FeatureConfigV2(sequence_length=sequence_length)
    extractor = HolisticLandmarkExtractor(config)

    new_X = []
    new_y = []
    new_paths = []
    new_statuses = []
    new_valid_frames = []
    new_quality_rows = []
    try:
        for idx, (video_path, label, logged_status) in enumerate(tests, 1):
            print(f"[{idx}/{len(tests)}] {label}: {video_path.name}")
            result = extractor.extract_video(video_path, sample_frames=args.sample_frames)
            if result.status != "ok":
                print(f"  [skip] extractor status={result.status} logged_status={logged_status}")
                continue
            new_X.append(result.features)
            new_y.append(label_to_id[label])
            new_paths.append(str(video_path.relative_to(ROOT) if video_path.is_relative_to(ROOT) else video_path))
            new_statuses.append("ok")
            new_valid_frames.append(result.valid_frames)
            new_quality_rows.append(result.quality)
    finally:
        extractor.close()

    if not new_X:
        print("No extractor-ok webcam tests were merged.")
        return

    quality_keys = base["quality_keys"].tolist() if "quality_keys" in base.files else []
    if not quality_keys:
        quality_keys = sorted({key for row in new_quality_rows for key in row})
    new_quality = np.asarray([[row.get(key, 0.0) for key in quality_keys] for row in new_quality_rows], dtype=np.float32)

    X = np.concatenate([base["X"].astype(np.float32), np.asarray(new_X, dtype=np.float32)], axis=0)
    y = np.concatenate([base["y"].astype(np.int64), np.asarray(new_y, dtype=np.int64)], axis=0)
    paths = np.concatenate([base["paths"].astype(str), np.asarray(new_paths)])
    signers = np.concatenate([base["signers"].astype(str), np.asarray(["webcam_test"] * len(new_X))])
    statuses = np.concatenate([base["statuses"].astype(str), np.asarray(new_statuses)])
    valid_frames = np.concatenate([base["valid_frames"].astype(np.int32), np.asarray(new_valid_frames, dtype=np.int32)])
    quality = np.concatenate([base["quality"].astype(np.float32), new_quality], axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        paths=paths,
        signers=signers,
        statuses=statuses,
        valid_frames=valid_frames,
        labels=base["labels"],
        feature_dim=base["feature_dim"],
        sequence_length=base["sequence_length"],
        schema_version=base["schema_version"],
        quality_keys=np.asarray(quality_keys),
        quality=quality,
        schema_metadata=base["schema_metadata"],
    )

    print(f"Merged {len(new_X)} webcam test samples.")
    print(f"Output: {args.out}")
    print("Retrain command:")
    print(
        "python -m vsl_mvp.train "
        f"--features {args.out} "
        "--model lite_transformer "
        "--out-dir runs/vsl_mvp30_v2_lite_transformer_webcam_tests "
        "--epochs 35"
    )


if __name__ == "__main__":
    main()
