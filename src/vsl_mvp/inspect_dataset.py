from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from .utils import ensure_parent

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ID_RE = re.compile(r"(\d{6})")


def _clip_id(path_or_text: str) -> str | None:
    match = ID_RE.search(path_or_text)
    return match.group(1) if match else None


def _flatten_json_records(value, hint: str = ""):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_json_records(item, hint)
    elif isinstance(value, dict):
        if any(k.lower() in value for k in ("gloss", "label", "class", "word", "signer", "signer_id", "file", "video")):
            record = dict(value)
            if hint:
                record.setdefault("_hint_id", hint)
            yield record
        for key, item in value.items():
            if isinstance(item, (list, dict)):
                yield from _flatten_json_records(item, str(key))


def _get_first(record: dict, names: tuple[str, ...]) -> str | None:
    lowered = {str(k).lower(): v for k, v in record.items()}
    for name in names:
        if name in lowered and lowered[name] not in (None, ""):
            return str(lowered[name])
    return None


def load_metadata(dataset_root: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for json_path in dataset_root.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for record in _flatten_json_records(payload):
            text = " ".join(str(v) for v in record.values())
            clip_id = _clip_id(str(record.get("_hint_id", ""))) or _clip_id(text) or _clip_id(json_path.stem)
            if not clip_id:
                continue
            gloss = _get_first(record, ("gloss", "label", "class", "word", "sign", "name"))
            signer = _get_first(record, ("signer", "signer_id", "subject", "participant", "person_id"))
            metadata[clip_id] = {
                "clip_id": clip_id,
                "gloss": gloss,
                "signer_id": signer,
                "metadata_path": str(json_path),
            }
    return metadata


def build_manifest(dataset_root: Path, view: str) -> pd.DataFrame:
    search_root = dataset_root / view if (dataset_root / view).exists() else dataset_root
    metadata = load_metadata(dataset_root)
    rows = []
    for video_path in search_root.rglob("*"):
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        clip_id = _clip_id(video_path.stem)
        meta = metadata.get(clip_id or "", {})
        gloss = meta.get("gloss")
        if gloss is None:
            parent = video_path.parent.name
            gloss = parent if parent != view else None
        rows.append(
            {
                "clip_id": clip_id or video_path.stem,
                "video_path": str(video_path.resolve()),
                "gloss": gloss,
                "signer_id": meta.get("signer_id"),
                "view": view if view in str(video_path) else "",
                "metadata_path": meta.get("metadata_path"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["gloss"].notna()].sort_values(["gloss", "clip_id", "video_path"])
    return df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Inspect VSL400 and build a trainable manifest CSV.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--view", default="front_view")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    df = build_manifest(args.dataset_root, args.view)
    ensure_parent(args.out)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {args.out}")
    if not df.empty:
        print(df["gloss"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
