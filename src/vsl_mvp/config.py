from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class FeatureConfig:
    sequence_length: int = 64
    hand_dims: int = 21 * 3 * 2
    pose_landmark_indices: tuple[int, ...] = (11, 12, 13, 14, 15, 16)
    pose_dims: int = 6 * 4
    feature_dim: int = hand_dims + pose_dims
    min_valid_frames: int = 8

    def to_json(self, path: str | Path) -> None:
        data = asdict(self)
        data["pose_landmark_indices"] = list(self.pose_landmark_indices)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "pose_landmark_indices" in data:
            data["pose_landmark_indices"] = tuple(data["pose_landmark_indices"])
        return cls(**data)


@dataclass(frozen=True)
class FeatureConfigV2:
    schema_version: str = "v2_holistic_subset"
    sequence_length: int = 64
    hand_dims: int = 21 * 3 * 2
    pose_landmark_indices: tuple[int, ...] = (0, 11, 12, 13, 14, 15, 16, 23, 24)
    pose_dims: int = 9 * 4
    face_landmark_indices: tuple[int, ...] = (
        10,
        33,
        61,
        78,
        81,
        82,
        87,
        88,
        91,
        95,
        133,
        146,
        159,
        160,
        161,
        173,
        178,
        181,
        185,
        191,
        199,
        263,
        291,
        308,
        311,
        312,
        317,
        318,
        321,
        324,
        362,
        374,
        386,
        387,
        388,
        398,
        402,
        405,
        409,
        415,
        454,
    )
    face_dims: int = 41 * 3
    motion_dims: int = 18
    geometry_dims: int = 16
    quality_dims: int = 8
    min_valid_frames: int = 8
    min_hand_frame_ratio: float = 0.35
    trim_motion_threshold: float = 0.015
    trim_margin: int = 4
    max_interp_gap: int = 5

    @property
    def base_dims(self) -> int:
        return self.hand_dims + self.pose_dims + self.face_dims

    @property
    def feature_dim(self) -> int:
        return self.base_dims + self.motion_dims + self.geometry_dims + self.quality_dims

    def to_json(self, path: str | Path) -> None:
        data = asdict(self)
        data["pose_landmark_indices"] = list(self.pose_landmark_indices)
        data["face_landmark_indices"] = list(self.face_landmark_indices)
        data["feature_dim"] = self.feature_dim
        data["base_dims"] = self.base_dims
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureConfigV2":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("feature_dim", None)
        data.pop("base_dims", None)
        if "pose_landmark_indices" in data:
            data["pose_landmark_indices"] = tuple(data["pose_landmark_indices"])
        if "face_landmark_indices" in data:
            data["face_landmark_indices"] = tuple(data["face_landmark_indices"])
        return cls(**data)
