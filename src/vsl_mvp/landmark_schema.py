from __future__ import annotations

from dataclasses import dataclass

from .config import FeatureConfigV2


@dataclass(frozen=True)
class FeatureSlices:
    left_hand: slice
    right_hand: slice
    pose: slice
    face: slice
    motion: slice
    geometry: slice
    quality: slice


def feature_slices(config: FeatureConfigV2) -> FeatureSlices:
    start = 0
    left_hand = slice(start, start + 21 * 3)
    start = left_hand.stop
    right_hand = slice(start, start + 21 * 3)
    start = right_hand.stop
    pose = slice(start, start + config.pose_dims)
    start = pose.stop
    face = slice(start, start + config.face_dims)
    start = face.stop
    motion = slice(start, start + config.motion_dims)
    start = motion.stop
    geometry = slice(start, start + config.geometry_dims)
    start = geometry.stop
    quality = slice(start, start + config.quality_dims)
    return FeatureSlices(left_hand, right_hand, pose, face, motion, geometry, quality)


QUALITY_NAMES = (
    "left_hand_detected",
    "right_hand_detected",
    "pose_detected",
    "face_detected",
    "any_hand_detected",
    "both_hands_detected",
    "hand_missing_rate",
    "face_missing_rate",
)


MOTION_NAMES = (
    "left_hand_vx",
    "left_hand_vy",
    "left_hand_speed",
    "right_hand_vx",
    "right_hand_vy",
    "right_hand_speed",
    "left_wrist_vx",
    "left_wrist_vy",
    "left_wrist_speed",
    "right_wrist_vx",
    "right_wrist_vy",
    "right_wrist_speed",
    "hands_distance_delta",
    "left_to_chest_delta",
    "right_to_chest_delta",
    "left_to_mouth_delta",
    "right_to_mouth_delta",
    "mean_hand_speed",
)


GEOMETRY_NAMES = (
    "hand_center_distance",
    "left_to_chest",
    "right_to_chest",
    "left_to_mouth",
    "right_to_mouth",
    "left_to_chin",
    "right_to_chin",
    "left_to_head",
    "right_to_head",
    "left_to_left_shoulder",
    "right_to_right_shoulder",
    "left_hand_spread",
    "right_hand_spread",
    "left_palm_proxy",
    "right_palm_proxy",
    "torso_angle",
)


def schema_metadata(config: FeatureConfigV2) -> dict:
    slices = feature_slices(config)
    return {
        "schema_version": config.schema_version,
        "sequence_length": config.sequence_length,
        "feature_dim": config.feature_dim,
        "base_dims": config.base_dims,
        "pose_landmark_indices": list(config.pose_landmark_indices),
        "face_landmark_indices": list(config.face_landmark_indices),
        "slices": {
            name: [value.start, value.stop]
            for name, value in vars(slices).items()
        },
        "motion_names": list(MOTION_NAMES),
        "geometry_names": list(GEOMETRY_NAMES),
        "quality_names": list(QUALITY_NAMES),
    }
