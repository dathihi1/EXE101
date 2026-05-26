from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import FeatureConfigV2
from .landmark_schema import feature_slices
from .landmarks import resample_sequence


@dataclass
class ExtractResultV2:
    features: np.ndarray
    valid_frames: int
    status: str
    quality: dict[str, float]


class HolisticLandmarkExtractor:
    def __init__(self, config: FeatureConfigV2 | None = None):
        self.config = config or FeatureConfigV2()
        import mediapipe as mp

        self.mp = mp
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def close(self) -> None:
        self.holistic.close()

    def extract_video(self, video_path: str | Path, sample_frames: int = 0) -> ExtractResultV2:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self._empty("cannot_open")

        rows: list[np.ndarray] = []
        if sample_frames > 0:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > sample_frames:
                frame_indices = np.linspace(0, total_frames - 1, num=sample_frames, dtype=np.int32)
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if ok:
                        rows.append(self.extract_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                rows.extend(self._read_all_frames(cap))
        else:
            rows.extend(self._read_all_frames(cap))
        cap.release()
        return self._finalize(rows)

    def extract_frames(self, bgr_frames: list[np.ndarray]) -> ExtractResultV2:
        rows = [self.extract_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in bgr_frames]
        return self._finalize(rows, empty_status="empty_recording")

    def _read_all_frames(self, cap: cv2.VideoCapture) -> list[np.ndarray]:
        rows = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rows.append(self.extract_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return rows

    def extract_frame(self, rgb: np.ndarray) -> np.ndarray:
        result = self.holistic.process(rgb)
        cfg = self.config

        left = np.zeros((21, 3), dtype=np.float32)
        right = np.zeros((21, 3), dtype=np.float32)
        pose = np.zeros((len(cfg.pose_landmark_indices), 4), dtype=np.float32)
        face = np.zeros((len(cfg.face_landmark_indices), 3), dtype=np.float32)

        if result.left_hand_landmarks:
            left = np.asarray([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark], dtype=np.float32)
        if result.right_hand_landmarks:
            right = np.asarray([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark], dtype=np.float32)
        if result.pose_landmarks:
            for out_idx, lm_idx in enumerate(cfg.pose_landmark_indices):
                lm = result.pose_landmarks.landmark[lm_idx]
                pose[out_idx] = (lm.x, lm.y, lm.z, lm.visibility)
        if result.face_landmarks:
            for out_idx, lm_idx in enumerate(cfg.face_landmark_indices):
                lm = result.face_landmarks.landmark[lm_idx]
                face[out_idx] = (lm.x, lm.y, lm.z)

        quality = np.asarray(
            [
                float(result.left_hand_landmarks is not None),
                float(result.right_hand_landmarks is not None),
                float(result.pose_landmarks is not None),
                float(result.face_landmarks is not None),
                float(result.left_hand_landmarks is not None or result.right_hand_landmarks is not None),
                float(result.left_hand_landmarks is not None and result.right_hand_landmarks is not None),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([left.reshape(-1), right.reshape(-1), pose.reshape(-1), face.reshape(-1), quality])

    def _finalize(self, rows: list[np.ndarray], empty_status: str = "empty_video") -> ExtractResultV2:
        if not rows:
            return self._empty(empty_status)

        raw = np.asarray(rows, dtype=np.float32)
        base, quality = split_raw(raw, self.config)
        quality[:, 6] = 1.0 - quality[:, 4]
        quality[:, 7] = 1.0 - quality[:, 3]
        base = normalize_base_sequence(base, quality, self.config)
        base = interpolate_missing_groups(base, quality, self.config)

        valid_frames = int(np.sum(quality[:, 4] > 0.5))
        base, quality = trim_idle_frames(base, quality, self.config)
        base = resample_sequence(base, self.config.sequence_length)
        quality = resample_sequence(quality, self.config.sequence_length)
        quality[:, :6] = (quality[:, :6] >= 0.5).astype(np.float32)
        quality[:, 6:] = np.clip(quality[:, 6:], 0.0, 1.0)

        derived = build_derived_features(base, quality, self.config)
        features = np.concatenate([base, derived, quality], axis=1).astype(np.float32)
        quality_summary = summarize_quality(quality)

        if valid_frames < self.config.min_valid_frames:
            status = "too_few_valid_frames"
        elif quality_summary["hand_frame_ratio"] < self.config.min_hand_frame_ratio:
            status = "low_hand_ratio"
        else:
            status = "ok"
        return ExtractResultV2(features, valid_frames, status, quality_summary)

    def _empty(self, status: str) -> ExtractResultV2:
        features = np.zeros((self.config.sequence_length, self.config.feature_dim), dtype=np.float32)
        return ExtractResultV2(features, 0, status, summarize_quality(None))


def split_raw(raw: np.ndarray, config: FeatureConfigV2) -> tuple[np.ndarray, np.ndarray]:
    base = raw[:, : config.base_dims]
    quality = raw[:, config.base_dims : config.base_dims + config.quality_dims]
    return base.copy(), quality.copy()


def normalize_base_sequence(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    seq = base.copy()
    slices = feature_slices(config)
    left = seq[:, slices.left_hand].reshape(seq.shape[0], 21, 3)
    right = seq[:, slices.right_hand].reshape(seq.shape[0], 21, 3)
    pose = seq[:, slices.pose].reshape(seq.shape[0], len(config.pose_landmark_indices), 4)
    face = seq[:, slices.face].reshape(seq.shape[0], len(config.face_landmark_indices), 3)

    left_shoulder_idx = config.pose_landmark_indices.index(11) if 11 in config.pose_landmark_indices else -1
    right_shoulder_idx = config.pose_landmark_indices.index(12) if 12 in config.pose_landmark_indices else -1

    for idx in range(seq.shape[0]):
        center, scale = frame_anchor(left[idx], right[idx], pose[idx], face[idx], quality[idx], left_shoulder_idx, right_shoulder_idx)
        if center is None:
            continue
        scale = max(float(scale), 1e-3)
        if quality[idx, 0] > 0.5:
            left[idx, :, :2] = (left[idx, :, :2] - center) / scale
            left[idx, :, 2] = left[idx, :, 2] / scale
        if quality[idx, 1] > 0.5:
            right[idx, :, :2] = (right[idx, :, :2] - center) / scale
            right[idx, :, 2] = right[idx, :, 2] / scale
        if quality[idx, 2] > 0.5:
            pose_valid = pose[idx, :, 3] > 0.2
            pose[idx, pose_valid, :2] = (pose[idx, pose_valid, :2] - center) / scale
            pose[idx, pose_valid, 2] = pose[idx, pose_valid, 2] / scale
        if quality[idx, 3] > 0.5:
            face[idx, :, :2] = (face[idx, :, :2] - center) / scale
            face[idx, :, 2] = face[idx, :, 2] / scale

    seq[:, slices.left_hand] = left.reshape(seq.shape[0], -1)
    seq[:, slices.right_hand] = right.reshape(seq.shape[0], -1)
    seq[:, slices.pose] = pose.reshape(seq.shape[0], -1)
    seq[:, slices.face] = face.reshape(seq.shape[0], -1)
    return np.nan_to_num(seq, copy=False)


def frame_anchor(
    left: np.ndarray,
    right: np.ndarray,
    pose: np.ndarray,
    face: np.ndarray,
    quality: np.ndarray,
    left_shoulder_idx: int,
    right_shoulder_idx: int,
) -> tuple[np.ndarray | None, float]:
    if (
        quality[2] > 0.5
        and left_shoulder_idx >= 0
        and right_shoulder_idx >= 0
        and pose[left_shoulder_idx, 3] > 0.2
        and pose[right_shoulder_idx, 3] > 0.2
    ):
        center = (pose[left_shoulder_idx, :2] + pose[right_shoulder_idx, :2]) / 2.0
        scale = np.linalg.norm(pose[left_shoulder_idx, :2] - pose[right_shoulder_idx, :2])
        return center, scale

    points = []
    if quality[0] > 0.5:
        points.append(left[:, :2])
    if quality[1] > 0.5:
        points.append(right[:, :2])
    if quality[3] > 0.5:
        points.append(face[:, :2])
    if not points:
        return None, 1.0
    valid = np.concatenate(points, axis=0)
    center = valid.mean(axis=0)
    scale = np.linalg.norm(valid.max(axis=0) - valid.min(axis=0))
    return center, scale


def interpolate_missing_groups(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    seq = base.copy()
    slices = feature_slices(config)
    groups = (
        (slices.left_hand, quality[:, 0] > 0.5),
        (slices.right_hand, quality[:, 1] > 0.5),
        (slices.pose, quality[:, 2] > 0.5),
        (slices.face, quality[:, 3] > 0.5),
    )
    for group_slice, valid in groups:
        seq[:, group_slice] = interpolate_short_gaps(seq[:, group_slice], valid, config.max_interp_gap)
    return seq


def interpolate_short_gaps(values: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    out = values.copy()
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 2:
        return out
    for start, end in zip(valid_idx[:-1], valid_idx[1:]):
        gap = end - start - 1
        if gap <= 0 or gap > max_gap:
            continue
        for offset in range(1, gap + 1):
            alpha = offset / (gap + 1)
            out[start + offset] = (1.0 - alpha) * out[start] + alpha * out[end]
    return out


def trim_idle_frames(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> tuple[np.ndarray, np.ndarray]:
    if len(base) <= 4:
        return base, quality
    motion = hand_motion_per_frame(base, quality, config)
    if float(np.max(motion)) < config.trim_motion_threshold:
        return base, quality
    active = np.flatnonzero(motion >= config.trim_motion_threshold)
    if len(active) == 0:
        return base, quality
    start = max(0, int(active[0]) - config.trim_margin)
    end = min(len(base), int(active[-1]) + config.trim_margin + 1)
    if end - start < max(8, config.min_valid_frames):
        return base, quality
    return base[start:end], quality[start:end]


def hand_motion_per_frame(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    centers = hand_centers(base, quality, config)
    deltas = np.zeros(len(base), dtype=np.float32)
    for hand_idx in range(2):
        points = centers[:, hand_idx]
        valid = np.any(points != 0.0, axis=1)
        diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
        pair_valid = valid[1:] & valid[:-1]
        deltas[1:] += diff * pair_valid
    return deltas / 2.0


def build_derived_features(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    motion = build_motion_features(base, quality, config)
    geometry = build_geometry_features(base, quality, config)
    return np.concatenate([motion, geometry], axis=1).astype(np.float32)


def build_motion_features(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    centers = hand_centers(base, quality, config)
    wrists = hand_wrists(base, quality, config)
    geometry = build_geometry_features(base, quality, config)
    out = np.zeros((base.shape[0], config.motion_dims), dtype=np.float32)

    center_delta = np.zeros_like(centers)
    wrist_delta = np.zeros_like(wrists)
    center_delta[1:] = centers[1:] - centers[:-1]
    wrist_delta[1:] = wrists[1:] - wrists[:-1]
    geom_delta = np.zeros_like(geometry)
    geom_delta[1:] = geometry[1:] - geometry[:-1]

    out[:, 0:2] = center_delta[:, 0]
    out[:, 2] = np.linalg.norm(center_delta[:, 0], axis=1)
    out[:, 3:5] = center_delta[:, 1]
    out[:, 5] = np.linalg.norm(center_delta[:, 1], axis=1)
    out[:, 6:8] = wrist_delta[:, 0]
    out[:, 8] = np.linalg.norm(wrist_delta[:, 0], axis=1)
    out[:, 9:11] = wrist_delta[:, 1]
    out[:, 11] = np.linalg.norm(wrist_delta[:, 1], axis=1)
    out[:, 12] = geom_delta[:, 0]
    out[:, 13] = geom_delta[:, 1]
    out[:, 14] = geom_delta[:, 2]
    out[:, 15] = geom_delta[:, 3]
    out[:, 16] = geom_delta[:, 4]
    out[:, 17] = (out[:, 2] + out[:, 5]) / 2.0
    return np.nan_to_num(out, copy=False)


def build_geometry_features(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    slices = feature_slices(config)
    left = base[:, slices.left_hand].reshape(base.shape[0], 21, 3)
    right = base[:, slices.right_hand].reshape(base.shape[0], 21, 3)
    pose = base[:, slices.pose].reshape(base.shape[0], len(config.pose_landmark_indices), 4)
    face = base[:, slices.face].reshape(base.shape[0], len(config.face_landmark_indices), 3)

    left_center = masked_mean(left[:, :, :2], quality[:, 0] > 0.5)
    right_center = masked_mean(right[:, :, :2], quality[:, 1] > 0.5)
    chest = pose_anchor(pose, quality, config, 11, 12)
    head = pose_point(pose, quality, config, 0)
    mouth = face_anchor(face, quality, config, (61, 291, 13, 14, 78, 308))
    chin = face_anchor(face, quality, config, (199, 152, 175))
    left_shoulder = pose_point(pose, quality, config, 11)
    right_shoulder = pose_point(pose, quality, config, 12)

    out = np.zeros((base.shape[0], config.geometry_dims), dtype=np.float32)
    out[:, 0] = pair_distance(left_center, right_center)
    out[:, 1] = pair_distance(left_center, chest)
    out[:, 2] = pair_distance(right_center, chest)
    out[:, 3] = pair_distance(left_center, mouth)
    out[:, 4] = pair_distance(right_center, mouth)
    out[:, 5] = pair_distance(left_center, chin)
    out[:, 6] = pair_distance(right_center, chin)
    out[:, 7] = pair_distance(left_center, head)
    out[:, 8] = pair_distance(right_center, head)
    out[:, 9] = pair_distance(left_center, left_shoulder)
    out[:, 10] = pair_distance(right_center, right_shoulder)
    out[:, 11] = hand_spread(left, quality[:, 0] > 0.5)
    out[:, 12] = hand_spread(right, quality[:, 1] > 0.5)
    out[:, 13] = palm_proxy(left, quality[:, 0] > 0.5)
    out[:, 14] = palm_proxy(right, quality[:, 1] > 0.5)
    out[:, 15] = torso_angle(pose, quality, config)
    return np.nan_to_num(out, copy=False)


def hand_centers(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    slices = feature_slices(config)
    left = base[:, slices.left_hand].reshape(base.shape[0], 21, 3)[:, :, :2]
    right = base[:, slices.right_hand].reshape(base.shape[0], 21, 3)[:, :, :2]
    return np.stack((masked_mean(left, quality[:, 0] > 0.5), masked_mean(right, quality[:, 1] > 0.5)), axis=1)


def hand_wrists(base: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    slices = feature_slices(config)
    left = base[:, slices.left_hand].reshape(base.shape[0], 21, 3)[:, 0, :2]
    right = base[:, slices.right_hand].reshape(base.shape[0], 21, 3)[:, 0, :2]
    left[quality[:, 0] <= 0.5] = 0.0
    right[quality[:, 1] <= 0.5] = 0.0
    return np.stack((left, right), axis=1)


def masked_mean(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((points.shape[0], 2), dtype=np.float32)
    out[valid] = points[valid].mean(axis=1)
    return out


def pair_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    valid = np.any(a != 0.0, axis=1) & np.any(b != 0.0, axis=1)
    out = np.zeros(a.shape[0], dtype=np.float32)
    out[valid] = np.linalg.norm(a[valid] - b[valid], axis=1)
    return out


def pose_anchor(pose: np.ndarray, quality: np.ndarray, config: FeatureConfigV2, left_idx: int, right_idx: int) -> np.ndarray:
    left = pose_point(pose, quality, config, left_idx)
    right = pose_point(pose, quality, config, right_idx)
    valid = np.any(left != 0.0, axis=1) & np.any(right != 0.0, axis=1)
    out = np.zeros_like(left)
    out[valid] = (left[valid] + right[valid]) / 2.0
    return out


def pose_point(pose: np.ndarray, quality: np.ndarray, config: FeatureConfigV2, landmark_idx: int) -> np.ndarray:
    out = np.zeros((pose.shape[0], 2), dtype=np.float32)
    if landmark_idx not in config.pose_landmark_indices:
        return out
    idx = config.pose_landmark_indices.index(landmark_idx)
    valid = (quality[:, 2] > 0.5) & (pose[:, idx, 3] > 0.2)
    out[valid] = pose[valid, idx, :2]
    return out


def face_anchor(face: np.ndarray, quality: np.ndarray, config: FeatureConfigV2, landmark_indices: tuple[int, ...]) -> np.ndarray:
    local_indices = [config.face_landmark_indices.index(idx) for idx in landmark_indices if idx in config.face_landmark_indices]
    out = np.zeros((face.shape[0], 2), dtype=np.float32)
    if not local_indices:
        return out
    valid = quality[:, 3] > 0.5
    out[valid] = face[valid][:, local_indices, :2].mean(axis=1)
    return out


def hand_spread(hand: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(hand.shape[0], dtype=np.float32)
    fingertips = hand[:, [4, 8, 12, 16, 20], :2]
    out[valid] = np.linalg.norm(fingertips[valid].max(axis=1) - fingertips[valid].min(axis=1), axis=1)
    return out


def palm_proxy(hand: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(hand.shape[0], dtype=np.float32)
    wrist = hand[:, 0, :2]
    middle = hand[:, 9, :2]
    vec = middle - wrist
    out[valid] = np.arctan2(vec[valid, 1], vec[valid, 0]) / np.pi
    return out


def torso_angle(pose: np.ndarray, quality: np.ndarray, config: FeatureConfigV2) -> np.ndarray:
    left_shoulder = pose_point(pose, quality, config, 11)
    right_shoulder = pose_point(pose, quality, config, 12)
    vec = right_shoulder - left_shoulder
    valid = np.any(left_shoulder != 0.0, axis=1) & np.any(right_shoulder != 0.0, axis=1)
    out = np.zeros(pose.shape[0], dtype=np.float32)
    out[valid] = np.arctan2(vec[valid, 1], vec[valid, 0]) / np.pi
    return out


def summarize_quality(quality: np.ndarray | None) -> dict[str, float]:
    if quality is None or len(quality) == 0:
        return {
            "left_hand_ratio": 0.0,
            "right_hand_ratio": 0.0,
            "hand_frame_ratio": 0.0,
            "both_hands_ratio": 0.0,
            "pose_ratio": 0.0,
            "face_ratio": 0.0,
            "mean_hand_missing_rate": 1.0,
            "mean_face_missing_rate": 1.0,
        }
    return {
        "left_hand_ratio": float(np.mean(quality[:, 0] > 0.5)),
        "right_hand_ratio": float(np.mean(quality[:, 1] > 0.5)),
        "hand_frame_ratio": float(np.mean(quality[:, 4] > 0.5)),
        "both_hands_ratio": float(np.mean(quality[:, 5] > 0.5)),
        "pose_ratio": float(np.mean(quality[:, 2] > 0.5)),
        "face_ratio": float(np.mean(quality[:, 3] > 0.5)),
        "mean_hand_missing_rate": float(np.mean(quality[:, 6])),
        "mean_face_missing_rate": float(np.mean(quality[:, 7])),
    }
