from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import FeatureConfig


@dataclass
class ExtractResult:
    features: np.ndarray
    valid_frames: int
    status: str


class LandmarkExtractor:
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        import mediapipe as mp

        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def close(self) -> None:
        self.hands.close()
        self.pose.close()

    def extract_video(self, video_path: str | Path, sample_frames: int = 0) -> ExtractResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return ExtractResult(np.zeros((self.config.sequence_length, self.config.feature_dim), dtype=np.float32), 0, "cannot_open")

        frames: list[np.ndarray] = []
        valid_frames = 0
        if sample_frames > 0:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > sample_frames:
                frame_indices = np.linspace(0, total_frames - 1, num=sample_frames, dtype=np.int32)
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    features, has_signal = self.extract_frame(rgb)
                    frames.append(features)
                    valid_frames += int(has_signal)
            else:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    features, has_signal = self.extract_frame(rgb)
                    frames.append(features)
                    valid_frames += int(has_signal)
        else:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                features, has_signal = self.extract_frame(rgb)
                frames.append(features)
                valid_frames += int(has_signal)
        cap.release()

        if not frames:
            status = "empty_video"
            sequence = np.zeros((self.config.sequence_length, self.config.feature_dim), dtype=np.float32)
        else:
            sequence = resample_sequence(np.asarray(frames, dtype=np.float32), self.config.sequence_length)
            sequence = normalize_sequence(sequence)
            status = "ok" if valid_frames >= self.config.min_valid_frames else "too_few_valid_frames"
        return ExtractResult(sequence.astype(np.float32), valid_frames, status)

    def extract_frames(self, bgr_frames: list[np.ndarray]) -> ExtractResult:
        rows = []
        valid_frames = 0
        for frame in bgr_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, has_signal = self.extract_frame(rgb)
            rows.append(features)
            valid_frames += int(has_signal)
        if not rows:
            sequence = np.zeros((self.config.sequence_length, self.config.feature_dim), dtype=np.float32)
            return ExtractResult(sequence, 0, "empty_recording")
        sequence = normalize_sequence(resample_sequence(np.asarray(rows, dtype=np.float32), self.config.sequence_length))
        status = "ok" if valid_frames >= self.config.min_valid_frames else "too_few_valid_frames"
        return ExtractResult(sequence.astype(np.float32), valid_frames, status)

    def extract_frame(self, rgb: np.ndarray) -> tuple[np.ndarray, bool]:
        hands_result = self.hands.process(rgb)
        pose_result = self.pose.process(rgb)

        left = np.zeros((21, 3), dtype=np.float32)
        right = np.zeros((21, 3), dtype=np.float32)
        has_hand = False
        if hands_result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
                handedness = "Right"
                if hands_result.multi_handedness and idx < len(hands_result.multi_handedness):
                    handedness = hands_result.multi_handedness[idx].classification[0].label
                values = np.asarray([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                if handedness == "Left":
                    left = values
                else:
                    right = values
                has_hand = True

        pose_values = np.zeros((len(self.config.pose_landmark_indices), 4), dtype=np.float32)
        if pose_result.pose_landmarks:
            for out_idx, lm_idx in enumerate(self.config.pose_landmark_indices):
                lm = pose_result.pose_landmarks.landmark[lm_idx]
                pose_values[out_idx] = (lm.x, lm.y, lm.z, lm.visibility)

        row = np.concatenate([left.reshape(-1), right.reshape(-1), pose_values.reshape(-1)]).astype(np.float32)
        return row, has_hand


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    if len(sequence) == target_len:
        return sequence
    if len(sequence) == 1:
        return np.repeat(sequence, target_len, axis=0)
    old_idx = np.linspace(0.0, 1.0, num=len(sequence))
    new_idx = np.linspace(0.0, 1.0, num=target_len)
    out = np.empty((target_len, sequence.shape[1]), dtype=np.float32)
    for dim in range(sequence.shape[1]):
        out[:, dim] = np.interp(new_idx, old_idx, sequence[:, dim])
    return out


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    seq = sequence.copy()
    xyz = seq[:, : 21 * 3 * 2].reshape(seq.shape[0], 42, 3)
    pose = seq[:, 21 * 3 * 2 :].reshape(seq.shape[0], 6, 4)

    for idx in range(seq.shape[0]):
        points = xyz[idx]
        valid = np.any(points != 0, axis=1)
        if pose[idx, 0, 3] > 0.2 and pose[idx, 1, 3] > 0.2:
            center = (pose[idx, 0, :2] + pose[idx, 1, :2]) / 2.0
            scale = np.linalg.norm(pose[idx, 0, :2] - pose[idx, 1, :2])
        elif valid.any():
            center = points[valid, :2].mean(axis=0)
            mins = points[valid, :2].min(axis=0)
            maxs = points[valid, :2].max(axis=0)
            scale = float(np.linalg.norm(maxs - mins))
        else:
            continue
        scale = max(scale, 1e-3)
        points[valid, :2] = (points[valid, :2] - center) / scale
        points[valid, 2] = points[valid, 2] / scale
        pose_valid = pose[idx, :, 3] > 0.2
        pose[idx, pose_valid, :2] = (pose[idx, pose_valid, :2] - center) / scale
        pose[idx, pose_valid, 2] = pose[idx, pose_valid, 2] / scale

    seq[:, : 21 * 3 * 2] = xyz.reshape(seq.shape[0], -1)
    seq[:, 21 * 3 * 2 :] = pose.reshape(seq.shape[0], -1)
    return np.nan_to_num(seq, copy=False)
