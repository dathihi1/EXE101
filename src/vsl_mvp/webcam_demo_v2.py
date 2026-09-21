from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import platform
import time
import unicodedata
import sys
import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FeatureConfigV2
from .infer import OnnxSignRecognizer
from .landmarks import LandmarkExtractor
from .landmarks_v2 import HolisticLandmarkExtractor

# Prevent UnicodeEncodingError on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

HAND_FEATURE_DIMS = 21 * 3 * 2
REFERENCE_THUMB_WIDTH = 180
SAMPLE_THUMB_WIDTH = 240
DEFAULT_PRACTICE_VIDEO_DIR = Path("data/practice_videos")
DEFAULT_TEST_OUTPUT_DIR = Path("runs/webcam_tests/hidden_data")


@dataclass
class HandBox:
    label: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    size: tuple[float, float]
    landmarks: np.ndarray


@dataclass
class ReferenceSample:
    frame: np.ndarray
    hand_boxes: list[HandBox]


@dataclass
class SampleVideo:
    path: Path
    frames: list[np.ndarray]
    features: np.ndarray
    frame_index: int = 0


class LiveHandTracker:
    def __init__(self) -> None:
        import mediapipe as mp

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def close(self) -> None:
        self.hands.close()

    def detect(self, bgr_frame: np.ndarray) -> list[HandBox]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        boxes = []
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = f"Hand {idx + 1}"
            if result.multi_handedness and idx < len(result.multi_handedness):
                label = result.multi_handedness[idx].classification[0].label
            points = np.asarray([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            pad_x = max((x2 - x1) * 0.18, 0.025)
            pad_y = max((y2 - y1) * 0.18, 0.025)
            x1 = float(np.clip(x1 - pad_x, 0.0, 1.0))
            y1 = float(np.clip(y1 - pad_y, 0.0, 1.0))
            x2 = float(np.clip(x2 + pad_x, 0.0, 1.0))
            y2 = float(np.clip(y2 + pad_y, 0.0, 1.0))
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            size = (x2 - x1, y2 - y1)
            boxes.append(HandBox(label=label, bbox=(x1, y1, x2, y2), center=center, size=size, landmarks=points))
        return boxes


def load_font(size: int = 24):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_hud(
    frame: np.ndarray,
    target_label: str,
    status_msg: str,
    last_result: dict | None,
    phase: str,
    recording: bool,
    frames_count: int,
) -> None:
    # Convert BGR to RGB for PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    
    # Load fonts
    font_large = load_font(28)  # For Target and Result
    font_small = load_font(18)  # For instructions
    
    h, w, _ = frame.shape
    
    top_banner_h = 75
    bottom_banner_h = 105
    
    # Render Target Label
    target_text = f"MỤC TIÊU CẦN THỰC HÀNH: {target_label.upper()}" if target_label else "MỤC TIÊU: CHƯA CHỌN"
    draw.text((20, 20), target_text, font=font_large, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    
    # Determine Result Text and Color
    result_text = ""
    result_color = (255, 255, 255)  # White default
    
    if recording:
        result_text = f"TRẠNG THÁI: ĐANG GHI HÌNH... ({frames_count} frames)"
        result_color = (255, 60, 60)  # Red
    elif phase == "capture_reference":
        if "capturing" in status_msg.lower() or "giây" in status_msg.lower():
            result_text = f"TRẠNG THÁI: ĐANG CHỤP TƯ THẾ BẮT ĐẦU..."
            result_color = (255, 200, 0)  # Yellow
        else:
            result_text = "TRẠNG THÁI: Hãy để tay vào khung hình và nhấn 'C' để chụp tư thế bắt đầu"
            result_color = (200, 200, 200)  # Gray
    elif last_result:
        if last_result["status"] == "ok":
            if last_result.get("verified"):
                result_text = f"KẾT QUẢ: {last_result['label'].upper()} (CHÍNH XÁC! - {last_result['confidence']*100:.1f}%)"
                result_color = (30, 240, 80)  # Green
            else:
                result_text = f"KẾT QUẢ: {last_result['label'].upper()} ({last_result['confidence']*100:.1f}%)"
                result_color = (255, 200, 0)  # Yellow
        elif last_result.get("label"):
            result_text = f"KẾT QUẢ: {last_result['label'].upper()} (CHƯA ĐÚNG - {last_result['confidence']*100:.1f}%)"
            result_color = (255, 100, 100)  # Red-Orange
        else:
            result_text = "KẾT QUẢ: Không thể nhận diện chính xác. Hãy thực hiện lại!"
            result_color = (255, 100, 100)  # Red
    else:
        # Determine from status msg
        if "stable" in status_msg.lower() or "ổn định" in status_msg.lower() or "đứng im" in status_msg.lower():
            result_text = "Tư thế bắt đầu ổn định! Nhấn 'C' để chụp tư thế"
            result_color = (100, 200, 255)  # Light Blue
        else:
            result_text = "TRẠNG THÁI: Sẵn sàng. Nhấn [SPACE] để bắt đầu ghi hình"
            result_color = (255, 255, 255)
            
    # Draw Result Text
    draw.text((20, h - bottom_banner_h + 15), result_text, font=font_large, fill=result_color, stroke_width=2, stroke_fill=(0, 0, 0))
    
    # Draw Instructions at the very bottom
    instr_text = "Điều khiển: [SPACE] Ghi hình/Dừng | [C] Chụp tư thế | [R] Reset | [N] Từ tiếp | [P] Từ trước | [Q] Thoát"
    draw.text((20, h - 35), instr_text, font=font_small, fill=(240, 240, 240), stroke_width=1, stroke_fill=(0, 0, 0))
    
    # Convert PIL back to BGR
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

def draw_hand_overlay(frame: np.ndarray, hand_boxes: list[HandBox], reference_ok: bool | None) -> None:
    height, width = frame.shape[:2]
    color = (60, 220, 80)
    if reference_ok is False:
        color = (30, 120, 255)
    elif reference_ok is True:
        color = (255, 190, 40)

    for hand in hand_boxes:
        x1, y1, x2, y2 = hand.bbox
        px1, py1 = int(x1 * width), int(y1 * height)
        px2, py2 = int(x2 * width), int(y2 * height)
        cx, cy = int(hand.center[0] * width), int(hand.center[1] * height)
        points = np.asarray(hand.landmarks, dtype=np.float32)
        if points.shape == (21, 2):
            pixel_points = [(int(x * width), int(y * height)) for x, y in points]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, pixel_points[start], pixel_points[end], color, 2)
            for point_idx, point in enumerate(pixel_points):
                radius = 4 if point_idx in (0, 4, 8, 12, 16, 20) else 3
                cv2.circle(frame, point, radius, (255, 255, 255), -1)
                cv2.circle(frame, point, radius, color, 1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, hand.label, (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_reference_thumbnail(frame: np.ndarray, reference: ReferenceSample | None) -> None:
    if reference is None:
        return

    ref = reference.frame.copy()
    draw_hand_overlay(ref, reference.hand_boxes, True)
    ref_h, ref_w = ref.shape[:2]
    scale = REFERENCE_THUMB_WIDTH / max(ref_w, 1)
    thumb_w = REFERENCE_THUMB_WIDTH
    thumb_h = max(1, int(ref_h * scale))
    thumb = cv2.resize(ref, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    margin = 12
    x1 = max(0, frame.shape[1] - thumb_w - margin)
    y1 = margin
    x2 = x1 + thumb_w
    y2 = y1 + thumb_h
    frame[y1:y2, x1:x2] = thumb
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 190, 40), 2)
    cv2.putText(frame, "Reference", (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 190, 40), 2)


def draw_sample_video(frame: np.ndarray, sample_video: SampleVideo | None, mirror: bool = False) -> None:
    if sample_video is None or not sample_video.frames:
        return

    sample_frame = sample_video.frames[sample_video.frame_index % len(sample_video.frames)]
    sample_video.frame_index += 1
    if mirror:
        sample_frame = cv2.flip(sample_frame, 1)
    sample_h, sample_w = sample_frame.shape[:2]
    scale = SAMPLE_THUMB_WIDTH / max(sample_w, 1)
    thumb_w = SAMPLE_THUMB_WIDTH
    thumb_h = max(1, int(sample_h * scale))
    thumb = cv2.resize(sample_frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    margin = 12
    x1 = max(0, frame.shape[1] - thumb_w - margin)
    y1 = frame.shape[0] - thumb_h - 42
    x2 = x1 + thumb_w
    y2 = y1 + thumb_h
    if y1 < 0:
        return
    frame[y1:y2, x1:x2] = thumb
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 210, 255), 2)
    title = "Sample video (mirror)" if mirror else "Sample video"
    cv2.putText(frame, title, (x1, y2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 210, 255), 2)


def hand_frame_ratio(sequence: np.ndarray) -> float:
    # landmarks: (T, D)
    # hand features: first 21*3*2 = 126 dims
    T = len(sequence)
    detected = 0
    for frame in sequence:
        lh = frame[0:63]
        rh = frame[63:126]
        # if not all zeros
        if float(np.abs(lh).sum()) > 1e-5 or float(np.abs(rh).sum()) > 1e-5:
            detected += 1
    return detected / max(1, T)


def hand_motion_score(sequence: np.ndarray) -> float:
    T = len(sequence)
    if T < 2:
        return 0.0
    scores = []
    for t in range(1, T):
        lh_curr = sequence[t, 0:63]
        lh_prev = sequence[t - 1, 0:63]
        rh_curr = sequence[t, 63:126]
        rh_prev = sequence[t - 1, 63:126]
        d_lh = float(np.sqrt(np.sum((lh_curr - lh_prev) ** 2))) if float(np.abs(lh_curr).sum()) > 1e-5 and float(np.abs(lh_prev).sum()) > 1e-5 else 0.0
        d_rh = float(np.sqrt(np.sum((rh_curr - rh_prev) ** 2))) if float(np.abs(rh_curr).sum()) > 1e-5 and float(np.abs(rh_prev).sum()) > 1e-5 else 0.0
        scores.append(max(d_lh, d_rh))
    return float(np.mean(scores))


def union_hand_box(hand_boxes: list[HandBox]) -> HandBox | None:
    if not hand_boxes:
        return None
    x1 = min(box.bbox[0] for box in hand_boxes)
    y1 = min(box.bbox[1] for box in hand_boxes)
    x2 = max(box.bbox[2] for box in hand_boxes)
    y2 = max(box.bbox[3] for box in hand_boxes)
    return HandBox(
        label="Union",
        bbox=(x1, y1, x2, y2),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        size=(x2 - x1, y2 - y1),
        landmarks=np.zeros((0, 2), dtype=np.float32),
    )


def text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).replace("đ", "d").replace("Đ", "D")
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.replace("_", " ").casefold().split())


def sample_label_from_path(path: Path) -> str:
    return path.stem.split("__", 1)[0].replace("_", " ")


def find_sample_video_path(label: str, sample_dir: Path) -> Path | None:
    if not sample_dir.exists():
        return None

    wanted = text_key(label)
    # exact search
    for path in sample_dir.rglob("*.mp4"):
        if text_key(sample_label_from_path(path)) == wanted:
            return path
    # substring search
    for path in sample_dir.rglob("*.mp4"):
        candidate = text_key(sample_label_from_path(path))
        if wanted in candidate or candidate in wanted:
            return path
    return None


def read_sample_frames(path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    if not cap.isOpened():
        return []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def load_sample_video(label: str, sample_dir: Path, extractor, max_frames: int) -> tuple[SampleVideo | None, str]:
    path = find_sample_video_path(label, sample_dir)
    if path is None:
        return None, "sample_not_found"
    frames = read_sample_frames(path, max_frames)
    if not frames:
        return None, "sample_empty"
    extract_result = extractor.extract_frames(frames)
    if extract_result.status != "ok":
        return None, "sample_extract_failed"
    return SampleVideo(path=path, frames=frames, features=extract_result.features), "ok"


def label_index(labels: list[str], wanted_label: str | None) -> int:
    if not labels or not wanted_label:
        return 0
    wanted_key = text_key(wanted_label)
    for idx, label in enumerate(labels):
        if text_key(label) == wanted_key:
            return idx
    for idx, label in enumerate(labels):
        label_key = text_key(label)
        if wanted_key in label_key or label_key in wanted_key:
            return idx
    return 0


def load_practice_sample(
    labels: list[str],
    practice_index: int,
    sample_dir: Path,
    extractor,
    max_frames: int,
) -> tuple[str, SampleVideo | None, str]:
    if not labels:
        return "", None, "no_labels"
    label = labels[practice_index % len(labels)]
    sample_video, sample_status = load_sample_video(label, sample_dir, extractor, max_frames)
    return label, sample_video, sample_status


def sample_match_score(user_features: np.ndarray, sample_features: np.ndarray) -> float:
    # Dynamic Time Warping distance on normalized hand coordinates
    # simple MSE on padded/interpolated features for MVP
    # user_features: (U, D), sample_features: (S, D)
    # resample both to 64 frames
    from scipy.interpolate import interp1d

    U, D = user_features.shape
    S, _ = sample_features.shape

    u_hands = user_features[:, :126]
    s_hands = sample_features[:, :126]

    # helper to resample
    def resample(x: np.ndarray, target: int) -> np.ndarray:
        L = len(x)
        if L == target:
            return x
        grid_in = np.linspace(0, 1, L)
        grid_out = np.linspace(0, 1, target)
        f = interp1d(grid_in, x, axis=0, bounds_error=False, fill_value="extrapolate")
        return f(grid_out)

    u_norm = resample(u_hands, 64)
    s_norm = resample(s_hands, 64)

    # MSE of coordinate features
    return float(np.mean((u_norm - s_norm) ** 2))


def safe_slug(text: str, fallback: str = "unknown") -> str:
    cleaned = text_key(text).replace(" ", "_")
    return cleaned if cleaned else fallback


def save_test_video(frames: list[np.ndarray], output_dir: Path, label: str, phase: str, fps: float) -> Path | None:
    if not frames:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = safe_slug(label)
    filename = f"{slug}__{phase}__{stamp}.mp4"
    path = output_dir / filename
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not out.isOpened():
        return None
    try:
        for f in frames:
            out.write(f)
    finally:
        out.release()
    return path


def append_test_log(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_logs.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def log_attempt(
    output_dir: Path,
    phase: str,
    target_label: str,
    sample_video: SampleVideo | None,
    video_path: Path | None,
    result_status: str,
    prediction: dict | None = None,
    quality: dict | None = None,
    reference_score: float | None = None,
) -> Path:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "target_label": target_label,
        "sample_video": str(sample_video.path) if sample_video else None,
        "test_video": str(video_path) if video_path else None,
        "result_status": result_status,
        "prediction": prediction,
        "quality": quality,
        "reference_score": reference_score,
    }
    return append_test_log(output_dir, payload)


def stable_reference_ready(history: deque[HandBox], tolerance: float) -> bool:
    if len(history) < history.maxlen:
        return False
    # check standard deviation of hand center positions
    centers = np.array([box.center for box in history])
    stds = centers.std(axis=0)
    # also check size consistency
    sizes = np.array([box.size for box in history])
    size_stds = sizes.std(axis=0)
    return float(stds.max()) <= tolerance and float(size_stds.max()) <= tolerance * 0.75


def reference_alignment(reference: ReferenceSample | None, hand_boxes: list[HandBox], tolerance: float) -> tuple[bool, float]:
    if reference is None:
        return True, 0.0
    u_curr = union_hand_box(hand_boxes)
    u_ref = union_hand_box(reference.hand_boxes)
    if u_curr is None or u_ref is None:
        return False, float("inf")
    # distance between centers
    dx = u_curr.center[0] - u_ref.center[0]
    dy = u_curr.center[1] - u_ref.center[1]
    dist = float(np.sqrt(dx*dx + dy*dy))
    return dist <= tolerance, dist


def best_reference_alignment(references: list[ReferenceSample], hand_boxes: list[HandBox], tolerance: float) -> tuple[bool, float]:
    if not references:
        return True, 0.0
    best_dist = float("inf")
    any_ok = False
    for ref in references:
        ok, dist = reference_alignment(ref, hand_boxes, tolerance)
        if ok:
            any_ok = True
        best_dist = min(best_dist, dist)
    return any_ok, best_dist


def make_reference(frame: np.ndarray, hand_boxes: list[HandBox]) -> ReferenceSample | None:
    if not hand_boxes:
        return None
    return ReferenceSample(frame=frame.copy(), hand_boxes=list(hand_boxes))


def build_extractor(recognizer: OnnxSignRecognizer):
    schema_version = recognizer.config.get("schema_version", "v2_holistic_subset")
    seq_len = int(recognizer.config.get("sequence_length", 64))
    if schema_version == "v2_holistic_subset":
        from .config import FeatureConfigV2
        config = FeatureConfigV2(sequence_length=seq_len)
        extractor = HolisticLandmarkExtractor(config)
        return extractor, "v2_holistic_subset", "HolisticLandmarkExtractor(v2)"
    else:
        from .config import FeatureConfig
        config = FeatureConfig(sequence_length=seq_len)
        extractor = LandmarkExtractor(config)
        return extractor, "v1_hands_pose", "LandmarkExtractor(v1)"


def sequence_quality(result, schema_version: str, args: argparse.Namespace) -> tuple[dict[str, float], str]:
    if schema_version != "v2_holistic_subset":
        ratio = hand_frame_ratio(result.features)
        motion = hand_motion_score(result.features)
        metrics = {"hand_ratio": ratio, "face_ratio": 1.0, "motion": motion}
        if ratio < args.min_hand_frame_ratio:
            return metrics, "low_hand_ratio"
        if motion < args.min_hand_motion:
            return metrics, "low_hand_motion"
        return metrics, "ok"

    # For V2, we have precise features quality metrics
    q_features = result.features[:, 319:327] # quality slice
    # quality fields:
    # 0: left_hand_detected, 1: right_hand_detected, 2: pose_detected, 3: face_detected,
    # 4: any_hand_detected, 5: both_hands_detected, 6: hand_missing_rate, 7: face_missing_rate
    any_hand_detected = q_features[:, 4]
    face_detected = q_features[:, 3]
    
    hand_ratio = float(any_hand_detected.mean())
    face_ratio = float(face_detected.mean())
    
    # Estimate hand motion using the motion slice [285:303]
    motion_features = result.features[:, 285:303]
    # speed indices: left_hand_speed=2, right_hand_speed=5
    speed = np.maximum(motion_features[:, 2], motion_features[:, 5])
    motion = float(speed.mean())

    metrics = {"hand_ratio": hand_ratio, "face_ratio": face_ratio, "motion": motion}
    if hand_ratio < args.min_hand_frame_ratio:
        return metrics, "low_hand_ratio"
    if face_ratio < args.min_face_frame_ratio:
        return metrics, "low_face_ratio"
    if motion < args.min_hand_motion:
        return metrics, "low_hand_motion"
    return metrics, "ok"


def low_quality_result(status: str, quality: dict[str, float] | None = None) -> dict:
    return {
        "label": "",
        "confidence": 0.0,
        "confidence_margin": 0.0,
        "top3": [],
        "status": status,
        "quality": quality or {"hand_ratio": 0.0, "face_ratio": 0.0, "motion": 0.0},
    }


def intent_is_confident(prediction: dict, args: argparse.Namespace) -> bool:
    top3 = prediction["top3"]
    if not top3:
        return False
    best_conf = float(top3[0]["confidence"])
    second_conf = float(top3[1]["confidence"]) if len(top3) > 1 else 0.0
    return best_conf >= args.intent_confidence_threshold and (best_conf - second_conf) >= args.intent_margin_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local webcam sign recognition demo (Clean V2 HUD).")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--min-hand-frame-ratio", default=0.35, type=float)
    parser.add_argument("--min-face-frame-ratio", default=0.10, type=float)
    parser.add_argument("--min-hand-motion", default=0.015, type=float)
    parser.add_argument("--confidence-threshold", default=0.25, type=float)
    parser.add_argument("--confidence-margin-threshold", default=0.03, type=float)
    parser.add_argument("--reference-stable-frames", default=12, type=int)
    parser.add_argument("--reference-position-tolerance", default=0.35, type=float)
    parser.add_argument("--intent-confidence-threshold", default=0.65, type=float)
    parser.add_argument("--intent-margin-threshold", default=0.12, type=float)
    parser.add_argument("--reference-capture-seconds", default=2.0, type=float)
    parser.add_argument("--sample-video-dir", default=DEFAULT_PRACTICE_VIDEO_DIR, type=Path)
    parser.add_argument("--sample-video-max-frames", default=96, type=int)
    parser.add_argument("--sample-match-threshold", default=0.85, type=float)
    parser.add_argument("--practice-label", default=None)
    parser.add_argument("--auto-detect-intent", action="store_true")
    parser.add_argument("--test-output-dir", default=DEFAULT_TEST_OUTPUT_DIR, type=Path)
    parser.add_argument("--test-video-fps", default=20.0, type=float)
    parser.add_argument("--no-mirror-camera", action="store_true")
    parser.add_argument("--no-mirror-sample-video", action="store_true")
    args = parser.parse_args()

    recognizer = OnnxSignRecognizer(args.model, args.labels, args.config)
    if args.confidence_threshold is not None:
        recognizer.threshold = args.confidence_threshold
    recognizer.margin_threshold = args.confidence_margin_threshold
    extractor, schema_version, extractor_name = build_extractor(recognizer)
    hand_tracker = LiveHandTracker()

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"Opened camera {args.camera} with {extractor_name}. Press Q in the webcam window to quit.", flush=True)
    cv2.startWindowThread()
    cv2.namedWindow("VSL MVP Demo - Ban V2", cv2.WINDOW_NORMAL)

    labels = list(recognizer.labels)
    guided_mode = not args.auto_detect_intent
    practice_index = label_index(labels, args.practice_label)
    sample_video: SampleVideo | None = None
    sample_status = ""
    locked_label = ""
    if guided_mode:
        locked_label, sample_video, sample_status = load_practice_sample(
            labels,
            practice_index,
            args.sample_video_dir,
            extractor,
            args.sample_video_max_frames,
        )

    phase = "capture_reference"
    recording = False
    frames: list[np.ndarray] = []
    last_result = None
    reference: ReferenceSample | None = None
    references: list[ReferenceSample] = []
    reference_history: deque[HandBox] = deque(maxlen=max(1, args.reference_stable_frames))
    recording_phase = ""
    recording_target = ""
    recording_reference_score: float | None = None
    last_saved_video: Path | None = None
    last_log_path: Path | None = None
    capture_reference_at: float | None = None
    if guided_mode:
        status = f"Follow sample: {locked_label}. Press C for 2s reference capture"
    else:
        status = "Press C for 2s starting hand capture"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Camera {args.camera} opened but did not return frames")

            model_frame = frame.copy()
            display_frame = cv2.flip(frame, 1) if not args.no_mirror_camera else frame.copy()
            display_clean_frame = display_frame.copy()
            hand_boxes = hand_tracker.detect(display_clean_frame)
            current_union = union_hand_box(hand_boxes)
            if phase == "capture_reference" and current_union is not None:
                reference_history.append(current_union)
                stable_count = len(reference_history)
                if stable_count >= args.reference_stable_frames and stable_reference_ready(reference_history, args.reference_position_tolerance):
                    status = "Starting pose is stable. Press C to capture"
                elif stable_count >= args.reference_stable_frames:
                    status = "Hand detected. Press C when the starting pose looks right"
                else:
                    status = f"Hold starting pose steady before C: {stable_count}/{args.reference_stable_frames}"
            elif phase == "capture_reference":
                reference_history.clear()
                status = "Press C for 2s starting hand capture"

            if phase == "capture_reference" and capture_reference_at is not None:
                remaining = capture_reference_at - time.monotonic()
                if remaining <= 0:
                    new_reference = make_reference(display_clean_frame, hand_boxes)
                    capture_reference_at = None
                    if new_reference is None:
                        status = "Cannot capture reference: no hand detected"
                    else:
                        reference = new_reference
                        references.append(new_reference)
                        reference_history.clear()
                        phase = "verify" if guided_mode else "detect_intent"
                        if not guided_mode:
                            locked_label = ""
                            sample_video = None
                            sample_status = ""
                        last_result = None
                        recording = False
                        frames = []
                        if guided_mode:
                            status = f"Reference captured. Follow sample {locked_label}, then press Space to record"
                        else:
                            status = "Reference captured. Press Space to record the first action"
                else:
                    status = f"Capturing starting pose in {remaining:.1f}s"

            reference_ok = None
            alignment_score = float("inf")
            if phase == "verify" and not recording:
                reference_ok, alignment_score = best_reference_alignment(references, hand_boxes, args.reference_position_tolerance)

            if recording:
                frames.append(model_frame)
                status = f"Recording {len(frames)} frames"

            # Overlay nothing on the camera frame besides HUD
            draw_hand_overlay(display_frame, hand_boxes, reference_ok)
            draw_reference_thumbnail(display_frame, reference)
            draw_sample_video(display_frame, sample_video, mirror=not args.no_mirror_sample_video)

            # Draw the beautiful clean V2 HUD
            draw_hud(
                display_frame,
                target_label=locked_label,
                status_msg=status,
                last_result=last_result,
                phase=phase,
                recording=recording,
                frames_count=len(frames)
            )
            
            cv2.imshow("VSL MVP Demo - Ban V2", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if guided_mode and key in (ord("n"), ord("p")) and not recording:
                if labels:
                    step = 1 if key == ord("n") else -1
                    practice_index = (practice_index + step) % len(labels)
                    locked_label, sample_video, sample_status = load_practice_sample(
                        labels,
                        practice_index,
                        args.sample_video_dir,
                        extractor,
                        args.sample_video_max_frames,
                    )
                    phase = "capture_reference"
                    frames = []
                    last_result = None
                    reference = None
                    references = []
                    reference_history.clear()
                    capture_reference_at = None
                    if sample_video is None:
                        status = f"Sample {locked_label} unavailable ({sample_status})"
                    else:
                        status = f"Sample changed: {locked_label}. Press C for 2s reference capture"
                continue
            if key == ord("r"):
                phase = "capture_reference"
                frames = []
                last_result = None
                recording = False
                reference = None
                references = []
                reference_history.clear()
                capture_reference_at = None
                recording_phase = ""
                recording_target = ""
                recording_reference_score = None
                if guided_mode:
                    locked_label, sample_video, sample_status = load_practice_sample(
                        labels,
                        practice_index,
                        args.sample_video_dir,
                        extractor,
                        args.sample_video_max_frames,
                    )
                    status = f"Follow sample: {locked_label}. Press C for 2s reference capture"
                else:
                    locked_label = ""
                    sample_video = None
                    sample_status = ""
                    status = "Press C for 2s starting hand capture"
                continue
            if key == ord("c"):
                if recording:
                    status = "Cannot capture reference while recording"
                    continue
                phase = "capture_reference"
                last_result = None
                frames = []
                recording = False
                reference_history.clear()
                capture_reference_at = time.monotonic() + max(0.0, args.reference_capture_seconds)
                status = f"Capturing starting pose in {args.reference_capture_seconds:.1f}s"
                continue
            if key != 32:
                continue

            if phase == "capture_reference":
                status = "Press C to start the 2s reference capture first"
                continue
            if phase == "verify" and not recording:
                reference_ok, alignment_score = best_reference_alignment(references, hand_boxes, args.reference_position_tolerance)
                if not reference_ok:
                    status = f"Recording with reference offset {alignment_score:.2f}"

            if not recording:
                frames = []
                last_result = None
                recording = True
                recording_phase = phase
                recording_target = locked_label
                recording_reference_score = None if alignment_score == float("inf") else alignment_score
                if not status.startswith("Recording with reference offset"):
                    status = "Recording"
                continue

            recording = False
            status = "Processing"
            last_saved_video = save_test_video(frames, args.test_output_dir, recording_target, recording_phase, args.test_video_fps)
            result = extractor.extract_frames(frames)
            if result.status != "ok":
                last_result = low_quality_result(result.status)
                last_log_path = log_attempt(
                    args.test_output_dir,
                    phase=recording_phase,
                    target_label=recording_target,
                    sample_video=sample_video,
                    video_path=last_saved_video,
                    result_status=result.status,
                    reference_score=recording_reference_score,
                )
                status = f"Bad recording: {result.status}"
                continue

            quality, quality_status = sequence_quality(result, schema_version, args)
            if quality_status != "ok":
                last_result = low_quality_result(quality_status, quality)
                last_log_path = log_attempt(
                    args.test_output_dir,
                    phase=recording_phase,
                    target_label=recording_target,
                    sample_video=sample_video,
                    video_path=last_saved_video,
                    result_status=quality_status,
                    quality=quality,
                    reference_score=recording_reference_score,
                )
                status = f"Bad recording: {quality_status}"
                continue

            prediction = recognizer.predict(result.features)
            prediction["quality"] = quality

            if phase == "detect_intent":
                if intent_is_confident(prediction, args):
                    locked_label = prediction["label"]
                    prediction["verified"] = False
                    last_result = prediction
                    phase = "verify"
                    sample_video, sample_status = load_sample_video(
                        locked_label,
                        args.sample_video_dir,
                        extractor,
                        args.sample_video_max_frames,
                    )
                    if sample_video is None:
                        status = f"Target locked: {locked_label}. No sample video ({sample_status}); press Space to verify"
                    else:
                        status = f"Target locked: {locked_label}. Follow sample, then press Space to verify"
                else:
                    prediction["status"] = "uncertain_intent"
                    last_result = prediction
                    status = "Intent not confident. Record the first action again"
                last_log_path = log_attempt(
                    args.test_output_dir,
                    phase=recording_phase,
                    target_label=locked_label or recording_target,
                    sample_video=sample_video,
                    video_path=last_saved_video,
                    result_status=prediction["status"],
                    prediction=prediction,
                    quality=quality,
                    reference_score=recording_reference_score,
                )
                continue

            if phase == "verify":
                prediction["expected_label"] = locked_label
                sample_score = sample_match_score(result.features, sample_video.features) if sample_video else None
                if sample_score is not None:
                    prediction["sample_score"] = sample_score
                    prediction["sample_ok"] = sample_score <= args.sample_match_threshold
                else:
                    prediction["sample_score"] = None
                    prediction["sample_ok"] = True
                prediction["verified"] = (
                    prediction["status"] == "ok"
                    and prediction["label"] == locked_label
                )
                if prediction["verified"]:
                    last_result = prediction
                    status = f"Correct: {locked_label}. Press N for next sample or Space to retry"
                else:
                    if prediction["label"] != locked_label:
                        prediction["status"] = "wrong_target"
                    last_result = prediction
                    status = f"Try again. Expected {locked_label}. Press Space to retry or N for next sample"
                last_log_path = log_attempt(
                    args.test_output_dir,
                    phase=recording_phase,
                    target_label=locked_label,
                    sample_video=sample_video,
                    video_path=last_saved_video,
                    result_status=prediction["status"],
                    prediction=prediction,
                    quality=quality,
                    reference_score=recording_reference_score,
                )
    finally:
        cap.release()
        hand_tracker.close()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
