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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FeatureConfigV2
from .infer import OnnxSignRecognizer
from .landmarks import LandmarkExtractor
from .landmarks_v2 import HolisticLandmarkExtractor


HAND_FEATURE_DIMS = 21 * 3 * 2
REFERENCE_THUMB_WIDTH = 180
SAMPLE_THUMB_WIDTH = 240
DEFAULT_PRACTICE_VIDEO_DIR = Path("data/practice_videos")
DEFAULT_TEST_OUTPUT_DIR = Path("runs/webcam_tests/hidden_data")
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


def draw_lines(frame: np.ndarray, lines: list[str]) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = load_font(24)
    y = 16
    for text in lines:
        draw.text((16, y), text, font=font, fill=(30, 240, 80), stroke_width=2, stroke_fill=(0, 0, 0))
        y += 32
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


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
    hands = sequence[:, :HAND_FEATURE_DIMS].reshape(sequence.shape[0], 42, 3)
    has_hand = np.any(hands != 0.0, axis=2).any(axis=1)
    return float(np.mean(has_hand))


def hand_motion_score(sequence: np.ndarray) -> float:
    hands = sequence[:, :HAND_FEATURE_DIMS].reshape(sequence.shape[0], 42, 3)
    valid = np.any(hands != 0.0, axis=2)
    centers = []
    for idx in range(sequence.shape[0]):
        if valid[idx].any():
            centers.append(hands[idx, valid[idx], :2].mean(axis=0))
    if len(centers) < 2:
        return 0.0
    centers_np = np.asarray(centers)
    deltas = np.linalg.norm(np.diff(centers_np, axis=0), axis=1)
    return float(np.mean(deltas))


def union_hand_box(hand_boxes: list[HandBox]) -> HandBox | None:
    if not hand_boxes:
        return None
    x1 = min(hand.bbox[0] for hand in hand_boxes)
    y1 = min(hand.bbox[1] for hand in hand_boxes)
    x2 = max(hand.bbox[2] for hand in hand_boxes)
    y2 = max(hand.bbox[3] for hand in hand_boxes)
    return HandBox(
        label="Both hands" if len(hand_boxes) > 1 else hand_boxes[0].label,
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

    label_key = text_key(label)
    candidates = sorted(sample_dir.glob("*.mp4"))
    for path in candidates:
        if text_key(sample_label_from_path(path)) == label_key:
            return path
    for path in candidates:
        sample_key = text_key(sample_label_from_path(path))
        if label_key in sample_key or sample_key in label_key:
            return path
    return None


def read_sample_frames(path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    frames: list[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int32)
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    cap.release()
    return frames


def load_sample_video(label: str, sample_dir: Path, extractor, max_frames: int) -> tuple[SampleVideo | None, str]:
    path = find_sample_video_path(label, sample_dir)
    if path is None:
        return None, "sample_not_found"
    extract_result = extractor.extract_video(path)
    if extract_result.status != "ok":
        return None, f"sample_{extract_result.status}"
    frames = read_sample_frames(path, max_frames)
    if not frames:
        return None, "sample_cannot_open"
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
    if user_features.shape != sample_features.shape:
        return float("inf")
    dims = min(HAND_FEATURE_DIMS, user_features.shape[1])
    user_hands = user_features[:, :dims]
    sample_hands = sample_features[:, :dims]
    valid = (np.any(user_hands != 0.0, axis=1)) & (np.any(sample_hands != 0.0, axis=1))
    if not np.any(valid):
        return float("inf")
    return float(np.mean(np.linalg.norm(user_hands[valid] - sample_hands[valid], axis=1)))


def safe_slug(text: str, fallback: str = "unknown") -> str:
    key = text_key(text)
    chars = [char if char.isalnum() else "_" for char in key]
    slug = "_".join("".join(chars).split("_"))
    return slug or fallback


def save_test_video(frames: list[np.ndarray], output_dir: Path, label: str, phase: str, fps: float) -> Path | None:
    if not frames:
        return None

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{safe_slug(label)}_{safe_slug(phase)}.mp4"
    path = videos_dir / filename
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return None
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    return path


def append_test_log(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "test_log.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return log_path


def log_attempt(
    output_dir: Path,
    *,
    phase: str,
    target_label: str,
    sample_video: SampleVideo | None,
    video_path: Path | None,
    result_status: str,
    prediction: dict | None = None,
    quality: dict[str, float] | None = None,
    reference_score: float | None = None,
) -> Path:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "target_label": target_label,
        "sample_video": str(sample_video.path) if sample_video else "",
        "test_video": str(video_path) if video_path else "",
        "result_status": result_status,
        "reference_score": reference_score,
        "quality": quality or {},
    }
    if prediction:
        payload.update(
            {
                "predicted_label": prediction.get("label", ""),
                "confidence": prediction.get("confidence", 0.0),
                "confidence_margin": prediction.get("confidence_margin", 0.0),
                "verified": bool(prediction.get("verified", False)),
                "sample_score": prediction.get("sample_score"),
                "sample_ok": prediction.get("sample_ok"),
                "top3": prediction.get("top3", []),
            }
        )
    return append_test_log(output_dir, payload)


def stable_reference_ready(history: deque[HandBox], tolerance: float) -> bool:
    if not history:
        return False
    centers = np.asarray([box.center for box in history], dtype=np.float32)
    sizes = np.asarray([box.size for box in history], dtype=np.float32)
    max_center_span = float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))
    max_size_span = float(np.max(np.linalg.norm(sizes - sizes.mean(axis=0), axis=1)))
    return max_center_span <= tolerance * 0.25 and max_size_span <= tolerance * 0.25


def reference_alignment(reference: ReferenceSample | None, hand_boxes: list[HandBox], tolerance: float) -> tuple[bool, float]:
    ref_box = union_hand_box(reference.hand_boxes) if reference else None
    cur_box = union_hand_box(hand_boxes)
    if ref_box is None or cur_box is None:
        return False, float("inf")

    ref_diag = max(float(np.linalg.norm(ref_box.size)), 0.01)
    center_delta = float(np.linalg.norm(np.asarray(cur_box.center) - np.asarray(ref_box.center)) / ref_diag)
    size_delta = float(np.linalg.norm(np.asarray(cur_box.size) - np.asarray(ref_box.size)) / ref_diag)
    score = center_delta + 0.5 * size_delta
    return score <= tolerance, score


def best_reference_alignment(references: list[ReferenceSample], hand_boxes: list[HandBox], tolerance: float) -> tuple[bool, float]:
    if not references:
        return False, float("inf")
    results = [reference_alignment(reference, hand_boxes, tolerance) for reference in references]
    return min(results, key=lambda item: item[1])


def make_reference(frame: np.ndarray, hand_boxes: list[HandBox]) -> ReferenceSample | None:
    if not hand_boxes:
        return None
    return ReferenceSample(frame=frame.copy(), hand_boxes=list(hand_boxes))


def build_extractor(recognizer: OnnxSignRecognizer):
    schema_version = str(recognizer.config.get("schema_version", "v1_hands_pose"))
    if schema_version.startswith("v2"):
        config = FeatureConfigV2(sequence_length=int(recognizer.config["sequence_length"]))
        return HolisticLandmarkExtractor(config), schema_version, "MediaPipe Holistic V2"
    return LandmarkExtractor(), schema_version, "MediaPipe Hands+Pose V1"


def sequence_quality(result, schema_version: str, args: argparse.Namespace) -> tuple[dict[str, float], str]:
    if schema_version.startswith("v2"):
        hand_ratio = float(result.quality.get("hand_frame_ratio", hand_frame_ratio(result.features)))
        face_ratio = float(result.quality.get("face_ratio", 0.0))
    else:
        hand_ratio = hand_frame_ratio(result.features)
        face_ratio = 0.0
    motion = hand_motion_score(result.features)
    quality = {"hand_ratio": hand_ratio, "face_ratio": face_ratio, "motion": motion}

    if hand_ratio < args.min_hand_frame_ratio:
        return quality, "low_hand_ratio"
    if schema_version.startswith("v2") and face_ratio < args.min_face_frame_ratio:
        return quality, "low_face_ratio"
    if motion < args.min_hand_motion:
        return quality, "low_motion"
    return quality, "ok"


def low_quality_result(status: str, quality: dict[str, float] | None = None) -> dict:
    result = {"label": "", "confidence": 0.0, "top3": [], "status": status}
    if quality is not None:
        result["quality"] = quality
    return result


def intent_is_confident(prediction: dict, args: argparse.Namespace) -> bool:
    top3 = prediction.get("top3", [])
    if not top3:
        return False
    best_conf = float(top3[0]["confidence"])
    second_conf = float(top3[1]["confidence"]) if len(top3) > 1 else 0.0
    return best_conf >= args.intent_confidence_threshold and (best_conf - second_conf) >= args.intent_margin_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local webcam sign recognition demo.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--min-hand-frame-ratio", default=0.45, type=float)
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
    cv2.namedWindow("VSL MVP Demo", cv2.WINDOW_NORMAL)

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

            draw_hand_overlay(display_frame, hand_boxes, reference_ok)
            draw_reference_thumbnail(display_frame, reference)
            draw_sample_video(display_frame, sample_video, mirror=not args.no_mirror_sample_video)

            lines = [
                "Space: record/stop | C: 2s reference | N/P: sample | R: reset | Q: quit",
                f"Mode: {'guided sample' if guided_mode else 'auto detect'}",
                f"Preview: {'mirrored' if not args.no_mirror_camera else 'normal'} | Model input: normal",
                f"Step: {phase}",
                status,
            ]
            if locked_label:
                lines.append(f"Target: {locked_label}")
            if sample_video is not None:
                lines.append(f"Sample: {sample_video.path.name}")
            elif sample_status:
                lines.append(f"Sample: {sample_status}")
            if last_saved_video is not None:
                lines.append(f"Saved test: {last_saved_video.name}")
            if last_log_path is not None:
                lines.append(f"Log: {last_log_path.name}")
            if references:
                lines.append(f"References: {len(references)}")
            if phase == "verify" and not recording and reference_ok is False:
                lines.append(f"Reference offset: {alignment_score:.2f} (you can still record)")
            if last_result:
                if last_result["status"] == "ok":
                    prefix = "Correct" if last_result.get("verified") else "Locked"
                    lines.append(f"{prefix}: {last_result['label']} ({last_result['confidence']:.2f})")
                elif last_result.get("label"):
                    lines.append(f"Detected: {last_result['label']} ({last_result['confidence']:.2f})")
                else:
                    lines.append("Prediction: not confident / try again")
                if last_result["top3"]:
                    lines.append("Top3: " + ", ".join(f"{item['label']} {item['confidence']:.2f}" for item in last_result["top3"]))
                if "quality" in last_result:
                    q = last_result["quality"]
                    lines.append(f"Quality: hand {q['hand_ratio']:.2f} | face {q['face_ratio']:.2f} | motion {q['motion']:.2f}")
                if last_result.get("sample_score") is not None:
                    sample_ok = "ok" if last_result.get("sample_ok") else "mismatch"
                    lines.append(f"Sample match: {last_result['sample_score']:.2f} ({sample_ok})")
                if last_result["status"] != "ok":
                    lines.append(f"Reason: {last_result['status']}")
            draw_lines(display_frame, lines)
            cv2.imshow("VSL MVP Demo", display_frame)

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
