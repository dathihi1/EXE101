from __future__ import annotations

import argparse
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FeatureConfigV2
from .infer import OnnxSignRecognizer
from .landmarks import LandmarkExtractor
from .landmarks_v2 import HolisticLandmarkExtractor


HAND_FEATURE_DIMS = 21 * 3 * 2


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


def draw_lines(frame, lines):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = load_font(24)
    y = 16
    for text in lines:
        draw.text((16, y), text, font=font, fill=(30, 240, 80), stroke_width=2, stroke_fill=(0, 0, 0))
        y += 32
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


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


def build_extractor(recognizer: OnnxSignRecognizer):
    schema_version = str(recognizer.config.get("schema_version", "v1_hands_pose"))
    if schema_version.startswith("v2"):
        config = FeatureConfigV2(sequence_length=int(recognizer.config["sequence_length"]))
        return HolisticLandmarkExtractor(config), schema_version, "MediaPipe Holistic V2"
    return LandmarkExtractor(), schema_version, "MediaPipe Hands+Pose V1"


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
    args = parser.parse_args()

    recognizer = OnnxSignRecognizer(args.model, args.labels, args.config)
    if args.confidence_threshold is not None:
        recognizer.threshold = args.confidence_threshold
    recognizer.margin_threshold = args.confidence_margin_threshold
    extractor, schema_version, extractor_name = build_extractor(recognizer)

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"Opened camera {args.camera} with {extractor_name}. Press Q in the webcam window to quit.", flush=True)
    cv2.startWindowThread()
    cv2.namedWindow("VSL MVP Demo", cv2.WINDOW_NORMAL)

    recording = False
    frames = []
    last_result = None
    status = "Ready"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Camera {args.camera} opened but did not return frames")
            if recording:
                frames.append(frame.copy())
                status = f"Recording {len(frames)} frames"

            lines = ["Space: start/stop | R: reset | Q: quit", status]
            if last_result:
                if last_result["status"] == "ok":
                    lines.append(f"Prediction: {last_result['label']} ({last_result['confidence']:.2f})")
                elif last_result.get("label"):
                    lines.append(f"Suggestion: {last_result['label']} ({last_result['confidence']:.2f})")
                else:
                    lines.append("Prediction: Không chắc / thử lại")
                if last_result["top3"]:
                    lines.append("Top3: " + ", ".join(f"{item['label']} {item['confidence']:.2f}" for item in last_result["top3"]))
                if "quality" in last_result:
                    q = last_result["quality"]
                    if "face_ratio" in q:
                        lines.append(f"Quality: hand {q['hand_ratio']:.2f} | face {q['face_ratio']:.2f} | motion {q['motion']:.2f}")
                    else:
                        lines.append(f"Quality: hand {q['hand_ratio']:.2f} | motion {q['motion']:.2f}")
                if last_result["status"] != "ok":
                    lines.append(f"Reason: {last_result['status']}")
            draw_lines(frame, lines)
            cv2.imshow("VSL MVP Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                frames = []
                last_result = None
                recording = False
                status = "Ready"
            if key == 32:
                if not recording:
                    frames = []
                    last_result = None
                    recording = True
                    status = "Recording"
                else:
                    recording = False
                    status = "Processing"
                    result = extractor.extract_frames(frames)
                    if result.status != "ok":
                        last_result = {"label": "", "confidence": 0.0, "top3": [], "status": result.status}
                        status = f"Bad recording: {result.status}"
                    else:
                        if schema_version.startswith("v2"):
                            hand_ratio = float(result.quality.get("hand_frame_ratio", hand_frame_ratio(result.features)))
                            face_ratio = float(result.quality.get("face_ratio", 0.0))
                        else:
                            hand_ratio = hand_frame_ratio(result.features)
                            face_ratio = 0.0
                        motion = hand_motion_score(result.features)
                        quality = {"hand_ratio": hand_ratio, "face_ratio": face_ratio, "motion": motion}
                        if hand_ratio < args.min_hand_frame_ratio:
                            last_result = {
                                "label": "",
                                "confidence": 0.0,
                                "top3": [],
                                "status": "low_hand_ratio",
                                "quality": quality,
                            }
                        elif schema_version.startswith("v2") and face_ratio < args.min_face_frame_ratio:
                            last_result = {
                                "label": "",
                                "confidence": 0.0,
                                "top3": [],
                                "status": "low_face_ratio",
                                "quality": quality,
                            }
                        elif motion < args.min_hand_motion:
                            last_result = {
                                "label": "",
                                "confidence": 0.0,
                                "top3": [],
                                "status": "low_motion",
                                "quality": quality,
                            }
                        else:
                            last_result = recognizer.predict(result.features)
                            last_result["quality"] = quality
                        status = "Ready"
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
