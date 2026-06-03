import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set UTF-8 encoding for Vietnamese terminal output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Import modules from src
sys.path.append(str(Path(__file__).parent.parent / "src"))
from vsl_mvp.config import FeatureConfigV2
from vsl_mvp.landmarks_v2 import HolisticLandmarkExtractor

def load_font(size: int = 22):
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
    font = load_font(20)
    y = 16
    for text in lines:
        draw.text((16, y), text, font=font, fill=(30, 240, 80), stroke_width=2, stroke_fill=(0, 0, 0))
        y += 28
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def select_class(classes):
    print("=" * 60)
    print(" DANH SÁCH 30 CỬ CHỈ ĐANG HỌC:")
    print("=" * 60)
    for idx, c in enumerate(classes):
        print(f"  {idx + 1:2d}. {c}")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Nhập số thứ tự của từ bạn muốn ghi hình (hoặc 'q' để thoát): ").strip()
            if choice.lower() == 'q':
                return None
            num = int(choice)
            if 1 <= num <= len(classes):
                return classes[num - 1]
            else:
                print(f"Vui lòng chọn từ 1 đến {len(classes)}.")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ.")

def main():
    workspace_dir = Path(__file__).parent.parent
    classes_path = workspace_dir / "data/processed/selected_classes_vsl_mvp30.json"
    custom_dir = workspace_dir / "data/custom_recorded"
    
    if not classes_path.exists():
        print(f"Lỗi: Không tìm thấy file {classes_path}")
        return
        
    with open(classes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = sorted(data["classes"])
    
    # Class Selection
    label = select_class(classes)
    if not label:
        print("Đã thoát trình ghi hình.")
        return
        
    print(f"\n>> Bắt đầu ghi hình cử chỉ: '{label}'")
    print("OpenCV window sẽ mở ra. Thực hiện các thao tác sau:")
    print("  - Phím [Space]: Bắt đầu ghi hình / Dừng ghi hình.")
    print("  - Phím [R]: Reset ghi lại từ đầu.")
    print("  - Phím [S]: Lưu mẫu ghi âm (chỉ khả dụng sau khi dừng ghi thành công).")
    print("  - Phím [Q]: Thoát.")
    
    # Initialize Camera
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera!")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    config = FeatureConfigV2()
    extractor = HolisticLandmarkExtractor(config)
    
    cv2.namedWindow("VSL Custom Recorder", cv2.WINDOW_NORMAL)
    
    recording = False
    frames = []
    status = "Ready"
    last_res = None
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lỗi camera!")
                break
                
            if recording:
                frames.append(frame.copy())
                status = f"REC: {len(frames)} frames"
                
            lines = [
                f"Gesture: {label}",
                "Space: REC / STOP | R: Reset | S: Save | Q: Quit",
                f"Status: {status}"
            ]
            
            if last_res:
                lines.append(f"Landmarks: {last_res['status']}")
                if last_res["status"] == "ok":
                    q = last_res["quality"]
                    lines.append(f"Quality: hand {q.get('hand_frame_ratio',0.0):.2f} | face {q.get('face_ratio',0.0):.2f}")
                    lines.append(f"Motion: {q.get('hand_motion_mean',0.0):.3f}")
                    lines.append(">> PRESS 'S' TO SAVE THIS SAMPLE! <<")
                else:
                    lines.append(f"Reason: {last_res['status']}")
                    lines.append(">> PRESS 'R' TO RECORD AGAIN <<")
                    
            draw_lines(frame, lines)
            cv2.imshow("VSL Custom Recorder", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Space to Toggle recording
            if key == 32:
                if not recording:
                    frames = []
                    last_res = None
                    recording = True
                    status = "RECORDING..."
                else:
                    recording = False
                    status = "Extracting landmarks..."
                    cv2.imshow("VSL Custom Recorder", frame)
                    cv2.waitKey(1)
                    
                    # Extract frames
                    result = extractor.extract_frames(frames)
                    if result.status == "ok":
                        last_res = {
                            "status": "ok",
                            "features": result.features,
                            "quality": result.quality,
                            "valid_frames": result.valid_frames
                        }
                        status = "Success (Press S to Save)"
                    else:
                        last_res = {
                            "status": result.status
                        }
                        status = f"Extraction Failed: {result.status}"
                        
            # Reset
            elif key == ord("r"):
                frames = []
                last_res = None
                recording = False
                status = "Ready"
                print("Reset recording.")
                
            # Save
            elif key == ord("s"):
                if last_res and last_res["status"] == "ok":
                    label_dir = custom_dir / label.replace(" ", "_")
                    label_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = int(time.time())
                    filename = label_dir / f"custom_{timestamp}.npz"
                    
                    quality_keys = sorted(last_res["quality"].keys())
                    quality_vals = [last_res["quality"][k] for k in quality_keys]
                    
                    np.savez_compressed(
                        filename,
                        X=np.asarray(last_res["features"], dtype=np.float32),
                        label=label,
                        quality_keys=np.asarray(quality_keys),
                        quality=np.asarray(quality_vals, dtype=np.float32),
                        valid_frames=np.asarray([last_res["valid_frames"]], dtype=np.int32)
                    )
                    
                    print(f"Đã lưu mẫu custom thành công tại: {filename}")
                    status = "SAVED SUCCESS!"
                    last_res = None
                    frames = []
                else:
                    print("Không có mẫu ghi thành công nào để lưu!")
                    
            # Quit
            elif key == ord("q"):
                break
                
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
        print("Đóng chương trình ghi hình.")

if __name__ == "__main__":
    main()
