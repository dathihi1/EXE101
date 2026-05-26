import sys
from pathlib import Path

# Set console output to UTF-8 to support Vietnamese characters on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Add src/ to python path so we can import vsl_mvp
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
from vsl_mvp.config import FeatureConfig, FeatureConfigV2
from vsl_mvp.infer import OnnxSignRecognizer
from vsl_mvp.landmarks import LandmarkExtractor
from vsl_mvp.landmarks_v2 import HolisticLandmarkExtractor


def build_extractor(recognizer: OnnxSignRecognizer):
    schema_version = str(recognizer.config.get("schema_version", "v1_hands_pose"))
    if schema_version.startswith("v2"):
        config = FeatureConfigV2(sequence_length=int(recognizer.config["sequence_length"]))
        return HolisticLandmarkExtractor(config), schema_version, "MediaPipe Holistic V2"
    return LandmarkExtractor(), schema_version, "MediaPipe Hands+Pose V1"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test sign language recognition on sample videos.")
    parser.add_argument("--model-dir", default="runs/vsl_mvp30_v2_lite_transformer", type=str,
                        help="Path to the trained model directory containing model.onnx, config.json, labels.json")
    parser.add_argument("--video-dir", default="data/processed/sample_videos", type=str,
                        help="Directory containing test MP4 videos")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    video_dir = Path(args.video_dir)

    model_path = model_dir / "model.onnx"
    labels_path = model_dir / "labels.json"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        print(f"Error: ONNX model not found at {model_path}")
        return

    print("=" * 60)
    print(f"Loading Model: {model_dir.name}")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = OnnxSignRecognizer(model_path, labels_path, config_path)
    
    # Initialize extractor based on model schema
    extractor, schema_version, extractor_name = build_extractor(recognizer)
    print(f"Schema version: {schema_version}")
    print(f"Landmark Extractor: {extractor_name}")
    print("-" * 60)

    # Find videos
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 videos found in {video_dir}")
        return

    print(f"Found {len(video_files)} test videos. Starting extraction and inference...\n")
    
    results = []
    correct_count = 0
    total_valid = 0

    for video_file in sorted(video_files):
        # Extract ground truth label from filename, e.g. "Anh__000000.mp4" -> "Anh"
        filename = video_file.name
        ground_truth = filename.split("__")[0].replace("_", " ") # Handles things like "Chủ_nhật" -> "Chủ nhật"
        
        print(f"Testing video: {video_file.name}")
        print(f"  └─ Ground Truth: '{ground_truth}'")
        
        # Run extractor
        extract_res = extractor.extract_video(video_file)
        
        if extract_res.status != "ok":
            print(f"  └─ Extraction Failed: {extract_res.status}")
            quality_str = ""
            if hasattr(extract_res, "quality"):
                q = extract_res.quality
                quality_str = f" (Hand: {q.get('hand_frame_ratio', 0.0):.2f}, Face: {q.get('face_ratio', 0.0):.2f})"
            print(f"     Metrics: {quality_str}")
            results.append({
                "file": video_file.name,
                "ground_truth": ground_truth,
                "status": f"Extraction Fail ({extract_res.status})",
                "pred": "N/A",
                "conf": 0.0,
                "correct": False
            })
            print("-" * 60)
            continue
        
        total_valid += 1
        # Predict using ONNX recognizer
        pred_res = recognizer.predict(extract_res.features)
        
        pred_label = pred_res["label"]
        confidence = pred_res["confidence"]
        status = pred_res["status"]
        top3 = pred_res["top3"]
        
        is_correct = (pred_label.strip().lower() == ground_truth.strip().lower())
        if is_correct:
            correct_count += 1
            match_marker = "✅ MATCH"
        else:
            match_marker = "❌ MISMATCH"
            
        print(f"  └─ Status: {status}")
        print(f"  └─ Predicted: '{pred_label}' with confidence {confidence:.4f} ({match_marker})")
        print("  └─ Top 3 Predictions:")
        for idx, item in enumerate(top3):
            print(f"     {idx+1}. '{item['label']}': {item['confidence']:.4f}")
            
        results.append({
            "file": video_file.name,
            "ground_truth": ground_truth,
            "status": status,
            "pred": pred_label,
            "conf": confidence,
            "correct": is_correct
        })
        print("-" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total files:            {len(video_files)}")
    print(f"Successfully processed: {total_valid}/{len(video_files)}")
    if total_valid > 0:
        accuracy = correct_count / total_valid * 100
        print(f"Inference Accuracy:     {correct_count}/{total_valid} ({accuracy:.2f}%)")
    else:
        print("No videos were successfully processed.")
    print("=" * 60)
    
    # Clean up
    extractor.close()

if __name__ == "__main__":
    main()
