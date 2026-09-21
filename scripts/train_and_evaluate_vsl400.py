import os
import sys
import subprocess
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

run_dir = ROOT / "runs" / "vsl_mvp400_v2_lite_transformer"
features = ROOT / "data" / "processed" / "features_vsl400_v2_sf8.npz"
checkpoint = run_dir / "best_prev.pt"

print("=" * 70, flush=True)
print("BẮT ĐẦU QUY TRÌNH TIẾP TỤC HUẤN LUYỆN & KIỂM THỬ MÔ HÌNH 400 TỪ", flush=True)
print("=" * 70, flush=True)

# 1. Resume training for 10 epochs
train_cmd = [
    PYTHON, "-m", "src.vsl_mvp.train",
    "--features", str(features),
    "--model", "lite_transformer",
    "--out-dir", str(run_dir),
    "--checkpoint", str(checkpoint),
    "--epochs", "10",
    "--batch-size", "64",
    "--lr", "5e-4",
    "--confidence-threshold", "0.35",
    "--augment-copies", "2"
]
print("\n[Bước 1/3] Đang nạp trọng số và huấn luyện 10 epochs tiếp theo...", flush=True)
print("Lệnh:", " ".join(train_cmd), flush=True)
t0 = time.time()
res = subprocess.run(train_cmd, cwd=str(ROOT))
if res.returncode != 0:
    print(f"Lỗi khi huấn luyện (mã lỗi {res.returncode})", flush=True)
    sys.exit(res.returncode)
print(f"Huấn luyện hoàn tất trong {(time.time() - t0)/60:.1f} phút!", flush=True)

# 2. Export ONNX
print("\n[Bước 2/3] Đang xuất mô hình sang định dạng ONNX & INT8...", flush=True)
onnx_path = run_dir / "model.onnx"
int8_path = run_dir / "model.int8.onnx"

export_cmd = [
    PYTHON, "-m", "src.vsl_mvp.export_onnx",
    "--run-dir", str(run_dir),
    "--out", str(onnx_path)
]
subprocess.run(export_cmd, cwd=str(ROOT), check=True)

quant_cmd = [
    PYTHON, "-m", "src.vsl_mvp.quantize_onnx",
    "--model", str(onnx_path),
    "--out", str(int8_path)
]
subprocess.run(quant_cmd, cwd=str(ROOT), check=True)
print(f"Xuất ONNX và INT8 thành công vào {run_dir}!", flush=True)

# 3. Quick test on sample videos
print("\n[Bước 3/3] Đang chạy kiểm thử tự động trên video mẫu...", flush=True)
test_cmd = [
    PYTHON, "scripts/test_video.py",
    "--model-dir", str(run_dir),
    "--video-dir", "data/practice_videos"
]
subprocess.run(test_cmd, cwd=str(ROOT))

print("\n" + "=" * 70, flush=True)
print("TOÀN BỘ QUY TRÌNH HUẤN LUYỆN VÀ KIỂM THỬ ĐÃ HOÀN TẤT THÀNH CÔNG!", flush=True)
print("=" * 70, flush=True)
