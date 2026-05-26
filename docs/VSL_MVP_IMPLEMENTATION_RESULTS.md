# VSL MVP Implementation Results

Date: 2026-05-23

## Implemented Scope

Week 1-3 implementation is now covered for the landmark-first MVP path:

- MediaPipe Holistic V2 feature extraction.
- Explicit feature schema with hands, pose, face subset, motion, geometry, and quality masks.
- MVP-30 and MVP-50 class selection.
- V2 feature caches.
- Lite Transformer classifier.
- GRU baseline for MVP-30.
- Ablation feature generation.
- ONNX export.
- INT8 dynamic quantization.
- V2-aware webcam demo.

## Main Artifacts

### MVP-30

```text
Classes:
data/processed/selected_classes_vsl_mvp30.json

Features:
data/processed/features_vsl_mvp30_v2_sf8.npz
data/processed/features_vsl_mvp30_v2_sf8_report.json

Best deploy run:
runs/vsl_mvp30_v2_lite_transformer

Exported models:
runs/vsl_mvp30_v2_lite_transformer/model.onnx
runs/vsl_mvp30_v2_lite_transformer/model.int8.onnx
```

Feature quality:

```text
samples: 701
classes: 30
feature_dim: 327
status: 701 ok
pose_ratio: 1.000
face_ratio: 0.983
hand_frame_ratio: 0.967
both_hands_ratio: 0.839
```

Best model result:

```text
model: lite_transformer
best_top1: 0.901
best_epoch: 25
top3_at_best: 1.000
macro_f1_at_best: 0.895
split: group_shuffle_by_signer
```

Baseline:

```text
model: gru
best_top1: 0.893
best_epoch: 15
top3_at_best: 0.992
macro_f1_at_best: 0.882
split: group_shuffle_by_signer
```

### MVP-50

```text
Classes:
data/processed/selected_classes_vsl_mvp50.json

Features:
data/processed/features_vsl_mvp50_v2_sf8.npz
data/processed/features_vsl_mvp50_v2_sf8_report.json

Best deploy run:
runs/vsl_mvp50_v2_lite_transformer

Exported models:
runs/vsl_mvp50_v2_lite_transformer/model.onnx
runs/vsl_mvp50_v2_lite_transformer/model.int8.onnx
```

Feature quality:

```text
samples: 1126
classes: 50
feature_dim: 327
status: 1126 ok
pose_ratio: 1.000
face_ratio: 0.977
hand_frame_ratio: 0.966
both_hands_ratio: 0.845
```

Best model result:

```text
model: lite_transformer
best_top1: 0.871
best_epoch: 9
top3_at_best: 0.970
macro_f1_at_best: 0.866
split: group_shuffle_by_signer
```

## Model Size

```text
MVP-30 lite_transformer ONNX FP32: about 1.00 MB
MVP-30 lite_transformer ONNX INT8: about 0.40 MB

MVP-50 lite_transformer ONNX FP32: about 1.00 MB
MVP-50 lite_transformer ONNX INT8: about 0.40 MB
```

## Classifier Latency

Measured with ONNX Runtime CPU provider on 128 cached samples:

```text
MVP-30 FP32: about 0.523 ms/sample
MVP-30 INT8: about 0.617 ms/sample
MVP-50 FP32: about 0.614 ms/sample
MVP-50 INT8: about 0.608 ms/sample
```

The classifier is already very fast. End-to-end webcam latency will mostly come from MediaPipe Holistic extraction, not the classifier.

## Ablation Runs

Short 10-epoch ablation runs were created to validate the tooling:

```text
runs/ablation_vsl_mvp30_hands_lite
runs/ablation_vsl_mvp30_hands_pose_lite
runs/ablation_vsl_mvp30_hands_pose_face_lite
runs/ablation_vsl_mvp30_full_lite
```

These are not final scientific ablations because they use fewer epochs than the main model. They confirm that the ablation pipeline works and can be re-run longer.

## Demo Commands

Recommended MVP-30 demo:

```powershell
python -m vsl_mvp.webcam_demo `
  --model runs\vsl_mvp30_v2_lite_transformer\model.onnx `
  --config runs\vsl_mvp30_v2_lite_transformer\config.json `
  --labels runs\vsl_mvp30_v2_lite_transformer\labels.json `
  --confidence-threshold 0.25 `
  --confidence-margin-threshold 0.03
```

Smaller INT8 model:

```powershell
python -m vsl_mvp.webcam_demo `
  --model runs\vsl_mvp30_v2_lite_transformer\model.int8.onnx `
  --config runs\vsl_mvp30_v2_lite_transformer\config.json `
  --labels runs\vsl_mvp30_v2_lite_transformer\labels.json `
  --confidence-threshold 0.25 `
  --confidence-margin-threshold 0.03
```

MVP-50 demo:

```powershell
python -m vsl_mvp.webcam_demo `
  --model runs\vsl_mvp50_v2_lite_transformer\model.onnx `
  --config runs\vsl_mvp50_v2_lite_transformer\config.json `
  --labels runs\vsl_mvp50_v2_lite_transformer\labels.json `
  --confidence-threshold 0.25 `
  --confidence-margin-threshold 0.03
```

## Recommendation

Use MVP-30 for the first public/demo flow because it has the strongest validation result:

```text
top1: 0.901
top3: 1.000
model size INT8: about 0.40 MB
classifier latency: under 1 ms/sample
```

Keep MVP-50 as the expansion model. It is already usable for testing, but it should receive more data, longer ablations, and confusion-matrix cleanup before being the main demo.

## Next Technical Steps

1. Run real webcam validation with 3-5 people.
2. Tune `confidence_threshold` and `confidence_margin_threshold` on live recordings.
3. Review confusion matrices for MVP-30 and MVP-50.
4. Re-run ablations for 25 epochs each if deciding whether face/motion features should stay in the final schema.
5. Add auto segmentation only after manual start/stop demo is stable.
