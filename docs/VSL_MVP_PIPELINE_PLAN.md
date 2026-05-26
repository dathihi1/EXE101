# VSL MVP Pipeline Plan

Date: 2026-05-23

## 1. Goal

Build a Vietnamese Sign Language isolated-word recognition MVP that is:

- Accurate enough for a controlled demo.
- Fast enough for webcam use on a normal laptop CPU.
- Small enough to export to ONNX/TFLite and ship locally.
- Designed well from the beginning so it can grow from 30-50 classes to 400 classes later.

The system should not train a hand/body/face detector from scratch. It should reuse a strong existing landmark extractor, then train only the lightweight sign classifier.

Recommended target for the first MVP:

```text
30-50 clean VSL classes
manual start/stop gesture recording
top-3 predictions
confidence threshold
local webcam demo
ONNX or TFLite export
```

## 2. Current Project Context

Current local data from `Part_1 + Part_2`:

```text
clips: 7073
classes: 400
signers: 9
view: front_view only
average clips/class: about 17.7
current feature dim: 150
current feature source: MediaPipe Hands + upper-body Pose
```

Current pipeline:

```text
Video/Webcam
-> MediaPipe Hands + MediaPipe Pose
-> left hand + right hand + shoulders/elbows/wrists
-> normalize to 64 frames
-> GRU or Transformer
-> ONNX webcam demo
```

Main limitation:

```text
No face/non-manual markers yet.
No dedicated missing-landmark repair.
No explicit quality mask.
No motion/relative-geometry feature layer.
400 classes are too many for a high-confidence MVP with only Part_1 + Part_2.
```

## 3. Research Takeaways

1. Use MediaPipe Holistic or MediaPipe Tasks Holistic Landmarker as the perception layer. It provides pose, face, and both hands in real time.

2. Use landmark/keypoint data instead of raw RGB for the MVP. Landmark models are smaller, faster, and need less data than RGB video models.

3. Add non-manual features carefully. Sign language can depend on head, shoulders, torso, eyebrows, eyes/gaze, eyelids, nose, mouth/lips, cheeks, and chin. But VSL400 anonymizes facial regions, so face landmark quality must be measured before making face features dominant.

4. Preprocessing matters as much as the classifier. Anchor normalization, missing-keypoint reconstruction, dynamic trimming, and augmentation can improve accuracy significantly.

5. Keep an honest validation split. Prefer signer-based validation when possible; random splits can make accuracy look better than real-world performance.

## 4. Target Pipeline

```text
Video/Webcam
-> frame sampling
-> frame quality check
-> MediaPipe Holistic
-> LandmarkSchemaV2
-> landmark quality masks
-> missing landmark interpolation
-> anchor normalization
-> dynamic onset/offset trimming
-> resample/pad to 64 frames
-> motion + relative geometry features
-> lightweight temporal classifier
-> confidence calibration
-> top-k + reject logic
-> ONNX/TFLite deployment
```

### 4.1 LandmarkSchemaV2

The feature schema should be explicit and versioned.

```text
hands:
  left hand: 21 x 3
  right hand: 21 x 3

upper_body:
  shoulders
  elbows
  wrists
  neck/chest anchor
  nose/head anchor
  optional hips for torso angle

face_subset:
  lips/mouth
  eyebrows
  eyes/eyelids
  nose anchor
  chin
  selected face contour points

motion:
  hand center velocity
  wrist velocity
  hand-to-hand distance change
  hand-to-mouth distance change
  hand-to-chest distance change

relative_geometry:
  hand-to-mouth distance
  hand-to-chin distance
  hand-to-eye/forehead distance
  hand-to-shoulder distance
  hand-to-chest distance
  palm/wrist orientation proxy

quality_masks:
  left_hand_detected
  right_hand_detected
  pose_detected
  face_detected
  per-frame missing rate
```

Do not feed all 468 face landmarks into the MVP model by default. Start with a stable face subset. Add full-face features only if ablation proves they help.

## 5. Normalization Rules

Use anchor-based normalization instead of raw pixel coordinates.

Primary anchor:

```text
center = midpoint(left_shoulder, right_shoulder)
scale = distance(left_shoulder, right_shoulder)
```

Fallback anchors:

```text
if shoulders missing:
  use visible hand points + face anchor

if face missing:
  use shoulders + wrists

if too many landmarks missing:
  mark frame as low quality
```

Each valid coordinate should become relative to the body:

```text
x_norm = (x - center_x) / scale
y_norm = (y - center_y) / scale
z_norm = z / scale
```

Keep detection masks separate. Do not hide missing data by silently filling everything with zeros.

## 6. Missing Landmark Repair

Use interpolation only when the gap is short.

```text
short gap: 1-5 frames -> linear interpolation
medium gap: 6-10 frames -> interpolate but mark low confidence
long gap: >10 frames -> keep missing and rely on masks
```

Important:

- Never interpolate across the whole gesture if a hand is absent for most frames.
- Track missing rate per body part.
- Use missing-rate thresholds to reject bad recordings before classification.

## 7. Dynamic Trimming

Before resampling to 64 frames, remove idle frames at the beginning and end.

Motion score candidates:

```text
mean wrist velocity
mean hand center velocity
change in hand-to-chest distance
change in hand-to-face distance
```

Suggested logic:

```text
1. Compute motion score per frame.
2. Smooth motion score.
3. Find first and last frame above threshold.
4. Add a small margin before/after.
5. Resample trimmed segment to 64 frames.
```

This reduces noise from waiting posture and makes the model learn the sign itself.

## 8. Model Architecture

Use a teacher-student setup.

### 8.1 Teacher Model

Purpose: highest accuracy during training and ablation.

```text
Input: [batch, 64 frames, feature_dim + masks]

hand encoder MLP
pose encoder MLP
face encoder MLP
motion encoder MLP
-> concat
-> temporal Conv1D stem
-> Transformer Encoder
-> attention pooling
-> classification head
```

Suggested config:

```text
model_dim: 192
layers: 4
heads: 4
dropout: 0.25-0.35
label_smoothing: 0.05
```

### 8.2 Student Model

Purpose: small, fast exported model.

Option A: Lite Transformer

```text
model_dim: 96 or 128
layers: 2 or 3
heads: 4
dropout: 0.20-0.30
```

Option B: BiGRU

```text
hidden_dim: 96 or 128
layers: 2
dropout: 0.25
```

Recommended MVP deployment:

```text
Use Lite Transformer if accuracy is clearly better.
Use BiGRU if speed/size matters more and accuracy is close.
```

### 8.3 Knowledge Distillation

Train the teacher first, then train a smaller student using both hard labels and teacher logits.

```text
loss = CE(student_logits, label)
     + alpha * KL(student_logits / T, teacher_logits / T)
```

Suggested:

```text
temperature: 2.0-4.0
alpha: 0.3-0.7
```

This can help the small model perform better without increasing deployment size.

## 9. Training Strategy

### 9.1 Class Selection

For the MVP, create clean class subsets:

```text
VSL-MVP-30
VSL-MVP-50
VSL-MVP-100 later
```

Selection criteria:

- Enough clips per class.
- Enough signer diversity.
- Low missing-hand rate.
- Low face/pose tracking failure.
- Avoid highly similar signs in the first demo unless they are intentionally tested.

The current local data has only about 17.7 clips/class on average, so 400-class high-accuracy recognition should be treated as a later research target, not the first product demo.

### 9.2 Splits

Preferred:

```text
signer-holdout validation
```

If the subset becomes too small:

```text
use grouped split by signer where possible
also report random split separately, but do not treat it as real-world accuracy
```

### 9.3 Augmentation

Use landmark-safe augmentations:

```text
time shift
time stretch/compress
random frame dropout
small coordinate noise
small global scale jitter
part dropout for face/pose
hand landmark dropout
```

Use horizontal flip carefully:

- It may help reduce handedness bias.
- It can be harmful if left/right hand identity changes the sign meaning.
- Test it as an ablation, do not enable blindly.

## 10. Evaluation

Track both ML quality and product quality.

ML metrics:

```text
top1 accuracy
top3 accuracy
macro F1
per-class accuracy
confusion matrix
calibration / confidence reliability
```

Product metrics:

```text
model file size
CPU inference latency
MediaPipe extraction latency
end-to-end webcam latency
bad-recording reject rate
unknown/low-confidence behavior
```

Target MVP metrics:

```text
VSL-MVP-30:
  top1 >= 88% on controlled validation
  top3 >= 95%

VSL-MVP-50:
  top1 >= 80-85% on controlled validation
  top3 >= 92%

Deployment:
  exported classifier <= 3 MB after quantization if possible
  classifier inference comfortably real-time on CPU
```

## 11. Ablation Plan

Run these experiments before locking the model:

```text
E1: hands only
E2: hands + upper body
E3: hands + upper body + face subset
E4: hands + upper body + face subset + motion
E5: E4 + missing landmark interpolation
E6: E5 + dynamic trimming
E7: E6 + distillation
E8: E7 + quantization
```

Keep results in a small table:

```text
experiment | classes | top1 | top3 | macro_f1 | size_mb | latency_ms | notes
```

## 12. Export And Compression

Preferred export path:

```text
PyTorch checkpoint
-> ONNX FP32
-> ONNX dynamic quantization INT8
```

Optional mobile/web path:

```text
PyTorch
-> ONNX
-> TensorFlow/TFLite conversion
-> TFLite INT8 or FP16
```

Do not optimize the classifier alone and ignore MediaPipe latency. Benchmark both:

```text
landmark extraction time
classifier inference time
total webcam loop time
```

## 13. Three-Week MVP Plan

### Week 1: Pipeline V2

Deliverables:

```text
LandmarkSchemaV2
Holistic extractor
face subset selection
quality masks
anchor normalization
missing landmark interpolation
dynamic trimming
VSL-MVP-30 and VSL-MVP-50 class files
features_vsl_mvp30_v2.npz
features_vsl_mvp50_v2.npz
```

Tasks:

- Add `FeatureConfigV2`.
- Add `HolisticLandmarkExtractor`.
- Add `landmark_schema.py` for named landmark groups.
- Add feature cache metadata: schema version, selected parts, masks, normalization mode.
- Add dataset selection script for clean MVP subsets.
- Add a small inspection report for missing rates and class counts.

### Week 2: Training And Model Selection

Deliverables:

```text
runs/vsl_mvp30_teacher/
runs/vsl_mvp30_student/
runs/vsl_mvp50_teacher/
runs/vsl_mvp50_student/
ablation_report.md
best ONNX model
```

Tasks:

- Train current GRU baseline on V2 features.
- Train Multi-stream Transformer teacher.
- Train Lite Transformer or BiGRU student.
- Add distillation training.
- Run ablations E1-E7.
- Choose the model for webcam demo.

### Week 3: Demo And Deployment

Deliverables:

```text
webcam demo with V2 extractor
ONNX/TFLite exported classifier
quantized model
latency/size report
README demo instructions
```

Tasks:

- Update `webcam_demo.py` to use V2 features.
- Add top-3 predictions and confidence margin.
- Add reject logic for low hand/face/pose quality.
- Add temporal smoothing across repeated predictions.
- Benchmark on laptop CPU.
- Record known limitations.

## 14. After MVP

Phase 1: improve accuracy

```text
30 classes -> 50 -> 100
add more signers/data
use confusion-matrix-driven class cleanup
improve face subset only if it helps
```

Phase 2: improve robustness

```text
auto start/stop detection
unknown gesture rejection
better calibration
lighting/background tests
```

Phase 3: scale vocabulary

```text
train on all available VSL400 parts
use multi-view if local storage allows
consider RGB crop branch for handshape if landmark-only plateaus
```

Phase 4: productization

```text
learning/practice UI
recording quality guidance
per-class feedback
local-only privacy defaults
packaged desktop or web app
```

## 15. Risk Register

Risk: Face landmarks may be unreliable because VSL400 anonymizes facial regions.

Mitigation:

```text
measure face tracking quality
use face subset only
keep masks
run ablation before relying on face features
```

Risk: 400 classes with only Part_1 + Part_2 will have low real-world accuracy.

Mitigation:

```text
ship MVP on 30-50 classes
keep 400-class training as research track
scale only after more data or better filtering
```

Risk: Random validation split can overstate accuracy.

Mitigation:

```text
prefer signer-holdout
report split method in every run config
```

Risk: Lightweight model loses accuracy.

Mitigation:

```text
teacher-student distillation
quantization after accuracy is stable
compare Lite Transformer vs BiGRU
```

## 16. Source Links

- MediaPipe Holistic Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
- VSL400 dataset: https://zenodo.org/records/17943574
- SPOTER paper: https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf
- MediaPipe keypoint preprocessing for ISLR: https://www.sign-lang.uni-hamburg.de/lrec/pub/24052.html
- MediaPipe landmarks with RNN for dynamic sign recognition: https://www.mdpi.com/2079-9292/11/19/3228
- ASL Fingerspelling 1st place solution: https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution
- Non-manual markers overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC2837852/

## 17. Immediate Next Steps

1. Implement `FeatureConfigV2`.
2. Implement `HolisticLandmarkExtractor`.
3. Create `selected_classes_vsl_mvp30.json` and `selected_classes_vsl_mvp50.json`.
4. Extract V2 features for both subsets.
5. Train baseline GRU and Lite Transformer.
6. Compare ablations and choose the first demo model.
