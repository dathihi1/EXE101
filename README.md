# VSL MVP Sign Recognition

MVP training and demo pipeline for Vietnamese Sign Language isolated word recognition using VSL400 front-view videos.

The project is intentionally small and practical for a 2-week demo:

1. Scan VSL400 metadata/videos.
2. Select 10-15 stable classes.
3. Extract MediaPipe hand + upper-body pose landmarks.
4. Train a GRU baseline or a small Temporal Transformer.
5. Export ONNX.
6. Run a webcam demo with manual start/stop gesture recording.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Google Colab, upload this repo/folder and run:

```bash
pip install -r requirements.txt
```

## Expected VSL400 Layout

Download VSL400 from Zenodo and extract at least the front-view files:

```text
data/
  raw/
    VSL400/
      front_view/
        ...
      *.json
```

The scanner is permissive: it recursively searches for video files and JSON metadata, then matches records by six-digit clip IDs or filename stems.

Dataset source: https://zenodo.org/records/17943574  
License: CC BY 4.0. Keep attribution in any product or demo.

## Training Pipeline

```powershell
# 1. Build a manifest from extracted VSL400 files
python -m vsl_mvp.inspect_dataset --dataset-root data/raw/VSL400 --view front_view --out data/processed/manifest.csv

# 2. Choose 15 candidate classes by sample count
python -m vsl_mvp.select_classes --manifest data/processed/manifest.csv --num-classes 15 --out data/processed/selected_classes.json

# 3. Extract landmarks and cache training tensors
python -m vsl_mvp.extract_features --manifest data/processed/manifest.csv --classes data/processed/selected_classes.json --out data/processed/features_vsl15.npz

# 4. Train baseline or transformer
python -m vsl_mvp.train --features data/processed/features_vsl15.npz --model transformer --out-dir runs/vsl15_transformer

# 5. Export best PyTorch checkpoint to ONNX
python -m vsl_mvp.export_onnx --run-dir runs/vsl15_transformer --out runs/vsl15_transformer/model.onnx

# 6. Run webcam demo
python -m vsl_mvp.webcam_demo --model runs/vsl15_transformer/model.onnx --config runs/vsl15_transformer/config.json --labels runs/vsl15_transformer/labels.json
```

## Webcam Controls

- `Space`: start/stop recording one gesture.
- `R`: reset current recording.
- `Q`: quit.

The MVP uses manual segmentation so the model is judged on recognition, not on brittle auto-start/auto-stop logic.

## Product Notes

- Process camera frames locally by default.
- Do not store or upload user videos/landmarks without explicit consent.
- Present results as an AI learning/practice aid, not an official interpreter.
- Keep VSL400 and open-source attribution in the product.
