# VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing

A research-grade dataset and reproducible engineering pipeline for Vietnamese Sign Language Processing (SLP), with dual-track evaluation for retrieval and generation settings.

Language: **English** | [Tiếng Việt](README.vi.md)

---

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Problem Statement and Research Motivation](#2-problem-statement-and-research-motivation)
- [3. Contributions](#3-contributions)
- [4. Dataset Access and Scope](#4-dataset-access-and-scope)
- [5. Data Schema and Storage Layout](#5-data-schema-and-storage-layout)
- [6. System Architecture and Pipeline Design](#6-system-architecture-and-pipeline-design)
- [7. Environment and Installation](#7-environment-and-installation)
- [8. End-to-End Reproduction Guide](#8-end-to-end-reproduction-guide)
- [9. Technical Documentation (Modules and Scripts)](#9-technical-documentation-modules-and-scripts)
- [10. Evaluation Protocol and Metrics](#10-evaluation-protocol-and-metrics)
- [11. Available Results and Artifacts](#11-available-results-and-artifacts)
- [12. Reproducibility Checklist](#12-reproducibility-checklist)
- [13. Troubleshooting](#13-troubleshooting)
- [14. Limitations and Ethics](#14-limitations-and-ethics)
- [15. Citation](#15-citation)

---

## 1. Abstract

This repository presents VSWD, a Vietnamese weather-domain sign language dataset and a reproducible processing/evaluation toolkit for SLP research. The project is designed as a full scientific workflow rather than a single model implementation. It includes scene-level extraction from raw videos, metadata alignment, keypoint preparation, quality analysis, deterministic split generation, and benchmark-ready reporting artifacts.

The release supports two complementary evaluation perspectives:

1. **Retrieval-oriented SLP**: semantic alignment between text queries and sign/pose clips.
2. **Production-oriented SLP**: generation quality and motion-distribution plausibility for text-to-pose workflows.

Together, these tracks provide evidence of both semantic discriminability and generative learnability for the dataset.

---

## 2. Problem Statement and Research Motivation

Vietnamese SLP research often faces three bottlenecks:

- limited publicly usable domain datasets,
- fragmented preprocessing pipelines that are hard to reproduce,
- missing standardized reporting artifacts for thesis/paper integration.

VSWD addresses these issues by combining a practical data-building pipeline with report-grade outputs, enabling researchers to move from raw media to reproducible benchmark tables in one repository.

---

## 3. Contributions

1. **Dataset construction pipeline** for weather sign scenes from raw broadcast videos.
2. **Metadata-centric processing design** to keep all downstream steps auditable and reproducible.
3. **Technical scripts for quality/statistics/splits/keypoint preparation** with explicit I/O conventions.
4. **Paper-ready benchmark outputs** in both retrieval and production tracks.
5. **Integrated docs + artifacts** suitable for thesis and research reporting.

---

## 4. Dataset Access and Scope

### 4.1 Dataset reference link

- VSWD dataset folder (Google Drive):
  - https://drive.google.com/drive/folders/1c45THLWH5vTPxlbrOdU4tlgACAi4rRFG?usp=sharing

### 4.2 Expected repository data root

All scripts assume data is mounted under:

- `SignWeather/data/`

### 4.3 Recommended minimum content

- raw videos
- scene videos (original/pose)
- scene keypoints
- scene metadata CSV files
- train/val/test list files

---

## 5. Data Schema and Storage Layout

## 5.1 Directory structure

```text
SignWeather/
├── classifier_ends/
├── stats_and_eval/
├── docs/
├── results/
└── data/
    ├── raw_videos/
    ├── scene_videos_orginal/
    ├── scene_videos_pose/
    ├── scene_keypoints/
    ├── metadata/
    └── lists/
```

## 5.2 Core metadata files

- `data/metadata/scene_metadata.csv` (canonical source used by most scripts)
- `data/metadata/vswd_final_filtered.csv` (filtered metadata snapshot)
- `data/metadata/vswd_final_split.csv` (split-aware metadata)

## 5.3 Typical metadata columns

| Column | Description |
|---|---|
| `path` | Relative path to scene clip |
| `text` | Vietnamese sentence/transcript |
| `quality_level` | Quality tag (e.g., HIGH, MEDIUM) |
| `content_label` | Content category (e.g., WEATHER_CORE, WEATHER_SUPPORT) |
| `split` | Data split (train/val/test), if available |

---

## 6. System Architecture and Pipeline Design

The processing design follows a metadata-first pipeline:

1. **Scene inference and segmentation** from raw videos.
2. **Scene extraction and mapping synchronization**.
3. **Metadata refinement and quality annotation**.
4. **Pose/keypoint preparation** for training/evaluation compatibility.
5. **Statistics and split generation** for reproducible experiments.
6. **Benchmark/report export** for publication-ready outputs.

This architecture minimizes hidden state by encoding workflow state in CSV/manifest artifacts.

---

## 7. Environment and Installation

## 7.1 Prerequisites

- Linux (recommended)
- Python 3.8+
- FFmpeg available in shell

Install FFmpeg on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## 7.2 Clone + Python environment

```bash
git clone https://github.com/derricky2004/VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP.git
cd VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP/SignWeather
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Conda alternative:

```bash
conda create -n vswd python=3.10 -y
conda activate vswd
pip install -r requirements.txt
```

## 7.3 Optional API configuration

Some quality workflows may use OpenAI-based processing. If needed:

```bash
export OPENAI_API_KEY="<your_key>"
```

---

## 8. End-to-End Reproduction Guide

This section provides a full run sequence from data placement to report artifacts.

## 8.1 Prepare directory skeleton

```bash
mkdir -p data/raw_videos \
         data/scene_videos_orginal \
         data/scene_videos_pose \
         data/scene_keypoints \
         data/metadata \
         data/lists
```

## 8.2 Populate data from Drive

Download dataset files from the Drive link and place them into corresponding folders under `data/`.

## 8.3 Run scene-level pipeline

```bash
python classifier_ends/run_full_pipeline.py
```

## 8.4 Rebuild scene pose overlays (if required)

```bash
python rebuild_and_add_pose.py
```

## 8.5 Build train/val/test split

```bash
python stats_and_eval/stats/train_val_split.py \
  --metadata scene_metadata.csv \
  --output vswd_final_split.csv \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --random-seed 42
```

Outputs:

- `data/lists/train.csv`, `val.csv`, `test.csv`
- `data/lists/train_paths.txt`, `val_paths.txt`, `test_paths.txt`
- `data/metadata/vswd_final_split.csv`

## 8.6 Prepare keypoints for evaluation-ready folders

```bash
python stats_and_eval/stats/prepare_eval.py \
  --metadata data/metadata/scene_metadata.csv \
  --scene_keypoints data/scene_keypoints \
  --scene_videos data/scene_videos_pose \
  --output data/keypoints_processed
```

## 8.7 Compute dataset statistics

```bash
python stats_and_eval/stats/data_stats.py
```

Output:

- `docs/dataset_full_stats.md`

## 8.8 Compute transcript quality metrics (optional)

```bash
python stats_and_eval/stats/quality_metrics.py
```

## 8.9 Plot distribution figures for reporting

```bash
python stats_and_eval/stats/plot_data_distribution.py
```

---

## 9. Technical Documentation (Modules and Scripts)

## 9.1 `classifier_ends/`

| Script | Purpose | Primary Input | Primary Output |
|---|---|---|---|
| `run_full_pipeline.py` | Main scene processing workflow | raw videos + metadata | scene metadata updates and scene clips |
| `refine_scenes.py` | Scene refinement utilities | scene metadata/clips | refined scene metadata |
| `crop_scale_scenes.py` | Crop/scale processing for scenes | scene clips | normalized clips |
| `sort_metadata.py` | Metadata sorting/cleanup | metadata CSV | sorted metadata CSV |
| `sync_mapping.py` | Mapping consistency between IDs and files | mapping CSV + metadata | synchronized metadata |
| `visualize_inference.py` | Quick visual check for inference stage | scene frames/clips | visualization outputs |

## 9.2 `stats_and_eval/stats/`

| Script | Purpose | Key Arguments | Output |
|---|---|---|---|
| `train_val_split.py` | Stratified split generation | `--metadata`, ratios, seed | split CSVs + path lists |
| `prepare_eval.py` | Convert/match keypoints into split folders | `--metadata`, `--scene_keypoints`, `--output` | split mapping CSVs + `.npy` files |
| `data_stats.py` | Aggregate dataset-level statistics | built-in metadata path | `docs/dataset_full_stats.md` |
| `quality_metrics.py` | Transcript quality scoring/report | optional API key | `docs/quality_report.md` |
| `plot_data_distribution.py` | Distribution charts for report | built-in metadata path | report figures |

## 9.3 `results/`

| Folder | Content |
|---|---|
| `results/final_reports/` | final integrated markdown/CSV reports + figures |
| `results/retrieval_parallel_methods/` | retrieval model reports and CSV outputs |
| `results/production_parallel_methods/` | production benchmark summaries and stats |

---

## 10. Evaluation Protocol and Metrics

### Retrieval track (discriminative)

- Recall@1, Recall@5, Recall@10
- mAP@10
- BLEU
- WER
- DTW
- Approximate FID

### Production track (generative)

- Accuracy (train/val-test)
- FID (train/test)
- DTW norm
- Diversity and seed-level stability statistics

### Statistical aggregation

For repeated runs $x_1,\dots,x_n$:

$$
\mu = \frac{1}{n}\sum_{i=1}^{n}x_i, \qquad
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2, \qquad
\sigma = \sqrt{\sigma^2}
$$

---

## 11. Available Results and Artifacts

Final integrated report files currently include:

- `results/final_reports/final_experiment_report_retrieval_production_2026-03-05.md`
- `results/final_reports/final_benchmark_table_2026-03-05.csv`
- `results/production_parallel_methods/cross_model_benchmark_clean_2026-03-05.csv`
- `results/production_parallel_methods/cross_model_benchmark_stats_2026-03-05.csv`
- `results/production_parallel_methods/wsigngen_seed_stats_2026-03-05.csv`

These artifacts are ready for thesis tables and paper appendices.

---

## 12. Reproducibility Checklist

- Use a fixed split seed (default: 42).
- Preserve split ratios for official benchmark replication.
- Keep metadata version and script version in sync.
- Record package versions and environment details for each run.
- Save result files with timestamps under `results/`.

---

## 13. Troubleshooting

### FFmpeg not found

- Install FFmpeg and ensure it is on PATH.
- Verify with: `ffmpeg -version`

### Missing keypoints in `prepare_eval.py`

- Check filenames in `data/scene_keypoints/` and metadata `path` consistency.
- Confirm scene basename patterns are aligned.

### Split script ratio errors

- Ensure `train_ratio + val_ratio + test_ratio = 1.0`.

### OpenAI-related quality scoring issues

- Set `OPENAI_API_KEY` in your environment.
- If optional dependencies are missing, install required packages and rerun.

---

## 14. Limitations and Ethics

- Dataset domain is weather-focused; cross-domain generalization requires separate validation.
- Data quality varies by source segment and ASR/transcript reliability.
- Respect media rights, privacy constraints, and local institutional policies in downstream use.
- For high-stakes deployment, additional governance and independent evaluation are mandatory.

---

## 15. Citation

```bibtex
@misc{vswd2026,
  title={VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing},
  author={Quoc Dat Do},
  year={2026},
  howpublished={\url{https://github.com/derricky2004/VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP}}
}
```
