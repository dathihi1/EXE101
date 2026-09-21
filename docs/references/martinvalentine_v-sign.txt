# Visual Sign Language Recognition Model

This document details the architecture and usage instructions for the visual sign language recognition model.

## Prerequisites

- This project is implemented in Pytorch (it should be >=1.13 to be compatible with ctcdecode, or there may be errors) (**Install First**)

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode. (ctcdecode is only supported on the Linux platform.)
  Install to `./third_party` folder

- [Optional] sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link to the sclite: 
  ```bash
  cd /vsign
  mkdir ./third_party
  cd ./third_party && mkdir ./sofware
  ln -s PATH_TO_KALDI/tools/SCTK/bin/sclite ./software/sclite
  ```

   You may use the Python version evaluation tool for convenience (by setting 'evaluate_tool' as 'python' in line 16 of ./configs/baseline.yaml), but sclite can provide more detailed statistics.

- You can install other required modules by conducting 
   ```bash
  pip install -r requirements.txt
   ```
  or you can install the environment I'm currently working on:
    ```bash
    conda env create -f environment.yml (**will update later**)
    ```
  This will create a new conda env.

- You also need to install `src/vsign/` as a package. You can do this by running:
  ```bash
  pip install -e .
  ```

## Model Architecture

The model processes video frames through a series of layers to generate sign language predictions.

```text
Input: Video Frames [Batch x Time x Channels x Height x Width]
            │
            ▼
   ┌──────────────────────────────┐
   │     2D CNN Backbone (ResNet) │
   │    - Extracts frame features │
   │    - FC layer removed        │
   └──────────────────────────────┘
            │
            ▼
   Framewise Features [Batch x Channels x Time]
            │
            ▼
   ┌─────────────────────────────────────────┐
   │         TemporalConv Layer              │
   │ - 1D temporal convolutions              │
   │ - Input size: 512                       │
   │ - Hidden size: configurable (e.g. 1024) │
   └─────────────────────────────────────────┘
            │
            ├──────────────► conv_logits (used in ConvCTC loss and conv_sents)
            │
            ▼
   visual_feat [Time x Batch x Hidden]
            │
            ▼
   ┌─────────────────────────────────────────┐
   │         BiLSTM Temporal Layer           │
   │ - 2 layers                              │
   │ - Bidirectional                         │
   │ - Input/output: [Time x Batch x Hidden] │
   └─────────────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────┐
   │      Classification Layer    │
   │ - Linear or NormLinear       │
   │ - Output: num_classes logits │
   └──────────────────────────────┘
            │
            ├──────────────► sequence_logits (used in SeqCTC loss and recognized_sents)
            ▼
   ┌──────────────────────┐
   │   Beam Decoder       │
   │ - Decodes predictions│
   └──────────────────────┘
```
*Flow Diagram Simplified:*
```text
Video Frames
     │
     ▼
ResNet Backbone (Spatial Feat)
     │
     ▼
TemporalConv (Temporal Conv Feat)
     │
     ▼
BiLSTM (Temporal Context)
     │
     ▼
Classification (Predictions)
     │
     ▼
  Logits
```

## Preprocessing

**Important:** [vsl_preprocess.py](src/vsign/data/vsl_preprocess.py), [extract_frames.py](src/vsign/data/extract_frames.py) before running the following steps.

1.  **Extract Frames and Annotations:**
    ```bash
    python extract_frames.py
    ```

2.  **Extract Features:** (Resizes images, creates gloss dictionary, info file, and groundtruth)
    ```bash
    python src/vsign/data/vsl_preprocess.py --process-image --multiprocessing
    ```
    Remember to change parameters inside the file `vsl_preprocess.py`

3.  **Combine both steps:**
    ```bash
      python /home/kafka/Desktop/v-sign/src/vsign/data/vsl_preprocess_new.py --extract-frames --video-root /home/kafka/Desktop/v-sign/data/raw/VSL_V2 --dataset-root /home/kafka/Desktop/v-sign/data/interim/256x256px/VSL_V2 --processed-feature-root /home/kafka/Desktop/v-sign/data/processed/VSL_V2 --multiprocessing 
    ``` 

## Training

Run the training script using the baseline configuration.

```bash
python run_baseline.py \
  --config configs/baseline.yaml \
  --device 0
```

## Evaluation:
Evaluate the model's performance on the test set using saved weights.
```bash
python run_baseline.py \
  --config configs/baseline.yaml \
  --device 0 \
  --load-weights outputs/logs/baseline_res18/_best_model.pt \
  --phase test
```

## Demo:
Run the inference demo script with a trained model.
```bash
python src/vsign/inference/demo.py  \
   --model_path outputs/logs/baseline_res18/_best_model.pt  \
   --device 0 \
   --dict_path data/processed/VSL_V2/gloss_dict.npy
```

## Citation

```text
@inproceedings{hu2023continuous,
  title={Continuous Sign Language Recognition with Correlation Network},
  author={Hu, Lianyu and Gao, Liqing and Liu, Zekang and Feng, Wei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023},
}
```
