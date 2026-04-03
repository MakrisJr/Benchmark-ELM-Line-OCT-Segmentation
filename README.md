# Benchmark ELM Line OCT Dataset

Benchmarking automated detection of the retinal External Limiting Membrane (ELM) in 3D spectral-domain OCT volumes of full-thickness macular holes.

## Overview

This repository started from the original ELM segmentation benchmark project associated with:

> Singh, V. K., Kucukgoz, B., Murphy, D. C., Xiong, X., Steel, D. H., and Obara, B.  
> "Benchmarking Automated Detection Of The Retinal External Limiting Membrane In A 3D Spectral Domain Optical Coherence Tomography Image Dataset of Full Thickness Macular Holes"

The original codebase provided the core dataset handling, 2D training/inference pipeline, and baseline model implementations for ELM line segmentation.

On top of that original project, this repository has been extended with:

- metadata-driven train/validation fold splits
- 5-fold cross-validation training for both 2D and 3D models
- newer 3D architectures and hybrid 2.5D/3D approaches
- cross-validation evaluation scripts with per-eye and per-slice/per-volume CSV exports
- additional geometric and surface-based metrics such as ASSD, HD95, boundary F1, and surface Dice
- Grad-CAM utilities for model inspection
- utilities for exporting predictions and comparing models

## What Is In This Repo

The codebase now supports two related workflows:

- 2D ELM segmentation from individual OCT slices
- 3D ELM segmentation from full OCT volumes assembled from 49 slices per eye

Repository layout:

- `elm/` contains the reusable package code such as datasets, models, evaluation helpers, losses, and transforms
- top-level scripts such as `train.py`, `new-train.py`, `predict_cv2d.py`, and `predict_cv3d.py` remain as runnable entry points

Common patterns used across the updated code:

- dataset splits are read from `data_no_anomalies/metadata.csv`
- checkpoints are saved fold-by-fold under `elm-results/<model_name>/fold_<k>/checkpoints/`
- evaluation scripts write CSV summaries suitable for later statistical analysis
- reusable Python modules now live in the `elm/` package and are imported as `elm.*`

## Dataset Layout

The newer training and CV scripts expect a structure like:

```text
data_no_anomalies/
  metadata.csv
  all/
    image/
      001-0.png
      001-1.png
      ...
    mask/
      001-0.png
      001-1.png
      ...
```

Expected conventions:

- `metadata.csv` contains at least `patient_id` and `fold`
- patient IDs are treated as zero-padded strings such as `001`
- 2D slices follow the naming pattern `<patient_id>-<slice_idx>.png`
- 3D volumes are reconstructed by stacking slices `0..48` for each eye

## Main Scripts

The repository is now split into:

- `elm/`: reusable package code
- top-level `*.py` scripts: runnable entry points for training, inference, analysis, and utilities

If you are importing code from this project, prefer imports such as:

```python
from elm.dataset import D3Dataset
from elm.model import SwinUNETR3D
from elm.eval import eval_net
```

### Training

- `train.py`: 2D training with cross-validation support for models such as `SegNet`, `U_Net`, `R2U_Net`, `DeepLabv3_plus`, and `SwinEncoderUNet2D`
- `new-train.py`: 3D training with 5-fold cross-validation for `UNet3D`, `UNet3D_Aniso`, `UNet3D_Aniso2`, `UNet3DFrawley`, `UNet2DEnc3DDec`, `CSAM_UNet2p5D`, `UNet2p5D_SlidingWindow`, and `SwinUNETR3D`

### Inference And Evaluation

- `predict.py`: single-run 2D inference/evaluation
- `predict3D.py`: single-run 3D inference/evaluation
- `predict_cv2d.py`: fold-wise cross-validation evaluation for 2D models
- `predict_cv3d.py`: fold-wise cross-validation evaluation for 3D models

### Data And Core Components

- `elm/dataset.py`: 2D and 3D dataset classes, metadata-based split logic, transforms, and volume assembly
- `elm/model.py`: 2D, 2.5D, and 3D segmentation models used throughout the project
- `elm/dice_loss.py`: Dice coefficient and Dice loss
- `elm/eval.py`: validation helpers used during training
- `elm/transformation.py`: legacy and auxiliary transform utilities

### Analysis And Utilities

- `gradCAM_2D.py`: Grad-CAM visualisation for 2D models
- `gradCAM_3D.py`: Grad-CAM style analysis for 3D models
- `stack_to_tif.py`: stack 2D predictions into a 3D multi-page TIF
- `compare_csvs.py` and paired CSV files: model comparison and downstream analysis helpers

## Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
```

The older project notes referenced CUDA 10 and earlier PyTorch versions, but the current repository has evolved beyond that original minimal setup. In practice, you should treat `environment.yml` as the authoritative starting point.

From the repository root, the top-level scripts should work directly because they import from the local `elm` package.

## Quick Start

### 1. Train A 2D Model

```bash
python train.py --model SegNet --epochs 100
```

### 2. Train A 3D Model With Cross-Validation

```bash
python new-train.py --model SwinUNETR3D --epochs 100
```

### 3. Evaluate A 2D Cross-Validation Run

```bash
python predict_cv2d.py \
  --model_root elm-results/SegNet_Apr-01-2026_1639_model \
  --model SegNet
```

### 4. Evaluate A 3D Cross-Validation Run

```bash
python predict_cv3d.py \
  --model_root elm-results/SwinUNETR3D_Apr-02-2026_1646_model \
  --model SwinUNETR3D
```

## Metrics

The updated evaluation scripts report more than Dice alone. Depending on the script, outputs can include:

- Dice
- IoU
- sensitivity
- false positive rate
- RMSE
- ASSD
- Hausdorff distance
- HD95
- boundary F1
- surface Dice

Cross-validation scripts also export CSV summaries such as:

- `cv_fold_summary.csv`
- `cv_per_patient_all_folds.csv`
- `cv_per_slice_all_folds.csv` for 2D evaluation
- `cv_per_volume_all_folds.csv` for 3D evaluation
- `cv_oof_eye_summary.csv`

## Notes

- The 2D workflow operates on individual slices and aggregates results per eye.
- The 3D workflow operates on full reconstructed volumes, where one eye corresponds to one volume.
- Due to the repo being work in progress, some historical scripts in the repository predate the metadata-based CV pipeline. For new experiments, prefer `train.py`, `new-train.py`, `predict_cv2d.py`, and `predict_cv3d.py`.
- The dataset itself is not bundled here and is available on request from the corresponding author for non-commercial use.

## Citation

If you use this repository, please cite the original benchmark paper:

```bibtex
@article{singh2021benchmarking,
  title={Benchmarking Automated Detection Of The Retinal External Limiting Membrane In A 3D Spectral Domain Optical Coherence Tomography Image Dataset of Full Thickness Macular Holes},
  author={Singh, VK and Kucukgoz, B and Murphy, DC and Xiong, X and Steel, DH and Obara, B},
  journal={Computers in Biology and Medicine},
  year={2021},
  publisher={Newcastle University}
}
```
