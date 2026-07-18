# Benchmark ELM Line OCT Dataset

Benchmarking automated detection of the retinal External Limiting Membrane (ELM) in 3D spectral-domain OCT volumes of full-thickness macular holes.

## Overview

This repository started from the original ELM segmentation benchmark project associated with:

> Singh, V. K., Kucukgoz, B., Murphy, D. C., Xiong, X., Steel, D. H., and Obara, B.  
> "Benchmarking Automated Detection Of The Retinal External Limiting Membrane In A 3D Spectral Domain Optical Coherence Tomography Image Dataset of Full Thickness Macular Holes"

The original codebase provided the core dataset handling, 2D training/inference pipeline, and baseline model implementations for ELM line segmentation.

On top of that original project, this repository has been extended with:

- metadata-driven train/validation/test fold splits, rebuilt from a cleaned, anomaly-filtered dataset (`data_no_anomalies/`, `remove-anomalies.py`, `patient_anomalies.csv`)
- 5-fold cross-validation training for both 2D and 3D models, plus a 2.5D sliding-window model (`UNet2p5D_SlidingWindow`) and a cross-slice-attention model (`CSAM_UNet2p5D`)
- newer 3D architectures and hybrid 2.5D/3D approaches, including a SwinUNETR3D model with optional pretrained encoder weights
- an optional clDice loss term (topology-aware, skeleton-based) alongside BCE/Dice for both 2D and 3D training
- a parallel nnU-Net v2 baseline pipeline (`nnunet/`) using this repo's own CV folds, so it is directly comparable to the custom architectures
- cross-validation evaluation scripts with per-eye and per-slice/per-volume CSV exports, including a native-resolution scoring mode (`--native_res`) so all models can be compared at their original per-slice resolution
- additional geometric and surface-based metrics such as ASSD, HD95, boundary F1, and surface Dice
- macular-hole gap-decomposition metrics (`elm/hole_metrics.py`, `--hole_decomposition`) that score how well each model's predicted ELM line reproduces the annotated hole gap, plus aggregation scripts (`aggregate_native_results.py`, `aggregate_hole_results.py`) that roll per-model CV CSVs into headline comparison tables
- qualitative side-by-side comparison figures across every model (`qualitative/`)
- Grad-CAM utilities for model inspection
- utilities for exporting predictions and comparing models

## What Is In This Repo

The codebase now supports two related workflows:

- 2D ELM segmentation from individual OCT slices
- 3D ELM segmentation from full OCT volumes assembled from 49 slices per eye

Common patterns used across the updated code:

- dataset splits are read from `data_no_anomalies/metadata.csv`
- checkpoints are saved fold-by-fold under `elm-results/<model_name>/fold_<k>/checkpoints/`
- evaluation scripts write CSV summaries suitable for later statistical analysis
- reusable Python modules live in the `elm/` package and are imported as `elm.*`

## Repository Layout

- `elm/` -- reusable package code: dataset classes and CV split logic (`dataset.py`), model architectures including the CSAM cross-slice attention model (`model.py`, `csam.py`), losses including Dice and clDice (`dice_loss.py`), shared geometric/surface metrics (`metrics.py`), macular-hole gap-decomposition metrics (`hole_metrics.py`), and validation helpers (`eval.py`)
- `nnunet/` -- a self-contained nnU-Net v2 baseline pipeline (dataset conversion, preprocessing, training, and CV evaluation scripts) that reuses this repo's own fold splits; see [nnunet/README.md](nnunet/README.md)
- `qualitative/` -- generates side-by-side qualitative comparison figures across all baseline models and both nnU-Net configs for representative worst/median/best eyes; see [qualitative/README.md](qualitative/README.md)
- `data_no_anomalies/` -- the cleaned dataset used by the current pipeline: `metadata.csv` (patient IDs, base `split`, and 5 rotating `split_fold0..split_fold4` columns) plus `all/`, `train/`, `val/`, `test/` image/mask folders
- `data_original/` -- the original, unfiltered image/mask dataset before anomaly removal
- `data/` -- working copy of the dataset used by `run.sh` when staging a training run to fast local scratch disk
- `elm-results/` -- per-model training outputs: fold checkpoints, TensorBoard logs, and CV result CSVs (gitignored)
- `csvs/` and `figures/` -- exported per-patient result CSVs and summary comparison plots (boxplots, scatter plots) across models
- `checkpoint/` -- pretrained weights consumed at inference/training time (e.g. `model_swinvit_UNETR.pt`, the pretrained SwinViT encoder for `SwinUNETR3D`)
- `result/`, `eval-3d-outputs/`, `eval-window-outputs/` -- saved per-slice prediction outputs from `predict.py`, `predict3D.py`, and `predict3Dwindow.py` respectively (gitignored)
- `slurm-logs/` -- stdout/stderr logs from sbatch jobs (gitignored)

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

- `metadata.csv` contains `patient_id`, a base `split` column, and 5 rotating CV columns `split_fold0..split_fold4` (each a per-fold `train`/`val`/`test` label), generated by `make_cv_splits.py` with a 70/10/20 split where the test 20% rotates across folds
- patient IDs are treated as zero-padded strings such as `001`
- 2D slices follow the naming pattern `<patient_id>-<slice_idx>.png`
- 3D volumes are reconstructed by stacking slices `0..48` for each eye
- `data_no_anomalies/` was derived from `data_original/` by dropping patients flagged as annotation anomalies in `patient_anomalies.csv` via `remove-anomalies.py`

## Main Scripts

The repository is split into:

- `elm/`: reusable package code
- top-level `*.py` scripts: runnable entry points for training, inference, analysis, and utilities

If you are importing code from this project, prefer imports such as:

```python
from elm.dataset import D3Dataset
from elm.model import SwinUNETR3D
from elm.eval import eval_net
```

### Training

- `train2D.py`: 2D training with cross-validation support for models such as `SegNet`, `U_Net`, `AttU_Net`, `LinkNetImprove`, `U2NETP`, `R2U_Net`, `DeepLabv3_plus`, `FCN`, and `SwinEncoderUNet2D`, with an optional clDice loss term (`--cldice-weight`)
- `train3D.py`: 3D training with 5-fold cross-validation for `UNet3D`, `UNet3D_Aniso`, `UNet3D_Aniso2`, `UNet3DFrawley`, `UNet2DEnc3DDec`, `CSAM_UNet2p5D`, `UNet2p5D_SlidingWindow`, and `SwinUNETR3D`, also with an optional clDice loss term
- `train_cldice2d.sbatch` / `train_cldice3d.sbatch`: sbatch wrappers for the clDice pilot runs (`R2U_Net`, `SegNet`, `SwinEncoderUNet2D` for 2D)
- `train.sbatch`: general sbatch wrapper for 2D/3D training jobs
- `train-windows.py`: legacy single-run (non-CV) sliding-window 3D training script, predating the metadata-based CV pipeline
- `run.sh`: stages the dataset to fast local scratch disk before training and syncs `elm-results/` back afterwards

### Inference And Evaluation

- `predict.py`: single-run 2D inference/evaluation
- `predict3D.py`: single-run 3D inference/evaluation
- `predict3Dwindow.py`: legacy single-run sliding-window 3D inference script
- `predict_cv2d.py`: fold-wise cross-validation evaluation for 2D models, with optional `--native_res` (score at native per-slice resolution) and `--hole_decomposition` (macular-hole gap metrics) modes
- `predict_cv3d.py`: fold-wise cross-validation evaluation for 3D models, with the same `--native_res` and `--hole_decomposition` options
- `native_eval.sbatch`: re-scores each baseline's existing CV checkpoints with `--native_res`, for apples-to-apples comparison against nnU-Net
- `hole_eval.sbatch`: re-scores each baseline's checkpoints with `--native_res --hole_decomposition` for the macular-hole gap analysis

### Data And Core Components

- `elm/dataset.py`: 2D and 3D dataset classes, metadata-based split logic, transforms, and volume assembly
- `elm/model.py`: 2D, 2.5D, and 3D segmentation models used throughout the project
- `elm/csam.py`: `CSAM_UNet2p5D`'s cross-slice attention module (semantic/positional/slice branches with an uncertainty-aware slice branch)
- `elm/dice_loss.py`: Dice coefficient/loss and the clDice (soft-skeleton) loss
- `elm/metrics.py`: shared confusion-matrix, Dice/IoU/RMSE, and surface-distance metrics (ASSD, HD95, boundary F1, surface Dice) for 2D and 3D
- `elm/hole_metrics.py`: per-slice macular-hole gap comparison (predicted gap vs. annotated gap geometry, spurious-gap detection)
- `elm/eval.py`: validation helpers used during training
- `elm/transformation.py`: legacy and auxiliary transform utilities
- `make_cv_splits.py`: generates the 5-fold 70/10/20 `split_fold0..split_fold4` columns in `metadata.csv`
- `remove-anomalies.py`: drops patients listed in `patient_anomalies.csv` from `data_no_anomalies/`

### Analysis And Utilities

- `gradCAM_2D.py`: Grad-CAM visualisation for 2D models
- `gradCAM_3D.py`: Grad-CAM style analysis for 3D models
- `stack_to_tif.py`: stack 2D predictions into a 3D multi-page TIF
- `compare_csvs.py` and paired CSV files: paired statistical comparison between two models' per-patient CV results
- `aggregate_native_results.py`: collects each model's native-resolution CV results (baselines + both nnU-Net configs) into one comparison table
- `aggregate_hole_results.py`: turns each model's `--hole_decomposition` CV results into headline macular-hole gap-accuracy numbers, in both directions (hole-crossing slices and spurious-gap slices)

## nnU-Net Baseline

`nnunet/` converts `data_no_anomalies/` into nnU-Net v2 raw datasets (one 2D slice dataset, one 3D per-eye volume dataset) and overwrites nnU-Net's own CV splits with this repo's `split_fold0..split_fold4`, so the nnU-Net baseline is trained and evaluated on exactly the same folds as every other model here. See [nnunet/README.md](nnunet/README.md) for the full conversion/train/predict pipeline and the auto-generated network/loss/augmentation configuration.

## Qualitative Comparison Figures

`qualitative/` generates side-by-side figures comparing every baseline architecture plus both nnU-Net configs on a spread of worst/median/best eyes (picked by nnU-Net 2D per-eye Dice as a stable, independently-computed reference). See [qualitative/README.md](qualitative/README.md) for the 3-step pipeline (`qualitative_baselines.py` -> `qualitative_nnunet.py` -> `make_qualitative_grid.py`).

## Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
```

`current_requirements.txt` is a pip-freeze snapshot of the environment as actually installed on the training machine; treat `environment.yml` as the primary spec and `current_requirements.txt` as a reference for pinning exact versions if `environment.yml` alone doesn't reproduce the environment. The nnU-Net pipeline uses a separate `nnunet` conda environment (Python 3.10, since `nnunetv2` needs a newer Python than the `elm-gpu` env's 3.9 pin) -- see [nnunet/README.md](nnunet/README.md).

From the repository root, the top-level scripts should work directly because they import from the local `elm` package.

## Quick Start

### 1. Train A 2D Model

```bash
python train2D.py --model SegNet --epochs 100
```

### 2. Train A 3D Model With Cross-Validation

```bash
python train3D.py --model SwinUNETR3D --epochs 100
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
- macular-hole gap accuracy (width/margin error and spurious-gap rate, via `--hole_decomposition`)

Cross-validation scripts also export CSV summaries such as:

- `cv_fold_summary.csv`
- `cv_per_patient_all_folds.csv`
- `cv_per_slice_all_folds.csv` for 2D evaluation
- `cv_per_volume_all_folds.csv` for 3D evaluation
- `cv_oof_eye_summary.csv`

With `--native_res` these land under a separate `cv_eval_native`/`cv_eval_3d_native` output directory, and with `--hole_decomposition` under `cv_eval_hole`/`cv_eval_3d_hole`, so they don't overwrite the default 256x256 CV outputs.

## Notes

- The 2D workflow operates on individual slices and aggregates results per eye.
- The 3D workflow operates on full reconstructed volumes, where one eye corresponds to one volume.
- Due to the repo being work in progress, some historical scripts in the repository predate the metadata-based CV pipeline (`train-windows.py`, `predict3Dwindow.py`). For new experiments, prefer `train2D.py`, `train3D.py`, `predict_cv2d.py`, and `predict_cv3d.py`.
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
