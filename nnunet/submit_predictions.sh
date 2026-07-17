#!/bin/bash
# Submits nnunet/predict.sbatch once per trained (dataset, configuration)
# combo (dataset 1 2D, dataset 2 2D, dataset 2 3D) as separate sbatch jobs.
# Each job loops all 5 folds internally -- predict + score is much cheaper
# than training, so there's no need to split per-fold jobs like
# nnunet/submit_folds.sh does.
#
# Requires nnunet/submit_folds.sh to have finished training every fold
# (check with `squeue -u $USER` and that every
# nnunet/nnUNet_results/<dataset>/<trainer>__<plans>__<config>/fold_*/checkpoint_final.pth
# exists).
#
# Usage: ./nnunet/submit_predictions.sh

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch --gres=gpu:nvidia_h200:1 -p ICF-Research --job-name="nnunet-predict-2d-ds1" nnunet/predict.sbatch 1 2d
sbatch --gres=gpu:nvidia_h200:1 -p ICF-Research --job-name="nnunet-predict-2d-ds2" nnunet/predict.sbatch 2 2d
sbatch --gres=gpu:nvidia_h200:1 -p ICF-Research --job-name="nnunet-predict-3d-ds2" nnunet/predict.sbatch 2 3d_fullres

squeue -u "$USER"
