#!/bin/bash
# Submits nnunet/train.sbatch once per fold (and per config: dataset 1 2D,
# dataset 2 2D, dataset 2 3D) as separate sbatch jobs, so all 15 folds train
# in parallel on different GPUs instead of sequentially inside one job.
#
# Requires nnunet/preprocess.sbatch to have completed first (raw + preprocessed
# + splits_final.json for Dataset001_ELM2D / Dataset002_ELM3D), and that you've
# checked nnunet/nnUNet_preprocessed/Dataset002_ELM3D/nnUNetResEncUNetMPlans.json to confirm
# nnU-Net planned 3d_fullres (not a 3d_lowres/3d_cascade_fullres pair) --
# adjust the CONFIGURATION below if it did.
#
# Usage: ./nnunet/submit_folds.sh

set -euo pipefail
cd "$(dirname "$0")/.."

for FOLD in 0 1 2 3 4; do
    sbatch --gres=gpu:nvidia_h200:1 -p ICF-Research --job-name="nnunet-2d-ds1-f${FOLD}" nnunet/train.sbatch 1 2d "${FOLD}"
done

for FOLD in 0 1 2 3 4; do
    sbatch --gres=gpu:nvidia_h200:1 -p ICF-Research --job-name="nnunet-2d-ds2-f${FOLD}" nnunet/train.sbatch 2 2d "${FOLD}"
done

for FOLD in 0 1 2 3 4; do
    sbatch --gres=gpu:nvidia_h200:1 -p ICF-Research --job-name="nnunet-3d-f${FOLD}" nnunet/train.sbatch 2 3d_fullres "${FOLD}"
done

squeue -u "$USER"
