#!/bin/bash
#SBATCH --job-name=elm-qualitative-nnunet
#SBATCH --partition=ICF-Research
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --output=slurm-logs/%x-%j.out

set -euo pipefail

cd /home/s2036401/Benchmark-ELM-Line-OCT-Dataset

echo "Running on: $(hostname)"
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nnunet
source nnUnet.sh

# Requires qualitative_baselines.sbatch to have finished first (reads
# qualitative/images/manifest.json for the eye/slice/fold selection). Adds the
# nnU-Net (2d, 3d_fullres) prediction overlays for the same eyes.
python qualitative/qualitative_nnunet.py
