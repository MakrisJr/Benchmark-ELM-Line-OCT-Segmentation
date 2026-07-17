#!/bin/bash
#SBATCH --job-name=elm-qualitative-baselines
#SBATCH --partition=ICF-Research
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --output=slurm-logs/%x-%j.out

set -euo pipefail

cd /home/s2036401/Benchmark-ELM-Line-OCT-Dataset

echo "Running on: $(hostname)"
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate elm-gpu

# Saves native-resolution image/GT/prediction overlays for the 6 baseline
# architectures, for 2 representative eyes per category (worst/median/best
# nnU-Net Dice, 6 eyes total), into qualitative/images/. Follow with
# qualitative_nnunet.sbatch (nnunet env) and then make_qualitative_grid.py to
# add the nnU-Net panels and stitch everything into one comparison figure per eye.
python qualitative/qualitative_baselines.py --n_per_category 2
