#!/bin/bash
#SBATCH --job-name=nnunet-preprocess
#SBATCH --partition=ICF-Research
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/%x-%j.out

# No --gres here: dataset conversion + nnUNetv2_plan_and_preprocess are
# CPU-only. This cluster's partitions are all GPU nodes (see `sinfo`), so
# if the scheduler here requires an explicit --gres to place a job at all,
# add e.g. `#SBATCH --gres=gpu:0` -- just don't request an actual GPU for
# this step, it would sit idle.

set -euo pipefail # exit on error, unset variable, or pipe failure

cd /home/s2036401/Benchmark-ELM-Line-OCT-Dataset

echo "Running on: $(hostname)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nnunet
source nnUnet.sh

# 1. Convert this repo's data_no_anomalies/ into nnU-Net raw datasets.
python nnunet/prepare_2d.py
python nnunet/prepare_3d.py

# 2. Fingerprint + plan + resample. -np controls parallel worker processes,
# match it to --cpus-per-task above.
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -np 8 -pl nnUNetPlannerResEncM
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity -np 8 -pl nnUNetPlannerResEncM

# 3. Overwrite nnU-Net's auto-generated splits_final.json with this repo's
# own rotating 5-fold split, so results are comparable to the other models.
python nnunet/make_splits.py

echo "Preprocessing done. Inspect nnUNet_preprocessed/Dataset00{1,2}_ELM*/nnUNetPlans.json"
echo "before training -- check the planned 3d config (3d_fullres vs a"
echo "3d_lowres/cascade) and patch size given the placeholder 1x1x1 spacing."

# Command:
# sbatch --gres=gpu:nvidia_h200:1 -p ICF-Research nnunet/preprocess.sbatch; squeue
