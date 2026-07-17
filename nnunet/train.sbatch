#!/bin/bash
#SBATCH --job-name=nnunet-train
#SBATCH --partition=ICF-Research
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --output=slurm-logs/%x-%j.out

set -euo pipefail

cd /home/s2036401/Benchmark-ELM-Line-OCT-Dataset

if [ "$#" -ne 3 ]; then # 3 args
    echo "Usage: sbatch nnunet/train.sbatch <dataset_id> <configuration> <fold>" >&2
    echo "  e.g.: sbatch nnunet/train.sbatch 1 2d 0" >&2
    echo "  e.g.: sbatch nnunet/train.sbatch 2 3d_fullres 0" >&2
    exit 1
fi

DATASET=$1
CONFIGURATION=$2
FOLD=$3
PLANS=nnUNetResEncUNetMPlans

echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nnunet
source nnUnet.sh

python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

# Requires nnunet/preprocess.sbatch to have completed first (raw + preprocessed
# + splits_final.json for Dataset001_ELM2D / Dataset002_ELM3D).
#
# One fold per job (see nnunet/submit_folds.sh to launch all 10 folds --
# 2D x5 + 3D x5 -- as separate jobs in parallel across GPUs).
echo "Training dataset=${DATASET} configuration=${CONFIGURATION} fold=${FOLD} plans=${PLANS}"
nnUNetv2_train "${DATASET}" "${CONFIGURATION}" "${FOLD}" -p "${PLANS}"

# After training a fold, evaluate on that fold's held-out test cases (listed
# in nnunet/nnUNet_preprocessed/Dataset00{1,2}_ELM*/test_cases_per_fold.json) with
# nnUNetv2_predict, then score against labelsTr with this repo's own metrics.
