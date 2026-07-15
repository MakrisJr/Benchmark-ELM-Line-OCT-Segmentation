#!/bin/bash
#SBATCH --job-name=nnunet-predict
#SBATCH --partition=ICF-Research
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --output=slurm-logs/%x-%j.out

set -euo pipefail

cd /home/s2036401/Benchmark-ELM-Line-OCT-Dataset

if [ "$#" -ne 2 ]; then # 2 args
    echo "Usage: sbatch nnunet/predict.sbatch <dataset_id> <configuration>" >&2
    echo "  e.g.: sbatch nnunet/predict.sbatch 1 2d" >&2
    echo "  e.g.: sbatch nnunet/predict.sbatch 2 3d_fullres" >&2
    exit 1
fi

DATASET=$1
CONFIGURATION=$2
PLANS=nnUNetResEncUNetMPlans

echo "Running on: $(hostname)"
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nnunet
source nnUnet.sh

# Requires nnunet/submit_folds.sh to have finished (checkpoint_final.pth
# present for every fold of this dataset/configuration).
#
# Predicts + scores all 5 folds in one job (see nnunet/submit_predictions.sh
# to launch all 3 trained dataset/configuration combos in parallel).
echo "Predicting + scoring dataset=${DATASET} configuration=${CONFIGURATION} plans=${PLANS} (all 5 folds)"
python nnunet/predict_cv.py --dataset_id "${DATASET}" --configuration "${CONFIGURATION}" --plans "${PLANS}"
