#!/bin/bash
#SBATCH --job-name=nnunet-hole-eval
#SBATCH --partition=ICF-Research
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --output=slurm-logs/%x-%j.out

set -euo pipefail

cd /home/s2036401/Benchmark-ELM-Line-OCT-Dataset

if [ "$#" -ne 2 ]; then # 2 args
    echo "Usage: sbatch nnunet/hole_eval.sbatch <dataset_id> <configuration>" >&2
    echo "  e.g.: sbatch nnunet/hole_eval.sbatch 1 2d" >&2
    echo "  e.g.: sbatch nnunet/hole_eval.sbatch 2 3d_fullres" >&2
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

# Same as nnunet/predict.sbatch, but with --hole_decomposition: every eye in
# this dataset has a macular hole, but it only crosses some of an eye's
# slices, so on those slices this compares the model's own predicted gap in
# the ELM line against the GT gap -- whether it bridged straight across (no
# gap at all) and, if not, how its width and margins (ELM termination
# points) compare -- so the "models struggle with the hole boundaries"
# pattern from the qualitative figure can be quantified across all 99 eyes.
# Writes to <model_root>/cv_eval_hole (predict_cv.py's default out_dir when
# --hole_decomposition is set), so the canonical cv_eval/ numbers used
# elsewhere are untouched.
echo "Predicting + scoring (hole decomposition) dataset=${DATASET} configuration=${CONFIGURATION} plans=${PLANS} (all 5 folds)"
python nnunet/predict_cv.py \
  --dataset_id "${DATASET}" --configuration "${CONFIGURATION}" --plans "${PLANS}" \
  --hole_decomposition
