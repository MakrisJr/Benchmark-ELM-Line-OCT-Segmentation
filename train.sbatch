#!/bin/bash
#SBATCH --job-name=elm-train
#SBATCH --partition=ICF-Research
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --output=slurm-logs/%x-%j.out

set -euo pipefail

cd /home/s2036401/Benchmark-ELM-Line-OCT-Dataset

echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate elm-gpu

python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda device_count', torch.cuda.device_count())"

# python train.py --model SwinEncoderUNet2D --epochs 100 --run-all-folds
# python train.py --model R2U_Net --epochs 100 --run-all-folds
# python train.py --model SegNet --epochs 100 --run-all-folds
# python train3D.py --model SwinUNETR3D --epochs 100 --run-all-folds
# python train3D.py --model UNet2DEnc3DDec --epochs 100 --run-all-folds
# python train3D.py --model UNet3DFrawley --epochs 100 --run-all-folds
# python train3D.py --model CSAM_UNet2p5D --epochs 100 --run-all-folds
# new 3D models:
# python train3D.py --model UNet3D --epochs 100 --run-all-folds
# python train3D.py --model UNet3D_Aniso --epochs 100 --run-all-folds
# python train3D.py --model UNet3DFrawley --epochs 100 --run-all-folds

python train3D.py --model UNet2p5D_SlidingWindow --window-k 7 --run-all-folds

# submitted job 4165 for the 3 2D models
# submitted job 6741 for the 4 3D models
# submitted job 3500128 for the 3 new 3D models

# 2.5D sliding window: job 3501176