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

# python train.py --model SwinEncoderUNet2D --epochs 100
# python train.py --model R2U_Net --epochs 100
# python train.py --model SegNet --epochs 100
# python new-train.py --model SwinUNETR3D --epochs 100
python new-train.py --model UNet2DEnc3DDec --epochs 100