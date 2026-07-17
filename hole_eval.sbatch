#!/bin/bash
#SBATCH --job-name=elm-hole-eval
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

# Re-scores the same canonical checkpoints as native_eval.sbatch, at native
# per-slice resolution, but with --hole_decomposition: every eye in this
# dataset has a macular hole, but it only crosses some of an eye's slices, so
# on those slices this compares the model's own predicted gap in the ELM
# line against the GT gap -- whether it bridged straight across (no gap at
# all) and, if not, how its width and margins (ELM termination points)
# compare -- so the "models struggle with the hole boundaries" pattern from
# the qualitative figure can be quantified across all 99 eyes instead of
# eyeballed from 3 example slices.
#
# Writes to a separate cv_eval_hole / cv_eval_3d_hole dir per model, so the
# canonical cv_eval_native / cv_eval_3d_native outputs used elsewhere
# (aggregate_native_results.py) are untouched. Run aggregate_hole_results.py
# afterwards to turn these into the summary table.

echo "=== SegNet (2D) ==="
python predict_cv2d.py \
  --model_root elm-results/SegNet_Jun-04-2026_1243_model \
  --model SegNet \
  --native_res --hole_decomposition \
  --out_dir elm-results/SegNet_Jun-04-2026_1243_model/cv_eval_hole

echo "=== R2U_Net (2D) ==="
python predict_cv2d.py \
  --model_root elm-results/R2U_Net_Jun-04-2026_0441_model \
  --model R2U_Net \
  --native_res --hole_decomposition \
  --out_dir elm-results/R2U_Net_Jun-04-2026_0441_model/cv_eval_hole

echo "=== UNet2DEnc3DDec (3D) ==="
python predict_cv3d.py \
  --model_root elm-results/UNet2DEnc3DDec_Jun-07-2026_1946_model \
  --model UNet2DEnc3DDec \
  --native_res --hole_decomposition \
  --out_dir elm-results/UNet2DEnc3DDec_Jun-07-2026_1946_model/cv_eval_3d_hole

echo "=== UNet3DFrawley (3D) ==="
python predict_cv3d.py \
  --model_root elm-results/UNet3DFrawley_Jun-09-2026_1509_model \
  --model UNet3DFrawley \
  --native_res --hole_decomposition \
  --out_dir elm-results/UNet3DFrawley_Jun-09-2026_1509_model/cv_eval_3d_hole

echo "=== CSAM_UNet2p5D (3D) ==="
python predict_cv3d.py \
  --model_root elm-results/CSAM_UNet2p5D_Jun-08-2026_0152_model \
  --model CSAM_UNet2p5D \
  --native_res --hole_decomposition \
  --out_dir elm-results/CSAM_UNet2p5D_Jun-08-2026_0152_model/cv_eval_3d_hole

echo "=== UNet2p5D_SlidingWindow (3D) ==="
python predict_cv3d.py \
  --model_root elm-results/UNet2p5D_SlidingWindow_Jun-11-2026_0120_model \
  --model UNet2p5D_SlidingWindow \
  --window_k 7 \
  --native_res --hole_decomposition \
  --out_dir elm-results/UNet2p5D_SlidingWindow_Jun-11-2026_0120_model/cv_eval_3d_hole
