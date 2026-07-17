#!/bin/sh

# Activate environment
source /home/s2036401/miniconda3/etc/profile.d/conda.sh
conda activate elm-gpu

# Define directories
SCRATCH_DIR=/disk/scratch/s2036401/elm-data
DATASET_DIR=/home/s2036401/Benchmark-ELM-Line-OCT-Dataset/data
RESULTS_DIR=/home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results

# Prepare scratch space
mkdir -p $SCRATCH_DIR

# Copy dataset to scratch (fast local disk)
rsync -a --info=progress2  $DATASET_DIR $SCRATCH_DIR

# Ensure results copy back even on failure or interruption
trap 'echo "Syncing results back..."; rsync -av --progress $SCRATCH_DIR/elm-results/ $RESULTS_DIR/' EXIT

# Run training
python train3D.py \
    --base_dir $SCRATCH_DIR


