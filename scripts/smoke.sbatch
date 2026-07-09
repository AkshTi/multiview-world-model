#!/bin/bash
# Rung 0-2 validation on an L40S. Submit from the repo root:
#   sbatch scripts/smoke.sbatch
# or start an interactive node instead (see the srun line in the header comment below).
#
# Interactive equivalent:
#   srun --partition=mit_normal_gpu --account=mit_general --gres=gpu:l40s:1 \
#        --cpus-per-task=8 --mem=96G --time=01:00:00 --pty bash

#SBATCH --job-name=spt-rung2
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rung2_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

# Use the counterfactual conda env directly (no `conda activate` needed).
export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== python / gpu ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== Rung 1/2 CPU tests (math, freeze, latents) ==="
python tests/test_score.py
python tests/test_freeze.py
python tests/test_latents.py

echo "=== Rung 2 one-source training smoke (GPU) ==="
# Defaults point at the downloaded demo data (scripts/setup_data.sh). Change the clip if you like.
VIDEO_PATH="demo_videos/videos/video_0.mp4"
CAMERA_FILE="demo_videos/cameras/camera_extrinsics.json"
CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"

python scripts/rung2_smoke.py \
    --video_path "$VIDEO_PATH" \
    --caption "a video of a scene" \
    --ckpt "$CKPT" \
    --camera_file "$CAMERA_FILE" \
    --cam_type cam01 \
    --do_optimizer_step
