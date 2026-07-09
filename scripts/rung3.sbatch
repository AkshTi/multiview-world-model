#!/bin/bash
# Rung 3 validation on an L40S: N-source model edits.
#   sbatch scripts/rung3.sbatch

#SBATCH --job-name=spt-rung3
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rung3_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== python / gpu ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== CPU tests (math, freeze, latents) ==="
python tests/test_score.py
python tests/test_freeze.py
python tests/test_latents.py

echo "=== Rung 3 N-source equivalence + two-source smoke (GPU) ==="
VIDEO_PATH="demo_videos/videos/video_0.mp4"
CAMERA_FILE="demo_videos/cameras/camera_extrinsics.json"
CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"

python scripts/rung3_smoke.py \
    --video_path "$VIDEO_PATH" \
    --caption "a video of a scene" \
    --ckpt "$CKPT" \
    --camera_file "$CAMERA_FILE"
