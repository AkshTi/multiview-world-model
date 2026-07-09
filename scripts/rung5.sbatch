#!/bin/bash
# Rung 5 validation on an L40S: the K=1 DMD loop end-to-end.
#   sbatch scripts/rung5.sbatch

#SBATCH --job-name=spt-rung5
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rung5_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# Three resident 1.3B DiTs leave the student's 2-source backward fighting fragmentation;
# expandable segments reclaim the ~4 GB "reserved but unallocated" that OOM'd the first run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== python / gpu ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== CPU tests (incl. dmd + middle bank) ==="
python tests/test_score.py
python tests/test_freeze.py
python tests/test_latents.py
python tests/test_middle_bank.py
python tests/test_dmd.py

echo "=== Rung 5 K=1 DMD loop (GPU) ==="
VIDEO_PATH="demo_videos/videos/video_0.mp4"
CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"
BANK_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank"

python scripts/rung5_smoke.py \
    --video_path "$VIDEO_PATH" \
    --v0_id video_0 \
    --caption "a video of a scene" \
    --ckpt "$CKPT" \
    --bank_dir "$BANK_DIR" \
    --num_iters 3
