#!/bin/bash
# Rung 5 hardening: fp32 master weights + a longer run at a REALISTIC small lr.
# Student lr 1e-5, fake lr 1e-4 — both below the bf16 ulp of typical weights, so without
# master weights the fake net would never move and the arrow would stay 0. With master
# weights the updates accumulate in fp32; this run shows the arrow turning on within a few
# iters and the loop staying finite/stable over 30 iters. Targets an H200.
#   sbatch scripts/rung5_harden.sbatch

#SBATCH --job-name=spt-rung5h
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rung5h_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== python / gpu ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== CPU tests (incl. master weights) ==="
python tests/test_score.py
python tests/test_freeze.py
python tests/test_latents.py
python tests/test_middle_bank.py
python tests/test_dmd.py
python tests/test_master.py

echo "=== Rung 5 hardening: master weights, realistic lr, 30 iters (GPU) ==="
VIDEO_PATH="demo_videos/videos/video_0.mp4"
CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"
BANK_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank"

python scripts/rung5_smoke.py \
    --video_path "$VIDEO_PATH" \
    --v0_id video_0 \
    --caption "a video of a scene" \
    --ckpt "$CKPT" \
    --bank_dir "$BANK_DIR" \
    --master \
    --lr 1e-5 \
    --fake_lr 1e-4 \
    --num_iters 30
