#!/bin/bash
# AMP re-smoke (A7, 7/6 decision): rung6 K=2 DMD loop, ~30 iters, to re-measure peak VRAM
# and 30-iter stability under the new fp32-trainable + bf16-autocast recipe (replaces the
# pure-bf16 + MasterAdamW numerics). This is the "Ongoing gates" §14 first gate:
#   AMP re-smoke (peak + 30-iter stability) -> bank QC -> resume kill-test -> ...
# and the §10 hard rule it checks: measured peak <= 90% of an 80 GB card (72 GB).
#
# DO NOT SBATCH THIS WITHOUT REVIEW — prepared per the AMP refactor task, not yet submitted.
#   sbatch scripts/rung6_amp_resmoke.sbatch
#
# Runs on an H200 (141 GB) for headroom so the measured peak is a real number, not an
# OOM-truncated one; compare the printed "peak VRAM" line against the 72 GB budget to decide
# whether an 80 GB card (H100/A100-80) is enough or H200-only is required (§10 escalation
# ladder). Falls back to L40S by editing --gres + adding --offload_teacher.

#SBATCH --job-name=spt-rung6-amp-resmoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rung6_amp_resmoke_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== python / gpu ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== CPU tests (incl. freeze fp32-trainable + dmd accumulation) ==="
python tests/test_score.py
python tests/test_freeze.py
python tests/test_latents.py
python tests/test_middle_bank.py
python tests/test_dmd.py
python tests/test_master.py

echo "=== AMP re-smoke: Rung 6 K=2 DMD marginalization, 30 iters (GPU) ==="
VIDEO_PATH="demo_videos/videos/video_0.mp4"
CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"
BANK_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank"

python -u scripts/rung6_smoke.py \
    --video_path "$VIDEO_PATH" \
    --v0_id video_0 \
    --caption "a video of a scene" \
    --ckpt "$CKPT" \
    --bank_dir "$BANK_DIR" \
    --num_middles 2 \
    --num_iters 30

# rung6_smoke.py already prints, per iter: loss_G / loss_D / mean_arrow (+ per-middle arrows),
# and at the end: the RUNG 6 RESULT block incl. "peak VRAM" from torch.cuda.max_memory_allocated().
# Check against §10: peak VRAM <= 72 GB (90% of 80 GB) and loss_G/loss_D finite + stable
# over all 30 iters (no blow-up, arrow trending nonzero and not diverging).
