#!/bin/bash
# Turn-over test: first end-to-end run of scripts/train_dmd.py (6 steps, 1 video, K=2).
# GATE: submit only AFTER the AMP re-smoke (rung6_amp_resmoke) passes — the loop uses the same
# autocast forward, so there's no point running it until that numerics gate is green.
#   sbatch scripts/turnover.sbatch

#SBATCH --job-name=spt-turnover
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=logs/turnover_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

python -u scripts/train_dmd.py --config config/train/turnover.yaml

echo "=== turn-over metrics tail ==="
tail -8 /orcd/scratch/orcd/014/akshatat/counterfactual_models/runs/turnover/metrics.jsonl