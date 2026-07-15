#!/bin/bash
# Run 1 proper (PLAN section 9): K=2 pilot DMD over the 53-video split, 1500-step ceiling.
# ONE 6h H200 window per submission; train_dmd.py auto-resumes from the latest ckpt in
# out_dir, so a chain of these (afterany dependencies -- TIMEOUT must still trigger the next
# window) walks to max_steps unattended. A window that starts after completion no-ops.
#
#   scripts/launch_run1_chain.sh   # submits N chained windows
#   sbatch scripts/run1.sbatch     # or one window by hand

#SBATCH --job-name=spt-run1
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/run1_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# timeout margin: SIGTERM at 330m; the in-flight step (~2 min) + a possible in-flight
# sample pass (~8 min) + the graceful ckpt save (~3 min, 23 GB) must all fit before
# SLURM's 360m hard kill. train_dmd catches TERM and checkpoints at the step boundary.
timeout 330m python -u scripts/train_dmd.py --config config/train/run1.yaml || {
    rc=$?
    if [ $rc -eq 124 ]; then echo "window timeout hit (expected mid-run) -- next window resumes"; exit 0; fi
    exit $rc
}

echo "=== window done; metrics tail ==="
tail -4 /orcd/scratch/orcd/014/akshatat/counterfactual_models/runs/run1/metrics.jsonl
