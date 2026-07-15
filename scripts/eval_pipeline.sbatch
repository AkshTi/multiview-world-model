#!/bin/bash
# Run-1 pre-registered baselines: B0 (released SPT independent view) + B1 (student-at-init)
# rollouts on the 8 held-out v0s, then score them (PLAN_DMD_TRAINING.md section 8).
#
# PREREQUISITE: scripts/build_heldout_bank.sbatch must have already completed (v1 middles for
# the 8 held-out v0s at cam01/cam02). This job does NOT build that bank itself.
#
# Metric-1a runs in its ESTIMATED-H form (7/14 amendment): no intrinsics needed. The
# commanded-H form was retired after scripts/calibrate_fov.py measured that the released SPT
# renders 1.2-2.2x the commanded rotation with no stable implied FOV -- see
# spacetimepilot/eval/matching.py's docstring for the full rationale + the anti-copy guard.
#
#   sbatch scripts/eval_pipeline.sbatch

#SBATCH --job-name=spt-eval-pilot
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_pipeline_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/eval/pilot_baselines"

echo "=== CPU tests (geometry + matching math, quick sanity before spending GPU time) ==="
python tests/test_eval_geometry.py
python tests/test_eval_matching.py

echo "=== B0: released SPT, independent view ==="
python -u scripts/eval_rollout.py --method b0 --out_dir "$OUT_DIR"

echo "=== B1: student-at-init (2-source, released weights) ==="
python -u scripts/eval_rollout.py --method b1 --out_dir "$OUT_DIR"

echo "=== report (metric-1a estimated-H + metric-2) ==="
python -u scripts/eval_report.py --rollout_dir "$OUT_DIR" --methods b0 b1 \
    --out "$OUT_DIR/report_baselines.json"
