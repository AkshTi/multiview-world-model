#!/bin/bash
# Finish the eval after the save_video/bf16 fix: B0 already succeeded (16 videos on disk), so
# regenerate ONLY B1, roll out the diagnostic checkpoint (EMA), and score all three ->
# the before/after multiview table. No bank/train dependency: banks are built, ckpt exists.
#
#   sbatch scripts/eval_finish.sbatch

#SBATCH --job-name=spt-eval-finish
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_finish_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

RUN_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/runs/run1_diag"
EVAL_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/eval/pilot_baselines"

echo "=== CPU tests (geometry + matching) ==="
python tests/test_eval_geometry.py
python tests/test_eval_matching.py

echo "=== sanity: B0 already present? ==="
ls -1 "$EVAL_DIR/b0"/*.mp4 2>/dev/null | wc -l
test -f "$EVAL_DIR/b0/manifest.json" || { echo "B0 missing -- run eval_pipeline.sbatch instead"; exit 1; }

echo "=== B1: student-at-init (was the crash; now fixed) ==="
python -u scripts/eval_rollout.py --method b1 --out_dir "$EVAL_DIR"

CKPT=$(ls -1 "$RUN_DIR"/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
test -n "$CKPT" || { echo "no ckpt in $RUN_DIR"; exit 1; }
STEP=$(basename "$CKPT" .pt | sed 's/ckpt_//'); TAG="step${STEP}"
echo "=== trained student (EMA) rollout from $CKPT as $TAG ==="
python -u scripts/eval_rollout.py --method ckpt --train_ckpt "$CKPT" --use_ema \
    --tag "$TAG" --out_dir "$EVAL_DIR"

echo "=== REPORT: B0 vs B1 vs $TAG (metric-1a estimated-H + metric-2) ==="
python -u scripts/eval_report.py --rollout_dir "$EVAL_DIR" --methods b0 b1 "$TAG" \
    --out "$EVAL_DIR/report_b0_b1_${TAG}.json"

echo "=== DONE: before/after table -> $EVAL_DIR/report_b0_b1_${TAG}.json ==="
