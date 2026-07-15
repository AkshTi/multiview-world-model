#!/bin/bash
# Final link of the chain: roll out the DIAGNOSTIC checkpoint (EMA weights) at the fixed eval
# tuples and score it against the B0/B1 baselines -> the before/after multiview table.
# Chained AFTER both the diagnostic train AND the baselines (needs the ckpt + the b0/b1
# rollouts to compare against). Writes into the SAME dir the baselines used, so eval_report
# sees b0, b1, and the checkpoint side by side.
#
#   sbatch --dependency=afterok:<diag_train>:<baselines> scripts/eval_ckpt.sbatch

#SBATCH --job-name=spt-eval-ckpt
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_ckpt_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Overridable via --export (watcher/manual): RUN_DIR (which run), CKPT (exact ckpt file).
RUN_DIR="${RUN_DIR:-/orcd/scratch/orcd/014/akshatat/counterfactual_models/runs/run1}"
EVAL_DIR="${EVAL_DIR:-/orcd/scratch/orcd/014/akshatat/counterfactual_models/eval/pilot_baselines}"

# Latest ckpt_<step>.pt by NUMERIC step. (The old `sort -t_ -k2 -n` split on the PATH's
# underscores, not the step -- it once picked ckpt_50 over ckpt_150. Sort basenames.)
if [ -z "$CKPT" ]; then
    STEP_MAX=$(ls -1 "$RUN_DIR"/ckpt_*.pt 2>/dev/null | xargs -n1 basename 2>/dev/null \
               | sed 's/ckpt_//;s/\.pt//' | sort -n | tail -1)
    [ -n "$STEP_MAX" ] && CKPT="$RUN_DIR/ckpt_${STEP_MAX}.pt"
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no usable ckpt (CKPT='$CKPT', RUN_DIR='$RUN_DIR')"
    exit 1
fi
STEP=$(basename "$CKPT" .pt | sed 's/ckpt_//')
TAG="step${STEP}"
echo "=== scoring $CKPT as tag=$TAG ==="

echo "=== rollout: trained student (EMA) at eval tuples ==="
python -u scripts/eval_rollout.py --method ckpt --train_ckpt "$CKPT" --use_ema \
    --tag "$TAG" --out_dir "$EVAL_DIR"

echo "=== report: B0 vs B1 vs $TAG (metric-1a estimated-H + metric-2) ==="
python -u scripts/eval_report.py --rollout_dir "$EVAL_DIR" --methods b0 b1 "$TAG" \
    --out "$EVAL_DIR/report_b0_b1_${TAG}.json"

echo "=== DONE: before/after multiview table written to $EVAL_DIR/report_b0_b1_${TAG}.json ==="
