#!/bin/bash
# Stage A: build the FULL cfg=1 middle bank across the 53-video pilot split (PLAN A4).
# Production settings: cfg=1, 20 inference steps, K_bank=8 middles per v0. Fresh bank dir
# (middle_bank_cfg1) — the old .../middle_bank holds cfg=5 video_0 middles and MUST NOT be mixed.
#
# Job array of 4 disjoint shards (entries[i::4]); each task loads the frozen SPT pipe ONCE and
# loops its ~13-14 v0s. Shards touch disjoint v0 files, so concurrent writes are safe. Fully
# resumable: is_cached skips finished middles, so re-submitting continues where it left off.
# %4 caps concurrency at 4 GPUs (considerate on the shared partition).
#
#   sbatch scripts/build_full_bank.sbatch
# Resume (after a partial run): just re-submit the same line.

#SBATCH --job-name=spt-fullbank
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/fullbank_%A_%a.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

SPLIT="config/data/pilot_split.json"
BANK_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank_cfg1"
CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"

echo "=== python / gpu (shard ${SLURM_ARRAY_TASK_ID}/4) ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python -u scripts/build_full_bank.py \
    --split "$SPLIT" \
    --bank_dir "$BANK_DIR" \
    --ckpt "$CKPT" \
    --num_middles 8 \
    --num_inference_steps 20 \
    --cfg_scale 1 \
    --num_shards 4 \
    --shard_id "${SLURM_ARRAY_TASK_ID}"
