#!/bin/bash
# Run-1 diagnostic: short K=1 DMD probe (config/train/run1_diag.yaml). Chained AFTER the full
# train bank (build_full_bank.sbatch) completes -- the sampler needs >=1 middle for every
# train v0. Fits one 6h H200 window (200 K=1 steps). Resumable if it ever does spill.
#
#   sbatch --dependency=afterok:<train_bank_jobid> scripts/run1_diag.sbatch
# (scripts/chain_run1.sh wires the dependency automatically.)

#SBATCH --job-name=spt-run1-diag
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/run1_diag_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BANK_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank_cfg1"

echo "=== precheck: every train v0 has >=1 cached middle ==="
python - <<'PY'
import json, sys
sys.path.insert(0, ".")
from spacetimepilot.training import data, middle_bank as mbk
BANK = "/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank_cfg1"
entries = data.load_split("config/data/pilot_split.json", "train")
missing = [e.v0_id for e in entries if len(mbk.list_middles(BANK, e.v0_id)) < 1]
print(f"train v0s: {len(entries)}  missing middles: {len(missing)}")
if missing:
    print("MISSING:", missing[:10], "..." if len(missing) > 10 else "")
    sys.exit(1)
print("precheck OK -- bank covers all train v0s")
PY

echo "=== python / gpu ==="
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "=== K=1 diagnostic train (200 steps) ==="
python -u scripts/train_dmd.py --config config/train/run1_diag.yaml

echo "=== metrics tail ==="
tail -12 /orcd/scratch/orcd/014/akshatat/counterfactual_models/runs/run1_diag/metrics.jsonl
