#!/bin/bash
# Submit the whole Run-1-diagnostic DAG with SLURM dependencies, so it runs itself as each
# gate clears -- no babysitting. Idempotent-ish: safe to read the printed job ids into the
# monitor. Bank jobs are auto-detected from the queue (override by passing them as args).
#
#   bash scripts/chain_run1.sh [TRAIN_BANK_JOBID] [HELDOUT_BANK_JOBID]
#
# DAG:
#   train_bank(existing) ---afterok--> run1_diag(K=1 200 steps) --.
#   heldout_bank(existing) --afterok--> baselines(B0,B1,report) --+--afterok--> eval_ckpt(before/after table)
#
# afterok = a dependent runs ONLY if its prereq succeeded; if a prereq FAILS, dependents are
# cancelled by SLURM (state DependencyNeverSatisfied) -- the monitor watches for exactly that.

set -e
cd /home/akshatat/code/SpaceTimePilot

TRAIN_BANK="${1:-$(squeue -u "$USER" -h -n spt-fullbank -o '%A' | head -1 | cut -d_ -f1)}"
HELDOUT_BANK="${2:-$(squeue -u "$USER" -h -n spt-heldout-bank -o '%A' | head -1 | cut -d_ -f1)}"

if [ -z "$TRAIN_BANK" ] || [ -z "$HELDOUT_BANK" ]; then
    echo "ERROR: could not auto-detect bank jobs (train='$TRAIN_BANK' heldout='$HELDOUT_BANK')."
    echo "Pass them explicitly: bash scripts/chain_run1.sh <train_bank_id> <heldout_bank_id>"
    exit 1
fi
echo "depending on: train_bank=$TRAIN_BANK  heldout_bank=$HELDOUT_BANK"

# 1) Baselines B0+B1 (+ metric-2/1a report) once the held-out bank (v1 refs) is built.
BASELINES=$(sbatch --parsable --dependency=afterok:"$HELDOUT_BANK" scripts/eval_pipeline.sbatch)
echo "submitted baselines     : $BASELINES  (afterok:$HELDOUT_BANK)"

# 2) Diagnostic K=1 train once the full train bank is built.
DIAG=$(sbatch --parsable --dependency=afterok:"$TRAIN_BANK" scripts/run1_diag.sbatch)
echo "submitted diag train    : $DIAG  (afterok:$TRAIN_BANK)"

# 3) Score the trained ckpt vs B0/B1 once BOTH the train and the baselines are done.
EVALCKPT=$(sbatch --parsable --dependency=afterok:"$DIAG":"$BASELINES" scripts/eval_ckpt.sbatch)
echo "submitted eval-ckpt     : $EVALCKPT  (afterok:$DIAG:$BASELINES)"

echo ""
echo "CHAIN_JOBS train_bank=$TRAIN_BANK heldout_bank=$HELDOUT_BANK baselines=$BASELINES diag=$DIAG eval_ckpt=$EVALCKPT"
echo ""
squeue -u "$USER" -o "%.16i %.16j %.2t %.10M %.24R %.30E" 2>/dev/null
