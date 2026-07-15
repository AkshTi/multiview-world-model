#!/bin/bash
# Submit Run 1 as a chain of N 6h windows (afterany: a TIMEOUT window still triggers the
# next, which resumes from the latest ckpt; finished runs no-op). 1500 K=2 steps at ~116s
# ~= 48 GPU-h ~= 9 windows; submit 10 for slack.
#
#   bash scripts/launch_run1_chain.sh [N_WINDOWS]

set -e
cd /home/akshatat/code/SpaceTimePilot
N="${1:-10}"

PREV=""
JOBS=()
for i in $(seq 1 "$N"); do
    if [ -z "$PREV" ]; then
        J=$(sbatch --parsable scripts/run1.sbatch)
    else
        J=$(sbatch --parsable --dependency=afterany:"$PREV" scripts/run1.sbatch)
    fi
    JOBS+=("$J"); PREV="$J"
done
echo "RUN1_CHAIN ${JOBS[*]}"
squeue -u "$USER" -o "%.12i %.14j %.2t %.10M %.20R %.24E" 2>/dev/null | head -15
