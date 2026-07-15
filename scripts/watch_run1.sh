#!/bin/bash
# Milestone watcher for the Run-1 chain: every 30 min, (a) surface any FAILED window,
# (b) when ckpt_500 / ckpt_1000 / ckpt_1499 first appears, submit ONE eval job for it
# (b0/b1 rollouts on disk get rescored alongside it -> the growing before/after curve).
# Exits after the final milestone eval is submitted, on failure, or after ~66h.
#
#   bash scripts/watch_run1.sh <window_jobids...>

cd /home/akshatat/code/SpaceTimePilot
RUN_DIR=/orcd/scratch/orcd/014/akshatat/counterfactual_models/runs/run1
EVAL_DIR=/orcd/scratch/orcd/014/akshatat/counterfactual_models/eval/pilot_baselines
MARK=/tmp/.run1_eval_marks; mkdir -p "$MARK"
BAD="FAILED|OUT_OF_MEMORY|NODE_FAIL"
MILESTONES="250 500 999"           # ckpt steps for a 1000-step run (grid@250/500 + final@999)
FINAL_MILESTONE="999"

for i in $(seq 1 132); do
    for J in "$@"; do
        ST=$(sacct -j "$J" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
        if echo "$ST" | grep -qE "$BAD"; then
            echo "=== RUN1 WINDOW $J FAILED ($ST) at $(date '+%F %T') ==="
            LF=$(ls -1t logs/run1_${J}.out 2>/dev/null | head -1)
            [ -n "$LF" ] && { grep -nE "Traceback|Error|ABORT|assert" "$LF" | head; tail -15 "$LF"; }
            exit 2
        fi
    done
    LAST_DONE=1
    for S in $MILESTONES; do
        CK="$RUN_DIR/ckpt_${S}.pt"
        if [ -f "$CK" ] && [ ! -f "$MARK/eval_$S" ]; then
            EJ=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR",CKPT="$CK" scripts/eval_ckpt.sbatch)
            touch "$MARK/eval_$S"
            echo "=== MILESTONE ckpt_$S: eval job $EJ submitted at $(date '+%F %T') ==="
            # After the FINAL milestone's eval, chain the experiment summary (afterok) so the
            # run ends with the Hyunwoo-ready SUMMARY.md, no manual step.
            if [ "$S" = "$FINAL_MILESTONE" ]; then
                SJ=$(sbatch --parsable --dependency=afterok:"$EJ" \
                     --export=ALL,EVAL_DIR="$EVAL_DIR",RUN_DIR="$RUN_DIR" scripts/summarize.sbatch)
                echo "=== FINAL: summary job $SJ chained after eval $EJ ==="
            fi
        fi
        [ -f "$MARK/eval_$S" ] || LAST_DONE=0
    done
    if [ "$LAST_DONE" = "1" ]; then
        echo "=== all milestone evals submitted; watcher done at $(date '+%F %T') ==="
        exit 0
    fi
    LATEST=$(ls -1 "$RUN_DIR"/ckpt_*.pt 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/ckpt_//;s/\.pt//' | sort -n | tail -1)
    echo "tick $i $(date '+%T'): latest ckpt=${LATEST:-none}"
    sleep 1800
done
echo "watcher timed out (~66h)"; exit 3
