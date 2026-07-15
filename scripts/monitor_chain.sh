#!/bin/bash
# Watchdog for the Run-1-diagnostic chain. Polls every 10 min; exits (which notifies the
# session) on the FIRST decisive event: any chain job failing, the baselines completing (first
# multiview result), or the final eval-ckpt completing. Low-footprint: one sacct + one squeue
# per tick, capped at 90 ticks (~15h).
#
#   bash scripts/monitor_chain.sh <train_bank> <heldout_bank> <baselines> <diag> <eval_ckpt>

TRAIN="$1"; HELDOUT="$2"; BASELINES="$3"; DIAG="$4"; EVALCKPT="$5"
LOGD=/home/akshatat/code/SpaceTimePilot/logs
BAD="FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|DEADLINE|NODE_FAIL"

state() { sacct -j "$1" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}'; }
arr_has_bad() { sacct -j "$1" -X -n -o State 2>/dev/null | grep -qE "$BAD"; }

echo "monitor start $(date '+%F %T')  jobs: train=$TRAIN heldout=$HELDOUT baselines=$BASELINES diag=$DIAG eval=$EVALCKPT"

for i in $(seq 1 90); do
    # --- failure detection across the whole chain ---
    for J in "$TRAIN" "$HELDOUT" "$BASELINES" "$DIAG" "$EVALCKPT"; do
        if arr_has_bad "$J"; then
            echo "=== FAILURE: job $J in bad state at $(date '+%F %T') ==="
            sacct -j "$J" -X -n -o JobID,JobName%20,State,ExitCode 2>/dev/null
            echo "--- recent log lines ---"
            LF=$(ls -1t $LOGD/*"${J%_*}"*.out 2>/dev/null | head -1)
            [ -n "$LF" ] && tail -25 "$LF"
            exit 2
        fi
    done
    # a dependent that can never run (its afterok prereq failed)
    if squeue -u "$USER" -h -o "%i %R" 2>/dev/null | grep -q "DependencyNeverSatisfied"; then
        echo "=== FAILURE: a chain job has DependencyNeverSatisfied at $(date '+%F %T') ==="
        squeue -u "$USER" -o "%.16i %.16j %.24R %.30E" 2>/dev/null
        exit 2
    fi

    ESTATE=$(state "$EVALCKPT"); BSTATE=$(state "$BASELINES"); DSTATE=$(state "$DIAG")

    # --- final success ---
    if [ "$ESTATE" = "COMPLETED" ]; then
        echo "=== CHAIN COMPLETE (eval-ckpt done) at $(date '+%F %T') ==="
        LF=$(ls -1t $LOGD/eval_ckpt_*.out 2>/dev/null | head -1)
        [ -n "$LF" ] && tail -40 "$LF"
        exit 0
    fi

    # --- first-result milestone: baselines done ---
    if [ "$BSTATE" = "COMPLETED" ] && [ ! -f /tmp/.chain_baselines_seen_$BASELINES ]; then
        touch /tmp/.chain_baselines_seen_$BASELINES
        echo "=== MILESTONE: baselines COMPLETED at $(date '+%F %T') (first multiview result) ==="
        LF=$(ls -1t $LOGD/eval_pipeline_*.out 2>/dev/null | head -1)
        [ -n "$LF" ] && tail -40 "$LF"
        exit 10   # distinct code: report first result, then relaunch to watch diag+eval
    fi

    echo "tick $i $(date '+%T'): baselines=$BSTATE diag=$DSTATE eval=$ESTATE"
    sleep 600
done
echo "monitor timed out after ~15h without a terminal event"
exit 3
