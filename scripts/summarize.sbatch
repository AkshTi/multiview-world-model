#!/bin/bash
# Final stage of the experiment: collect all eval reports + the training curve into one
# SUMMARY.md. CPU-only (no GPU) -- chained afterok the last milestone eval by watch_run1.sh.
#
#   sbatch --export=ALL,EVAL_DIR=...,RUN_DIR=... scripts/summarize.sbatch

#SBATCH --job-name=spt-summary
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/summary_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH

EVAL_DIR="${EVAL_DIR:-/orcd/scratch/orcd/014/akshatat/counterfactual_models/eval/pilot_baselines}"
RUN_DIR="${RUN_DIR:-/orcd/scratch/orcd/014/akshatat/counterfactual_models/runs/run1}"

python -u scripts/summarize_experiment.py \
    --eval_dir "$EVAL_DIR" \
    --metrics  "$RUN_DIR/metrics.jsonl"
