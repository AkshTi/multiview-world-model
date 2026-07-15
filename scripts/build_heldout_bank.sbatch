#!/bin/bash
# Eval prereq: build cfg=1 middles for the 8 HELD-OUT v0s (build_full_bank.py only touches
# "train" by default — the eval pipeline needs v1 middles for held-out v0s too, at the cam
# indices used by config/eval/pilot_eval_tuples.json: cam01 and cam02 (crossing_cam_types(2)).
# Same bank dir as the train build (middle_bank_cfg1) — v0_ids never collide across the split.
#
# 8 v0s x 2 middles = 16 generations x ~2.3 min ~= 40 min on one GPU. Single shard (small).
#
#   sbatch scripts/build_heldout_bank.sbatch

#SBATCH --job-name=spt-heldout-bank
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/heldoutbank_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python -u scripts/build_full_bank.py \
    --split config/data/pilot_split.json \
    --split_key heldout \
    --bank_dir /orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank_cfg1 \
    --ckpt checkpoints/SpacetimePilot_1.3B_v1.ckpt \
    --num_middles 2 \
    --num_inference_steps 20 \
    --cfg_scale 1 \
    --num_shards 1 \
    --shard_id 0
