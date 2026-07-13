#!/bin/bash
# cfg ablation side-bank: build matched middles at cfg=1 vs cfg=3 to decide the
# production middle-bank guidance scale (PLAN_DMD_TRAINING.md A4 / ledger #6).
#
# Controlled comparison: same v0 / seed0 / cameras (crossing_cam_types is deterministic,
# seed = seed0+idx) and the SAME num_inference_steps (20 = production). The ONLY variable
# is --cfg_scale. The two arms MUST land in SEPARATE bank dirs because is_cached/middle_paths
# key on (bank_dir, v0_id, idx) only — a shared dir would make the 2nd cfg arm skip everything.
#
#   sbatch scripts/cfg_sidebank.sbatch

#SBATCH --job-name=spt-cfg-sidebank
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cfg_sidebank_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"
ROOT="/orcd/scratch/orcd/014/akshatat/counterfactual_models/cfg_sidebank"
NUM_MIDDLES=4
STEPS=20

# v0 subset spanning content types (crowd detail / single subject / motion+texture).
V0_IDS=(video_0 video_2 video_4)
CAPTIONS=(
  "The video opens with a vibrant street scene where a group of people are gathered. The crowd is diverse"
  "The video features a single rhinoceros"
  "The video features a person wearing a black helmet and dark clothing, riding a mountain bike through a forested area."
)

python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

for CFG in 1 3; do
  BANK="${ROOT}/cfg${CFG}"
  echo "===================== cfg_scale=${CFG}  ->  ${BANK} ====================="
  for i in "${!V0_IDS[@]}"; do
    ID="${V0_IDS[$i]}"
    echo "--- v0=${ID}  cfg=${CFG} ---"
    python -u scripts/build_middle_bank.py \
      --video_path "demo_videos/videos/${ID}.mp4" \
      --v0_id "${ID}" \
      --caption "${CAPTIONS[$i]}" \
      --ckpt "${CKPT}" \
      --bank_dir "${BANK}" \
      --num_middles "${NUM_MIDDLES}" \
      --num_inference_steps "${STEPS}" \
      --cfg_scale "${CFG}"
  done
done

echo "===================== QC: sharpness compare ====================="
python -u scripts/qc_cfg_compare.py \
  --cfg1_dir "${ROOT}/cfg1" \
  --cfg3_dir "${ROOT}/cfg3" \
  --v0_ids "${V0_IDS[@]}" \
  --num_middles "${NUM_MIDDLES}"
