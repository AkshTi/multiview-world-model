#!/bin/bash
# Rung 6 validation: K>1 DMD marginalization over banked middles.
# Targets an H200 (141 GB) so all three DiTs stay resident with no teacher offload and
# headroom to raise K. Falls back to L40S by editing --gres + adding --offload_teacher.
#   sbatch scripts/rung6.sbatch

#SBATCH --job-name=spt-rung6
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rung6_%j.out

set -e
cd /home/akshatat/code/SpaceTimePilot
mkdir -p logs

export PATH=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== python / gpu ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== CPU tests (incl. dmd accumulation) ==="
python tests/test_score.py
python tests/test_freeze.py
python tests/test_latents.py
python tests/test_middle_bank.py
python tests/test_dmd.py

echo "=== Rung 6 K>1 DMD marginalization (GPU) ==="
VIDEO_PATH="demo_videos/videos/video_0.mp4"
CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"
BANK_DIR="/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank"

python -u scripts/rung6_smoke.py \
    --video_path "$VIDEO_PATH" \
    --v0_id video_0 \
    --caption "a video of a scene" \
    --ckpt "$CKPT" \
    --bank_dir "$BANK_DIR" \
    --num_middles 2 \
    --num_iters 3
