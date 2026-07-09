#!/bin/bash
# One-time setup on the LOGIN node (no GPU needed):
#   * link the Wan2.1 base models,
#   * download the SPT checkpoint + demo videos from HuggingFace (to scratch),
#   * link them into the paths the code expects,
#   * run the CPU test suite (score / freeze / latents).
#
# Run:  bash scripts/setup_data.sh
# Then submit the GPU smoke:  sbatch scripts/smoke.sbatch
set -euo pipefail

REPO=/home/akshatat/code/SpaceTimePilot
ENV_BIN=/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual/bin
WAN_DIR=/orcd/scratch/orcd/014/akshatat/counterfactual_models/Wan-AI/Wan2.1-T2V-1.3B
SPT_HF=/orcd/scratch/orcd/014/akshatat/counterfactual_models/SpaceTimePilot_hf

export PATH="$ENV_BIN:$PATH"
cd "$REPO"

echo "== [1/4] link Wan2.1 base models =="
mkdir -p checkpoints
if [ ! -e checkpoints/wan2.1 ]; then
    ln -s "$WAN_DIR" checkpoints/wan2.1
    echo "  linked checkpoints/wan2.1 -> $WAN_DIR"
else
    echo "  checkpoints/wan2.1 already present"
fi
for f in diffusion_pytorch_model.safetensors models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth; do
    [ -e "checkpoints/wan2.1/$f" ] || { echo "  MISSING base file: $f"; exit 1; }
done

echo "== [2/4] download SPT checkpoint + demo videos (one-time, to scratch) =="
if [ ! -e "$SPT_HF/SpacetimePilot_1.3B_v1.ckpt" ]; then
    hf download zhening/SpaceTimePilot --repo-type model --local-dir "$SPT_HF"
else
    echo "  SPT repo already downloaded at $SPT_HF"
fi
[ -e checkpoints/SpacetimePilot_1.3B_v1.ckpt ] || \
    ln -s "$SPT_HF/SpacetimePilot_1.3B_v1.ckpt" checkpoints/SpacetimePilot_1.3B_v1.ckpt
[ -e demo_videos ] || ln -s "$SPT_HF/demo_videos" demo_videos
echo "  linked checkpoints/SpacetimePilot_1.3B_v1.ckpt and demo_videos/"

echo "== [3/4] CPU test suite (Rung 1 + Rung 2 scaffold) =="
python tests/test_score.py
python tests/test_freeze.py
python tests/test_latents.py

echo "== [4/4] setup complete. Next: run the GPU smoke =="
echo "  sbatch scripts/smoke.sbatch"
echo "  # or interactively:"
echo "  srun --partition=mit_normal_gpu --account=mit_general --gres=gpu:l40s:1 \\"
echo "       --cpus-per-task=8 --mem=96G --time=01:00:00 --pty bash"
