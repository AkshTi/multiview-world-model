"""Rung 2 GPU smoke test: prove SPT trains at the latent level with a correct grad mask.

Reuses the exact data loaders from single_video_test.py (video, source/target camera,
world-time), loads the pipeline the same way as inference, but:
  * does NOT call enable_vram_management() (inference-only offload; unsafe for backprop);
  * freezes the DiT and unfreezes only the SPT fine-tune set;
  * runs one_source_smoke_step (throwaway flow-matching MSE, a graph test);
  * backward() + assert_grad_mask() + reports peak VRAM.

For a pure graph test the "target" video is just the source video (the MSE is not the
objective). Run from the repo root on a GPU node.

Example:
  python scripts/rung2_smoke.py \
      --video_path demo_videos/<clip>.mp4 \
      --caption "a scene" \
      --ckpt checkpoints/SpacetimePilot_1.3B_v1.ckpt \
      --camera_file demo_videos/cameras/camera_extrinsics.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_video_test as svt  # noqa: E402  (loaders + constants + __main__ guard)
from spacetimepilot import ModelManager  # noqa: E402
from spacetimepilot.utils.builder import build_pipeline  # noqa: E402
from spacetimepilot.dataset.utils import (  # noqa: E402
    load_frames_using_imageio,
    process_camera_trajectory,
    compute_pose_embedding,
)
from spacetimepilot.training import freeze, steps  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Rung 2 one-source training smoke test")
    p.add_argument("--video_path", required=True)
    p.add_argument("--caption", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--camera_file", default=svt.DEFAULT_CAMERA_FILE)
    p.add_argument("--cam_type", default="cam01")
    p.add_argument("--temporal_control", default="forward")
    p.add_argument("--dit_path", default=svt.DEFAULT_DIT_PATH)
    p.add_argument("--text_encoder_path", default=svt.DEFAULT_TEXT_PATH)
    p.add_argument("--vae_path", default=svt.DEFAULT_VAE_PATH)
    p.add_argument("--num_train_steps", type=int, default=1000)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--do_optimizer_step", action="store_true")
    return p.parse_args()


def build_pipe(args):
    print("Loading Wan2.1 foundation models...")
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([args.dit_path, args.text_encoder_path, args.vae_path])
    pipe = build_pipeline({"type": svt.PIPELINE_VERSION}).from_model_manager(mm, device="cuda")
    print(f"Loading SPT checkpoint: {args.ckpt}")
    pipe.dit.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    pipe.to(dtype=torch.bfloat16)
    # Move every submodule to CUDA. Inference normally does this via
    # enable_vram_management(), which we skip for training (it breaks backprop), so we
    # must place the VAE / text encoder / DiT on the GPU ourselves.
    for m in (pipe.dit, pipe.vae, pipe.text_encoder, pipe.image_encoder):
        if m is not None:
            m.to("cuda")
    pipe.device = torch.device("cuda")
    # NOTE: intentionally NOT calling pipe.enable_vram_management() — training needs
    # gradients to reach the original parameters.
    return pipe


def build_batch(pipe, args):
    cam_idx = svt._parse_cam_type(args.cam_type)
    video = load_frames_using_imageio(
        args.video_path, max_num_frames=svt.NUM_FRAMES, start_frame_id=0, interval=1,
        num_frames=svt.NUM_FRAMES, frame_process=svt.FRAME_PROCESS,
        target_width=svt.WIDTH, target_height=svt.HEIGHT,
    )
    if video is None:
        raise ValueError(f"Could not load video: {args.video_path}")
    source_video = video.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")  # (1,C,81,H,W)

    src_cam = svt.make_identity_src_camera()
    source_camera = src_cam.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")  # (1,21,12)

    with open(args.camera_file) as f:
        cam_data = json.load(f)
    frame_indices = list(range(svt.NUM_FRAMES))[::4]
    tgt_cam = compute_pose_embedding(process_camera_trajectory(cam_data, frame_indices, cam_idx))
    tgt_cam = rearrange(tgt_cam, "b c d -> b (c d)")
    target_camera = tgt_cam.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")  # (1,21,12)

    src_time = torch.tensor(svt.get_time_pattern("forward", svt.NUM_FRAMES),
                            dtype=torch.float32).unsqueeze(0).to("cuda")  # (1,81)
    tgt_time = torch.tensor(svt.get_time_pattern(args.temporal_control, svt.NUM_FRAMES),
                            dtype=torch.float32).unsqueeze(0).to("cuda")

    return {
        "source_video": source_video,
        "target_video": source_video,  # graph test only; MSE is not the objective
        "source_camera": source_camera,
        "target_camera": target_camera,
        "src_time_embedding": src_time,
        "tgt_time_embedding": tgt_time,
        "prompt": args.caption,
    }


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node (see scripts/smoke.sbatch)"
    torch.cuda.reset_peak_memory_stats()

    pipe = build_pipe(args)
    batch = build_batch(pipe, args)

    trainable = freeze.set_trainable(pipe.dit, verbose=True)
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    pipe.scheduler.set_timesteps(args.num_train_steps, training=True, shift=args.sigma_shift)

    optimizer.zero_grad()
    out = steps.one_source_smoke_step(pipe, batch, pipe.scheduler)
    loss = out["loss"]
    loss.backward()

    assert pipe.dit.training, "DiT must be in train() mode for checkpointing to run"
    freeze.assert_grad_mask(pipe.dit)
    if args.do_optimizer_step:
        optimizer.step()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print("\n================= RUNG 2 SMOKE RESULT =================")
    print(f"  loss (throwaway MSE) : {loss.item():.6f}  (finite={torch.isfinite(loss).item()})")
    print(f"  sampled sigma        : {out['sigma'].item():.4f}")
    print(f"  grad mask            : PASS (no leak; trainable grads present)")
    print(f"  peak VRAM            : {peak:.2f} GB")
    print(f"  optimizer step       : {'done' if args.do_optimizer_step else 'skipped'}")
    print("======================================================")


if __name__ == "__main__":
    main()
