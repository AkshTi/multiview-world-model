"""Build the middle bank for one source video v0 (Rung 4, GPU).

Generates K crossing middles v1_k ~ p(v1|v0) with the FROZEN released SPT (this is p, not
the student), caches each as mp4 + json to the bank dir, and verifies one reload+encode.
Resumable: middles already cached (mp4+json) are skipped.

Uses the released inference pipe (enable_vram_management ON — pure @torch.no_grad
inference, no backprop), mirroring single_video_test.run_inference.

Example:
  python scripts/build_middle_bank.py \
      --video_path demo_videos/videos/video_0.mp4 --v0_id video_0 \
      --caption "a video of a scene" \
      --ckpt checkpoints/SpacetimePilot_1.3B_v1.ckpt \
      --num_middles 4 --num_inference_steps 50 \
      --bank_dir /orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank
"""

import argparse
import json
import os
import sys

import torch
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_video_test as svt  # noqa: E402
from spacetimepilot import ModelManager  # noqa: E402
from spacetimepilot.utils.builder import build_pipeline  # noqa: E402
from spacetimepilot.utils.misc import save_video  # noqa: E402
from spacetimepilot.dataset.utils import (  # noqa: E402
    load_frames_using_imageio,
    process_camera_trajectory,
    compute_pose_embedding,
)
from spacetimepilot.training import middle_bank as mbk  # noqa: E402
from spacetimepilot.training import latents as latent_utils  # noqa: E402

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    p = argparse.ArgumentParser(description="Build the middle bank for one v0")
    p.add_argument("--video_path", required=True)
    p.add_argument("--v0_id", required=True, help="Stable id for this source video (bank key)")
    p.add_argument("--caption", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--camera_file", default=svt.DEFAULT_CAMERA_FILE)
    p.add_argument("--bank_dir", required=True)
    p.add_argument("--num_middles", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--mid_time_pattern", default="forward")
    p.add_argument("--src_time_pattern", default="forward")
    p.add_argument("--seed0", type=int, default=0, help="seed for middle 0; middle k uses seed0+k")
    p.add_argument("--dit_path", default=svt.DEFAULT_DIT_PATH)
    p.add_argument("--text_encoder_path", default=svt.DEFAULT_TEXT_PATH)
    p.add_argument("--vae_path", default=svt.DEFAULT_VAE_PATH)
    return p.parse_args()


def build_inference_pipe(args):
    """Released inference pipe (mirrors single_video_test.run_inference)."""
    print("Loading Wan2.1 foundation models...")
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([args.dit_path, args.text_encoder_path, args.vae_path])
    pipe = build_pipeline({"type": svt.PIPELINE_VERSION}).from_model_manager(mm, device="cuda")
    print(f"Loading SPT checkpoint: {args.ckpt}")
    pipe.dit.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    pipe.to(dtype=torch.bfloat16)
    pipe.dit.to("cuda")
    pipe.device = torch.device("cuda")
    pipe.enable_vram_management()  # inference only — no gradients here
    return pipe


def target_camera_from_idx(cam_data, cam_idx):
    frame_indices = list(range(svt.NUM_FRAMES))[::4]
    cam = compute_pose_embedding(process_camera_trajectory(cam_data, frame_indices, cam_idx))
    cam = rearrange(cam, "b c d -> b (c d)")
    return cam.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node"
    os.makedirs(args.bank_dir, exist_ok=True)

    pipe = build_inference_pipe(args)

    video = load_frames_using_imageio(
        args.video_path, max_num_frames=svt.NUM_FRAMES, start_frame_id=0, interval=1,
        num_frames=svt.NUM_FRAMES, frame_process=svt.FRAME_PROCESS,
        target_width=svt.WIDTH, target_height=svt.HEIGHT,
    )
    if video is None:
        raise ValueError(f"Could not load video: {args.video_path}")
    source_video = video.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
    source_camera = svt.make_identity_src_camera().unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")

    src_time = torch.tensor(svt.get_time_pattern(args.src_time_pattern, svt.NUM_FRAMES),
                            dtype=torch.float32).unsqueeze(0).to("cuda")
    mid_time = torch.tensor(svt.get_time_pattern(args.mid_time_pattern, svt.NUM_FRAMES),
                            dtype=torch.float32).unsqueeze(0).to("cuda")

    with open(args.camera_file) as f:
        cam_data = json.load(f)

    cam_idxs = mbk.crossing_cam_types(args.num_middles)
    print(f"Crossing cam indices for {args.num_middles} middles: {cam_idxs}")

    generated = 0
    for idx, cam_idx in enumerate(cam_idxs):
        if mbk.is_cached(args.bank_dir, args.v0_id, idx):
            print(f"  [skip] middle {idx} (cam{cam_idx:02d}) already cached")
            continue
        seed = args.seed0 + idx
        print(f"  [gen ] middle {idx}: cam{cam_idx:02d}, seed={seed}")
        middle_camera = target_camera_from_idx(cam_data, cam_idx)
        frames = mbk.generate_middle(
            pipe, source_video, source_camera, middle_camera,
            src_time, mid_time, args.caption, NEGATIVE_PROMPT,
            seed=seed, num_inference_steps=args.num_inference_steps, cfg_scale=args.cfg_scale,
        )
        video_path, meta_path = mbk.middle_paths(args.bank_dir, args.v0_id, idx)
        save_video(frames, video_path, fps=30, quality=5,
                   ffmpeg_params=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        meta = mbk.build_meta(
            v0_id=args.v0_id, idx=idx, cam_idx=cam_idx, camera_file=args.camera_file,
            time_pattern=args.mid_time_pattern, src_time_pattern=args.src_time_pattern,
            seed=seed, num_inference_steps=args.num_inference_steps, cfg_scale=args.cfg_scale,
            caption=args.caption, source_cam_kind="identity")
        mbk.save_meta(meta_path, meta)
        generated += 1

    cached = mbk.list_middles(args.bank_dir, args.v0_id)
    print(f"\nBank now holds middles {cached} for v0_id={args.v0_id}")

    # Verify: reload one cached middle and encode to a latent of the expected shape.
    # enable_vram_management offloads the VAE to CPU after inference; bring it back first
    # (the inference path guards every VAE call the same way).
    pipe.load_models_to_device(["vae"])
    v_path, _ = mbk.middle_paths(args.bank_dir, args.v0_id, cached[0])
    rev = load_frames_using_imageio(
        v_path, max_num_frames=svt.NUM_FRAMES, start_frame_id=0, interval=1,
        num_frames=svt.NUM_FRAMES, frame_process=svt.FRAME_PROCESS,
        target_width=svt.WIDTH, target_height=svt.HEIGHT,
    ).unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
    v1_latent = latent_utils.encode_video_nograd(pipe, rev)

    print("\n================= RUNG 4 RESULT =================")
    print(f"  v0_id                 : {args.v0_id}")
    print(f"  middles generated now : {generated}")
    print(f"  middles cached total  : {len(cached)}  {cached}")
    print(f"  reload+encode v1 shape: {tuple(v1_latent.shape)}  (expect (1, 16, 21, 60, 104))")
    print(f"  bank dir              : {args.bank_dir}")
    print("=================================================")


if __name__ == "__main__":
    main()
