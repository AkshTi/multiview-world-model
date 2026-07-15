"""Build the FULL middle bank across one side of a split (Stage A train, or heldout for eval).

Loads the frozen SPT inference pipe ONCE and loops every v0 in --split_key ("train" or
"heldout"), generating num_middles crossing middles v1_k ~ p(v1|v0) per source video. Reuses
the exact tested generation path from scripts/build_middle_bank.py (build_inference_pipe /
generate_middle / is_cached / middle_paths / build_meta) — this is just the multi-video driver
around it, so the model is loaded once instead of once per video.

Resumable: middles already cached (mp4 + json) are skipped, so re-running continues where a
prior job left off. Shardable for a job array: --num_shards N --shard_id i processes
entries[i::N]. Seeds are keyed by per-v0 middle idx (seed0 + idx), so the produced bank is
byte-for-byte identical regardless of how it is sharded.

Plan A4 production settings: --cfg_scale 1 --num_inference_steps 20 --num_middles 8.

Example:
  python scripts/build_full_bank.py \
      --split config/data/pilot_split.json \
      --bank_dir /orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank_cfg1 \
      --ckpt checkpoints/SpacetimePilot_1.3B_v1.ckpt \
      --num_middles 8 --num_inference_steps 20 --cfg_scale 1 \
      --num_shards 4 --shard_id 0
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import build_middle_bank` works

import single_video_test as svt  # noqa: E402
import build_middle_bank as bmb  # noqa: E402  (reuse its tested pipe + gen path)
from spacetimepilot.utils.misc import save_video  # noqa: E402
from spacetimepilot.dataset.utils import load_frames_using_imageio  # noqa: E402
from spacetimepilot.training import middle_bank as mbk  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Build the full middle bank across a train split")
    p.add_argument("--split", required=True, help="split json (build_split_from_metadata format)")
    p.add_argument("--bank_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--camera_file", default=svt.DEFAULT_CAMERA_FILE)
    p.add_argument("--num_middles", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--mid_time_pattern", default="forward")
    p.add_argument("--src_time_pattern", default="forward")
    p.add_argument("--seed0", type=int, default=0, help="seed for middle 0 of each v0; middle k uses seed0+k")
    p.add_argument("--num_shards", type=int, default=1, help="split the v0 list into this many shards")
    p.add_argument("--shard_id", type=int, default=0, help="which shard this process handles (0..num_shards-1)")
    p.add_argument("--split_key", default="train", choices=["train", "heldout"],
                   help="which side of the split to build middles for (eval uses heldout)")
    p.add_argument("--dit_path", default=svt.DEFAULT_DIT_PATH)
    p.add_argument("--text_encoder_path", default=svt.DEFAULT_TEXT_PATH)
    p.add_argument("--vae_path", default=svt.DEFAULT_VAE_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node"
    assert 0 <= args.shard_id < args.num_shards, "shard_id must be in [0, num_shards)"
    os.makedirs(args.bank_dir, exist_ok=True)

    with open(args.split) as f:
        split = json.load(f)
    entries = split[args.split_key]
    shard = entries[args.shard_id::args.num_shards]
    print(f"split={args.split} [{args.split_key}]={len(entries)}  shard {args.shard_id}/{args.num_shards} -> {len(shard)} v0s")
    print(f"bank_dir={args.bank_dir}  num_middles={args.num_middles}  steps={args.num_inference_steps}  cfg={args.cfg_scale}")

    # crossing cam indices are deterministic in num_middles — same for every v0.
    cam_idxs = mbk.crossing_cam_types(args.num_middles)
    print(f"crossing cam indices ({args.num_middles} middles): {cam_idxs}")

    # Load the frozen inference pipe ONCE (reuses build_middle_bank's tested loader).
    pipe = bmb.build_inference_pipe(SimpleNamespace(
        dit_path=args.dit_path, text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path, ckpt=args.ckpt))

    # Source camera + time patterns are identical across all v0s — build once.
    source_camera = svt.make_identity_src_camera().unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
    src_time = torch.tensor(svt.get_time_pattern(args.src_time_pattern, svt.NUM_FRAMES),
                            dtype=torch.float32).unsqueeze(0).to("cuda")
    mid_time = torch.tensor(svt.get_time_pattern(args.mid_time_pattern, svt.NUM_FRAMES),
                            dtype=torch.float32).unsqueeze(0).to("cuda")
    with open(args.camera_file) as f:
        cam_data = json.load(f)
    middle_cameras = [bmb.target_camera_from_idx(cam_data, ci) for ci in cam_idxs]

    total_gen = 0
    total_skip = 0
    t_start = time.time()
    for vi, entry in enumerate(shard):
        v0_id = entry["v0_id"]
        video_path = entry["video_path"]
        caption = entry["caption"]

        if all(mbk.is_cached(args.bank_dir, v0_id, idx) for idx in range(args.num_middles)):
            print(f"[{vi + 1}/{len(shard)}] {v0_id}: all {args.num_middles} middles cached — skip")
            total_skip += args.num_middles
            continue

        video = load_frames_using_imageio(
            video_path, max_num_frames=svt.NUM_FRAMES, start_frame_id=0, interval=1,
            num_frames=svt.NUM_FRAMES, frame_process=svt.FRAME_PROCESS,
            target_width=svt.WIDTH, target_height=svt.HEIGHT,
        )
        if video is None:
            print(f"[{vi + 1}/{len(shard)}] {v0_id}: WARN could not load {video_path} — skipping v0")
            continue
        source_video = video.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")

        print(f"[{vi + 1}/{len(shard)}] {v0_id}: generating up to {args.num_middles} middles")
        for idx, cam_idx in enumerate(cam_idxs):
            if mbk.is_cached(args.bank_dir, v0_id, idx):
                total_skip += 1
                continue
            seed = args.seed0 + idx
            t0 = time.time()
            frames = mbk.generate_middle(
                pipe, source_video, source_camera, middle_cameras[idx],
                src_time, mid_time, caption, bmb.NEGATIVE_PROMPT,
                seed=seed, num_inference_steps=args.num_inference_steps, cfg_scale=args.cfg_scale,
            )
            video_out, meta_out = mbk.middle_paths(args.bank_dir, v0_id, idx)
            save_video(frames, video_out, fps=30, quality=5,
                       ffmpeg_params=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            meta = mbk.build_meta(
                v0_id=v0_id, idx=idx, cam_idx=cam_idx, camera_file=args.camera_file,
                time_pattern=args.mid_time_pattern, src_time_pattern=args.src_time_pattern,
                seed=seed, num_inference_steps=args.num_inference_steps, cfg_scale=args.cfg_scale,
                caption=caption, source_cam_kind="identity")
            mbk.save_meta(meta_out, meta)
            total_gen += 1
            print(f"    middle {idx} (cam{cam_idx:02d}, seed={seed}) -> {time.time() - t0:.1f}s")

    dt = time.time() - t_start
    print("\n================= FULL BANK SHARD RESULT =================")
    print(f"  shard                 : {args.shard_id}/{args.num_shards}  ({len(shard)} v0s)")
    print(f"  middles generated now : {total_gen}")
    print(f"  middles skipped/cached: {total_skip}")
    print(f"  wall time             : {dt / 60:.1f} min  ({dt / max(total_gen, 1):.1f}s/gen)")
    print(f"  bank dir              : {args.bank_dir}")
    print(f"  resumable: re-run this shard to continue; is_cached skips finished middles.")
    print("=========================================================")


if __name__ == "__main__":
    main()
