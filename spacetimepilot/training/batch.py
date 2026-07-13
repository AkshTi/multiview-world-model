"""Materialize a ``TrainingItem`` into the tensor dict ``steps.dmd_step_k`` consumes.

This formalizes ``rung6_smoke.build_batch`` (PLAN §7 item 3) into a reusable module: same
tensor constructions, but driven by a ``data.TrainingItem`` instead of argparse, and with
device/dtype parameterized. The output dict shape is the contract fixed by
``steps.dmd_step_k`` — DO NOT change keys without changing that consumer.

The primitives (pixel loader, pose-embedding, time embedding, identity source camera) live
in the top-level ``single_video_test`` script and ``spacetimepilot.dataset.utils``. Because a
package module importing a top-level script is fragile, ``single_video_test`` is imported
LAZILY (inside the calls) — the caller (train_dmd.py, a script) already puts the repo root on
``sys.path``, exactly as every other script here does.

Conventions carried over from the validated smokes:
  * pixels + cameras + world-time are all bf16 on the target device (memory gotcha: training
    has no ``enable_vram_management`` so WE cast world-time to bf16, unlike inference);
  * camera embedding = ``compute_pose_embedding(process_camera_trajectory(...))`` flattened;
  * v0's source camera = identity (A2 stores raw cam_idx; identity src matches how middles
    were generated). The middle-as-source camera representation is the still-open Rung-5 item;
    we use the trajectory embedding, same as the smokes.
  * RoPE positions stay released-default (R0 sequential) — no ``frame_positions`` passed.
"""

import json

import torch
from einops import rearrange

from spacetimepilot.dataset.utils import (
    load_frames_using_imageio,
    process_camera_trajectory,
    compute_pose_embedding,
)


def _svt():
    """Lazy import of the top-level inference script (repo root must be on sys.path)."""
    import single_video_test as svt
    return svt


def load_camera_data(camera_file):
    """Parse the camera_extrinsics.json once; pass the result into build_batch."""
    with open(camera_file) as f:
        return json.load(f)


def load_pixel_video(path, device="cuda", dtype=torch.bfloat16):
    """Load an 81-frame 480x832 clip as a (1, C, T, H, W) tensor (re-encoded like v0)."""
    svt = _svt()
    video = load_frames_using_imageio(
        path, max_num_frames=svt.NUM_FRAMES, start_frame_id=0, interval=1,
        num_frames=svt.NUM_FRAMES, frame_process=svt.FRAME_PROCESS,
        target_width=svt.WIDTH, target_height=svt.HEIGHT,
    )
    if video is None:
        raise ValueError(f"could not load video: {path}")
    return video.unsqueeze(0).to(dtype=dtype, device=device)


def camera_embedding(cam_data, cam_idx, device="cuda", dtype=torch.bfloat16):
    """Pose embedding for a pool trajectory idx -> (1, 21, 12)-style camera tensor."""
    svt = _svt()
    frame_indices = list(range(svt.NUM_FRAMES))[::4]
    cam = compute_pose_embedding(process_camera_trajectory(cam_data, frame_indices, cam_idx))
    cam = rearrange(cam, "b c d -> b (c d)")
    return cam.unsqueeze(0).to(dtype=dtype, device=device)


def time_embedding(pattern, device="cuda", dtype=torch.bfloat16):
    """World-time pattern -> bf16 (1, 81) tensor (bf16 matches the DMD-step contract)."""
    svt = _svt()
    return torch.tensor(svt.get_time_pattern(pattern, svt.NUM_FRAMES),
                        dtype=dtype).unsqueeze(0).to(device)


def identity_source_camera(device="cuda", dtype=torch.bfloat16):
    """v0's source camera = static identity (matches how bank middles were generated)."""
    svt = _svt()
    return svt.make_identity_src_camera().unsqueeze(0).to(dtype=dtype, device=device)


def build_batch(item, cam_data, device="cuda", dtype=torch.bfloat16):
    """Turn a ``data.TrainingItem`` into the K-middle dict ``steps.dmd_step_k`` consumes.

    Keys (fixed by the consumer): source_video, middle_videos[list], middle_cameras[list],
    mid_time_embeddings[list], source_camera, target_camera, src_time_embedding,
    tgt_time_embedding, prompt. v0/v2 fields are shared across the K middles.
    """
    middle_videos, middle_cameras, mid_times = [], [], []
    for m in item.middles:
        middle_videos.append(load_pixel_video(m.video_path, device, dtype))
        middle_cameras.append(camera_embedding(cam_data, m.cam_idx, device, dtype))
        mid_times.append(time_embedding(m.time_pattern, device, dtype))

    return {
        "source_video": load_pixel_video(item.video_path, device, dtype),
        "middle_videos": middle_videos,
        "middle_cameras": middle_cameras,
        "mid_time_embeddings": mid_times,
        "source_camera": identity_source_camera(device, dtype),
        "target_camera": camera_embedding(cam_data, item.target_cam_idx, device, dtype),
        "src_time_embedding": time_embedding(item.src_time_pattern, device, dtype),
        "tgt_time_embedding": time_embedding(item.target_time_pattern, device, dtype),
        "prompt": item.caption,
    }
