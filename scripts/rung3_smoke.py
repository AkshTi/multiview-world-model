"""Rung 3 GPU gates for the N-source model edits.

Two checks:

  (A) N=1 numerical equivalence. The edited model must not perturb released behavior:
        A1. dit.forward(single-tensor src)  ==  model_fn_wan_video(single-tensor src)
            -> the frame_positions RoPE hook (default None) is inert.
        A2. dit.forward([len-1 list] src)    ==  dit.forward(single-tensor src)
            -> the new N-source list path reduces exactly to the released 1-source path.
      Both run in eval()/no_grad; we report max|Δ| and require it ~0.

  (B) Two-source smoke. Build [target, src0, src1] (63 latent frames), run the edited
      DiTBlock N-source concat with gradients, backward, and assert the freeze mask
      (grads only on the SPT fine-tune modules, no leak).

Run on a GPU node from the repo root. Reuses build_pipe + loaders from rung2_smoke.
"""

import argparse
import json
import os
import sys

import torch
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_video_test as svt  # noqa: E402
from spacetimepilot.dataset.utils import (  # noqa: E402
    load_frames_using_imageio,
    process_camera_trajectory,
    compute_pose_embedding,
)
from spacetimepilot.model.spacetimepilot import model_fn_wan_video  # noqa: E402
from spacetimepilot.training import freeze, steps, latents as latent_utils  # noqa: E402
from scripts.rung2_smoke import build_pipe  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Rung 3 N-source equivalence + two-source smoke")
    p.add_argument("--video_path", required=True)
    p.add_argument("--caption", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--camera_file", default=svt.DEFAULT_CAMERA_FILE)
    p.add_argument("--dit_path", default=svt.DEFAULT_DIT_PATH)
    p.add_argument("--text_encoder_path", default=svt.DEFAULT_TEXT_PATH)
    p.add_argument("--vae_path", default=svt.DEFAULT_VAE_PATH)
    p.add_argument("--num_train_steps", type=int, default=1000)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--equiv_atol", type=float, default=1e-3)
    return p.parse_args()


def _load_video(args):
    video = load_frames_using_imageio(
        args.video_path, max_num_frames=svt.NUM_FRAMES, start_frame_id=0, interval=1,
        num_frames=svt.NUM_FRAMES, frame_process=svt.FRAME_PROCESS,
        target_width=svt.WIDTH, target_height=svt.HEIGHT,
    )
    if video is None:
        raise ValueError(f"Could not load video: {args.video_path}")
    return video.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")


def _camera_from_file(args, cam_type):
    with open(args.camera_file) as f:
        cam_data = json.load(f)
    frame_indices = list(range(svt.NUM_FRAMES))[::4]
    cam = compute_pose_embedding(
        process_camera_trajectory(cam_data, frame_indices, svt._parse_cam_type(cam_type)))
    cam = rearrange(cam, "b c d -> b (c d)")
    return cam.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")


def _time(pattern):
    # bf16 to match the released inference path (__call__ casts world-time to torch_dtype).
    return torch.tensor(svt.get_time_pattern(pattern, svt.NUM_FRAMES),
                        dtype=torch.bfloat16).unsqueeze(0).to("cuda")


def equivalence_check(pipe, args):
    """(A) N=1: edited paths must match the released single-source numerics."""
    dtype = pipe.torch_dtype
    video = _load_video(args)
    src_cam = svt.make_identity_src_camera().unsqueeze(0).to(dtype=dtype, device="cuda")
    tgt_cam = _camera_from_file(args, "cam01")
    src_time = _time("forward")
    tgt_time = _time("forward")

    with torch.no_grad():
        source_latent = latent_utils.encode_video_nograd(pipe, video)
        target_latent = latent_utils.encode_video_nograd(pipe, video)
        prompt_emb = pipe.encode_prompt(args.caption, positive=True)

        # A fixed noised target so all three forwards see identical input.
        pipe.scheduler.set_timesteps(args.num_train_steps, training=True, shift=args.sigma_shift)
        timestep, _, _ = steps.sample_training_timestep(pipe.scheduler, 1, "cuda")
        noise = torch.randn_like(target_latent)
        target_noised = pipe.scheduler.add_noise(target_latent, noise, timestep)
        latents_input = latent_utils.build_latent_input(target_noised, source_latent)
        ts = timestep.to(dtype=dtype)

        pipe.dit.eval()

        # A1: released inference entry point (unedited; single-tensor src).
        out_released = model_fn_wan_video(
            pipe.dit, latents_input, timestep=ts,
            src_camera_emb=src_cam, tgt_camera_emb=tgt_cam,
            src_time_embedding=src_time, tgt_time_embedding=tgt_time,
            context=prompt_emb["context"],
        )

        # A2 baseline: edited forward, single-tensor src.
        cam_tensor = {"tgt": tgt_cam, "src": src_cam}
        time_tensor = {"time_embedding_tgt": tgt_time, "time_embedding_src": src_time}
        out_forward_tensor = pipe.dit(
            latents_input, timestep=ts, cam_emb=cam_tensor,
            context=prompt_emb["context"], frame_time_embedding=time_tensor)

        # A2 list path: edited forward, length-1 list src (new N-source code path).
        cam_list = {"tgt": tgt_cam, "src": [src_cam]}
        time_list = {"time_embedding_tgt": tgt_time, "time_embedding_src": [src_time]}
        out_forward_list = pipe.dit(
            latents_input, timestep=ts, cam_emb=cam_list,
            context=prompt_emb["context"], frame_time_embedding=time_list)

    d_a1 = (out_forward_tensor.float() - out_released.float()).abs().max().item()
    d_a2 = (out_forward_list.float() - out_forward_tensor.float()).abs().max().item()
    return d_a1, d_a2


def two_source_smoke(pipe, args):
    """(B) N=2 forward + backward + grad-mask on the edited model."""
    video = _load_video(args)
    src_cam0 = svt.make_identity_src_camera().unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
    src_cam1 = _camera_from_file(args, "cam01")
    batch = {
        "target_video": video,
        "source_videos": [video, video],
        "target_camera": _camera_from_file(args, "cam01"),
        "source_cameras": [src_cam0, src_cam1],
        "tgt_time_embedding": _time("forward"),
        "src_time_embeddings": [_time("forward"), _time("reverse")],
        "prompt": args.caption,
    }

    trainable = freeze.set_trainable(pipe.dit, verbose=False)
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    pipe.scheduler.set_timesteps(args.num_train_steps, training=True, shift=args.sigma_shift)

    optimizer.zero_grad()
    out = steps.two_source_smoke_step(pipe, batch, pipe.scheduler)
    out["loss"].backward()
    assert pipe.dit.training, "DiT must be train() for checkpointing"
    freeze.assert_grad_mask(pipe.dit)
    optimizer.step()
    return out


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node"
    torch.cuda.reset_peak_memory_stats()

    pipe = build_pipe(args)

    d_a1, d_a2 = equivalence_check(pipe, args)
    equiv_ok = (d_a1 <= args.equiv_atol) and (d_a2 <= args.equiv_atol)

    out = two_source_smoke(pipe, args)
    peak = torch.cuda.max_memory_allocated() / 1e9

    print("\n================= RUNG 3 RESULT =================")
    print("  (A) N=1 equivalence:")
    print(f"      A1 forward-vs-released max|Δ| : {d_a1:.3e}")
    print(f"      A2 list-vs-tensor   max|Δ| : {d_a2:.3e}")
    print(f"      verdict (atol={args.equiv_atol:g})       : {'PASS' if equiv_ok else 'FAIL'}")
    print("  (B) two-source smoke:")
    print(f"      num sources                : {out['num_sources']}")
    print(f"      fused latent frames        : {out['fused_frames']}  (expect 63)")
    print(f"      loss (throwaway MSE)       : {out['loss'].item():.6f}  (finite={torch.isfinite(out['loss']).item()})")
    print(f"      grad mask                  : PASS (no leak; trainable grads present)")
    print(f"      peak VRAM                  : {peak:.2f} GB")
    print("================================================")
    if not equiv_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
