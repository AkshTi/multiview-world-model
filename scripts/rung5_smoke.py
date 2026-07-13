"""Rung 5 GPU smoke: the K=1 DMD loop end-to-end. See RUNG5_DESIGN.md.

Builds the three models — 2-source student (θ), frozen 1-source teacher (s_real), online
1-source fake-score net (s_fake) — pulls one cached middle v1 from the Rung-4 bank, and runs
a few DMD iterations. Each iteration: student one-step generation of v2, a
distribution-matching (G) update on the student, and a denoising (D) update on the fake-score
net.

Gates (see RUNG5_DESIGN.md §"Rung 5 gate"):
  * loss_G / loss_D finite over several iters;
  * student grad only on its unfrozen modules after the G-backward (no teacher/fake leak);
  * fake grad only on its unfrozen modules after the D-backward (no student/teacher leak);
  * teacher inert (no grad ever);
  * mean|v_real - v_fake| starts ~0 (teacher==fake at init) and grows;
  * fits in 48 GB.

Reuses build_pipe + loaders from rung2_smoke / rung3_smoke. Run on a GPU node from repo root.
"""

import argparse
import copy
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
from spacetimepilot.training import freeze, steps, middle_bank as mbk  # noqa: E402
from spacetimepilot.training.master import MasterAdamW  # noqa: E402
from scripts.rung2_smoke import build_pipe  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Rung 5 K=1 DMD loop smoke")
    p.add_argument("--video_path", required=True, help="v0 pixel video")
    p.add_argument("--v0_id", required=True, help="bank key for v0 (to pull its middles)")
    p.add_argument("--caption", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bank_dir", required=True)
    p.add_argument("--camera_file", default=svt.DEFAULT_CAMERA_FILE)
    p.add_argument("--target_cam", default="cam05", help="v2 novel-view camera")
    p.add_argument("--dit_path", default=svt.DEFAULT_DIT_PATH)
    p.add_argument("--text_encoder_path", default=svt.DEFAULT_TEXT_PATH)
    p.add_argument("--vae_path", default=svt.DEFAULT_VAE_PATH)
    p.add_argument("--num_iters", type=int, default=3)
    p.add_argument("--num_train_steps", type=int, default=1000)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--lr", type=float, default=1e-5, help="student (generator) lr")
    # AMP default (A7, 7/6 decision): fp32 trainable params (freeze.to_fp32_trainable) + bf16
    # autocast forwards make a realistic small lr representable, so 1e-4 is the default here.
    # --master switches to the OLD, now-retired recipe (bf16 params + fp32-master-copy
    # AdamW), kept only as an off-by-default regression comparison; that path still needs the
    # old large fake_lr=1e-2 hack (see master.py) to make its update representable, so the
    # default below resolves to 1e-2 automatically when --master is set.
    p.add_argument("--fake_lr", type=float, default=None,
                   help="fake-score lr (default: 1e-4 under AMP, 1e-2 under --master)")
    p.add_argument("--master", action="store_true",
                   help="[off by default] regression path: bf16 params + fp32-master-copy "
                        "AdamW (MasterAdamW, the pre-AMP numerics) instead of the AMP default")
    args = p.parse_args()
    if args.fake_lr is None:
        args.fake_lr = 1e-2 if args.master else 1e-4
    return args


def _load_video(path):
    video = load_frames_using_imageio(
        path, max_num_frames=svt.NUM_FRAMES, start_frame_id=0, interval=1,
        num_frames=svt.NUM_FRAMES, frame_process=svt.FRAME_PROCESS,
        target_width=svt.WIDTH, target_height=svt.HEIGHT,
    )
    if video is None:
        raise ValueError(f"Could not load video: {path}")
    return video.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")


def _camera_from_idx(cam_data, cam_idx):
    frame_indices = list(range(svt.NUM_FRAMES))[::4]
    cam = compute_pose_embedding(process_camera_trajectory(cam_data, frame_indices, cam_idx))
    cam = rearrange(cam, "b c d -> b (c d)")
    return cam.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")


def _time(pattern):
    # bf16 to match the released inference path (world-time is cast to torch_dtype).
    return torch.tensor(svt.get_time_pattern(pattern, svt.NUM_FRAMES),
                        dtype=torch.bfloat16).unsqueeze(0).to("cuda")


def build_batch(args):
    """v0 = source video; v1 = one cached middle from the bank; v2 = a novel target view."""
    cached = mbk.list_middles(args.bank_dir, args.v0_id)
    if not cached:
        raise RuntimeError(
            f"no cached middles for v0_id={args.v0_id} in {args.bank_dir}; run build_middle_bank first")
    idx = cached[0]
    v1_path, meta_path = mbk.middle_paths(args.bank_dir, args.v0_id, idx)
    meta = mbk.load_meta(meta_path)
    print(f"Using middle idx={idx}: cam{meta['cam_idx']:02d}, {v1_path}")

    with open(args.camera_file) as f:
        cam_data = json.load(f)

    v0_video = _load_video(args.video_path)
    v1_video = _load_video(v1_path)

    # OPEN item (Thursday / Rung 5): v1 conditions v2 as a *source*, but was generated as a
    # *target* under meta['cam_idx']. Source vs. target cameras use different SPT embeddings.
    # The smoke uses the trajectory embedding for v1's source camera; the exact source
    # representation (and the RoPE 3D-position convention) is still to be settled.
    return {
        "source_video": v0_video,
        "middle_video": v1_video,
        "source_camera": svt.make_identity_src_camera().unsqueeze(0).to(dtype=torch.bfloat16, device="cuda"),
        "middle_camera": _camera_from_idx(cam_data, meta["cam_idx"]),
        "target_camera": _camera_from_idx(cam_data, svt._parse_cam_type(args.target_cam)),
        "src_time_embedding": _time("forward"),
        "mid_time_embedding": _time(meta.get("time_pattern", "forward")),
        "tgt_time_embedding": _time("forward"),
        "prompt": args.caption,
    }


def _assert_no_grad(module, label):
    for name, p in module.named_parameters():
        if p.grad is not None and bool(torch.any(p.grad != 0)):
            raise AssertionError(f"{label} param has nonzero grad (leak): {name}")


def _assert_connected(module, label):
    """At least one trainable param has a populated .grad (graph reached it), even if zero.

    At iter 0 the teacher and fake net are identical, so the DMD arrow is exactly 0 and the
    student's grads are a *zero* tensor — connected but not nonzero (see RUNG5_DESIGN.md
    "arrow ≈ 0 at step 0"). Connectivity, not magnitude, is the right iter-0 check.
    """
    for name, p in module.named_parameters():
        if p.requires_grad and p.grad is not None:
            return
    raise AssertionError(f"{label}: no trainable param is connected to the graph (all grads None)")


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node"
    torch.cuda.reset_peak_memory_stats()

    pipe = build_pipe(args)

    # Three models from identical released+ckpt weights (deepcopy BEFORE unfreezing student).
    # Teacher starts on CPU; dmd_step_k1 moves it to GPU only for its one no_grad forward,
    # keeping it off-device during the student/fake backprop (memory).
    teacher_dit = copy.deepcopy(pipe.dit).to("cpu").requires_grad_(False).eval()
    fake_dit = copy.deepcopy(pipe.dit)

    student_trainable = freeze.set_trainable(pipe.dit, verbose=True)      # θ
    fake_trainable = freeze.set_trainable(fake_dit, verbose=False)        # φ (same mask; see design note)
    if args.master:
        # Regression path (off by default): retired pure-bf16 + fp32-master-copy numerics.
        # Params stay bf16; MasterAdamW bolts on fp32 masters (see master.py). Kept only so
        # this smoke can still reproduce the pre-AMP behavior for comparison.
        opt_G = MasterAdamW(student_trainable, lr=args.lr)
        opt_D = MasterAdamW(fake_trainable, lr=args.fake_lr)
        print(f"optimizer: MasterAdamW (bf16 params + fp32 master copies, REGRESSION path)  "
              f"student_lr={args.lr:g}  fake_lr={args.fake_lr:g}")
    else:
        # AMP default (A7, 7/6): cast the trainable subset to fp32 in place; the forward runs
        # under bf16 autocast (steps._dit_velocity). Plain AdamW now works at a realistic lr.
        student_trainable = freeze.to_fp32_trainable(pipe.dit)
        fake_trainable = freeze.to_fp32_trainable(fake_dit)
        opt_G = torch.optim.AdamW(student_trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
        opt_D = torch.optim.AdamW(fake_trainable, lr=args.fake_lr, betas=(0.9, 0.999), weight_decay=0.0)
        print(f"optimizer: AdamW (fp32 trainable params + bf16 autocast, AMP default)  "
              f"student_lr={args.lr:g}  fake_lr={args.fake_lr:g}")

    pipe.scheduler.set_timesteps(args.num_train_steps, training=True, shift=args.sigma_shift)
    batch = build_batch(args)

    def after_g():
        # No frozen-param leak, and the student graph is connected. Do NOT require a nonzero
        # student grad: at iter 0 teacher==fake => arrow==0 => grads are a zero tensor.
        freeze.assert_grad_mask(pipe.dit, require_any_nonzero=False)
        _assert_connected(pipe.dit, "student(G)")
        _assert_no_grad(teacher_dit, "teacher")
        _assert_no_grad(fake_dit, "fake(during G)")
        if not args.master:
            # A7 grad-clip 1.0 (AMP default only; --master is left unmodified as a clean
            # regression comparison against the pre-AMP recipe).
            torch.nn.utils.clip_grad_norm_(student_trainable, max_norm=1.0)

    def after_d():
        # The D-step is a real denoising MSE, so the fake net must get nonzero grads.
        freeze.assert_grad_mask(fake_dit)      # fake: grads only on unfrozen modules
        _assert_no_grad(teacher_dit, "teacher")
        _assert_no_grad(pipe.dit, "student(during D)")
        if not args.master:
            torch.nn.utils.clip_grad_norm_(fake_trainable, max_norm=1.0)

    history = []
    for it in range(args.num_iters):
        out = steps.dmd_step_k1(
            pipe, teacher_dit, fake_dit, batch, pipe.scheduler, opt_G, opt_D,
            after_g=after_g, after_d=after_d)
        history.append(out)
        print(f"  iter {it}: loss_G={out['loss_G']:+.4e}  loss_D={out['loss_D']:.4e}  "
              f"arrow|v_real-v_fake|={out['arrow_abs']:.4e}  "
              f"(sigma_T={out['sigma_T']:.3f}, sigma_g={out['sigma_g']:.3f})")

    peak = torch.cuda.max_memory_allocated() / 1e9
    losses_finite = all(
        torch.isfinite(torch.tensor([h["loss_G"], h["loss_D"]])).all().item() for h in history)
    # iter-0 arrow is exactly 0 (teacher==fake); the loop is "live" once the fake net
    # separates from the teacher and the arrow (hence the student's DMD signal) turns nonzero.
    arrow_became_nonzero = any(h["arrow_abs"] > 0 for h in history)
    first_nonzero = next((i for i, h in enumerate(history) if h["arrow_abs"] > 0), None)

    print("\n================= RUNG 5 RESULT =================")
    print(f"  iterations                 : {args.num_iters}")
    print(f"  arrow first nonzero at iter: {first_nonzero if first_nonzero is not None else 'never'}")
    print(f"  loss_G / loss_D finite     : {'PASS' if losses_finite else 'FAIL'}")
    print(f"  student grad mask (G)      : PASS (no leak; graph connected)")
    print(f"  fake grad mask (D)         : PASS (no leak; trainable grads present)")
    print(f"  teacher inert              : PASS (no grad)")
    print(f"  arrow |v_real-v_fake|      : {history[0]['arrow_abs']:.3e} -> {history[-1]['arrow_abs']:.3e}"
          f"  ({'became nonzero' if arrow_became_nonzero else 'STILL ZERO'})")
    print(f"  peak VRAM                  : {peak:.2f} GB")
    print("=================================================")
    if not (losses_finite and arrow_became_nonzero):
        sys.exit(1)


if __name__ == "__main__":
    main()
