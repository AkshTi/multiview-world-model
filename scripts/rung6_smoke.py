"""Rung 6 GPU smoke: K>1 DMD with Monte-Carlo marginalization over middles.

Generalizes the Rung-5 K=1 loop to K banked middles via gradient accumulation (see
steps.dmd_step_k and RUNG5_DESIGN.md §"Middle handling"). Each middle is an independent
sample v2_hat_k = G_θ(v0, v1_k, z_k); the student's marginal-over-v1 update is the mean of
the per-middle DMD arrows, accumulated over k with one student graph live at a time.

Gates (extend Rung 5):
  * loss_G / loss_D finite;
  * student grad only on unfrozen modules after the accumulated G-backward (no leak);
  * fake grad only on unfrozen modules after the accumulated D-backward (no leak);
  * teacher inert;
  * mean|v_real - v_fake| over the K middles becomes nonzero as the fake net separates;
  * K middles actually consumed.

Reuses build_pipe + loaders from rung2 / rung5_smoke. Run on a GPU node from repo root.
"""

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_video_test as svt  # noqa: E402
from spacetimepilot.training import freeze, steps, middle_bank as mbk  # noqa: E402
from scripts.rung2_smoke import build_pipe  # noqa: E402
from scripts.rung5_smoke import (  # noqa: E402
    _load_video, _camera_from_idx, _time, _assert_no_grad, _assert_connected,
)


def parse_args():
    p = argparse.ArgumentParser(description="Rung 6 K>1 DMD marginalization smoke")
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
    p.add_argument("--num_middles", type=int, default=2, help="K: middles to marginalize over")
    p.add_argument("--num_iters", type=int, default=3)
    p.add_argument("--num_train_steps", type=int, default=1000)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    # AMP default (A7, 7/6): fp32 trainable params (freeze.to_fp32_trainable) + bf16 autocast
    # (steps._dit_velocity) make these realistic lrs representable — no MasterAdamW bf16-hack
    # needed here (contrast the old fake_lr=1e-2 kept in rung5_smoke's --master regression path).
    p.add_argument("--lr", type=float, default=1e-5, help="student (generator) lr")
    p.add_argument("--fake_lr", type=float, default=1e-4, help="fake-score lr")
    p.add_argument("--offload_teacher", action="store_true",
                   help="keep the frozen teacher on CPU except for its forward (only needed on tight VRAM)")
    return p.parse_args()


def build_batch(args):
    """v0 = source; middles = K cached bank middles; v2 = a novel target view."""
    cached = mbk.list_middles(args.bank_dir, args.v0_id)
    if len(cached) < args.num_middles:
        raise RuntimeError(
            f"need {args.num_middles} middles for v0_id={args.v0_id} but bank has {len(cached)} "
            f"({cached}); run build_middle_bank with more --num_middles")
    idxs = cached[:args.num_middles]

    with open(args.camera_file) as f:
        cam_data = json.load(f)

    middle_videos, middle_cameras, mid_times = [], [], []
    for idx in idxs:
        v1_path, meta_path = mbk.middle_paths(args.bank_dir, args.v0_id, idx)
        meta = mbk.load_meta(meta_path)
        print(f"  middle idx={idx}: cam{meta['cam_idx']:02d}, {os.path.basename(v1_path)}")
        middle_videos.append(_load_video(v1_path))
        # OPEN item (Thursday/Rung 5): v1 as a *source* vs. *target* camera representation.
        middle_cameras.append(_camera_from_idx(cam_data, meta["cam_idx"]))
        mid_times.append(_time(meta.get("time_pattern", "forward")))

    return {
        "source_video": _load_video(args.video_path),
        "middle_videos": middle_videos,
        "middle_cameras": middle_cameras,
        "mid_time_embeddings": mid_times,
        "source_camera": svt.make_identity_src_camera().unsqueeze(0).to(dtype=torch.bfloat16, device="cuda"),
        "target_camera": _camera_from_idx(cam_data, svt._parse_cam_type(args.target_cam)),
        "src_time_embedding": _time("forward"),
        "tgt_time_embedding": _time("forward"),
        "prompt": args.caption,
    }


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node"
    torch.cuda.reset_peak_memory_stats()

    pipe = build_pipe(args)

    teacher_start = "cpu" if args.offload_teacher else "cuda"
    teacher_dit = copy.deepcopy(pipe.dit).to(teacher_start).requires_grad_(False).eval()
    fake_dit = copy.deepcopy(pipe.dit)

    freeze.set_trainable(pipe.dit, verbose=True)
    freeze.set_trainable(fake_dit, verbose=False)
    # AMP default (A7, 7/6): cast the trainable subset to fp32 in place; the forward runs
    # under bf16 autocast (steps._dit_velocity). Plain AdamW at a realistic lr.
    student_trainable = freeze.to_fp32_trainable(pipe.dit)
    fake_trainable = freeze.to_fp32_trainable(fake_dit)
    opt_G = torch.optim.AdamW(student_trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    opt_D = torch.optim.AdamW(fake_trainable, lr=args.fake_lr, betas=(0.9, 0.999), weight_decay=0.0)
    print(f"optimizer: AdamW (fp32 trainable params + bf16 autocast, AMP default)  "
          f"student_lr={args.lr:g}  fake_lr={args.fake_lr:g}")

    pipe.scheduler.set_timesteps(args.num_train_steps, training=True, shift=args.sigma_shift)
    print(f"Building batch with K={args.num_middles} middles:")
    batch = build_batch(args)

    def after_g():
        freeze.assert_grad_mask(pipe.dit, require_any_nonzero=False)
        _assert_connected(pipe.dit, "student(G)")
        _assert_no_grad(teacher_dit, "teacher")
        _assert_no_grad(fake_dit, "fake(during G)")
        torch.nn.utils.clip_grad_norm_(student_trainable, max_norm=1.0)  # A7 grad-clip 1.0

    def after_d():
        freeze.assert_grad_mask(fake_dit)
        _assert_no_grad(teacher_dit, "teacher")
        _assert_no_grad(pipe.dit, "student(during D)")
        torch.nn.utils.clip_grad_norm_(fake_trainable, max_norm=1.0)

    history = []
    for it in range(args.num_iters):
        out = steps.dmd_step_k(
            pipe, teacher_dit, fake_dit, batch, pipe.scheduler, opt_G, opt_D,
            after_g=after_g, after_d=after_d, offload_teacher=args.offload_teacher)
        history.append(out)
        arrows = ", ".join(f"{a:.3e}" for a in out["arrows"])
        print(f"  iter {it}: loss_G={out['loss_G']:+.4e}  loss_D={out['loss_D']:.4e}  "
              f"mean_arrow={out['arrow_abs']:.4e}  per-middle=[{arrows}]")

    peak = torch.cuda.max_memory_allocated() / 1e9
    losses_finite = all(
        torch.isfinite(torch.tensor([h["loss_G"], h["loss_D"]])).all().item() for h in history)
    arrow_became_nonzero = any(h["arrow_abs"] > 0 for h in history)

    print("\n================= RUNG 6 RESULT =================")
    print(f"  K (middles marginalized)   : {history[0]['K']}")
    print(f"  iterations                 : {args.num_iters}")
    print(f"  loss_G / loss_D finite     : {'PASS' if losses_finite else 'FAIL'}")
    print(f"  student grad mask (G, acc) : PASS (no leak; graph connected)")
    print(f"  fake grad mask (D, acc)    : PASS (no leak; trainable grads present)")
    print(f"  teacher inert              : PASS (no grad)")
    print(f"  mean arrow |v_real-v_fake| : {history[0]['arrow_abs']:.3e} -> {history[-1]['arrow_abs']:.3e}"
          f"  ({'became nonzero' if arrow_became_nonzero else 'STILL ZERO'})")
    print(f"  peak VRAM                  : {peak:.2f} GB")
    print("=================================================")
    if not (losses_finite and arrow_became_nonzero):
        sys.exit(1)


if __name__ == "__main__":
    main()
