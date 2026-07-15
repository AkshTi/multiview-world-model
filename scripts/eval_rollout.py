"""Generate B0 / B1 / checkpoint rollouts at the fixed eval tuples (PLAN_DMD_TRAINING.md
section 8, "Baselines BEFORE Run 1").

Three --method values, one script (so all three are produced by the exact same tensor path
per family, keeping the comparison fair):

  b0   Released SPT, INDEPENDENT view: v0 -> a2, no knowledge of v1 at all. Reuses
       middle_bank.generate_middle -- the SAME function (same pipe class, same cfg=1/20-step
       settings) used to build the middle bank, just targeting the eval tuple's a2 camera
       instead of a bank crossing camera. This is the floor: what agreement do you get with
       ZERO cross-view mechanism.

  b1   "Student-at-init" (2-source architecture, released weights, no DMD training): v1 =
       the cached heldout-bank middle at v1_cam, a2_hat = G_released(v0, v1, a2). Reuses
       train_dmd.sample_v2 with a manually-built data.TrainingItem (no random sampler --
       eval tuples are fixed) on the training-style pipe (rung2_smoke.build_pipe). This is
       the second floor: does v1-conditioning alone (no training) already buy consistency.

  ckpt Same 2-source path as b1, but loads a train_dmd.py checkpoint's student_trainable (or
       its EMA shadow) onto the pipe first. Use this after Run 1 has produced ckpt_*.pt to
       eval any step against the b0/b1 baselines with the identical tuple set.

Needs a heldout middle bank first (scripts/build_heldout_bank.sbatch) -- v1_cam middles for
the 8 held-out v0s must already be cached; this script never generates v1, only a2.

Usage:
  python -u scripts/eval_rollout.py --method b0 --out_dir <dir>
  python -u scripts/eval_rollout.py --method b1 --out_dir <dir>
  python -u scripts/eval_rollout.py --method ckpt --ckpt <ckpt_path> --tag step1000 --out_dir <dir>
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import single_video_test as svt  # noqa: E402
import build_middle_bank as bmb  # noqa: E402  (b0: reuses generate_middle + its inference pipe)
from spacetimepilot.training import middle_bank as mbk  # noqa: E402
from spacetimepilot.training import data, batch as batch_utils, freeze  # noqa: E402
from spacetimepilot.utils.misc import save_video  # noqa: E402
from scripts.rung2_smoke import build_pipe  # noqa: E402  (b1/ckpt: training-style pipe)
from scripts.train_dmd import sample_v2, _load_trainable_state  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Eval rollout: B0 / B1 / checkpoint at fixed eval tuples")
    p.add_argument("--method", required=True, choices=["b0", "b1", "ckpt"])
    p.add_argument("--tuples", default="config/eval/pilot_eval_tuples.json")
    p.add_argument("--split", default="config/data/pilot_split.json")
    p.add_argument("--bank_dir", default="/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank_cfg1")
    p.add_argument("--camera_file", default=svt.DEFAULT_CAMERA_FILE)
    p.add_argument("--ckpt", default="checkpoints/SpacetimePilot_1.3B_v1.ckpt", help="released SPT ckpt")
    p.add_argument("--train_ckpt", default=None, help="[--method ckpt] a train_dmd.py ckpt_*.pt to load")
    p.add_argument("--use_ema", action="store_true", help="[--method ckpt] load the EMA shadow, not raw student")
    p.add_argument("--tag", default=None, help="subfolder name for this rollout (default = --method)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_inference_steps", type=int, default=20, help="[b0 only] matches bank convention")
    p.add_argument("--cfg_scale", type=float, default=1.0, help="[b0 only] matches bank convention (A4)")
    p.add_argument("--seed0", type=int, default=1000, help="[b0 only] disjoint from bank-build seeds")
    p.add_argument("--dit_path", default=svt.DEFAULT_DIT_PATH)
    p.add_argument("--text_encoder_path", default=svt.DEFAULT_TEXT_PATH)
    p.add_argument("--vae_path", default=svt.DEFAULT_VAE_PATH)
    return p.parse_args()


def find_middle_by_cam(bank_dir, v0_id, cam_idx):
    """Bank idx of the cached middle at cam_idx for v0_id, or None if not cached."""
    for idx in mbk.list_middles(bank_dir, v0_id):
        _, meta_path = mbk.middle_paths(bank_dir, v0_id, idx)
        if int(mbk.load_meta(meta_path)["cam_idx"]) == cam_idx:
            return idx
    return None


def load_tuples(path, v0_lookup):
    with open(path) as f:
        payload = json.load(f)
    out = []
    for t in payload["tuples"]:
        out.append({**t, "video_path": v0_lookup[t["v0_id"]].video_path,
                    "caption": v0_lookup[t["v0_id"]].caption})
    return out


def run_b0(args, tuples, out_dir):
    from spacetimepilot.dataset.utils import load_frames_using_imageio

    pipe = bmb.build_inference_pipe(args)
    with open(args.camera_file) as f:
        cam_data = json.load(f)
    manifest = []
    for i, t in enumerate(tuples):
        video = load_frames_using_imageio(
            t["video_path"], max_num_frames=svt.NUM_FRAMES, start_frame_id=0, interval=1,
            num_frames=svt.NUM_FRAMES, frame_process=svt.FRAME_PROCESS,
            target_width=svt.WIDTH, target_height=svt.HEIGHT,
        )
        source_video = video.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        source_camera = svt.make_identity_src_camera().unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        src_time = torch.tensor(svt.get_time_pattern("forward", svt.NUM_FRAMES),
                                dtype=torch.float32).unsqueeze(0).to("cuda")
        mid_time = torch.tensor(svt.get_time_pattern("forward", svt.NUM_FRAMES),
                                dtype=torch.float32).unsqueeze(0).to("cuda")
        a2_camera = bmb.target_camera_from_idx(cam_data, t["a2_cam"])
        frames = mbk.generate_middle(
            pipe, source_video, source_camera, a2_camera, src_time, mid_time,
            t["caption"], bmb.NEGATIVE_PROMPT, seed=args.seed0 + i,
            num_inference_steps=args.num_inference_steps, cfg_scale=args.cfg_scale)
        video_out = os.path.join(out_dir, f"{t['tuple_id']}.mp4")
        save_video(frames, video_out, fps=30, quality=5,
                   ffmpeg_params=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        manifest.append({**t, "method": "b0", "a2_video": video_out})
        print(f"[{i + 1}/{len(tuples)}] b0 {t['tuple_id']}: v0={t['v0_id']} a2_cam={t['a2_cam']} -> {video_out}")
    return manifest


def run_student(args, tuples, out_dir):
    pipe = build_pipe(args)
    if args.method == "ckpt":
        assert args.train_ckpt, "--method ckpt requires --train_ckpt"
        freeze.set_trainable(pipe.dit, verbose=False)
        freeze.to_fp32_trainable(pipe.dit)  # ckpts save fp32 trainables (A7) -- upcast BEFORE
                                            # loading, exactly like train_dmd.py's own resume path,
                                            # or the copy_ silently downcasts fp32 -> bf16 on load.
        ck = torch.load(args.train_ckpt, map_location="cpu")
        key = "ema" if args.use_ema else "student_trainable"
        _load_trainable_state(pipe.dit, ck[key])
        print(f"loaded {key} from {args.train_ckpt} (step {ck.get('step')})")
    pipe.dit.eval()
    pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
    cam_data = batch_utils.load_camera_data(args.camera_file)

    manifest = []
    for i, t in enumerate(tuples):
        idx = find_middle_by_cam(args.bank_dir, t["v0_id"], t["v1_cam"])
        if idx is None:
            raise RuntimeError(
                f"no cached middle at cam{t['v1_cam']:02d} for v0_id={t['v0_id']} in {args.bank_dir} "
                f"-- run scripts/build_heldout_bank.sbatch first")
        v1_path, meta_path = mbk.middle_paths(args.bank_dir, t["v0_id"], idx)
        meta = mbk.load_meta(meta_path)
        item = data.TrainingItem(
            v0_id=t["v0_id"], video_path=t["video_path"], caption=t["caption"],
            target_cam_idx=t["a2_cam"], target_time_pattern="forward", src_time_pattern="forward",
            middles=[data.MiddleRef(idx=idx, cam_idx=t["v1_cam"],
                                    time_pattern=meta.get("time_pattern", "forward"),
                                    video_path=v1_path)])
        with torch.no_grad():
            b = batch_utils.build_batch(item, cam_data, device=pipe.device, dtype=pipe.torch_dtype)
            frames = sample_v2(pipe, b, pipe.scheduler)
        video_out = os.path.join(out_dir, f"{t['tuple_id']}.mp4")
        save_video(frames, video_out, fps=30, quality=5)
        manifest.append({**t, "method": args.method, "a2_video": video_out, "v1_video": v1_path})
        print(f"[{i + 1}/{len(tuples)}] {args.method} {t['tuple_id']}: v0={t['v0_id']} "
              f"v1_cam={t['v1_cam']} a2_cam={t['a2_cam']} -> {video_out}")
    return manifest


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node"
    tag = args.tag or args.method
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    entries = data.load_split(args.split, "heldout")
    v0_lookup = {e.v0_id: e for e in entries}
    tuples = load_tuples(args.tuples, v0_lookup)
    print(f"method={args.method}  tag={tag}  tuples={len(tuples)}  out_dir={out_dir}")

    if args.method == "b0":
        manifest = run_b0(args, tuples, out_dir)
    else:
        manifest = run_student(args, tuples, out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"method": args.method, "tag": tag, "train_ckpt": args.train_ckpt,
                   "use_ema": args.use_ema, "entries": manifest}, f, indent=2)
    print(f"\nwrote {len(manifest)} rollouts + manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
