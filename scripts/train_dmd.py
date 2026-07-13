"""DMD training entrypoint (PLAN §7 item 5, numerics A7). Run 1 = pilot DMD, no stitching.

Ties the pieces together:
  data.V0Sampler(step) -> batch.build_batch -> steps.dmd_step_k(student, teacher, fake)
with AMP numerics (freeze.to_fp32_trainable + autocast, already in steps._dit_velocity), EMA
on the student trainables, JSONL metrics, checkpoint/resume, and a step-5 peak-memory
auto-abort so a bad memory budget dies in seconds, not at hour 20.

Resume is exact by construction: BOTH the data draw (data.step_generator) and the per-step
noise/timestep draw (we re-seed the global RNG from the step index at the top of every step)
are pure functions of the step, so restoring only {step, params, opt, ema} reproduces the
stream — no RNG blobs to carry.

GPU-bound: needs a GPU node + SPT ckpt + a middle bank with >= k_step middles for the sampled
v0s. NOT login-node runnable (imports resolve on CPU; the loop does not). Stitching (Rung 7)
is intentionally NOT wired — `lambda_stitch` is an inert hook for the post-Run-1 item 8.

  python -u scripts/train_dmd.py --config config/train/run1.yaml
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_video_test as svt  # noqa: E402
from spacetimepilot.training import (  # noqa: E402
    freeze, steps, data, batch as batch_utils, latents as latent_utils, score as score_utils,
)
from spacetimepilot.utils.misc import save_video  # noqa: E402
from scripts.rung2_smoke import build_pipe  # noqa: E402  (student pipe: no vram management)

_STEP_NOISE_PRIME = 2_654_435_761  # decorrelate per-step noise seed from the sampler's stream


# --------------------------------------------------------------------------------------- #
# Config                                                                                    #
# --------------------------------------------------------------------------------------- #

DEFAULTS = dict(
    # data
    split="config/data/pilot_split.json", bank_dir=None, camera_file=svt.DEFAULT_CAMERA_FILE,
    k_step=2, base_seed=0, camera_pool=list(range(1, 11)),
    # model / ckpt
    ckpt=None, dit_path=svt.DEFAULT_DIT_PATH, text_encoder_path=svt.DEFAULT_TEXT_PATH,
    vae_path=svt.DEFAULT_VAE_PATH,
    # optim (A7)
    student_lr=1e-5, fake_lr=1e-4, grad_clip=1.0, ema_decay=0.999,
    num_train_steps=1000, sigma_shift=5.0, max_steps=3000,
    use_gradient_checkpointing=True, offload_teacher=False, ema_device="cuda",
    # stitching hook (inert until Rung 7 / item 8)
    lambda_stitch=0.0,
    # io / cadence
    out_dir=None, ckpt_every=250, sample_every=250, keep_last=2, log_every=1,
    # safety
    mem_budget_gb=72.0,  # 90% of an 80 GB card (§10); set to your card before running
)


def load_config():
    ap = argparse.ArgumentParser(description="DMD training (Run 1)")
    ap.add_argument("--config", default=None, help="yaml/json config; CLI flags override it")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(v, list):
            ap.add_argument(f"--{k}", nargs="+", type=int, default=None)
        else:
            ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            loaded = _load_yaml_or_json(f.read(), args.config)
        cfg.update({k: v for k, v in loaded.items() if k in cfg})
    for k in DEFAULTS:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    for required in ("bank_dir", "ckpt", "out_dir"):
        if not cfg[required]:
            raise SystemExit(f"--{required} is required (set it in the config or on the CLI)")
    return cfg


def _load_yaml_or_json(text, path):
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML not installed; use a .json config or `pip install pyyaml`")
    return yaml.safe_load(text)


def git_sha(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------- #
# EMA                                                                                       #
# --------------------------------------------------------------------------------------- #

class EMA:
    """Exponential moving average of the student's trainable params (fp32 shadow)."""

    def __init__(self, dit, decay, device="cuda"):
        self.decay = float(decay)
        self.device = device
        self.shadow = {n: p.detach().clone().float().to(device)
                       for n, p in dit.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, dit):
        for n, p in dit.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach().float().to(self.device),
                                                     alpha=1.0 - self.decay)

    @contextmanager
    def swapped_in(self, dit):
        """Temporarily load the EMA weights into ``dit`` (for sampling), then restore."""
        backup = {n: p.detach().clone() for n, p in dit.named_parameters() if n in self.shadow}
        try:
            with torch.no_grad():
                for n, p in dit.named_parameters():
                    if n in self.shadow:
                        p.copy_(self.shadow[n].to(p.dtype))
            yield
        finally:
            with torch.no_grad():
                for n, p in dit.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])

    def state_dict(self):
        return {n: t.cpu() for n, t in self.shadow.items()}

    def load_state_dict(self, sd):
        self.shadow = {n: t.clone().to(self.device) for n, t in sd.items()}


# --------------------------------------------------------------------------------------- #
# Checkpoint (compact: backbone is the fixed released+ckpt, only trainables + state move)   #
# --------------------------------------------------------------------------------------- #

def _trainable_state(dit):
    return {n: p.detach().cpu() for n, p in dit.named_parameters() if p.requires_grad}


def _load_trainable_state(dit, sd):
    with torch.no_grad():
        params = dict(dit.named_parameters())
        for n, t in sd.items():
            params[n].copy_(t.to(params[n].dtype).to(params[n].device))


def save_ckpt(path, step, student, fake, opt_G, opt_D, ema, cfg):
    tmp = path + ".tmp"
    torch.save({
        "step": step, "cfg": cfg,
        "student_trainable": _trainable_state(student),
        "fake_trainable": _trainable_state(fake),
        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
        "ema": ema.state_dict(),
    }, tmp)
    os.replace(tmp, path)  # atomic


def load_ckpt(path, student, fake, opt_G, opt_D, ema):
    ck = torch.load(path, map_location="cpu")
    _load_trainable_state(student, ck["student_trainable"])
    _load_trainable_state(fake, ck["fake_trainable"])
    opt_G.load_state_dict(ck["opt_G"])
    opt_D.load_state_dict(ck["opt_D"])
    ema.load_state_dict(ck["ema"])
    return ck["step"]


def latest_ckpt(out_dir):
    if not os.path.isdir(out_dir):
        return None
    cks = [f for f in os.listdir(out_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    if not cks:
        return None
    return os.path.join(out_dir, max(cks, key=lambda f: int(f[5:-3])))


def prune_ckpts(out_dir, keep_last):
    cks = sorted((int(f[5:-3]), f) for f in os.listdir(out_dir)
                 if f.startswith("ckpt_") and f.endswith(".pt"))
    for _, f in cks[:-keep_last] if keep_last > 0 else []:
        os.remove(os.path.join(out_dir, f))


# --------------------------------------------------------------------------------------- #
# Sampling (viz only; wrapped so a decode failure never kills training)                     #
# --------------------------------------------------------------------------------------- #

@torch.no_grad()
def sample_v2(pipe, batch, scheduler):
    """One-step student generation of v2 under the CURRENT student weights (EMA swapped by caller)."""
    device, dtype = pipe.device, pipe.torch_dtype
    v0 = latent_utils.encode_video_nograd(pipe, batch["source_video"])
    v1 = latent_utils.encode_video_nograd(pipe, batch["middle_videos"][0])
    prompt = pipe.encode_prompt(batch["prompt"], positive=True)["context"]
    B = v0.shape[0]
    sig_idx = int(torch.argmax(scheduler.sigmas))
    t_T = scheduler.timesteps[sig_idx].to(device).repeat(B)
    sigma_T = scheduler.sigmas[sig_idx].to(device)
    z = torch.randn((B, *v0.shape[1:]), dtype=v0.dtype, device=device)
    v_pred = steps._dit_velocity(
        pipe.dit, z, [v0, v1], batch["target_camera"], [batch["source_camera"], batch["middle_cameras"][0]],
        batch["tgt_time_embedding"], [batch["src_time_embedding"], batch["mid_time_embeddings"][0]],
        prompt, t_T, latent_utils.LATENT_FRAMES_PER_VIDEO, dtype, device, use_gradient_checkpointing=False)
    v2_hat = score_utils.x0_from_velocity(z, v_pred, sigma_T)
    return latent_utils.decode_video_nograd(pipe, v2_hat)


def maybe_sample(pipe, ema, scheduler, sampler, cam_data, out_dir, step, n=4):
    try:
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        pipe.dit.eval()
        with ema.swapped_in(pipe.dit):
            for j in range(n):
                item = sampler.sample(10**6 + j)  # fixed tuples, disjoint from training steps
                b = batch_utils.build_batch(item, cam_data, device=pipe.device, dtype=pipe.torch_dtype)
                frames = sample_v2(pipe, b, scheduler)
                save_video(frames, os.path.join(out_dir, "samples", f"s{step:06d}_{j}.mp4"),
                           fps=30, quality=5)
        pipe.dit.train()
    except Exception as e:  # viz must never crash a long run
        print(f"[warn] sampling at step {step} failed: {e}")


# --------------------------------------------------------------------------------------- #
# Train                                                                                     #
# --------------------------------------------------------------------------------------- #

def seed_step(base_seed, step):
    """Re-seed the global RNG from the step index so per-step noise is resume-exact."""
    s = (int(base_seed) * _STEP_NOISE_PRIME + int(step)) % (2**63)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    cfg = load_config()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # --- models: student (θ), frozen teacher (s_real), fake-score net (φ) ---
    class _A:  # build_pipe reads attributes off args
        pass
    a = _A(); a.__dict__.update(cfg)
    pipe = build_pipe(a)

    import copy
    teacher = copy.deepcopy(pipe.dit).to("cpu" if cfg["offload_teacher"] else "cuda")
    teacher.requires_grad_(False).eval()
    fake = copy.deepcopy(pipe.dit)

    freeze.set_trainable(pipe.dit, verbose=True)
    freeze.set_trainable(fake, verbose=False)
    student_trainable = freeze.to_fp32_trainable(pipe.dit)
    fake_trainable = freeze.to_fp32_trainable(fake)
    opt_G = torch.optim.AdamW(student_trainable, lr=cfg["student_lr"], betas=(0.9, 0.999), weight_decay=0.0)
    opt_D = torch.optim.AdamW(fake_trainable, lr=cfg["fake_lr"], betas=(0.9, 0.999), weight_decay=0.0)
    ema = EMA(pipe.dit, cfg["ema_decay"], device=cfg["ema_device"])

    pipe.scheduler.set_timesteps(cfg["num_train_steps"], training=True, shift=cfg["sigma_shift"])

    entries = data.load_split(cfg["split"], "train")
    sampler = data.V0Sampler(entries, bank_dir=cfg["bank_dir"], k_step=cfg["k_step"],
                             camera_pool=tuple(cfg["camera_pool"]), base_seed=cfg["base_seed"])
    cam_data = batch_utils.load_camera_data(cfg["camera_file"])

    # --- resume ---
    start = 0
    resume = latest_ckpt(cfg["out_dir"])
    if resume:
        start = load_ckpt(resume, pipe.dit, fake, opt_G, opt_D, ema) + 1
        print(f"resumed from {resume} at step {start}")

    log_path = os.path.join(cfg["out_dir"], "metrics.jsonl")
    with open(log_path, "a") as logf:
        logf.write(json.dumps({"_header": True, "git_sha": git_sha(repo_root),
                               "cfg": cfg, "resumed_at": start}) + "\n")
        logf.flush()

        def clip_and_norm(params):
            return float(torch.nn.utils.clip_grad_norm_(params, max_norm=cfg["grad_clip"]))

        norms = {}

        def after_g():
            freeze.assert_grad_mask(pipe.dit, require_any_nonzero=False)
            norms["G"] = clip_and_norm(student_trainable)

        def after_d():
            freeze.assert_grad_mask(fake)
            norms["D"] = clip_and_norm(fake_trainable)

        for step in range(start, cfg["max_steps"]):
            seed_step(cfg["base_seed"], step)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            item = sampler.sample(step)
            b = batch_utils.build_batch(item, cam_data, device=pipe.device, dtype=pipe.torch_dtype)
            out = steps.dmd_step_k(pipe, teacher, fake, b, pipe.scheduler, opt_G, opt_D,
                                   use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
                                   after_g=after_g, after_d=after_d,
                                   offload_teacher=cfg["offload_teacher"])
            ema.update(pipe.dit)

            peak = torch.cuda.max_memory_allocated() / 1e9
            if step % cfg["log_every"] == 0:
                logf.write(json.dumps({
                    "step": step, "loss_G": out["loss_G"], "loss_D": out["loss_D"],
                    "arrow_abs": out["arrow_abs"], "arrows": out["arrows"],
                    "grad_norm_G": norms.get("G"), "grad_norm_D": norms.get("D"),
                    "sigma_T": out["sigma_T"], "step_time_s": round(time.time() - t0, 3),
                    "peak_gb": round(peak, 2), "v0_id": item.v0_id,
                    "target_cam": item.target_cam_idx, "overlap": item.overlap_score,
                }) + "\n")
                logf.flush()
                print(f"step {step}: loss_G={out['loss_G']:+.3e} loss_D={out['loss_D']:.3e} "
                      f"arrow={out['arrow_abs']:.3e} peak={peak:.1f}GB {time.time()-t0:.1f}s")

            # §10 step-5 peak-memory auto-abort: fail fast, not at hour 20.
            if step == start + 5 and peak > cfg["mem_budget_gb"]:
                raise SystemExit(
                    f"ABORT: peak {peak:.1f} GB > budget {cfg['mem_budget_gb']} GB at step {step} "
                    f"(see §10 escalation ladder: offload_teacher / AC-offload / windowed attn / K=1)")

            if cfg["sample_every"] and step > start and step % cfg["sample_every"] == 0:
                maybe_sample(pipe, ema, pipe.scheduler, sampler, cam_data, cfg["out_dir"], step)

            if cfg["ckpt_every"] and step > start and step % cfg["ckpt_every"] == 0:
                save_ckpt(os.path.join(cfg["out_dir"], f"ckpt_{step}.pt"),
                          step, pipe.dit, fake, opt_G, opt_D, ema, cfg)
                prune_ckpts(cfg["out_dir"], cfg["keep_last"])

    print("training loop finished")


if __name__ == "__main__":
    main()
