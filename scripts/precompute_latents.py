"""Precompute + cache per-v0 VAE latents and text embeddings (PLAN §7 item 4 / B4).

Each training step re-encodes v0 and its middles through the VAE (steps.dmd_step_k calls
``encode_video_nograd`` every iter). v0's latent and its caption embedding are constant across
steps, so we cache them once to scratch and (later) let train_dmd.py read the cache instead of
re-encoding. This script only PRODUCES the cache; wiring it into the DMD step is a train_dmd.py
change (the step currently always encodes) — see the TODO there.

Cache layout (scratch, never POOL):  <cache_dir>/<v0_id>.pt  holding
    {"v0_id", "caption", "latent": (1,16,21,60,104) cpu bf16, "context": cpu}
Resumable: an existing cache file is skipped.

GPU-bound (VAE + text encoder). Mirrors build_middle_bank's inference pipe (enable_vram_
management ON), so the VAE / text encoder are CPU-offloaded until needed — hence the explicit
``load_models_to_device`` calls (the recurring gotcha). NOT login-node runnable.

  python scripts/precompute_latents.py \
      --split config/data/pilot_split.json --which both \
      --ckpt checkpoints/SpacetimePilot_1.3B_v1.ckpt \
      --cache_dir /orcd/scratch/orcd/014/akshatat/counterfactual_models/latent_cache
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_video_test as svt  # noqa: E402
from spacetimepilot import ModelManager  # noqa: E402
from spacetimepilot.utils.builder import build_pipeline  # noqa: E402
from spacetimepilot.training import data, latents as latent_utils, batch as batch_utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Cache per-v0 VAE latents + text embeddings")
    p.add_argument("--split", required=True, help="config/data/pilot_split.json")
    p.add_argument("--which", default="both", choices=["train", "heldout", "both"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--dit_path", default=svt.DEFAULT_DIT_PATH)
    p.add_argument("--text_encoder_path", default=svt.DEFAULT_TEXT_PATH)
    p.add_argument("--vae_path", default=svt.DEFAULT_VAE_PATH)
    return p.parse_args()


def build_inference_pipe(args):
    """Released inference pipe (mirrors build_middle_bank.build_inference_pipe)."""
    print("Loading Wan2.1 foundation models...")
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([args.dit_path, args.text_encoder_path, args.vae_path])
    pipe = build_pipeline({"type": svt.PIPELINE_VERSION}).from_model_manager(mm, device="cuda")
    print(f"Loading SPT checkpoint: {args.ckpt}")
    pipe.dit.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    pipe.to(dtype=torch.bfloat16)
    pipe.dit.to("cuda")
    pipe.device = torch.device("cuda")
    pipe.enable_vram_management()
    return pipe


def load_entries(args):
    sides = ["train", "heldout"] if args.which == "both" else [args.which]
    seen, out = set(), []
    for side in sides:
        for e in data.load_split(args.split, side):
            if e.v0_id not in seen:
                seen.add(e.v0_id)
                out.append(e)
    return out


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "no CUDA device — request a GPU node"
    os.makedirs(args.cache_dir, exist_ok=True)

    pipe = build_inference_pipe(args)
    entries = load_entries(args)
    print(f"Precomputing latents+text for {len(entries)} v0s -> {args.cache_dir}")

    done = 0
    for e in entries:
        out_path = os.path.join(args.cache_dir, f"{e.v0_id}.pt")
        if os.path.exists(out_path):
            print(f"  [skip] {e.v0_id}")
            continue
        video = batch_utils.load_pixel_video(e.video_path, device="cuda", dtype=torch.bfloat16)

        pipe.load_models_to_device(["vae"])            # gotcha: VAE offloaded until needed
        latent = latent_utils.encode_video_nograd(pipe, video)

        pipe.load_models_to_device(["text_encoder"])   # same for the text encoder
        with torch.no_grad():
            prompt_emb = pipe.encode_prompt(e.caption, positive=True)

        torch.save({
            "v0_id": e.v0_id,
            "caption": e.caption,
            "latent": latent.detach().to("cpu"),
            "context": prompt_emb["context"].detach().to("cpu"),
        }, out_path)
        done += 1
        print(f"  [ok ] {e.v0_id}  latent={tuple(latent.shape)}")

    # verify one reload
    sample = torch.load(os.path.join(args.cache_dir, f"{entries[0].v0_id}.pt"), map_location="cpu")
    print("\n================= PRECOMPUTE RESULT =================")
    print(f"  v0s cached now   : {done}")
    print(f"  v0s total        : {len(entries)}")
    print(f"  sample latent    : {tuple(sample['latent'].shape)}  (expect (1, 16, 21, 60, 104))")
    print(f"  sample context   : {tuple(sample['context'].shape)}")
    print(f"  cache dir        : {args.cache_dir}")
    print("====================================================")


if __name__ == "__main__":
    main()
