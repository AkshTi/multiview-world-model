# PLAN B (Track P) — Plain Finetune: our own mini-SPT from pretrained Wan
### Hyunwoo's "third option" (7/6 Slack): finetune pretrained Wan yourself at (T/2, H/2, W/2) —
### "~8× cheaper on GEMM and ~64× cheaper on attention… you can run experiments quickly.
### You can't run the full experiment every time."
### Companion: `PLAN_DMD_TRAINING.md` (Plan A — the research method). This track serves Plan A.

## 1. Purpose and non-purpose

**Purpose:** a small, fully-owned SPT-style model (camera+time-conditioned Wan) trained by
**plain supervised flow-matching MSE** — no DMD — used as:
1. **Fast-iteration proxy** for Plan A: replicate the DMD loop at mini scale where a full
   training run costs hours on an L40S instead of a day on a scarce 80 GB card. Tune the
   contested knobs there (RoPE OOD band / windowed attention, K, λ_stitch, cadence, teacher
   cfg), then confirm at full scale with short runs.
2. **Queue relief:** everything here targets **L40S-class GPUs (48 GB, plentiful)**, while
   Plan A waits on H100/H200 queues. This is a parallel lane, not a replacement.
3. **(Optional, high paper value)** a supervised **upper-bound baseline**: on synthetic data
   multi-view GT *does* exist, so a 2-source mini model trained WITH GT quantifies the gap
   DMD (which uses no multi-view GT) is trying to close.

**Non-purpose:** this does NOT replace released SPT in Plan A's full-scale runs, and it must
never block Plan A — separate budget, separate lane. If Track P slips, Plan A proceeds.

## 2. Model spec

- **Base (P0): Wan2.1-T2V-1.3B** — already on scratch, and the SPT codebase IS this
  architecture, so we reuse every conditioning module (cam_encoder, projector,
  frame_time_embedding, temporal_downsampler) **initialized from scratch with zero-init**,
  exactly SPT's own recipe. Trainable set = same as `freeze.py` (those modules + self_attn
  q/k/v/o); backbone frozen.
- **Recorded alternative (H): Wan2.2-5B** ("4× more params but 4× fewer tokens so it actually
  runs faster"). Switch criteria: P0 quality gate fails after the data-scale bump (§6 risk
  table), because porting SPT's conditioning modules to a new backbone is real integration
  work — only pay it if 1.3B is the proven bottleneck.
- **Resolution/length: 240×416, 41 frames.**
  - 41 = stride-2 subsample of 81 (frames 0,2,…,80 — exactly 41 ✓): keeps the FULL camera
    trajectory, keeps frame 0 (**attention-sink invariant, H**), satisfies Wan's 4k+1 rule
    → 11 latent frames.
  - Token math: latent 11×30×52 → patchify (1,2,2) → 11 × (15×26) = **4,290 tokens/video** vs
    32,760 at full res → ~7.6× fewer tokens, **~58× cheaper attention** (Hyunwoo's ~64×).
  - Conditioning resolutions scale consistently: camera enters pre-downsampled `[::4]` →
    41[::4] = 11 frames ✓; world-time enters at 41 and `temporal_downsampler` gives
    (41−1)/4+1 = 11 ✓.
- **Required code change (small, must be equivalence-gated):** the DiT hardcodes the full-res
  token grid — `.repeat(1,1,30,52,1)` sites and patchify assumptions in
  `spacetimepilot/model/spacetimepilot.py` (~:329, :340 and the `:314/:323`-era sites).
  Parameterize grid (f, h, w) from the latent shape. **Gate:** at full res the parameterized
  path must be BIT-IDENTICAL to released behavior (extend the existing
  `test_one_source_equivalence` pattern) — this protects Plan A's frozen teacher, which shares
  the file.

## 3. Data (supervised pairs — this is where plain finetuning is possible at all)

- **Source: MultiCamVideo** (`KlingTeam/MultiCamVideo-Dataset`; H-approved). 10 synchronized
  cameras per scene ⇒ real supervised pairs, ReCamMaster/SPT-style:
  **input = video from cam_i + target trajectory of cam_j → target = video from cam_j.**
  Ordered pairs i≠j ⇒ up to 90 pairs/scene.
- **Scale:** pilot **2,000 scenes** (~50 GB, scratch has 875 GB free; never POOL — it is 100%
  full). Bump to 5,000 if underfitting (§6).
- **Preprocess:** stride-2 → 41 frames; center-crop 1280² → 738×1280 → resize **240×416**;
  extrinsics subsampled with the same stride; VAE-latent + text-embedding cache to scratch
  (same `precompute_latents.py` job as Plan A, different config). **Convention check (shared
  with Plan A B2, 7/8):** verify MultiCamVideo's extrinsics layout/units match
  `process_camera_trajectory` before training, and normalize trajectories relative to the
  source cam's frame-0 pose (SPT convention) — do NOT assume the demo pool's absolute units.
- **Time axis honesty:** MultiCamVideo has NO frozen-time (bullet-time) renders — only
  synchronized real-time views. So mini-SPT learns camera control + simple retimings (extra
  stride/offset augmentations), not the full Cam×Time grid. Fine for purpose: Plan A's DMD
  pipeline is forward-time everywhere.
- **Captions:** none shipped → same VLM captioning job as Plan A B2 (1/scene, spot-check 20,
  template fallback).
- **Split:** 1,900 train / 100 held-out scenes (seed-0, committed json).

## 4. Training recipe (deliberately boring)

- **Objective:** standard flow-matching denoising MSE on the target latents — literally the
  Rung-2 step (`steps.py::one_source_smoke_step`) promoted to a real loop. Noise target only;
  source stays clean; slice to target frames before loss.
- **Numerics (H 7/6):** fp32 trainable params + bf16 autocast; backbone bf16 frozen; AC on;
  grad-clip 1.0; wd 0; EMA 0.999.
- **LRs:** 1e-4 for from-scratch conditioning modules, 2e-5 for self_attn q/k/v/o (pretrained);
  constant with 500-step warmup.
- **Batch:** measure at start; expect ≥4/GPU at 4.3k tokens (peak well under 48 GB with AC —
  step-5 auto-abort assert same as Plan A).
- **Steps:** 20k pilot (≈ 8–12 h on one L40S at batch 4, est — measure). Ckpt/resume/JSONL/
  fixed-tuple samples: identical harness to `train_dmd.py` (write the harness once, share it).
- **Sbatch:** L40S partition, 24 h, resumable; `expandable_segments`; `python -u`.

## 5. Eval (cheap, GT-backed — synthetic data pays off here)

- **Held-out 100 scenes.** Generate cam_j view from cam_i + GT trajectory; compare against the
  ACTUAL cam_j render (exists — synced cameras): PSNR / LPIPS, plus epipolar error from GT
  extrinsics+intrinsics (dataset ships focal lengths).
- **Baselines:** zero-shot Wan2.1 (no camera conditioning — sanity floor) and, for reference,
  released SPT at full res on the same scenes (not apples-to-apples, just orientation).
- **Success gate for "usable proxy":** clear camera obedience (epipolar error well below the
  no-conditioning baseline), no collapse/fog, LPIPS-to-GT in a sane band on ≥80% of held-out
  scenes. The proxy does NOT need to be beautiful — it needs to be *controllable and stable*.

## 6. The payoff: mini-DMD (once §5 gate passes)

Replicate Plan A at mini scale — mini teacher = this model (frozen), mini student = its
2-source extension (same edit already in the codebase), mini fake net = deepcopy:
- Whole DMD loop (Runs 1–3 analogues) at ~2–4 h/run on ONE L40S (3×1.3B at 4.3k tokens is
  small; measure peak, expect <20 GB).
- **What gets decided here cheaply:** RoPE OOD band behavior at 3 views (33 latent frames) +
  windowed-attention fallback; K ablation; λ_stitch + blur ablation; cadence; teacher-cfg knob.
- **Transfer protocol (honest):** conclusions transfer as *priors, not proofs* — each adopted
  setting gets one short confirmation run at full scale in Plan A. Scale-dependent effects are
  the known threat; never skip the confirmation.
- **(Optional) supervised upper bound:** train the 2-source mini student directly on GT
  triplets (v0=cam_i, v1=cam_j, target=cam_k — all exist!) and compare to mini-DMD. This
  quantifies "how much does lacking multi-view GT cost" — a figure the paper wants anyway.

## 7. Schedule & budget (parallel lane, L40S only)

| Step | What | Est |
|---|---|---|
| P1 | Download 2k scenes + preprocess + caption + cache | ~1 day wall, ~6 GPU-h |
| P2 | Grid-parameterization edit + bit-identical full-res equivalence test | 0.5–1 day |
| P3 | Train mini-SPT 20k steps + eval §5 | ~12 GPU-h |
| P4 | Mini-DMD replicas of Runs 1–3 | ~2–4 h each, as needed |
| P5 (opt) | Supervised 2-source upper bound | ~12 GPU-h |

Total ≈ 30–60 L40S-hours. Decision rule: if the H100/H200 queue keeps Plan A Run 1 waiting
> ~3 days, Track P becomes the *primary* iteration vehicle in the meantime.

## 8. Risks

| Risk | Signal | Response |
|---|---|---|
| Mini quality too low to be a meaningful proxy | §5 gate fails | scenes 2k→5k; longer training; only then consider Wan2.2-5B port |
| Grid-parameterization breaks full-res path | equivalence test | hard gate — bit-identical or don't merge (protects Plan A's teacher) |
| Transfer failure (mini conclusions wrong at scale) | full-scale confirmation runs disagree | treat mini results as priors only; full-scale gates in Plan A unchanged |
| Scope creep — Track P delays Plan A | calendar | separate lane/budget; Plan A never waits on P |
| Stride-2 retiming confuses world-time semantics | eval videos play at wrong speed | world-time embedding fed in subsampled frame units, consistently at train+eval |
| No frozen-time data | — | accepted limitation (§3); forward-time only, matches Plan A usage |

## 9. Verification
- P2 equivalence test bit-identical at full res (extends `test_one_source_equivalence`).
- Existing CPU suite stays green (shared files!).
- P3: loss curve + fixed-tuple samples every 250 steps; §5 gate numbers vs both baselines.
- Every run logs config + git SHA; same JSONL harness as Plan A.
