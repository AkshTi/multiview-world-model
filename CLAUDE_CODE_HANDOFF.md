# Project Handoff: Multi-View-Consistent World Model via DMD Distillation

> **Read this whole document before writing code.** It contains the research goal, the
> mathematical derivations the implementation must realize, a detailed map of the existing
> SpaceTimePilot (SPT) codebase, the concrete implementation plan, and the working
> constraints. The project is a research implementation, not a product. Correctness of the
> *training objective* matters more than speed.

---

## 0. Who I am and how to work with me

- MIT grad student. Advisor/collaborator on this project: **Hyunwoo Ryu**. Weekly meeting Thursdays 11am ET.
- I learn by being able to **teach the derivation/implementation back**, not just run it. When something is subtle, explain the *why* underneath, build intuition before algebra/code, go one idea at a time. If I say I'm confused, stop and rebuild from a lower level rather than pushing forward.
- I prefer honest "this part is open / approximate / unverified" over confident overstatement.
- For implementation: **build incrementally, verify each rung before adding the next.** Do not write the entire DMD loop in one shot. Get a forward pass working, then noising, then one score, then the next, checking shapes and sanity at each step.

---

## 1. The research goal in one paragraph

Existing camera-conditioned video models (SPT / SpaceTimePilot, BulletTime, ReCamMaster)
can re-shoot a factual video `v0` from a new camera path + world-time to invent a
counterfactual view. **Problem:** each invented view silently commits to a *different*
underlying 3D world (an underdetermined posterior over worlds), so stitching views at the
same world-time gives inconsistent geometry / flicker. **Goal:** force every invented
video to be a render of *one shared world*. **Method:** fine-tune from an existing
2-video model and **distill** (no multi-view ground truth exists) toward a *consistent*
teacher, using **DMD** over a *marginalized* distribution, plus a **stitching** term to
prevent blurry collapse, plus **crossing constraints** for geometric truth, scaled past
two views by **pairwise-window** distillation.

**Critical reframe that drives the whole implementation:** because no multi-view ground
truth exists, the training loop is **NOT a plain denoising-MSE fine-tune**. It is a **DMD
distillation loop**. Plain denoising-MSE only appears *inside* one auxiliary network (the
fake-score network). Do not build a vanilla "corrupt target, MSE against truth" trainer as
the main objective — that requires ground-truth targets we do not have.

---

## 2. Notation

- Videos `v0` (factual), `v1, v2, ...` (counterfactual). Action `a_n = (g_n, t_n)` = camera
  pose + world-time. (BulletTime/SPT disentangle pose from time.)
- Want the joint `p(v0,...,vN | a0,...,aN)` so we can *demand* a shared world. We *have*:
  `p(v0)`, and `p(v1|v0)` (SPT/ReCamMaster). We *lack*: `p(vn | v0,...,v_{n-1})`
  (many-video conditioning) → must **distill**, not train.
- **Score / "arrow"** `s(v) = ∇_v log p(v)` = "direction to nudge a video to make it more
  realistic." Central object because the density needs the intractable partition function
  `Z`, but the arrow doesn't (`∇_v log Z = 0`). **A trained diffusion model IS its arrow
  field** — its noise/velocity prediction is the score (up to a known scaling).

---

## 3. The derivations the code must realize

### D1 — DMD gradient (DONE; off-the-shelf, not the contribution)
Minimizing `KL(q_θ || p)` is intractable (needs `Z`), but its *gradient* collapses to a
difference of arrows. Result:

```
∇_θ L = E_z[ (s_fake − s_real) · ∂_θ G_θ(z) ]
```

Descent nudges each generated video along `A = s_real − s_fake` (uphill on the teacher,
away from the student's own pile-up), pushed through the generator. Implement as a
**detached surrogate**: `E_z[ stop-grad(s_fake − s_real)ᵀ · G_θ(z) ]`.

- `s_real`: frozen teacher score — free (a trained diffusion model is its arrow field).
- `s_fake = ∇_v log q_θ`: **NOT free** — the generator only *samples*, never reports its
  own density's score. So train a **second small score network online** (denoising-score-
  matching on the student's own outputs) that chases the shifting student distribution.
  This is the VSD/DMD two-network structure; the `−s_fake` term is the anti-blur force.

### D2 — Monte-Carlo–marginalized DMD (THIS is the contribution)

> **CORRECTED to Hyunwoo's written branch.** See `HYUNWOO_RECONCILIATION.md`. An earlier
> draft put the marginal on the *teacher* (average two-video arrows over middles). Hyunwoo's
> `counterfactual.txt` (lines 40–43) puts the marginal on the **student**, with a **direct**
> teacher. Follow that.

The marginalized object is the **student** joint, marginalized over the middle `v1`:

```
p'(v2|v0) = ∫ q_θ(v2|v0,v1) p(v1|v0) dv1
```

DMD matches this to the **direct** released teacher `p(v2|v0)`:
`min_θ D_KL( p'(v2|v0) || p(v2|v0) )`. So:

- `s_real = ∇_{v2} log p(v2|v0)` — a **single** frozen-SPT call, one source = the real `v0`.
  No averaging over middles. In-distribution (v0 is real footage).
- `s_fake = ∇_{v2} log p'(v2|v0)` — the **marginal student score**, obtained by the MSE
  trick: train a **one-source** fake-score net conditioned on `v0` ONLY (hide `v1`) via
  denoising-MSE on the student's `v2` samples. Least-squares with `v1` hidden returns the
  average over `v1` = the marginal score. Conditioning the net on `(v0,v1)` would give a
  *conditional* score — the wrong object.

**Novelty:** the Monte-Carlo marginalization on the student side (`counterfactual.txt` L43,
"nobody has tried monte-carlo marginalized DMD").

**Bias note:** the prior-vs-posterior worry does NOT arise here — the student marginal is
*defined* by `p(v1|v0)`, so sampling `v1 ~ p(v1|v0)` and hiding it is the exact generative
process; the MSE optimum is the true marginal score, no posterior reweighting.

**Only the STUDENT is N-source.** Teacher and fake-score net both stay one-source.

### D3 — stitching score transport (DONE)
Consistency alone has a cheat: all videos collapse to the same gray blur (two blurs
"agree"). Stitching forbids it. Take **one frame from each of N jointly-generated videos**
at the shared world-time, assemble into one strip `x = Wv` (`W` = linear selection/masking
matrix, mostly zeros with a single 1 per kept pixel), and demand `x` "looks like a real
single video" (sharp + temporally smooth). Blur fails that test; sharp-consistent passes.

**Transport result:** because `W` is a clean (disjoint, un-blended) selection,
`WWᵀ = I`, so noising commutes with slicing and the strip's score is the joint score
sliced the same way:

```
s_x(x_t) = W · s_v(v_t)
```

So you **reuse the joint score, no new network** for the stitching term. Same DPS /
linear-inverse-problem machinery. **OPEN:** exact conditions on `W` (slice-then-average vs.
average-then-slice equality holds at high noise / genuinely disjoint frames) — same shape
of gap as D2's posterior-vs-prior.

### D4 — scaling past two views (stated as an honest bet, NOT a theorem)
Teacher conditions on ≤2 videos; student should generate 3+. Plan: generate N videos, and
enforce the two-video teacher (D1/D2/D3 machinery) on **every overlapping window** (sliding
123, 234, 345…). **Claim:** overlapping-pairwise consistency ⇒ global N-view consistency,
**by transitivity**. **Status: assumption, not theorem.** Justified by (a) window *overlap*
pinning shared videos across adjacent windows, and (b) crossing constraints giving
chain-distance-independent anchors. **Failure mode to watch: drift** — tolerated slack
accumulating coherently across the chain so distant videos disagree while all neighbors
pass. Self-Forcing reports the analogous length-mismatch scheme working; that's the
precedent. There is **no boxed equation** for D4 — the honest output is the precisely
stated bet with its failure mode named.

---

## 4. The architecture: Wan 2.1 + SPT (what the code IS)

### Stack (three historical layers, all in the SPT files)
- **Stock Wan 2.1** (1.3B): a latent video diffusion model. Frozen **3D-VAE** compresses
  video → latent; **DiT** (transformer) denoises in latent space. This is the borrowed
  "what video looks like" knowledge. **Never modified.**
- **ReCamMaster** layer: added video-conditioning (concat a source latent onto the target)
  + camera injection inside each DiT block. Inherited.
- **SPT** layer: added explicit **world-time** control + **source-aware** cameras
  (separate src/tgt cameras instead of one duplicated). This is the model we fine-tune from.

### Shapes that govern everything (from the VAE, memorize these)
- VAE: **81 RGB frames → 21 latent frames**, 8× spatial (480×832 → 60×104), **16 latent
  channels**. Formula in code: `(num_frames - 1)//4 + 1 = (81-1)//4+1 = 21` (first frame
  kept alone, then every 4 frames → 1).
- Patchify uses patch size `(1,2,2)` → spatial grid becomes **30×52** (these appear
  hardcoded in the DiTBlock; they assume 480×832 resolution).
- One video latent: `(B, 16, 21, 60, 104)`. Two concatenated: `(B, 16, 42, 60, 104)`.
  N videos: `(B, 16, 21·N, 60, 104)`.
- Token sequence length after patchify = `f·h·w` where `f` is the **concatenated** frame
  count. Attention is **quadratic** in sequence length → doubling videos ~quadruples
  attention cost. **This is why the teacher caps at 2 videos and D4 scaling is pairwise.**

### Key architectural facts (verified by reading the code)
- **Fusion is latent-channel concat along the frame axis, done ONCE outside the DiT**, not
  token concat and not a special module. The line is
  `latents_input = torch.cat([latents, source_latents], dim=2)` (dim 2 = frame axis).
  Self-attention then mixes the two videos because they're in one sequence. **Fusion =
  stock self-attention on a concatenated sequence.**
- **Conditioning (camera, world-time) is injected ADDITIVELY inside every DiTBlock**, via
  a 5-step recipe: embed → (compress to 21 frames) → concat `[tgt, src]` → broadcast across
  the 30×52 grid → add onto tokens. The camera path and time path are structurally
  identical.
- **Order invariant:** latents, camera, and time are all concatenated in the SAME `[tgt,
  src]` order. Any N-video extension MUST use one canonical ordering identically in all
  concat sites, or video i's conditioning lands on video j's tokens (silent bug, no error).
- **RoPE is index-based, NOT world-coordinate-based.** `precompute_freqs_cis` uses
  `torch.arange(end)` — sequence indices only. So two patches at the same world point but
  different sequence positions get *different* RoPE. **The crossing constraint cannot ride
  on RoPE**; world info enters only through the additive camera embedding (or must be
  enforced via the loss). (This corrected an earlier wrong assumption that RoPE was
  world-coordinate-based.)
- **Zero-init trick:** new conditioning modules start as no-ops so fine-tuning begins at
  pretrained behavior and "turns on" gradually. In `DiTBlock.__init__`:
  `cam_encoder` weights/bias zeroed, `projector` = identity (`torch.eye(dim)`), zero bias.
  **This is the template for adding ANY new module (N-th video conditioning, crossing
  signal): zero-init it.**
- **The released `__call__` is `@torch.no_grad()` INFERENCE** — it starts from noise and
  runs ~50 scheduler steps to generate. The training loop must be written separately,
  WITH gradients, single forward call per step.

### File map (in `spacetimepilot/pipelines/` and `spacetimepilot/wan/`)
- `base.py` — shared building blocks: `RMSNorm`, `rope_apply`, `modulate`,
  `CrossAttention`, `MLP`, `Head`, `BasePipeline`, `TeaCache`, plus a *baseline* DiTBlock.
  Mostly imported by the other files. The attention-backend dispatcher
  (`flash_attention`, FlashAttn-3/2/Sage/SDPA fallbacks) lives here too.
- `recammaster.py` — registered `'baseline'`: camera-only conditioning, single `cam_emb`,
  no time path. This is the "before SPT" reference. `__call__` takes only `target_camera`.
- `spacetimepilot.py` — registered `'spacetimepilot_1dconv'`: **the model we use.** Adds
  `frame_time_embedding` MLP + `TemporalDownsampler` (81→21, VAE-matched), splits camera
  into tgt/src. `__call__` takes both `target_camera` and `source_camera` plus
  `src_time_embedding`/`tgt_time_embedding`.
- `spacetimepilot/wan/` subpackage — the stock Wan internals (VAE
  `wan/models/wan_video_vae.py`, DiT, text encoder, `FlowMatchScheduler`,
  `sinusoidal_embedding_1d`, etc.). Usually accessed only via call sites; dive in only if a
  question forces it.

### The three N-video edit sites (for later; NOT the first task)
1. **Fusion line** in pipeline `__call__`: `torch.cat([latents, source_latents], dim=2)` →
   `torch.cat([latents, src1, src2, ...], dim=2)` in canonical order; encode N sources.
2. **The two concats in `DiTBlock.forward`**: `torch.cat([tgt_*, src_*], dim=1)` for both
   camera and time → concat N entries in the same canonical order. Broadcast/flatten stay
   the same; only the frame count grows.
3. **The two dicts in `model_fn_wan_video`**: `{"tgt":..., "src":...}` → N entries
   (dict or list). Pure glue; must stay consistent with sites 1 and 2.
   Everything else (attention, RoPE, patchify, FFN, VAE, downsampler) adapts automatically
   because it's written against an arbitrary sequence length.

---

## 5. Repo / environment facts

- Repo: `https://github.com/ZheningHuang/SpaceTimePilot` (CVPR 2026, Apache-2.0).
- **Inference code + checkpoint are released; TRAINING CODE AND DATASETS ARE NOT.** So the
  training loop is built by us on top of the released model definition. CamXTime *eval*
  data is on HF; full dataset is request-access.
- Requirements: **Linux + an 80GB GPU** for inference. Training with gradients needs more
  headroom → **gradient checkpointing is mandatory** (already supported via
  `use_gradient_checkpointing` in `WanModel.forward`).
- Setup uses `uv`; checkpoint via `hf download zhening/SpaceTimePilot
  SpacetimePilot_1.3B_v1.ckpt`; Wan2.1 base via `spacetimepilot/wan/download_wan2.1.py`.
- Entry points: `single_video_test.py`, `inference_batch.py`. Model config loaded is the
  **1.3B** (`dim = 1536*2`, `num_layers = 30`).

---

## 6. The implementation ladder (DO IN THIS ORDER)

> Each rung must work and be sanity-checked before the next. Do not skip ahead.

**Rung 0 — Make SPT run at all.**
Get the released inference example executing. Put prints/breakpoints at the fusion line and
inside one DiTBlock; confirm real tensor shapes match Section 4 (latent `(B,16,21,60,104)`,
concatenated `(B,16,42,...)`, token sequence `f·h·w`). This converts static code-reading
into living understanding and surfaces all environment problems early.

**Rung 1 — A training harness around the frozen model (no novel loss yet).**
Build the scaffolding the release omits, *without* committing to the wrong objective:
- A `training_step` method (NOT `@torch.no_grad()`, gradients live) separate from `__call__`.
- A dataloader yielding `(source_video, target_video, source_camera, target_camera,
  src_time, tgt_time, caption)`. Cameras as flattened 3×4 (12-number) extrinsics per frame;
  time signals at **81-frame** granularity (the downsampler compresses to 21 itself);
  videos preprocessed to 81 frames @ 480×832, normalized to [-1,1].
- Freeze/unfreeze: set `requires_grad=False` on the whole DiT, then re-enable only the
  conditioning-relevant modules — `cam_encoder`, `projector`, `frame_time_embedding`,
  `temporal_downsampler`, and the self-attention `q/k/v/o`. (Mirrors what the SPT paper
  trains; also what makes it fit in memory.)
- Turn ON `use_gradient_checkpointing`. AdamW over only the unfrozen params.
- **Verify a single training step runs, loss is finite, gradients are non-zero on the
  unfrozen modules and zero/None on frozen ones.** Use a tiny toy batch.

**Rung 2 — The DMD distillation loop (the actual method). THREE SPT instances.**
This realizes D1 + D2 (Hyunwoo's direct-teacher / student-marginal branch). Per step:
- **Student `G_θ`** (trainable, **N-source** SPT conditioning on `(v0,v1)`): generates `v2`
  (few-step). Sampling `v1 ~ p(v1|v0)` from the middle bank realizes the marginal
  `p'(v2|v0)`.
- **Frozen teacher** (for `s_real`): the released SPT called **once**, one source = the real
  `v0`, target = `v2`. `s_real = ∇ log p(v2|v0)`. No averaging over middles.
- **Online fake-score net** (for `s_fake`): a **one-source** SPT (conditioned on `v0`,
  `v1` hidden) trained online with **denoising-MSE on the student's current `v2` outputs**
  (THIS is the only place plain-MSE lives). By the MSE trick its optimum is the marginal
  student score `∇ log p'(v2|v0)`. It must keep chasing the moving student distribution.
- **Student update:** detached surrogate `stop-grad(s_fake − s_real)ᵀ · G_θ(z)` →
  `loss.backward()`. (D1.)
- **Score-net update:** separate optimizer step on the fake-score net via its denoising loss.
- Get the **noise-level** versions right (scores at sampled timestep `t`, per the
  noise-level D2). Pin the FlowMatchScheduler velocity/target convention exactly — a wrong
  sign/scale here silently trains toward garbage.

**Rung 3 — Stitching term (D3).**
Add the strip probe: select one frame per generated video at the shared world-time → `x =
Wv`. Use the transport `s_x = W · s_v` to reuse the joint score (no new net). Add this score
term into the student update. Watch for the open `W` condition (high-noise / disjoint-frame
regime). This is the anti-blur force that kills the degenerate gray-fog solution.

**Rung 4 — N-video extension (the three edit sites in Section 4).**
Only after 2-video DMD+stitching works. Extend the fusion concat, the DiTBlock concats, and
the model_fn dicts to N videos under one canonical ordering. Zero-init any new modules.

**Rung 5 — Scaling via pairwise windows (D4).**
Generate N, enforce the (D1/D2/D3) machinery on every overlapping window (123, 234, …), sum
per-window losses. Watch for **drift** (the named failure mode). Reference Self-Forcing for
the length-mismatch mechanism.

---

## 7. Things that are easy to get wrong (guardrails)

- **Two unrelated "time" concepts.** The DiT's `time_embedding`/`time_projection` /
  `timestep` is the **diffusion noise-level clock** (internal). The
  `frame_time_embedding` / `src_time_embedding` / `tgt_time_embedding` is **world/animation
  time** (SPT's conditioning). They are different variables doing different jobs — never
  conflate them.
- **Don't build a vanilla MSE fine-tune as the main objective.** No multi-view ground truth
  exists; the main objective is DMD. MSE only lives inside the fake-score net.
- **The `[tgt, src]` ordering invariant** across all concat sites. For N videos, write the
  canonical order down once and obey it in the fusion line, both DiTBlock concats, and the
  model_fn dicts.
- **Slice to the target frames** (`pred[:, :, :tgt_latent_length, ...]`) — the source half
  is context with no learning target.
- **Source latents are clean (not noised)** in the conditioning path; only the target gets
  corrupted.
- **Zero-init every new conditioning module** so you start from pretrained behavior.
- **FlowMatchScheduler convention** (velocity vs noise vs x0 target, and its scaling) must
  be read from the scheduler code, not assumed.
- **Gradient checkpointing ON** for any training; otherwise OOM at these sequence lengths.
- **`enable_vram_management`** was built for inference offload — verify it cooperates with
  backprop before relying on it during training.

---

## 8. Open questions to track (and raise with Hyunwoo)

1. **D2 posterior vs. prior** sampling of middles `v1` — the cheap recipe is biased; the
   crossing constraint is meant to make it safe. Which estimator do we commit to?
2. **D3 exact conditions on `W`** — when does slice-then-average equal average-then-slice?
3. **D4 transitivity / drift** — does pairwise-window consistency actually buy global
   consistency, and how do we measure drift?
4. **Detach vs. backprop through the conditioning video `v1`** in the teacher path.
5. **Few-step student availability** for the backbone (or do we distill one ourselves).
6. **Crossing constraint enforcement** given RoPE is index-based — via camera embedding or
   via loss?

---

## 9. Immediate next action for this Claude Code session

Start at **Rung 0 / Rung 1**. Concretely:
- Confirm the repo is cloned and the environment/compute is available (flag immediately if
  the 80GB GPU or checkpoint is missing — nothing downstream moves without it).
- Get one inference forward pass running; print shapes at the fusion line and in one
  DiTBlock to confirm Section 4's shapes.
- Then scaffold the `training_step` + dataloader + freeze/unfreeze + gradient-checkpointing,
  and verify a single training step produces a finite loss with gradients only on the
  unfrozen modules. Use a toy batch.
- **Do not** implement the DMD loop (Rung 2) until Rung 1 is verified. **Do not** implement
  a vanilla-MSE main objective at all.

Build incrementally, check shapes and finiteness at every step, and explain the *why* as you
go (I want to be able to teach it back).
