# Plan: Multi-View-Consistent World Model via DMD Distillation on SPT

## Context - what we are building

SpaceTimePilot (SPT) can re-shoot a real video `v0` from a new camera path and world-time, but
independent generations `v1...vN` can commit to different hidden 3D worlds. The goal is to
fine-tune SPT so counterfactual videos behave like renders of one shared world, without
multi-view ground truth. The method is DMD/VSD-style distillation: a trainable student generator,
a frozen teacher score, and an online fake-score network. Plain denoising MSE is never the main
objective; it only trains the fake-score net or exercises the graph in early smoke tests.

The released repo is inference-first. The core code sites are:

- `RESEARCH_CONTEXT.md`: source-of-truth research context integrated from handoff, meeting,
  derivations, Slack, and code. Read it before changing the objective.
- `spacetimepilot/model/spacetimepilot.py`: SPT pipeline, DiT block, `model_fn_wan_video`.
- `spacetimepilot/wan/schedulers/flow_match.py`: flow-matching scheduler, noising, training target.
- `single_video_test.py`: camera/time preprocessing used by released inference.

## Critical Corrections To The Earlier Plan

1. **The DMD student-loss sign was wrong.** The plan's arrow formula was right, but the surrogate
   loss line used the arrow instead of its negative. We must implement
   `loss_student = stop_grad((1 - sigma) / sigma * (v_real - v_fake)) dot x0_hat`, because this is
   `(s_fake - s_real) dot x0_hat`; gradient descent then moves `x0_hat` along
   `s_real - s_fake`. Add a sign unit test before training.
2. **`model_fn_wan_video` does not use gradient checkpointing.** `WanModel.forward` supports
   `use_gradient_checkpointing=True`, but `model_fn_wan_video` manually loops over blocks and
   bypasses that path. Training must use either `dit.forward(...)` or a new
   `training_model_fn_wan_video(...)` that preserves CFG/TeaCache-free glue while checkpointing
   blocks. Otherwise Rung 1 is likely to OOM.
3. **`enable_vram_management()` is unsafe to assume for training.** The wrapper can copy modules
   during `forward` and was designed for inference offload. Disable it for training until a small
   backprop test proves gradients still hit the original parameters.
4. **Do not assume a 4-step student works from the released checkpoint.** Flow matching helps, but
   the released pipeline defaults to 20-50 inference steps. Start with a no-grad prefix plus a
   gradient-tracked final step, then sweep fewer steps only after generated `x0_hat` is not garbage.
5. **The D2 teacher remains approximate.** The exact marginal score needs posterior-weighted
   middles `p(v1 | x_t, v0)`. Prior-sampled cached middles are biased. This is acceptable only as a
   named approximation with diagnostics; it is not "solved" by code inspection.
6. **The marginalization side is still an open research fork.** `RESEARCH_CONTEXT.md` §4.5 says
   Hyunwoo's Slack framing may put the integral on the **student/fake-score** side, with direct
   teacher `p(v2 | v0)`, while the handoff/derivation plan puts the integral on the **teacher**
   side, averaging over middles. These are not equivalent unless SPT is already self-consistent,
   which is exactly what we do not believe. Implementation must keep this swappable until Hyunwoo
   confirms Q1.
7. **The crossing constraint is not automatic.** RoPE is sequence-index-based. World geometry enters
   through camera/time embeddings, and camera preprocessing is relative to a first frame. Crossing
   must be enforced after the same normalization convention used by SPT, not merely by matching raw
   extrinsics in a JSON file.

## Feasibility Check - Could This Implementation Work?

Yes, but only if the first implementation is narrower than the full research story. The parts that
look implementable from the released code are:

- A one-source training harness: `WanModel.forward` already supports gradient checkpointing, and
  the scheduler gives the exact noising/velocity target.
- A two-source student/fake extension: the model is mostly sequence-length agnostic, and the only
  required N-source edits are the latent, camera, and time concats.
- A DMD-style loss: velocity-to-score conversion is algebraically pinned by `flow_match.py`.

The parts that can break the project if assumed too early are:

- **Q1 marginalization side:** teacher-averaged middles vs direct teacher with marginalized
  student score. This changes which SPT calls are made in Rung 2.
- **Memory:** three 1.3B SPTs plus 3-video attention can exceed 80GB if the training path bypasses
  checkpointing or if VAE/text encoder are left in the graph.
- **Few-step generation:** a 4-step student is not guaranteed from the released checkpoint; use a
  known-working sampler length with last-step gradients first.
- **Crossing:** camera equality must be checked after SPT's relative pose preprocessing, otherwise
  the "same world point" may not be what the model receives.
- **Fake-score net size:** the "small score net" is not small if mirrored with full SPT. Treat it
  as a full/frozen-mostly SPT first, or explicitly introduce LoRA/adapters later as a separate
  engineering decision.

## Decision 0 - Who Conditions On What

The released SPT conditions a target on exactly one source video:

```python
source_latents = self.encode_video(source_video, **tiler_kwargs)
latents_input = torch.cat([latents, source_latents], dim=2)
```

**RESOLVED to Hyunwoo's written branch (`counterfactual.txt` L40–43; see
`HYUNWOO_RECONCILIATION.md`). Direct teacher, student-side Monte-Carlo marginalization:**

- **Frozen teacher (`s_real`):** released SPT called **once**, one source = the real `v0`,
  target = `v2`. `s_real = grad_{v2} log p(v2 | v0)`. No averaging over middles. This is the
  "clean target"; `v0` is real footage, so it is in-distribution.
- **Student:** extended SPT conditioning target `v2` on two sources `(v0, v1)`. OOD relative
  to the checkpoint, so it must pass a 2-source forward/shape/quality gate before DMD. The
  student marginal `p'(v2|v0) = ∫ q_theta(v2|v0,v1) p(v1|v0) dv1` is realized by sampling
  middles `v1 ~ p(v1|v0)` from the bank.
- **Fake-score net (`s_fake`):** a **one-source** net conditioned on `v0` ONLY (`v1` hidden),
  trained online by denoising-MSE on the student's `v2` samples. By the MSE trick its optimum
  is the marginal student score `grad log p'(v2|v0)`. Do NOT condition it on `(v0,v1)` — that
  gives a conditional score, the wrong object.

So **only the student is N-source**; teacher and fake-score net stay one-source. The
prior-vs-posterior bias does not arise (the student marginal is defined by `p(v1|v0)`).

A `score_provider` interface may still wrap `s_real`/`s_fake` for testability, but the mode is
fixed to Hyunwoo's `direct_teacher` branch; the marginal-teacher branch is retired (kept only
as a historical note in `RESEARCH_CONTEXT.md` §4.5).

## Flow Matching: Velocity To Score

`FlowMatchScheduler.add_noise` implements
`x_t = (1 - sigma) * x0 + sigma * eps`, and `training_target` is `eps - x0`. Therefore SPT predicts
velocity `v = eps - x0`.

```text
x0 = x_t - sigma * v_pred
eps = x_t + (1 - sigma) * v_pred
s(x_t) = -eps / sigma = -(x_t + (1 - sigma) * v_pred) / sigma
s_real - s_fake = (1 - sigma) / sigma * (v_fake - v_real)
s_fake - s_real = (1 - sigma) / sigma * (v_real - v_fake)
```

Implementation requirements:

- Always recover `sigma` from `scheduler.sigmas[timestep_id]`; do not use `timestep / 1000`.
- Slice predictions to target frames before score conversion:
  `pred[:, :, :tgt_latent_length, ...]`.
- Avoid `sigma = 0` for DMD score levels.
- Unit-test score conversion and DMD sign on known `(x0, eps, sigma)` before any training run.

## Architecture Rung A - Minimal N-Source Extension

Extend the model under one canonical order:

```text
[target, source0=v0, source1=v1, ...]
```

Required edit sites in `spacetimepilot/model/spacetimepilot.py`:

1. Fusion: `torch.cat([latents, source_latents], dim=2)` becomes concat of all source latents.
2. `DiTBlock.forward`: time concat currently `torch.cat([tgt_time_embedding, src_time_embedding], dim=1)`.
3. `DiTBlock.forward`: camera concat currently `torch.cat([cam_emb_tgt, cam_emb_src], dim=1)`.
4. `model_fn_wan_video`: replace fixed `{"tgt": ..., "src": ...}` glue with ordered source lists
   or explicit keys that cannot reorder silently.
5. **RoPE positions (per Hyunwoo's Slack):** the freqs are gathered by array index
   (`self.freqs[0][:f]`). To let the crossing constraint ride on RoPE, index `self.freqs` with an
   **explicit per-token 3D/temporal position** so tokens at the same world point across videos get
   the same positional encoding. Keep the camera embedding too. (Exact position convention =
   confirm Thursday; wire the plumbing to accept explicit positions now.)

No new *learned* module should be added first. If a new view-ID/crossing module becomes necessary
later, it must be zero-initialized and gated behind an ablation.

## Rung 0 - Repo, Checkpoints, And Inference Shape Probe

Before training in the active worktree/cluster environment:

- Confirm Wan2.1 base weights, SPT checkpoint, demo/eval video, and camera JSON exist on the GPU
  node.
- Run released `single_video_test.py` with `num_inference_steps=20` or the known-working setting.
- Add temporary shape prints/hooks at VAE encode, fusion concat, and one `DiTBlock.forward`.
- Verify expected shapes: video latent `(B,16,21,60,104)`, 2-video latent
  `(B,16,42,60,104)`, tokens `42 * 30 * 52`.

Gate: do not build training until released inference works in the same environment.

## Rung 1a - One-Source Training Harness Smoke Test

Build `spacetimepilot/training/` without changing inference `__call__`.

The first harness should use the released one-source shape, not the 2-source extension, so failures
are not confused with N-source OOD behavior:

- Load one `(source_video, target_video, source_camera, target_camera, src_time, tgt_time, caption)`.
- Encode source and target with VAE under no grad.
- Sample timestep index, noise only the target latent with `add_noise`; keep source latent clean.
- Run a checkpointed DiT forward and slice to target frames.
- Use throwaway flow-matching MSE against `training_target(target_latent, noise, timestep)`.

This MSE is only a graph test. It must be commented as "not the research objective."

Training forward requirement:

- Prefer calling `dit.forward(..., use_gradient_checkpointing=True)` for the smoke test.
- If CFG or custom score glue needs `model_fn_wan_video`, create a training-specific equivalent
  that calls checkpointed blocks. Do not use the inference `model_fn_wan_video` loop for grad runs.

Freeze/unfreeze:

- Freeze all DiT parameters.
- Unfreeze only `cam_encoder`, `projector`, `frame_time_embedding`, `temporal_downsampler`, and
  self-attention `q/k/v/o`.
- Do not train VAE or text encoder.

Gate:

- Finite loss.
- Gradients nonzero only on the intended trainable modules.
- Checkpointed path actually used.
- `enable_vram_management` disabled or explicitly proven compatible with backprop.

## Rung 1b - Two-Source Student/Fake Harness

After Rung 1a passes, implement Rung A and rerun the same harness with two clean sources
`(v0, v1)` and target `v2`. For this smoke test, `v1` can be a duplicate or cached teacher output;
the point is shape/gradient correctness, not objective quality.

Gate:

- 3-video latent input has frame length `63`.
- Time/camera embeddings also have frame length `63`.
- Target-frame slice remains exactly first `21` latent frames.
- Outputs match the one-source harness when source1 is disabled/omitted, if a compatibility path is
  kept.

## Data And Middle Bank

A DMD training item is not a multi-view ground-truth tuple. It is:

- real source video `v0`;
- source action `a0`;
- sampled middle actions `a1^(1...K)`;
- target action `a2`;
- cached middles `v1^(1...K)` generated offline by frozen one-source SPT from `v0`.

Crossing constraint:

- Choose a shared world-time frame and camera pose where `a1` and `a2` should intersect.
- Apply the same camera preprocessing convention used by SPT before checking equality:
  source cameras are made relative to the first camera; target camera embeddings use
  `compute_pose_embedding` relative to the trajectory's first camera.
- Store normalized camera/time embeddings with the cached middle, not only raw camera IDs.
- Add a cache validation script that reports the max embedding mismatch at the crossing frame.

Middle bank:

- Generate K middles per `v0` offline with frozen released SPT.
- Store source latents when possible, not only decoded videos, so teacher/student score calls can
  avoid repeated VAE encode. Also store prompt/camera/time metadata and the initial noise/seed used
  for deterministic ODE/sampler replay when needed.
- Detach cached `v1` always. It is data, not part of the student graph.

## Rung 2 - DMD Loop (D1 + Q1-Dependent D2)

Use three SPT instances: student, frozen teacher, fake-score net. Start with `K=1`, no stitching,
`cfg_scale=1`, and a configurable score-provider mode.

Per step:

1. Load `v0`, one cached `v1`, and target action `a2`.
2. Generate `x0_hat = G_theta(z | v0, v1, a2)`.
   - First implementation: run many scheduler steps with no grad, then rerun or track only the last
     step with grad (DMD2-style) to keep memory bounded.
   - Sweep down from the known-working inference step count; do not start by assuming 4 steps.
3. Sample DMD noise level `sigma`, noise `x0_hat` to `x_t`.
4. Real score `s_real` (direct teacher, Hyunwoo's branch):
   - Frozen one-source SPT with `source=v0`, target=`x_t`, action=`a2`. A single call.
   - Run under `no_grad`; add CFG only after sign/variance tests pass.
5. Fake score `s_fake` (marginal student score via the MSE trick):
   - **One-source** fake-score net conditioned on `v0` ONLY (`v1` hidden), target=`x_t`.
     Trained on the student's `v2` samples, so its optimum is the marginalized student score
     `grad log p'(v2|v0)`. Do NOT feed it `v1`.
   - The Monte-Carlo marginalization comes from the student having sampled `v1 ~ p(v1|v0)` in
     step 2, then the fake-score net hiding `v1` — this is exact, not a biased approximation.
   - During student update, fake-score net is evaluated without parameter grads because the DMD
     vector is detached.
6. Student loss:

```text
loss_student = mean(stop_grad((1 - sigma) / sigma * (v_real - v_fake)) * x0_hat)
```

This is the detached surrogate for `(s_fake - s_real) dot G_theta(z)`. Gradient descent moves the
student along `s_real - s_fake`.

7. Fake-score update:
   - Detach `x0_hat`.
   - Noise it at a fresh or reused timestep.
   - Train fake-score net with flow-matching MSE to predict `eps - x0_hat`.

Gates:

- Q1 mode is explicit in logs/checkpoints.
- DMD vector norm finite and not dominated by tiny `sigma`.
- Flipping the loss sign makes behavior worse in a toy/small test.
- Fake-score loss decreases on cached student outputs.
- Student samples do not immediately collapse or explode.

## Rung 2b - K>1 Monte-Carlo Marginalization (student side)

Only after K=1 is stable. K>1 does NOT touch the teacher (it stays a single direct `p(v2|v0)`
call). K is the number of Monte-Carlo middle samples `v1 ~ p(v1|v0)` used to reduce variance in
the **student marginal** and its fake-score estimate:

- Per step, draw K crossing-constrained middles from the bank; generate one `v2` per middle.
- Train the one-source fake-score net on all K `v2` samples with `v1` hidden — more samples =
  lower-variance estimate of `grad log p'(v2|v0)` (the MSE-trick optimum is unchanged).
- The student loss averages the K per-sample DMD vectors.

Track: fake-score estimator variance vs K; consistency vs K; whether crossing-constrained middles
beat random middles. There is no prior-vs-posterior bias to correct here — `v1 ~ p(v1|v0)` is the
exact generative process being marginalized.

## Rung 3 - Stitching Term (D3)

Add stitching only after DMD is stable.

- Select one latent/video frame from each generated view at the shared world-time.
- Use a disjoint binary selection `W`; no blending, interpolation, or averaging at first.
- Reuse the joint score by slicing: `s_x = W s_v`.
- Add the stitched DMD vector into the student surrogate with a small weight schedule.

Gate:

- With stitching off, measure blur/consistency baseline.
- With stitching on, strip sharpness improves without breaking per-view video quality.
- Ablation should show the term is doing work; do not assume blur returning is guaranteed.

## Rung 4 - More Views And Pairwise Windows (D4)

Only after two-source DMD + stitching works:

- Generate N views with the student.
- Apply D1/D2/D3 on overlapping windows, e.g. `(v1,v2,v3)`, `(v2,v3,v4)`.
- Sum window losses.

This is an empirical bet, not a theorem. Measure drift by chain distance:

- crossing-frame embedding mismatch;
- feature/optical-flow consistency if available;
- human/visual stitched strip inspection;
- degradation from adjacent windows to far-apart windows.

## Build Order

1. Rung 0: released inference + shape probes.
2. Crux tests: flow score conversion and DMD sign.
3. Rung 1a: one-source training harness with checkpointed forward.
4. Rung A + Rung 1b: 2-source forward/training smoke test.
5. (Q1 already resolved to direct-teacher / student-marginal — see HYUNWOO_RECONCILIATION.md.)
6. Middle-bank generation and crossing validation.
7. Rung 2: K=1 DMD loop (direct teacher, one-source fake-score net hiding v1).
8. Rung 2b: K>1 Monte-Carlo student-side marginalization.
9. Rung 3: stitching term.
10. Rung 4: N-view pairwise windows.

## Open Questions For Hyunwoo

1. (RESOLVED per `counterfactual.txt`: direct teacher `p(v2|v0)`, marginalization on the
   student side via the MSE trick. No longer open.)
2. (RESOLVED: no prior-vs-posterior issue on this branch — the student marginal is defined by
   `p(v1|v0)`, so hiding `v1` in the fake-score net is exact.)
3. Crossing constraint: what exact normalized camera/time equality is sufficient, given SPT's
   relative camera embeddings?
4. Few-step student: should we first distill a few-step generator, or is last-step-gradient DMD2
   enough?
5. D3: under what noise levels and selection rules does `s_x = W s_v` remain a good approximation?
6. D4: what drift metric best predicts global consistency failure?

## Guardrails

- Never conflate diffusion timestep with SPT world-time embedding.
- Keep sources clean; noise only the target latent.
- Keep `[target, source0, source1, ...]` order identical across latents, camera embeddings, and
  time embeddings.
- Slice model outputs to target frames before losses or score conversion.
- Disable TeaCache in training.
- Use CFG only after the unconditioned `cfg_scale=1` path passes sign and stability tests.
- Cache prompt embeddings/source latents where possible; do not backprop through VAE/text encoder.
- Treat all D2/D3/D4 claims as approximations until ablations verify them.
- Keep `RESEARCH_CONTEXT.md` and `PLAN.md` paired: context explains why; plan says what to build.
