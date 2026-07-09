# End-to-End Implementation Plan — Multi-View-Consistent World Model via MC-Marginalized DMD

This is the executable, presentation-ready plan. It assumes the resolutions in
`HYUNWOO_RECONCILIATION.md` (direct teacher, student-side Monte-Carlo marginalization,
one-source fake-score net, explicit-position RoPE for crossing). `RESEARCH_CONTEXT.md` = why;
`PLAN.md` = the ladder; this file = the single accurate build sheet + the result each step yields.

---

## 0. What we are building (one screen)

A trainable **student** SPT that conditions a new view on *two* videos and, applied
autoregressively, renders `v0, v1, …, vN` as one shared 3D world — the thing released SPT fails.

Three model instances:

| Instance | Sources | Trainable? | Role |
|---|---|---|---|
| **Student `G_θ`** | `(v0, v1)` — 2-source | yes | generates `v2`; its `v1`-marginal is what we distill |
| **Teacher `p`** | `v0` — 1-source | no (frozen) | `s_real = ∇log p(v2|v0)`, single call |
| **Fake-score net** | `v0` — 1-source, `v1` hidden | yes (online) | `s_fake = ∇log p'(v2|v0)`, marginal student score via the MSE trick |

Objective: `min_θ D_KL( p'(v2|v0) ‖ p(v2|v0) )` with
`p'(v2|v0) = ∫ q_θ(v2|v0,v1) p(v1|v0) dv1`, optimized by DMD. The Monte-Carlo marginalization on
the student side is the novelty. Later: **stitching (D3)** kills blur; **pairwise windows (D4)**
scale past two views.

---

## 1. Ground truth from the code (verified — do not re-derive)

- **Trainable forward:** `WanModel.forward` (`spacetimepilot/model/spacetimepilot.py:202`) supports
  `use_gradient_checkpointing`, gated on `self.training` (`:233`). Signature takes `cam_emb` (dict
  `{"tgt","src"}`) and `frame_time_embedding` (dict `{"time_embedding_src","time_embedding_tgt"}`),
  passed straight to blocks. **Use this for training**, NOT `model_fn_wan_video` (`:758`, no
  checkpointing) and NOT `__call__` (`:629`, `@torch.no_grad`).
- **Edit sites for N-source:** fusion concat `:730`, time concat `:313`, camera concat `:321`,
  `model_fn` dict glue `:792`, **RoPE gather `:226`/`:784`** (index by explicit positions).
- **Conditioning resolutions (asymmetric!):** camera enters at **21 latent frames**
  (pre-downsampled `::4`, `compute_pose_embedding → (21,3,4)=12/frame` → `cam_encoder=Linear(12,dim)`);
  world-time enters at **81 frames** and is downsampled to 21 *inside* the block
  (`temporal_downsampler`). Feed each at the right resolution.
- **Shapes:** 1 video `(B,16,21,60,104)`; 2-video `42`; 3-video `63` latent frames. Patchify
  `(1,2,2)` → **30×52** spatial grid (hardcoded `.repeat(1,1,30,52,1)` at `:314/:323`, so resolution
  is locked to 480×832). Tokens = `latent_frames · 30 · 52`.
- **RoPE table:** `precompute_freqs_cis_3d(head_dim, end=1024)` — supports ≤1024 frames, so 63+ is
  safe. Currently gathered by `arange` (`self.freqs[0][:f]`).
- **Scheduler** (`spacetimepilot/wan/schedulers/flow_match.py`): `add_noise` → `x_t=(1-σ)x0+σε`;
  `training_target` → `ε - x0` (velocity); `σ = self.sigmas[timestep_id]`.
- **Freeze set:** per-block `cam_encoder, projector, frame_time_embedding, temporal_downsampler,
  self_attn q/k/v/o` (30 blocks). Everything else frozen. Never train VAE/text encoder.
- **Env:** conda `counterfactual` (torch 2.4.1+cu121), MIT ORCD/SLURM, 80GB GPU via `srun`/`sbatch`
  (login node has no GPU). Wan2.1 base + ReCamMaster on scratch; **download SPT ckpt** first:
  `hf download zhening/SpaceTimePilot SpacetimePilot_1.3B_v1.ckpt`.

## 2. The math to hard-code once (unit-tested before any training)

```
x0  = x_t − σ·v_pred
ε   = x_t + (1−σ)·v_pred
s(x_t) = −ε/σ = −(x_t + (1−σ)·v_pred)/σ
A = s_real − s_fake = −((1−σ)/σ)·(v_real − v_fake)
loss_student = mean( stop_grad( (1−σ)/σ · (v_real − v_fake) ) · x0_hat )   # descent moves along s_real−s_fake
```
`v_real` = teacher velocity (source=`v0`); `v_fake` = 1-source fake-score velocity (source=`v0`,
`v1` hidden). Slice every prediction to target frames `pred[:,:,:21]` before conversion. Recover
`σ` from `scheduler.sigmas[timestep_id]`, never `timestep/1000`. Avoid `σ=0`.

---

## 3. Build ladder (each rung: what to build → gate → result to show Hyunwoo)

### Rung 0 — Environment, checkpoint, inference shape probe
- Download SPT ckpt to scratch. Run `single_video_test.py` on a GPU node
  (`num_inference_steps=20` / known-working). Add temporary shape prints at VAE encode, fusion
  concat, one `DiTBlock.forward`.
- **Gate:** released inference reproduces a clean video in the `counterfactual` env; shapes match §1.
- **Result:** shape table + one released-SPT sample. *(Largely done in a prior worktree.)*

### Rung 1 — Crux math unit tests (pure tensors, no model)
- `spacetimepilot/training/score.py`: `velocity_to_score`, `dmd_vector_from_velocities`,
  `student_dmd_loss`. `tests/test_score.py`: reconstruct analytic score from known `(x0,ε,σ)`;
  assert DMD loss gradient moves `x0_hat` along `s_real−s_fake`; assert flipped sign reverses it.
- **Gate:** tests green.
- **Result:** a passing sign/scale test suite — the "we can't be silently training toward garbage"
  guarantee (RESEARCH_CONTEXT §7).

### Rung 2 — One-source training smoke test (graph test)
- `spacetimepilot/training/`: `latents.py` (VAE encode under `no_grad`, build latent input, slice
  target), `freeze.py` (freeze all → unfreeze the set → assert grad mask), `steps.py`
  (`one_source_smoke_step`), `data.py` (tiny batch loader).
- Encode `source,target` no_grad → sample timestep → noise **target only** → `latents_input =
  cat([tgt_noised, src], dim=2)` → `dit.train()` →
  `dit.forward(..., use_gradient_checkpointing=True)` → slice `[:,:,:21]` → throwaway MSE vs
  `training_target` → backward.
- **Gate:** finite loss; grads nonzero **only** on unfrozen modules, zero/None elsewhere;
  checkpointing actually active; no VAE/text-encoder grads; `enable_vram_management` OFF; fits 80GB.
- **Result:** "SPT trains outside inference, gradients land exactly where intended." (This MSE is a
  graph test, explicitly *not* the objective.)

### Rung 3 — N-source student extension + two-source smoke
- Edit `spacetimepilot/model/spacetimepilot.py` at the **5 sites** (§1). Source handling becomes an
  ordered list `[target, source0, source1, …]` for latents, camera, time. **RoPE:** allow an
  explicit per-token position tensor into the freq gather; default (arange) reproduces released
  behavior.
- **Hard backward-compat gate:** with exactly one source, output is **numerically identical** to
  released SPT (add `test_one_source_equivalence`). This protects the frozen teacher.
- Two-source smoke: sources `(v0, v1_dup)`; verify 63-frame latent, cam/time cover 63 frames, target
  slice = first 21; grads flow; memory OK.
- **Result:** the extended student runs on 3 videos; a (blurry, as Hyunwoo predicts) 2-source
  sample; the 1-source equivalence test passes.

### Rung 4 — Middle bank + crossing
- `scripts/generate_middle_bank.py` + `spacetimepilot/training/middle_bank.py`.
- Per `v0`: **heuristically** choose crossing trajectories (`a1` shares a `(camera, world-time)`
  point with `a2`; Slack: "camera trajectory needs to be heuristically chosen"). Generate `K`
  middles `v1` with the **frozen released** SPT offline. Cache: source latents, camera/time
  embeddings, prompt embedding, crossing-frame index, seed. Always **detach** `v1`.
- Validation script: report max camera/time embedding mismatch at the crossing frame (in the
  processed space, not raw JSON).
- **Result:** a reusable bank + a crossing-validation pass/fail report.

### Rung 5 — K=1 DMD loop (the core method)
- `steps.py`: `dmd_step_k1`, `fake_score_update`. `score.py`: `ScoreProvider` fixed to
  `direct_teacher`.
- Per step: load `v0`, one `v1`, action `a2`. **Student** generates `x0_hat` (multi-step no_grad
  prefix + last-step gradient, DMD2-style; start from known-working step count, sweep down). Sample
  DMD `σ`, noise `x0_hat→x_t`. **`s_real`** = teacher(source=`v0`, target=`x_t`) under `no_grad`.
  **`s_fake`** = fake-score net(source=`v0`, `v1` hidden, target=`x_t`), params detached.
  `loss_student` per §2. **Fake-score update:** detach `x0_hat`, noise at a fresh timestep, MSE to
  predict `ε − x0_hat` conditioned on `v0`. Two AdamW optimizers, 1:1 cadence. `cfg_scale=1` first.
- Stabilizers: standard DMD per-sample gradient normalization; sample `σ` in a mid-range.
- **Gate:** DMD-vector norm finite, not dominated by tiny `σ`; flipped sign is worse on a toy run;
  fake-score loss decreases on cached student outputs; student doesn't collapse/explode; branch +
  `σ` logged.
- **Result:** the first real training curve; `v2` samples improving over the released baseline —
  "MC-marginalized DMD trains."

### Rung 6 — K>1 Monte-Carlo marginalization (student side)
- Draw `K` middles/step; generate one `v2` per middle; train the 1-source fake-score net on all `K`
  (with `v1` hidden) → lower-variance marginal score; student loss averages the `K` DMD vectors.
  Teacher stays a single `p(v2|v0)` call.
- **Result:** ablation K=1 vs K>1 (fake-score variance, consistency) — the "marginalization helps"
  evidence and the paper's novelty in action. No prior/posterior bias to correct here.

### Rung 7 — Stitching term (D3, anti-degeneracy)
- Select one frame per generated view at the shared world-time → `x = W[v0…vN]ᵀ`, `W` a disjoint
  binary selection (`WWᵀ=I`). Reuse the joint score by slicing: `s_x = W·s_v` (compute `s_v` with
  the flow-matching formula). Add the stitched DMD vector to the student surrogate on a small weight
  schedule. Conceptually: align the stitched marginal `p(x)` to the single-video `p_1(x)`.
- **Gate:** stitching-off baseline is blurry; stitching-on sharpens the strip without breaking
  per-view quality; ablation shows the term does work.
- **Result:** the blur ablation (off→gray-fog, on→sharp+consistent) — the degeneracy-prevention
  result Hyunwoo asked about.

### Rung 8 — N-view pairwise windows (D4, scaling)
- Generate `N` views with the student; apply D1/D2/D3 on overlapping windows (`123, 234, …`); sum
  window losses. Augment by swapping/cropping trajectories across times to manufacture more pairs.
- Measure **drift** as a function of chain distance.
- **Result:** N-view consistency vs chain distance — tests Hyunwoo's own open transitivity bet.

---

## 4. Evaluation — the results that defend this to a PhD

Baseline = released SPT generating views independently (the inconsistent model Hyunwoo says "we
already checked" fails). Report, per rung:

1. **Multi-view geometric consistency** (the headline): at shared world-times across generated
   views, measure cross-view correspondence / reprojection error (point tracking or feature
   matching), and/or COLMAP reconstructability of the generated set. Student should beat baseline.
2. **Stitched-strip quality** (D3): sharpness + temporal smoothness of `x = Wv` vs a real single
   video. Shows consistency isn't bought by collapse.
3. **Per-view video quality**: FVD-style / visual, so improvements aren't degeneracy.
4. **Ablations**: K=1 vs K>1; stitching on/off; crossing-constrained vs random middles;
   direct-teacher sign test. Each isolates one claimed mechanism.

Frame the talk honestly (matches Hyunwoo's own hedges): DMD keeps samples realistic and prevents
collapse; consistency is an **emergent, measured** property of multi-view conditioning + crossing +
stitching + pairwise windows — validated by metric 1, not asserted. Transitivity → global
consistency (Rung 8) and marginalization tractability are the two open bets the experiments test.

## 5. Cross-cutting engineering decisions (standard defaults — stated so they're deliberate)

- **Memory:** bf16; batch=1; gradient checkpointing on student + fake-score; teacher and fake-score
  run under `no_grad` during the student step; sequential residency; 3×1.3B fits 80GB. `enable_vram
  _management` **OFF** for training. `dit.train()` on student+fake (checkpointing needs it; LayerNorm
  only, no BN/dropout, so it's safe), `teacher.eval()`. **TeaCache OFF** in training.
- **Generation:** DMD2 last-step gradient; don't assume a 4-step student — start from a known-working
  step count and sweep down.
- **Optimizers:** AdamW over unfrozen params only; a **separate** optimizer for the fake-score net.
- **Invariants:** keep `[target, source0, source1, …]` order identical across latents, camera, time,
  and RoPE positions; sources clean (noise only the target); slice to target frames before any loss.

## 6. The one open detail (Thursday 10:30 ET meeting)

The exact **3D-position convention** to feed into RoPE for the crossing constraint (which
world/temporal coordinates, how normalized). Not a blocker: Rungs 0–3 and even Rung 5 (K=1 DMD)
run with heuristic crossing + camera embedding + the default `arange` RoPE; explicit-position RoPE
is wired as plumbing now and switched on once the convention is set. It bites hardest at Rung 4+
(making `v1` a genuinely informative anchor).

## 7. Suggested milestone order for the presentation

1. Rung 0 shapes + Rung 1 green math tests. 2. Rung 2/3: "SPT is trainable and extends to N
sources, 1-source stays exact." 3. Rung 4 bank + crossing validation. 4. **Rung 5 K=1 DMD training
curve + samples vs baseline** (the centerpiece). 5. Rung 6 K-ablation (the novelty). 6. Rung 7 blur
ablation. 7. Rung 8 drift-vs-chain-distance. Metric 1 (geometric consistency) reported from Rung 5
onward.
