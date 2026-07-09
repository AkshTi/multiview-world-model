# Rung 5 — the K=1 DMD loop (design note)

The centerpiece. Everything before this rung was scaffolding to make *this* loop
runnable and checkable. This note pins the update order, which model sees which inputs,
and the exact tensor flow, so the code is a transcription of an agreed structure rather
than an improvisation. It is written to be reusable in the Hyunwoo presentation.

Branch: **direct-teacher / student-side Monte-Carlo marginalization** (Q1 resolved, see
`HYUNWOO_RECONCILIATION.md`). K=1 means one middle `v1` drawn per step from the bank;
Rung 6 averages over K.

## Objective

Force the student's *marginal-over-v1* distribution to match the direct teacher:

    minimize  D_KL( q_θ(v2|v0)  ||  p(v2|v0) ),
    where     q_θ(v2|v0) = ∫ q_θ(v2|v0,v1) p(v1|v0) dv1   (marginalize the middle).

The DMD gradient of this KL is the two-arrow update evaluated at student samples v2:

    ∇_θ KL  ∝  E_{v2~q_θ, t}[ ( s_fake(v2_t) − s_real(v2_t) ) · ∂v2/∂θ ]

- `s_real = ∇log p(v2_t|v0)`  — the frozen **teacher** (1-source, source = v0).
- `s_fake = ∇log q_θ(v2_t|v0)` — the **fake-score net** (1-source, source = v0, **v1 hidden**),
  learned online by denoising the student's own samples (MSE trick → marginal score).

Descent moves the student *up* the teacher's log-density and *away* from its own pile-up.

## Three models (who sees what)

| model        | # sources | conditioning        | params        | mode    |
|--------------|-----------|---------------------|---------------|---------|
| **student** G_θ | 2         | v0 **and** v1        | fine-tune subset trainable | `.train()` |
| **teacher** s_real | 1      | v0 only             | **frozen** (released SPT) | `.eval()` |
| **fake-score** s_fake | 1   | v0 only (**v1 hidden**) | trainable (φ) | `.train()` |

- The **student is the only N-source model** (the Rung-3 edited DiT). It is the only place
  v1 enters. This is what makes s_fake, seeing only v0, learn the *marginal* score: v1 is a
  hidden variable of the sample-generating process, so denoising-MSE regression recovers the
  conditional mean over v1 → the marginal.
- Teacher and fake-score are **separate DiT instances** both built from released weights.
  They must be separate: the teacher must stay at released weights (it is `p`), while the
  student's fine-tune modules drift, so the student cannot double as the teacher.
- **Fake-score trainable set (smoke choice):** the same Rung-2 freeze mask as the student
  (`freeze.set_trainable`), so one `assert_grad_mask` covers both and memory stays bounded.
  *Production option:* full fake-score or a LoRA fake-score (DMD2 uses LoRA) — the fake net
  should ideally be more expressive than the student's adapter. Flagged, not required for the
  gate.

## The student is a one-step generator

DMD distills the multi-step teacher into a **one-step** conditional generator. To get
`∂v2/∂θ` we need a differentiable map noise → v2; a full multi-step sampler is not
differentiable-through cheaply, so (as in DMD/DMD2) the student runs **once at the maximum
noise level** and reads off the clean sample:

    z = x_T ~ N(0, I)                        # shape = target latent (B,16,21,60,104)
    v_pred = student([x_T, v0, v1], t=T)     # 2-source fused forward, sliced to target frames
    v2_hat = x_T − σ_T · v_pred              # x0_from_velocity  (score.py, tested)

`t=T` is `scheduler.timesteps[0]` / `sigmas[0]` (the highest σ ≈ 1). `v2_hat` carries
gradient through the student's trainable modules only. (Few-step DMD2 generation is a later
refinement; K=1 one-step is the gate.)

## One DMD iteration (exact order)

    # --- generate (grad on θ) ---
    z      = randn(target_shape)
    v2_hat = x0_from_velocity(z, student([z, v0, v1], t=T), σ_T)

    # --- G-step: distribution-matching gradient on the student ---
    t'   = random timestep;  eps ~ N(0,I)
    x_t' = (1−σ') v2_hat + σ' eps                     # scheduler.add_noise(v2_hat, eps, t')
    with no_grad:
        v_real = teacher([x_t', v0], t')              # 1-source
        v_fake = fake(   [x_t', v0], t')              # 1-source, v1 hidden
    loss_G = student_dmd_loss(v2_hat, v_real, v_fake, σ', normalize=True)   # score.py
    loss_G.backward();  opt_G.step();  opt_G.zero_grad()

    # --- D-step: teach the fake-score net the student's current distribution ---
    t''   = random timestep;  eps2 ~ N(0,I)
    x_t'' = (1−σ'') v2_hat.detach() + σ'' eps2
    v_fp  = fake([x_t'', v0], t'')                    # grad on φ
    loss_D = MSE( slice_target(v_fp), eps2 − v2_hat.detach() )   # flow-match target = eps − x0
    loss_D.backward();  opt_D.step();  opt_D.zero_grad()

Ordering rationale:
- **Two independent backward graphs, never alive at once** — `v_real`/`v_fake` are `no_grad`
  in the G-step; the D-step uses `v2_hat.detach()`. Keeps peak VRAM to one live graph on the
  48 GB L40S.
- G before D means s_fake is always one step behind the generator — standard DMD, harmless.
- The score→velocity algebra is entirely inside `student_dmd_loss` (x_t terms cancel;
  `normalize=True` is the DMD per-sample-magnitude stabilizer that stands in for the
  weighting w(t)). Sign already unit-tested in `tests/test_score.py`.

## Why the arrow ≈ 0 at step 0 (a sanity check, not a bug)

Teacher and fake-score start from the *same* released weights, so at iter 0
`v_real ≈ v_fake` ⇒ `loss_G ≈ 0`. The signal appears as the fake-score net adapts to the
student's outputs and the two scores separate. The smoke reports `mean|v_real − v_fake|`
growing over a few iters as evidence the loop is live.

## Rung 5 gate (what the GPU smoke asserts)

1. `loss_G`, `loss_D` finite over several iters.
2. **Student grad mask:** after `loss_G.backward()`, grads only on the student's unfrozen
   modules; none on frozen student params, none leaking from the teacher/fake forwards.
3. **Fake grad mask:** after `loss_D.backward()`, grads only on `fake_dit`'s unfrozen
   modules; none on the student, none on the teacher.
4. **Teacher inert:** every teacher param `requires_grad == False`, no grad.
5. `mean|v_real − v_fake|` starts ≈ 0 and increases.
6. Fits in 48 GB (three 1.3B DiTs + one live graph).

CPU-testable core (no SPT, no GPU): the new fake-score loss target and the detachment
structure (G-loss reaches `v2_hat` only; D-loss reaches `v_fake_pred` only; no cross-leak).
See `tests/test_dmd.py`. The two-arrow math is already covered by `tests/test_score.py`.

## Middle handling (K=1)

Pull one cached middle from the Rung-4 bank per step: decode-free path is not needed — we
re-encode the mp4 exactly like v0 (`encode_video_nograd`). Camera representation for v1 as a
*source* is still the OPEN Thursday item (source vs. target camera embeddings differ); the
smoke uses the middle's stored `cam_idx` with the source-camera embedding path and flags the
choice. Rung 6 replaces the single v1 with a K-average of banked middles.
