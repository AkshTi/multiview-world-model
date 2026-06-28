# Research Context: Multi-View-Consistent World Model via DMD Distillation

> **What this document is.** A single integrated source-of-truth for the project, weaving
> together four primary sources plus the code reality:
> - **[HANDOFF]** — the original project handoff (`CLAUDE_CODE_HANDOFF.md`).
> - **[MEETING]** — the recorded advising call with Hyunwoo Ryu (`transcript.txt`).
> - **[DERIV]** — Akshata's worked derivations (D1–D3).
> - **[SLACK]** — Hyunwoo's written follow-up to those derivations.
> - **[CODE]** — facts verified by reading the SPT source in this repo.
>
> Inline tags mark provenance so a future agent knows what is *settled math*, what is
> *Hyunwoo's verbal intuition*, what is an *engineering fact*, and what is *open*. The
> companion `PLAN.md` is the execution ladder; this file is the *why* behind it. Read this
> before reasoning about the objective.

---

## 0. How to work on this project (people + style)

- **[HANDOFF]** Akshata is an MIT undergrad; advisor/collaborator **Hyunwoo Ryu**; weekly
  meeting **Thursdays 11am ET** (set in **[MEETING]** ~47:00).
- **[HANDOFF]** Learning style: understand the *why* before algebra/code; be able to teach
  the derivation back. When confused, rebuild from a lower level rather than pushing forward.
  Prefer honest "this part is open/approximate/unverified" over confident overstatement.
- **[HANDOFF]** Build **incrementally, verify each rung before the next.** Do not write the
  whole DMD loop in one shot. Forward pass → noising → one score → next score, checking
  shapes and finiteness at each step.
- **[MEETING ~28:00]** Hyunwoo: "I'm giving you the most challenging problem… tell me if it's
  too difficult." The difficulty is acknowledged; the derivations are genuinely open research.

---

## 1. The research goal (one paragraph)

**[HANDOFF]** Camera-conditioned video models (SPT / SpaceTimePilot, BulletTime, ReCamMaster)
can re-shoot a factual video `v0` from a new camera path + world-time to invent a
counterfactual view. **Problem:** each invented view silently commits to a *different*
underlying 3D world (an underdetermined posterior over worlds), so stitching views at the
same world-time gives inconsistent geometry / flicker. **Goal:** force every invented video
to be a render of *one shared world*. **Method:** fine-tune from an existing 2-video model and
**distill** (no multi-view ground truth exists) toward a *consistent* teacher, using **DMD**
over a *marginalized* distribution, plus a **stitching** term to prevent blurry collapse, plus
**crossing constraints** for geometric truth, scaled past two views by **pairwise-window**
distillation.

**[HANDOFF] Critical reframe:** because no multi-view ground truth exists, the training loop
is **NOT a plain denoising-MSE fine-tune**. It is a **DMD distillation loop**. Plain
denoising-MSE only appears *inside* one auxiliary network (the fake-score net). Do not build a
vanilla "corrupt target, MSE against truth" trainer as the main objective.

**[MEETING ~15:31–21:01] The intuition Hyunwoo gave for the goal:** think of it as building a
world *incrementally* — "autoregression not over frames but over videos." First generate one
video `v0`; generate a different view; then, **conditioning on those videos, build the next
view, and recurse** — many videos rendering the same world consistently, distilling the
capability out of a pre-trained 2-video model. (He floated a "pre-render anchor views for a
playable world" framing then **explicitly dropped it as confusing** ~20:24 — ignore it; the
clean statement is just autoregressive-over-videos.)

---

## 2. Notation

- **[HANDOFF/DERIV]** Videos `v0` (factual — actually happened, real footage), `v1, v2, …`
  (counterfactual — invented views of the same world). Action `a_n = (g_n, t_n)` = camera
  pose `g_n` + world-time `t_n`. (BulletTime/SPT disentangle pose from time.)
- **[DERIV]** `p(·)` = how plausible the model thinks a video is. `p(v1|v0)` = plausibility of
  `v1` given `v0`.
- **[DERIV]** The object ultimately wanted is the **joint**, conditioned on camera/time
  instructions: `p(v0,…,vN | a0,…,aN)`. Modeling jointly lets us *demand* a shared world and
  penalize internal disagreement. Factorized:
  `p(v0,v1,v2|a) = p(v0)·p(v1|v0)·p(v2|v0,v1)`.
- **[HANDOFF]** We *have* `p(v0)` and `p(v1|v0)` (SPT/ReCamMaster). We *lack*
  `p(vn|v0,…,v_{n-1})` (many-video conditioning) → must **distill**, not train.
- **[HANDOFF/DERIV] Score / "arrow"** `s(v) = ∇_v log p(v)` = direction to nudge a video to
  make it more plausible. Central because the density needs the intractable partition function
  `Z`, but the arrow doesn't (`∇_v log Z = 0`). **A trained diffusion model IS its arrow
  field** — its noise/velocity prediction is the score, up to a known scaling (see §7).

---

## 3. Derivation D1 — the DMD gradient (settled; off-the-shelf, not the contribution)

**Goal [DERIV]:** train `G_θ` so the student distribution `q_θ` matches the teacher `p` by
minimizing `KL(q_θ ‖ p)`. `q_θ` is the trainable student; `p` is the frozen teacher (SPT/BT).

```
L(θ) = KL(q_θ ‖ p) = E[ log q_θ(v) − log p(v) ]
```

Write each student sample as `v = G_θ(z)`, `z ~ N(0,I)`:

```
L(θ) = E_z[ log q_θ(G_θ(z)) − log p(G_θ(z)) ]
∇_θ L = E_z[ ∇_θ log q_θ(G_θ(z)) − ∇_θ log p(G_θ(z)) ]
```

**Teacher term:** `∇_θ log p(G_θ(z)) = s_real · ∂_θ G_θ(z)`, with `s_real = ∇_v log p`.

**Student term** has two pieces (chain rule + explicit θ-dependence of `q_θ`):
`∇_θ log q_θ(G_θ(z)) = [∂_θ log q_θ(v)]_explicit + s_fake · ∂_θ G_θ(z)`,
with `s_fake = ∇_v log q_θ`.

**The explicit piece vanishes** [DERIV]:
```
E[ ∂_θ log q_θ(v) ] = ∫ q_θ ∂_θ log q_θ dv = ∫ ∂_θ q_θ dv = ∂_θ ∫ q_θ dv = ∂_θ(1) = 0
```
(total probability is 1 → its θ-derivative is 0). So we have **no dependence on the student's
normalizer or density value** — only on its *arrow*:

```
∇_θ L = E_z[ (s_fake − s_real) · ∂_θ G_θ(z) ]
```

Descent nudges each generated video along **`A = s_real − s_fake`** (uphill on the teacher,
away from the student's own pile-up), pushed through the generator.

**[HANDOFF] Implement as a detached surrogate:**
`E_z[ stop-grad(s_fake − s_real)ᵀ · G_θ(z) ]`, then `.backward()`.

- `s_real`: frozen teacher score — free (the diffusion model *is* its arrow field).
- `s_fake = ∇_v log q_θ`: **NOT free** — the generator only *samples*, never reports its own
  density's score. Train a **second small score network online** (denoising-score-matching on
  the student's own outputs) that chases the shifting student distribution. This is the
  VSD/DMD two-network structure; the `−s_fake` term is the **anti-blur force**.

**[MEETING ~25:48–26:35]** Hyunwoo on the DMD paper: "the motivation is just matching the
score… in practice you take the inner product and **detach** the score functions." He also
predicted (correctly, see D3) that "we might end up with a *linear-projected* version of the
score function" once stitching is added.

---

## 4. Derivation D2 — the marginal (consistent) teacher score (THE contribution)

### 4.1 The single most useful trick [MEETING ~29:41–35:47] — "the transcript trick"

Hyunwoo drew this on screen and called it the single most useful trick in the
diffusion/score-matching literature. **Least-squares regression returns the conditional
mean:**
```
min_f  E[ ‖ f(x) − y ‖² ]   ⟹   f*(x) = E[ y | x ]
```
Pointwise proof: `∂/∂c E[‖c − y‖² | x] = 2(c − E[y|x]) = 0 ⟹ c = E[y|x]`.

**Why it matters here:** if you train a network with least-squares but *hide* a variable, it
outputs the **average over that hidden variable**. "You just get rid of the variable you want
to marginalize and train your NN on it — then you get the marginalized score function." And
**[MEETING ~34:08–35:42]**: the *joint* score is the sum of conditional scores
(`s_{x,y} = s_x + s_y`), which is easy; what you actually want is the *marginal* score, which
this regression trick gives you.

### 4.2 The marginal teacher [DERIV]

The *consistent* teacher `p'` is a two-hop marginal — a mixture, one simple two-video teacher
per middle `v1`:
```
p'(v2|v0) = ∫ p(v2|v0,v1) · p(v1|v0) dv1
```
The objective is **training away the inconsistency** between `p'` (consistent) and `p`.

Its arrow (slope ÷ height):
```
∇_{v2} log p' = ∇_{v2} p' / p'
∇_{v2} p'      = ∫ [∇_{v2} p(v2|v0,v1)] p(v1|v0) dv1                       (differentiate inside)
              = ∫ p(v2|v0,v1) [∇_{v2} log p(v2|v0,v1)] p(v1|v0) dv1        (slope = height × arrow)
```
Divide by `p'` to get a weighted average of arrows:
```
s_real = ∫  [ p(v2|v0,v1) p(v1|v0) / p'(v2|v0) ]  ∇_{v2} log p(v2|v0,v1)  dv1
```
The bracket **is a Bayes posterior**:
```
p(v2|v0,v1) p(v1|v0) / p'(v2|v0) = p(v1 | v2, v0)      ("how much to trust this middle, given the v2 we drew")
```
So:
```
s_real = E_{ v1 ~ p(v1|v2,v0) } [ ∇_{v2} log p(v2|v0,v1) ]
```

**Plain words [HANDOFF]:** "consistent arrow = average of two-video arrows over middles,
weighted by how well each middle fits." **In code:** call the frozen two-video teacher several
times with different middles `v1`, read each velocity/score, average them. The averaging is
*exactly* the regression trick of §4.1 made explicit.

### 4.3 Noise-level version (needed for real DMD) [HANDOFF]

Identical derivation with the diffusion clock `t` on `v2`, so the posterior conditions on the
*noised* `v2,t`:
```
s_real = E_{ v1 ~ p(v1 | v2,t, v0) } [ ∇_{v2,t} log p_t(v2,t | v0,v1) ]
```

### 4.4 The open subtlety (prior vs posterior) — DO NOT silently pick one

**[HANDOFF]** The exact estimator averages over the **posterior** `p(v1|v2,t,v0)`; the cheap
recipe samples the **prior** `p(v1|v0)` — which is **biased**. The **crossing constraint**
(middles share a (camera, world-time) point with `v2`) is what's meant to make the cheap
version safe. There is a tension between two artifacts: the transcript trick (regression
optimum = conditional expectation = *unbiased* posterior average) vs. the doc's
prior-sample-and-average (biased). **First implementation: prior-sampling with
crossing-constrained middles, marked in code as an approximation.** Posterior-correct version
is **Open Q1** for Hyunwoo. (The principled route is to *train a net* to regress the two-video
velocity from `(v2,t, v0)` marginalizing `v1` — by §4.1 its optimum is the posterior average.)

### 4.5 Hyunwoo's written sharpening [SLACK]

> "Keep `q` the model you'd like to train and `p` what you have. The essence of DMD is a
> relaxation of `D_KL(q_θ ‖ p)`. `q_θ` and `p` should represent the **same joint
> distribution, factorized differently.**"

Concretely:
- Teacher joint you *have*: `p(v2,v0) = p(v2|v0) p(v0)`.
- Student joint you *want*: `q_θ(v2,v0) = ∫ dv1 · q_θ(v2|v1,v0) · q_θ(v1|v0) · q_θ(v0)`.
- **Reuse** `p(v1|v0)` and `p(v0)` for `q_θ(v1|v0)` and `q_θ(v0)` (we already have them) → the
  **only piece actually trained is `q_θ(v2|v1,v0)`.**
- **[SLACK] Caveat (important, flagged but unresolved):** while the score of
  `p(v2,v0)=p(v2|v0)p(v0)` is the *sum* of the scores of `p(v2|v0)` and `p(v0)`, **this is NOT
  generally true for the *diffused* distribution.** ⟹ the marginalized/joint score must be
  **estimated** (e.g., by the fake-score net / regression trick), not assembled by adding
  conditional scores. This is the theoretical crux behind why `s_fake` needs its own network.

---

## 5. Derivation D3 — stitching score transport (settled)

**[MEETING 2:24–2:43] Why "degenerate" = blurry, not just different:** "two gray squares can
match each other, and that would solve the objective." Consistency alone has a cheat — all
videos collapse to the same gray blur (two blurs trivially "agree"). **Stitching forbids it.**

**[DERIV/HANDOFF]** Take **one frame from each of N jointly-generated videos** at the shared
world-time, assemble into one strip `x = Wv`, where `v` stacks all the videos' numbers and `W`
is a linear selection/masking matrix (rows each holding a single `1` — the kept pixel).
Demand `x` "looks like a real single video" (sharp + temporally smooth). Blur fails; sharp +
consistent passes.

**Transport result [DERIV].** Noising: `v_t = v + σε`, `ε ~ N(0,I)`. Then
```
W v_t = W v + σ W ε = x + σ W ε ,        Cov(W ε) = W Wᵀ = I   (disjoint, un-blended selection)
```
Because `WWᵀ = I`, the strip's noise is again standard, so noising **commutes** with slicing.
With the noising-defined arrow `−(v_t − v)/σ²`:
```
W( −(v_t − v)/σ² ) = −(W v_t − W v)/σ² = −(x_t − x)/σ²    ⟹    s_x(x_t) = W · s_v(v_t)
```
So you **reuse the joint score — no new network** for the stitching term. Same DPS /
linear-inverse-problem machinery. **Open Q2:** exact conditions on `W` (slice-then-average vs.
average-then-slice equality holds at high noise / genuinely disjoint frames).

**[MEETING 21:26–27:35] The framing Hyunwoo gave for D3 — why linearity is the whole point.**
Picking one frame per video = a **linear projection** `y = Wx`, exactly the setup of
**inverse problems with diffusion priors**. He pointed to **DPS** ("you should know this if
you work on diffusion models") and the in-painting benchmark: observe only a masked linear
projection, recover the full signal using the diffusion prior. Analogies he gave: MRI/CT
(linear 3D→2D projection), Gaussian blur (convolution = linear), super-resolution (linear).
"Masking is a linear operator… that's why linearity is very important — there's a very
successful treatment in inverse problems." This is *why* the clean `WWᵀ=I` transport exists
and `s_x = W s_v` works.

---

## 6. Derivation D4 — scaling past two views (honest bet, NOT a theorem)

**[HANDOFF]** The teacher conditions on ≤2 videos; the student should generate 3+. Plan:
generate N videos, enforce the two-video teacher (D1/D2/D3 machinery) on **every overlapping
window** (sliding 123, 234, 345…). **Claim:** overlapping-pairwise consistency ⇒ global
N-view consistency, **by transitivity**. **Status: assumption, not theorem.** Justified by
(a) window *overlap* pinning shared videos across adjacent windows, (b) crossing constraints
giving chain-distance-independent anchors. **Failure mode to watch — drift:** tolerated slack
accumulating coherently across the chain so distant videos disagree while all neighbors pass.

**[MEETING 37:11–43:06] Hyunwoo's version, matching D4 exactly.** `P` = teacher (freeze; SPT /
ReCamMaster / any 2-video model). `P_fake` = the autoregressive student. The lengths must
match for a KL, but we only have 2-video teachers, so: **generate 3+ with the student, align
all valid two-video combinations.** He pointed to **Self-Forcing**, which addresses the
**length mismatch** between a student that generates a longer trajectory and a teacher fixed at
a shorter context, via **sliding-window distillation (123, 234, …)**. "Window length = the
video length, and we have 2 videos only." Plus an SPT/BulletTime-specific augmentation: swap /
crop different trajectories across times to manufacture more 2-video pairs to be consistent on.

---

## 7. The velocity → score bridge (the math-to-code crux) [CODE]

Every derivation needs a **score** `∇_{x_t} log p_t`, but SPT outputs **velocity**. This is
the single most error-prone junction. From `spacetimepilot/wan/schedulers/flow_match.py`:
- forward / `add_noise`: `x_t = (1−σ)·x_0 + σ·ε`, `ε ~ N(0,I)`
- `training_target = noise − sample`, i.e. the model learns `v = ε − x_0` (flow matching).

Invert the two linear relations (`x_t = (1−σ)x_0 + σε`, `v = ε − x_0`):
```
x_0 = x_t − σ·v_pred
ε   = x_t + (1−σ)·v_pred
score  s(x_t) = −E[ε|x_t]/σ = −( x_t + (1−σ)·v_pred ) / σ
```
Therefore the **DMD arrow collapses to a velocity difference** (the `x_t` term cancels):
```
A = s_real − s_fake = −((1−σ)/σ) · ( v_real − v_fake )
```
**Read `v_pred` from `model_fn_wan_video`'s output sliced to target frames**
`[:,:,:tgt_latent_length]`. **Unit-test this first** (feed known `x_0, ε`, recover the
analytic score). A wrong sign/scale here silently trains toward garbage — `flow_match.py` is
the authority, not assumptions. **This is the concrete realization of [HANDOFF]'s "a trained
diffusion model IS its arrow field, up to a known scaling" and [SLACK]'s warning that scores
must be read carefully under diffusion.**

---

## 8. Architecture — Wan 2.1 + SPT (what the code IS) [HANDOFF + CODE]

### Stack (three historical layers, all in the SPT files)
- **Stock Wan 2.1 (1.3B):** latent video diffusion. Frozen **3D-VAE** compresses video →
  latent; **DiT** transformer denoises in latent space. Borrowed "what video looks like"
  knowledge. **Never modified.**
- **ReCamMaster layer:** added video-conditioning (concat a source latent onto the target) +
  camera injection inside each DiT block. Inherited.
- **SPT layer:** added explicit **world-time** control + **source-aware** cameras (separate
  src/tgt). This is the model we fine-tune from.

### Shapes that govern everything (memorize) [HANDOFF, confirmed CODE]
- VAE: **81 RGB frames → 21 latent frames**, 8× spatial (480×832 → 60×104), **16 latent
  channels**. `out_T = (num_frames−1)//4 + 1 = (81−1)//4+1 = 21`. (Verified in
  `wan/models/wan_video_vae.py`, `tiled_encode`.)
- Patchify `(1,2,2)` → spatial grid **30×52** (hardcoded for 480×832).
- One video latent: `(B,16,21,60,104)`. Two concatenated: `(B,16,42,60,104)`. N videos:
  `(B,16,21·N,60,104)`.
- Token sequence length after patchify = `f·h·w` with `f` = **concatenated** frame count
  (e.g. 2-video → `42·30·52 ≈ 65.5k`). **Attention is quadratic in sequence length** →
  doubling videos ~quadruples attention cost. **This is why the teacher caps at 2 and D4
  scaling is pairwise.**

### Key architectural facts [HANDOFF, verified CODE — file is `spacetimepilot/model/spacetimepilot.py`]
- **Fusion = latent-channel concat along the frame axis, ONCE outside the DiT.**
  Line `:730`: `latents_input = torch.cat([latents, source_latents], dim=2)`. Self-attention
  then mixes the two videos because they share one sequence. (Sources are VAE-encoded at
  `:687`; `__call__` is `@torch.no_grad()` at `:629` — INFERENCE only.)
- **Conditioning (camera, world-time) injected ADDITIVELY inside every DiTBlock.** Two concat
  sites: time `:313` (`torch.cat([tgt_time, src_time], dim=1)`), camera `:321`
  (`torch.cat([cam_tgt, cam_src], dim=1)`); additive injection at `:316` and `:326`. The
  `model_fn_wan_video` dicts are built at `:792–796` (`{tgt, src}` for time and camera).
- **Order invariant:** latents, camera, time all concatenated in the SAME `[tgt, src]` order.
  Any N-video extension MUST use one canonical ordering at every concat site, or video i's
  conditioning lands on video j's tokens (silent bug, no error).
- **[CORRECTED — RoPE is index-based, NOT world-coordinate-based.]** `precompute_freqs_cis`
  uses `torch.arange(end)` — sequence indices only. Two patches at the same world point but
  different sequence positions get *different* RoPE. **The crossing constraint cannot ride on
  RoPE**; world info enters only through the additive **camera embedding** (or must be enforced
  via the loss). ⚠️ **[MEETING 3:31–4:47]** Hyunwoo verbally assumed the *opposite* ("same 3D
  rope/world position for two videos at the same timestamp, distinguished only by camera
  embedding"). The **code wins** — RoPE is positional. His operational conclusion still holds
  (camera embedding distinguishes views), just not via RoPE. (**Open Q6.**)
- **Zero-init trick:** new conditioning modules start as no-ops so fine-tuning begins at
  pretrained behavior and "turns on" gradually. In `DiTBlock.__init__` (`:279–282`):
  `cam_encoder` weights/bias zeroed, `projector = torch.eye(dim)`, zero bias. **Template for
  ANY new module.** **[MEETING 4:58–5:57]** Hyunwoo's variant: an optional intra-video
  attention layer (attends only within one video) initialized with **zero layer-scale** (a
  multiplier on the output), so it gradually turns on — "but I don't think it'll be necessary;
  you don't want to distinguish which video is which when rendering the same world."
- **`__call__` is `@torch.no_grad()` INFERENCE** (~50 scheduler steps from noise, `:727` loop,
  CFG = two forward passes `nega + cfg·(posi−nega)` `:741`). The training loop is written
  separately, WITH gradients, single forward per step.

### Decision 0 — who conditions on what (resolves a derivation-vs-code tension) [CODE + reasoning]
The derivation wants `p(v2|v0,v1)` (two conditioning videos), but the **released SPT conditions
a target on exactly one `source_video`** (`:686`). Resolution that keeps the method buildable:
- **Teacher = the FROZEN released 1-source SPT, applied to pairs.** A teacher arrow is
  `∇_{v2} log p(v2|v1)` (source = `v1`, target = `v2`) — stays in-distribution. `v0` enters via
  *which middles* `v1` are drawn (from `p(v1|v0)`) **and** the crossing constraint (each `v1`
  shares a (camera,world-time) point with `v2`, making it a sufficient local summary of `v0`).
  Gap to the ideal 2-source `p(v2|v0,v1)` = a named approximation (sits with Q1).
- **Student `G_θ` = the N-source-extended SPT (trainable)**, conditioning on `(v0,v1)`. Mildly
  OOD at first; the DMD loss fine-tunes it in (see §9 OOD reassurance). **Fake-score net** = a
  second extended SPT mirroring the student.
- This is exactly [HANDOFF/MEETING]'s "teacher caps at 2, student generates 3+, D4 pairwise."

### The three N-video edit sites (Rung 4; NOT the first task) [HANDOFF, verified CODE]
1. **Fusion** `:730`: `torch.cat([latents, source_latents], dim=2)` → cat N sources in
   canonical order; encode N sources.
2. **The two DiTBlock concats** `:313` (time) and `:321` (camera): cat N entries, same order.
3. **The two model_fn dicts** `:792–796`: `{tgt, src}` → N entries.
Everything else (attention, RoPE, patchify, FFN, VAE, downsampler) **adapts automatically** —
written against an arbitrary sequence length. No new modules required (see §9).

---

## 9. Will putting N videos in even work? (the OOD question) [MEETING 6:19–11:10]

Akshata's worry: feeding 2+ videos is out-of-distribution; the model was never trained for it,
so it'll produce garbage. Hyunwoo's answer (with an on-screen training video):
- **Initially yes** — outputs look like a blurry overlap of two videos. **But minimal
  fine-tuning clears it fast:** ~500 steps already much better, ~5000 steps clean.
  "Transformers are really good at this kind of extrapolation."
- **Precedent (Evan):** needed a multi-view video model without multi-view data → split a
  single video into **4 quadrants**, placed 4 multiview videos there, fine-tuned **~6000
  iterations, batch size ~8** — successfully generalized. "Video models really understand the
  world; fine-tuning is quicker than you'd worry about."
- **Caveat — length extrapolation is the hard part.** Models overfit to the **81-frame**
  training length. Self-Forcing & autoregressive video models did real engineering to extend
  length — e.g. using the **first frame as an attention sink** so patches that don't want to
  attend dump "garbage attention" there. *Changing the number of videos* should be fine;
  *changing the length* may need this kind of engineering. "Try it; if it doesn't work, we'll
  find a better solution."

**Takeaway:** the N-source architecture extension (§8 edit sites) needs **no new modules** —
the existing zero-init pattern + brief fine-tuning is expected to absorb the OOD-ness.

---

## 10. Where the training signal comes from (no GT data) [MEETING 14:56–15:31]

Akshata's confusion: when fine-tuning, what is the ground-truth target for `v2`? Hyunwoo:
**"We don't need ground-truth data. The source of truth is the ODE-initialization solution."**
Start from the baseline model, integrate the ODE/sampler to get the paired map from initial
noise → final deterministic denoised video. **That pair is what you distill from** — no
multi-view GT. This is the practical basis for **[PLAN]**'s "a training example = one real
`v0` + sampled actions, and the student *generates* `v2`; no GT `v2`," and for the **middle
bank** (pre-generate middles `v1` offline since they don't depend on student params).

---

## 11. File map [CODE — corrected paths]

The code is under `spacetimepilot/model/` and `spacetimepilot/wan/` (the **[HANDOFF]** said
`pipelines/` — that was drift; the real package is `model/`).
- `spacetimepilot/model/spacetimepilot.py` — **the model we use** (`'spacetimepilot_1dconv'`).
  Pipeline `__call__` `:629` (`@torch.no_grad()`), DiTBlock conditioning `:259`+, fusion `:730`,
  `model_fn_wan_video` `:758`, `WanModel.forward` `:210` (with `use_gradient_checkpointing`).
  Adds `frame_time_embedding` MLP (`:267`) + `TemporalDownsampler` (81→21, `:272`), splits
  camera into tgt/src. `__call__` takes `target_camera`, `source_camera`,
  `src_time_embedding`, `tgt_time_embedding`.
- `spacetimepilot/wan/schedulers/flow_match.py` — `FlowMatchScheduler`: `add_noise` (`:62`,
  `(1−σ)x+σε`), `training_target` (`:71`, `noise−sample` = velocity), `step` (`:40`).
  **Authority for the §7 conversion.**
- `spacetimepilot/wan/models/wan_video_vae.py` — VAE; `tiled_encode` (81→21, 16ch, 8×),
  `tiled_decode` (`out_T = T·4 − 3`).
- `spacetimepilot/pipelines/recammaster.py` / `base.py` (per HANDOFF) — ReCamMaster baseline
  (camera-only) and shared blocks (`RMSNorm`, `rope_apply`, attention dispatcher, `TeaCache`).
- Entry points: `single_video_test.py` (one inference; loaders
  `load_frames_using_imageio`, `load_src_camera`, `process_camera_trajectory`,
  `compute_pose_embedding`; cameras flattened 3×4 = 12 numbers/frame, downsampled `::4`; time
  at 81-frame granularity), `inference_batch.py`. Config = **1.3B** (`dim=1536*2`,
  `num_layers=30`).

---

## 12. Environment & repo facts [CODE]

- This fork: `github.com/AkshTi/multiview-world-model` (private). Upstream
  `ZheningHuang/SpaceTimePilot` (CVPR 2026, Apache-2.0) kept as the `upstream` remote.
- **Inference code + checkpoint released; TRAINING CODE AND DATASETS ARE NOT** → we build the
  training loop on the released model definition. CamXTime *eval* data on HF; full dataset
  request-access.
- **Cluster = MIT ORCD/SLURM. Login node has NO GPU** — all model-running work via
  `srun`/`sbatch` on an ~80GB GPU node (A100-80GB/H100). **Gradient checkpointing mandatory**
  for training (supported via `use_gradient_checkpointing` in `WanModel.forward`).
- conda env **`counterfactual`** at `/orcd/scratch/orcd/014/akshatat/conda_envs/counterfactual`
  — torch 2.4.1+cu121.
- On scratch already: **Wan2.1 base** (`Wan2.1_VAE.pth`, `diffusion_pytorch_model.safetensors`,
  `models_t5_umt5-xxl-enc-bf16.pth`) and a **ReCamMaster** ckpt, at
  `/orcd/scratch/orcd/014/akshatat/counterfactual_models/`. **SPT checkpoint NOT yet
  downloaded** → `hf download zhening/SpaceTimePilot SpacetimePilot_1.3B_v1.ckpt` (to scratch).
- SPT inference already runs (Rung 0 done in a separate worktree).

---

## 13. Implementation ladder (see `PLAN.md` for the executable detail) [HANDOFF + PLAN]

0. **Rung 0** — make SPT run, confirm Section-8 shapes. *(done)*
1. **Rung 1** — training harness around the frozen model (gradients live, freeze/unfreeze,
   grad-checkpointing, throwaway flow-matching MSE just to exercise the graph — **NOT the
   objective**).
2. **Rung A** — N-source architecture (three edit sites; no new modules).
3. **Rung 2** — DMD loop (D1 + D2): three SPT instances (student, frozen 1-source teacher on
   pairs for `s_real` via marginal averaging, online fake-score net for `s_fake`). Gate K=1
   (vanilla 2-source DMD) before K>1 (the marginal contribution).
4. **Rung 3** — stitching term (D3): strip `x=Wv`, `s_x = W·s_v`, anti-blur. Ablation must
   restore blur.
5. **Rung 4/5** — N-video + pairwise-window scaling (D4); watch drift.

**Freeze/unfreeze [HANDOFF]:** freeze the whole DiT, re-enable only `cam_encoder`, `projector`,
`frame_time_embedding`, `temporal_downsampler`, and self-attn `q/k/v/o`. AdamW over those.

---

## 14. Open questions to track (with provenance) [HANDOFF + SLACK + reasoning]

- **Q1 — D2 posterior vs prior** sampling of middles `v1` (and the related 1-source `p(v2|v1)`
  vs ideal 2-source `p(v2|v0,v1)` from Decision 0). Cheap recipe is biased; crossing constraint
  is meant to make it safe; transcript trick gives the unbiased route at the cost of a net.
  **Sharpened by [SLACK]:** score-of-diffused-joint ≠ sum-of-conditional-scores → must estimate.
- **Q2 — D3 exact conditions on `W`** (slice-then-average vs average-then-slice; high-noise /
  disjoint-frame regime).
- **Q3 — D4 transitivity / drift** — does pairwise-window consistency buy global consistency,
  and how do we *measure* drift?
- **Q4 — detach vs backprop through the middle `v1`** in the teacher path.
- **Q5 — few-step student availability** — distill one ourselves, or unroll the sampler with
  backprop-through-last-step?
- **Q6 — crossing-constraint enforcement** given RoPE is index-based — via camera embedding or
  via loss?

---

## 15. Guardrails (easy to get wrong) [HANDOFF]

- **Two unrelated "time" concepts.** The DiT's `time_embedding`/`timestep` is the **diffusion
  noise-level clock** (internal). `frame_time_embedding`/`src_time_embedding`/
  `tgt_time_embedding` is **world/animation time** (SPT's conditioning). Never conflate them.
- **Don't build a vanilla-MSE main objective.** No multi-view GT; the main objective is DMD.
  MSE only lives inside the fake-score net.
- **The `[tgt, src…]` ordering invariant** across all concat sites (fusion, both DiTBlock
  concats, both model_fn dicts).
- **Slice to the target frames** `pred[:,:,:tgt_latent_length,...]` — the source half is
  context with no learning target.
- **Source latents are clean (not noised)**; only the target gets corrupted.
- **Zero-init every new conditioning module** (or zero layer-scale) so you start at pretrained
  behavior.
- **FlowMatchScheduler convention** (velocity vs noise vs x0, and scaling) is read from
  `flow_match.py`, not assumed (§7).
- **Gradient checkpointing ON** for any training; else OOM at these sequence lengths.
- **`enable_vram_management`** was built for inference offload — verify it cooperates with
  backprop before relying on it during training.
