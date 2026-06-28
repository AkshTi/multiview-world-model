# Implementation Explainer: What We Are Actually Building

This document is a plain-English guide to the project. It is separate from `PLAN.md`.

Use it when the plan starts to feel like a pile of symbols. The goal here is to explain:

- what problem we are solving;
- what the model already does;
- what DMD is doing;
- why there are three SPT models in training;
- what the confusing open questions mean;
- what you actually need to implement, in order.

`RESEARCH_CONTEXT.md` is the detailed source-of-truth context. `PLAN.md` is the execution ladder.
This file is the bridge between them.

## 0. If You Only Read One Section

You are not implementing "a new video model" all at once.

You are implementing a **training scaffold** around released SpaceTimePilot, then slowly replacing
smoke-test losses with the real DMD objective.

The immediate work is:

```text
1. Make SPT trainable at the latent level.
2. Extend SPT from one source video to two source videos.
3. Build a cached bank of middle videos.
4. Build a DMD training step with explicit teacher/fake-score calls.
```

The concrete files you will touch first are:

```text
spacetimepilot/training/                 new package for training code
spacetimepilot/training/train_step.py    new latent-level training step
spacetimepilot/training/data.py          new toy/eval dataloader helpers
spacetimepilot/training/score.py         new score conversion + score providers
spacetimepilot/model/spacetimepilot.py   edit only when adding N-source support
tests/ or scripts/                       small shape/sign/gradient checks
```

The concrete file you should **not** build training by calling:

```text
spacetimepilot/model/spacetimepilot.py::__call__
```

because it is inference-only and decorated with `@torch.no_grad()`.

The first real deliverable should be boring:

```text
A one-source training smoke test that:
  - VAE-encodes source and target videos under no_grad;
  - noises only the target latent;
  - calls dit.forward(..., use_gradient_checkpointing=True);
  - computes throwaway flow-matching MSE;
  - proves gradients appear only on the intended trainable modules.
```

That smoke test is not the research objective. It only proves the training graph works.

## 0.1 Implementation Map By File

### `spacetimepilot/model/spacetimepilot.py`

This is the only SPT model file you should edit for this project.

Use it for:

- adding N-source support;
- preserving one-source teacher behavior;
- optionally adding a checkpointed training-specific model helper if `dit.forward(...)` is not
  enough for CFG/score calls.

Important locations:

```text
WanModel.forward(...)                         line ~202
  - has use_gradient_checkpointing
  - use this path for training

DiTBlock.forward(...)                         line ~294
  - time concat currently [target, source]
  - camera concat currently [target, source]
  - must become [target, source0, source1, ...]

WanVideoReCamMasterPipeline.__call__(...)     line ~629
  - inference only
  - do not use for training

latents_input = torch.cat(...)                line ~730
  - latent concat currently [target, source]

model_fn_wan_video(...)                       line ~758
  - inference helper
  - does not checkpoint blocks
```

Do not accidentally edit the lookalike definitions in:

```text
spacetimepilot/model/base.py
spacetimepilot/model/recammaster.py
```

Those are not the SPT path for `spacetimepilot_1dconv`.

### `spacetimepilot/wan/schedulers/flow_match.py`

Use this for the noising math and DMD sign.

Important functions:

```text
add_noise(original_samples, noise, timestep)
  x_t = (1 - sigma) * x0 + sigma * eps

training_target(sample, noise, timestep)
  target = eps - x0

step(model_output, timestep, sample)
  sampler update for generation
```

You should not guess the score conversion from memory. Use this file.

### `single_video_test.py`

Use this as a reference for:

- video loading/preprocessing;
- camera trajectory helpers;
- source camera loading;
- world-time patterns;
- checkpoint loading;
- prompt encoding workflow.

Do not copy the full inference flow for training, because it eventually calls inference-only
pipeline `__call__`.

### New `spacetimepilot/training/` package

This is where most new code should go.

Suggested files:

```text
spacetimepilot/training/__init__.py
spacetimepilot/training/batch.py
spacetimepilot/training/latents.py
spacetimepilot/training/freeze.py
spacetimepilot/training/score.py
spacetimepilot/training/steps.py
spacetimepilot/training/middle_bank.py
```

What each file should do:

```text
batch.py
  load tiny training batches from videos/cameras/time/prompt

latents.py
  encode videos under no_grad
  cache/reuse source latents
  build [target, source0, source1, ...] latent tensors

freeze.py
  freeze all DiT params
  unfreeze cam_encoder/projector/frame_time_embedding/temporal_downsampler/self_attn qkv/o
  assert gradient masks

score.py
  velocity_to_score(...)
  dmd_vector_from_velocities(...)
  ScoreProvider with marginal_teacher/direct_teacher modes

steps.py
  one_source_smoke_step(...)
  two_source_smoke_step(...)
  dmd_step_k1(...)
  fake_score_update(...)

middle_bank.py
  offline generation metadata
  crossing validation
  cached latent/video loading
```

You do not need perfect file names, but you do need this separation of concerns.

## 0.2 The Implementation Ladder As Task Cards

### Task 1: One-source training smoke test

Purpose:

```text
Prove gradients can flow through SPT's DiT outside of inference.
```

Build in:

```text
spacetimepilot/training/steps.py
spacetimepilot/training/latents.py
spacetimepilot/training/freeze.py
```

Use:

```text
spacetimepilot/model/spacetimepilot.py::WanModel.forward
spacetimepilot/wan/schedulers/flow_match.py::add_noise
spacetimepilot/wan/schedulers/flow_match.py::training_target
```

Do not use:

```text
WanVideoReCamMasterPipeline.__call__
```

Pseudocode:

```python
with torch.no_grad():
    source_latent = vae.encode(source_video)
    target_latent = vae.encode(target_video)
    prompt_emb = text_encoder(prompt)

timestep = sample_scheduler_timestep()
noise = torch.randn_like(target_latent)
target_noised = scheduler.add_noise(target_latent, noise, timestep)

latents_input = torch.cat([target_noised, source_latent], dim=2)

dit.train()
pred = dit(
    latents_input,
    timestep=timestep,
    cam_emb={"tgt": target_camera, "src": source_camera},
    frame_time_embedding={
        "time_embedding_tgt": target_time,
        "time_embedding_src": source_time,
    },
    context=prompt_emb,
    use_gradient_checkpointing=True,
)

pred_target = pred[:, :, :target_latent.shape[2]]
target = scheduler.training_target(target_latent, noise, timestep)
loss = mse(pred_target, target)
loss.backward()
assert_grad_mask()
```

Done when:

- loss is finite;
- checkpointing path is actually active;
- gradients are nonzero only where expected;
- no VAE/text encoder gradients exist;
- memory fits.

### Task 2: N-source-compatible model path

Purpose:

```text
Let student/fake-score condition on [v0, v1] while teacher still works with one source.
```

Edit:

```text
spacetimepilot/model/spacetimepilot.py
```

Change these concepts:

```text
source_latents -> source_latents_list
src_camera_emb -> source_camera_embs list/dict
src_time_embedding -> source_time_embeddings list/dict
```

Keep order fixed:

```text
[target, source0, source1, ...]
```

Hard requirement:

```text
If source list length is 1, output must match old one-source path.
```

Done when:

- one-source path still works;
- 3-video latent input has 63 latent frames;
- time/camera embeddings also cover 63 latent frames after downsampling;
- target output slice is still first 21 frames;
- no silent reorder is possible.

### Task 3: Two-source training smoke test

Purpose:

```text
Prove the extended student/fake architecture can train before adding DMD.
```

Build in:

```text
spacetimepilot/training/steps.py
```

Use fake/simple data:

```text
source0 = real v0
source1 = duplicate of v0 or cached v1
target = any real target for graph testing
```

Still use throwaway flow-matching MSE. Still not the research objective.

Done when:

- same checks as Task 1 pass;
- one-source compatibility still passes;
- memory is acceptable.

### Task 4: Middle bank generation and validation

Purpose:

```text
Pre-generate middle videos v1 so DMD training does not run 50-step SPT K times every step.
```

Build in:

```text
spacetimepilot/training/middle_bank.py
scripts/generate_middle_bank.py
```

Use released SPT inference for this offline step.

Cache:

```text
v0 id
v1 video path and/or latent path
prompt or prompt embedding
source/target camera embeddings
source/target time embeddings
crossing frame index
seed / initial noise if needed
```

Validation:

```text
At crossing frame:
  camera embedding mismatch should be small
  world-time mismatch should be zero/small
```

Done when:

- you can sample K middles for one `v0`;
- metadata is enough to reproduce teacher/student score calls;
- crossing validation prints a clear pass/fail.

### Task 5: Score conversion tests

Purpose:

```text
Prevent a silent sign/scale bug.
```

Build in:

```text
spacetimepilot/training/score.py
tests/test_score_conversion.py or scripts/check_score_conversion.py
```

Test with known tensors:

```text
x_t = (1 - sigma) * x0 + sigma * eps
v = eps - x0
score = -(x_t + (1 - sigma) * v) / sigma
```

Also test DMD sign:

```text
student loss uses (v_real - v_fake)
gradient descent should move x0_hat along s_real - s_fake
```

Done when:

- score conversion exactly matches analytic score;
- flipped sign test fails as expected.

### Task 6: K=1 DMD training step

Purpose:

```text
First real objective.
```

Build in:

```text
spacetimepilot/training/steps.py
spacetimepilot/training/score.py
```

Inputs:

```text
v0 latent
one cached v1 latent
target action a2
prompt embedding
initial noise z
```

Outputs:

```text
loss_student
loss_fake
metrics
```

Start with:

```text
K = 1
cfg_scale = 1
no stitching
known-working sampler length
last-step gradient only
```

Done when:

- DMD vector norm is finite;
- fake-score loss decreases on cached student outputs;
- student does not explode/collapse immediately;
- Q1 mode is written into logs/checkpoints.

### Task 7: K>1 and stitching

Only after Task 6 works.

K>1:

```text
average teacher arrows over cached middles
track teacher-arrow variance
compare random middles vs crossing-constrained middles
```

Stitching:

```text
select one frame per view
apply disjoint W slice
reuse score slice W s_v
add with small weight
```

Done when:

- K>1 improves consistency or gives useful diagnostics;
- stitching improves sharpness/strip quality without destroying per-view quality.

## 1. The Big Picture

SpaceTimePilot takes one real video and re-renders it from a new camera path and a new world-time
path.

For example:

- input `v0`: a real video of a scene;
- instruction `a1`: "show the same scene from camera path 1, maybe frozen at time 40";
- output `v1`: an invented view of that scene.

The problem is that if you ask SPT for multiple invented views separately:

```text
v1 = SPT(v0, action a1)
v2 = SPT(v0, action a2)
v3 = SPT(v0, action a3)
```

each output may look plausible alone, but they may not describe the same hidden 3D world. A chair
could be in one place in `v1`, another place in `v2`, or a person's pose could flicker between
views.

The project goal is:

```text
Make all generated views behave like renders of one shared world.
```

The hard part:

```text
We do not have ground-truth multi-view training data.
```

So this is not a normal supervised fine-tune where we corrupt a target video and train the model to
recover the real target. For the key multi-view objective, no real target exists.

Instead, we use a distillation method called DMD.

## 2. What SPT Already Gives Us

Released SPT can do one-source generation:

```text
source video v0 + target camera/time action a1 -> generated video v1
```

In the code, released SPT has exactly one source video:

```python
source_latents = self.encode_video(source_video, **tiler_kwargs)
latents_input = torch.cat([latents, source_latents], dim=2)
```

That means the model input is:

```text
[target latent being denoised, source latent]
```

The concat is along the latent frame dimension.

Important shapes for 81-frame, 480x832 videos:

```text
video frames:      (B, 3, 81, 480, 832)
VAE latent:        (B, 16, 21, 60, 104)
2-video latent:    (B, 16, 42, 60, 104)
3-video latent:    (B, 16, 63, 60, 104)
patch tokens:      latent_frames * 30 * 52
```

So a released SPT teacher call works on two latent videos total:

```text
[target, source]
```

To train a student that can condition on two sources, we need an extended input:

```text
[target, source0=v0, source1=v1]
```

That is the first architectural change.

## 3. What We Want The Student To Learn

The student should learn:

```text
q_theta(v2 | v0, v1)
```

Meaning:

```text
Generate a new view v2, conditioned on both the original real video v0 and one already-generated
middle view v1.
```

This is "autoregression over videos":

```text
start with v0
generate v1
then generate v2 using v0 and v1
then generate v3 using previous views
...
```

The model is not autoregressing over frames. It is autoregressing over generated videos/views.

Why include `v1` at all?

Because `v1` is another sample from the hidden world implied by `v0`. If `v1` is forced to cross
or overlap with the target view at a known camera/time point, it can act as a local anchor for the
world.

## 4. Why We Need DMD

Normally, if we had real multi-view data, we could train:

```text
student(v0, v1, action a2) -> real ground-truth v2
```

But we do not have real `v2`.

Instead, DMD gives us a way to train a generator using score directions.

A score is:

```text
s(x) = grad_x log p(x)
```

Plain English:

```text
The score is the direction to nudge a sample so it becomes more likely under a model.
```

Diffusion/flow models do not give us the probability density directly, but they do give us a
vector field that can be converted into this score.

DMD says:

```text
To move the student distribution q toward the teacher distribution p,
move generated samples in the direction:

s_real - s_fake
```

where:

- `s_real` is the teacher score;
- `s_fake` is the student's own score.

The teacher tells us:

```text
which direction makes this generated video more realistic / more teacher-like?
```

The fake-score net tells us:

```text
which direction is just making the student pile up / blur / collapse?
```

So the DMD update uses the difference:

```text
s_real - s_fake
```

That is the "arrow" we push the student sample along.

## 5. Why There Are Three SPT Models

The DMD training loop uses three SPT instances.

### 5.1 Student

The student is trainable.

It generates:

```text
x0_hat = G_theta(z | v0, v1, action a2)
```

This is the generated target view `v2`.

It is the model we ultimately want.

### 5.2 Frozen Teacher

The teacher is the released SPT checkpoint, frozen.

It gives the real/teacher score `s_real`.

The teacher stays one-source because released SPT was trained one-source. Feeding it two sources
would be out-of-distribution. So teacher calls look like:

```text
teacher(target=x_t, source=v1)
```

or, under the direct-teacher interpretation:

```text
teacher(target=x_t, source=v0)
```

The Q1 question decides which of those is conceptually correct.

### 5.3 Fake-Score Net

The fake-score net estimates `s_fake`, the score of the student distribution.

The student generator can sample videos, but it does not tell us its own density score. DMD needs
that score. So we train a second model online to approximate it.

The fake-score net is trained using normal flow-matching MSE on the student's generated outputs.

Important:

```text
This is the only place plain MSE belongs in the actual DMD method.
```

The main student objective is not MSE-to-ground-truth.

## 6. The Flow-Matching Conversion

SPT does flow matching. The scheduler says:

```text
x_t = (1 - sigma) * x0 + sigma * eps
target velocity v = eps - x0
```

The model predicts velocity `v_pred`.

From the two equations:

```text
x0  = x_t - sigma * v_pred
eps = x_t + (1 - sigma) * v_pred
```

The score is:

```text
s(x_t) = -eps / sigma
       = -(x_t + (1 - sigma) * v_pred) / sigma
```

When we subtract fake score from real score, the `x_t` terms cancel:

```text
s_real - s_fake = (1 - sigma) / sigma * (v_fake - v_real)
```

DMD gradient descent needs the surrogate for:

```text
(s_fake - s_real) dot x0_hat
```

So the student loss should use:

```text
loss_student = mean(stop_grad((1 - sigma) / sigma * (v_real - v_fake)) * x0_hat)
```

This sign matters. If the sign is flipped, training pushes away from the teacher.

## 7. The Most Confusing Part: Q1

There are two possible interpretations in the research context.

This is the biggest conceptual fork in the project.

### 7.1 Marginal-Teacher Version

This is the version the original handoff/plan was building.

The teacher score is an average over middle views:

```text
s_real = average over k of teacher_score(target=v2, source=v1_k)
```

Here:

```text
v1_k ~ p(v1 | v0)
```

So we pre-generate a bank of middle videos `v1_1, ..., v1_K` from `v0`.

Then, for the current generated target `v2`, we ask:

```text
Which direction would each middle view v1_k push v2?
```

Average those arrows.

Problem:

The mathematically exact average should be over:

```text
p(v1 | v2, v0)
```

That is posterior over middles given the generated/noised target. But the cheap implementation
samples middles from:

```text
p(v1 | v0)
```

That is a prior over middles.

So the cheap version is biased. The crossing constraint is meant to make this less bad.

### 7.2 Direct-Teacher Version

Hyunwoo's Slack explanation may imply a different setup.

The teacher distribution you directly have is:

```text
p(v2 | v0)
```

So the real score is just:

```text
s_real = teacher_score(target=v2, source=v0)
```

Then the marginalization over `v1` belongs on the student/fake side:

```text
q_theta(v2, v0) = integral over v1 of q_theta(v2 | v1, v0) p(v1 | v0) p(v0)
```

In this version, the hard score is `s_fake`, because it needs to be the score of the student after
marginalizing over middles.

Problem:

A fake-score net conditioned on one sampled `v1` is not exactly this marginal score. It is only a
conditional approximation. The principled version needs the regression trick: train a model to
predict the averaged/marginalized velocity when `v1` is hidden.

### 7.3 What To Do About Q1 In Code

Do not hard-code one interpretation deep in the training loop.

Implement a `score_provider` abstraction with two modes:

```text
marginal_teacher:
    s_real = average teacher(target=x_t, source=v1_k)
    s_fake = fake_score(target=x_t, sources=(v0, v1))

direct_teacher:
    s_real = teacher(target=x_t, source=v0)
    s_fake = marginalized-student score estimate
```

At first, you can implement the marginal-teacher mode because it matches the handoff plan. But the
code should make the Q1 choice explicit in config/logs/checkpoints.

## 8. The Crossing Constraint

The crossing constraint says:

```text
Middle view v1 and target view v2 should share a known camera/world-time point.
```

Why?

Because if `v1` and `v2` intersect at a known point, then `v1` gives useful information about the
same world `v2` should render.

Important code fact:

RoPE in SPT is not world-coordinate-based. It is sequence-index-based:

```text
precompute_freqs_cis uses torch.arange(...)
```

So the model does not automatically know that two patches in different videos correspond to the
same 3D world point through RoPE.

World/camera information enters through camera embeddings and time embeddings.

Therefore, crossing must be enforced in the camera/time conditioning that the model actually sees.
That means checking equality after SPT's pose preprocessing, not just checking raw camera JSON.

Practical requirement:

```text
When creating the middle bank, store the normalized camera/time embeddings and validate that the
crossing frame matches in embedding space.
```

## 9. What Rung A Actually Changes

Released SPT has one source:

```text
[target, source]
```

Student/fake-score need two sources:

```text
[target, source0, source1]
```

So Rung A changes four places in `spacetimepilot/model/spacetimepilot.py`.

### 9.1 Latent concat

Current:

```python
latents_input = torch.cat([latents, source_latents], dim=2)
```

Needed:

```python
latents_input = torch.cat([target_latents, source0_latents, source1_latents, ...], dim=2)
```

### 9.2 Time embedding concat

Current:

```python
frame_time_embedding = torch.cat([tgt_time_embedding, src_time_embedding], dim=1)
```

Needed:

```python
frame_time_embedding = torch.cat(
    [tgt_time_embedding, src0_time_embedding, src1_time_embedding, ...],
    dim=1,
)
```

### 9.3 Camera embedding concat

Current:

```python
cam_emb = torch.cat([cam_emb_tgt, cam_emb_src], dim=1)
```

Needed:

```python
cam_emb = torch.cat([cam_emb_tgt, cam_emb_src0, cam_emb_src1, ...], dim=1)
```

### 9.4 Model function glue

Current:

```python
frame_time_embedding = {
    "time_embedding_src": src_time_embedding,
    "time_embedding_tgt": tgt_time_embedding,
}
camera_emb = {"tgt": tgt_camera_emb, "src": src_camera_emb}
```

Needed:

Use ordered source lists or explicit keys. The order must always be:

```text
[target, source0, source1, ...]
```

If latents, cameras, and times are not in the same order, the model will silently condition the
wrong video tokens on the wrong camera/time path.

## 10. Why Rung A Must Preserve One-Source Behavior

The frozen teacher must remain the released one-source SPT.

If you edit `DiTBlock` in place to support N sources, that same class is used by:

- the student;
- the fake-score net;
- the frozen teacher.

Therefore, the N-source implementation must be backward-compatible with one source.

Hard gate:

```text
With exactly one source, the new N-source code should match the old one-source code numerically.
```

Otherwise teacher scores change, and the whole DMD objective moves under your feet.

## 11. What Not To Call During Training

Do not call the released pipeline `__call__` for student training.

Why?

`__call__` is decorated with:

```python
@torch.no_grad()
```

So gradients are dead.

Also, `__call__` accepts videos and encodes them internally. That is fine for inference, but bad
for training. Training should work mostly at the latent level:

```text
video -> VAE latent, under no_grad, cached if possible
latent concat
DiT forward with gradients
scheduler step
loss on target latent frames
```

The training harness should implement explicit latent-level helpers instead of reusing `__call__`.

## 12. The First Thing To Implement

Do not start with the full DMD loop.

Start with a one-source training smoke test.

Goal:

```text
Can we run one gradient-carrying DiT forward on SPT latents and get gradients only where expected?
```

Implementation:

1. Load one source video and one target video.
2. VAE-encode both under `torch.no_grad()`.
3. Sample a timestep from the scheduler.
4. Noise only the target latent.
5. Concatenate:

```text
[noised target, clean source]
```

6. Run `dit.forward(...)`, not pipeline `__call__`.
7. Slice output to target frames.
8. Compare to `training_target(target_latent, noise, timestep)` with MSE.
9. Backprop.
10. Assert gradients exist only on unfrozen modules.

This MSE is just a graph test. It is not the final objective.

## 13. The Second Thing To Implement

After one-source training works, implement the N-source extension.

Then run the same training smoke test with:

```text
[noised target, clean source0, clean source1]
```

Do not care about quality yet.

Care about:

- shapes are correct;
- output target slice is correct;
- gradients flow;
- one-source compatibility is preserved;
- memory does not explode immediately.

## 14. The Third Thing To Implement

Generate the middle bank.

For each real `v0`:

1. Sample K middle actions `a1_1, ..., a1_K`.
2. Make sure each middle action crosses the target action `a2` at a known camera/world-time point.
3. Use frozen released SPT to generate `v1_k`.
4. Cache:

```text
v1 video or latent
camera/time embeddings
prompt embedding or prompt text
seed / initial noise if needed
crossing metadata
```

Validate:

```text
At the crossing frame, camera/time embeddings match according to SPT's preprocessing.
```

## 15. The Fourth Thing To Implement

Only after the above gates, implement K=1 DMD.

K=1 means:

```text
Use one middle v1.
Do not average over multiple middles yet.
Do not add stitching yet.
Do not generate N views yet.
```

The training step is:

1. Load `v0`, cached `v1`, and target action `a2`.
2. Student generates `x0_hat`.
3. Noise `x0_hat` to `x_t`.
4. Teacher computes `v_real` on `x_t`.
5. Fake-score net computes `v_fake` on `x_t`.
6. Student loss:

```text
mean(stop_grad((1 - sigma) / sigma * (v_real - v_fake)) * x0_hat)
```

7. Separately train fake-score net with flow-matching MSE on detached `x0_hat`.

Start with:

```text
cfg_scale = 1
no stitching
K = 1
known-working number of sampler steps
last-step gradient only
```

## 16. The Fifth Thing To Implement

If K=1 DMD is stable, try K>1.

In marginal-teacher mode:

```text
for each v1_k:
    teacher gives an arrow for x_t
average the arrows
```

Track:

- variance across teacher arrows;
- whether K>1 improves consistency;
- whether crossing-constrained middles help more than random middles.

If this does not help, the marginal-teacher approximation may be wrong or too biased.

## 17. Stitching Comes Later

Stitching is the anti-blur term.

The failure mode is:

```text
All generated views become the same gray blurry thing.
```

They are consistent, but useless.

Stitching says:

```text
Take one frame from each generated view at the same world-time.
Make a strip.
Require that strip to look like a real video/frame sequence.
```

Mathematically:

```text
x = Wv
```

If `W` is a clean disjoint selection, then:

```text
s_x = W s_v
```

So we can reuse the joint score by slicing.

But do not implement this until DMD without stitching is stable.

## 18. Scaling To More Views Comes Last

Once two-source DMD plus stitching works, scale to N views.

The idea:

```text
Generate many views.
Apply the DMD/stitching losses to overlapping windows.
```

Example:

```text
(v1, v2, v3)
(v2, v3, v4)
(v3, v4, v5)
```

This is not a theorem. It is an empirical bet that local overlap forces global consistency.

The failure mode is drift:

```text
neighboring views agree, but far-apart views slowly become inconsistent.
```

So measure consistency as a function of chain distance.

## 19. Concrete Code Hazards

### 19.1 There are duplicate model definitions

There are lookalike functions/classes in:

- `spacetimepilot/model/base.py`;
- `spacetimepilot/model/recammaster.py`;
- `spacetimepilot/model/spacetimepilot.py`.

For this project, edit the SPT file:

```text
spacetimepilot/model/spacetimepilot.py
```

Do not accidentally patch the baseline ReCamMaster or base helper.

### 19.2 `model_fn_wan_video` does not checkpoint

`model_fn_wan_video` manually loops over blocks.

`WanModel.forward` supports:

```python
use_gradient_checkpointing=True
```

For training, call `dit.forward(...)` or write a checkpointed training model function.

### 19.3 Checkpointing only happens in train mode

The code checks:

```python
if self.training and use_gradient_checkpointing:
```

So for student/fake DiTs:

```python
dit.train()
```

Then freeze unwanted parameters with `requires_grad_(False)`.

Teacher can stay:

```python
teacher.eval()
```

### 19.4 VRAM management is inference-only until proven otherwise

`enable_vram_management` wraps modules and may copy modules during forward.

That can break gradients.

Do not use it in training unless a small backprop test proves gradients reach the original
parameters.

### 19.5 Fake-score net is not actually small

If it mirrors SPT, it is another 1.3B model.

The plan may say "small score net" because DMD papers often use smaller score networks. Here, the
safest first implementation is a full SPT-shaped fake-score net with most modules frozen/unfrozen
like the student.

Later, you can consider LoRA/adapters to reduce memory. That is a separate engineering decision.

## 20. Minimal Implementation Order

If you forget everything else, implement in this order:

1. Verify released inference and shapes.
2. Write unit tests for flow score conversion and DMD sign.
3. Build one-source latent-level training smoke test.
4. Make checkpointing actually happen.
5. Verify freeze/unfreeze gradients.
6. Implement N-source support while preserving one-source behavior exactly.
7. Run two-source training smoke test.
8. Generate and validate middle bank.
9. Implement explicit `score_provider` with Q1 mode in config/logs.
10. Run K=1 DMD, no CFG, no stitching.
11. Add K>1 only if K=1 works.
12. Add stitching only if DMD works.
13. Add N-view windows only if two-source stitching works.

## 21. The Short Version

You are building a training system around released SPT.

First, make SPT trainable at the latent level.

Second, extend it from:

```text
[target, source]
```

to:

```text
[target, source0, source1]
```

Third, train a student with DMD:

```text
student update = teacher score - fake/student score
```

Fourth, use cached middle views so teacher scores can enforce consistency without multi-view
ground truth.

Fifth, later add stitching to prevent blur and pairwise windows to scale beyond two generated
views.

The two most important things not to mess up:

```text
1. The DMD sign.
2. The Q1 choice: where the marginalization lives.
```

Everything else is engineering, memory, and careful shape management.
