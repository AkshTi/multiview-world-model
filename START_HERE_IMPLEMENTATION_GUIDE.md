# Start Here: The Plain-English Implementation Guide

This is the simplest version of the project.

If `PLAN.md` and `IMPLEMENTATION_EXPLAINER.md` feel too abstract, start here.

This guide explains:

- what problem we are solving;
- why the method makes sense;
- what the code already does;
- what new code needs to be written;
- what each piece is for;
- what to implement first.

I will avoid math unless it is necessary. When there is math, I will explain what it means in
words.

## 1. The Problem In One Sentence

SpaceTimePilot can generate a new view of a video, but if you generate several new views, they may
not agree about the same underlying world.

Example:

```text
Input real video:
  v0 = a person dancing in a room

Ask SPT:
  v1 = show this from camera angle 1
  v2 = show this from camera angle 2
  v3 = show this from camera angle 3
```

Each output might look good alone.

But together, they may disagree:

```text
v1 says the chair is on the left
v2 says the chair is farther back
v3 changes the person's pose slightly
```

That means the model is not really imagining one stable 3D scene. It is inventing a new plausible
scene every time.

We want:

```text
All generated videos should look like different camera views of the same hidden world.
```

## 2. What SpaceTimePilot Already Does

SpaceTimePilot already knows how to do this:

```text
given one source video, generate one new target video
```

In symbols:

```text
v1 = SPT(source video v0, target camera/time action a1)
```

In code, this happens inside:

```text
spacetimepilot/model/spacetimepilot.py
```

The key thing to know is that SPT works in **latent space**, not directly on pixels.

That means:

```text
video frames -> VAE -> smaller latent tensor -> DiT denoises latent -> VAE decodes to video
```

For an 81-frame video:

```text
regular video shape:
  (batch, 3 color channels, 81 frames, 480 height, 832 width)

latent video shape:
  (batch, 16 channels, 21 latent frames, 60 height, 104 width)
```

So when we say "video" during training, we often really mean:

```text
latent video
```

because that is what the DiT transformer sees.

## 3. The Important Code Path In Released SPT

Released SPT inference does this:

1. Encode the source video into a latent.
2. Start the target latent from noise.
3. Concatenate target and source latents.
4. Run the DiT denoiser many times.
5. Decode the final target latent into video frames.

The key code is:

```python
source_latents = self.encode_video(source_video, **tiler_kwargs)
latents_input = torch.cat([latents, source_latents], dim=2)
```

This means the model sees:

```text
[target being generated, source video]
```

The concat dimension is the latent-frame dimension.

So if one latent video has 21 latent frames:

```text
target: 21 latent frames
source: 21 latent frames
combined: 42 latent frames
```

## 4. What We Need To Change

We need a student model that can look at **two source videos**, not one.

Instead of:

```text
[target, source]
```

we need:

```text
[target, source0, source1]
```

where:

```text
source0 = v0, the original real video
source1 = v1, one generated middle view
target  = v2, the new view we want the student to generate
```

So the student learns:

```text
generate v2 while looking at both v0 and v1
```

Why does this help?

Because `v1` gives another clue about the same world. If `v0` and `v1` both describe the world,
then using both should make `v2` more consistent.

## 5. Why We Cannot Just Train With Normal Supervision

Normal training would need examples like:

```text
input:
  v0 and v1

ground truth:
  the real correct v2
```

But we do not have real multi-view ground-truth videos.

We only have:

```text
v0 = real video
generated v1, v2, etc. = model inventions
```

So we cannot say:

```text
student output should match this real v2
```

because there is no real `v2`.

This is why the project uses DMD.

## 6. What DMD Is Trying To Do

DMD is a way to train a generator without ground-truth examples.

Instead of saying:

```text
match this exact target
```

DMD says:

```text
move the generated output in a direction that makes it more like the teacher distribution
```

Think of it like this:

```text
The student generates an image/video.
The teacher gives an arrow saying "move it this way to look more realistic."
The fake-score model gives another arrow saying "this is where the student is already piling up."
The useful direction is teacher arrow minus fake/student arrow.
```

That direction is:

```text
s_real - s_fake
```

Where:

```text
s_real = teacher's arrow
s_fake = student's own arrow, estimated by the fake-score net
```

Plain English:

```text
s_real pulls the sample toward what the teacher likes.
s_fake pushes against collapse/blurry pile-up.
```

## 7. Why There Are Three SPT Models

The training loop has three SPT-like models.

### 7.1 Student

This is the model we actually train.

It takes:

```text
v0, v1, target camera/time action
```

and generates:

```text
v2
```

This is the model we want to improve.

### 7.2 Frozen Teacher

This is the released SPT checkpoint.

It is frozen.

It gives the "real" arrow:

```text
s_real
```

Important: the teacher should stay one-source because released SPT was trained with one source.

So the teacher should see:

```text
[target, source]
```

not:

```text
[target, source0, source1]
```

### 7.3 Fake-Score Net

DMD needs to know the student's own score/arrow.

The student can generate samples, but it does not directly tell us:

```text
what direction increases the student's own probability?
```

So we train another model, called the fake-score net, to estimate that.

This model is trained with ordinary flow-matching MSE on the student's generated videos.

Important:

```text
The fake-score net uses MSE.
The main student does not use normal MSE to ground truth.
```

## 8. The Key DMD Formula Without The Confusing Math

SPT predicts a velocity.

The scheduler says:

```text
noised_video = partly clean video + partly random noise
```

The model predicts:

```text
velocity = noise - clean_video
```

From this velocity, we can compute the teacher/fake arrows.

After simplifying, the student loss uses:

```text
v_real - v_fake
```

The important line is:

```text
loss_student = mean(stop_grad((1 - sigma) / sigma * (v_real - v_fake)) * x0_hat)
```

Where:

```text
x0_hat = the student's generated clean target latent
v_real = teacher velocity prediction
v_fake = fake-score velocity prediction
sigma  = noise level
```

The sign matters.

If you accidentally use:

```text
v_fake - v_real
```

inside the loss, gradient descent pushes in the wrong direction.

## 9. The Biggest Conceptual Fork: Q1

There are two possible ways to set up the teacher/fake scores.

This is the biggest thing that still needs Hyunwoo's confirmation.

### 9.1 Option A: Marginal-Teacher Mode

This was the original plan.

You generate several middle videos from `v0`:

```text
v1_1, v1_2, v1_3, ...
```

Then the teacher gives one arrow for each middle:

```text
teacher arrow using v1_1
teacher arrow using v1_2
teacher arrow using v1_3
```

Then you average those arrows.

In code:

```text
s_real = average teacher_score(target=x_t, source=v1_k)
```

This tries to say:

```text
The correct direction for v2 is the average opinion of many possible middle views.
```

Problem:

The ideal math wants middles sampled based on the current generated `v2`.

But the cheap version samples middles from `v0` only.

So this is approximate.

### 9.2 Option B: Direct-Teacher Mode

Hyunwoo's Slack message may imply this instead.

Here the teacher is simpler:

```text
s_real = teacher_score(target=x_t, source=v0)
```

So the teacher directly says:

```text
Given the original video v0, what direction makes v2 more likely?
```

Then the marginalization over middles belongs to the fake/student side.

Problem:

That makes `s_fake` harder. It must estimate the student's score after averaging over possible
middles.

A fake-score net conditioned on one sampled `v1` is not really enough.

### 9.3 Practical Coding Rule

Do not hide this choice deep in the training code.

Make a simple switch:

```text
score_mode = "marginal_teacher"
or
score_mode = "direct_teacher"
```

Then implement a `ScoreProvider` object that decides how to compute:

```text
v_real
v_fake
```

That way, if Hyunwoo says "use the other interpretation," you do not rewrite everything.

## 10. What Each New Code Piece Should Do

This is the part to use when coding.

### 10.1 New folder: `spacetimepilot/training/`

Most new code should live here.

Suggested files:

```text
spacetimepilot/training/__init__.py
spacetimepilot/training/latents.py
spacetimepilot/training/freeze.py
spacetimepilot/training/score.py
spacetimepilot/training/steps.py
spacetimepilot/training/middle_bank.py
```

### 10.2 `latents.py`

This file should handle anything about VAE latents.

It should have helpers like:

```python
encode_video_to_latent(pipe, video) -> latent
build_latent_input(target_latent, source_latents) -> combined_latent
slice_target(pred, target_length) -> pred_target
```

Why this file matters:

Training must happen at the latent level. Do not call pipeline `__call__` during training.

### 10.3 `freeze.py`

This file should handle trainable parameters.

It should:

1. freeze the whole DiT;
2. unfreeze only the modules we want;
3. create an optimizer from only trainable parameters;
4. assert that gradients match the intended mask after backward.

Trainable modules:

```text
cam_encoder
projector
frame_time_embedding
temporal_downsampler
self_attn.q
self_attn.k
self_attn.v
self_attn.o
```

Important:

Checkpointing only runs if:

```python
dit.train()
```

So the student and fake-score DiTs should be in train mode, even if most parameters are frozen.

The frozen teacher can stay in eval mode.

### 10.4 `score.py`

This file should contain the DMD math.

It should have:

```python
velocity_to_score(x_t, v_pred, sigma)
dmd_vector_from_velocities(v_real, v_fake, sigma)
student_dmd_loss(x0_hat, v_real, v_fake, sigma)
```

It should also contain:

```python
class ScoreProvider:
    mode: "marginal_teacher" or "direct_teacher"
```

This class should be the only place where Q1 mode changes behavior.

### 10.5 `steps.py`

This file should contain training steps.

Start with:

```python
one_source_smoke_step(...)
```

Then:

```python
two_source_smoke_step(...)
```

Then:

```python
dmd_step_k1(...)
fake_score_update(...)
```

Do not start with `dmd_step_k1`.

### 10.6 `middle_bank.py`

This file should help create and load cached middle views.

It should store:

```text
which v0 this came from
the generated v1 video or latent
camera embeddings
time embeddings
crossing frame
prompt or prompt embedding
seed / noise if needed
```

It should also validate:

```text
At the crossing frame, do v1 and target action a2 really match in the representation SPT sees?
```

## 11. What To Edit In `spacetimepilot.py`

Only edit:

```text
spacetimepilot/model/spacetimepilot.py
```

Do not edit the lookalike versions in:

```text
spacetimepilot/model/base.py
spacetimepilot/model/recammaster.py
```

The needed SPT edits are:

### 11.1 Latent concat

Current:

```python
latents_input = torch.cat([latents, source_latents], dim=2)
```

Needed:

```python
latents_input = torch.cat([target_latents] + source_latents_list, dim=2)
```

### 11.2 Time embeddings

Current:

```python
src_time_embedding = frame_time_embedding["time_embedding_src"]
tgt_time_embedding = frame_time_embedding["time_embedding_tgt"]
...
frame_time_embedding = torch.cat([tgt_time_embedding, src_time_embedding], dim=1)
```

Needed:

```python
tgt_time_embedding = ...
source_time_embeddings = [...]
frame_time_embedding = torch.cat([tgt_time_embedding] + source_time_embeddings, dim=1)
```

### 11.3 Camera embeddings

Current:

```python
cam_emb_tgt = self.cam_encoder(cam_emb["tgt"])
cam_emb_src = self.cam_encoder(cam_emb["src"])
cam_emb = torch.cat([cam_emb_tgt, cam_emb_src], dim=1)
```

Needed:

```python
cam_emb_tgt = self.cam_encoder(cam_emb["tgt"])
cam_emb_sources = [self.cam_encoder(src) for src in cam_emb["srcs"]]
cam_emb = torch.cat([cam_emb_tgt] + cam_emb_sources, dim=1)
```

### 11.4 Backward compatibility

The new code must still work when there is exactly one source.

That means this:

```text
new code with one source
```

should behave like:

```text
old released SPT code
```

Why?

Because the frozen teacher still uses one-source SPT.

If you break one-source behavior, teacher scores are no longer the released SPT scores.

## 12. First Implementation: One-Source Smoke Test

This should be the first real code you write.

Goal:

```text
Can we run SPT with gradients, outside the inference-only __call__?
```

Pseudo-code:

```python
def one_source_smoke_step(batch, pipe, scheduler):
    # 1. Encode videos to latents. No gradients through VAE.
    with torch.no_grad():
        source_latent = pipe.vae.encode(batch.source_video)
        target_latent = pipe.vae.encode(batch.target_video)
        prompt_emb = pipe.text_encoder(batch.prompt)

    # 2. Make noisy target.
    timestep = sample_timestep(scheduler)
    noise = torch.randn_like(target_latent)
    target_noised = scheduler.add_noise(target_latent, noise, timestep)

    # 3. Build released SPT input shape: [target, source].
    latents_input = torch.cat([target_noised, source_latent], dim=2)

    # 4. Run DiT with gradients and checkpointing.
    pipe.dit.train()
    pred = pipe.dit(
        latents_input,
        timestep=timestep,
        cam_emb={
            "tgt": batch.target_camera,
            "src": batch.source_camera,
        },
        frame_time_embedding={
            "time_embedding_tgt": batch.target_time,
            "time_embedding_src": batch.source_time,
        },
        context=prompt_emb,
        use_gradient_checkpointing=True,
    )

    # 5. Only target frames matter.
    pred_target = pred[:, :, :target_latent.shape[2]]

    # 6. Throwaway graph-test loss.
    target_velocity = scheduler.training_target(target_latent, noise, timestep)
    loss = mse(pred_target, target_velocity)
    loss.backward()

    return loss
```

If this does not work, do not move on.

## 13. Second Implementation: Two-Source Smoke Test

After one-source works, add N-source support.

Then run the same kind of test with:

```text
[target, source0, source1]
```

Use any simple `source1` at first:

```text
source1 = duplicate of v0
```

or:

```text
source1 = one cached generated v1
```

This is not testing quality. It is testing:

```text
Can the model run with 3 latent videos in the sequence?
Can gradients flow?
Do shapes line up?
Does memory fit?
```

## 14. Third Implementation: Middle Bank

Once the student can accept two sources, create cached middle views.

For each real video `v0`:

```text
generate v1_1
generate v1_2
generate v1_3
...
```

Each `v1_k` is generated by frozen released SPT.

Cache the results because running frozen SPT every training step would be too slow.

Also cache the camera/time metadata, because the crossing constraint depends on it.

## 15. Fourth Implementation: K=1 DMD

Now implement the first real DMD step.

Use only one middle:

```text
K = 1
```

No stitching.

No CFG.

No many-view windows.

The step is:

```text
1. Student generates x0_hat using v0 and v1.
2. Noise x0_hat to x_t.
3. Teacher predicts v_real for x_t.
4. Fake-score net predicts v_fake for x_t.
5. Student loss uses v_real - v_fake.
6. Fake-score net separately trains on detached x0_hat with flow-matching MSE.
```

Only when this is stable should you add K>1, stitching, or many views.

## 16. Motivation Recap

Why all this work?

Because we want a model that can build a world view by view.

SPT already knows:

```text
how to generate one plausible new view
```

But it does not guarantee:

```text
all generated views describe the same world
```

The student learns to use multiple views as context.

The teacher provides realism arrows.

The fake-score net prevents collapse and gives the DMD correction.

The middle bank gives multiple possible views of the world without needing real multi-view data.

The crossing constraint makes those middle views useful anchors rather than random unrelated
samples.

## 17. The Real Order Of Work

Do this:

```text
1. One-source training smoke test.
2. N-source model extension.
3. Two-source training smoke test.
4. Score conversion and DMD sign tests.
5. Middle bank generation and crossing validation.
6. K=1 DMD.
7. K>1 DMD.
8. Stitching.
9. More-than-two-view windows.
```

Do not do this:

```text
1. Full DMD loop first.
2. Stitching first.
3. N-view scaling first.
4. Training through pipeline __call__.
5. Editing base.py or recammaster.py by accident.
```

## 18. The Mental Model

The whole project can be remembered as:

```text
Released SPT:
  one source -> one generated view

Our student:
  two sources -> one generated view

Training signal:
  teacher arrow - fake/student arrow

No ground truth:
  use generated middles and DMD instead

Implementation:
  latent-level training, not pipeline inference
```

