"""Training steps (Rung 2 onward).

``one_source_smoke_step`` is the Rung 2 gate: prove SPT trains at the latent level with
gradients flowing only through the intended modules. It uses the *released* one-source
shape, so it runs against the unmodified model (no N-source edits needed yet). The MSE
here is a throwaway graph test, NOT the research objective.

NOTE: this module imports/executes the SPT DiT and therefore needs a GPU node + the SPT
checkpoint. It is not CPU/login-node runnable. The pure math it relies on
(``score``, ``latents``, ``freeze``) is unit-tested separately on CPU.

Two-source smoke (``two_source_smoke_step``) and the DMD loop (``dmd_step_k1``) are added
after the N-source model edits (Rung 3) and the middle bank (Rung 4).

A7 (7/6 AMP refactor): ``_dit_velocity`` — the shared forward used by ``dmd_step_k1`` /
``dmd_step_k`` — runs under ``torch.autocast(bf16)``. Trainable params on the student/fake
DiTs are stored fp32 (``freeze.to_fp32_trainable``); the frozen teacher stays plain bf16.
See the docstring on ``_dit_velocity`` for the mixed fp32/bf16-param hazard this can raise,
GPU-only and unverified from this login node.
"""

import torch
import torch.nn.functional as F

from . import latents as latent_utils
from . import score as score_utils


def sample_training_timestep(scheduler, batch_size=1, device="cpu", generator=None):
    """Draw a random diffusion timestep index and return (timestep, sigma, timestep_id).

    ``scheduler.set_timesteps(..., training=True)`` must have been called so
    ``scheduler.timesteps`` / ``scheduler.sigmas`` are populated. We sample a single index
    and broadcast it across the batch (matching the released inference, which uses one
    timestep per denoising call).
    """
    n = len(scheduler.timesteps)
    idx = int(torch.randint(0, n, (1,), generator=generator).item())
    timestep = scheduler.timesteps[idx].to(device).repeat(batch_size)
    sigma = scheduler.sigmas[idx].to(device)
    return timestep, sigma, idx


def one_source_smoke_step(pipe, batch, scheduler, use_gradient_checkpointing=True):
    """One latent-level training step on the released one-source shape [target, source].

    ``batch`` provides (already on the right device or CPU tensors we move):
        source_video, target_video : pixel videos for the VAE
        target_camera, source_camera : camera embeddings at 21 latent frames
        tgt_time_embedding, src_time_embedding : world-time at 81 frames
        prompt : text string

    Returns a dict with the throwaway MSE loss and the sampled sigma. Caller does
    ``loss.backward()`` then ``freeze.assert_grad_mask(pipe.dit)``.

    Gate (Rung 2): finite loss; checkpointing active (pipe.dit.train()); grads only on the
    unfrozen modules; no VAE/text-encoder grads; enable_vram_management OFF; fits 80GB.
    """
    device, dtype = pipe.device, pipe.torch_dtype

    # 1. Encode source + target and the prompt under no_grad (no VAE/text-encoder grads).
    source_latent = latent_utils.encode_video_nograd(pipe, batch["source_video"])
    target_latent = latent_utils.encode_video_nograd(pipe, batch["target_video"])
    with torch.no_grad():
        prompt_emb = pipe.encode_prompt(batch["prompt"], positive=True)  # {"context": ...}

    # 2. Noise ONLY the target latent; keep the source clean.
    B = target_latent.shape[0]
    timestep, sigma, _ = sample_training_timestep(scheduler, batch_size=B, device=device)
    noise = torch.randn_like(target_latent)
    target_noised = scheduler.add_noise(target_latent, noise, timestep)

    # 3. Build the released one-source input [target, source].
    latents_input = latent_utils.build_latent_input(target_noised, source_latent)

    # 4. Checkpointed DiT forward WITH gradients (dit.train() enables the checkpoint path).
    pipe.dit.train()
    cam_emb = {
        "tgt": batch["target_camera"].to(dtype=dtype, device=device),
        "src": batch["source_camera"].to(dtype=dtype, device=device),
    }
    frame_time_embedding = {
        "time_embedding_tgt": batch["tgt_time_embedding"].to(dtype=dtype, device=device),
        "time_embedding_src": batch["src_time_embedding"].to(dtype=dtype, device=device),
    }
    pred = pipe.dit(
        latents_input,
        timestep=timestep.to(dtype=dtype),
        cam_emb=cam_emb,
        context=prompt_emb["context"],
        frame_time_embedding=frame_time_embedding,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    # 5. Only the target frames carry a learning signal.
    pred_target = latent_utils.slice_target(pred, target_latent.shape[2])

    # 6. Throwaway flow-matching MSE (graph test, NOT the objective).
    target_velocity = scheduler.training_target(target_latent, noise, timestep)
    loss = F.mse_loss(pred_target.float(), target_velocity.float())

    return {"loss": loss, "sigma": sigma.detach(), "timestep": timestep.detach()}


def two_source_smoke_step(pipe, batch, scheduler, use_gradient_checkpointing=True):
    """N-source (N>=2) latent training step against the edited model.

    Same shape as ``one_source_smoke_step`` but the source side is a *list*: the fused
    input is [target, src0, src1, ...] and the camera / time dicts carry per-source lists,
    exercising the N-source concat in the edited DiTBlock. Gate (Rung 3): finite loss,
    grads only on the unfrozen modules, fits in VRAM.

    ``batch`` provides:
        target_video : pixel video for the VAE
        source_videos : list of pixel videos (len N)
        target_camera : (B,21,12)
        source_cameras : list of (B,21,12)
        tgt_time_embedding : (B,81)
        src_time_embeddings : list of (B,81)
        prompt : text string
    """
    device, dtype = pipe.device, pipe.torch_dtype

    target_latent = latent_utils.encode_video_nograd(pipe, batch["target_video"])
    source_latents = [latent_utils.encode_video_nograd(pipe, v) for v in batch["source_videos"]]
    with torch.no_grad():
        prompt_emb = pipe.encode_prompt(batch["prompt"], positive=True)

    B = target_latent.shape[0]
    timestep, sigma, _ = sample_training_timestep(scheduler, batch_size=B, device=device)
    noise = torch.randn_like(target_latent)
    target_noised = scheduler.add_noise(target_latent, noise, timestep)

    latents_input = latent_utils.build_latent_input(target_noised, source_latents)

    pipe.dit.train()
    cam_emb = {
        "tgt": batch["target_camera"].to(dtype=dtype, device=device),
        "src": [c.to(dtype=dtype, device=device) for c in batch["source_cameras"]],
    }
    frame_time_embedding = {
        "time_embedding_tgt": batch["tgt_time_embedding"].to(dtype=dtype, device=device),
        "time_embedding_src": [s.to(dtype=dtype, device=device) for s in batch["src_time_embeddings"]],
    }
    pred = pipe.dit(
        latents_input,
        timestep=timestep.to(dtype=dtype),
        cam_emb=cam_emb,
        context=prompt_emb["context"],
        frame_time_embedding=frame_time_embedding,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    pred_target = latent_utils.slice_target(pred, target_latent.shape[2])
    target_velocity = scheduler.training_target(target_latent, noise, timestep)
    loss = F.mse_loss(pred_target.float(), target_velocity.float())

    return {
        "loss": loss,
        "sigma": sigma.detach(),
        "timestep": timestep.detach(),
        "num_sources": len(source_latents),
        "fused_frames": latents_input.shape[2],
    }


def _dit_velocity(dit, target_noised, source_latents, tgt_cam, src_cams,
                  tgt_time, src_times, context, timestep, target_frames,
                  dtype, device, use_gradient_checkpointing):
    """One DiT forward returning velocity on the TARGET frames only.

    ``source_latents``/``src_cams``/``src_times`` are ordered lists (len 1 = the released
    one-source path, byte-identical per Rung 3; len N = student). Fusion order
    [target, src0, src1, ...] must line up across the three lists (see latents.build_latent_input).

    A7 (7/6 AMP recipe): the forward runs under ``torch.autocast(bf16)``. This is the single
    call site used by teacher/student/fake in dmd_step_k1 / dmd_step_k, so autocast lands
    everywhere uniformly — harmless for the frozen bf16 teacher (no_grad already), and what
    lets the student/fake nets compute in bf16 while their trainable params are stored fp32
    (freeze.to_fp32_trainable). Loss/score math stays fp32 via the explicit .float() calls in
    score.py, so autocast never touches the loss itself.

    KNOWN HAZARD (GPU-only, unverified — see scratchpad/amp_refactor_notes.md): student/fake
    now hold fp32 trainable + bf16 frozen params in the same module. autocast picks op dtype
    by op TYPE, not param dtype, so an elementwise/residual/norm op combining an fp32
    activation (from a fp32 trainable layer) with a bf16 one (from a frozen layer) can raise
    a dtype-mismatch error that only surfaces on a real forward. If that happens, see the
    scratchpad note for the two fallbacks (widen this autocast region vs. keep the whole
    module fp32).
    """
    latents_input = latent_utils.build_latent_input(target_noised, source_latents)
    cam_emb = {
        "tgt": tgt_cam.to(dtype=dtype, device=device),
        "src": [c.to(dtype=dtype, device=device) for c in src_cams],
    }
    frame_time_embedding = {
        "time_embedding_tgt": tgt_time.to(dtype=dtype, device=device),
        "time_embedding_src": [s.to(dtype=dtype, device=device) for s in src_times],
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = dit(
            latents_input,
            timestep=timestep.to(dtype=dtype),
            cam_emb=cam_emb,
            context=context,
            frame_time_embedding=frame_time_embedding,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    return latent_utils.slice_target(pred, target_frames)


def dmd_step_k1(student_pipe, teacher_dit, fake_dit, batch, scheduler, opt_G, opt_D,
                use_gradient_checkpointing=True, generator=None,
                after_g=None, after_d=None, offload_teacher=True):
    """One K=1 DMD iteration (Rung 5). See RUNG5_DESIGN.md for the full derivation.

    Three models: the 2-source **student** ``student_pipe.dit`` (θ, generates v2 from noise
    conditioned on v0 AND v1), the frozen 1-source **teacher** ``teacher_dit`` (s_real, v0
    only), and the online 1-source **fake-score** net ``fake_dit`` (s_fake, v0 only, v1
    hidden). One middle v1 is drawn per step (K=1).

    ``batch`` provides (pixel videos for the VAE, camera/time embeddings, prompt):
        source_video   : v0 pixels          middle_video   : v1 pixels
        target_camera  : v2 cam (B,21,12)   source_camera  : v0 cam    middle_camera : v1 cam
        tgt_time_embedding : v2 time (B,81)  src_time_embedding : v0 time  mid_time_embedding : v1 time
        prompt : text string

    The two backwards are done HERE, in order, so only one graph is ever live (48 GB L40S):
      1. build v2_hat (student one-step, grad on θ),
      2. G-step: loss_G.backward(); after_g(); opt_G.step(); opt_G.zero_grad(),
      3. D-step: fresh fake forward on v2_hat.detach(); loss_D.backward(); after_d();
         opt_D.step(); opt_D.zero_grad().
    ``after_g``/``after_d`` are optional callbacks run while grads are live (e.g. grad-mask
    asserts). Returns detached diagnostics only (no graph escapes).
    """
    pipe = student_pipe
    device, dtype = pipe.device, pipe.torch_dtype

    # --- encode conditioning (no VAE grads) ---
    v0_latent = latent_utils.encode_video_nograd(pipe, batch["source_video"])
    v1_latent = latent_utils.encode_video_nograd(pipe, batch["middle_video"])
    with torch.no_grad():
        prompt_emb = pipe.encode_prompt(batch["prompt"], positive=True)
    context = prompt_emb["context"]

    B = v0_latent.shape[0]
    target_frames = latent_utils.LATENT_FRAMES_PER_VIDEO

    tgt_cam, v0_cam, v1_cam = batch["target_camera"], batch["source_camera"], batch["middle_camera"]
    tgt_time, v0_time, v1_time = (
        batch["tgt_time_embedding"], batch["src_time_embedding"], batch["mid_time_embedding"])

    # --- generate v2_hat: student one-step at the max-sigma timestep (grad on θ) ---
    sig_idx = int(torch.argmax(scheduler.sigmas))
    t_T = scheduler.timesteps[sig_idx].to(device).repeat(B)
    sigma_T = scheduler.sigmas[sig_idx].to(device)
    z = torch.randn((B, *v0_latent.shape[1:]), generator=generator,
                    dtype=v0_latent.dtype, device=device)

    pipe.dit.train()
    v_pred = _dit_velocity(
        pipe.dit, z, [v0_latent, v1_latent], tgt_cam, [v0_cam, v1_cam],
        tgt_time, [v0_time, v1_time], context, t_T, target_frames,
        dtype, device, use_gradient_checkpointing)
    v2_hat = score_utils.x0_from_velocity(z, v_pred, sigma_T)  # carries grad on θ

    # --- G-step: distribution-matching gradient on the student ---
    opt_G.zero_grad(set_to_none=True)
    tg, sigma_g, _ = sample_training_timestep(scheduler, batch_size=B, device=device, generator=generator)
    eps_g = torch.randn_like(v2_hat)
    x_tg = scheduler.add_noise(v2_hat, eps_g, tg)
    with torch.no_grad():
        # The frozen teacher is only needed for this one no_grad forward. Keep it on CPU the
        # rest of the time so its ~2.6 GB isn't resident during the student/fake backprop
        # (three 1.3B DiTs + optimizer state otherwise OOM the 48 GB L40S).
        if offload_teacher:
            teacher_dit.to(device)
        teacher_dit.eval()
        v_real = _dit_velocity(
            teacher_dit, x_tg, [v0_latent], tgt_cam, [v0_cam],
            tgt_time, [v0_time], context, tg, target_frames,
            dtype, device, use_gradient_checkpointing=False)
        if offload_teacher:
            teacher_dit.to("cpu")
            torch.cuda.empty_cache()
        fake_dit.eval()
        v_fake = _dit_velocity(
            fake_dit, x_tg, [v0_latent], tgt_cam, [v0_cam],
            tgt_time, [v0_time], context, tg, target_frames,
            dtype, device, use_gradient_checkpointing=False)
    loss_G = score_utils.student_dmd_loss(v2_hat, v_real, v_fake, sigma_g, normalize=True)
    arrow_abs = (v_real.float() - v_fake.float()).abs().mean().item()
    loss_G_val = loss_G.item()
    loss_G.backward()
    if after_g is not None:
        after_g()
    opt_G.step()
    opt_G.zero_grad(set_to_none=True)

    # v2_hat's graph is now freed; the D-step regresses on its detached values.
    v2_detached = v2_hat.detach()
    del v2_hat, v_pred, x_tg, v_real, v_fake, loss_G

    # --- D-step: teach the fake-score net the student's current distribution ---
    opt_D.zero_grad(set_to_none=True)
    td, sigma_d, _ = sample_training_timestep(scheduler, batch_size=B, device=device, generator=generator)
    eps_d = torch.randn_like(v2_detached)
    x_td = scheduler.add_noise(v2_detached, eps_d, td)
    fake_dit.train()
    v_fake_pred = _dit_velocity(
        fake_dit, x_td, [v0_latent], tgt_cam, [v0_cam],
        tgt_time, [v0_time], context, td, target_frames,
        dtype, device, use_gradient_checkpointing)
    loss_D = score_utils.fake_score_loss(v_fake_pred, v2_detached, eps_d)
    loss_D_val = loss_D.item()
    loss_D.backward()
    if after_d is not None:
        after_d()
    opt_D.step()
    opt_D.zero_grad(set_to_none=True)

    return {
        "loss_G": loss_G_val,
        "loss_D": loss_D_val,
        "sigma_T": float(sigma_T),
        "sigma_g": float(sigma_g),
        "sigma_d": float(sigma_d),
        "arrow_abs": arrow_abs,
    }


def dmd_step_k(student_pipe, teacher_dit, fake_dit, batch, scheduler, opt_G, opt_D,
               use_gradient_checkpointing=True, generator=None,
               after_g=None, after_d=None, offload_teacher=False):
    """One K>=1 DMD iteration with Monte-Carlo marginalization over K middles (Rung 6).

    The direct-teacher branch marginalizes the student joint over the middle view:

        p'(v2|v0) = ∫ q_θ(v2|v0,v1) p(v1|v0) dv1  ≈  (1/K) Σ_k q_θ(v2|v0, v1_k),  v1_k ~ p(v1|v0).

    Each middle is an INDEPENDENT sample: draw fresh noise z_k, generate v2_hat_k = G_θ(v0,v1_k,z_k),
    and apply the two-arrow gradient at v2_hat_k. The faithful marginal update is the mean of
    the per-middle gradients, so we **accumulate** ``loss/K`` over k and step once — identical
    in expectation to K=1-per-step, lower variance, and (crucially) only one student graph is
    ever live, so peak memory is independent of K. The fake-score net sees only v0 (v1 hidden)
    across all K samples, so its denoising regression learns the marginal score directly.

    ``batch`` extends the K=1 batch: the middle fields are length-K lists
        middle_videos : [v1_0 pixels, ...]   middle_cameras : [v1_0 cam, ...]
        mid_time_embeddings : [v1_0 time, ...]
    and v0 / v2 fields (source_*, target_*) are shared across the K middles.

    Returns diagnostics: mean loss_G / loss_D over the K middles, per-middle and mean arrow.
    """
    pipe = student_pipe
    device, dtype = pipe.device, pipe.torch_dtype

    middle_videos = batch["middle_videos"]
    middle_cameras = batch["middle_cameras"]
    mid_times = batch["mid_time_embeddings"]
    K = len(middle_videos)
    if not (K == len(middle_cameras) == len(mid_times)) or K < 1:
        raise ValueError(f"middle_videos/cameras/times must be equal-length, len>=1 (got {K})")

    # --- encode shared conditioning once (no VAE grads) ---
    v0_latent = latent_utils.encode_video_nograd(pipe, batch["source_video"])
    v1_latents = [latent_utils.encode_video_nograd(pipe, v) for v in middle_videos]
    with torch.no_grad():
        prompt_emb = pipe.encode_prompt(batch["prompt"], positive=True)
    context = prompt_emb["context"]

    B = v0_latent.shape[0]
    target_frames = latent_utils.LATENT_FRAMES_PER_VIDEO
    tgt_cam, v0_cam = batch["target_camera"], batch["source_camera"]
    tgt_time, v0_time = batch["tgt_time_embedding"], batch["src_time_embedding"]

    sig_idx = int(torch.argmax(scheduler.sigmas))
    t_T = scheduler.timesteps[sig_idx].to(device).repeat(B)
    sigma_T = scheduler.sigmas[sig_idx].to(device)

    if offload_teacher:
        teacher_dit.to(device)
    teacher_dit.eval()

    # --- G-step: accumulate the distribution-matching gradient over K middles ---
    opt_G.zero_grad(set_to_none=True)
    detached_samples = []
    arrows = []
    loss_G_sum = 0.0
    for k in range(K):
        z = torch.randn((B, *v0_latent.shape[1:]), generator=generator,
                        dtype=v0_latent.dtype, device=device)
        pipe.dit.train()
        v_pred = _dit_velocity(
            pipe.dit, z, [v0_latent, v1_latents[k]], tgt_cam, [v0_cam, middle_cameras[k]],
            tgt_time, [v0_time, mid_times[k]], context, t_T, target_frames,
            dtype, device, use_gradient_checkpointing)
        v2_hat = score_utils.x0_from_velocity(z, v_pred, sigma_T)

        tg, sigma_g, _ = sample_training_timestep(scheduler, batch_size=B, device=device, generator=generator)
        eps_g = torch.randn_like(v2_hat)
        x_tg = scheduler.add_noise(v2_hat, eps_g, tg)
        with torch.no_grad():
            v_real = _dit_velocity(
                teacher_dit, x_tg, [v0_latent], tgt_cam, [v0_cam],
                tgt_time, [v0_time], context, tg, target_frames,
                dtype, device, use_gradient_checkpointing=False)
            fake_dit.eval()
            v_fake = _dit_velocity(
                fake_dit, x_tg, [v0_latent], tgt_cam, [v0_cam],
                tgt_time, [v0_time], context, tg, target_frames,
                dtype, device, use_gradient_checkpointing=False)
        loss_G = score_utils.student_dmd_loss(v2_hat, v_real, v_fake, sigma_g, normalize=True) / K
        loss_G_sum += loss_G.item()
        arrows.append((v_real.float() - v_fake.float()).abs().mean().item())
        loss_G.backward()  # accumulates into θ.grad (no zero between middles -> sum = mean*K/K)
        detached_samples.append(v2_hat.detach())
        del v2_hat, v_pred, x_tg, v_real, v_fake, loss_G
    if offload_teacher:
        teacher_dit.to("cpu")
        torch.cuda.empty_cache()
    if after_g is not None:
        after_g()
    opt_G.step()
    opt_G.zero_grad(set_to_none=True)

    # --- D-step: accumulate the fake-score denoising gradient over the same K samples ---
    opt_D.zero_grad(set_to_none=True)
    loss_D_sum = 0.0
    for k in range(K):
        v2 = detached_samples[k]
        td, sigma_d, _ = sample_training_timestep(scheduler, batch_size=B, device=device, generator=generator)
        eps_d = torch.randn_like(v2)
        x_td = scheduler.add_noise(v2, eps_d, td)
        fake_dit.train()
        v_fake_pred = _dit_velocity(
            fake_dit, x_td, [v0_latent], tgt_cam, [v0_cam],
            tgt_time, [v0_time], context, td, target_frames,
            dtype, device, use_gradient_checkpointing)
        loss_D = score_utils.fake_score_loss(v_fake_pred, v2, eps_d) / K
        loss_D_sum += loss_D.item()
        loss_D.backward()
        del v_fake_pred, x_td, loss_D
    if after_d is not None:
        after_d()
    opt_D.step()
    opt_D.zero_grad(set_to_none=True)

    return {
        "K": K,
        "loss_G": loss_G_sum,
        "loss_D": loss_D_sum,
        "sigma_T": float(sigma_T),
        "arrows": arrows,
        "arrow_abs": sum(arrows) / K,
    }
