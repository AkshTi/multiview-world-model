"""Latent-level helpers for training (Rung 2).

Training happens at the VAE-latent level, never through the pipeline ``__call__`` (which
is ``@torch.no_grad``). VAE encode/decode run under ``no_grad`` and are cached where
possible; only the DiT forward carries gradients.

Convention (must match the model's concat sites in spacetimepilot.py):
  * fusion is a concat along the latent-frame axis dim=2, order [target, source0, source1, ...]
  * one video = 21 latent frames; predictions are sliced to the first 21 (the target) before
    any loss or score conversion.

The pure-tensor helpers (build_latent_input, slice_target) are unit-tested in
tests/test_latents.py; encode/decode wrap the pipeline VAE and need the model + GPU.
"""

import torch

LATENT_FRAMES_PER_VIDEO = 21  # (81 - 1) // 4 + 1


@torch.no_grad()
def encode_video_nograd(pipe, video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    """VAE-encode a pixel video to a latent, detached, on the pipe's dtype/device.

    ``video`` is whatever ``pipe.encode_video`` expects (same tensor the inference path
    feeds it). Returns a latent of shape (B, 16, 21, 60, 104) for an 81-frame 480x832 clip.
    """
    video = video.to(dtype=pipe.torch_dtype, device=pipe.device)
    latents = pipe.encode_video(video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    return latents.to(dtype=pipe.torch_dtype, device=pipe.device).detach()


@torch.no_grad()
def decode_video_nograd(pipe, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    return pipe.decode_video(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)


def build_latent_input(target_latent, source_latents):
    """Concat [target, source0, source1, ...] along the latent-frame axis (dim=2).

    ``source_latents`` may be a single tensor or an ordered list/tuple of tensors. The
    order here MUST match the camera/time/RoPE-position order elsewhere, or a video's
    conditioning lands on another video's tokens (a silent bug).
    """
    if torch.is_tensor(source_latents):
        source_latents = [source_latents]
    else:
        source_latents = list(source_latents)
    if len(source_latents) == 0:
        raise ValueError("need at least one source latent")
    parts = [target_latent] + source_latents
    c = target_latent.shape[1]
    for i, p in enumerate(parts):
        if p.shape[1] != c:
            raise ValueError(f"channel mismatch at part {i}: {p.shape[1]} != {c}")
        if p.shape[0] != target_latent.shape[0]:
            raise ValueError(f"batch mismatch at part {i}")
    return torch.cat(parts, dim=2)


def slice_target(pred, target_frames=LATENT_FRAMES_PER_VIDEO):
    """Keep only the target-video frames of a prediction: pred[:, :, :target_frames]."""
    return pred[:, :, :target_frames, ...]


def num_source_videos(latent_input, target_frames=LATENT_FRAMES_PER_VIDEO):
    """Infer the source count from a fused latent's frame length (sanity helper)."""
    total = latent_input.shape[2]
    if total % target_frames != 0:
        raise ValueError(f"frame length {total} not a multiple of {target_frames}")
    return total // target_frames - 1
