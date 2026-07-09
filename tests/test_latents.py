"""CPU tests for the pure-tensor latent helpers (Rung 2).

Verifies the fusion concat and target slice produce the documented frame lengths
(1 video -> 21, 2 -> 42, 3 -> 63; slice -> first 21) without the VAE or a GPU.

Run:  python tests/test_latents.py    (or pytest tests/test_latents.py)
"""

import importlib.util
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, os.pardir, "spacetimepilot", "training", "latents.py")
_spec = importlib.util.spec_from_file_location("spt_latents", _PATH)
lat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lat)

B, C, F, H, W = 2, 16, 21, 60, 104


def _vid():
    return torch.randn(B, C, F, H, W)


def test_two_video_fusion_is_42_frames():
    x = lat.build_latent_input(_vid(), _vid())
    assert x.shape == (B, C, 42, H, W)
    assert lat.num_source_videos(x) == 1


def test_three_video_fusion_is_63_frames():
    x = lat.build_latent_input(_vid(), [_vid(), _vid()])
    assert x.shape == (B, C, 63, H, W)
    assert lat.num_source_videos(x) == 2


def test_slice_target_keeps_first_21():
    x = lat.build_latent_input(_vid(), [_vid(), _vid()])
    tgt = lat.slice_target(x)
    assert tgt.shape == (B, C, 21, H, W)


def test_order_is_target_first():
    tgt = torch.zeros(B, C, F, H, W)
    src = torch.ones(B, C, F, H, W)
    x = lat.build_latent_input(tgt, src)
    assert torch.all(x[:, :, :21] == 0)   # target block first
    assert torch.all(x[:, :, 21:] == 1)   # source block after


def test_channel_mismatch_raises():
    tgt = torch.randn(B, C, F, H, W)
    bad = torch.randn(B, C + 1, F, H, W)
    raised = False
    try:
        lat.build_latent_input(tgt, bad)
    except ValueError:
        raised = True
    assert raised


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} latent-helper tests passed.")


if __name__ == "__main__":
    _run_all()
