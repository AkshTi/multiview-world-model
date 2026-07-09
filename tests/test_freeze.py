"""CPU tests for the freeze/unfreeze + grad-mask logic (Rung 2).

Uses a mock module that mirrors the DiTBlock submodule *names* the real model uses
(cam_encoder, projector, frame_time_embedding, temporal_downsampler, self_attn.q/k/v/o,
plus frozen-only ffn/cross_attn), so the name-matching is verified without loading the
1.3B DiT or a GPU.

Run:  python tests/test_freeze.py    (or pytest tests/test_freeze.py)
"""

import importlib.util
import os

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, os.pardir, "spacetimepilot", "training", "freeze.py")
_spec = importlib.util.spec_from_file_location("spt_freeze", _PATH)
freeze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze)


class _SelfAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)  # intentionally NOT in the trainable set

    def forward(self, x):
        return self.o(self.q(x) + self.k(x) + self.v(x))


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cam_encoder = nn.Linear(dim, dim)
        self.projector = nn.Linear(dim, dim)
        self.frame_time_embedding = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.temporal_downsampler = nn.Linear(dim, dim)
        self.self_attn = _SelfAttn(dim)
        self.cross_attn = nn.Linear(dim, dim)  # frozen-only
        self.ffn = nn.Linear(dim, dim)          # frozen-only

    def forward(self, x):
        x = x + self.projector(self.self_attn(x))
        x = x + self.cross_attn(x)
        x = x + self.ffn(x)
        return x


class _DiT(nn.Module):
    def __init__(self, dim=8, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def _expected_trainable(name):
    return any(
        s in "." + name
        for s in (
            ".cam_encoder.",
            ".projector.",
            ".frame_time_embedding.",
            ".temporal_downsampler.",
            ".self_attn.q.",
            ".self_attn.k.",
            ".self_attn.v.",
            ".self_attn.o.",
        )
    )


def test_set_trainable_marks_exactly_the_right_params():
    dit = _DiT()
    freeze.set_trainable(dit)
    for name, p in dit.named_parameters():
        assert p.requires_grad == _expected_trainable(name), name
    # sanity: frozen-only modules and norm_q are indeed frozen
    frozen = dict(dit.named_parameters())
    assert not frozen["blocks.0.ffn.weight"].requires_grad
    assert not frozen["blocks.0.cross_attn.weight"].requires_grad
    assert not frozen["blocks.0.self_attn.norm_q.weight"].requires_grad
    assert frozen["blocks.1.self_attn.q.weight"].requires_grad
    assert frozen["blocks.0.cam_encoder.weight"].requires_grad


def test_grad_mask_passes_after_backward():
    dit = _DiT()
    trainable = freeze.set_trainable(dit)
    opt = torch.optim.AdamW(trainable, lr=1e-3)  # optimizer over trainable only
    opt.zero_grad()
    x = torch.randn(4, 8)
    dit(x).pow(2).mean().backward()
    assert freeze.assert_grad_mask(dit) is True


def test_grad_mask_catches_a_leak():
    dit = _DiT()
    freeze.set_trainable(dit)
    # simulate a bug: someone re-enabled a frozen module
    dit.blocks[0].ffn.weight.requires_grad_(True)
    x = torch.randn(4, 8)
    dit(x).pow(2).mean().backward()
    raised = False
    try:
        freeze.assert_grad_mask(dit)
    except AssertionError:
        raised = True
    assert raised, "assert_grad_mask should have flagged the leaked frozen param"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} freeze tests passed.")


if __name__ == "__main__":
    _run_all()
