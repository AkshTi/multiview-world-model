"""Freeze/unfreeze the DiT and assert the gradient mask (Rung 2).

Freeze the whole DiT, then re-enable only the modules SPT fine-tunes:
``cam_encoder``, ``projector``, ``frame_time_embedding``, ``temporal_downsampler`` and
the self-attention ``q/k/v/o`` (each present once per DiTBlock). Matched by name suffix,
verified in ``tests/test_freeze.py`` against a mock with the same submodule names.

Checkpointing only runs in train mode (``if self.training and use_gradient_checkpointing``
in ``WanModel.forward``), so the student and fake-score DiTs must be ``.train()`` even
though most of their parameters are frozen; the frozen teacher stays ``.eval()``.
"""

import torch

DEFAULT_TRAINABLE_PATTERNS = (
    ".cam_encoder.",
    ".projector.",
    ".frame_time_embedding.",
    ".temporal_downsampler.",
    ".self_attn.q.",
    ".self_attn.k.",
    ".self_attn.v.",
    ".self_attn.o.",
)


def _matches(name, patterns):
    padded = "." + name  # so a leading module name also matches ".pattern"
    return any(p in padded for p in patterns)


def set_trainable(dit, patterns=DEFAULT_TRAINABLE_PATTERNS, verbose=False):
    """Freeze all of ``dit``, then unfreeze only params whose name matches a pattern.

    Returns the list of trainable parameters for the optimizer.
    """
    dit.requires_grad_(False)
    trainable = []
    for name, p in dit.named_parameters():
        if _matches(name, patterns):
            p.requires_grad_(True)
            trainable.append(p)
    if not trainable:
        raise RuntimeError(
            "No trainable parameters matched the patterns; check them against the model's "
            "named_parameters()."
        )
    if verbose:
        n = sum(p.numel() for p in trainable)
        tot = sum(p.numel() for p in dit.parameters())
        print(f"[freeze] trainable: {n:,} / {tot:,} params ({100.0 * n / tot:.2f}%)")
    return trainable


def trainable_parameter_names(dit, patterns=DEFAULT_TRAINABLE_PATTERNS):
    return [name for name, _ in dit.named_parameters() if _matches(name, patterns)]


def assert_grad_mask(dit, patterns=DEFAULT_TRAINABLE_PATTERNS, require_any_nonzero=True):
    """After ``loss.backward()``: assert gradients did not leak to frozen params.

    Checks:
      * every frozen param has ``requires_grad == False`` and no nonzero grad;
      * at least one intended-trainable param received a nonzero grad
        (``require_any_nonzero``), so the graph really reached the unfrozen modules.

    Raises ``AssertionError`` listing every violation.
    """
    problems = []
    trainable_with_grad = 0
    for name, p in dit.named_parameters():
        should_train = _matches(name, patterns)
        nonzero_grad = p.grad is not None and bool(torch.any(p.grad != 0))
        if should_train:
            if p.requires_grad and nonzero_grad:
                trainable_with_grad += 1
        else:
            if p.requires_grad:
                problems.append(f"frozen param still requires_grad: {name}")
            if nonzero_grad:
                problems.append(f"frozen param has nonzero grad (leak): {name}")
    if require_any_nonzero and trainable_with_grad == 0:
        problems.append("no trainable param received a nonzero gradient")
    if problems:
        raise AssertionError("grad-mask violations:\n  " + "\n  ".join(problems))
    return True
