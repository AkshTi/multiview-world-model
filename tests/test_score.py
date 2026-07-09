"""Unit tests for the flow-matching velocity<->score conversion and DMD sign (Rung 1).

These are pure-tensor tests (CPU, no model, no GPU). They must pass before any training:
a wrong sign or scale in score.py silently trains toward garbage.

We load score.py directly by file path so the test does not trigger the heavy
``spacetimepilot`` package __init__ (which imports the full DiT model).

Run:  python tests/test_score.py     (or pytest tests/test_score.py)
"""

import importlib.util
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCORE_PATH = os.path.join(_HERE, os.pardir, "spacetimepilot", "training", "score.py")
_spec = importlib.util.spec_from_file_location("spt_score", _SCORE_PATH)
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

TOL = dict(rtol=1e-5, atol=1e-5)


def _make_case(B=2, C=16, F=3, H=4, W=4, seed=0, per_sample_sigma=True):
    """Return (x0, eps, x_t, v, sigma) consistent with the scheduler conventions."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(B, C, F, H, W, generator=g, dtype=torch.float64)
    eps = torch.randn(B, C, F, H, W, generator=g, dtype=torch.float64)
    if per_sample_sigma:
        sigma = torch.rand(B, generator=g, dtype=torch.float64) * 0.9 + 0.05  # (B,) in [0.05, 0.95]
        s = sigma.view(B, 1, 1, 1, 1)
    else:
        sigma = float(torch.rand(1, generator=g).item() * 0.9 + 0.05)
        s = sigma
    x_t = (1.0 - s) * x0 + s * eps
    v = eps - x0
    return x0, eps, x_t, v, sigma


def test_x0_eps_reconstruction():
    x0, eps, x_t, v, sigma = _make_case()
    assert torch.allclose(score.x0_from_velocity(x_t, v, sigma), x0, **TOL)
    assert torch.allclose(score.eps_from_velocity(x_t, v, sigma), eps, **TOL)


def test_velocity_to_score_matches_analytic():
    x0, eps, x_t, v, sigma = _make_case()
    s = sigma.view(-1, 1, 1, 1, 1) if torch.is_tensor(sigma) else sigma
    analytic = -eps / s
    assert torch.allclose(score.velocity_to_score(x_t, v, sigma), analytic, **TOL)


def test_scalar_sigma_also_works():
    x0, eps, x_t, v, sigma = _make_case(per_sample_sigma=False)
    assert torch.allclose(score.x0_from_velocity(x_t, v, sigma), x0, **TOL)
    assert torch.allclose(score.velocity_to_score(x_t, v, sigma), -eps / sigma, **TOL)


def test_dmd_arrow_cancels_x_t():
    """s_real - s_fake must equal -(1-sigma)/sigma (v_real - v_fake), independent of x_t."""
    x0, eps, x_t, v, sigma = _make_case(seed=1)
    v_real = torch.randn_like(v)
    v_fake = torch.randn_like(v)
    s_real = score.velocity_to_score(x_t, v_real, sigma)
    s_fake = score.velocity_to_score(x_t, v_fake, sigma)
    coeff = score.dmd_coefficient(v_real, v_fake, sigma)  # == s_fake - s_real
    assert torch.allclose(s_real - s_fake, -coeff, **TOL)
    # And the cancellation really is x_t-independent: recompute with a different x_t.
    x_t2 = x_t + 3.14
    s_real2 = score.velocity_to_score(x_t2, v_real, sigma)
    s_fake2 = score.velocity_to_score(x_t2, v_fake, sigma)
    assert torch.allclose(s_real2 - s_fake2, s_real - s_fake, **TOL)


def test_dmd_descent_moves_toward_teacher():
    """Gradient DESCENT on the student loss must move x0_hat along s_real - s_fake."""
    x0, eps, x_t, v, sigma = _make_case(seed=2)
    v_real = torch.randn_like(v)
    v_fake = torch.randn_like(v)

    x0_hat = x0.clone().requires_grad_(True)
    loss = score.student_dmd_loss(x0_hat, v_real, v_fake, sigma, normalize=False)
    loss.backward()

    n = x0_hat.numel()
    coeff = score.dmd_coefficient(v_real, v_fake, sigma)
    # d/dx0_hat mean(c * x0_hat) = c / N
    assert torch.allclose(x0_hat.grad, coeff / n, **TOL)

    # descent direction = -grad; teacher-improving direction = s_real - s_fake = -coeff.
    descent = -x0_hat.grad
    teacher_dir = -coeff
    # positive alignment: descent points the same way as (s_real - s_fake)
    assert torch.sum(descent * teacher_dir) > 0
    assert torch.allclose(descent, teacher_dir / n, **TOL)


def test_flipped_sign_moves_away_from_teacher():
    """The buggy sign (v_fake - v_real) must move AWAY from the teacher."""
    x0, eps, x_t, v, sigma = _make_case(seed=3)
    v_real = torch.randn_like(v)
    v_fake = torch.randn_like(v)
    teacher_dir = -score.dmd_coefficient(v_real, v_fake, sigma)  # s_real - s_fake

    x0_hat = x0.clone().requires_grad_(True)
    # swap the arguments == flip the sign inside the coefficient
    bad_loss = score.student_dmd_loss(x0_hat, v_fake, v_real, sigma, normalize=False)
    bad_loss.backward()
    descent = -x0_hat.grad
    assert torch.sum(descent * teacher_dir) < 0  # anti-aligned -> wrong way


def test_normalize_preserves_sign():
    x0, eps, x_t, v, sigma = _make_case(seed=4)
    v_real = torch.randn_like(v)
    v_fake = torch.randn_like(v)
    teacher_dir = -score.dmd_coefficient(v_real, v_fake, sigma)

    x0_hat = x0.clone().requires_grad_(True)
    loss = score.student_dmd_loss(x0_hat, v_real, v_fake, sigma, normalize=True)
    loss.backward()
    descent = -x0_hat.grad
    assert torch.sum(descent * teacher_dir) > 0  # same sign as unnormalized


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} score/DMD tests passed.")


if __name__ == "__main__":
    _run_all()
