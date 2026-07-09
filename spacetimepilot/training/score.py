"""Velocity <-> score conversion and the DMD student loss (Rung 1).

Authority for the conventions is the flow-matching scheduler,
``spacetimepilot/wan/schedulers/flow_match.py``:

    add_noise:        x_t = (1 - sigma) * x0 + sigma * eps        (eps ~ N(0, I))
    training_target:  v   = eps - x0                              (model predicts velocity)

Inverting those two linear relations:

    x0  = x_t - sigma * v_pred
    eps = x_t + (1 - sigma) * v_pred
    s(x_t) = -eps / sigma = -(x_t + (1 - sigma) * v_pred) / sigma

so the DMD arrow collapses to a velocity difference (the x_t term cancels):

    A = s_real - s_fake = -((1 - sigma) / sigma) * (v_real - v_fake)

A wrong sign/scale here silently trains toward garbage, so every function below is
covered by ``tests/test_score.py`` and must be run before any training.

Q1 is resolved to the direct-teacher / student-side-marginal branch
(see HYUNWOO_RECONCILIATION.md): ``v_real`` is the direct teacher velocity
(source = v0), ``v_fake`` is the one-source marginal fake-score velocity (source = v0,
v1 hidden). This module is agnostic to how those velocities are produced.
"""

import torch


def _broadcast_sigma(sigma, ref):
    """Return sigma shaped to broadcast against ``ref``.

    Accepts a python float, a 0-d tensor, or a per-sample 1-d tensor of shape (B,),
    which is reshaped to (B, 1, 1, ...) to line up with a (B, C, F, H, W) latent.
    """
    if not torch.is_tensor(sigma):
        return sigma
    if sigma.ndim == 0:
        return sigma
    if sigma.ndim == 1:
        return sigma.view(sigma.shape[0], *([1] * (ref.ndim - 1)))
    return sigma


def x0_from_velocity(x_t, v_pred, sigma):
    """x0 = x_t - sigma * v_pred."""
    s = _broadcast_sigma(sigma, x_t)
    return x_t - s * v_pred


def eps_from_velocity(x_t, v_pred, sigma):
    """eps = x_t + (1 - sigma) * v_pred."""
    s = _broadcast_sigma(sigma, x_t)
    return x_t + (1.0 - s) * v_pred


def velocity_to_score(x_t, v_pred, sigma):
    """Flow-matching score s(x_t) = -eps / sigma = -(x_t + (1 - sigma) v_pred) / sigma."""
    s = _broadcast_sigma(sigma, x_t)
    return -(x_t + (1.0 - s) * v_pred) / s


def dmd_coefficient(v_real, v_fake, sigma):
    """Detached DMD coefficient c = (1 - sigma)/sigma * (v_real - v_fake).

    Algebraically c == s_fake - s_real (the x_t terms cancel), so the surrogate loss
    ``<stop_grad(c), x0_hat>`` has gradient c w.r.t. x0_hat, and gradient DESCENT moves
    x0_hat along ``-c = s_real - s_fake`` (uphill on the teacher, away from the
    student's own pile-up).
    """
    s = _broadcast_sigma(sigma, v_real)
    return (1.0 - s) / s * (v_real - v_fake)


def student_dmd_loss(x0_hat, v_real, v_fake, sigma, normalize=True, eps=1e-8):
    """DMD student loss in detached-surrogate form.

        loss = mean( stop_grad(c) * x0_hat ),   c = (1 - sigma)/sigma * (v_real - v_fake)

    Only ``x0_hat`` carries gradient; ``v_real``/``v_fake``/``sigma`` are constants
    (in real training they are computed at x_t = noise(x0_hat) under no_grad and
    detached). Minimizing this moves the student along ``s_real - s_fake``.

    Parameters
    ----------
    normalize : bool
        Divide the coefficient by its per-sample mean-abs magnitude (a standard DMD
        stabilizer so small ``sigma`` / large arrows don't dominate the step). This is
        a positive per-sample rescale, so it never changes the gradient's sign.
    """
    c = dmd_coefficient(v_real, v_fake, sigma).detach()
    if normalize:
        dims = tuple(range(1, c.ndim)) if c.ndim > 1 else None
        if dims is None:
            scale = c.abs().mean().clamp_min(eps)
        else:
            scale = c.abs().mean(dim=dims, keepdim=True).clamp_min(eps)
        c = c / scale
    return (c * x0_hat).mean()


def flow_match_target(x0, eps):
    """Flow-matching velocity target v = eps - x0 (matches scheduler.training_target)."""
    return eps - x0


def fake_score_loss(v_fake_pred, x0, eps):
    """Denoising-MSE loss that trains the fake-score net (Rung 5, D-step).

    The fake-score net is a 1-source diffusion model regressed on the STUDENT's samples
    ``x0`` (here ``v2_hat.detach()``) with the middle v1 hidden. Regressing velocity on a
    process whose sample depends on a hidden variable recovers the conditional mean over
    that variable — i.e. the *marginal* student score ``s_fake = ∇log q_θ(v2|v0)`` used in
    the two-arrow update (the "MSE trick").

    Both ``v_fake_pred`` (the fake net's velocity prediction at the noised sample) and the
    target ``eps - x0`` are on the target-video frames only; ``x0``/``eps`` must be detached
    from the generator graph so this loss updates only φ, never θ.
    """
    return torch.mean((v_fake_pred.float() - flow_match_target(x0, eps).float()) ** 2)
