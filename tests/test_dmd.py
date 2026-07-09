"""CPU tests for the Rung 5 DMD loop's new math + detachment contract.

The two-arrow score math is already covered by tests/test_score.py. Here we cover:
  * fake_score_loss computes the flow-match target (eps - x0) and updates only its prediction;
  * the G-step / D-step detachment discipline that dmd_step_k1 implements — G-loss reaches
    the student sample only, D-loss reaches the fake prediction only, and neither leaks into
    the "teacher". Uses toy tensors + tiny Linear stand-ins (no SPT, no GPU).
"""

import importlib.util
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, os.pardir, "spacetimepilot", "training", "score.py")
_spec = importlib.util.spec_from_file_location("spt_score", _PATH)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def _add_noise(x0, eps, sigma):
    return (1.0 - sigma) * x0 + sigma * eps


def test_flow_match_target_is_eps_minus_x0():
    x0 = torch.randn(2, 3)
    eps = torch.randn(2, 3)
    assert torch.allclose(sc.flow_match_target(x0, eps), eps - x0)


def test_fake_score_loss_zero_at_perfect_prediction():
    x0 = torch.randn(2, 4)
    eps = torch.randn(2, 4)
    v_pred = eps - x0  # perfect velocity prediction
    assert sc.fake_score_loss(v_pred, x0, eps).item() < 1e-12


def test_fake_score_loss_grad_only_through_prediction():
    x0 = torch.randn(2, 4, requires_grad=True)
    eps = torch.randn(2, 4, requires_grad=True)
    v_pred = torch.randn(2, 4, requires_grad=True)
    loss = sc.fake_score_loss(v_pred, x0.detach(), eps.detach())
    loss.backward()
    assert v_pred.grad is not None and torch.any(v_pred.grad != 0)
    assert x0.grad is None and eps.grad is None  # detached targets carry no grad


def test_g_step_updates_student_not_fake_or_teacher():
    """G-loss backward reaches the student sample only (teacher/fake are no_grad)."""
    torch.manual_seed(0)
    student = torch.nn.Linear(4, 4)
    fake = torch.nn.Linear(4, 4)
    teacher = torch.nn.Linear(4, 4)

    z = torch.randn(3, 4)
    sigma_T = torch.tensor(1.0)
    v_pred = student(z)
    v2_hat = sc.x0_from_velocity(z, v_pred, sigma_T)  # grad on student

    sigma_g = torch.tensor(0.4)
    eps_g = torch.randn(3, 4)
    x_tg = _add_noise(v2_hat, eps_g, sigma_g)
    with torch.no_grad():
        v_real = teacher(x_tg)
        v_fake = fake(x_tg)
    loss_G = sc.student_dmd_loss(v2_hat, v_real, v_fake, sigma_g, normalize=True)
    loss_G.backward()

    assert student.weight.grad is not None and torch.any(student.weight.grad != 0)
    assert fake.weight.grad is None
    assert teacher.weight.grad is None


def test_d_step_updates_fake_not_student_or_teacher():
    """D-loss backward reaches the fake net only (student sample detached)."""
    torch.manual_seed(1)
    student = torch.nn.Linear(4, 4)
    fake = torch.nn.Linear(4, 4)
    teacher = torch.nn.Linear(4, 4)

    z = torch.randn(3, 4)
    v2_hat = sc.x0_from_velocity(z, student(z), torch.tensor(1.0)).detach()

    sigma_d = torch.tensor(0.6)
    eps_d = torch.randn(3, 4)
    x_td = _add_noise(v2_hat, eps_d, sigma_d)
    v_fake_pred = fake(x_td)
    loss_D = sc.fake_score_loss(v_fake_pred, v2_hat, eps_d)
    loss_D.backward()

    assert fake.weight.grad is not None and torch.any(fake.weight.grad != 0)
    assert student.weight.grad is None
    assert teacher.weight.grad is None


def test_gradient_accumulation_over_k_equals_mean_loss():
    """Rung 6: accumulating (loss_k / K).backward() over K middles == grad of the mean loss.

    This is the identity dmd_step_k relies on to marginalize over middles with only one
    student graph live at a time (peak memory independent of K).
    """
    torch.manual_seed(2)
    K = 3
    zs = [torch.randn(3, 4) for _ in range(K)]
    v_real = [torch.randn(3, 4) for _ in range(K)]
    v_fake = [torch.randn(3, 4) for _ in range(K)]
    sigma = torch.tensor(0.5)
    lin = torch.nn.Linear(4, 4)

    def loss_k(k):
        x0 = sc.x0_from_velocity(zs[k], lin(zs[k]), torch.tensor(1.0))
        return sc.student_dmd_loss(x0, v_real[k], v_fake[k], sigma, normalize=False)

    lin.zero_grad()
    for k in range(K):
        (loss_k(k) / K).backward()
    g_acc = lin.weight.grad.clone()

    lin.zero_grad()
    total = sum(loss_k(k) / K for k in range(K))
    total.backward()
    g_mean = lin.weight.grad.clone()

    assert torch.allclose(g_acc, g_mean, atol=1e-6)


def test_arrow_is_zero_when_scores_agree():
    """At init teacher == fake => arrow == 0 => G-loss coefficient == 0 (sanity gate)."""
    x0_hat = torch.randn(2, 4, requires_grad=True)
    v = torch.randn(2, 4)
    sigma = torch.tensor(0.5)
    loss = sc.student_dmd_loss(x0_hat, v_real=v, v_fake=v.clone(), sigma=sigma, normalize=False)
    loss.backward()
    assert torch.allclose(x0_hat.grad, torch.zeros_like(x0_hat.grad))


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} DMD tests passed.")


if __name__ == "__main__":
    _run_all()
