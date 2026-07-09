"""CPU tests for the stitching term D3 (Rung 7).

Pins the settled D3 properties: the disjoint one-frame-per-view selection is orthonormal
(W Wᵀ = I), noising commutes with selection, and the strip's score equals the joint score
sliced the same way (s_x = W s_v) — so the stitching term reuses the per-view score with no
new network. Also checks the stitched DMD loss carries θ-grad only through the strip.
"""

import importlib.util
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(mod_name, rel):
    path = os.path.join(_HERE, os.pardir, "spacetimepilot", "training", rel)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# stitch imports `from . import score`, so load as a proper package-ish pair.
import sys  # noqa: E402
sys.path.insert(0, os.path.join(_HERE, os.pardir))
from spacetimepilot.training import stitch as st  # noqa: E402
from spacetimepilot.training import score as sc   # noqa: E402


def _views(n, B=2, C=3, F=5, H=4, W=4):
    return [torch.randn(B, C, F, H, W) for _ in range(n)]


def test_stitch_selects_the_right_frames():
    views = _views(3)
    picks = [0, 2, 4]
    strip = st.stitch(views, picks)
    assert strip.shape == (2, 3, 3, 4, 4)
    for i, f in enumerate(picks):
        assert torch.allclose(strip[:, :, i], views[i][:, :, f])


def test_disjoint_selection_is_orthonormal():
    """W Wᵀ = I for a disjoint one-frame-per-view selection (the property D3 rests on)."""
    picks = [1, 3, 0, 4]
    W = st.selection_matrix(picks, frames_per_view=5)
    WWt = W @ W.T
    assert torch.allclose(WWt, torch.eye(len(picks)))


def test_noising_commutes_with_selection():
    """stitch(add_noise(v, eps)) == add_noise(stitch(v), stitch(eps)) for shared noise."""
    views = _views(3)
    eps = _views(3)
    sigma = 0.37
    picks = [0, 1, 2]
    noised_then_stitched = st.stitch([(1 - sigma) * v + sigma * e for v, e in zip(views, eps)], picks)
    stitched_then_noised = (1 - sigma) * st.stitch(views, picks) + sigma * st.stitch(eps, picks)
    assert torch.allclose(noised_then_stitched, stitched_then_noised, atol=1e-6)


def test_score_slicing_identity_s_x_equals_W_s_v():
    """s_x(x_t) = W s_v(v_t): score of the stitched strip == stitched per-view scores."""
    views_t = _views(3)
    vels = _views(3)
    sigma = torch.tensor(0.42)
    picks = [4, 1, 3]

    s_x = st.stitched_score(views_t, vels, sigma, picks)                      # via slicing
    per_view_scores = [sc.velocity_to_score(vt, vv, sigma) for vt, vv in zip(views_t, vels)]
    s_v_sliced = st.stitch(per_view_scores, picks)                            # slice the joint score
    assert torch.allclose(s_x, s_v_sliced, atol=1e-6)


def test_stitching_loss_grad_only_through_strip():
    picks = [0, 1, 2]
    # per-view x0_hat carries grad (stands in for the generated views)
    views = [torch.randn(2, 3, 5, 4, 4, requires_grad=True) for _ in range(3)]
    strip_x0 = st.stitch(views, picks)
    v_real = st.stitch(_views(3), picks).detach()
    v_fake = st.stitch(_views(3), picks).detach()
    loss = st.stitching_student_loss(strip_x0, v_real, v_fake, torch.tensor(0.5), weight=0.1)
    loss.backward()
    assert all(v.grad is not None for v in views)  # grad reaches the generated views


def test_zero_arrow_gives_zero_stitching_grad():
    picks = [0, 1]
    views = [torch.randn(1, 2, 3, 2, 2, requires_grad=True) for _ in range(2)]
    strip_x0 = st.stitch(views, picks)
    v = st.stitch(_views(2, B=1, C=2, F=3, H=2, W=2), picks).detach()
    loss = st.stitching_student_loss(strip_x0, v, v.clone(), torch.tensor(0.5),
                                     weight=1.0, normalize=False)
    loss.backward()
    assert all(torch.allclose(v_.grad, torch.zeros_like(v_.grad)) for v_ in views)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} stitching tests passed.")


if __name__ == "__main__":
    _run_all()
