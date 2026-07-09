"""Stitching term (D3, Rung 7): forbid the gray-blur cheat.

"Consistent" has a trivial degenerate solution — make every view a blur; two blurs agree.
The stitching term forbids it: take ONE frame from each jointly generated view at a shared
world-time, lay them end-to-end into a strip x, and demand the strip look like a REAL single
video (sharp, temporally smooth) by aligning its marginal p(x) to the single-video law p_1(x)
with DMD. A blur fails that test; a genuinely-consistent set passes.

    x = W · [v0 v1 ... vN]ᵀ ,   W = disjoint binary "pick one frame per view" selection.

Because W is a disjoint linear selection, W Wᵀ = I, so noising commutes with slicing and the
strip's score is the joint score sliced the same way:

    s_x(x_t) = W · s_v(v_t).

``velocity_to_score`` (score.py) is elementwise, so it commutes with any frame selection —
that IS the ``s_x = W s_v`` identity in code, and it means the stitching term reuses the
existing per-view score with NO new network (the DPS "linear inverse problem with a diffusion
prior" machinery). Derivation D3 is settled.

Pure / CPU-tested here (tests/test_stitch.py). The one GPU/convention-dependent piece is the
single-video-prior teacher velocity on the strip (``s_real`` for p_1): presenting a synthetic
N-frame strip to the released model needs the world-time/RoPE convention that is still an open
Thursday item, so ``stitching_student_loss`` takes that ``v_real_strip`` as an argument rather
than inventing a camera/time convention here.
"""

import torch

from . import score as score_utils


def stitch(views, frame_picks):
    """W·[v0..vN]: take frame ``frame_picks[i]`` from view i and stack into a strip.

    ``views``       : list of N latents, each (B, C, F, H, W).
    ``frame_picks`` : list of N ints — the latent-frame index taken from each view (the frames
                      sharing the world-time). One output frame per view ⇒ the selection is
                      disjoint across output positions ⇒ W Wᵀ = I.
    Returns the strip (B, C, N, H, W) with output frame i = views[i][:, :, frame_picks[i]].
    """
    if len(views) != len(frame_picks):
        raise ValueError("views and frame_picks must have equal length")
    if len(views) == 0:
        raise ValueError("need at least one view")
    cols = []
    for v, f in zip(views, frame_picks):
        if not (0 <= f < v.shape[2]):
            raise ValueError(f"frame pick {f} out of range for a view with {v.shape[2]} frames")
        cols.append(v[:, :, f:f + 1])   # (B, C, 1, H, W)
    return torch.cat(cols, dim=2)       # (B, C, N, H, W)


def selection_matrix(frame_picks, frames_per_view):
    """Build the explicit W (N, N*frames_per_view) for the disjoint selection (for tests/analysis).

    Row i is a one-hot over the joint frame index of view i, so distinct rows hit distinct
    joint indices ⇒ ``W @ W.T == I_N``. This is the concrete witness that a disjoint one-frame
    -per-view selection is orthonormal (the property the score-slicing identity rests on).
    """
    n = len(frame_picks)
    W = torch.zeros(n, n * frames_per_view)
    for i, f in enumerate(frame_picks):
        if not (0 <= f < frames_per_view):
            raise ValueError(f"frame pick {f} out of range [0,{frames_per_view})")
        W[i, i * frames_per_view + f] = 1.0
    return W


def stitched_score(view_latents_t, view_velocities, sigma, frame_picks):
    """s_x(x_t) = W · s_v(v_t): the strip's score by slicing the per-view scores.

    ``view_latents_t`` are the NOISED per-view latents, ``view_velocities`` the per-view
    predicted velocities. Equals ``velocity_to_score(stitch(v_t), stitch(vel), sigma)`` because
    the score formula is elementwise; provided both ways agree in tests.
    """
    x_t = stitch(view_latents_t, frame_picks)
    v_x = stitch(view_velocities, frame_picks)
    return score_utils.velocity_to_score(x_t, v_x, sigma)


def stitching_student_loss(strip_x0_hat, v_real_strip, v_fake_strip, sigma,
                           weight=1.0, normalize=True):
    """Stitched DMD term added to the student surrogate on a (small) weight schedule.

    Same two-arrow DMD loss as the main objective (``score.student_dmd_loss``) but applied to
    the STRIP: pull the stitched-marginal p(x) toward the single-video law p_1(x).

    ``strip_x0_hat`` : the strip built from the generated views' clean samples (carries θ-grad,
        via ``stitch`` of the per-view x0_hat).
    ``v_real_strip`` : single-video-prior velocity on the strip (teacher for p_1) — supplied by
        the caller; presenting a synthetic strip to the teacher is the open convention piece.
    ``v_fake_strip`` : the student/fake velocity on the strip, e.g. ``stitch`` of the per-view
        fake velocities (reuses the score, no new net).
    ``weight``       : schedule coefficient (start small; D3 is a regularizer on top of D1/D2).
    """
    return weight * score_utils.student_dmd_loss(
        strip_x0_hat, v_real_strip, v_fake_strip, sigma, normalize=normalize)
