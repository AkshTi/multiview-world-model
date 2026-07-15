"""Correspondence utilities for the eval metrics: SIFT matching + RANSAC homography.

Shared by scripts/eval_report.py (metric-1a scoring) and scripts/calibrate_fov.py
(FOV/self-calibration diagnostics) so there is exactly one matching implementation.

Why metric 1a estimates H from the images instead of deriving it from the commanded
rotation (the 7/14 redesign, measured evidence in scripts/calibrate_fov.py):
self-calibration on 4 v0s x cam01-04 showed the released SPT renders rotations at
~1.2-2.2x the COMMANDED magnitude (scene- and segment-dependent) with no stable implied
FOV -- but individual videos are locally homography-consistent where matches are dense.
So "the two views agree under SOME rotation homography" is measurable; "the two views
agree under the commanded homography" mostly measures the teacher's command disobedience,
which student and baselines all inherit. Estimating H drops the need for K entirely
(intrinsics prereq PLAN section 13 #7 becomes moot for 1a). The copy-collapse loophole
this opens (a2 == v1 makes H_est == I and scores perfectly) is closed by the displacement
guard: genuine rotated views must show substantial matched-point displacement.
"""

import cv2
import numpy as np

# 7/15 robustness pass: the first baseline eval scored only 5/16 (step50) and 2/16 (b1)
# tuples because SIFT's default contrastThreshold (0.04) finds too few keypoints on SOFT
# one-step student generations (sharpness ~120 vs ~170 for 20-step B0 renders). A metric
# that returns null on most tuples can't detect any effect. Lowering the contrast
# threshold + the match floor trades a little per-match purity for coverage; RANSAC
# (estimate_homography) remains the actual outlier gate.
MIN_MATCHES = 10


def make_sift():
    return cv2.SIFT_create(contrastThreshold=0.02), cv2.BFMatcher()


def sift_matches(frame_a, frame_b, sift, matcher, ratio=0.75, min_matches=MIN_MATCHES):
    """Ratio-test-filtered SIFT correspondences: (Nx2 ptsA, Nx2 ptsB), or None if too few."""
    ga = cv2.cvtColor(np.asarray(frame_a), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(np.asarray(frame_b), cv2.COLOR_RGB2GRAY)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < min_matches or len(kb) < min_matches:
        return None
    pts_a, pts_b = [], []
    for m in matcher.knnMatch(da, db, k=2):
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            pts_a.append(ka[m[0].queryIdx].pt)
            pts_b.append(kb[m[0].trainIdx].pt)
    if len(pts_a) < min_matches:
        return None
    return np.asarray(pts_a, dtype=np.float64), np.asarray(pts_b, dtype=np.float64)


def estimate_homography(pts_a, pts_b, ransac_px=3.0, min_inliers=10):
    """RANSAC homography A->B. Returns (H, inlier_mask) or (None, None)."""
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_px)
    if H is None:
        return None, None
    mask = mask.ravel().astype(bool)
    if mask.sum() < min_inliers:
        return None, None
    return H, mask


def displacement_stats(pts_a, pts_b, mask):
    """Inlier matched-point displacement summary. `median_px` is the anti-copy guard:
    H_est ~= I (student copied v1 verbatim) shows up as near-zero displacement, whereas two
    genuinely rotated views displace by f*tan(theta) ~ tens of px."""
    d = pts_b[mask] - pts_a[mask]
    mags = np.linalg.norm(d, axis=1)
    return {
        "median_px": float(np.median(mags)),
        "mean_vec": (float(d[:, 0].mean()), float(d[:, 1].mean())),
        "n_inliers": int(mask.sum()),
        "inlier_frac": float(mask.mean()),
    }


def direction_cosine(observed_vec, predicted_vec):
    """Cosine between observed mean displacement and the commanded rotation's predicted
    displacement direction (obedience signal: sign/direction is FOV-insensitive even though
    magnitude is not). Returns nan if either vector is degenerate."""
    o, p = np.asarray(observed_vec, dtype=np.float64), np.asarray(predicted_vec, dtype=np.float64)
    no, np_ = np.linalg.norm(o), np.linalg.norm(p)
    if no < 1e-6 or np_ < 1e-6:
        return float("nan")
    return float(np.dot(o, p) / (no * np_))
