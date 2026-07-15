"""CPU tests for spacetimepilot/eval/matching.py (metric 1a's estimated-H machinery).

Synthetic ground truth, no bank/GPU needed: build a textured image, warp it by a KNOWN
rotation homography, and check the SIFT+RANSAC path recovers the geometry. Also pins the two
guards the 7/14 metric-1a amendment leans on:
  * anti-copy guard: identical frames -> H_est ~= I -> near-zero median displacement;
  * direction_cosine: aligned vectors -> +1, opposed -> -1, degenerate -> nan.

Runnable via pytest OR directly: `python tests/test_eval_matching.py`.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetimepilot.eval import geometry, matching

W, H_IMG = 832, 480


def _textured_image(seed=0):
    """Blob-rich random texture (SIFT needs structure; pure noise has none)."""
    rng = np.random.RandomState(seed)
    small = rng.randint(0, 255, (H_IMG // 8, W // 8, 3), dtype=np.uint8)
    img = cv2.resize(small, (W, H_IMG), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


def _rotation_h(pan_deg=4.0, fov_deg=60.0):
    K = geometry.intrinsics_matrix(fov_deg, W, H_IMG)
    th = np.deg2rad(pan_deg)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    return geometry.homography_from_rotation(K, R), R


def test_estimated_h_recovers_known_warp():
    img = _textured_image()
    H_true, _ = _rotation_h()
    warped = cv2.warpPerspective(img, H_true, (W, H_IMG))

    sift, matcher = matching.make_sift()
    m = matching.sift_matches(img, warped, sift, matcher)
    assert m is not None, "SIFT must find matches on a textured image and its warp"
    H_est, mask = matching.estimate_homography(m[0], m[1])
    assert H_est is not None

    # H_est must act like H_true on interior points (compare actions, not matrix entries).
    pts = np.array([[200.0, 150.0, 1], [600.0, 300.0, 1], [416.0, 240.0, 1]]).T
    a = (H_true @ pts); a = (a[:2] / a[2]).T
    b = (H_est @ pts); b = (b[:2] / b[2]).T
    assert np.max(np.linalg.norm(a - b, axis=1)) < 3.0, "estimated H deviates from ground truth"

    stats = matching.displacement_stats(m[0], m[1], mask)
    # 4 deg pan at fov 60 (f ~= 720 px) displaces ~ f*tan(4deg) ~= 50 px -- far above the guard.
    assert stats["median_px"] > 20.0, "real rotation must show substantial displacement"
    assert stats["inlier_frac"] > 0.5


def test_copy_guard_trips_on_identical_frames():
    img = _textured_image(seed=1)
    sift, matcher = matching.make_sift()
    m = matching.sift_matches(img, img.copy(), sift, matcher)
    assert m is not None
    H_est, mask = matching.estimate_homography(m[0], m[1])
    assert H_est is not None
    stats = matching.displacement_stats(m[0], m[1], mask)
    assert stats["median_px"] < 1.0, "identical frames must show ~zero displacement (copy guard)"


def test_direction_cosine():
    assert abs(matching.direction_cosine((10, 0), (3, 0)) - 1.0) < 1e-9
    assert abs(matching.direction_cosine((10, 0), (-3, 0)) + 1.0) < 1e-9
    assert np.isnan(matching.direction_cosine((0, 0), (1, 0)))


def test_predicted_center_displacement_direction_is_fov_insensitive():
    _, R = _rotation_h(pan_deg=5.0, fov_deg=60.0)
    d1 = np.array(geometry.predicted_center_displacement(R, 40.0, W, H_IMG))
    d2 = np.array(geometry.predicted_center_displacement(R, 90.0, W, H_IMG))
    cos = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
    assert cos > 0.999, "direction must agree across very different fov assumptions"
    assert abs(np.linalg.norm(d1) - np.linalg.norm(d2)) > 1.0, \
        "magnitudes SHOULD differ across fovs (that's why we only use direction)"


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(_TESTS)} eval-matching tests passed")
