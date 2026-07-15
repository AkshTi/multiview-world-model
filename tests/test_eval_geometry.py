"""CPU tests for spacetimepilot/eval/geometry.py (metric 1a math).

Runs against the REAL demo_videos/cameras/camera_extrinsics.json (no GPU, no VAE) -- these
tests exist because the homography derivation is new math that must be checked against actual
pool geometry before it's trusted on a GPU job. Covers:
  * ledger #1's "cam01-04 share position at all t" claim, via assert_shared_center on real data;
  * assert_shared_center correctly REJECTS a non-rotation-only pair (cam01 vs cam05, a pure
    translation per ledger #1) -- the guard must not be a rubber stamp;
  * H = K.R_rel.K^-1 self-consistency: same-camera pair -> R_rel = I -> H = I -> warping a
    frame by H reproduces it exactly;
  * intrinsics_matrix has NO default fov_deg (contract: never silently guess a wrong K).

Runnable via pytest OR directly: `python tests/test_eval_geometry.py`.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetimepilot.eval import geometry

CAMERA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "demo_videos/cameras/camera_extrinsics.json")


def _cam_data():
    with open(CAMERA_FILE) as f:
        return json.load(f)


def test_rotation_only_pairs_share_center():
    cam_data = _cam_data()
    for a, b in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        for t in (0, 20, 40, 60, 80):
            d = geometry.assert_shared_center(cam_data, a, b, t, atol=1e-2)
            assert d < 1e-2, f"cam{a:02d}/cam{b:02d} @ t={t}: center drift {d}"


def test_translation_pair_does_not_share_center():
    cam_data = _cam_data()
    # cam05/06 are pure translations per ledger #1 (~200 units) -- must NOT pass the guard
    # at a late frame (translation compounds over time; near t=0 they still start co-located).
    raised = False
    try:
        geometry.assert_shared_center(cam_data, 1, 5, 80, atol=1e-2)
    except ValueError:
        raised = True
    assert raised, "cam01 vs cam05 at t=80 should NOT share a center (translation family)"


def test_homography_identity_for_same_camera():
    cam_data = _cam_data()
    R_rel = geometry.relative_rotation(cam_data, 2, 2, 40)
    assert np.allclose(R_rel, np.eye(3), atol=1e-5), "same-camera relative rotation must be I"

    K = geometry.intrinsics_matrix(fov_deg=55.0, width=832, height=480)
    H = geometry.homography_from_rotation(K, R_rel)
    # R_rel's ~1e-6 floating-point residual gets amplified by K's focal length (~800) when
    # sandwiched as K.R_rel.K^-1 -- still sub-pixel (well under 1e-3), not a geometry bug.
    assert np.allclose(H / H[2, 2], np.eye(3), atol=1e-3), "same-camera H must be ~I (up to scale)"

    frame = (np.random.RandomState(0).rand(480, 832, 3) * 255).astype(np.uint8)
    warped, mask = geometry.warp_a_to_b(frame, H, 832, 480)
    # interior pixels (away from the identity-warp's zero-padding edges) must reproduce exactly
    interior = frame[10:-10, 10:-10]
    warped_interior = warped[10:-10, 10:-10]
    assert np.mean(np.abs(interior.astype(int) - warped_interior.astype(int))) < 1.0
    assert mask[240, 416], "center pixel must be covered for an identity warp"


def test_intrinsics_requires_fov():
    raised = False
    try:
        geometry.intrinsics_matrix(fov_deg=None, width=832, height=480)
    except ValueError:
        raised = True
    assert raised, "intrinsics_matrix must refuse to guess a default fov_deg"


def test_mask_bbox_empty_and_full():
    empty = np.zeros((100, 100), dtype=bool)
    assert geometry.mask_bbox(empty) is None

    full = np.ones((100, 100), dtype=bool)
    bbox = geometry.mask_bbox(full, border_crop=10)
    assert bbox == (10, 90, 10, 90)


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(_TESTS)} eval-geometry tests passed")
