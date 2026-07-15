"""Metric 1a: rotation-homography agreement (PLAN_DMD_TRAINING.md section 8, ledger #2).

cam01-04 are rotation-only pan/tilt cameras that share their position at every timestep
(ledger #1, direct measurement over all 45 pairs x 81^2 time pairs). Two matched-time frames
rendered from two such cameras are therefore related by an EXACT homography
H = K . R_rel . K^-1 (pure rotation about a shared center -- no depth or scene-motion
dependence): warp one onto the other, score PSNR/LPIPS on the valid-overlap region. This is
the primary Run-1 consistency metric -- it tests whether the student's 2-source rollout
(v1 fixed, a2 = G(v0,v1,a2)) agrees with v1 where their fields of view overlap, the way two
views of ONE consistent 3D world must. Frame 0 is excluded everywhere here: it's the shared
sink/anchor every method copies from v0 verbatim (ledger #2) -- zero discriminative power.

INTRINSICS: RESOLVED 2026-07-14, but not the way section 13 #7 anticipated. Empirical
calibration (scripts/calibrate_fov.py, intra-video pairs, 4 v0s x cam01-04) showed there is
NO stable K to recover: the released SPT renders rotations at ~1.2-2.2x the commanded
magnitude with a scene-/segment-dependent implied FOV (IQR 59-97 deg), while individual
videos ARE locally homography-consistent where matches are dense. Consequently metric-1a is
scored with an ESTIMATED homography (spacetimepilot/eval/matching.py) and needs no K at all.
The commanded-H path below remains for tests, controls, and direction (not magnitude)
predictions -- and `fov_deg` stays a REQUIRED argument with no default ANYWHERE in this
module: any commanded-H number is convention-dependent, so the convention must be stated
explicitly at every call site.
"""

import cv2
import numpy as np

from spacetimepilot.dataset.utils import process_camera_trajectory


def c2w(cam_data, cam_idx, frame_idx):
    """4x4 camera-to-world matrix for one (cam, frame) via the repo's own parsing/coord-fix path."""
    cams = process_camera_trajectory(cam_data, [frame_idx], cam_idx)
    return cams[0].c2w_mat


def camera_center(cam_data, cam_idx, frame_idx):
    return c2w(cam_data, cam_idx, frame_idx)[:3, 3]


def assert_shared_center(cam_data, cam_a, cam_b, frame_idx, atol=1e-3):
    """Guard ledger #1's claim (cam01-04 share position at all t) -- fail loud, not silent,
    if a supposedly rotation-only pair doesn't share a center at this frame (would break the
    pure-rotation homography assumption)."""
    ca = camera_center(cam_data, cam_a, frame_idx)
    cb = camera_center(cam_data, cam_b, frame_idx)
    d = float(np.linalg.norm(ca - cb))
    if d > atol:
        raise ValueError(
            f"cam{cam_a:02d} and cam{cam_b:02d} centers differ by {d:.4f} at frame {frame_idx} "
            f"(> {atol}) -- not a shared-center pair; metric-1a's pure-rotation assumption breaks")
    return d


def relative_rotation(cam_data, cam_a, cam_b, frame_idx):
    """R_rel taking a camera-A-frame direction to the same direction expressed in camera-B's frame."""
    Ra = c2w(cam_data, cam_a, frame_idx)[:3, :3]
    Rb = c2w(cam_data, cam_b, frame_idx)[:3, :3]
    return Rb.T @ Ra


def relative_rotation_frames(cam_data, cam_idx, frame_a, frame_b):
    """R_rel between two TIMES of one rotation-only trajectory (frame_a's camera frame ->
    frame_b's camera frame). Valid for the same reason as the cross-camera version: cam01-04
    keep a static center over time, so intra-video frame pairs are pure-rotation related."""
    Ra = c2w(cam_data, cam_idx, frame_a)[:3, :3]
    Rb = c2w(cam_data, cam_idx, frame_b)[:3, :3]
    return Rb.T @ Ra


def assert_static_center(cam_data, cam_idx, frame_a, frame_b, atol=1e-3):
    """Guard that one trajectory's center does not move between two frames (rotation-only)."""
    ca = camera_center(cam_data, cam_idx, frame_a)
    cb = camera_center(cam_data, cam_idx, frame_b)
    d = float(np.linalg.norm(ca - cb))
    if d > atol:
        raise ValueError(
            f"cam{cam_idx:02d} center moved {d:.4f} between frames {frame_a}->{frame_b} "
            f"(> {atol}) -- not rotation-only; intra-video homography assumption breaks")
    return d


def intrinsics_matrix(fov_deg, width, height):
    """Pinhole K, centered principal point, HORIZONTAL fov_deg, square pixels. See module
    docstring -- fov_deg has no default and must be passed explicitly by the caller."""
    if fov_deg is None:
        raise ValueError(
            "fov_deg is required and has no default (intrinsics convention is an open "
            "external prereq -- PLAN_DMD_TRAINING.md section 13 #7). Do not guess.")
    fx = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0],
                      [0.0, fx, height / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)


def homography_from_rotation(K, R_rel):
    """H mapping image-A pixel coords to image-B pixel coords (up to scale)."""
    return K @ R_rel @ np.linalg.inv(K)


def predicted_center_displacement(R_rel, fov_deg, width, height):
    """Where the commanded rotation moves the image-center pixel: (dx, dy) in px.

    Used ONLY for its DIRECTION (matching.direction_cosine): the direction of the center
    pixel's motion under a small rotation is insensitive to fov_deg, while the magnitude is
    not -- so callers must not compare magnitudes computed here against observations.
    fov_deg still has no default (module rule); pass a stated nominal value.
    """
    K = intrinsics_matrix(fov_deg, width, height)
    H = homography_from_rotation(K, R_rel)
    c = np.array([width / 2.0, height / 2.0, 1.0])
    p = H @ c
    return float(p[0] / p[2] - c[0]), float(p[1] / p[2] - c[1])


def warp_a_to_b(frame_a, H, width, height):
    """Warp frame_a into image-B's plane; returns (warped, coverage_mask).

    coverage_mask marks pixels of the OUTPUT that have real (in-bounds) source content --
    cv2.warpPerspective leaves everything else at borderValue=0, which is NOT "real black
    pixels", so callers must intersect this with anything downstream, not treat 0 as data.
    """
    warped = cv2.warpPerspective(np.asarray(frame_a), H, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.full(np.asarray(frame_a).shape[:2], 255, dtype=np.uint8)
    coverage = cv2.warpPerspective(ones, H, (width, height), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, coverage > 0


def mask_bbox(mask, border_crop=16):
    """Tight bounding box of `mask`, shrunk by `border_crop` px (plan: "large overlap;
    border-crop" -- a homography warp's edges are its least reliable region). Returns None
    if nothing survives (degenerate / near-zero overlap)."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = mask.shape
    y0, x0 = max(y0 + border_crop, 0), max(x0 + border_crop, 0)
    y1, x1 = min(y1 - border_crop, h), min(x1 - border_crop, w)
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def overlap_psnr(frame_b, warped_a, bbox):
    y0, y1, x0, x1 = bbox
    a = warped_a[y0:y1, x0:x1].astype(np.float64)
    b = np.asarray(frame_b)[y0:y1, x0:x1].astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def crop_overlap(frame, bbox):
    y0, y1, x0, x1 = bbox
    return np.asarray(frame)[y0:y1, x0:x1]
