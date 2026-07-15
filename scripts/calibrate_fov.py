"""Empirically calibrate the demo pool's FOV from frozen-teacher generations (resolves the
metric-1a intrinsics gap, PLAN_DMD_TRAINING.md section 13 #7).

Why this is sound: the model is conditioned on EXTRINSICS ONLY (compute_pose_embedding
flattens a 3x4 relative pose -- no K anywhere in the conditioning), so "the pool's FOV" is
not a recorded constant anywhere; it is an empirical property of what the generator renders.
For a rotation-only camera (cam01-04: static center, ledger #1), two frames related by a
KNOWN pure rotation obey H(fov) = K(fov) . R_rel . K(fov)^-1, leaving fov as the single
unknown: SIFT-match the frames, sweep fov, and the argmin of a robust reprojection error IS
the teacher's rendering convention -- measured, with a residual, instead of guessed.

MODES (this matters -- the first calibration attempt failed for a subtle reason):
  intra (default)  Match frame t vs frame t+delta of the SAME generated middle. One video =
                   one internally-consistent world, and cam01-04 rotate in place over time,
                   so intra-video pairs are true pure-rotation pairs. Moving objects violate
                   the static-world assumption per pair, but they land in the truncated tail
                   of the objective; the static background carries the fov signal.
  cross            Match matched-time frames of DIFFERENT middles (cam a vs cam b). Kept as
                   a diagnostic only: different middles are INDEPENDENT samples of p(v1|v0)
                   -- different counterfactual worlds (making them agree is the project's
                   whole goal!) -- so cross errors are dominated by world divergence, not
                   projection. Measured on 4 v0s: cross ~106 px median vs flipped-control
                   719 px: direction convention right, but no shared world. The GAP between
                   intra and cross error is itself the baseline inconsistency the DMD
                   training is supposed to shrink.

Calibrating on teacher generations and later scoring methods with the fitted K is not
circular: K becomes a global constant shared by every method (B0/B1/student), and a method
that disobeys the commanded rotation cannot be rescued by any single K.

Diagnostics built in (each failure mode has a distinct signature):
  * per-sample argmin spread  -- tight = a real shared convention exists; wide = the teacher
                                 is not projectively consistent even within one video ->
                                 metric-1a's premise is shaky, better to learn that NOW;
  * flipped-rotation control  -- err(R_rel.T) >> err(R_rel) confirms the direction convention;
                                 if the flip fits BETTER, fix the axis convention, don't calibrate;
  * inlier fraction at 5 px   -- how much of the image actually obeys the fitted H.

CPU-only (SIFT + a 1-D sweep), login-node safe, bounded (caps on v0s/pairs/frames).

  python scripts/calibrate_fov.py \
      --bank_dir /orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank_cfg1 \
      --write config/eval/intrinsics.json
"""

import argparse
import glob
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_video_test as svt  # noqa: E402
from spacetimepilot.training import middle_bank as mbk  # noqa: E402
from spacetimepilot.eval import geometry  # noqa: E402
from spacetimepilot.eval.matching import make_sift, sift_matches, MIN_MATCHES  # noqa: E402

ROTATION_CAMS = (1, 2, 3, 4)  # shared-center family (ledger #1)
CROSS_FRAME_INDICES = (20, 40, 60, 80)      # cross mode: matched-time, t>0 (ledger #2)
INTRA_FRAME_PAIRS = ((0, 40), (20, 60), (40, 80), (0, 80))  # intra mode: same video, known R(t)
TRUNC_PX = 20.0   # truncated-mean cap: moving-object matches saturate here instead of voting
INLIER_PX = 5.0


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate demo-pool FOV from teacher generations")
    p.add_argument("--bank_dir", required=True)
    p.add_argument("--camera_file", default=svt.DEFAULT_CAMERA_FILE)
    p.add_argument("--mode", default="intra", choices=["intra", "cross"])
    p.add_argument("--max_v0s", type=int, default=6)
    p.add_argument("--fov_lo", type=float, default=20.0)
    p.add_argument("--fov_hi", type=float, default=120.0)
    p.add_argument("--coarse_step", type=float, default=1.0)
    p.add_argument("--width", type=int, default=svt.WIDTH)
    p.add_argument("--height", type=int, default=svt.HEIGHT)
    p.add_argument("--write", default=None,
                   help="write the calibrated value + provenance to this json (config/eval/intrinsics.json)")
    return p.parse_args()


def rotation_middles_by_v0(bank_dir, max_v0s):
    """{v0_id: {cam_idx: mp4_path}} for v0s whose FULL rotation family (cam01-04) is cached."""
    by_v0 = {}
    for meta_path in sorted(glob.glob(os.path.join(bank_dir, "*__mid*.json"))):
        meta = mbk.load_meta(meta_path)
        cam = int(meta["cam_idx"])
        if cam in ROTATION_CAMS:
            v_path, _ = mbk.middle_paths(bank_dir, meta["v0_id"], int(meta["idx"]))
            by_v0.setdefault(meta["v0_id"], {})[cam] = v_path
    complete = {v0: cams for v0, cams in by_v0.items() if len(cams) == len(ROTATION_CAMS)}
    return dict(sorted(complete.items())[:max_v0s])


def load_frames(path, indices):
    reader = imageio.get_reader(path)
    try:
        return {i: np.asarray(reader.get_data(i)) for i in indices}
    finally:
        reader.close()


def collect_intra(groups, cam_data, sift, matcher):
    """Samples from (t, t+delta) pairs WITHIN each middle video (one video = one world)."""
    samples, rejected = [], 0
    needed = sorted({f for pair in INTRA_FRAME_PAIRS for f in pair})
    for v0, cams in groups.items():
        for cam, path in sorted(cams.items()):
            frames = load_frames(path, needed)
            for fa, fb in INTRA_FRAME_PAIRS:
                geometry.assert_static_center(cam_data, cam, fa, fb, atol=1e-2)
                m = sift_matches(frames[fa], frames[fb], sift, matcher)
                if m is None:
                    rejected += 1
                    continue
                samples.append({"v0": v0, "cams": (cam,), "frames": (fa, fb),
                                "pts_a": m[0], "pts_b": m[1],
                                "R_rel": geometry.relative_rotation_frames(cam_data, cam, fa, fb)})
    return samples, rejected


def collect_cross(groups, cam_data, sift, matcher):
    """Samples from matched-time frames ACROSS middles (diagnostic only -- see module docstring)."""
    samples, rejected = [], 0
    for v0, cams in groups.items():
        frames = {c: load_frames(p, CROSS_FRAME_INDICES) for c, p in cams.items()}
        for ai in range(len(ROTATION_CAMS)):
            for bi in range(ai + 1, len(ROTATION_CAMS)):
                a, b = ROTATION_CAMS[ai], ROTATION_CAMS[bi]
                for t in CROSS_FRAME_INDICES:
                    geometry.assert_shared_center(cam_data, a, b, t, atol=1e-2)
                    m = sift_matches(frames[a][t], frames[b][t], sift, matcher)
                    if m is None:
                        rejected += 1
                        continue
                    samples.append({"v0": v0, "cams": (a, b), "frames": (t, t),
                                    "pts_a": m[0], "pts_b": m[1],
                                    "R_rel": geometry.relative_rotation(cam_data, a, b, t)})
    return samples, rejected


def project(H, pts):
    """Apply homography H to Nx2 points."""
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def match_errors(fov, sample, width, height, flip=False):
    K = geometry.intrinsics_matrix(fov, width, height)
    R = sample["R_rel"].T if flip else sample["R_rel"]
    H = geometry.homography_from_rotation(K, R)
    return np.linalg.norm(project(H, sample["pts_a"]) - sample["pts_b"], axis=1)


def sample_error(fov, sample, width, height, flip=False):
    """Truncated-mean reprojection error: moving-object outliers saturate at TRUNC_PX."""
    return float(np.minimum(match_errors(fov, sample, width, height, flip), TRUNC_PX).mean())


def aggregate_error(fov, samples, width, height, flip=False):
    return float(np.mean([sample_error(fov, s, width, height, flip) for s in samples]))


def main():
    args = parse_args()
    with open(args.camera_file) as f:
        cam_data = json.load(f)

    groups = rotation_middles_by_v0(args.bank_dir, args.max_v0s)
    if not groups:
        raise SystemExit("no v0 with a complete cam01-04 rotation family in the bank yet -- "
                         "wait for the bank build to cover at least one v0")
    print(f"mode={args.mode}  calibrating on {len(groups)} v0s: {sorted(groups)}")

    sift, matcher = make_sift()
    collect = collect_intra if args.mode == "intra" else collect_cross
    samples, rejected = collect(groups, cam_data, sift, matcher)
    print(f"matched samples: {len(samples)} (rejected {rejected} with <{MIN_MATCHES} matches)")
    if len(samples) < 10:
        raise SystemExit("too few matched samples to calibrate -- need more bank coverage")

    # Coarse sweep -> refine around the minimum at 0.1 deg.
    fovs = np.arange(args.fov_lo, args.fov_hi + 1e-9, args.coarse_step)
    errs = [aggregate_error(f, samples, args.width, args.height) for f in fovs]
    i = int(np.argmin(errs))
    lo, hi = fovs[max(i - 1, 0)], fovs[min(i + 1, len(fovs) - 1)]
    fine = np.arange(lo, hi + 1e-9, 0.1)
    fine_errs = [aggregate_error(f, samples, args.width, args.height) for f in fine]
    j = int(np.argmin(fine_errs))
    best_fov, best_err = float(fine[j]), float(fine_errs[j])

    # Diagnostics.
    per_sample_argmin = [float(fovs[int(np.argmin([sample_error(f, s, args.width, args.height)
                                                   for f in fovs]))]) for s in samples]
    spread = (float(np.percentile(per_sample_argmin, 25)),
              float(np.percentile(per_sample_argmin, 50)),
              float(np.percentile(per_sample_argmin, 75)))
    flipped_err = aggregate_error(best_fov, samples, args.width, args.height, flip=True)
    all_errs = np.concatenate([match_errors(best_fov, s, args.width, args.height) for s in samples])
    inlier_frac = float((all_errs < INLIER_PX).mean())
    med_inlier = float(np.median(all_errs[all_errs < TRUNC_PX])) if (all_errs < TRUNC_PX).any() else float("nan")

    print("\n================= FOV CALIBRATION RESULT =================")
    print(f"  mode / samples             : {args.mode} / {len(samples)}")
    print(f"  best fov (horizontal)      : {best_fov:.1f} deg")
    print(f"  truncated-mean error       : {best_err:.2f} px (cap {TRUNC_PX:g})")
    print(f"  median non-truncated error : {med_inlier:.2f} px")
    print(f"  inlier fraction (<{INLIER_PX:g} px)  : {inlier_frac:.2%}")
    print(f"  per-sample argmin 25/50/75 : {spread[0]:.1f} / {spread[1]:.1f} / {spread[2]:.1f} deg")
    print(f"  flipped-rotation control   : {flipped_err:.2f} px (should be near the {TRUNC_PX:g} cap)")
    iqr = spread[2] - spread[0]
    ok = (best_err < 10.0 and iqr < 12.0 and flipped_err > 1.5 * best_err and inlier_frac > 0.25)
    print(f"  VERDICT: {'PASS -- a consistent rendering convention exists' if ok else 'WEAK -- inspect before trusting metric-1a'}")
    print("==========================================================")

    if args.write:
        if not ok:
            print(f"NOT writing {args.write}: calibration verdict is WEAK (see diagnostics above)")
            sys.exit(3)
        payload = {
            "fov_deg": round(best_fov, 1),
            "convention": "horizontal fov, centered principal point, square pixels "
                          "(spacetimepilot/eval/geometry.py intrinsics_matrix)",
            "provenance": {
                "method": f"empirical ({args.mode}-video): SIFT matches under known pure rotation; "
                          "fov = argmin of truncated-mean reprojection error under "
                          "H = K.R_rel.K^-1 (scripts/calibrate_fov.py)",
                "bank_dir": args.bank_dir,
                "v0s": sorted(groups),
                "n_samples": len(samples),
                "truncated_mean_err_px": round(best_err, 2),
                "median_err_px": round(med_inlier, 2),
                "inlier_frac_5px": round(inlier_frac, 3),
                "argmin_iqr_deg": round(iqr, 1),
                "flipped_control_px": round(flipped_err, 2),
            },
        }
        os.makedirs(os.path.dirname(args.write) or ".", exist_ok=True)
        with open(args.write, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote calibrated intrinsics -> {args.write}")


if __name__ == "__main__":
    main()
