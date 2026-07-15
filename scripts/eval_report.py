"""Score B0 / B1 / checkpoint rollouts (from scripts/eval_rollout.py) against the Run-1
eval protocol (PLAN_DMD_TRAINING.md section 8, metric-1a as amended 2026-07-14).

Metric 1a (rotation-homography agreement, ESTIMATED-H form -- no intrinsics needed):
  For each eval tuple, every method is compared against the SAME reference: the cached bank
  middle v1 at v1_cam (a released-SPT independent view). At matched times t>0, SIFT-match
  v1's frame to the method's a2 frame, RANSAC-estimate the homography, and report
    1a_psnr      warp-PSNR under H_est on the valid-overlap region ("do the two views depict
                 one world related by SOME rotation homography");
    1a_inlier    RANSAC inlier fraction of the matches (consistency evidence);
    1a_disp      median inlier displacement in px -- the ANTI-COPY GUARD: a student that
                 copies v1 verbatim gets H_est ~= I and a perfect 1a_psnr, but ~0 px
                 displacement exposes it (genuine rotated views displace by tens of px);
    1a_dircos    cosine between observed mean displacement and the COMMANDED rotation's
                 predicted direction (obedience: direction is FOV-insensitive, magnitude is
                 not -- magnitude is meaningless here per the calibrate_fov.py finding that
                 SPT renders 1.2-2.2x the commanded rotation).
  Why estimated H and not commanded H: scripts/calibrate_fov.py (2026-07-14, 4 v0s) showed
  the teacher is not metrically camera-obedient, so a commanded-H score mostly measures
  inherited command disobedience, not cross-view consistency. See matching.py's docstring.

Metric 2 (quality guard):
  Laplacian sharpness of each method's a2 (fog detector, same convention as qc_cfg_compare)
  + LPIPS-to-teacher = LPIPS(method's a2, B0's a2) -- B0 IS the teacher's independent
  generation, so this is literally the plan's "LPIPS-to-teacher".

Run-1 success rule (plan section 8, printed as a note): 1a beats B0 and B1 on >=6/8 held-out
v0s; metric-2 within 10% of B1; not fog; PLUS (amendment) no copy-guard trip. Metric 1b
(epipolar, translating pairs) is out of scope here.

Usage:
  python -u scripts/eval_report.py --rollout_dir <out_dir from eval_rollout.py> \
      --methods b0 b1 --out report.json
"""

import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_video_test as svt  # noqa: E402
from spacetimepilot.training import middle_bank as mbk  # noqa: E402
from spacetimepilot.eval import geometry, quality, matching  # noqa: E402

FRAME_INDICES = (10, 20, 30, 40, 50, 60, 70, 80)  # t>0 only (frame 0 = shared sink, ledger #2);
# 8 frames (7/15, was 4): more chances for SIFT to find enough structure on soft generations.
SHARPNESS_STRIDE = 5
COPY_GUARD_PX = 5.0    # median displacement below this = suspicious (H_est ~= identity)
NOMINAL_FOV_DEG = 60.0  # for DIRECTION prediction only (direction is fov-insensitive)


def parse_args():
    p = argparse.ArgumentParser(description="Score eval rollouts: metric-1a (estimated-H) + metric-2")
    p.add_argument("--rollout_dir", required=True, help="--out_dir passed to eval_rollout.py")
    p.add_argument("--methods", nargs="+", default=["b0", "b1"],
                   help="subfolder tags under --rollout_dir to score (e.g. b0 b1, or b0 b1 step1000)")
    p.add_argument("--bank_dir", default="/orcd/scratch/orcd/014/akshatat/counterfactual_models/middle_bank_cfg1")
    p.add_argument("--camera_file", default=svt.DEFAULT_CAMERA_FILE)
    p.add_argument("--width", type=int, default=svt.WIDTH)
    p.add_argument("--height", type=int, default=svt.HEIGHT)
    p.add_argument("--out", default=None, help="write the full per-tuple report as json")
    return p.parse_args()


def load_frames(path, indices):
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(reader.get_data(i)) for i in indices]
    finally:
        reader.close()


def load_all_frames(path, stride):
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(f) for i, f in enumerate(reader) if i % stride == 0]
    finally:
        reader.close()


def find_middle_by_cam(bank_dir, v0_id, cam_idx):
    for idx in mbk.list_middles(bank_dir, v0_id):
        _, meta_path = mbk.middle_paths(bank_dir, v0_id, idx)
        if int(mbk.load_meta(meta_path)["cam_idx"]) == cam_idx:
            return idx
    return None


def metric_1a(cam_data, v1_path, v1_cam, a2_path, a2_cam, sift, matcher, width, height):
    """Estimated-H agreement between the v1 reference and a method's a2, per matched time."""
    v1_frames = load_frames(v1_path, FRAME_INDICES)
    a2_frames = load_frames(a2_path, FRAME_INDICES)
    psnrs, inliers, disps, dircos = [], [], [], []
    matched_frames = 0
    for fi, f_v1, f_a2 in zip(FRAME_INDICES, v1_frames, a2_frames):
        geometry.assert_shared_center(cam_data, v1_cam, a2_cam, fi)
        m = matching.sift_matches(f_v1, f_a2, sift, matcher)
        if m is None:
            continue
        H, mask = matching.estimate_homography(m[0], m[1])
        if H is None:
            continue
        matched_frames += 1
        stats = matching.displacement_stats(m[0], m[1], mask)
        inliers.append(stats["inlier_frac"])
        disps.append(stats["median_px"])
        R_cmd = geometry.relative_rotation(cam_data, v1_cam, a2_cam, fi)
        pred = geometry.predicted_center_displacement(R_cmd, NOMINAL_FOV_DEG, width, height)
        dircos.append(matching.direction_cosine(stats["mean_vec"], pred))
        warped, cov = geometry.warp_a_to_b(f_v1, H, width, height)
        bbox = geometry.mask_bbox(cov)
        if bbox is not None:
            psnrs.append(geometry.overlap_psnr(f_a2, warped, bbox))
    n = len(FRAME_INDICES)
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else None,
        "inlier_frac": float(np.mean(inliers)) if inliers else None,
        "median_disp_px": float(np.median(disps)) if disps else None,
        "dir_cos": float(np.nanmean(dircos)) if dircos else None,
        "matched_frame_frac": matched_frames / n,
        "copy_flag": bool(disps and np.median(disps) < COPY_GUARD_PX),
    }


def main():
    args = parse_args()
    with open(args.camera_file) as f:
        cam_data = json.load(f)
    sift, matcher = matching.make_sift()

    manifests = {}
    for m in args.methods:
        path = os.path.join(args.rollout_dir, m, "manifest.json")
        with open(path) as f:
            manifests[m] = json.load(f)["entries"]
    tuple_ids = [e["tuple_id"] for e in manifests[args.methods[0]]]
    by_method_tuple = {m: {e["tuple_id"]: e for e in entries} for m, entries in manifests.items()}

    b0_available = "b0" in manifests
    rows = []
    for tid in tuple_ids:
        ref = by_method_tuple[args.methods[0]][tid]
        v0_id, v1_cam, a2_cam = ref["v0_id"], ref["v1_cam"], ref["a2_cam"]
        v1_idx = find_middle_by_cam(args.bank_dir, v0_id, v1_cam)
        if v1_idx is None:
            print(f"[warn] {tid}: no cached v1 at cam{v1_cam:02d} for {v0_id} -- skipping")
            continue
        v1_path, _ = mbk.middle_paths(args.bank_dir, v0_id, v1_idx)

        row = {"tuple_id": tid, "v0_id": v0_id, "v1_cam": v1_cam, "a2_cam": a2_cam}
        b0_frames_full = load_all_frames(by_method_tuple["b0"][tid]["a2_video"], SHARPNESS_STRIDE) \
            if b0_available else None

        for m in args.methods:
            a2_path = by_method_tuple[m][tid]["a2_video"]
            one_a = metric_1a(cam_data, v1_path, v1_cam, a2_path, a2_cam,
                              sift, matcher, args.width, args.height)
            row[f"{m}_1a_psnr"] = one_a["psnr"]
            row[f"{m}_1a_inlier"] = one_a["inlier_frac"]
            row[f"{m}_1a_disp"] = one_a["median_disp_px"]
            row[f"{m}_1a_dircos"] = one_a["dir_cos"]
            row[f"{m}_copy_flag"] = one_a["copy_flag"]
            row[f"{m}_sharpness"] = quality.video_sharpness(load_all_frames(a2_path, SHARPNESS_STRIDE))
            if b0_available and m != "b0":
                m_frames = load_all_frames(a2_path, SHARPNESS_STRIDE)
                row[f"{m}_lpips_to_teacher"] = quality.lpips_distance(m_frames, b0_frames_full)
        rows.append(row)
        print(f"scored {tid} ({v0_id})")

    metric_cols = [f"{m}_1a_psnr" for m in args.methods] \
        + [f"{m}_1a_inlier" for m in args.methods] \
        + [f"{m}_1a_disp" for m in args.methods] \
        + [f"{m}_1a_dircos" for m in args.methods] \
        + [f"{m}_sharpness" for m in args.methods] \
        + [f"{m}_lpips_to_teacher" for m in args.methods if m != "b0"]

    print("\n" + "=" * 110)
    header = ["tuple_id", "v0_id"] + metric_cols
    print("  ".join(f"{h:>20}" for h in header))
    for row in rows:
        cells = []
        for h in header:
            v = row.get(h)
            cells.append(f"{v:>20.3f}" if isinstance(v, float) else f"{str(v if v is not None else '-'):>20}")
        print("  ".join(cells))
    print("=" * 110)

    summary = {}
    for h in metric_cols:
        vals = [row[h] for row in rows if isinstance(row.get(h), float)]
        summary[h] = float(np.mean(vals)) if vals else None
    print("\nmean over held-out tuples:")
    for k, v in summary.items():
        print(f"  {k:<28} {v:.3f}" if v is not None else f"  {k:<28} -")

    copy_trips = {m: sum(bool(row.get(f"{m}_copy_flag")) for row in rows) for m in args.methods}
    print(f"\ncopy-guard trips (median disp < {COPY_GUARD_PX:g} px): {copy_trips}")

    if "b0" in args.methods and len(args.methods) >= 2:
        print("\nRun-1 success-rule note (plan section 8 as amended 7/14; per-v0 >=6/8 rule needs "
              "the per-tuple rows above, this is the mean-only view):")
        for m in [x for x in args.methods if x != "b0"]:
            b0_mean, m_mean = summary.get("b0_1a_psnr"), summary.get(f"{m}_1a_psnr")
            if b0_mean is not None and m_mean is not None:
                print(f"  {m} vs b0: mean 1a PSNR {m_mean:.2f} vs {b0_mean:.2f} "
                      f"-> {'BEATS' if m_mean > b0_mean else 'does NOT beat'} b0"
                      f"{'  [COPY-GUARD TRIPPED -- psnr not trustworthy]' if copy_trips.get(m) else ''}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "summary": summary, "methods": args.methods,
                       "copy_guard_px": COPY_GUARD_PX, "nominal_fov_for_direction": NOMINAL_FOV_DEG},
                      f, indent=2)
        print(f"\nwrote full report -> {args.out}")


if __name__ == "__main__":
    main()
