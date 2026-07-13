"""QC compare for the cfg side-bank: is cfg=1 sharp enough, or is it fog?

For each (v0, middle idx) it reads the matched cfg=1 and cfg=3 middles plus the source v0
and reports a *sharpness* proxy = mean variance of a 4-neighbour discrete Laplacian over
sampled frames (higher = sharper; the classic blur/fog detector). We compare against v0's
own sharpness because absolute Laplacian variance is scene-dependent.

Decision rule (PLAN A4 QC gate): cfg=1 is what the marginalization math actually wants
(cfg>1 tilts the mixing distribution — ledger #6), so we PREFER cfg=1 unless it's degraded.
Flag "FOG" when a cfg=1 middle's sharpness < 0.5 * its v0. If cfg=1 passes on all/most
middles -> use cfg=1 (principled). If it fails -> cfg=3 is the quality hedge (accept the bias).

CPU-only, login-node safe. Reads mp4 via imageio.
"""

import argparse
import os
import sys

import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spacetimepilot.training import middle_bank as mbk  # noqa: E402

V0_VIDEO_DIR = "demo_videos/videos"


def laplacian_var(frame_rgb):
    """Variance of a 4-neighbour discrete Laplacian on the luma channel (blur detector)."""
    g = frame_rgb.astype(np.float32)
    g = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    lap = (-4.0 * g[1:-1, 1:-1]
           + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def video_sharpness(path, frame_stride=5):
    """Mean Laplacian-variance over every `frame_stride`-th frame of an mp4."""
    if not os.path.exists(path):
        return None
    vals = []
    reader = imageio.get_reader(path)
    try:
        for i, frame in enumerate(reader):
            if i % frame_stride == 0:
                vals.append(laplacian_var(np.asarray(frame)))
    finally:
        reader.close()
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg1_dir", required=True)
    ap.add_argument("--cfg3_dir", required=True)
    ap.add_argument("--v0_ids", nargs="+", required=True)
    ap.add_argument("--num_middles", type=int, default=4)
    ap.add_argument("--fog_ratio", type=float, default=0.5,
                    help="flag FOG when cfg middle sharpness < fog_ratio * v0 sharpness")
    args = ap.parse_args()

    header = f"{'v0':<10} {'mid':>3} {'v0_sharp':>10} {'cfg1':>10} {'cfg3':>10} {'c1/v0':>7} {'c3/v0':>7}  flag"
    print(header)
    print("-" * len(header))

    c1_ratios, c3_ratios, fog_c1, fog_c3 = [], [], 0, 0
    for v0 in args.v0_ids:
        v0_sharp = video_sharpness(os.path.join(V0_VIDEO_DIR, f"{v0}.mp4"))
        for idx in range(args.num_middles):
            p1, _ = mbk.middle_paths(args.cfg1_dir, v0, idx)
            p3, _ = mbk.middle_paths(args.cfg3_dir, v0, idx)
            s1, s3 = video_sharpness(p1), video_sharpness(p3)
            r1 = (s1 / v0_sharp) if (s1 and v0_sharp) else float("nan")
            r3 = (s3 / v0_sharp) if (s3 and v0_sharp) else float("nan")
            flags = []
            if s1 is not None and v0_sharp and s1 < args.fog_ratio * v0_sharp:
                flags.append("FOG:cfg1"); fog_c1 += 1
            if s3 is not None and v0_sharp and s3 < args.fog_ratio * v0_sharp:
                flags.append("FOG:cfg3"); fog_c3 += 1
            if not np.isnan(r1):
                c1_ratios.append(r1)
            if not np.isnan(r3):
                c3_ratios.append(r3)
            print(f"{v0:<10} {idx:>3} {(v0_sharp or 0):>10.1f} {(s1 or 0):>10.1f} "
                  f"{(s3 or 0):>10.1f} {r1:>7.2f} {r3:>7.2f}  {' '.join(flags)}")

    print("-" * len(header))
    m1 = np.mean(c1_ratios) if c1_ratios else float("nan")
    m3 = np.mean(c3_ratios) if c3_ratios else float("nan")
    print(f"mean sharpness ratio vs v0   cfg1={m1:.2f}   cfg3={m3:.2f}")
    print(f"FOG flags (< {args.fog_ratio:g}x v0)   cfg1={fog_c1}   cfg3={fog_c3}   "
          f"(of {len(args.v0_ids) * args.num_middles} middles)")
    if fog_c1 == 0:
        print("VERDICT hint: cfg=1 passes the fog gate -> use cfg=1 (principled, unbiased marginal).")
    else:
        print("VERDICT hint: cfg=1 shows fog -> consider cfg=3 as a sharpness hedge (accepts mixing-dist bias).")


if __name__ == "__main__":
    main()
