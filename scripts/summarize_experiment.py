"""Collect the whole Run-1 experiment into ONE summary (scripts/*.sbatch call this last).

Reads every eval report (report_*.json written by eval_report.py) + the training metrics
(run1/metrics.jsonl) and emits eval/pilot_baselines/SUMMARY.md: the consistency-vs-training-
step curve as a table, the B0/B1/student comparison, and -- critically -- the HONEST paired
analysis, not the raw subset means. The subset-mean trap (means computed over different
tuples per method because metric-1a returns null where SIFT can't match) burned us once; this
script computes the triple-paired comparison so the headline can't mislead.

CPU-only, login-node safe. Idempotent: re-run anytime; it summarizes whatever reports exist.

  python scripts/summarize_experiment.py \
      --eval_dir /orcd/scratch/orcd/014/akshatat/counterfactual_models/eval/pilot_baselines \
      --metrics  /orcd/scratch/orcd/014/akshatat/counterfactual_models/runs/run1/metrics.jsonl
"""

import argparse
import glob
import json
import os
import re
import statistics as st


def parse_args():
    p = argparse.ArgumentParser(description="Summarize the Run-1 experiment into one markdown")
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--metrics", default=None, help="run1 metrics.jsonl (training curve); optional")
    p.add_argument("--out", default=None, help="default: <eval_dir>/SUMMARY.md")
    return p.parse_args()


def load_reports(eval_dir):
    """All report_*.json -> {method: {tuple_id: row}} merged across reports (latest wins)."""
    by_method = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "report_*.json"))):
        rep = json.load(open(path))
        for row in rep.get("rows", []):
            for m in rep.get("methods", []):
                if f"{m}_1a_psnr" in row or f"{m}_sharpness" in row:
                    by_method.setdefault(m, {})[row["tuple_id"]] = row
    return by_method


def step_of(method):
    """Sort key: baselines first (b0<b1), then step<N> ascending, else name."""
    if method == "b0": return (0, 0)
    if method == "b1": return (0, 1)
    m = re.match(r"step(\d+)", method)
    return (1, int(m.group(1))) if m else (2, method)


def num(row, key):
    v = row.get(key)
    return v if isinstance(v, (int, float)) else None


def coverage_means(by_method):
    """Per-method mean over WHATEVER tuples that method has (the naive view -- labeled as such)."""
    out = {}
    for m, rows in by_method.items():
        for metric in ("1a_psnr", "1a_inlier", "1a_dircos", "sharpness", "lpips_to_teacher"):
            vals = [num(r, f"{m}_{metric}") for r in rows.values()]
            vals = [v for v in vals if v is not None]
            out[(m, metric)] = (st.mean(vals), len(vals)) if vals else (None, 0)
    return out


def paired_vs_untrained(by_method):
    """The honest comparison: each trained checkpoint vs B1 (same architecture, only training
    differs) on the tuples where BOTH have a valid 1a_psnr. Returns per-method paired stats."""
    if "b1" not in by_method:
        return {}
    b1 = by_method["b1"]
    res = {}
    for m, rows in by_method.items():
        if not m.startswith("step"):
            continue
        deltas, wins = [], 0
        for tid, row in rows.items():
            a, b = num(b1.get(tid, {}), "b1_1a_psnr"), num(row, f"{m}_1a_psnr")
            if a is None or b is None:
                continue
            deltas.append(b - a)
            wins += (b > a)
        if deltas:
            res[m] = {"n": len(deltas), "wins": wins,
                      "mean_delta": st.mean(deltas), "verdict": _verdict(deltas, wins)}
    return res


def _verdict(deltas, wins):
    n = len(deltas)
    frac = wins / n
    md = st.mean(deltas)
    if n < 4:
        return "too few paired tuples to call"
    if frac >= 0.75 and md > 1.0:
        return "improves over untrained"
    if frac <= 0.25 and md < -1.0:
        return "WORSE than untrained"
    return "within noise (trends up)" if md > 0 else "within noise"


def training_curve(metrics_path):
    if not metrics_path or not os.path.exists(metrics_path):
        return None
    steps = []
    for line in open(metrics_path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in d and "arrow_abs" in d:
            steps.append(d)
    if not steps:
        return None
    last = steps[-1]
    arrows = [d["arrow_abs"] for d in steps]
    return {"n_steps": last["step"] + 1, "last_arrow": last["arrow_abs"],
            "mean_arrow": st.mean(arrows), "last_lossD": last.get("loss_D"),
            "peak_gb": max(d.get("peak_gb", 0) for d in steps),
            "mean_step_s": st.mean(d.get("step_time_s", 0) for d in steps)}


def main():
    args = parse_args()
    out = args.out or os.path.join(args.eval_dir, "SUMMARY.md")
    by_method = load_reports(args.eval_dir)
    methods = sorted(by_method, key=step_of)
    cov = coverage_means(by_method)
    paired = paired_vs_untrained(by_method)
    curve = training_curve(args.metrics)

    L = []
    L.append("# Run 1 — multi-view DMD pilot: results summary\n")
    L.append("_Auto-generated by scripts/summarize_experiment.py. Numbers are honest paired "
             "comparisons where it matters; naive subset means are labeled as such._\n")

    if curve:
        L.append("## Training (run1)\n")
        L.append(f"- steps completed: **{curve['n_steps']}**")
        L.append(f"- DMD arrow |s_real−s_fake|: mean {curve['mean_arrow']:.3f}, "
                 f"last {curve['last_arrow']:.3f} (nonzero + bounded = healthy)")
        L.append(f"- last loss_D: {curve['last_lossD']:.3f} | peak {curve['peak_gb']:.1f} GB | "
                 f"{curve['mean_step_s']:.0f}s/step\n")

    L.append("## Cross-view consistency vs training step\n")
    L.append("### The honest comparison: trained vs UNTRAINED (B1), paired per tuple")
    L.append("_Same 2-source architecture and rollout; only DMD training differs. Only tuples "
             "where both have a valid homography match are counted._\n")
    if paired:
        L.append("| checkpoint | paired n | trained wins | mean ΔPSNR (dB) | verdict |")
        L.append("|---|---|---|---|---|")
        for m in sorted(paired, key=step_of):
            p = paired[m]
            L.append(f"| {m} | {p['n']} | {p['wins']}/{p['n']} | {p['mean_delta']:+.2f} | {p['verdict']} |")
        L.append("")
    else:
        L.append("_No trained-checkpoint reports found yet._\n")

    L.append("### Naive per-method means (DIFFERENT tuple subsets per method — do not compare directly)\n")
    L.append("| method | 1a_psnr (n) | inlier | dircos | sharpness | lpips→teacher |")
    L.append("|---|---|---|---|---|---|")
    for m in methods:
        def cell(metric):
            v, n = cov.get((m, metric), (None, 0))
            return f"{v:.2f} ({n})" if v is not None and metric == "1a_psnr" else (f"{v:.3f}" if v is not None else "—")
        L.append(f"| {m} | {cell('1a_psnr')} | {cell('1a_inlier')} | {cell('1a_dircos')} "
                 f"| {cell('sharpness')} | {cell('lpips_to_teacher')} |")
    L.append("")

    # LPIPS-to-teacher trend (full coverage -> trustworthy) for step checkpoints
    lp = [(m, cov[(m, "lpips_to_teacher")][0]) for m in methods
          if m != "b0" and cov.get((m, "lpips_to_teacher"), (None, 0))[0] is not None]
    if len(lp) >= 2:
        L.append("## Distillation convergence (LPIPS-to-teacher, full coverage = trustworthy)\n")
        L.append("_Lower = student output closer to the teacher distribution; monotone ↓ = DMD working._\n")
        L.append("  ".join(f"{m}={v:.3f}" for m, v in lp))
        mono = all(lp[i][1] >= lp[i + 1][1] for i in range(len(lp) - 1))
        L.append(f"\n\n**{'Monotonic ↓ — distillation is converging.' if mono else 'Non-monotonic — inspect.'}**\n")

    L.append("## Honest caveats (carry these to any discussion)\n")
    L.append("- **B0 confound:** B0 renders at 20 denoise steps; the student is one-step. "
             "Warp-PSNR partly rewards crispness, so \"beat B0\" is not apples-to-apples. The "
             "trained-vs-B1 pairing above is the fair test.")
    L.append("- **Two-view, rotation-only.** cam01–04 pairs. N>2 global consistency and "
             "translating/parallax pairs (metric-1b, epipolar) are not built yet.")
    L.append("- **Estimated homography, not commanded:** released SPT over-rotates 1.2–2.2× vs "
             "command with no stable FOV (scripts/calibrate_fov.py), so 1a fits H from the "
             "images. This is a standalone finding worth reporting.")
    L.append("- Metric-1a coverage is ~10–14/16 tuples; a null tuple means SIFT couldn't match, "
             "not zero consistency.\n")

    with open(out, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {out}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
