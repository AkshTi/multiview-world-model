# PLAN A — DMD Finetune (the research method)
## Multi-View-Consistent World Model via MC-Marginalized DMD — complete build sheet
### Revised after the 2026-07-06 Slack sync with Hyunwoo. Companion: `PLAN_PLAIN_FINETUNE.md` (Track P, fast-iteration proxy).

**Provenance tags** — the defensibility spine:
- **(H)** = Hyunwoo's spec/answer verbatim (`counterfactual.txt`, transcript, Slack incl. **7/6 sync**).
- **(F)** = hole we filled; each has a rationale + the cheap experiment that retires it.
- **(O)** = open in Hyunwoo's own framing; measured, never assumed.

## 0. What the 7/6 Slack sync changed (delta summary)

1. **(H) Numerics bug found: we were doing pure-bf16 training.** Correct recipe at 1.3B: **fp32
   trainable params + bf16 autocast (standard AMP)** — no stochastic-rounding, no fp32-master
   bolt-on. Replaces `MasterAdamW`. Also ~2× speed expected.
2. **(H) RoPE world-time idea (R1) demoted.** Wan2.1 is "very overfitted to the rope frequency
   it was trained on"; shared world-time positions "can catastrophically break the model."
   **R0 (sequential positions) is THE convention.** Run 0 A/B canceled.
3. **(H) Extrapolated positions 42–62 for the second source: "for now let's just try it."**
   RoPE is relative — only relative distances matter; distances >41 are the OOD part. Fallback
   ladder if pathological: **windowed attention** (mask pairs with RoPE distance >41 — which is
   also D4's pairwise-window idea at attention level), then **YaRN-style frequency rescaling**.
4. **(H) NEW hard constraint — attention-sink latent:** Wan uses the **first VAE latent as the
   attention sink**; dropping it degrades badly. Invariant: every video-like input (v0, middles,
   the stitched strip, any window) keeps a *real first latent frame in position 0*.
5. **(H) Stitching rule confirmed:** "any slice of frames from multiple videos that have
   continuous trajectory of (camera, time) coordinate should be fine." Our crossing-switch
   construction satisfies it.
6. **(H) Dataset swap approved:** MultiCamVideo ("Kling is a renowned company so the data
   quality should be good"). HF path confirmed live: `KlingTeam/MultiCamVideo-Dataset`.
7. **(H) Compute reality:** schadenfreude (CSAIL group server) is also H200-scarce. Post-AMP,
   re-measure peak and target **any 80 GB GPU (H100/A100-80/H200)**. FSDP2 unnecessary at 1.3B.
8. **(H) New optional track:** finetune **your own mini-SPT from pretrained Wan** at
   (T/2, H/2, W/2) — ~64× cheaper attention — for iteration velocity ("you can't run the full
   experiment every time"). → Split out as **`PLAN_PLAIN_FINETUNE.md`** (Track P).

## 0b. What the 7/8 self-audit changed (delta summary)

New tag: **(V)** = verified by direct measurement on this repo/data (code cited).

1. **(V) Pool geometry measured** (`demo_videos/cameras/camera_extrinsics.json`, all 45 pairs ×
   81² time pairs): all 10 trajectories start at ONE shared pose (frame-0 spread = 0.0) and
   **never re-cross at t>0**. Families: cam01–04 rotation-only pan/tilt (position static,
   ±10–20°), cam05–06 pure translations (~200 units), cam07–10 arcs (14–30°, 100–206 units).
2. **Consequence A — metric 1a redesigned.** "Crossing-frame agreement" was degenerate: the
   only shared (camera,time) point is frame 0, which every view copies from v0 (sink+anchor) —
   zero discriminative power. Replaced by **rotation-homography agreement** (§8): cam01–04
   share their POSITION at ALL times, so matched-time frames between two pan/tilt views are
   related by `H = K·R_rel·K⁻¹` — exact, GT-free, **depth- and motion-independent** (pure
   rotation about a shared center). Needs the same one-time intrinsics check as 1b.
3. **Consequence B — "crossing" re-scoped to "overlap".** `crossing_cam_types` cycles the pool;
   with no t>0 crossings, middles are same-origin diverging views, and ledger #7's
   crossing-vs-random ablation had no distinction to test. Re-scoped to **high-vs-low
   viewpoint-overlap** middles (§4 A5, ledger #7). True-crossing pool extension (mirrored
   S-curves that intersect mid-path at equal world time) = (O) backlog, OOD risk, raise w/ Hyunwoo.
4. **Fixed-bank bias acknowledged (ledger #3 corrected).** K middles/step drawn from a bank of
   only 4 frozen middles per v0 = distilling a fixed 4-atom mixture — **bias, not variance** —
   and the K-ablation confounded per-step K with bank coverage. Fix: decouple **K_bank=8** from
   **K_step∈{1,2,4}** (bank one-time cost; frozen teacher ⇒ extendable, never stale).
5. **Git hygiene is a hard gate now (§7 item 0).** Everything since 7/2 (training/, tests/,
   scripts/, both plans, the N-source model edits) is uncommitted ⇒ SHA-stamped configs are
   currently meaningless. Also `logs/` needs .gitignore (done 7/8). `assets/*` "modified" status
   is a harmless pre-existing phantom diff (`.gitattributes` declares LFS tracking for
   `*.png`/`*.gif`, but HEAD holds the real binaries, never actually run through LFS clean —
   verified via `git cat-file -p HEAD:assets/teaser.gif`, full GIF data, not a pointer) —
   correction of an earlier wrong read from `git diff --stat` alone; leave untouched, do not
   stage.
6. **Run-4 action space specified (§5 B2):** MultiCamVideo uses scene-native sibling
   trajectories normalized to SPT convention — the demo pool's absolute units (positions
   ~(3390,1380,240)) can't be assumed transferable. One-time convention check + per-pool
   geometry report (same script as (V) above).
7. Two ledger additions: direct fake-lag monitor (replay-set MSE, #4) and fake-net trainable
   set = student freeze mask tagged (F) (#15).

## 1. Context

Rungs 0–6 GPU-validated; Rung 7 (D3) math core CPU-tested. Between here and the first real
finetune: the AMP numerics refactor, final conventions (now mostly (H)-settled), data at scale,
production middle bank, real training entrypoint, pre-registered eval. Constraints: 80 GB-class
GPU, ~24 h jobs, resume-first; no gated CamXTime — pilot on 61 local demo videos, scale on
MultiCamVideo. Verified: SPT ckpt on scratch; `build_middle_bank.py` resumable; demo corpus =
61 videos + captions + cam pool (10 trajectories, coincide at frame 0) + `src_cam/`.

## 2. Formal objective (fixed)

**(H)** v0 = real source video; a = action = (pool camera trajectory, world-time pattern);
v1 ~ p(v1|v0) from frozen released SPT; v2 = new view:

```
min_θ  E_{v0,a2} [ D_KL( p'_θ(v2|v0,a2) ‖ p(v2|v0,a2) ) ],
p'_θ(v2|v0,a2) = ∫ q_θ(v2|v0,v1,a2) p(v1|v0) dv1     (marginalization on the STUDENT side)
```
DMD arrow `A = s_real − s_fake`; surrogate `loss_G = mean(stop_grad((1−σ)/σ·(v_real−v_fake))·x0_hat)`.
- **s_real (H):** ONE frozen teacher call, source=v0, action a2. Never sees v1.
- **s_fake (H):** 1-source net on (v0,a2), **v1 hidden**, denoising-MSE on student samples —
  MSE optimum = marginal score. Both scores condition on a2; only v1 is marginalized.
- **Student (H):** only N-source model; one-step generator `v2̂ = z − σ_T·v_pred`, σ_T=sigmas[0].
- Velocity→score: `s = −(x_t+(1−σ)v)/σ`; σ from scheduler table; slice to `[:,:,:21]` first.

## 3. Approximation ledger

| # | Item | Status | Handling |
|---|---|---|---|
| 1 | DMD gradient + detached surrogate | (H) exact up to score est. | unit tests green |
| 2 | Marginal score via v1-hidden MSE | (H) exact at MSE optimum | est. error monitored via loss_D |
| 3 | Finite-K MC from a FIXED bank | per-step K = variance; **finite frozen bank = bias** (7/8 fix) | K_bank=8 ≥ 2×max K_step; Run 3 varies K_step {1,2,4} at fixed K_bank — deconfounded; bank extendable (frozen teacher ⇒ never stale) |
| 4 | Fake-net lag (non-stationary) | standard DMD; loss_D alone is a weak proxy (also rises when student broadens — desired early) | 1:1 cadence; 2:1 trigger needs BOTH rising loss_D (200-step window) AND rising replay-set MSE (fake net scored on frozen student samples from 500 steps back, logged in JSONL) |
| 5 | One-step generator at σ_T | (F) DMD-style choice | few-step w/ last-step grad = backlog knob |
| 6 | Middles at **cfg=1** ≡ p(v1|v0) | (F); cfg>1 tilts the mixing dist | fallback cfg=3 recorded; teacher target unaffected. **Unasked — ratify** |
| 7 | ~~Crossing~~ **Overlap** heuristic; informativeness of v1 | (V) pool has NO t>0 crossings — middles are same-origin diverging views; (O) informativeness | overlap-swap ablation: high-overlap middles (pan/tilt pairs — same position all t) vs low-overlap (divergent arcs, e.g. cam09 vs cam10); overlap score = mean pose proximity over time, ranks all pool pairs |
| 8 | RoPE: R0 sequential; 42–62 extrapolated | **(H) "let's just try it"** — relative-distance OOD >41 is the risk | Run-1 watch signals; fallback: windowed attn → YaRN. R1 world-time = demoted backlog experiment |
| 9 | Strip score s_x = W·s_v, disjoint W | exact (WWᵀ=I, CPU-proven) | — |
| 10 | Strip plausibility under p₁ | **(H) rule:** continuous (camera,time) trajectory suffices | crossing-switch construction + sink invariant; teacher-strip sanity sample |
| 11 | Strip s_fake = same fake net, strip diet | (F) MSE-trick on student-induced strips | detachment tested; blur-ablation gate. **Unasked — ratify** |
| 12 | Pairwise ⇒ global consistency | (O) transitivity bet | Run 5 drift-vs-chain-distance |
| 13 | Numerics: fp32 params + bf16 autocast | **(H) 7/6 recipe** | replaces MasterAdamW; re-smoke + re-measure peak |
| 14 | 63-latent-frame attention window (> SPT's 42, ≫ Wan's 21) | (O) — SPT itself survived 42; 63 unproven | Run-1 watch signals + #8 fallback ladder |
| 15 | Fake-net trainable set = student freeze mask | (F) — shift lives in the same subspace as the student's; init=released keeps iter-0 arrow ≡ 0 sanity. DMD refs often train more of the fake net | if replay-set MSE plateaus high / arrow stuck: widen fake trainable set (recorded knob). **Unasked — ratify** |

## 4. Decisions locked

### A1. RoPE / positions → **R0 sequential (H)**; OOD-window fallback ladder
Positions stay released-style: `[target 0–20][src0 21–41][src1 42–62]`. RoPE is relative (H);
>41 (target↔src1) is the OOD band SPT never finetuned. **Proceed (H: "just try it").** Watch
signals in Run 1: arrow blow-up, attention collapse on/off the src1 block, v1-invariance of
samples. Fallbacks in order: (1) **windowed attention** — mask pairs with RoPE distance >41
(v0↔v2̂ indirect only; this IS D4's pairwise-window structure, so the fallback previews Rung 8);
(2) **YaRN-style rescaling** of OOD frequencies. R1 (shared world-time axis) = low-priority
backlog experiment, catastrophic-break risk flagged (H).

### A2. Middle-as-source camera → same encoder path, same world frame, src slot (F)
Bank `cam_idx` → pool extrinsics → identical `compute_pose_embedding → cam_encoder` path, src
list slot. *Retire (CPU):* degenerate equivalence (v1 on v0's trajectory ⇒ bit-identical src
embeddings) + shape/dtype contract through `build_batch`.

### A3. Strip (D3) → S1 crossing-switch blocks ((H)-compliant) + S2 shared fake net (F)
- **S1:** strip = 21 latent frames, contiguous per-view blocks switching only at crossing
  points — satisfies Hyunwoo's continuous-(camera,time) rule verbatim. Teacher sees an ordinary
  1-source call (source=v0, stitched per-frame cam/time tracks). **Sink invariant (H):** strip
  position 0 = v0's REAL first latent frame — v0 block first, includes latent 0, always.
  `WWᵀ=I` preserved; `stitch.py` unchanged.
- **S2 (unasked — ratify):** same 1-source fake net also trains on strip samples (conditioned
  as the teacher sees them). Anchors constant ⇒ gradient reaches only v2̂ rows (tested).
- **λ schedule:** 0 for 500 steps → linear ramp over 500 to λ* with stitched-arrow norm ≈ 0.2×
  main arrow (band 0.1–0.3). *Retire:* CPU tests + teacher-strip sanity + blur pregate
  (2×500 steps λ=0 vs λ*; fail ⇒ halve λ*).

### A4. Middle bank → cfg=1, 20 steps, **K_bank=8** (7/8: decoupled from K_step), forward time (F — ratify cfg)
`--cfg_scale 1 --num_inference_steps 20 --num_middles 8` (NB: script defaults are 50/5.0 —
**must override**), seeds `seed0+k`, pool cycle (8 of 10 cams per v0 ⇒ spans all four trajectory
families). ~1.5–2 min/middle est (measure on video 1). **K_bank vs K_step:** the bank is the
finite sample of p(v1|v0) (ledger #3 bias, shrinks with K_bank); K_step middles are drawn from
it per step w/o replacement. K_bank=8 = 2×max K_step keeps the Run-3 ablation honest. Frozen
teacher ⇒ the bank never goes stale and extends incrementally if #3 bias shows.
**QC gates:** no-fog (Laplacian var ≥ 0.5× v0 + eyeball 10), crossing-validation report,
reload+encode → `(1,16,21,60,104)`. cfg=5 side-bank (8 v0s) for the middle-quality ablation.

### A5. Action space + sampling (F)
a2 = (pool trajectory minus v1's, forward time). Prompt = v0's caption. v0 uniform over train
set; **K_step middles w/o replacement from the K_bank=8 bank**; all RNG **keyed by global step**
(exact resume). Sink corollary: never slice away any video's first latent in batch assembly.
**Overlap bookkeeping (7/8):** log the (v1_cam, a2) pool-pair overlap score with every step so
the Run-3 overlap-swap ablation (ledger #7) and the "student ignores v1" diagnosis can slice
metrics by overlap for free. **(O) backlog, raise Thursday:** extend the pool with a mirrored
S-curve pair (same start pose, intersect mid-path at equal world time) — the only construction
that yields TRUE t>0 crossings under the shared-start constraint; unknown whether novel smooth
trajectories are in-distribution for SPT ⇒ gate on a teacher sanity sample before any use.

### A6. Rollout for N views (F — ratify)
N=2: v1 = fresh frozen-SPT middle (fixed seeds), v2̂ = G^EMA(v0,v1,a2). N>2 (Run 5):
**v0-anchored chain** `v_{i+1}=G(v0,v_i,a_{i+1})`; sliding-window `(v_{i-1},v_i)` is the
recorded alternative (Run 5 ablation; dovetails with the windowed-attention fallback).

### A7. Numerics + optimization (H 7/6 — supersedes MasterAdamW)
**fp32 trainable params + `torch.autocast(bf16)`** on student & fake nets; frozen teacher stays
bf16 (+`.eval()`, no_grad). Retire `master.py` from the loop (keep tests as regression
history); expect ≈2× step-time improvement; re-measure peak (re-smoke before budgeting).
AdamW: student lr 1e-5, fake lr 1e-4, β=(0.9,0.999), wd=0, grad-clip 1.0.
σ: `id ~ U{⌈0.02N⌉..⌊0.98N⌋}`. EMA 0.999 on student trainables; eval always from EMA. Fake init
= deepcopy released (iter-0 arrow ≡ 0 = sanity signal). Teacher score cfg=1 (**unasked —
ratify**; guided-teacher = backlog knob). batch=1; AC on (trainable nets `.train()`);
AC-offload + `offload_teacher` in reserve; `expandable_segments`; `python -u`. No FSDP2 (H).

## 5. Data plan
- **B1 Pilot:** demo 61 → **53 train / 8 held-out** (seed-0 split, committed
  `config/data/pilot_split.json`); captions `metadata.csv`; pool `cameras/camera_extrinsics.json`.
- **B2 Scale (H-approved): MultiCamVideo** (`KlingTeam/MultiCamVideo-Dataset`; Apache-2.0,
  ungated; 13.6k UE5 scenes × 10 synced cams, **81 frames** = SPT-native, 1280×1280@15fps,
  extrinsics JSON). **500-scene subset** (~12 GB); v0 = cam01 per scene; other 9 views reserved
  as bonus-eval GT. Preprocess: center-crop → 738×1280 → resize 480×832; 81 frames as-is.
  Captions: VLM on middle frame, ≤2 sentences, one GPU job; spot-check 20; template fallback.
  **Action space for Run 4 (7/8 fix — was unspecified):** actions/middles use the scene's own
  9 sibling trajectories, normalized to SPT's convention (relative to v0=cam01's frame-0 pose)
  — NOT the demo pool, whose absolute units (positions ~(3390,1380,240)) have no defined scale
  in MultiCamVideo's UE5 worlds. One-time checks before Stage B: (a) extrinsics convention
  (row-vector layout, translation row, units) matches what `process_camera_trajectory` expects;
  (b) rerun the pool-geometry report (same script as the 7/8 audit) on a few scene pools —
  crossings/overlap structure decides which eval metrics apply at scale. Demo-pool-rescaled
  trajectories = recorded fallback if scene-native ones misbehave.
- **B3 Eval-only:** CamXTime benchmark (`zhening/CamxTime`, ~5.5 GB): `eval_input` +
  `full_grid_renders`. Never trained on. (Access form: optional, via Hyunwoo, for the paper.)
- **B4 Caching:** per-v0 VAE latents + text embeddings on scratch; middles keep mp4 + latents.
  Pilot ≪10 GB; scale 30–60 GB; ckpts ~10 GB × (last 2 + best).
  **Quota (verified 7/7): SCRATCH 148.6/1024 GB — ample. POOL is 100% FULL — write NOTHING to
  pool; all artifacts → `/orcd/scratch/orcd/014/akshatat/`.** HOME 112/200 GB — code only.

## 6. Middle-bank builds
- **Stage A:** 53×8 = 424 middles ≈ 14 GPU-h (7/8: K_bank=8) — one resumable job + QC report
  (hard gate). cfg=5 side-bank (32).
- **Stage B (post-Run-2):** 500×(2·K*) where K* = Run-3 winning K_step (K*=2 ⇒ 2000 ≈ 67 GPU-h
  → ~8 h lanes; K*=4 ⇒ 4000 ≈ 134 GPU-h — budget before committing).

## 7. Training system (coding backlog, in order)
Reuse: `steps.py`, `score.py`, `freeze.py`, `latents.py`, `middle_bank.py`, `stitch.py`,
smokes, `build_middle_bank.py`, `compute_metrics_camxtime.py`, sbatch templates.
0. **Git hygiene (7/8 — HARD GATE before any long run, incl. the Stage-A bank; DONE 7/8):**
   the entire implementation since 7/2 was uncommitted, so SHA-stamped configs described
   nothing. (a) `logs/` added to `.gitignore` (`__pycache__/` was already covered);
   (b) `assets/*` left untouched — its "modified" status is a harmless pre-existing phantom
   diff from `.gitattributes` LFS rules vs. real binaries already in HEAD, not real corruption
   (see §0b item 5) — staging it would actually convert them to LFS pointers, an unrelated and
   unintended change; (c) committed `training/`, `tests/`, `scripts/`, model edits, both PLAN
   files, reconciliation docs; (d) pushed to `AkshTi/multiview-world-model`. §13.3 holds by
   routine going forward.
1. **AMP refactor (new #1 priority):** fp32 trainables + autocast in `dmd_step_k`; drop
   MasterAdamW from the loop; re-smoke + re-measure peak → sets the GPU class.
2. `training/data.py` — V0Dataset, A5 samplers, step-keyed RNG.
3. `build_batch` formalized from `rung6_smoke` (A1 R0 positions, A2, sink invariant).
4. `scripts/precompute_latents.py`.
5. `scripts/train_dmd.py` + `config/train/run1.yaml` — clip, EMA, λ hook, JSONL metrics
   (|arrow| mean & per-middle, loss_G, loss_D, grad norms, σ, step time, peak mem) + config &
   git-SHA header; 4 fixed (v0,v1,a2,z) sample mp4s / 250 steps; ckpt / 250 steps (θ, φ, opt,
   EMA, RNG, step; last 2 + best); full resume; **step-5 peak-memory auto-abort**.
6. **Resume kill-test** (hard gate).
7. `scripts/eval_baselines.py` + `scripts/eval_consistency.py` (§8).
8. Rung 7 wiring per A3 + test extensions (post-Run-1).

## 8. Eval protocol (pre-registered; frozen before Run 1)
- **Held-out:** 8 demo v0s + 4 CamXTime scenes recurring; full 32-scene suite at run ends.
- **Baselines BEFORE Run 1:** B0 = released SPT independent views; B1 = student-at-init.
- **Metrics:** **1a Rotation-homography agreement** (7/8 redesign — replaces crossing-frame
  agreement, which was (V)-degenerate: the pool's only shared (camera,time) point is frame 0,
  copied from v0 by every method). cam01–04 are rotation-only ⇒ any two views generated under
  pan/tilt cams share their camera CENTER at all t; matched-time frames are related by
  `H = K·R_rel·K⁻¹` — exact, GT-free, depth- and scene-motion-independent. Warp one frame onto
  the other, PSNR/LPIPS on the valid-overlap mask (±10–20° ⇒ large overlap; border-crop),
  student vs B0/B1. **Eval tuples pin (v1_cam, a2) pairs from cam01–04 for 1a**; translating
  pairs are covered by 1b. **1b Epipolar consistency** (point tracks + known pool extrinsics;
  median px error, inlier% <2 px; the one-time intrinsics-convention check serves 1a AND 1b).
  **2 Quality guard:** Laplacian sharpness + LPIPS-to-teacher (FVD at Run 4). **3 Strip quality**
  (Rung 7 gate). **4 No-regression:** `compute_metrics_camxtime.py` + generated-pair vs
  GT-grid-pair features.
- Rollout per A6, EMA weights, fixed seeds. Cadence: iter-0 (=baselines) → every 1k steps →
  full at end.
- **Run-1 success rule:** 1a AND 1b beat B0 and B1 on ≥6/8 held-out v0s; metric-2 within 10% of
  B1; not fog. Less ⇒ diagnose (incl. #14 RoPE signals), don't scale.

## 9. Run schedule + timeline (80 GB GPU, 24 h jobs)
| Run | What | Notes |
|---|---|---|
| ~~0~~ | ~~RoPE A/B~~ | **CANCELED** — R0 locked by (H); hours go to the AMP re-smoke |
| 1 | Pilot DMD, no stitching: N=2, K=2, 53 v0s, 3000 steps | AMP recipe; measure s/step at start; step-500 gates (arrow band, loss_D bounded, non-fog) + RoPE-OOD watch — kill early on violation |
| 2 | + stitching: blur pregate (2×500) then 3000 steps, λ ramp | sink invariant enforced in strip builder |
| 3 | Ablations: K_step∈{1,2,4} ×1500 at fixed K_bank=8; overlap-swap (high vs low); cfg-5 bank arm | ledger #3/#6/#7 |
| 4 | Scale: MultiCamVideo-500, winning config, 10–20k steps | post-Run-2 gate |
| 5 | Rung 8 / D4: pairwise windows on N=4 chains, drift curve | windowed-attention machinery may exist via A1 fallback |
| P | **→ `PLAN_PLAIN_FINETUNE.md`** (parallel lane, L40S-class) | mini-SPT proxy for fast iteration |

Budget through Run 3 ≈ 60–90 GPU-h (80 GB class) + ~22 GPU-h utility (Stage-A bank doubled to
K_bank=8). Calendar (part-time):
W1 = AMP refactor + backlog 2–7 + Stage-A bank; W2 = baselines + Run 1; W3 = Rung 7 + Run 2 +
Run 3 start; W4+ = Run 4. Estimates, not commitments.

## 10. Compute targets & memory budget
- **Any 80 GB GPU** (H100/A100-80/H200) after AMP re-measure — widens both queues.
- **schadenfreude.csail.mit.edu** (Hyunwoo's group server): also H200-scarce (H); unreachable
  from ORCD (port 22 refused — different network; reach it from laptop/CSAIL VPN, likely via
  `login.csail.mit.edu` jump). Adoption checklist: GPU ≥80 GB (else offload_teacher + K=1),
  ~100 GB storage + restage path (cannot mount /orcd/scratch), CUDA ≥12.1 env mirror
  (conda-pack), tmux/nohup + resume harness (no SLURM assumed).

### Memory budget & cautions (per Hyunwoo 7/6)
- **Hard rule: measured peak ≤ 90% of card** (72 GB on 80 GB). Measured at: AMP re-smoke
  (K=1, K=2), blur pregate (strip term adds teacher/fake strip forwards), and the **step-5
  auto-abort assert** in `train_dmd.py` (die immediately, not at hour 20).
- Reference peaks (pre-AMP): 46.7 GB (K=1+offload) / 52.6 (K=2) / 60.5 (masters). AMP expected
  similar-or-lower — the re-smoke number is the only one that counts.
- Standing rules: `expandable_segments` always; K via gradient accumulation ⇒ **peak
  K-independent — never batch middles**; one live autograd graph at a time; VAE resident only
  during encode/decode then offloaded (`load_models_to_device` gotcha — bit us twice); text
  encoder never loaded in training; sample decodes under no_grad, VAE freed after; EMA bf16
  (CPU-resident fallback if within 2 GB of budget); per-step peak-mem in JSONL.
- Escalation ladder: offload_teacher (−2.6 GB) → AC-offload (`save_on_cpu`) → EMA to CPU →
  windowed attention (also ~2× attention FLOPs cut) → K=1 → last resort H200-only.
- **Not** doing (H): pure-bf16 training, FSDP2, bf16 downcasting as a memory fix.

## 11. Risk register
| Risk | Signal | Response |
|---|---|---|
| RoPE-OOD at 63 frames (#14) | arrow blow-up; attention collapse on src1; v1-invariance | windowed attention → YaRN (A1 ladder) |
| Sink-latent violation | sudden quality cliff in any slicing path | CPU assert in build_batch/stitch: every input keeps real latent 0 |
| Fake net lags | loss_D AND replay-set MSE rising 200 steps (either alone is ambiguous — ledger #4) | cadence 2:1 → fake lr ×2 → widen fake trainable set (ledger #15) |
| Student ignores v1 | per-middle arrows identical AND v1-swap invariant | raise K_step; slice metrics by logged overlap score (A5); Run 3 overlap-swap |
| Gray fog pre-stitch | sharpness dives <1500 steps | front-load Rung 7 |
| cfg=1 middles mushy | bank QC | cfg=3 fallback, recorded |
| AMP peak surprises | re-smoke measure | §10 ladder; worst case H200-only |
| Teacher-strip OOD | strip sanity sample garbage | shrink blocks / switches nearer crossings |
| Step time ≫ est | Run-1 start measurement | halve steps, never loosen gates |
| Preemption / 24 h | — | resume kill-test = hard gate |
| Dull samples (teacher cfg=1) | quality flat | guided-teacher knob (backlog) |
| Iteration too slow at full res | wall-clock per experiment | Track P (`PLAN_PLAIN_FINETUNE.md`) |

## 12. Remaining ratification list
Q1/Q2/dataset/numerics answered 7/6. Still to ratify (async fine): **S2** (strip fake score via
same net), **cfg=1 middles**, **teacher cfg=1**, **A6 rollout**. Added 7/8: **metric-1a
homography redesign** (touches the pre-registered eval — must settle BEFORE freeze), **crossing
→ overlap re-scope + S-curve pool-extension idea** (ledger #7, A5), **K_bank=8** (A4),
**MultiCamVideo scene-native actions** (B2), **fake-net trainable set** (ledger #15).
Logistics: CamXTime form co-sign (optional).

## 13. External prerequisites (outside the code, easy to forget)
1. **Env pinning:** `conda env export` → `env/counterfactual.yml` in repo; new deps to install
   & pin: `lpips`, a matcher/tracker for metric 1b (kornia-LoFTR = simplest install), VLM
   captioner (Qwen2.5-VL-class via transformers — check compat with torch 2.4.1) — install
   BEFORE freezing eval code.
2. **CSAIL access test (user, from laptop/VPN):** `ssh atiwari7@schadenfreude.csail.mit.edu`,
   then `nvidia-smi -L`, `df -h`. Only then run the §10 checklist.
3. **Git hygiene:** commit + push before every long run; run configs embed the SHA.
4. **Downloads to scratch (never pool):** MultiCamVideo subset (~12 GB), CamXTime eval (5.5 GB).
5. **Thursday agenda:** §12 ratification list + Track P blessing.
6. **Logging:** JSONL primary (no internet assumed on compute nodes); wandb offline-sync optional.
7. **One-time:** verify intrinsics convention for the epipolar metric (MultiCamVideo ships
   focal lengths; demo pool = SPT convention).
8. **POOL cleanup** (100% full) — hygiene, not blocking.

## 14. Verification

### Pre-flight audit — DONE 2026-07-07 (all load-bearing claims checked against code/data/env)
- ✅ SPT ckpt on scratch (`SpaceTimePilot_hf/SpacetimePilot_1.3B_v1.ckpt`).
- ✅ Camera pool = exactly 10 trajectories, 81 frames extrinsics; `src_cam/video_*_extrinsics.npy`.
- ✅ `frame_positions` hook live (`spacetimepilot.py:212,232-235`), default = released arange ⇒
  R0 costs zero code.
- ✅ `stitch.py` API supports contiguous blocks incl. latent 0 ⇒ S1 + sink invariant, no rewrite.
- ✅ Bank meta already stores cam_idx/camera_file/time_pattern/seed/steps/cfg/caption.
- ✅ `freeze.py` trainable set matches plan.
- ✅ Scheduler exposes `sigmas` (`set_timesteps(training=…)`, shift=3.0); `sample_training_timestep` exists.
- ✅ AMP refactor contained: no autocast in `training/` today; touchpoints = load-cast site,
  MasterAdamW wiring, autocast ctx in `dmd_step_*`.
- ✅ `KlingTeam/MultiCamVideo-Dataset` live: 13.6k scenes, 10 cams, 81 frames, 1280², Apache-2.0, ungated.
- ✅ 7 CPU test files; `build_batch` in rung5/6 smokes to formalize.
- ⚠️ POOL quota 100% full — scratch-only discipline stated as invariant.

### 7/8 self-audit addenda (measured)
- **(V)** Pool geometry: 10 trajectories, one shared frame-0 pose, NO t>0 crossings. Families:
  cam01–04 rotation-only (±10–20°, position static), cam05–06 translate (~200 units), cam07–10
  arcs (14–30°, 100–206 units). Basis for the 1a redesign + overlap re-scope (§0b).
- ⚠️ Repo uncommitted since 7/2 (67cbefd) — §7 item 0 is a hard gate before any long run.
- ⚠️ `assets/{concept-diagram.png,logo.png,teaser.gif}` shrunk to ~131 B = git-LFS pointer
  artifacts, not real edits — restore via `git checkout -- assets/`.

### Ongoing gates (strict order)
AMP re-smoke (peak + 30-iter stability) → bank QC → resume kill-test → iter-0 baselines →
Run-1 step-500 gates (incl. RoPE watch) → Run-1 success rule → teacher-strip sanity + blur
pregate → Run 2. Config + git SHA in every JSONL header; eval tuples + metric code frozen
before Run 1; all randomness step-keyed.
