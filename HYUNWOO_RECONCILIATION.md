# Reconciliation With Hyunwoo's Primary Sources

Purpose: lock the one place where the current plan/handoff diverged from what Hyunwoo
actually specified, before implementation. Sources of authority, in order:
`counterfactual.txt` (Hyunwoo's written framing), `transcript.txt` (the advising call),
Slack (quoted in `RESEARCH_CONTEXT.md` §4.5). `Problem Statement.txt` D2 is Akshata's
derivation and took the *other* branch; where it conflicts, Hyunwoo's written doc wins.

---

## THE HOLE (resolved): the marginalization was built on the wrong side

`CLAUDE_CODE_HANDOFF.md` (lines 75–90, 265) and `PLAN.md` Decision 0 build the
**marginal-teacher** branch:
`s_real = E_{v1 ~ p(v1|v2,v0)}[ ∇_{v2} log p(v2|v0,v1) ]` — average K frozen two-video
teacher calls over middles — with a **2-source** fake-score net conditioned on `(v0,v1)`.

Hyunwoo's written spec is the **direct-teacher / student-side-marginalization** branch.

`counterfactual.txt` (lines 40–43, verbatim):
```
marginalize v1 from p(v0,v1,v2)  =>  p'(v0,v2) = ∫ p(v2|v1,v0) p(v1|v0) p(v0) dv1
Now we have clean target p(v2|v0).
Now match D_KL( p'(v2|v0) || p(v2|v0) )   => Use DMD.
Novelty: (monte-carlo) marginalized version of DMD.
```
`transcript.txt` 37:56–38:44: "**P** ... freeze ... SPT ... 2-video generator model ...
**P_fake** would be the [aut]oregressive version."

Decision: **follow `counterfactual.txt`. The marginalization lives on the STUDENT side.**

---

## Resolved definitions (use these everywhere)

- **Teacher `p` (frozen) → `s_real` = ∇ log p(v2|v0).**
  A *single* released-SPT call, one source = the real `v0`, target = `v2`. No averaging
  over middles. This is Hyunwoo's "clean target p(v2|v0)". It is one-source, and strictly
  in-distribution (v0 is real footage).

- **Student `q_θ` = p'(v2|v0) = ∫ q_θ(v2|v0,v1) p(v1|v0) dv1.**
  The N-source-extended SPT conditioning on `(v0,v1)`. Its marginal over `v1` is realized
  by sampling middles `v1 ~ p(v1|v0)` (from the middle bank) and generating `v2|v0,v1`.
  This marginal is the thing DMD matches to the direct teacher.

- **Fake-score net → `s_fake` = ∇ log p'(v2|v0), via the MSE trick.**
  (`transcript.txt` 29:41–35:47; `counterfactual.txt` 57–66.) A **one-source** net
  conditioned on `v0` ONLY (v1 hidden), trained by denoising MSE on the student's `v2`
  samples. Least-squares with `v1` hidden returns the average over `v1` = the marginal
  score. Conditioning it on `(v0,v1)` would give a *conditional* score — the wrong object.

- **Novelty:** Monte-Carlo–marginalized DMD, marginalization on the student side
  (`counterfactual.txt` line 43). The marginal-teacher build discards this.

## Concrete corrections to Decision 0 / PLAN Rung 2 / handoff

1. `s_real`: one direct teacher call, `source = v0` — NOT a K-way average over middles.
2. Fake-score net is **1-source** (`source = v0`, hide `v1`) — NOT the 2-source `(v0,v1)`
   net in Decision 0.
3. **Only the STUDENT needs the N-source extension.** Teacher and fake-score net both stay
   one-source. This simplifies Rung A: the three DiTBlock edit sites are exercised by the
   student only.
4. Middle bank role: cached `v1 ~ p(v1|v0)` are Monte-Carlo samples for the *student's*
   marginalization and for training the fake-score net — not inputs to averaged teacher
   arrows.

## What is UNCHANGED (fork-independent)

- D1 DMD gradient, the detached surrogate, and the sign.
- Velocity→score conversion (`RESEARCH_CONTEXT.md` §7 / `flow_match.py`).
- `loss_student = mean( stop_grad((1-σ)/σ · (v_real − v_fake)) · x0_hat )`, now with
  `v_real` = direct teacher (source=v0), `v_fake` = 1-source marginal net.
- All Rung 0 / Rung 1 smoke-test scaffolding, freeze/unfreeze, gradient checkpointing.

## A bias that DISSOLVES under this branch

The prior-vs-posterior worry in `PLAN.md`/D2 §4.4 was specific to the marginal-teacher
branch (it wanted `v1 ~ p(v1|v2,v0)`). Here the student marginal is *defined* by
`p(v1|v0)`, so sampling `v1 ~ p(v1|v0)` and hiding it is the exact generative process —
the MSE optimum is the true marginal score, no posterior reweighting. One fewer
approximation to defend.

## Update from Hyunwoo's Slack (July)

The Slack thread **confirms** this branch verbatim (7:51–7:57 AM: "keep `q` the model to train,
`p` what you have"; `p(v2,v0)=p(v2|v0)p(v0)` direct teacher; `q_θ(v2,v0)=∫dv1 q_θ(v2|v1,v0)
p(v1|v0)p(v0)`; caveat "score of the diffused joint ≠ sum of conditional scores → must
estimate"). It also adds three concrete items:

1. **NEW — RoPE: feed explicit 3D token positions (supersedes "camera embedding only").**
   Architecture answer: "Just use pure DiT... they already have 3D RoPE. Some implementations put
   RoPE based on the array shape. **In that case you may have to explicitly feed the 3D position
   of each token.**" SPT's RoPE is array-shape-based (`torch.arange`, gathered by `:f`). So the
   crossing constraint **should ride on RoPE** by indexing `self.freqs` with an explicit per-token
   3D/temporal position instead of `arange`, *plus* keep the camera embedding. The earlier docs'
   hard claim "crossing cannot ride on RoPE; world info enters only via camera embedding" is
   retired. Exact positions/normalization = confirm Thursday.

2. **RESOLVED — crossing trajectories are heuristic.** "The camera trajectory needs to be
   heuristically chosen." Not learned, not training-stage-dependent. Pick fixed crossing
   trajectories by hand (choose `a_i, a_j` with an overlap point `a_{i;m}==a_{j;n}`).

3. **CONFIRMED + enriched — stitching (D3) is itself a marginalized-DMD alignment.** Jointly
   generate N videos, pick one frame each → `x = W[vid_1..vid_N]ᵀ` (`W` linear). Align the
   *marginal* `p(x)=∫ p_N(vid_{1:N}) p(x|vid_{1:N})` to the single-video distribution `p_1(x)`
   via DMD, using `s(x) = W·s(vid_1..vid_N)`. This is the anti-degeneracy force and reuses the
   joint score — exactly PLAN Rung 3.

## NOT holes — Hyunwoo's own open questions (leave open, do not "solve")

- Global consistency from pairwise windows: "assuming transitivity ... that is the
  question" (`transcript.txt` 41:27–41:38).
- Feasibility of the marginalization itself: "I don't know exactly how to do the
  marginalization ... I'm not sure it's possible" (29:48).
- RoPE mechanism: Hyunwoo verbally assumed RoPE carries world position (3:31–4:47); the
  code shows RoPE is sequence-index-based. His operational conclusion (camera embedding
  aligns views) still holds, and the plan already enforces crossing via the camera
  embedding — faithful as written.
