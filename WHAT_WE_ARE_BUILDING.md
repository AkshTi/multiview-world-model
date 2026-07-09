# What We Are Building — and How to Defend It

A from-scratch explanation of the project, in plain language first, then the math, then a
defense guide. Everything here is grounded in Hyunwoo's own sources (`counterfactual.txt`,
the meeting transcript, his Slack) — see `HYUNWOO_RECONCILIATION.md` for provenance. Nothing
in the *method* is invented here; the only opinions are in the "how to defend" section, which
is clearly marked.

---

## 1. The one-sentence version

We are fine-tuning a camera-controllable video model so that the several videos it invents
from a single real video all look like views of **one shared 3D world**, and we do it with a
distillation trick (DMD) because no multi-view ground-truth data exists.

---

## 2. The problem, concretely

SpaceTimePilot (SPT) takes one real video `v0` and a camera/time instruction, and generates a
new view of that same scene:

```
v1 = SPT(v0, "orbit left, freeze time at frame 40")
v2 = SPT(v0, "orbit right")
```

Each output looks fine on its own. But `v1` and `v2` don't agree with each other about the
world: a chair sits in one place in `v1` and a different place in `v2`, a person's pose
flickers between them. The model is **inventing a fresh, slightly different 3D world every
time you ask**, instead of committing to one.

That matters because there are *many* 3D worlds consistent with the limited evidence in `v0`
(an "underdetermined posterior over worlds"). SPT picks one at random each call. We want it to
pick the **same** one every time, so the views stitch together into a coherent scene.

**Goal (Hyunwoo's words):** force every invented video to be a picture of one shared world.
Formally, we want the joint distribution over all the videos, conditioned on their camera/time
actions `a`:

```
p(v0, v1, ..., vN | a0, ..., aN)
```

Modeling them *jointly* is the point — it lets us *demand* they share a world and penalize
disagreement. Factorized, that joint is:

```
p(v0, v1, v2) = p(v0) · p(v1|v0) · p(v2|v0, v1)
```

---

## 3. Why we can't just train it the normal way

Normal fine-tuning needs input/target pairs: "given `v0` and `v1`, here is the correct `v2`."
But **we have no real multi-view `v2`** — nobody filmed the same event from all these invented
camera angles. So there is no ground truth to regress against for the thing we actually care
about.

- We **have** `p(v0)` (real videos / the data) and `p(v1|v0)` (that's what SPT already is: one
  source video → one new view).
- We **lack** `p(v2 | v0, v1)` — a model that conditions on *two* videos and generates a third
  consistent one. That is the thing we need to build.

Since we can't train it with targets, we **distill** it out of the model we already have. The
distillation tool is **DMD**.

---

## 4. The core idea: "autoregression over videos"

Hyunwoo's framing (transcript ~15:31): this is autoregression **not over frames, but over
videos**. Start with `v0`. Generate `v1`. Then, conditioning on `v0` and `v1`, generate `v2`.
Then `v3` from earlier views. Recurse. Each step adds a view that is consistent with the ones
before it, incrementally building one world.

So the model we train — call it the **student** `q_θ` — takes **two** source videos and
produces a third:

```
student:  (v0, v1)  ->  v2
```

Why include `v1` at all? Because `v1` is another glimpse of the same hidden world. If `v1` is
chosen to *overlap* with `v2`'s viewpoint at a shared camera/time point (the "crossing
constraint", §9), then `v1` pins down the world that `v2` must also respect.

---

## 5. DMD in plain words: two arrows

DMD (Distribution Matching Distillation) trains a generator **without ground-truth targets**.
Instead of "match this exact video," it says "nudge each generated video in a direction that
makes it more like the teacher's distribution."

That direction is built from two **scores** (a score = the direction that makes a sample more
probable under a model — `s(v) = ∇_v log p(v)`):

- **`s_real`** — the **teacher's** arrow: "move the video this way to look more like what the
  teacher considers a valid, realistic view."
- **`s_fake`** — the **student's own** arrow: "this is the direction the student is already
  piling probability, i.e. where it's collapsing / blurring."

The useful update direction is the difference:

```
A = s_real − s_fake
```

Intuition: `s_real` pulls toward realistic; `−s_fake` pushes *away* from the student's own
pile-up (this is the anti-collapse / anti-blur force). We move each generated `v2` along `A`.

The clean fact that makes DMD work (Derivation D1, standard/off-the-shelf): even though the KL
divergence between student and teacher is intractable, its **gradient** reduces to exactly this
arrow difference pushed back through the generator:

```
∇_θ KL(q_θ ‖ p) = E[ (s_fake − s_real) · ∂_θ G_θ(z) ]
```

We implement it as a **detached surrogate**: compute `A`, stop its gradient, dot it with the
generated sample, and call `.backward()`.

---

## 6. The exact objective — Hyunwoo's branch (this is the important part)

Here is the precise setup, straight from `counterfactual.txt` (lines 40–43). Keep `q` = the
model you want to train, `p` = what you already have.

- **What we already have (the teacher):** the direct, one-hop model `p(v2 | v0)` — released SPT
  making a new view from `v0` alone. So `p(v2, v0) = p(v2|v0) · p(v0)`.
- **What we want (the student):** `q_θ(v2 | v0, v1)`, the two-source model. We don't have a
  target for it. But we can write the student's *joint* by **marginalizing out the middle**
  `v1`:

```
q_θ(v2, v0) = ∫ q_θ(v2 | v0, v1) · p(v1|v0) · p(v0)  dv1
            \_______________/   \____________________/
             the ONLY trained     reuse what we have
                 piece
```

- **The objective:** make the student's joint equal the teacher's joint — match them with DMD:

```
minimize  D_KL( q_θ(v2|v0)  ‖  p(v2|v0) )
          \_____________/     \________/
           student's           direct
           marginal            teacher
```

Read it in words: *"Generate `v2` by drawing a middle `v1 ~ p(v1|v0)` and then running the
two-source student; on average (marginalized over `v1`), that should look exactly like the
plain one-hop teacher `p(v2|v0)`."*

**Novelty (Hyunwoo, line 43):** nobody has done a **Monte-Carlo marginalized version of DMD**.
The marginalization over `v1` — done by sampling middles — is the contribution.

So the two arrows become:

- **`s_real = ∇ log p(v2|v0)`** — one frozen-SPT call, source = the real `v0`. Simple.
- **`s_fake = ∇ log q_θ(v2|v0)`** — the score of the **marginalized** student. This is the hard
  one, and §7 explains the trick that gives it to us.

---

## 7. The single most useful trick: marginalize by hiding a variable

(Transcript ~29:41–35:47, and `counterfactual.txt` "Key Trick 1".)

Least-squares regression returns the **conditional mean**:

```
minimize E[ ‖ f(x) − y ‖² ]   ⟹   f*(x) = E[ y | x ]
```

The consequence: **if you train a network with MSE but *hide* a variable, its optimum is the
average over that hidden variable.** For scores specifically,

```
s(x_t) = E_{ x0 | x_t } [ s(x_t | x0) ]     — the marginal score is the average of conditional scores.
```

We use this to get `s_fake`. The **fake-score net** is trained by ordinary denoising-MSE on the
student's generated `v2` samples, but conditioned on **`v0` only, with `v1` hidden**. By the
trick, its optimum is the **marginal** student score `∇ log q_θ(v2|v0)` — exactly the object we
need to compare against the direct teacher.

> This is why the fake-score net hides `v1`. If we fed it `v1`, it would learn a *conditional*
> score `q_θ(v2|v0,v1)`, which lives in a different conditioning context than the teacher
> `p(v2|v0)` — the comparison would be meaningless. Hiding `v1` is not a shortcut; it *is* the
> marginalization.

---

## 8. The three models

The training loop runs three SPT instances. **Only the student is two-source.**

| Model | Conditions on | Trainable? | Gives | Why |
|---|---|---|---|---|
| **Student `G_θ`** | `(v0, v1)` | yes | the generated `v2` | the model we ultimately want |
| **Teacher `p`** | `v0` (one source) | no, frozen | `s_real = ∇log p(v2\|v0)` | the "realistic view" arrow; it's the released SPT |
| **Fake-score net** | `v0` (one source, `v1` hidden) | yes, online | `s_fake = ∇log q_θ(v2\|v0)` | the student's own arrow, marginalized via §7 |

The fake-score net "chases" the student: as the student changes, the fake-score net keeps
re-learning the student's current distribution. Plain MSE lives **only** here — never as the
main objective.

---

## 9. The math-to-code bridge: velocity → score (the one thing not to get wrong)

SPT is a flow-matching model. It does not output a score directly; it outputs a **velocity**.
The scheduler defines:

```
noised video:     x_t = (1 − σ)·x0 + σ·ε        (ε = noise, σ = noise level)
model predicts:   v   = ε − x0                   (velocity)
```

Invert those two lines to recover the score:

```
x0 = x_t − σ·v
ε  = x_t + (1 − σ)·v
score  s(x_t) = −ε/σ = −( x_t + (1 − σ)·v ) / σ
```

And the beautiful part — when you subtract the two arrows, the `x_t` term cancels, so the whole
DMD update is just a **velocity difference**:

```
A = s_real − s_fake = −((1 − σ)/σ) · ( v_real − v_fake )
```

giving the student loss we actually code (a detached surrogate; only `x0_hat` carries gradient):

```
loss_student = mean( stop_grad( (1−σ)/σ · (v_real − v_fake) ) · x0_hat )
```

where `v_real` = teacher velocity (source `v0`), `v_fake` = marginal fake-score velocity
(source `v0`, `v1` hidden). **This sign is load-bearing**; we unit-test it before any training
(and those tests already pass — see `tests/test_score.py`).

---

## 10. Preventing the cheat: stitching (anti-degeneracy)

There's a trivial way to be "consistent": make every view a gray blur. Two blurs trivially
agree. So consistency alone is not enough — we must forbid the blur.

Hyunwoo's fix (transcript 21:26–27:35; `counterfactual.txt` "Key Trick 2"): **stitching.**
Jointly generate the views, then take **one frame from each view at the same world-time** and
lay them end-to-end into a short strip:

```
x = W · [v0, v1, ..., vN]ᵀ         (W = a linear "pick one frame per video" selection matrix)
```

Demand that this strip **looks like a real single video** (sharp, temporally smooth). A blur
fails that test; a sharp, genuinely-consistent set passes. Concretely we align the *marginal*
distribution of the stitched strip `p(x)` to the distribution of real single videos `p_1(x)`,
again with DMD.

Why the fuss about `W` being **linear**? Because for a clean disjoint selection `W·Wᵀ = I`, so
noising commutes with slicing and the strip's score is just the joint score sliced the same
way:

```
s_x(x_t) = W · s_v(v_t)
```

No new network — we reuse the same score. This is the standard machinery of **linear inverse
problems with diffusion priors** (DPS): observe a linear projection, recover the whole signal
using the diffusion prior. (Derivation D3 — settled.)

---

## 11. Anchoring the world: the crossing constraint

For `v1` to actually inform `v2`, their camera trajectories must **overlap** — there must be at
least one shared `(camera pose, world-time)` point (`counterfactual.txt` 45–51). At that
crossing point, `v1` and `v2` are looking at the same bit of the world, so `v1` becomes a
useful anchor rather than a random unrelated view. The crossing trajectories are chosen
**heuristically** (Hyunwoo's Slack), not learned.

One code subtlety (Hyunwoo's Slack): the model's positional encoding (RoPE) is based on array
position, so two videos don't automatically know they share a world point. His instruction:
**explicitly feed the 3D position of each token** so the crossing rides on RoPE, alongside the
camera embedding. (Exact convention = a Thursday-meeting detail.)

---

## 12. Scaling past two views: pairwise windows

The teacher only ever handles two videos (one source + one target). The student should generate
many. Hyunwoo's plan (transcript 37:11–43:06, à la **Self-Forcing**): let the student generate
`N` videos, then enforce the two-video machinery on every **overlapping window** — `(1,2,3)`,
`(2,3,4)`, `(3,4,5)`, … — and sum those losses. The bet is that overlapping-pairwise
consistency propagates into global consistency (Derivation D4).

---

## 13. What's proven vs. what's a bet (read this before you defend it)

Being honest about this is what makes the project defensible — and it matches Hyunwoo's own
hedges.

**Settled / off-the-shelf (you can derive these on the board):**
- D1: the DMD gradient reduces to the detached arrow difference.
- The velocity→score conversion (§9) and the DMD sign — unit-tested.
- The MSE/regression trick (§7): hidden-variable least-squares → marginal.
- D3: the stitching transport `s_x = W s_v` under `WWᵀ = I`.

**The contribution (novel, but a design, not a theorem):**
- Monte-Carlo–marginalized DMD, with the marginalization on the **student** side and the
  fake-score net realizing it via the MSE trick.

**Honest bets (Hyunwoo states these as open himself):**
- That routing generation through a second, crossing-anchored view (+ stitching) actually
  produces geometric 3D consistency. DMD by itself gives *realism and anti-collapse*;
  consistency is an **emergent** property we **measure**, not something the loss proves.
- That pairwise-window consistency buys **global** consistency ("assuming transitivity… that is
  the question," 41:27).
- That the marginalization is tractable in practice ("I'm not sure it's possible," 29:48).

The clean way to state the whole thing in a talk: *DMD keeps the generated views realistic and
stops them collapsing; the multi-view conditioning, the crossing anchors, and the stitching
term are what push toward one shared world; and we validate that consistency with a metric
against the released baseline — we don't assert it.*

---

## 14. Defense Q&A (the questions a committee will ask)

**Q: You're distilling toward the released teacher, which is itself inconsistent. How can that
produce consistency?**
The teacher's job is only to keep `v2` realistic and faithful to `v0`. The consistency pressure
comes from elsewhere: the student is forced to route through the `(v0, v1)` conditioning whose
marginal must match the teacher, plus the stitching term (forbids blur) and the crossing anchor
(ties `v1` and `v2` to the same world point). DMD is the realism/anti-collapse force, not the
consistency force — by design.

**Q: Why marginalize on the student side instead of averaging teacher calls over middles?**
Because the teacher we actually have is the direct `p(v2|v0)`, and the object we're building is
`q_θ(v2|v0,v1)` whose marginal we can form and match. Averaging teachers would require a
two-video teacher we don't have, and stack an extra approximation. This is Hyunwoo's written
choice (`counterfactual.txt`).

**Q: Why does the fake-score net hide `v1`?**
Because of the regression trick: MSE with `v1` hidden makes the optimum the average over `v1` =
the *marginal* student score, which is the correct object to compare against the direct teacher.
Feeding `v1` would give a conditional score in the wrong conditioning context.

**Q: Where's your ground truth?**
There is none for multi-view — that's the whole reason for DMD. The only place plain MSE appears
is inside the fake-score net, trained on the student's own samples.

**Q: How do you stop degenerate blur?**
The stitching term: a strip of one frame per view must look like a real video; blur fails it.

**Q: The teacher handles two videos; how do you get to N consistent views?**
Generate N with the student, enforce the two-video objective on sliding overlapping windows
(Self-Forcing style), and measure drift as a function of chain distance.

**Q: How do you know you succeeded?**
Measure multi-view geometric consistency (cross-view correspondence / reprojection error, or
3D-reconstructability) against the released SPT baseline, plus a stitched-strip sharpness check
so gains aren't just collapse.

---

## 15. Symbol glossary

| Symbol | Meaning |
|---|---|
| `v0` | the real ("factual") input video |
| `v1, v2, …` | invented ("counterfactual") views of the same world |
| `a_n = (g_n, t_n)` | action for view n: camera pose `g_n` + world-time `t_n` |
| `p(·)` | the frozen teacher's probability (how plausible a video is) |
| `q_θ` | the trainable student's distribution |
| `s(v) = ∇_v log p(v)` | score / "arrow": direction that makes `v` more plausible |
| `s_real`, `s_fake` | teacher score and (marginal) student score |
| `σ` | diffusion noise level |
| `v` (in §9) | flow-matching **velocity** the model predicts (`ε − x0`), not a video |
| `W` | linear "pick one frame per video" stitching matrix |

---

### Where this fits with the other docs
- `WHAT_WE_ARE_BUILDING.md` (this file) — the concept, for understanding and defending.
- `HYUNWOO_RECONCILIATION.md` — which branch we chose and why (provenance).
- `IMPLEMENTATION_PLAN.md` — the rung-by-rung build sheet + results.
- `RESEARCH_CONTEXT.md` / `PLAN.md` — the detailed why / the execution ladder.
