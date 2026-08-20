I'll look at the step 02 report and the plan.

Here's the plain-language version.

## First, what the thing being studied actually is

You have a neural network (really 100 of them, an "ensemble") that takes:

- **Seven numbers at each of 96 points** going around a flux tube — the magnetic geometry along a field line. Think of it as a 7-channel signal sampled at 96 positions, like seven audio tracks 96 samples long.
- **Two scalar knobs**, the temperature and density gradients (`g_T`, `g_n`) — how hard the plasma is being driven.

and predicts the ITG turbulent heat flux, specifically `max(log Q, -2)`. The clipping at -2 matters: about a third of the data sits on that floor, meaning "stable, essentially no turbulence." Those rows have to be analyzed separately, because a model can look great on them by just outputting the floor.

Some jargon you'll need:

- **Convolution**: the network slides a small pattern-detector along the 96-point grid, applying the same detector at every position. That's why it's "position-aware" but not tied to absolute position.
- **Cyclic**: the grid wraps around — position 95 is adjacent to position 0. Physically correct, since you're going around in the field-line coordinate.
- **Pooling**: after each convolution the network halves the resolution by keeping the larger of each adjacent pair. Five such blocks: 96 → 48 → 24 → 12 → 6 → 3.
- **Bottleneck / GAP**: after those blocks, the network averages over the remaining positions ("global average pooling") to get a handful of numbers (7 to 32 of them, depending on the member). Those numbers plus the two gradients go into a small dense network ("the MLP head") that produces the prediction.
- **Ensemble**: 100 independently trained copies. You usually quote their average, but they are 100 different functions and can disagree.

## What step 02 was for

Every later step in this project asks "what geometric feature is the network looking at, and *where* along the field line?" Before you can answer that, you have to pin down **exactly which function you are explaining**. Step 02 was the setup step that fixes that object once and for all, plus a set of sanity checks on the network's structure.

The specific worry: the physics has a symmetry. If you rotate the whole geometry signal around the flux tube by some number of grid points, nothing physical has changed, so the prediction shouldn't change either. Convolutions *look* like they respect that, but the pooling steps break it — after halving five times, shifting the input by 1 point lands you on a different "phase" of the pooling grid and you get a different answer. That's an artifact of the architecture, not physics. If you don't fix it first, every later "the network cares about position 47" claim would be partly measuring that artifact.

## What was found

**1. The shift symmetry is only partly there, and the leak is not small.**

Shifts of exactly 0, 32, or 64 points leave the output unchanged to ~1e-5 — numerically exact. (32 = 2⁵, the total pooling stride, so those shifts land on the same pooling phase.) But *arbitrary* shifts move a single member's output by roughly **half of that member's own typical error** — median ratio 0.51. That's large. It's an artifact, and the fix was needed.

Two important refinements:
- Averaging over 32 shifts is identical to averaging over all 96, so all later work uses 32 phases. (Cheaper, and it's an exact statement, not an approximation.)
- The ensemble average hides most of this — members' shift-artifacts partly cancel. So the artifact would have been easy to overlook if you only looked at ensemble numbers.

**2. Rows with fixed vs. varied gradients behave differently, but not dramatically.** Fixed-gradient cases shift by ~0.082 in output; varied-gradient cases by ~0.138, so roughly **1.7× more**. An earlier pooled headline number was withdrawn because of this — the two sets still can't be averaged together.

> **This finding was corrected on 2026-08-20, and the original version was wrong in substance, not just in its digits.** It first read "~0.012 versus ~0.14, about 10× more." Those fixed-gradient rows had been fed to the network at an $a/L_T$ it was never trained on, which flattens every member against the output floor: the ensemble mean lands within a 0.097-wide band at the floor, and no member varies even as much as the *least*-varying member does at the correct input. (Individual members are not all sitting exactly on the floor — 37 of 100 are — but every one of them has stopped responding.) What looked like "fixed-gradient cases are far less sensitive to geometry" was the networks being saturated. Measured properly, the gap is a factor of about two — a real effect of the different drive, but an ordinary one. Details in [S03_fixed_gradient_decision.md](reports/xai/S03_fixed_gradient_decision.md). Nothing else in step 02 changed: the varied-gradient numbers were never affected, and the refresh reproduced every previously published varied row exactly.

**3. A cheap exactly-symmetric version of the model was built and adopted.** Three candidates were compared:

| | what it is | cost |
|---|---|---|
| `f` | the original network | 1× |
| `bar_f` | run the network on all 32 shifts and average the predictions | 28× |
| `tilde_f` | average the *internal* feature maps over shifts, then run the head once | 1.25× |

`tilde_f` is exactly shift-invariant, essentially free, and is now the canonical model. Accuracy is unchanged: R² ≈ 0.989 for all three.

**4. A genuinely interesting negative result on accuracy.** Making the model symmetric improved **every single one of the 100 individual members** (each got ~4% smaller residual error, and this survived proper resampling uncertainty). Yet the *ensemble* got very slightly worse — not statistically resolvable, but the sign flipped. The explanation: symmetrization changed how the members' errors cancel against each other. Part of the ensemble's advantage was members' shift-artifacts pointing in opposite directions. This is exactly the kind of thing that vanishes if you only ever look at the ensemble mean, which is why the project rules insist on keeping per-member signed results.

**5. A "density" was built so later steps can say *where*.** Normally the network throws away position information at the GAP step. Step 02 reimplemented the pooling chain at full 96-point resolution (an "à trous" or dilated version — same math, no downsampling), producing a per-position quantity ρ whose average over position exactly reproduces what the original network computed. Verified to ~2e-6. This is the object steps 05 and 07 will use to make position-resolved claims.

**A bug was found and fixed here by external review**, and it's worth knowing about: the first implementation got the *average* right but had the whole position axis rotated by 1–15 points for 91 of the 100 members. Every accuracy result was unaffected (a rotation doesn't change an average), so the original checks all passed — but every future "the network responds to position 47" statement would have been wrong. The fix was a padding convention, and a third check was added that pins the density to the original network's positions directly.

**6. Parity (the mirror symmetry) is real but only approximate.** Flipping z → −z with sign flips on two channels changes predictions by ~0.22 residual-standard-deviations; doing the *wrong* flip (no sign changes) changes them by ~1.79, eight times more. So the networks clearly learned the physical rule — but it isn't exact, partly because the data itself doesn't obey it exactly (the two sign-flipping channels have ~4–8% mismatch). It's a useful control, not a hard constraint.

**7. Structural census.** All 100 members see the entire flux tube by the last block (89 of 100 already by block 4) — so there's no member that's only looking locally. Bottleneck widths run 7–32. About 12% of bottleneck units are **dead** (293 of 2,449) — they output essentially zero always, wasted capacity. Notably, none of this correlates with how accurate a member is (rank correlations all under 0.15), so "bigger bottleneck" or "fewer dead units" does not mean "better member."

## The one caveat carried forward

The reference cohort is still interpolation: every equilibrium in it also appeared in training. Step 02 didn't fix that, and nothing here is a claim about physical causation — it's all about what the network does.