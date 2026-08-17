I'll look at the plan and the step 03 report.

## The setting, in plain terms

You have a neural network that takes in the *shape of a magnetic field* along a field line — seven different geometric quantities sampled at 96 points going around the line — plus two "drive" numbers (how steep the temperature and density gradients are). It outputs a predicted turbulent heat flux, specifically `max(log Q, -2)`.

Nobody knows *what about the shape* the network is reading. That's the whole project. A few terms you'll need:

- **Panel** — a frozen set of 1,000 test cases (one per equilibrium), fixed in advance so nobody can cherry-pick results later.
- **Residual standard deviation** — how far off the network typically is when it's just doing its job. Roughly 0.3 in log units. This is the natural yardstick: *if I mess with the input and the output moves by less than this, the network basically didn't notice; if it moves by several times this, I broke something the network genuinely relies on.* All the numbers below are in these units.
- **Off-manifold** — an input that's physically impossible, e.g. a magnetic field shape that no real stellarator could have. You can still feed it to the network. The answer tells you about *the network*, not about *plasma physics*. Step 03 tags every edit with whether it's physical or not, and is disciplined about not confusing the two.

## What step 03 is for

It's a cheap, early **upper-bound exercise**, done before any expensive fine-grained analysis. The idea: rather than asking "which input point matters most" (hard, slow), first **destroy** whole categories of structure and see how much the answer moves. That tells you which *kinds* of information are on the table at all, so later steps don't waste effort chasing something the network never used.

The name "ladder" is because the destructions are graded from gentlest to most brutal, so you get a spectrum rather than a yes/no.

Along the way it also builds the reusable toolkit — the "perturbation API" — that every later step will use to modify inputs safely, including the rule that you must never blank an input to zero (magnetic field strength is a positive quantity; zero isn't a neutral input, it's nonsense).

## What was found

**1. Order along the field line genuinely matters.** The key test: shuffle the 96 positions randomly, keeping each position's seven values glued together. Every individual "point" is still present and unchanged — only their sequence is scrambled. Output moves by **3.3 residual SDs**, a large effect. So the network is not just averaging over points; it reads the *arrangement*.

**2. Broad structure carries much of the signal.** Shuffling in contiguous blocks instead of point-by-point gives: blocks of 2 → 2.93, 4 → 2.42, 8 → 1.83, and 16 → 1.26. The 32-point edit is not another scale rung: with only three blocks it is a reversal control (1.17). Complementary Fourier evidence gives raw effects 3.85, 1.22, and 0.099 for removing low, middle, and high bands. Dose-normalized multipliers depend strongly on whether dose is reduced by RMS or a median, so they are not used as headline evidence. **Takeaway for later steps: examine broad, smooth features, while treating normalization-dependent efficiency ranks cautiously.**

**3. Symmetries the model should ignore, it does ignore.** Rigidly rotating the whole field line by one full symmetry period changes nothing (error ~1e-5, i.e. numerical noise). Mirror-reflecting it with the physically correct sign flips gives 0.14 — small. The deliberately *wrong* mirror (reflect without the sign flips) gives 1.26 — nearly 10× larger. That's a nice sanity check: the network has internalized a real physical symmetry.

**4. Cross-channel phase evidence is suggestive, not decisive.** A corrected matched calculation uses one random phase tensor for both controls. Across three repeats, per-channel minus common-phase effects have paired member medians 0.36, 0.29, and 0.55, but only 8/10, 6/10, and 9/10 members agree. This does not establish a dominant mechanism. Likewise there is no unique channel ranking: RMS dose is dominated by rare `gds2` extremes, while median dose produces a different order. Both are retained as sensitivity views rather than promoted as importance.

**5. Effects differ by regime, with a floor caveat.** Joint permutation is 0.996 residual SD on stable rows and 3.54 on unstable rows. These are kept separate, but the stable ratio is not a clean drive-effect estimate because clipping compresses both its numerator and its residual-standard-deviation denominator.

**6. An honest negative result.** They built a "support score" meant to flag when a modified input has drifted off into physically impossible territory. It fails at exactly the case that matters: a randomly shuffled field line — obviously unphysical — gets flagged only 10.5% of the time, barely above the 11.4% baseline for *unmodified* inputs. The report states plainly that a low warning score is not evidence an input is realizable. That limitation is now known before it could quietly corrupt a later step.

## Why the report reads as it does

Much of it is machinery you can skim: operators were validated on toy models *before* touching the real network; deterministic endpoints are hashed; uncertainty resamples whole equilibria rather than individual field lines; and the original ladder plus corrected matched-phase slice are connected to the committed summaries through an explicit derived-artifact manifest. Three independent review rounds changed several secondary claims while leaving the main ordering result intact.

The net contribution: the later, expensive steps now know to look for **broad, long-wavelength, order-dependent structure**, chiefly in unstable cases — and they have a vetted toolkit plus a documented list of things that don't work.
