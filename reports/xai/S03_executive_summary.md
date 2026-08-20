# Step 03 Executive Summary

## What step 03 was

Step 03 is called **"the structure-destroying counterfactual ladder."** It's the third of the numbered experiments in [PLAN.md:568](PLAN.md:568), and its full write-up is [S03_ladder.md](reports/xai/S03_ladder.md), with a plain-language version already in [S03_executive_summary.md](reports/xai/S03_executive_summary.md).

### The vocabulary you need first

**The thing being explained.** The network takes 7 geometric quantities sampled at 96 points along a field line, plus 2 drive numbers ($a/L_T$, $a/L_n$), and outputs `max(log Q, -2)`. That clipping at −2 is a *floor*: below some heat flux the training data just says "essentially zero," so all near-zero cases pile up at the same output value. Cases sitting on that floor are called **stable / near-floor** rows and have to be reported separately, because a quantity that can't move can't show sensitivity.

**Counterfactual.** A fake input you invent to ask "what would the network have said if the input had been different?" You feed it in, you're not claiming it's a real plasma.

**Residual standard deviation (residual SD).** How wrong the network typically is on ordinary honest inputs — about 0.3 in log units here. It's used as the ruler. Every number in step 03 is "output change, measured in units of the network's own typical error." A perturbation that moves the output by 0.1 residual SD is lost in the noise; one that moves it by 3 residual SD broke something the network genuinely depends on.

**On- vs off-manifold.** The "manifold" is the set of inputs that correspond to real, physically achievable stellarator geometries. An off-manifold input is one no real device could produce. The network will happily give you a number for it, but that number tells you about *the network's internal function*, not about plasma physics. Step 03 tags every edit with which kind it is — this is the discipline that keeps "the network is sensitive to X" from being mistaken for "physics depends on X."

**Multiset.** A bag of items where you keep the values but throw away the order. Here: the 96 seven-component vectors, shuffled. If the network gave the same answer on the shuffled version, it would mean the network only cares about *what values occur along the field line*, not *where they occur or in what pattern* — an enormously simplifying discovery.

### Why do it this way

The natural instinct in explainable-AI work is to go straight to "which input point mattered most" (attribution methods, saliency maps). That's expensive and fragile. Step 03 is a cheap upper-bound exercise done first: instead of asking which detail matters, **delete an entire category of information and see how much the answer moves.** If deleting all fine-scale detail changes nothing, later steps never need to look for fine-scale mechanisms.

It's a "ladder" because the destructions are graded from gentle to brutal, so you get a *spectrum over length scales* rather than a yes/no.

Step 03 also builds the reusable machinery the later steps depend on — the perturbation/baseline API, including the rule that you never blank an input to zero (magnetic field strength is positive; zero isn't "neutral," it's nonsense) — plus the wrapped-around-the-circle masks and the four validity tags.

## What was concluded

**1. The network is not a multiset model.** Randomly shuffling the 96 joint vectors moves the output by a median **3.26 residual SD** — huge. So position/pattern information is genuinely used. But the report is careful: that shuffle destroys *two* things at once (the ordering, and the smooth large-scale envelope), so it can't say which of the two the network was using.

**2. Broad, long-wavelength structure carries most of the signal.** Shuffling in contiguous blocks instead of point-by-point gives a clean decay: blocks of 2 → 2.93, 4 → 2.42, 8 → 1.83, 16 → 1.26. (The 32 case is only 3 blocks, so it's a reversal control, not another rung.) Independently, removing Fourier bands gives 3.85 for the low band, 1.22 for the middle, **0.099 for the high band** — i.e. the fine wiggles are essentially irrelevant. This is the main actionable result for later steps: look at smooth, broad features.

**3. Cross-channel alignment matters.** Shifting each of the 7 channels independently around the circle — which preserves each channel's own profile and spectrum perfectly, and only breaks their *relative* registration — still moves the output 2.41 residual SD. So the network cares about which geometric quantities line up with which, not just each one's shape in isolation.

**4. The symmetries it should ignore, it does ignore.** Rotating the whole field line by a full symmetry period: error ~1e-5, pure numerical noise. Mirror-reflecting with the physically correct sign flips: 0.14, small. The deliberately *wrong* mirror (reflect without the sign flips): 1.26, nearly 10× larger. The network has internalized a real physical symmetry — and the wrong-parity control is what makes that claim mean something, since a network that ignored everything would also score low on the correct mirror.

**5. Honest limits, kept rather than buried.**
- The phase-scrambling comparison (per-channel vs common phase) is *suggestive, not decisive*: medians 0.36/0.29/0.55 across three replicates, with only 8/10, 6/10, 9/10 members agreeing. Not promoted to a mechanism claim.
- There is **no unique channel importance ranking** — normalizing by RMS gives one order (dominated by rare `gds2` extremes), by median another. Both retained as sensitivity views.
- Stable rows behave differently (0.996 vs 3.54 residual SD), but that ratio isn't a clean effect estimate because the floor compresses numerator and denominator together.
- **A genuine negative result:** the "support score," built to warn when an edited input has wandered off-manifold, *fails on the case it exists for.* A randomly shuffled field line — plainly unphysical — is flagged only 10.5% of the time, versus 11.4% for untouched inputs. So a low warning score is not evidence an input is realizable. Better to know that now than to have it quietly corrupt step 07.

**6. A registered-premise conflict was found and escalated, not patched.** The shared loader supplies $a/L_T = -3$ for fixed-gradient cases, but the serialized training tensors contain only nonnegative $a/L_T$, and $-3$ saturates the members at the clipped floor while $+3$ recovers R² of 0.978–0.985. All fixed-gradient results in step 03 were therefore **withdrawn**, and the fix was written up as an open decision gate for you ([S03_fixed_gradient_decision.md](reports/xai/S03_fixed_gradient_decision.md)) rather than changed inside the step — because editing that loader would silently alter a premise that S00–S02 already inherited.

*Since resolved.* You approved the correction on 2026-08-20 and it landed in its own prerequisite branch: fixed rows now reach the network at $+3$, where all 100 members score R² 0.973–0.987 on them, and the S01 and S02 artifacts that carried the old convention were refreshed. Tracing the legacy code also showed *why* the wrong value was there — the negation was added to the training script after the ensemble was trained, purely so the paper's test score would use varied rows only. Step 03's own fixed-gradient results remain withdrawn; restoring them means rerunning this ladder on fixed rows, which is a step's worth of work rather than a repair.

## The net contribution

Later, expensive steps now enter with the hypothesis space narrowed: look for **broad, long-wavelength geometric structure and cross-channel alignment, chiefly in unstable cases**; don't chase high-frequency detail. One thing remains explicitly unresolved — whether the permutation response is specifically about *ordering along the field line* or about the *smooth envelope*, since no rung in this ladder separates the two.

#########################################################

# Old version of the summary

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

**1. The network is not a multiset-only model, and channel alignment matters.** Randomly shuffling the 96 joint seven-channel vectors moves output by **3.3 residual SDs**, so preserving the vector multiset is not enough. That edit destroys both ordering and the low-frequency envelope, however, so it cannot say which caused the response. The cleaner alignment test shifts each channel independently while preserving every channel's complete profile and spectrum; its 2.4-SD effect shows that cross-channel alignment matters.

**2. Broad structure carries much of the signal.** Shuffling in contiguous blocks instead of point-by-point gives: blocks of 2 → 2.93, 4 → 2.42, 8 → 1.83, and 16 → 1.26. The 32-point edit is not another scale rung: with only three blocks it is a reversal control (1.17). Complementary Fourier evidence gives raw effects 3.85, 1.22, and 0.099 for removing low, middle, and high bands. Dose-normalized multipliers depend strongly on whether dose is reduced by RMS or a median, so they are not used as headline evidence. **Takeaway for later steps: examine broad, smooth features, while treating normalization-dependent efficiency ranks cautiously.**

**3. Symmetries the model should ignore, it does ignore.** Rigidly rotating the whole field line by one full symmetry period changes nothing (error ~1e-5, i.e. numerical noise). Mirror-reflecting it with the physically correct sign flips gives 0.14 — small. The deliberately *wrong* mirror (reflect without the sign flips) gives 1.26 — nearly 10× larger. That's a nice sanity check: the network has internalized a real physical symmetry.

**4. Cross-channel phase evidence is suggestive, not decisive.** A corrected matched calculation uses one random phase tensor for both controls. Across three repeats, per-channel minus common-phase effects have paired member medians 0.36, 0.29, and 0.55, but only 8/10, 6/10, and 9/10 members agree. This does not establish a dominant mechanism. Likewise there is no unique channel ranking: RMS dose is dominated by rare `gds2` extremes, while median dose produces a different order. Both are retained as sensitivity views rather than promoted as importance.

**5. Effects differ by regime, with a floor caveat.** Joint permutation is 0.996 residual SD on stable rows and 3.54 on unstable rows. These are kept separate, but the stable ratio is not a clean drive-effect estimate because clipping compresses both its numerator and its residual-standard-deviation denominator.

**6. An honest negative result.** They built a "support score" meant to flag when a modified input has drifted off into physically impossible territory. It fails at exactly the case that matters: a randomly shuffled field line — obviously unphysical — gets flagged only 10.5% of the time, barely above the 11.4% baseline for *unmodified* inputs. The report states plainly that a low warning score is not evidence an input is realizable. That limitation is now known before it could quietly corrupt a later step.

## Why the report reads as it does

Much of it is machinery you can skim: operators were validated on toy models *before* touching the real network; deterministic endpoints are hashed; uncertainty resamples whole equilibria rather than individual field lines; and the original ladder plus corrected matched-phase slice are connected to the committed summaries through an explicit derived-artifact manifest. Four independent review rounds sharpened the interpretation and corrected the machine-readable evidence.

The net contribution: the later, expensive steps now know to look for **broad, long-wavelength structure and cross-channel alignment**, chiefly in unstable cases. Whether permutation's additional response is specifically ordering-dependent remains unresolved.
