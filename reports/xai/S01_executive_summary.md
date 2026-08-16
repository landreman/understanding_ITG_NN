Here's the plain-language version. I'll define jargon as it comes up.

## The setting

You have 100 neural networks (an "ensemble") that were trained to predict ITG turbulent heat flux from flux-tube geometry. The whole project is about reverse-engineering *what those networks actually learned* — which geometric features they pay attention to.

Step 01 does none of that reverse-engineering. It's the **bookkeeping step that comes first**: before you can say "the network responds to feature X," you have to nail down exactly which data you're explaining, which networks, and how much you can trust the numbers you're comparing against. It's the equivalent of calibrating your instruments and writing down your sample list before running the experiment.

## What it actually did, and what it found

**1. Rebuilt the original train/validation/test split — and found it leaks.**

Some jargon first. When you train a network you split your data three ways: *training* data it learns from, *validation* data used to tune and pick models, and *test* data held back to give an honest score. The whole point of the test set is that the network has never seen it, so the score tells you how well it generalizes to genuinely new cases.

The audit found that separation is much weaker than it looks:

- The dataset has two "gradient sets" — *fixed* gradient and *varied* gradient — that share the exact same geometry rows. Nearly 34% of matched fixed/varied pairs land on opposite sides of the split. So for ~90% of test rows, the network already saw that identical geometry during training, just with different drive parameters.
- More broadly, the data is organized by *equilibrium files* (one magnetic configuration) each containing many *flux tubes* (samples from within that configuration). 19,571 of 23,577 equilibria appear in more than one split. **Every single** test row's equilibrium is also present in training.

The consequence, stated carefully in the report: the published R² of 0.989 is a real number, but it measures *interpolation to new flux tubes within already-seen equilibria*, not the ability to generalize to a brand-new stellarator configuration. Step 01 diagnosed this and deliberately did not retrain or restate the paper's headline score — that's out of scope. But it means later steps must be worded carefully.

**2. Confirmed the ensemble reproduces its published score exactly.** R² matched to 1e-10. This is the "our pipeline is wired up correctly" check — everything downstream rests on it.

Small but important nuance: about a third of the data sits at a *floor*. The network predicts `max(log Q, -2)`, i.e. log heat flux clipped at −2 — physically, these are the stable cases where turbulence is essentially off. Averaging those in with the unstable cases hides real differences. Split apart: error is ~2.5× larger on unstable rows than on floored ones. Also, R² is meaningless on the floored subset (it's a ratio against the data's variance, and the floored data barely varies — you get R² = −877, which is an artifact, not a failure). They kept that absurd number visible in the table rather than quietly deleting it.

**3. The big finding — the "top 10 best members" ranking is largely noise.**

Someone previously ranked the 100 networks by their validation scores and treated the top 10 as the good ones. Step 01 re-scored all 100 on held-out data and asked how stable that ranking is.

*Bootstrap* is the technique used: repeatedly re-draw a random sample of your data and re-compute the ranking, 500 times, to see how much it wobbles. Crucially they resampled whole *equilibria*, not individual flux tubes — tubes from the same equilibrium aren't independent, so treating them as independent would fake extra confidence.

Results:

- Rank correlation between validation score and held-out score: **0.57**. Loose, not tight.
- **Zero** of 500 bootstrap resamples reproduced the original top-10 set.
- The network ranked #1 on validation is actually **#31** on held-out data, with only a 4% chance of even making the top 10.
- Meanwhile the network ranked #48 on validation is genuinely #5, making the top 10 in 71% of resamples.

The scores are all so close together that ranking them is mostly reading noise. The practical decision: keep the original top-10 as the headline group for continuity, but treat "is this a top-10 member" as a soft hint, never as a quality boundary. In later steps, if a finding only holds for the top 10 and not the other 90, that's probably not a real finding.

**4. Froze an "interpretation panel."** 2,000 hand-selected rows — 1,000 varied-gradient cases from 1,000 distinct equilibria, plus their 1,000 fixed-gradient twins. The twins matter: same geometry, different drive, so you can isolate geometry effects at constant drive. The selection deliberately spans stable, near-threshold, and low/medium/high flux; all five equilibrium classes; and includes cases where the networks do badly or disagree with each other. Locked in *before* anyone ran an explanation method, so the panel can't be tuned to flatter a result.

**5. Established that the seven geometry channels can't be compared raw.** The channels have wildly different natural sizes — `gds2` reaches 1,212, while `bmag` varies over a range of ~0.19. If you asked "which channel does the network care most about" by comparing raw sensitivities, you'd just be measuring which channel has bigger numbers. Step 01 computed and registered proper rescaling factors so later comparisons are fair.

**6. Verified a mathematical identity in the data.** There's a quantity, ⟨|∇x|⟩ (flux-surface-averaged gradient), that should be exactly reconstructible from two of the input channels. It checks out across all 100,705 geometries to 1 part in 10⁷. This is valuable because it gives a **known-correct answer to test explanation methods against**: if a method claims to find what a network computes, and you feed it a case where you already know the answer, you can check whether the method lies. Three rows showed slightly worse precision and one was a clear outlier — noted, kept, and later steps are told to use the stored value rather than the reconstruction.

## The short version

Step 01 produced no physics. It produced trustworthy foundations, and along the way found two things that constrain everything after it: **the train/test split leaks, so the headline accuracy measures interpolation rather than generalization to new configurations**, and **the ranking of the 100 networks is close to arbitrary, so "the best member" isn't a meaningful category.** Both are negative results, and both were kept rather than smoothed over — which is the point of running the audit before the interesting work rather than after.