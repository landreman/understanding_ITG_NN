# Claude response

I'll read the S06b plan section and report (skipping executive summaries).

## What S06b was, and what it was for

**The object being explained.** You have an ensemble of cyclic-convolutional networks that take a flux tube's seven geometry channels along the field line (each a function of `z`) plus the two drive scalars `a/L_T`, `a/L_n`, and predict the GX heat flux in the form `max(log Q, −2)`. That clipped log is the *native output* — everything in S06b explains that quantity, not `Q`, and not `exp(prediction)`.

**"Attribution" — the core idea you need.** An *attribution method* takes one prediction and splits it into per-input contributions: a number for every (channel, z) cell, saying how much that cell contributed. Positive means "pushed the predicted log-flux up", negative means "pushed it down". So an attribution map has the same shape as the input geometry, and you can ask questions like "does the network care about `gbdrift` near the outboard midplane?"

The crucial subtlety, which the report hammers repeatedly: **attribution is always relative to a baseline.** There is no such thing as "the importance of this input cell" in the absolute. What you can measure is *how the prediction changes as you move the input from some reference state to the actual state*, and split that change across cells. Change the reference and you change the answer. S06a (the earlier step) benchmarked candidate methods and baselines; **S06b was the scaling step** — take the two methods S06a registered, run them across the whole ensemble and a large panel of geometries, and report member-level results with honest uncertainty.

**The two registered methods:**

- **Low-pass Integrated Gradients (IG).** IG walks the input along a straight path from a reference to the real input (here 64 steps), measures the model's local slope at each step, and averages. The reference here is the *frequency-8 low-pass version of the same geometry* — the real geometry with its short-wavelength structure smoothed away. So IG here answers: "when we restore short-scale structure to this geometry, where does the prediction change come from?" That path is deliberately **off-manifold** — a smoothed geometry need not be a realizable equilibrium — so the results describe the *network's* sensitivity, not plasma causality.
- **A periodic extremal mask** (a perturbation method: find the set of cells that, when replaced by a background, most changes the prediction). Registered only as an explicitly *secondary* diagnostic, because it failed a symmetry test (below). It is magnitude-only — no sign.

**Scale of the run:** top 10 ensemble members (ranked by their stored validation scores) × 1,000 panel geometries under varied drives + the same 1,000 under fixed drives, plus a smaller 256-row sensitivity run on five members from ranks 11–50 and five from 51–100. ~4,646 s of compute, 304 MB of signed maps kept per member before any averaging.

## What was found

**1. The members agree about *which channels* matter, but not about *where along z*.**

Two agreement statistics, and the gap between them is the headline:

- *Channel-rank agreement* (do two members order the seven channels the same way by mean absolute attribution?): median pairwise Spearman **0.964** on varied rows, **1.000** on fixed. Near-perfect.
- *Cellwise sign agreement* (at a given channel and z, do members agree on the sign of the contribution?): **0.749** varied, **0.745** fixed — against a null of **0.623**, which is what you'd get from ten members flipping coins independently. So the observed agreement is only about a third of the way from chance to perfect.

Read together: the ensemble broadly agrees on the coarse channel ordering, but the members do **not** implement one identical signed, position-by-position mechanism. The report also notes the channel statistic is coarse — with seven channels, 0.964 is literally one adjacent swap, and 0.0357 is the smallest possible step below 1.

**2. The channel ordering itself (varied, unstable rows), with hierarchical bootstrap intervals:**

`gbdrift0_over_shat` (0.00267) > `gbdrift` (0.00223) > `gds21_over_shat` (0.00155) > `gds2` (0.00148) > `cvdrift` (0.00112) > `gds22_over_shat_squared` (0.00092) > `bmag` (0.00058).

A *hierarchical bootstrap* here means: to get error bars, resample both the ensemble members and whole `equilibrium_files` (not individual flux tubes or rows) with replacement, 500 times, and look at the spread. Grouping by equilibrium matters because near-duplicate rows from one equilibrium are not independent evidence — resampling by row would make the intervals look tighter than they are.

But the ordering is **not universal**: `gbdrift0_over_shat` is largest in equilibrium classes 0 and 2, `gds2` in class 1, `gds21_over_shat` in classes 3 and 4. That contradiction is explicitly kept, and it blocks any claim that one geometry channel is *the* shared feature of the ensemble.

Also note the signed median over z is small for every channel, because positive and negative locations cancel. Signed and absolute summaries are reported in separate columns throughout.

**3. Response size tracks regime — and tracks model failure.**

The low-pass response *decreases* with heat flux (varied: 0.00348 / 0.00231 / 0.00190 across low/medium/high unstable-flux bins; fixed: 0.00303 / 0.00166 / 0.00115), while keeping `gbdrift0_over_shat` on top in all six bins.

More interesting: the response is **2.4× larger** on rows in the top member-error tertile than the bottom, and **2.6×** larger (varied) / **3.2×** (fixed) comparing high- to low-ensemble-spread rows. In plain terms — where the members disagree with each other and with the truth, their predictions are also more sensitive to short-scale geometry structure. The report flags this as descriptive only: flux and other covariates still vary across those tertiles, so this is a correlation, not an isolated effect.

**4. Stable/near-floor rows are published but carry no conclusion.**

A third of the varied-gradient data sits at the clipped floor, `log Q = −2` — the model predicts the floor value and the output barely moves. Here the attribution magnitude is 0.376× (varied) and 0.128× (fixed) the unstable value, and S06a had already shown that on these rows low-pass IG does **not** beat a *network-free control* — a map that simply ranks cells by `|X − B|`, distance from the baseline, with no trained network involved at all. When a method can't beat a map containing zero model information, the map is telling you about the baseline, not the network. So every stable-row artifact carries `feature_claims_permitted = false`, and no feature-level conclusion in the report rests on those 240 varied / 23 fixed rows.

**5. Drive sensitivities behave sensibly.** Local derivative of the native output w.r.t. `a/L_T`, scaled by a robust spread of the reference cohort: **+2.15** varied (member range 2.07–2.28), **+1.74** fixed. For `a/L_n`: **−0.22** varied, **−0.74** fixed. Signs are consistent across all ten members. These are local slopes of the network, not finite physical interventions.

**6. Validation rank does not predict attribution stability.** Spearman correlation between a member's stored validation R² and how well its map agrees with the 20-member median: **−0.074**. That is a null, and it was the predicted null — S01 had already shown the validation-score range is too narrow for the ranking to be meaningful. A better-scoring member is not a more trustworthy explainer.

**7. The symmetry check separates two versions of the network.** S02 established that the physically meaningful object is `invariant_tilde_f`, a version exactly invariant to cyclic shifts of `z`; the original network has a pooling-phase dependence and is only approximately invariant. S06b re-checks this at the *explanation* level: shift the input, does the explanation shift with it?

- Canonical (`tilde_f`) low-pass IG: median co-shift error **9.1e−7** — float32 roundoff, i.e. a pass.
- Original network: median **0.874** — an order-one failure, exactly as S02 predicts.
- The secondary mask with its registered *fixed* background: median **0.931** — also an order-one failure. This is why the mask stayed secondary.

So the canonical and original explanations are genuinely different objects and are never substituted for one another, even though their channel orderings correlate 0.93–0.96.

## Conclusions, including the honest caveats

The defensible claims are narrow, and the report is deliberate about keeping them narrow:

- **The ensemble shares a coarse channel-level story but not a fine position-level one.** Any downstream physics interpretation has to work at the channel/regime level, or else confront the sign disagreement and the equilibrium-class contradiction head-on.
- **Every number is a statement about "restoring short-scale content from a low-pass reference,"** not baseline-free importance and not plasma causality. Baseline sensitivity is a known, unresolved limitation: the low-pass map and the robust-constant map agree only 0.432, and the pilot picked a different baseline than the production run did.
- **Faithfulness evidence is qualified, not clean.** Low-pass IG beat the network-free control only on unstable rows. Its *parameter-randomization* response — reinitialize the network's weights at random and check the map changes — was only partial (rank correlation 0.406 with the randomized-model map; a fully model-driven explanation should go near zero), because the input-minus-baseline factor in IG carries structure regardless of the model. No perturbation method passed the complete S06a gate at all.
- **Scaling bought member-level uncertainty, not validity.** The report says this explicitly: S06b "preserves that verdict rather than converting scale into validity." Running a qualified method on 20× more data makes the error bars honest; it does not upgrade the method.

Seven acceptance criteria were graded one by one: symmetry, uncertainty, signed/absolute separation, and the "no feature called common without an agreement statistic" criterion all pass; faithfulness-vs-control, parameter randomization, and baseline sensitivity pass only as qualified/partial, under the researcher's approved 2026-08-22 decision.

Six numbered corrections are kept in the report, several of them consequential — most notably a NumPy indexing bug that had swapped the row and member axes in the stratified table (one sign-agreement figure moved 0.530 → 0.730), and an automated-review finding that the covariate tertiles still contained floor rows, which cut the headline error ratio from 5.0× to 2.4×. Nothing was deferred from the step's tasks.

**Where this leaves you:** S06b produces position-resolved, member-level, signed maps with real uncertainty, plus the strong caveat that they describe network sensitivity along an off-manifold smoothing path. S07 is the step that takes those maps and compares them against actual physics fields and GX `Q(z)` — no physics claim is made here.

--------------------

# Codex response — what the ensemble's input maps agree about

## What this step was for

The neural-network ensemble receives seven geometry curves, each sampled at 96
positions along a periodic magnetic field line, plus the two driving gradients
`a/L_T` and `a/L_n`. S06a first tested several ways of assigning credit to those
inputs. S06b applies the selected methods to the stored-validation top ten and
asks a different question: do independently trained members point to the same
channels and positions, and does that agreement change with the physical and
model-error regime?

The primary method is **Integrated Gradients from a low-pass reference**. A
low-pass reference is the same geometry with its shorter-scale wiggles smoothed
away. Integrated Gradients averages the network's slope while moving from that
smoothed curve back to the original, then assigns the resulting prediction
change to input cells. Its map keeps signs: positive cells push this path's
prediction upward and negative cells push it downward.

The secondary method is a **periodic extremal mask**. It searches for a compact
set of cells whose replacement changes the prediction. The mask reports
magnitude—important or not—not direction. S06a found that no mask passed every
selection test, so this one remains a labeled secondary check, not a co-equal
primary explanation.

All results explain the network's native output, `max(log Q, -2)`. They are
reported for the exactly shift-invariant canonical network and the original
network separately. The 1,000 varied-gradient rows and their 1,000 fixed-drive
twins are also kept separate.

## The main result: broad agreement at channel scale, incomplete agreement at position scale

The top ten agree strongly about the *ordering of whole channels*. For the
canonical low-pass method, median pairwise rank agreement is 0.964 on varied
rows and 1.000 on fixed rows; 1 would mean identical ordering. This is strong
evidence that these independently trained members respond to a similar coarse
set of inputs along the registered smoothing path.

They agree less strongly about the sign of every precise channel-position cell.
Mean sign agreement is about 0.749 on varied rows and 0.745 on fixed rows. In
plain terms, roughly three quarters of the members agree on whether a typical
cell pushes the prediction up or down after each member's map has first been
averaged over the chosen rows. Because this statistic takes the larger of the
positive and negative fractions, ten independent random signs already average
0.623. The observed 0.749 is only about one third of the way from that null to
perfect agreement, so it is not enough to describe the ensemble as one shared
position-by-position mechanism.

On varied unstable rows, `gbdrift0_over_shat` has the largest mean absolute
response to restoring the short-scale content removed by the low-pass
reference, followed by `gbdrift`. However, this is not a universal channel
ordering: `gds2` is largest in one equilibrium class, and
`gds21_over_shat` is largest in two others. The careful conclusion is therefore
that a radial-drift-related channel is often prominent along this particular
network probe, not that it is the single physical cause of ITG heat flux.

That distinction matters. Smoothing a geometry does not generally create
another realizable magnetic equilibrium. These maps tell us what the network
responds to when short-scale content is restored; they do not show what would
happen if a plasma equilibrium could be changed in that cellwise way. S07 will
make the more physical comparison with GX fields and diagnostics.

## Where the response is largest

The largest-channel response is higher in the low unstable-flux third than in
the high unstable-flux third. For varied rows, the values are about 0.00348,
0.00231, and 0.00190 across low, medium, and high unstable flux. Fixed rows show
the same direction: 0.00303, 0.00166, and 0.00115. This does **not** mean that
geometry becomes less physically important at high flux; it means the network's
prediction changes less along this smoothing path.

The response is also larger on unstable rows where members make larger errors or
disagree more. The registered thirds are defined on the full panel and then all
stable/near-floor rows are excluded from the feature calculation. On varied
unstable rows, the largest-channel value is 0.00363 versus 0.00151 in the high-
and low-error thirds (2.4-fold), and 0.00362 versus 0.00140 in the high- and
low-ensemble-spread thirds (2.6-fold). That makes the maps potentially useful
for the later disagreement study, but it does not establish cause: flux and
other regimes still differ across those groups.

## Stable rows still cannot support feature claims

The networks predict a clipped floor value on many stable or nearly stable
rows. There the low-pass reference barely changes the prediction, so dividing
credit among input cells is close to asking how to divide almost nothing. S06a
also showed that the primary map does not beat a simple network-free control on
those rows.

S06b reports all 240 varied and 23 fixed stable/near-floor panel rows separately,
but makes no feature claim from them. Their total channel-level response is only
about 38% of the unstable value on varied rows and 13% on fixed rows. Every
machine-readable stable summary explicitly says that feature claims are not
permitted.

## The two scalar drives behave consistently

The canonical network's local sensitivity to `a/L_T` is positive for every top
member, while sensitivity to `a/L_n` is negative. After scaling each drive by a
robust reference-cohort range, the median member values are about +2.15 and
−0.22 on varied rows, and +1.74 and −0.73 on fixed rows. These are local slopes
of the trained network in its native clipped-log output, not simulated finite
changes to the plasma.

## Validation rank does not predict explanation quality

S06b added five rank-spaced members from validation ranks 11–50 and five from
ranks 51–100. All twenty were compared on the same smaller panel. The
correlation between stored validation score and agreement with the common
attribution ordering is −0.074—effectively zero. This supports S01's conclusion
that tiny differences in stored validation score should not be treated as a
scientifically meaningful hierarchy among these models.

## The symmetry check and the mask's negative result

Rotating all seven input curves together should only change where a feature is
drawn, not what the canonical network predicts. The canonical low-pass map obeys
this: its median rotation error is about nine parts in ten million. The original
network's map error is about 0.87 because its stride-two pooling retains the
starting-point dependence established in S02. That contrast is why the
canonical network remains the primary object.

The mask can rotate correctly only when its replacement background rotates too.
With the registered background held fixed, its median error is 0.93—an order-one
failure. This negative result is preserved. The mask can show whether a broad
magnitude pattern supports the primary map, but it cannot establish a
shift-consistent feature on its own.

## What to carry forward

- The top members share a strong whole-channel ordering, but their exact signed
  spatial mechanisms are not identical.
- `gbdrift0_over_shat` is often the largest response along the low-pass path,
  but equilibrium-class contradictions prevent a universal feature claim.
- Stable/near-floor rows remain report-only and support no feature conclusion.
- Attribution stability is unrelated to stored validation rank.
- The primary map is explicitly baseline-dependent and off the physical data
  manifold; the secondary mask retains a major symmetry failure.

S06b therefore narrows the next question without pretending to answer the final
physics question. S07 should test whether the ensemble-supported spatial
patterns align with held-out GX quantities and whether those relationships
survive equilibrium class, drive, and flux stratification.
