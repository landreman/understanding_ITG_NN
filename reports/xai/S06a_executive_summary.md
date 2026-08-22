# Claude response

## What step S06a was for

The project's larger goal is to figure out *what geometric features* the trained neural networks look at when they predict ITG heat flux. The most direct family of tools for that is **attribution** (also called "saliency"): methods that, for one input sample, assign each input cell — each of the seven geometry channels at each position along the field line — a number saying how much that cell contributed to the network's output. The output being explained is the network's native quantity, max(log Q, −2).

The problem is that there are a dozen popular attribution methods, and they often disagree — and some produce plausible-looking heat maps that are actually meaningless. S06a's job was therefore *not* to draw any physics conclusions, but to run a **benchmark**: put ~12 candidate methods through a battery of quantitative sanity checks on one network (the top ensemble member) and a 128-row panel of geometries, and pre-register which one or two methods the scaled-up follow-on step (S06b, which runs the top 10 members) is allowed to use. Picking the method *before* the big run, by fixed criteria, is a guard against cherry-picking the method whose pictures look nicest.

## The methods being compared, in plain terms

Two families were tested:

- **Gradient/path methods.** The simplest is the gradient: how much does the output change if you nudge each input cell? A refinement called **Integrated Gradients (IG)** averages those gradients along a path from a "baseline" input to the actual input. The **baseline** (or reference) is a deliberately bland stand-in geometry meaning "no signal here" — and there's no unique right choice. S06a tried four: a constant profile (per-channel medians), a matched real geometry from the dataset, a "medoid" (the most central real geometry), and a **low-pass** version of the input itself (the same geometry with its fine-scale wiggles smoothed away, so the attribution highlights what the wiggles contribute). **Expected Gradients** is IG averaged over several real-data baselines.
- **Perturbation methods.** Instead of gradients, these actually *edit* the input and watch the output move. **Occlusion** replaces one window of cells at a time; a **mask** method searches for the smallest set of cells whose replacement most changes the prediction.

## The 12 registered candidates

**Gradient/path methods (7):**

1. **Signed robust-scaled gradient** — the plain gradient of the output with respect to each input cell, kept with its sign, and rescaled per channel by S01's interquartile ranges (needed because the seven geometry channels differ in magnitude by three orders of magnitude, so raw gradients aren't comparable across channels).
2. **Integrated Gradients, robust-constant baseline** — IG starting from a flat geometry set to each channel's median value.
3. **Integrated Gradients, matched-observed baseline** — IG starting from a real geometry drawn from the dataset and matched to the sample.
4. **Integrated Gradients, medoid baseline** — IG starting from the single most "central" real geometry in the reference cohort.
5. **Integrated Gradients, low-pass baseline** — IG starting from a smoothed copy of the sample's own geometry, so the attribution measures what the fine-scale structure adds. (This was the eventual winner.)
6. **GradientSHAP / Expected Gradients** — IG averaged over eight different real-data baselines instead of committing to one, which also adds small random noise along each path.
7. **Robust-noise VarGrad** — compute the gradient many times with small random perturbations added to the input, and report the *variance* of the results per cell; a magnitude-only "where is the network twitchy" map.

**Perturbation methods (3):**

8. **Cyclic channel-by-window occlusion** — slide a window around the (periodic) field line, replace the cells inside it channel by channel, and record how much the prediction moves. (Failed the analytic-toy position test, average precision 0.6125 vs. the required 0.75, so it was ineligible from the start of the rerun.)
9. **Periodic extremal mask, matched-observed background** — an optimizer searches for the smallest smooth, periodic set of cells whose replacement with a matched real geometry most changes the prediction. (Retained only as the secondary fallback.)
10. **Periodic extremal mask, robust-constant background** — the same mask search but replacing cells with the fixed z-median profile, added in the rerun specifically because a constant background is shift-invariant by construction. (Failed stable insertion and the symmetry tolerance anyway.)

**Temporal-saliency-rescaled variants (2):**

11. **TSR gradient** — the gradient method with "temporal saliency rescaling," a two-stage scheme borrowed from time-series explanation: first score whole positions along the field line by perturbing them, then weight the per-cell gradient map by those position scores.
12. **TSR robust-reference IG** — the same rescaling applied on top of Integrated Gradients from the robust-constant baseline.

A thirteenth family, **LRP** (Layer-wise Relevance Propagation, which propagates the output backward through the network layer by layer using per-layer rules), was deliberately *not* included: the plan only permitted it after documenting that its rules correctly cover this architecture's circular convolutions, max pooling, biases, and signed inputs, and that documentation was never done — so it's listed as deferred rather than tested.

## The tests each method had to pass

- **Toy recovery:** on a hand-built analytic function where the truly relevant cells are known by construction, does the method find them? (11 of 12 methods aced this, so it's a floor, not a ranking.)
- **Deletion/insertion faithfulness:** rank the cells by claimed importance, then progressively delete (or insert) them in that order and watch the prediction. If the method is honest, deleting its "important" cells should move the prediction much faster than deleting cells in random order.
- **Parameter randomization:** re-run the attribution on a copy of the network with its trained weights scrambled. A method whose map barely changes isn't explaining the *trained* network at all.
- **Symmetry (equivariance):** S02 established the network is exactly invariant to cyclic shifts along the field line. A trustworthy explanation should shift along with a shifted input.
- **Uncertainty:** every headline margin got a confidence interval from a **bootstrap** — recomputing the number on 500 random re-samplings of the data, resampled by whole equilibrium (so correlated rows from the same equilibrium travel together).

## What was found

**1. The selected pair.** The winner among gradient/path methods was **Integrated Gradients with the low-pass baseline** (64 steps): it passed toy recovery, both faithfulness directions in both row strata, the symmetry check under its natural convention, and randomization, then won the registered tie-break. The **periodic mask** was retained only as a secondary sensitivity check, because both of its background variants failed part of the gate — notably, the mask is only shift-equivariant if its background geometry is shifted along with the input; with the background held fixed its symmetry error is huge (1.009, i.e. order-one, versus essentially zero for the co-shifted convention).

**2. The most important negative finding: a "dumb" control nearly matched the winner.** The team built a network-free control map, |X − B| — literally just "how far does each input cell sit from the baseline," computed without ever consulting the network. This control passed the toy test perfectly and passed the earlier random-order faithfulness gates. On **stable/near-floor rows** (the third of the data where the true output is clipped at the −2 floor), low-pass IG did *not* beat this control at all. Only on unstable rows does it demonstrably add network-derived information beyond the control, and even there its edge over the control (~0.01 in native units) is 25–60× smaller than the other baselines' edges. This forced a mid-step correction (the researcher-approved "control-aware selection" rerun): the selection rule was amended so every candidate must beat its own network-free control, with a bootstrap interval excluding zero, in both directions on unstable rows.

**3. Stable rows carry no usable explanation.** For rows sitting at the clipped floor, the network's output barely differs from its output at the baseline (median endpoint difference 0.0014 native units), so attribution maps there are effectively noise divided by nearly zero. The registered standing caveat: S06b must *report* the stable stratum but may make **no feature-level claim** from any method on those rows.

**4. Baseline choice materially changes the map.** The winning low-pass IG map agrees with the constant-baseline IG map only at rank correlation 0.432 — moderate at best. Worse, a rerun of the selection on a separate 64-row pilot panel picked a *different* winner (medoid IG). This pilot-to-production instability is itself a recorded negative result: the selected method is *not* baseline-independent, and S06b is required to carry the medoid and robust-constant baselines along as sensitivity analyses rather than trusting any one map.

**5. The winner's pass on randomization is qualified.** Low-pass IG's map still correlates 0.406 with the map from a weight-scrambled network — the weakest response among eligible baselines — and much of that residual correlation traces to structure baked into the baseline itself rather than learned parameters.

**6. The process caught many real bugs.** Sixteen failures were found and fixed before the final run, including a subtle Captum library issue (Expected Gradients pairing geometry paths with the wrong gradient-drive values), a normalization that flipped faithfulness rankings for stable rows, and a pooled statistic whose denominator canceled between strata and produced spurious verdicts. All are documented, and eleven deliberate code mutations were confirmed to turn the tests red.

## The conclusions

- **Registered for S06b:** low-pass Integrated Gradients as the primary attribution method; the matched-background periodic mask as a labeled secondary perturbation check; medoid and constant-baseline IG carried as mandatory sensitivity analyses.
- **Everything is explicitly a statement about the network, not the plasma.** Both selected methods edit geometries in physically unrealizable ways ("off-manifold" edits — smoothed or cell-replaced geometries no real equilibrium would produce), so their maps diagnose what the *network* responds to, not what drives real ITG turbulence. The physics comparison is deferred to S07.
- **No feature claims from stable/near-floor rows**, ever, from any method.
- **Method selection here is honest but not unbiased:** the corrected rules were designed after seeing results on these 128 rows (a "post-run selection rule"), which the report states plainly rather than hiding.

The overall picture is a benchmark that succeeded at its narrow job — picking a
workable method with eyes open — while surfacing sobering caveats: attribution
maps are substantially shaped by an arbitrary baseline choice, a network-free
control can pass most standard checks, and a third of the dataset is effectively
unexplainable by these tools. Those caveats now constrain how much S06b's
scaled-up maps can be trusted to mean.

----------------------

# Codex response — choosing trustworthy input explanations

## What this step was about

Each network receives seven geometric quantities sampled at 96 positions along
a magnetic field line. An input-attribution method tries to say which of those
672 numbers pushed a prediction up or down. The result is often shown as a heat
map, but a visually appealing heat map can be wrong: it can move when the field
line is merely rotated, ignore a feature whose removal matters, or stay almost
unchanged after the network's learned parameters are erased.

S06a therefore did not interpret any particular geometric pattern. Its purpose
was to choose explanation methods by tests with right and wrong answers before
using them across the ensemble in S06b.

All predictions and explanations use the network's actual output,
$\max(\log Q,-2)$. The exactly rotation-invariant version of the top network,
called $\tilde f$, is the primary object, and the original network $f$ is kept as
a comparison. Stable cases near the clipped floor are reported separately from
unstable cases.

## The path method selected, and the perturbation fallback

The benchmark retained one primary path method. No perturbation method passed
the complete gate, so one mask remains only as a secondary sensitivity.

The researcher-approved rerun compares every path method with a deliberately
simple map that knows nothing about the trained network: it ranks cells only by
how far they lie from that method's reference geometry. A path method must beat
this control in both deletion and insertion on unstable rows, with uncertainty
intervals that stay above zero. Four paths pass; low-pass Integrated Gradients
retains the lowest infidelity score and remains primary.

This rule was written after results on the 128 production rows were already
known. The 64-row pilot was regenerated under it, so the pilot checks the code
and reveals panel sensitivity but is not an independent pre-production test.
The chosen method is a registered S06b benchmark choice, not an unbiased method-
selection estimate.

**Integrated Gradients from a low-pass reference** starts with the real geometry,
smooths away its shorter-scale variation, and measures how the prediction
changes while moving from that smoothed reference back to the original. It
keeps positive and negative contributions. “Integrated” means it averages the
network's local slope along the entire path instead of trusting the slope at
only one point.

**A periodic extremal mask** searches for a compact set of channel-position
cells whose replacement most changes the prediction. Its smoothness penalty
wraps around the field-line boundary, so positions 95 and 0 are treated as
neighbors. It is a magnitude map—important or not—not a signed contribution.
The original matched-background mask remains useful as a secondary diagnostic,
but it is not a symmetry-conforming primary explanation.

The two methods answer different questions. Later work may compare them, but
feature claims must rest on the path primary and ensemble agreement, not on the
secondary mask alone.

## Why they were selected

On a synthetic circular example where the correct channel and four wrapped
positions were known exactly, both methods found the right channel and achieved
perfect position-recovery score (1.0). A random map scored 0.044.

The methods passed the random-order part of the “faithfulness” tests. The
benchmark removes cells in the
order a method calls important and compares that curve with removing cells in a
random order; insertion performs the reverse experiment. On the canonical
128-row panel, the post-run gate is evaluated separately on the 33
stable/near-floor rows and 95 unstable rows. Low-pass Integrated Gradients beat
random by 1.503/0.211 on deletion/insertion in the stable group and 0.420/0.420
in the unstable group. The mask's corresponding margins were 0.431/0.055 and
1.744/1.527. The mask was optimized using the same replacement operation that
this curve scores, so its margins are in-sample optimization results, not an
independent validation.

The previous pooled mask margins, 5.247/5.293, are not interpretable headline
numbers. The stable and unstable endpoint differences have opposite signs
(-0.4915 and +0.2349 native units), so pooling leaves a small denominator and
inflates the ratio. The corrected gate uses each stratum and leaves the selected
pair unchanged.

Uncertainty is also reported without dividing by that unstable denominator.
Because 38% of low-pass rows have a negative endpoint difference, each row is
oriented separately before averaging. Under this convention all four low-pass
stable/unstable deletion/insertion intervals are positive, and 76–80% of rows
favour the method. Both unstable-mask intervals are positive; both stable-mask
intervals include zero. Orienting only by the cohort-mean endpoint instead gives
a weaker low-pass result, so both conventions are published as a material
aggregation sensitivity. The mask evidence remains an in-sample diagnostic
rather than independent validation.

A stronger network-free control ranks cells only by $|X-B|$, deleting the
gradient or learned mask from the explanation. Low-pass IG does not beat this
control on stable rows: the paired deletion and insertion intervals both cross
zero. It beats the control decisively on unstable rows. All five path baselines
were rerun against their own controls; robust-constant, medoid, low-pass, and
Expected Gradients clear the complete rule, and low-pass wins the unchanged
infidelity tie-break. Matched-observed IG beats its control but fails deletion
against random order.

The earlier negative result matters: the simple $|X-B|$ control itself gets
perfect toy recovery and clears all four low-pass random-order faithfulness
cells. The old rule excluded it only because its map is unchanged when the
network is randomized (correlation exactly 1.000). That is why direct
method-versus-control evidence is now part of selection.

The stable-row effect-size outcome is specific to low-pass: its paired estimates
are near zero, while robust-constant, medoid, Expected Gradients, and even the
otherwise-ineligible matched-observed path have estimates roughly two orders of
magnitude larger on those same 33 rows. At only 500 bootstrap resamples, however,
the smallest positive lower bounds (about 0.0005) are within Monte Carlo
resolution, so the distinction does not rest on which intervals cross zero. If
stable rows had also been gated, robust-constant IG—not low-pass—would have won.
This control metric does not show that clipped rows contain feature information,
so it does not relax the researcher-registered stable-row caveat.

The unstable pass is also much smaller in native units for low-pass. Its method-
minus-control gains are about 0.010/0.011, versus 0.24–0.60 for the other
eligible paths. Low-pass wins on normalized infidelity, a relative score that
can favour its unusually short path, so “winner” does not mean largest native
effect. Each path also faces a different $|X-B|$ control, so passing is not a
uniform-difficulty contest across baseline families.

The Integrated Gradients margins are normalized by a small native-unit change.
The low-pass reference changes the canonical output by median 0.0014 and mean
0.0175 across the registered rows. Its median numerical completeness error is
about 5.6% of that median difference. The direction of the faithfulness result
reproduces, but the normalized headline should not be mistaken for a large heat-
flux effect.

Finally, the explanations changed when all learned parameters were reset. Their
rank correlation with the randomized-network maps was 0.406 for Integrated
Gradients and 0.099 for the mask; a value near 1 would have meant the map barely
noticed that learning had been erased. The 0.406 result is the weakest among the
eligible Integrated Gradients baselines. Moreover, the simple difference between
the input and its low-pass reference correlates more strongly with the
randomized-network map (0.816) than with the trained-network map (0.477). Thus
the selected map still responds to learning, but much of its structure is
imposed by the baseline itself.

## The symmetry check

Rotating all seven geometry channels together is physically the same field line
with a different starting point. A good map for the canonical network should
rotate with the input and otherwise remain unchanged. Low-pass Integrated
Gradients passes when its input-derived reference rotates with the input. The
mask also passes only when its matched-observed background is rotated too
(error about $3\times10^{-7}$). The maps intended for S06b keep that background
fixed; under that convention the mask error is 1.009, so fixed-background mask
equivariance fails.

The rerun also tried replacing masked cells with a per-channel constant median,
which has no preferred starting position. This reduces the fixed-background
error to about 0.001 but does not reach S02's $2\times10^{-5}$ tolerance; it also
makes stable insertion worse than random (-0.155). The new mask therefore fails
two clauses and is kept as a negative result.

The original network's errors are about 0.82 for both methods. That is not a new
bug—it is the pooling-phase dependence established in S02. It demonstrates why
the canonical invariant network must remain the primary explained function.

## The most important caveats

The first caveat is **baseline sensitivity**. Integrated Gradients needs a
starting geometry, and different plausible choices give meaningfully different
maps. The selected low-pass map has rank correlation only 0.432 with the map
from a robust constant reference. The regenerated 64-row pilot applies the same
control-aware rule and selects a medoid reference, while the production panel
selects low-pass; the two panels overlapped by only 7 rows. S06b
must carry the other baselines as sensitivity analyses; the chosen map is not a
unique, baseline-free truth.

Stable/near-floor rows remain a standing caveat for **every method**. The
selected low-pass output barely changes under its edits (median endpoint
difference 0.0014 native units). Other baselines have much larger control-
comparison effect estimates, but this faithfulness metric does not establish
feature information in clipped rows. S06b must report the stratum but may not
base feature-level claims on it.

The second caveat is **physical validity**. Smoothing a geometry or replacing
individual cells does not generally produce another realizable magnetic
equilibrium. The mask edits travel as far as 3.8 robust input-scale units. The
reported PCA warning cannot establish off-manifold drift—it is high even for
some observed support rows—so the caveat rests on the artificial edits
themselves, not that warning. These maps explain what the trained network does
under those edits. They do not show that changing a stellarator in that way
would change the plasma.

The third caveat is that only one network was benchmarked. S06a can select
methods, but it cannot say a feature is common across the ensemble. S06b must
run the path primary across the top 10 members, retain signed member-level
results, and quantify member and equilibrium uncertainty; any mask comparison
remains secondary.

## Negative results worth keeping

The simple cyclic occlusion method was not good enough. It recovered the correct
toy channel but its position score was only 0.613, below the registered 0.75
threshold. Its real deletion margins are actually positive in both strata; the
former negative pooled value was a denominator-cancellation artifact. The rerun
therefore removed it from the perturbation candidate list while retaining its
published sensitivity results; the toy failure already made it ineligible.
Integrated Gradients from a matched observed geometry failed deletion in the
unstable stratum.
An observed endpoint does not make the artificial path between two geometries
physically valid or numerically useful.

The implementation and automated review caught four subtle failures: Captum's
Integrated Gradients expanded the input batch without expanding the two scalar
drives; Captum's Expected Gradients used a different batch ordering that paired
some paths with the wrong drives; Expected Gradients also used an unseeded NumPy
random draw; and the first faithfulness normalization reversed the
interpretation when a stable prediction lay below its baseline. Each was fixed,
tested, and the pilot and production artifacts were regenerated. The corrected
Expected Gradients result remained eligible but did not replace either selected
method.

Later review added controls that materially sharpen the conclusion: a
mixed-sign example now pins absolute-magnitude deletion ordering, a nonlinear
example pins the fallback integration rule, and symmetry is reported separately
for co-shifted and fixed baselines. It also exposed the pooled denominator
cancellation, moved the gate to the two floor strata, published per-row-oriented
denominator-free intervals, and pinned whole-equilibrium resampling with sibling
flux tubes. A negative-endpoint analytic case pins the orientation sign. A final
network-free displacement control separates learned ordering on unstable rows
from unresolved stable-row evidence.

## What comes next

The researcher decision is complete. S06b should apply low-pass Integrated
Gradients to the registered top 10 networks, with medoid and robust-reference
Integrated Gradients retained as baseline checks. The fixed-background mask may
be carried only as a secondary perturbation sensitivity method, not described
as a symmetry-conforming primary map. S06b will ask whether the signed maps
agree across networks, how that agreement changes between stable and unstable
conditions, and whether explanation stability has much relationship to the
networks' validation ranking. Only after that ensemble-level agreement exists
should S07 compare the maps with physical fields and GX heat-flux profiles.
