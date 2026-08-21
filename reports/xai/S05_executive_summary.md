# Claude summary

## The background you need

**The network.** The trained model takes a flux tube — seven geometric quantities (B, |∇x|², curvature drifts, metric coefficients…) sampled at 96 points along the field line — plus two scalar drives (temperature and density gradient, g_T and g_n), and predicts `max(log Q, −2)`, the heat flux on a log scale with a floor for "essentially stable."

**Convolutional, cyclic, and the "bottleneck."** Internally the network does two things in sequence:

1. A stack of *convolutional* layers slides small filters along the 96-point field line. A filter is a little weighted template; sliding it produces a new 96-point trace saying "how strongly does this pattern occur here?" Stacking layers builds up more complex patterns. "Cyclic" means the field line is treated as a loop that wraps around, so the answer doesn't depend on where you chose to start indexing.
2. Then everything is **averaged over the 96 positions** into a short vector of numbers — roughly 10–23 of them per network — and only that short vector (plus g_T, g_n) is fed to the final small dense network that emits the prediction.

That averaging step is the **bottleneck**: it is the only channel through which geometry reaches the answer. Each number in it is a **unit**. Because it's a spatial average, each unit has a hidden pre-average 96-point trace — call it a **density** ρ(z) — and the unit's value is just the mean of its density. A unit that is always zero is "dead"; one that varies is "live." Of the two networks studied here there are 9 and 20 live units respectively, 29 total.

**Ensemble members.** The model is 100 separately trained networks whose predictions get averaged. Two of them are studied here: `2864601_0.437` (best on stored validation, "top member") and `2864601_0.371` ("replication member"). Anything true of only one is not a fact about the model.

**What S04 established (the prior step).** The bottleneck units account for ~19.5% of the output variance (the two gradient scalars dominate), and the bottleneck strongly encodes the human-designed geometric features from the paper — in particular you can decode log f_Q from it with R² ≈ 0.88, and deleting that direction really does change the output. So the bottleneck *contains* recognizable physics. S04 also ranked the units by importance.

## What S05 was for

S04 said "the bottleneck knows about f_Q-like geometry" collectively. S05 asks the next, harder question, one unit at a time:

> **Does each important bottleneck unit correspond to a nameable local physics quantity?**

Concretely: take unit `u001`'s 96-point density ρ(z), compute 75 candidate physics traces from the same geometry (bad curvature, flux compression |∇x| and its powers, the f_Q integrand, radial drift / geodesic curvature, local shear, B minima and wells, parallel roughness/Fourier scale — each also smoothed over 9- and 25-point windows), and ask whether ρ(z) tracks any of them along the field line. If yes, you've named that unit — "this neuron measures bad curvature." That's the dictionary the whole XAI program wants.

Three technical choices matter for reading the results:

- **Correlation along z, per flux tube.** Not "does this unit's average value correlate with f_Q across tubes" (that's S04's question) but "does its spatial pattern within a tube track the concept's spatial pattern." Values run −1 to +1; sign just tells you whether high density goes with high or low concept.
- **Lag.** The match is searched over all 96 circular shifts, because a deep convolutional stack can respond to a feature at an offset from where the feature sits. The best shift is reported, not assumed zero.
- **Bootstrap over equilibria.** Uncertainty is estimated by resampling whole stellarator equilibria (not individual flux tubes), because tubes from the same equilibrium aren't independent. **Recurrence** = the fraction of 500 such resamples where the same concept still wins.

Two thresholds were fixed *before* running: |correlation| ≥ 0.20 and recurrence ≥ 0.50.

## The conclusions

**1. There is no clean one-name-per-unit dictionary.** Only **6 of 29** live units clear those deliberately modest thresholds. The other 23 are left explicitly unnamed rather than given plausible-sounding labels.

**2. The named ones all fall in the paper's own feature family.** Five align with bad-curvature / flux-compression / f_Q-integrand, one with radial drift / geodesic curvature. So where the network is legible at all, it is legible in the physics vocabulary humans already wrote down — consistent with S04.

**3. The best single result.** The top member's most important unit, `u001`, correlates **−0.369** with the 25-point smoothed f_Q integrand at lag +23, bootstrap interval [−0.396, −0.344], recurrence 0.994. Three of that member's top five units pass.

**4. It doesn't replicate.** In the second member, only one of the top five units gets a supported name, and the *dominant* unit misses at +0.192 — just under the 0.20 line, and deliberately not rounded up. So: the *concept family* recurs across members; the *specific identity of the most important unit* does not. Different trained networks build different individual neurons out of the same physics ingredients.

**5. Lag is real, not a technicality.** The six supported lags are +18, +23, −39, −15, −1, +11 grid points on a 96-point loop. For `u001`, the zero-lag correlation is **−0.006** — nothing — versus −0.369 at lag +23. Practically: you cannot say "this unit fires where the bad curvature is." And the reason large lags are possible is that by the final layer each unit's receptive field (the span of input it can see) wraps the entire domain — formally 180 and 330 points against a 96-point loop. Every unit is globally connected, so a matched lag may reflect distributed computation rather than a local detector.

**6. Even the six supported alignments are weak descriptions.** Three independent checks all say "correlated, not identical":

- Correlation of 0.3–0.4 means the concept explains roughly 10–16% of the density's spatial variance.
- **Overlap check:** take the top 5% of positions by density and by concept, ask how much they coincide — 0.107 to 0.421, but the chance baseline for those same rows is already 0.071 to 0.379. Only **1.10–1.51× better than chance**.
- **Partial correlation:** after statistically removing the seven raw channel magnitudes at each position, correlations move to −0.442…+0.237. Four survive above 0.33; the weakest (`u011`) collapses from +0.257 to +0.123.

**7. Selection inflation was checked, partially.** Searching 75 concepts × 96 lags will find *something* even in noise. Eight row-permutation draws repeating the entire search show the real correlations are 4.09–6.69× the null's 95th percentile — reassuring, but with eight draws it is a calibration diagnostic, not a p-value, and the report says so.

**8. No evidence for clean polysemantic units either.** Clustering each unit's 16 strongest natural exemplars mostly yields 15+1 or 14+2 splits — the algorithm isolates one odd equilibrium rather than finding two genuine recurring motifs. That's negative evidence for "this neuron does two distinct jobs," not proof it does one.

**9. Stable rows behave differently.** The panel has 240 stable/near-floor rows (at the log-flux floor) and 760 unstable. Signs and lags mostly hold across both, but for the top member's `u000` the winning concept recurs in 90.4% of unstable-row bootstraps and only 13.6% of stable-row ones. The name is a claim about the unstable regime.

## The honest summary

S05 was supposed to convert "important units" into "named local physics densities." It succeeded for a minority — six units, all in the paper's own curvature/compression/f_Q family, each at a large and explicitly reported spatial lag — and failed for the rest, including the dominant unit of the second network. Combined with the globally-connected receptive fields and the barely-above-chance overlap enrichment, the reading is that these networks don't store their geometry knowledge as a tidy set of individually nameable local feature detectors. The knowledge is there (S04 showed the bottleneck is highly decodable), but it's distributed across units rather than factored into them.

Two things were deliberately dropped and recorded under `## Deferred`: the per-unit local surrogate model (pointless when the receptive field is global — it would need 672 raw inputs, not a small local regression) and NMF/dictionary learning (not justified once the clustering diagnostic came back negative).

One process note worth knowing: the automated review caught a memory-allocation
bug in the ranking code that could leave partial-correlation ranks
uninitialized. The production run was **discarded and regenerated** — that's
what commits `f0de959`/`5be3fbb` on this branch are. The numbers above are from
the corrected run.

-------------------

# Codex summary — What do the network's internal units recognize?

## What this step was about

The neural networks first turn the seven geometry curves into a small set of
internal numbers, called bottleneck units, before predicting the heat flux. S04
showed which of those units matter to the prediction. S05 asked the next natural
question: can we say what each important unit is measuring along the field line?

We compared each unit's spatial activation pattern with physically motivated
patterns from the paper. These included bad magnetic curvature, flux-surface
compression $|\nabla x|$, the paper's combined $f_Q$ feature, geodesic
curvature, local shear, magnetic wells, and parallel length scale. We studied
the best validation-ranked network and repeated the analysis in the
second-ranked network, using 1,000 real flux tubes from 1,000 different
equilibria.

## Main conclusion

There is a real recurring theme, but not a clean dictionary.

Six of the 29 active internal units could be matched reliably enough to a named
geometric pattern. Five of those six involved the paper's main theme: strong
flux-surface compression in bad-curvature regions. The sixth involved radial
drift, which is closely related to geodesic curvature. This is encouraging
agreement between the neural network and the earlier, independent feature
analysis in the paper.

However, 23 units could not be named reliably. More importantly, the result did
not replicate cleanly for the most influential unit. In the first network, the
dominant unit matched a smoothed version of the paper's $f_Q$ feature. In the
second network, the dominant unit's best match was too weak to accept. Only one
of that second network's five most important units received a supported name.
The honest conclusion is therefore: the networks repeatedly use the same broad
physics family, but they divide and encode it differently.

## Why spatial lag matters

A convolutional network can respond to a feature some distance away from the
position where its unit activates. We therefore allowed each candidate pattern
to slide around the periodic field line and recorded the best shift, called the
lag.

This changed the answer substantially. The six accepted matches had lags from
-39 to +23 of the 96 grid points. For the strongest unit in the first network,
the Pearson correlation at the same position was almost zero (-0.006), but it
became -0.369 after the correct 23-point shift. A plot that silently assumed
zero lag would have missed the match or placed it at the wrong physical
location.

These large shifts are plausible because the final units can see the entire
field line. They are also a warning: these are distributed computations, not
simple local detectors.

## How strong are the names?

The accepted names passed several checks:

- the winning concept recurred in 95% to 100% of 500 resamples of whole
  equilibria;
- each unit was illustrated by 16 naturally occurring examples from 16
  different equilibria;
- shifting the input around the periodic domain shifted the unit patterns with
  maximum numerical error only $1.0\times10^{-5}$; and
- each observed correlation was at least 4.09 times the 95th percentile of a
  diagnostic null that repeated the complete search after permuting sample
  pairings.

But the names remain shorthand, not complete definitions. ReLU produces many
ties at zero, so a tie-aware nominal top-5% mask often contains more than 5% of
positions. After using each mask's actual size, unit/concept overlap is only
1.10 to 1.51 times its chance baseline. A second check accounting for the seven
channel magnitudes at the unit's activation position leaves four relationships
strong, but weakens one sparse unit from +0.257 to +0.123. It does not control
the geometry at the shifted feature's source position. Thus no unit is a pure
"bad-curvature detector" or "$f_Q$ detector."

## Stable and unstable cases

The analysis kept stable or near-stable simulations separate from unstable
ones, because the model output is clipped at the stable floor. The accepted
relationships generally kept the same sign in both regimes. Still, one unit's
winning concept was much less stable under resampling within the 240 stable
cases than within the 760 unstable cases. The overall unit names should not be
read as equally secure descriptions at the stability boundary.

## Natural examples and possible multiple meanings

For each unit we collected its 16 strongest naturally occurring activations and
aligned them only by a recorded circular shift. We did not optimize synthetic
geometry curves. A simple two-group analysis usually separated one unusual
equilibrium from the other 15, rather than finding two balanced, recurring
motifs. That is negative evidence for cleanly separated multiple meanings, but
it is not proof that the units have only one meaning.

## Relationship to the original network

The main analysis used the exactly shift-invariant version chosen in S02. We
also evaluated the original trained network. Their predictions differed by
about 0.10 in the native clipped-log units for each member, with smaller
differences on stable cases. The important original and invariant bottleneck
values were correlated above 0.99, but not identical. The report keeps these as
separate objects rather than silently treating the invariant model as the
original one.

## Bottom line

S05 strengthens the case that these two members have learned the paper's main
geometric vocabulary—bad curvature, surface compression, and geodesic
curvature. It also shows why that statement must remain at the concept-family
level. Individual important units are broad, shifted,
member-specific mixtures, and most resist a reliable one-line name. That
negative result is useful: later steps should test concepts across layers and
members rather than assume that a unit label discovered in one network transfers
to another.
