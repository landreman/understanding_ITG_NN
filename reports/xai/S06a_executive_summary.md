# S06a executive summary — choosing trustworthy input explanations

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

Stable/near-floor rows are a standing caveat, not an interpretation cohort.
Their clipped output barely changes under low-pass edits (median endpoint
difference 0.0014 native units), so S06b must report them but must not base
feature-level claims on their maps.

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
former negative pooled value was a denominator-cancellation artifact.
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
