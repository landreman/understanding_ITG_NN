# S07 executive summary — Does the network focus where GX transports heat?

## The question

Earlier steps found two kinds of spatial pattern inside the neural network:

1. A **density** is where an internal network unit activates along the 96-point
   periodic field line.
2. An **attribution** is where an input cell receives credit or blame for a
   prediction. Here the primary attribution method is Integrated Gradients: it
   divides the prediction change from a reference geometry among the input
   cells. That reference path is **off-manifold** (not guaranteed to consist of
   valid plasma equilibria), so the result describes how the network behaves
   along a constructed path, not how the plasma responds to a realizable edit.

S07 compares both patterns with `Q_avgs_vs_z`, the heat flux as a function of
position computed by GX. GX $Q(z)$ is a held-out physical diagnostic. The
network never sees that position-resolved curve as an input and was not trained
to reproduce it position by position. Similarity is therefore interesting
independent evidence—but it is still association, not causation.

The analysis uses three independently trained ensemble members and 1,000
equilibria. It keeps stable/near-floor and unstable simulations separate, keeps
signed member results before any summary, and explains the native model output
`max(log Q, -2)` rather than exponentiating it.

## Main answer

Some internal activation densities resemble GX $Q(z)$, but the prediction
attributions do not provide the signed, member-replicated evidence needed to
call this a physical mechanism.

The top member's most important internal unit has a spatial rank correlation of
**-0.361** with GX $Q(z)$ on varied-gradient unstable simulations. Rank
correlation measures whether two curves order positions similarly; -1 is
perfect reverse ordering and +1 is perfect matching ordering. The 95% interval
is **[-0.388, -0.333]**, obtained by resampling whole equilibria to estimate
uncertainty. The best match occurs at lag **+22** of 96 grid points, not at the
same coordinate. Its association is negative: high activation goes with low
$Q(z)$. After sign-flipping $Q(z)$ to measure that inverse pattern, their
top-10% regions overlap **1.63 times chance**. Without the sign flip, high
activation and high $Q(z)$ overlap only **0.54 times chance**—they avoid one
another more than chance predicts.

A second unit previously associated with radial drift/geodesic curvature has
correlation **-0.268 [-0.294, -0.241]** at lag **+44**. The same densities are
nearly unchanged in the fixed-gradient comparison because density depends only
on geometry; this is a robustness comparison against a second GX field, not an
independent replication. These are credible activation-to-physics associations.

But the off-manifold attribution diagnostic gives a different picture. Keeping
the full sign of Integrated Gradients, its correlations with $Q(z)$ are only **-0.021,
-0.013, and -0.012** in the three members, with incompatible lags **-36, +47,
and +48** on 760 unstable equilibria. Selecting the largest value from 96 lags
can create a small peak by chance, so we also broke the equilibrium pairing 200
times and repeated the full search. The -0.013 peak lies at the estimated null
threshold, so its binary pass/fail label is not stable; the -0.012 peak is below
the threshold. The -0.021 peak clears it, but is still negligible and does not
replicate.

If all negative contributions in the off-manifold diagnostic are thrown away, the correlations rise to
**+0.266, +0.280, and +0.262** at lag 0 or +1. That is a genuine and repeatable
resemblance along that constructed reference path: positively contributing
input cells tend to sit where $Q(z)$ is positive. It is also only half the
network explanation, not plasma evidence. Negative contributions are
part of the model's prediction, so the positive-only result cannot replace the
nearly null signed result.

## The zonal-flow hypothesis

The geodesic-curvature candidate is modestly associated with the GX zonal-flow
observable in the varied-gradient unstable panel: **-0.122 [-0.183, -0.060]**.
In the fixed-gradient panel it becomes much stronger, **-0.513 [-0.564,
-0.461]**. The top bad-curvature candidate changes from an unresolved
**+0.032 [-0.040, +0.099]** to **+0.310 [+0.247, +0.372]**.

This drive dependence is scientifically useful negative evidence. It argues
against reading either unit as a simple geometry-only monitor of zonal flow.
The observed relationship depends strongly on which simulation drives are
present.

## What the fixed/varied pairs show

For the same geometry, the fixed and varied simulations have substantially
different heat flux. Across all rows, fixed minus varied GX flux is **+1.559
[+1.434, +1.694]** in the native clipped-log units. It is +3.356 when either
simulation is stable/near-floor and +0.957 when both are unstable, so the pooled
value must not be read as a single-regime effect. Each member predicts almost
the same all-row difference. The physical $Q(z)$ curves remain strongly related:
spatial rank correlation **0.736 [0.710, 0.760]** over all rows and **0.874
[0.860, 0.888]** over the 749 both-unstable pairs, both at lag 0.

This is a natural paired comparison, not a constant-drive comparison within
each pair. The fixed panel uses $(a/L_T,a/L_n)=(3,0.9)$ across geometries, while
the paired varied row can have different drives. It controls geometry but does
not by itself isolate the effect of drive.

## Contradictions were kept

For each of the two physical hypotheses, S07 publishes five naturally occurring
equilibria that support the population sign and five that contradict it. Some
contradictions are strong: for the dominant unit, supporting per-row
correlations reach about -0.95, while a contradicting row reaches +0.86. These
are not discarded as outliers. They show why a population association is not a
one-to-one physical definition of the unit.

## Bottom line

The networks contain activation patterns that track real GX structure, especially
for the two bad-curvature/flux-compression and geodesic-curvature candidates
supported in S05. Four other selected units were already unresolved in S05, and
the three units from the third member were not studied there. The cross-member
comparison is therefore conditional on an importance-ranked, mostly unnamed set,
not an independent replication of the two named candidates. Moreover:

- density signs and lags do not replicate across members;
- complete signed off-manifold attributions have almost no spatial association
  with $Q(z)$; one peak is below the 96-lag null, one is at its threshold, and
  the resolved peak remains negligible;
- the stronger attribution result appears only after negative evidence is
  removed;
- the zonal-flow relationships change greatly with the drive panel; and
- strong natural contradictions exist for both hypotheses.

The appropriate conclusion is therefore narrower than “the network discovered
the GX transport mechanism.” It discovered internal spatial representations
that covary with GX transport structure, but S07 does not establish where or
how the model uses geometry in the same sense that the plasma produces
$Q(z)$. No candidate is promoted to a physically supported mechanism.

The full numerical record, including all lags and negative results, is in
[the technical report](S07_physics_alignment.md) and
[the registered artifacts](S07_artifacts/manifest.json).
