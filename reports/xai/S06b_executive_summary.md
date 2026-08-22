# S06b executive summary — what the ensemble's input maps agree about

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
averaged over the chosen rows. That is substantial, but not enough to describe
the ensemble as one shared position-by-position mechanism.

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

The response is also larger on rows where members make larger errors or disagree
more. On varied rows, the largest-channel value is about five times higher in
the high-error third than in the low-error third, and about five times higher in
the high-ensemble-spread third than in the low-spread third. That makes the maps
potentially useful for the later disagreement study, but it does not establish
cause: flux regime and stability status also differ across those groups.

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
