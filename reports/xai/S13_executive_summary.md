# Codex summary — Which learned ideas now have physical evidence?

## What this step was for

Earlier steps showed what the neural networks use. That is not yet the same as
showing what controls turbulence in a real plasma equilibrium. S13 moved one
step closer by asking whether the leading candidate geometry quantities track
independent GX simulation results when the temperature and density gradients
are held fixed.

The analysis used the same 1,000 equilibria registered in S01. Every fixed-drive
simulation has $(a/L_T,a/L_n)=(3,0.9)$, so differences across rows are associated
with geometry rather than different imposed drives. The primary quantity is GX
`max(log Q,-2)`: the clipped logarithmic heat flux that the networks were
trained to predict. There are 23 rows at or near the floor and 977 unstable
rows; they are kept separate.

Four candidates were tested:

1. the largest 25-point running average of the paper's $f_Q$ integrand—a measure
   of the worst localized patch of bad-curvature-weighted compression;
2. the paper's $f_{\rm stab}$ stability feature;
3. variation in bad-curvature/compression along the field line; and
4. geodesic curvature weighted by flux-surface compression.

All comparisons use naturally occurring, force-balanced equilibria. No geometry
channel was edited by itself.

## Main conclusion

All four candidates have physical **observational** support, but none is ready
to be called a physical mechanism.

At fixed drive, their rank correlations with true GX heat flux are:

- localized $f_Q$ peak: **0.807**;
- bad-curvature/compression variation: **0.808**;
- $f_{\rm stab}$: **0.659**; and
- geodesic-curvature/compression: **0.519**.

A rank correlation measures whether larger values of one quantity tend to go
with larger values of another, from -1 for perfect reverse ordering to +1 for
perfect matching. These are strong positive associations. Their uncertainty
ranges, obtained by resampling whole equilibria 500 times, all stay above zero.

But the natural comparisons do not isolate one feature from the others. The
features are tightly bundled together across the available equilibria. That
confounding—several geometry properties changing together—is the dominant result
of the stricter checks.

## What matching showed

We paired equilibria from the high and low quarter of each candidate while
trying to keep twelve other geometry summaries and equilibrium class similar.
This nearest-neighbour matching produced 185–215 disjoint pairs per candidate.
The high-candidate member of each pair had 1.32–1.88 higher clipped-log heat
flux on average.

However, matching did not make the pairs genuinely comparable. Balance is
measured by a standardized mean difference: a difference divided by the typical
spread of that nuisance variable. Values below 0.5 were required before calling
a candidate ready for intervention. The best candidate still had **1.07**;
the others ranged from **2.19 to 4.15**. In three cases the worst imbalance
became larger after matching because no low-candidate equilibria with genuinely
similar nuisance geometry exist in this panel.

That failure was kept visible. The precise matched differences are not causal
effects.

## The adjustment check is specification-sensitive

A second analysis used AIPW, a “doubly robust” estimator. It combines two fitted
models: one predicts whether a row belongs to the high or low candidate tail,
and the other predicts the GX outcome from the nuisance variables. “Doubly
robust” means one of those two fitted models may be wrong under certain
assumptions; it does not protect against missing confounders or lack of
comparable rows.

Comparable high/low rows were scarce. The fraction with acceptable overlap was
only **0.232–0.478**, below the required 0.8. The adjusted numbers are therefore
sensitivity warnings rather than reliable effect estimates.

Both potential-outcome models now use one common scale fitted on the complete
training fold. Under that specification, geodesic/compression has adjusted
contrast **+0.559 [0.318, 0.764]** and bad-curvature/compression **+0.203
[0.001, 0.476]**. The localized peak is unresolved: **+0.084 [-0.145, 0.380]**
over all rows and **-0.011 [-0.161, 0.141]** among unstable rows. Scaling each
tail separately had produced a resolved -1.09 unstable value. Its disappearance
under common scaling shows that the apparent sign reversal came from nuisance-
model specification, not a measured physical contradiction.

Matching has a parallel warning. The candidate/heat-flux difference grows with
leftover nuisance distance (rank correlations **0.71–0.79**). In the closest
quarter of pairs, the mean contrasts shrink from **1.32–1.88** to **0.41–0.79**,
although all four signs remain positive. Much of the full matched association is
therefore concentrated where comparison quality is worst.

## Which feature adds information beyond the published formulas?

We also asked whether each candidate predicts the part of true GX heat flux left
unexplained by the paper's selected quantities. The baseline contains the two
drives, $\log f_Q$, $f_{\rm stab}$, and
$\log\langle|\nabla x|\rangle$. An Explainable Boosting Machine—a regression
model that learns a smooth curve for each named input—was fitted in five folds,
always holding out complete equilibria.

The baseline reproduces **81.25%** of fixed-panel heat-flux variation. Adding
geodesic-curvature/compression raises this to **82.65%**, a gain of **1.39
percentage points**. The mean squared error improvement is **0.0188**, with a
95% range **[0.0062, 0.0325]** that stays above zero.

Under the registered fold assignment, the localized peak and
bad-curvature/compression variation each add about 0.6 percentage point, but
their uncertainty ranges include no improvement. That "only geodesic resolves"
statement is fold-dependent: over seven assignments geodesic resolves **7/7**
times and has the largest gain every time, bad-curvature resolves **5/7**, and
the localized peak **4/7**. The EBM's own random seed is held fixed, so this
changes only which equilibria enter each fold. Geodesic's first place is robust; exclusivity under
one split is not.

The result is panel-dependent too. On the varied-gradient panel all three
non-baseline candidates resolve, and geodesic has the smallest gain
($\Delta R^2=0.00633$ versus **0.00978** and **0.01053**).

This does not make the geodesic candidate causal. It makes it the best next
candidate to test.

## Other physical diagnostics

`Q_stds`, which measures variability during a GX simulation, tracks heat flux
so closely that its candidate correlations nearly duplicate those for heat
flux. It is not independent confirmation.

The localized $f_Q$ peak and bad-curvature/compression variation correlate with
how concentrated $Q(z)$ is along the field line at about **0.49–0.51**. The
geodesic candidate is only **0.10**. None of the four predicts what fraction of
the field line has positive $Q(z)$.

For zonal-flow magnitude, $f_{\rm stab}$ has the strongest raw association
(**0.592**), while the geodesic candidate is **0.315**. So the fixed-drive data
do not single out the geodesic candidate as a special zonal-flow mechanism.

## Claim grades

The report uses three levels:

- **Model-mechanistic:** the networks use the feature.
- **Observational-physical:** the feature also covaries with held-out GX physics
  among observed equilibria.
- **Intervention-ready:** natural comparisons are sufficiently balanced and
  overlapping to give a defensible expected intervention.

All four candidates reach observational-physical. None reaches
intervention-ready. Geodesic-curvature/compression ranks first because its
fixed-drive residual gain is largest and resolved in all seven fold assignments.
Bad-curvature/compression ranks second because it has the strongest raw
correlation, a resolved common-scale adjusted contrast, and resolves beyond the
paper baseline in five assignments. The localized peak and $f_{\rm stab}$ tie
behind them. $f_{\rm stab}$ is
already part of the full paper baseline, so its gain against the weaker
$f_Q$-only baseline is published but does not receive a comparable ranking
point; candidate name is used only to display the tie deterministically.

This physical ranking is deliberately not the same as upstream network
recurrence. In S12, geodesic/compression recurred in only **0.27–0.60** of
member bootstrap fits, versus **0.83–0.87** for the localized peak and
**0.83–0.90** for $f_{\rm stab}$. Testing geodesic first asks whether a less
consistently learned signal is nevertheless real physics; it does not call that
signal the ensemble's most stable rule.

Five supporting and five contradicting matched pairs for every candidate are
published; contradictory cases were not discarded.

## The proposed decisive GX experiment

The next calculation should compare the top two competing directions directly:

1. change geodesic-curvature/compression while holding bad-curvature/compression
   and key global properties close to the anchor;
2. change bad-curvature/compression while holding the geodesic candidate and the
   same global properties close.

The change must be made through a VMEC boundary continuation, meaning the
boundary shape is changed and a new force-balanced equilibrium is solved each
time. Geometry channels must never be edited independently.

The second direction still has a feasibility risk: bad-curvature/compression
has rank correlation **0.9532** with global $\log f_Q$, and its heat-flux rank
correlation after holding $\log f_Q$ fixed is only **0.1592**. For geodesic the
partial correlation is **0.3343**. The pre-budget VMEC screen is therefore a
real decision gate, not paperwork.

Before asking for a GX allocation, VMEC-only searches must show that both signed
directions are realizable at all three anchors. The candidate must change by at
least **0.5 panel IQR** while every constrained quantity changes by at most
**0.1 panel IQR**. If either direction fails, it is replaced or dropped and the
proposal returns to the researcher without launching GX.

The minimal GX design uses three typical unstable anchor equilibria, positive and
negative changes for each of the two candidates, and two drive points. That is
**24 standard GX runs**. Six decisive cases are repeated with doubled resolution
and longer averaging. Controls include a zero-change continuation, an
orthogonal direction, equal-size plus/minus boundary changes, and rerunning the
original anchors.

A decisive response must be at least **0.2 native-log units** and exceed two
combined `Q_stds` standard errors; those are prospective decision thresholds,
not effects already measured here.

The planning envelope is **32.5 Perlmutter node-hours**: 12 for standard runs,
12 for convergence runs, 2 for equilibrium searches, and 25% contingency. This
is not based on a measured Perlmutter pilot. If approved, one standard and one
high-resolution run should be timed first, and the estimate should be replaced
with those measurements.

## Bottom line

The fixed-drive panel confirms that the learned candidates live in physically
meaningful parts of geometry space, but it also shows why observational
correlation is not enough. The geometry quantities are too tightly linked to
separate their effects from existing data. Geodesic-curvature/compression is the
best next test because it adds a small, repeatable amount beyond the paper's
selected formulas. The localized $f_Q$ peak is the essential competing test
because it has the strongest raw support and the strongest sign contradiction.

No new equilibrium or GX simulation has been launched. The proposed 32.5-node-
hour experiment remains behind the VMEC realizability check and then the
researcher approval gate.
