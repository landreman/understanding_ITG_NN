# S10 executive summary — what the networks share

## What this step asked

The ensemble contains 100 separately trained neural networks. They make very
similar heat-flux predictions, but that does not guarantee that they learned the
same geometric reasoning. S10 asked whether their internal representations and
their important bottleneck units correspond across networks, and whether any
shared pattern persists outside the ten networks with the best stored validation
scores.

The analysis explains each network's native prediction
$\max(\log Q,-2)$. It uses the exactly shift-invariant version established in
S02, so moving the common origin around the periodic flux tube cannot change the
object being compared. The 1,000-row S01 panel contains one tube from each of
1,000 equilibria: 240 rows where the model is at or near the clipped output floor
and 760 unstable rows above it.

## What “the same internal feature” means here

A bottleneck unit is one of the 7–32 numbers through which all geometric
information must pass before the network makes its prediction. Two units were
matched only when several kinds of evidence agreed:

- they varied similarly over the same equilibria;
- they still varied similarly after removing simple associations with heat flux
  and the two gradient inputs;
- they had similar relationships to named geometry concepts and spatial scales;
  and
- replacing each unit by its typical value changed the two networks' predictions
  similarly.

That replacement is a diagnostic edit inside a network. It is not a physically
realizable change to a stellarator equilibrium, so it explains the networks and
does not by itself establish plasma causality.

The matching was repeated while resampling whole equilibria. This estimates how
often a match would recur if the panel contained a different sample of
equilibria. Units were allowed to remain unmatched; the calculation did not force
every internal number into a shared story.

## Main result: shared motifs exist, but far fewer than a simple comparison suggests

The first comparison found 582 plausible unit pairs among the top ten networks.
497 looked stable under equilibrium resampling and under a compact summary of
their effects. A stricter audit then found a serious problem: that compact
summary could look similar even when the signed effects opposed one another
within the stable or unstable regime.

After requiring agreement separately in both regimes, only **163 pairs** remained.
They form **eight shared motifs**, with at most one unit from each network in a
motif. Five motifs occur in at least four of the top ten networks; the largest
occurs in nine. The correction rejected 334 preliminary edges and is scientifically
important: averaging across regimes would have produced 33 apparently shared
motifs instead of eight.

Only one of the eight motifs contains a unit that S05 had given a supported
physical name. It includes the leading network's unit associated with a
parallel-window average of the paper's $f_Q$ integrand, and corresponding units
occur in six other networks. But those six units have not independently earned
that name. It is a promising anchor, not a seven-network identification. The
other seven motifs remain unresolved by the current concept vocabulary.

## The networks look similar in broad outline

Centered Kernel Alignment, or CKA, compares the geometry of two internal
representation spaces even when they contain different numbers of units. A CKA
score near one means that examples have similar relative arrangements in the two
spaces; it does not prove that the networks perform the same computation.

Across all 4,950 pairs of networks, median CKA decreases from **0.948** in the
first spatial layer to **0.814** at the final invariant bottleneck. The networks
therefore share a broad representational scaffold, while becoming more individual
deeper in the calculation. Removing the 5% most extreme probe examples changes
the median score by only 0.006–0.022, so this pattern is not usually driven by a
few outliers.

The uncertainty ranges for CKA use only 20 resampled panels. They are a coarse
sensitivity check, not a precise confidence interval. The exact all-pair scores
are complete; a planned 100-resample version exceeded the step's computation
budget and is recorded as deferred.

## Validation rank is not the organizing principle

Ninety-five of the 100 networks fall into one main cluster when predictions,
scaled input sensitivities, bottleneck interventions, and concept profiles are
combined. Five are outliers. Distance from the most central member has only a
weak relationship with stored validation rank (rank correlation 0.118, with
$p=0.243$). Networks ranked 51–100 are no less similar at the bottleneck than
the top ten: their median within-cohort CKA is 0.816, compared with 0.796 for the
top ten.

The four narrow networks, with at most 11 bottleneck units, are more unusual in
the combined evidence. Their median distance to wide networks is 3.153, compared
with 1.177 between two wide networks, and three of the four sit outside the main
cluster. Yet their bottleneck CKA with wide networks is almost unchanged
(0.813 versus 0.814 between wide networks). The narrow networks do not simply
lose the common representational scaffold; they differ in how they use it.

## What to carry forward

The strongest conclusion is methodological and scientific at once: shared
activation patterns are abundant, but shared signed effects in both physical
regimes are much rarer. S11 and S12 should use the eight strict motifs, retain
the regime-specific signs, and keep the other 334 preliminary correspondences as
negative evidence rather than silently recovering them through averaging.

The seven unnamed motifs are also useful. They say that the networks repeatedly
compute something similar that the current physical vocabulary has not captured.
That is a concrete target for later disagreement analysis and interpretable
distillation, not a reason to attach a post-hoc name.
