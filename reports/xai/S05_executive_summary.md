# S05 executive summary — What do the network's internal units recognize?

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
the correlation at the same position was almost zero (-0.018), but it became
-0.369 after the correct 23-point shift. A plot that silently assumed zero lag
would have missed the match or placed it at the wrong physical location.

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
- the relationships remained after accounting for the magnitudes of the seven
  individual geometry channels.

But the names remain shorthand, not complete definitions. When we selected only
the most active 5% of positions, unit and concept overlapped at just 7.9% to
13.0% of positions. Random equal-sized masks would overlap at 5%. Thus the
accepted concepts explain part of a unit's organization, but no unit is a pure
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

S05 strengthens the case that the ensemble has learned the paper's main
geometric vocabulary—bad curvature, surface compression, and geodesic
curvature. It also shows why that statement must remain at the ensemble or
concept-family level. Individual important units are broad, shifted,
member-specific mixtures, and most resist a reliable one-line name. That
negative result is useful: later steps should test concepts across layers and
members rather than assume that a unit label discovered in one network transfers
to another.

