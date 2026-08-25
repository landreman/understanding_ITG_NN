# S12 executive summary — Can the networks be rewritten in physical language?

## What this step was for

The neural networks make accurate heat-flux predictions, but their internal
calculations are difficult to read. This step asked whether a short list of
physically named, cyclic-invariant quantities could reproduce what the networks
do. “Cyclic-invariant” means that the quantity does not change when the starting
point along the periodic flux tube is moved; that is an exact symmetry the
physics should respect.

We built a fixed table of 17 quantities. It includes the temperature and density
gradients, the paper's $f_Q$ and $f_{\rm stab}$ features, flux-surface
compression in bad curvature, geodesic curvature, parallel length-scale
summaries, local shear, and a few windowed combinations suggested by earlier
steps. The list was fixed before the final calculation.

We then used an Explainable Boosting Machine, or EBM. An EBM is a flexible
regression model that draws one response curve for each input quantity and adds
only a small specified set of two-feature interactions. It is much easier to
inspect than a neural network, while still allowing curved rather than purely
linear relationships.

## Main conclusion

The compact physical vocabulary reproduces the outputs of three top neural
networks surprisingly well, but it does not give a simple interpretation to
every internal unit.

On equilibria excluded from each EBM fit, the fraction of member-output
variation reproduced was:

- 86.03% for member `2864601_0.437`;
- 85.61% for member `2864601_0.371`; and
- 86.36% for member `2864601_0.409`.

The three-member mean was 86.32%. Thus the result is not an artifact of averaging
different networks together: all three individual networks are reproduced at
nearly the same level.

The same feature table predicts the true clipped logarithmic GX heat flux at
83.92%. This is a separate question. The approximately 86% numbers say how well
the readable model imitates the neural networks; 83.92% says how well it predicts
the physical simulation target. Keeping those two scores separate prevents good
imitation of a network from being mistaken for new physical validation.

## What seems stable

We repeatedly refit the EBM after resampling complete equilibria. This procedure
is called a bootstrap: it tests whether a finding survives plausible changes in
which equilibria are represented.

The two drive terms, $a/L_T$ and $a/L_n$, appeared among the five most important
terms in every refit for every target. Among geometry quantities,
$f_{\rm stab}$ appeared in 83–90% of member refits. A 25-point-window summary of
the $f_Q$ integrand appeared in 83–87%. This is replicated evidence for a
feature family, not proof that one algebraic expression is uniquely correct;
several of the geometric quantities are correlated.

Only five two-feature interactions were allowed. The largest fitted interaction
surface for all three members combined density gradient with bad-curvature/
compression. This is an association found among observed equilibria. It does
not show what would happen if one geometry channel could be changed by itself,
and it should not be described as a causal plasma result.

## The important negative result

The compact vocabulary is much less successful at explaining individual
bottleneck units—the internal summary numbers immediately before a network's
final prediction layers.

There are 64 such units in the three networks. Five are completely inactive on
the panel. Among the 59 active units, the median fraction of variation reproduced
is only 59.42%, and just 13 of all 64 units reach 80%. The first member is easier
to summarize than the other two: its median is 78.21%, compared with 45.28% and
60.37%.

This is useful rather than disappointing. It says that a compact set of physical
features can reproduce much of the final behavior even though the networks do
not organize every internal unit as a clean, separately named physical detector.
That matches the earlier S05 result, where most units could not be given a
supported one-phrase interpretation.

Rows near the model's floor value also need care. Their output variance is so
small that the usual fraction-of-variation score becomes strongly negative and
misleading. The report therefore keeps those negative numbers visible but uses
mean squared error for that stratum. On unstable rows, where the score is
meaningful, the three readable models reproduce 81.15–82.68% of member-output
variation.

## What was not completed

The plan also proposed PySR, a symbolic-regression program that searches for
short algebraic expressions and produces a tradeoff between simplicity and
accuracy. The installed Julia version is 1.12.6, while PySR 1.5.10 supports
Julia 1.10.3–1.11. Forcing the newer Julia failed before any fit because its
package requirements were inconsistent.

The research plan explicitly says to deliver the complete EBM analysis when the
Julia toolchain fights back, so no Julia channel was changed in this step. A
follow-up should install a supported Julia version and first target the 13
bottleneck units that the EBM already reproduces at 80% or better. No symbolic
formula or symbolic stability claim is made here.

## Bottom line

Three independently fitted top networks share a readable, cyclic-invariant
approximation with about 86% fidelity. The most stable vocabulary contains the
two drives, $f_{\rm stab}$, and a windowed bad-curvature/compression quantity
related to $f_Q$. That is a strong model-mechanistic result: it describes common
behavior learned by the networks. It is not yet a physical causal law, and the
weak fit to many individual bottleneck units shows exactly where the simple
description remains incomplete.
