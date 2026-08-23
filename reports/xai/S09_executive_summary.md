# S09 executive summary — How complete is the concept vocabulary?

## What this step asked

Earlier steps showed that the networks encode and sometimes use familiar
geometry concepts. S09 asked the harder accounting question: how much of each
network's prediction can those concepts explain together, and does their effect
change with the temperature and density gradients that drive turbulence?

The target throughout is the network's own native prediction,
$\max(\log Q,-2)$, not the measured heat flux itself. This distinction matters:
the step explains what the trained network computes, not what causes transport
in a plasma.

## How completeness was measured

A simple decoder (a small fitted formula) tried to reproduce each of the top
three networks from progressively larger concept sets. Entire equilibria were
kept together, and every score came from equilibria excluded while that decoder
was fitted. This prevents nearly related rows from making the score look better
than it is.

The paper's starting vocabulary—temperature gradient, density gradient, and
$f_Q$—already explains a median **82.3%** of the variation in the member
predictions. The full candidate vocabulary explains **90.9%**. Its gain is
about **nine percentage points of $R^2$**, and the uncertainty interval excludes
zero in every member.

Completeness is bounded against the network's invariant bottleneck: the small
set of hidden numbers containing all geometric information used by the network.
The trained head reproduces its own output exactly, so the ceiling is 100%.
A separate simple decoder of that wider bottleneck reaches 87.0%. The fact that
the lower-dimensional concept decoder scores slightly better than that simple
bottleneck decoder reflects ease of fitting, not extra information; both
numbers are kept so that distinction is visible.

## The main qualification

The nine-point gain is not nine points of newly discovered geometry. The final
concept family includes the spatial concentration of GX $Q(z)$ and zonal-flow
magnitude. These are observed simulation diagnostics, not inputs supplied to
the network. They can summarize geometry correlated with the prediction, but
they cannot show that the network directly reads zonal flow. S08's stronger
use test rejected every zonal claim.

Restricting the comparison to geometry concepts gives a smaller gain over the
paper baseline: **1.2%, 2.2%, and 1.9% of $R^2$** in the three members. The gain
is statistically resolved in two members but not the third. This is the honest
answer to “what geometric information did the network learn beyond the paper's
baseline?”: something reproducible, but modest and member-dependent.

Adding $f_{\rm stab}$ and the known
$\log\langle|\nabla x|\rangle$ term immediately after $f_Q$ gives essentially
no gain. That does not contradict their hidden-layer use; it says they add
little unique predictive information once $f_Q$ and the two drives are already
in this particular decoder.

## How geometry changes with drive

The networks do not use one fixed geometry rule at every drive. The clearest
example is bad curvature. At low $a/L_T$, increasing the observed bad-curvature
score is associated with a lower prediction (median slope **-0.858**); at high
$a/L_T$, it is associated with a higher prediction (**+0.356**). All three
members show that reversal. Geodesic curvature strengthens sharply with
$a/L_T$, while cross-channel co-location remains positive but weakens.

These patterns reproduce impressively on unstable rows: every one of the 48
concept-by-drive-bin signs agrees across all three members. Near the clipped
output floor, only 36 of 48 agree. That weaker agreement is itself useful: the
network's stiffness behavior well above threshold is much more coherent than
its fine-grained geometry response where the output is compressed against the
floor.

The stable-row fidelity is therefore reported with mean squared error rather
than relying on $R^2$, whose denominator is nearly zero there. The candidate
decoder cuts median stable-row error from **0.848 to 0.385** and unstable-row
error from **0.691 to 0.371**.

## What this means

The tested vocabulary gives a compact, high-fidelity description of the three
networks, but most of the easy explanation was already present in the paper's
drive-plus-$f_Q$ baseline. The additional geometric story is real but small,
and the strongest remaining predictive gain comes from simulation diagnostics
that must not be mistaken for causal inputs.

The most valuable new result is consequently the drive dependence: shared
members reverse or strengthen particular geometry relationships as turbulence
drive changes. Those relationships are observed comparisons, not physical
interventions. They identify focused hypotheses for later compact formulas and
equilibrium-consistent GX tests; they do not yet establish plasma causality.

Nothing was deferred.
