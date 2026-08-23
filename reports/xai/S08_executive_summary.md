# S08 executive summary — Concepts inside the hidden layers

## What this step asked

Earlier steps found that individual network units often resemble familiar
geometry patterns, but resemblance alone does not show that a network uses a
pattern. S08 asked two separate questions at every convolutional layer:

1. **Is the concept encoded?** In plain language, can a simple readout recover
   the concept from the hidden activations on equilibria it did not fit?
2. **Is the concept used?** Does moving the hidden activation along that
   concept's direction reliably change the network's native
   $\max(\log Q,-2)$ prediction more than a random direction matched to the
   hidden units' ordinary activation scales?

Keeping these questions separate is the central safeguard of this step.

## What was tested

The top three ensemble members were evaluated on the frozen 1,000-equilibrium
varied-gradient panel. Ten concepts covered the paper's $f_Q$ and
$f_{\mathrm{stab}}$ features, compression, curvature, parallel scale,
cross-channel co-location, the known
$\log\langle|\nabla x|\rangle$ target term, local $Q(z)$ structure, and zonal
flows.

A sparse probe (a simple linear predictor encouraged to use only a few hidden
features) was trained and tested with entire equilibria kept together. TCAV
(Testing with Concept Activation Vectors) then measured how the network output
changes along hidden-layer directions associated with high versus low examples
of each concept. Those examples were matched on gradient drive, equilibrium
class, and a simple geometry scale. Finally, the same hidden-layer edits were
compared with ordinary random directions and a stronger control weighted by
each hidden unit's observed activation range. The derivative and intervention
use exactly the same concept direction. Projecting activations into its
orthogonal complement (removing the component along that direction) supplies a
second diagnostic.

The encoding analysis uses all 1,000 equilibria. The more expensive derivative
and intervention analysis uses a fixed random subset of 96: 25 stable/near-floor
and 71 unstable. The uncertainty calculation resamples those 96 equilibria, and
every artifact labels these counts explicitly.

## Main conclusions

The networks clearly encode most of the tested geometry vocabulary. Across all
150 member/layer/concept combinations, the median held-out probe score was
$R^2=0.747$. Permuted labels and completely random concepts both scored about
zero. Nine of the ten concepts met the encoding requirement at every layer in
every member.

The strongest evidence for both encoding and use belongs to
$f_{\mathrm{stab}}$, which passed the complete gate in all 15 member/layer
combinations. Compression, cross-channel co-location, and $\log f_Q$ passed in
12; geodesic curvature passed in 10. This supports the broad physical feature
families emphasized by the paper, while showing that no single concept
describes every layer or member.

The way concepts are used changes with depth. Bad-curvature,
geodesic-curvature, and cross-channel co-location directions tend to reduce the
prediction in the first layer but increase it in later layers. Parallel scale
has a negative early effect and almost no late effect. By contrast,
$f_{\mathrm{stab}}$ has a positive effect throughout. These sign changes are
important: averaging layers or taking absolute values would produce a much
simpler—and wrong—story.

Stable/near-floor rows are not pooled with unstable rows. Most geometric
concepts remain decodable in both regimes, but local $Q(z)$ concentration does
not: its median held-out $R^2$ is -0.087 near the prediction floor and 0.362 on
unstable rows. That agrees with the expectation that floor rows carry little
information about flux magnitude.

## The most useful negative result

Zonal-flow magnitude demonstrates why all the controls are needed. It is not
meaningfully encoded (median $R^2=0.031$), and its high/low examples retain too
much $a/L_T$ imbalance. Nevertheless, every raw zonal-direction intervention
beats the median random direction. If we had reported interventions alone, this
would look like strong evidence that the network uses zonal flow—even though
zonal flow was not a network input and cannot be read reliably from its hidden
state. The complete gate correctly permits **zero** zonal-use claims.

Overall, 120 of 150 raw concept interventions beat ordinary isotropic random
controls, and 119 beat the stronger activation-scale-matched controls. Only 83
pass every condition: held-out encoding, parent- and subset-level balanced
examples, stable sign, uncertainty excluding zero after controlling the false
discovery rate (limiting the expected share of chance findings among declared
results), and intervention superiority. The gap is not lost signal; it is the
value of the safeguards.

The intervention comparison is now also reported separately by regime. It
beats the scale-matched control in 116 of 150 stable/near-floor cells and 117 of
150 unstable cells, but 19 pooled passes fail on the stable/near-floor rows.
Accordingly, the 83-cell headline is an average-cohort statement, not evidence
that every permitted direction matters near the clipped output floor.

Removing each concept direction—the orthogonal-complement projection
diagnostic—changes the output more than the small directional edit in all 150
cells (median RMS 0.609 versus 0.140). Zonal flow has the second-largest median
removal effect, 0.896, despite failing the encoding gate. This reinforces the
negative-control lesson: a disruptive artificial hidden-state edit is not, by
itself, evidence that the network learned a physical concept.

## What these results do and do not mean

The results show how the trained networks organize and use information. They do
not show that changing one of these concepts would cause the same change in a
real plasma. The hidden-layer interventions are deliberately off the data
manifold: they create internal states that may not correspond to any valid
stellarator equilibrium. Physical causality still requires the observational
checks and equilibrium-consistent GX interventions planned later.

Network-dissection overlap tables were deferred to protect the complete
top-three encoding/use analysis within the step's budget. S05 already supplies
the nearest bottleneck-level mask-overlap result.
