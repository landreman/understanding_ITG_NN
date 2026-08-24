# Claude summary

## What Step 8 was for

The earlier steps established *what* the trained networks predict (the clipped log heat flux, max(log Q, −2)) and built a vocabulary of ten candidate physics "concepts" — quantities like compression, bad curvature, geodesic-curvature magnitude, cross-channel co-location, the parallel length scale, f_Q, f_stab, log⟨|∇x|⟩, local Q(z) concentration, and zonal-flow magnitude. Step 8 asked: **are these concepts actually represented inside the network's intermediate computations, and — a separate question — does the network actually *use* them to produce its prediction?**

Some background you need:

- A neural network transforms its input through a sequence of **layers**. Each layer produces a **hidden representation**: a long vector of numbers that is the network's internal working state at that depth. Here there are 5 convolutional layers, and the analysis was done for the top 3 members of the ensemble (the networks are an ensemble — several independently trained copies whose predictions get averaged), giving 3 members × 5 layers × 10 concepts = 150 "cells" to test.
- **"Encoded"** means: you can read the concept's value out of the hidden representation. This is tested with a **linear probe** — a simple straight-line fit (here a *sparse* one, forced to use only a few of the hidden numbers) that tries to predict the concept value from the hidden vector. Its quality is measured with **R²**, the fraction of the concept's variation the fit explains (1 = perfect, 0 = no better than guessing the average, negative = worse than that). Crucially, R² was measured on equilibria the probe never saw during fitting, so a good score means genuine generalization, not memorization.
- **"Used"** means: nudging the hidden representation along the concept's direction actually changes the network's output. The tool for this is **TCAV** (Testing with Concept Activation Vectors): from matched sets of high-concept and low-concept examples, you extract a direction in the hidden space that separates them (a **concept activation vector**), then measure the derivative of the output along that direction, and also directly shift the hidden state along it and rerun the rest of the network to see how much the prediction moves.

Encoded-but-not-used is entirely possible: information can sit in a representation without influencing the answer, the way a variable can be in scope without appearing in the final formula. Keeping those two verdicts as separate columns was an explicit acceptance criterion.

## Guarding against fooling yourself

A lot of the machinery exists to keep the answer honest, and it's worth understanding because it shapes the conclusions:

- **Null controls.** Probes were also fit to shuffled concept labels and to a made-up random concept. Both scored R² ≈ 0 (−0.002), so a real score like 0.75 means something.
- **Matched examples.** High-concept and low-concept example sets were matched on the temperature/density gradients and equilibrium class, so a concept direction isn't secretly just "high gradient direction." The report quantifies residual imbalance and gates on it.
- **Random-direction controls.** A hidden-state nudge along *any* direction moves the output some amount. So each concept intervention was compared against random directions of equal size — including a stronger control scaled to match how active each hidden unit actually is — and a concept only counts as "used" if it beats those.
- **A five-part claim gate.** A cell is declared "encoded and used" only if all of these hold: decodable on held-out equilibria, balanced matching, a sign-stable derivative across five resampled counterexample sets, a bootstrap confidence interval excluding zero (with **false-discovery-rate control** — a correction for running 150 tests at once, since 150 tests at 5% would produce several false hits by chance), and beating the scale-matched random controls.

One caveat the report is explicit about: shifting a hidden representation puts the network in a state no real equilibrium would produce ("off-manifold"). These interventions diagnose *the network's* wiring, not the plasma's causal physics.

## What was found

**Encoding is widespread.** Median held-out R² was **0.747** across the 150 cells, versus ~0 for both controls. Nine of the ten concepts are decodable at every member and layer.

**The clear exception is zonal-flow magnitude.** It is essentially not decodable (median R² = 0.031), its example matching failed the balance threshold, and it earned **0 of 15** use claims. Interestingly, its arbitrary directions still beat random interventions — which the report treats as a warning that the intervention ratio *alone* can manufacture a false "use" story, validating the strict gate.

**Use is narrower than encoding: 83 of 150 cells pass the full gate.** The ranking by concept (out of 15 member/layer cells each):

- **f_stab: 15/15** — the most robust finding; encoded and used everywhere, always pushing the output in the same (positive) direction.
- Compression, cross-channel co-location, and log f_Q: 12/15 each.
- Geodesic-curvature magnitude: 10/15; bad curvature and local Q(z) concentration: 7/15; parallel scale: 6/15; log⟨|∇x|⟩: 2/15; zonal magnitude: 0/15.

**The direction of a concept's effect flips with depth for several concepts.** Bad curvature's output derivative goes from **−0.46 at layer 1 to +0.14 at layer 5**; geodesic curvature from −0.55 to +0.15; co-location from −0.60 to +0.30. Parallel scale is strongly negative early and near zero late. Only f_stab is consistently positive at every depth. This matters methodologically: any summary that averaged over layers or dropped the sign would have hidden these opposing mechanisms — which is exactly why the project's rules insist on keeping signed, per-member, per-layer results.

**Parallel scale illustrates the encoded/used split cleanly:** it's decodable (R² = 0.544) but passes the use gate in only 6/15 cells. The information is there; the network mostly doesn't route it to the answer.

**Regime caveat.** The "used" verdicts are for the pooled cohort. On the stable/near-floor rows alone (where the model predicts the −2 floor), 14 of the 83 permitted claims fail the intervention comparison — so "used" should not be read as "used near the output floor." Local Q(z) concentration is also barely encoded on stable rows (R² = −0.087) versus 0.362 on unstable rows.

## Conclusions

1. The convolutional stack genuinely carries most of the candidate geometry concepts — this isn't imposed by the analysis, since permuted and random concepts read out at zero.
2. **f_stab is the network's most consistently used concept**, at every layer of every top member, always in the same direction. If one concept "explains" these networks, it's that one.
3. Curvature-type concepts (bad curvature, geodesic curvature, co-location) are used, but their role **reverses sign between early and late layers**, so the network's treatment of curvature is not a single monotone mechanism.
4. Encoding ≠ use (parallel scale), and passing an intervention test ≠ meaningful (zonal flow) — both cautionary results were kept and reported with equal prominence.
5. Zonal-flow magnitude is effectively absent from the hidden computation, a genuine negative result.

One planned item (a "network dissection" analysis mapping individual hidden units to concept masks) was deferred for budget reasons; the report says Step 5 already covers similar ground at the network's bottleneck. The step also survived two rounds of adversarial automated review, which caught real problems (mismatched directions between the derivative and intervention tests, mislabeled sample counts, pooled regimes) that were fixed by discarding and recomputing the affected production statistics — the [Failed checks section](reports/xai/S08_concepts.md:223) documents that history rather than hiding it.

-------


# Codex summary — Concepts inside the hidden layers

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
More directly, **14 of the 83 permitted pooled claims** fail the stable-row
intervention comparison, and **3 of 83** fail the unstable-row comparison.

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
