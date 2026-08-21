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

## The two ideas that were selected

The benchmark chose one method from each of two different families.

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

These two methods answer different questions. Their agreement in later work
will be more credible than two minor variations of the same gradient formula.

## Why they were selected

On a synthetic circular example where the correct channel and four wrapped
positions were known exactly, both methods found the right channel and achieved
perfect position-recovery score (1.0). A random map scored 0.044.

The methods also passed “faithfulness” tests. The benchmark removes cells in the
order a method calls important and compares that curve with removing cells in a
random order; insertion performs the reverse experiment. On the canonical
128-row panel, low-pass Integrated Gradients beat its random control by 0.572 on
deletion and 0.416 on insertion. The mask beat random by 5.247 and 5.293. Both
directions also remained positive when the 33 stable/near-floor rows and 95
unstable rows were analyzed separately.

Finally, the explanations changed when all learned parameters were reset. Their
rank correlation with the randomized-network maps was 0.406 for Integrated
Gradients and 0.099 for the mask; a value near 1 would have meant the map barely
noticed that learning had been erased.

## The symmetry check

Rotating all seven geometry channels together is physically the same field line
with a different starting point. A good map for the canonical network should
rotate with the input and otherwise remain unchanged. The selected methods pass:
their relative rotation errors are about $10^{-4}$ and $3\times10^{-7}$.

The original network's errors are about 0.82 for both methods. That is not a new
bug—it is the pooling-phase dependence established in S02. It demonstrates why
the canonical invariant network must remain the primary explained function.

## The most important caveats

The first caveat is **baseline sensitivity**. Integrated Gradients needs a
starting geometry, and different plausible choices give meaningfully different
maps. The selected low-pass map has rank correlation only 0.432 with the map
from a robust constant reference. The 64-row pilot even selected a medoid
reference, while the fixed rule selected low-pass on 128 rows. S06b must carry
the other baselines as sensitivity analyses; the chosen map is not a unique,
baseline-free truth.

The second caveat is **physical validity**. Smoothing a geometry or replacing
individual cells does not generally produce another realizable magnetic
equilibrium. The mask edits travel as far as 3.8 robust input-scale units and
receive high data-support warnings. These maps explain what the trained network
does off its observed manifold. They do not show that changing a stellarator in
that way would change the plasma.

The third caveat is that only one network was benchmarked. S06a can select
methods, but it cannot say a feature is common across the ensemble. S06b must
run the selected pair on the top 10 members, retain signed member-level results,
and quantify member and equilibrium uncertainty.

## Negative results worth keeping

The simple cyclic occlusion method was not good enough. It recovered the correct
toy channel but its position score was only 0.613, below the registered 0.75
threshold, and its real deletion curve was worse than random. Integrated
Gradients from a matched observed geometry also failed deletion faithfulness.
An observed endpoint does not make the artificial path between two geometries
physically valid or numerically useful.

The implementation itself caught three subtle failures: Captum expanded the
input batch without expanding the two scalar drives; Expected Gradients used an
unseeded NumPy random draw; and the first faithfulness normalization reversed
the interpretation when a stable prediction lay below its baseline. Each was
fixed, tested, and the pilot and production artifacts were regenerated.

## What comes next

S06b should apply low-pass Integrated Gradients and the periodic mask to the
registered top 10 networks, with medoid and robust-reference Integrated
Gradients retained as baseline checks. It will then ask whether the signed maps
agree across networks, how that agreement changes between stable and unstable
conditions, and whether explanation stability has much relationship to the
networks' validation ranking. Only after that ensemble-level agreement exists
should S07 compare the maps with physical fields and GX heat-flux profiles.
