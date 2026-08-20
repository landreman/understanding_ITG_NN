# S04 executive summary — what is inside the bottleneck

Each network compresses the entire magnetic geometry into only 7–32 numbers
before combining it with the temperature and density gradients. S04 opened that
small box completely: it saved those numbers for all 100 networks, measured how
the output changes when each one or each pair is edited, and asked which known
geometric quantities can be read back out.

The headline is that the box matters, but the two drive inputs matter more. In
the top 10 networks, geometry accounts for about **20% of the variation in the
prediction**; the temperature and density gradients account for the rest. This
is not saying geometry is physically unimportant. It says that, over this panel
where both geometry and drive vary, the trained network's final combination is
drive-dominated.

The unit ranking is unusually trustworthy for an explanation result. Two very
different tests agree: Shapley values ask how much credit each unit receives
across all possible coalitions, while ablation replaces one unit with its normal
panel value and measures the damage. Their median rank correlation is **0.89**.
For the top network, one unit is clearly dominant (0.634 RMS output change), one
is a distant second (0.273), and one is completely dead. The other networks show
the same broad pattern of a few strong units plus a shoulder, although the unit
numbers are network-specific.

The bottlenecks encode much of the paper's feature grammar. This is not itself a
discovery—especially for $f_Q$, whose role was already known—and S04 does not
yet ask what the network adds beyond that feature family. From the bottleneck
alone, a held-out decoder recovers:

- the paper's $f_Q$ feature with median $R^2=0.89$;
- the exactly known $\log\langle|\nabla x|\rangle$ factor with $R^2=0.85$; and
- the stability feature $f_{\rm stab}$ with $R^2=0.80$.

That is substantially better than reading out `nfp`, magnetic shear, or aspect
ratio. More importantly, S04 did not stop at “the information is present.” It
removed the bottleneck direction associated with each feature and asked whether
the output moved. Those edits have unequal sizes, so the raw changes are not
directly comparable. Once divided by the removed projection's RMS size, $f_Q$,
$f_{\rm stab}$, and the known gradient factor remain above random directions;
`nfp` and shear are below the control, while aspect ratio is approximately on
it. So the first group is both **encoded and used**; the second is at least
partly encoded but not demonstrably used. This is the cleanest negative result
in the step: being able to decode something from a hidden layer is not evidence
that the network bases its answer on it.

The dense head is also simpler than its architecture suggests. A linear model
of the bottleneck and drives reproduces about 80% of each top member's output
variation. Adding a small nonlinear decoder raises that by only about **0.012**
on average. Pair interactions are usually small too, although a sparse tail is
real, so “mostly additive” is fair and “linear” is not.

Stable or near-floor cases remain awkward. Geometry effects are smaller there,
and any statistic divided by the tiny output variance can become negative or
enormous. Those numbers were kept rather than cleaned up, but the interpretation
rests on native output changes and the 760 unstable rows, where the denominators
are meaningful.

What S04 does **not** tell us is what the important units see along the field
line. The hidden edits are network diagnostics, not physically realizable
stellarator changes. S05 now has a short, stable target list: start with the
high-Shapley/high-ablation units, inspect their full spatial densities, and ask
which broad geometric motifs produce them.
