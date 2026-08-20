# Step 02 Executive Summary

## The setup, in plain terms

Each of the 100 trained networks predicts turbulent ion heat flux from three things: a **flux tube's magnetic geometry** (7 physical quantities sampled at 96 points along a field line — so 672 numbers) and **two scalar drives**, the temperature gradient $a/L_T$ and the density gradient $a/L_n$.

The crucial architectural fact (structural fact 1 in [PLAN.md](PLAN.md)) is that a network doesn't blend geometry and drives freely. It first crushes all 672 geometry numbers down to a **bottleneck** of just 7–32 numbers, and only then feeds those, plus the two drives, into a small final network called the **head**:

$$f(\text{geometry}, g_T, g_n) = \underbrace{\text{MLP}}_{\text{the head}}\big(\underbrace{\bar u(\text{geometry})}_{\text{7–32 numbers}},\ g_T,\ g_n\big)$$

That bottleneck is a hard information limit: **everything the network knows about the magnetic geometry lives in those few numbers.** For the best-performing network it is only 10 numbers, of which one is entirely dead (always outputs zero).

That's an unusual gift. Interpreting a big neural network is normally hopeless because you're chasing millions of parameters. Here you can open a box with 10 numbers in it and enumerate essentially everything.

## What S04 was for

S04 is the step that exploits that gift. Its question is threefold:

1. **Which of those few numbers actually matter?** ("units" is the jargon for the individual numbers; `u001`, `u008`, etc. are their stable IDs.)
2. **Is anything recognizable stored in them** — specifically, can you read back out the hand-built geometric features that the original paper used, like $f_Q$?
3. **How does the head combine them** — is it doing something intricate, or something simple?

It's the pivot point of the whole program: earlier steps set up the machinery and the symmetry handling; S05 onward will ask what the *important* units physically measure along the field line. S04's job was to produce a short, trustworthy list of which units are worth that effort.

## The jargon you need

**Shapley values.** Borrowed from game theory. Imagine the 10 bottleneck numbers plus the 2 drives as 12 players cooperating to produce one prediction. To assign credit fairly, you consider every possible subset ("coalition") of players, and ask how much the prediction changes when player $j$ joins. Average that over all subsets, and you get player $j$'s share of the credit. With 12 players there are $2^{12}=4096$ coalitions — small enough to enumerate *exactly*, which S04 did for the top network. Wider networks (up to 33 players) needed random sampling instead, with reported error bars.

**Ablation.** Cruder and more direct: break one unit — set it to zero, or replace it with its average value across the dataset, or swap in another sample's value — and see how much the output moves. If the answer is "not at all," the unit isn't being used.

Shapley and ablation are conceptually very different, so if they agree, you can believe the ranking. **They agreed: rank correlation 0.89** (a Spearman correlation — 1.0 would be identical orderings). That's the credibility check that makes the rest of the step worth reading.

**Decoding / "probing".** Take the 10 bottleneck numbers and try to fit a simple formula from them to some known physics quantity — say the paper's $f_Q$ feature. If the fit is good, the information about $f_Q$ is *present* in the bottleneck. $R^2$ is the standard score: 1.0 = perfect prediction, 0 = no better than guessing the average, negative = worse than guessing the average.

**Encoded vs. used.** This distinction is the intellectual heart of the step, and it's a mistake people make constantly in interpretability work. Just because you can *decode* something from a hidden layer doesn't mean the network *uses* it. The information can be sitting there as an incidental byproduct while the head ignores it. So S04 measured both separately: decodability (encoded) and, independently, what happens to the output when you surgically remove that direction from the bottleneck (used).

**Native units.** All results are in $\max(\log Q, -2)$ — the network's actual output, log of heat flux, floored at $-2$. The floor exists because a third of the dataset is in stable conditions where flux is essentially zero, and $\log 0$ is undefined. Those "floored" rows behave statistically very differently and are reported separately throughout (240 of the 1,000 panel rows).

## The conclusions

**1. The head is drive-dominated.** Over the interpretation panel, the bottleneck geometry accounts for about **19.5%** of the variation in the prediction; the two gradient drives account for the rest. Read this carefully — it does *not* say geometry is physically unimportant. It says that across this particular set of cases, where both geometry and drive vary, the drives swing the answer more. On the unstable (non-floored) rows, geometry's share rises to ~22%.

**2. A couple of units do most of the geometry work.** In the top network, one unit (`u001`) dominates — breaking it moves the output by 0.634 RMS in native log units — a second (`u008`) is a distant second at 0.273, one is completely dead, and the rest form a shoulder. Same broad pattern in the other networks, though the specific unit indices differ per network since each was trained independently.

**3. The paper's own features are in there.** Decoding from the bottleneck alone, across all 100 networks:

| Target | median $R^2$ |
|---|---|
| $\log f_Q$ (the paper's bad-curvature/compression feature) | 0.89 |
| $\log\langle\|\nabla x\|\rangle$ (exactly known geometric factor) | 0.85 |
| $f_{\rm stab}$ (stability feature) | 0.80 |
| `nfp` (field periods) | 0.45 |
| magnetic shear $\hat s$ | 0.37 |
| aspect ratio | 0.28 |

The last three are the **controls** — quantities that shouldn't be central. And a "label permutation" control (scramble the answers, refit; a good method must then score ~0) came out at $R^2 = -0.05$, confirming the pipeline isn't manufacturing structure.

The $\log\langle|\nabla x|\rangle$ row is a nice sanity check: that quantity is *provably* baked into the prediction target by construction, so a method that failed to find it would be broken. It found it.

**4. But only some of that is actually used — the step's cleanest negative result.** Removing each decoded direction from the bottleneck and measuring the output change, then normalizing for the fact that the different edits have different sizes:

- $f_Q$: **0.479**, $f_{\rm stab}$: **0.396**, $\log\langle|\nabla x|\rangle$: **0.269** — all clearly above a random direction (**0.162**)
- `nfp` (0.147) and shear (0.122) — **below** the random-direction control; aspect ratio (0.171) sits about on it

So `nfp` and shear are demonstrably *encoded* — you can read them out at $R^2 \approx 0.4$–$0.5$ — but there's no evidence the network's answer depends on them. Had S04 stopped at decodability, it would have overclaimed their importance. That's a generalizable lesson about probing studies.

**5. The head is much simpler than its architecture.** It contains two ReLU layers, so it *could* be doing something quite nonlinear. It mostly isn't: a straight linear formula in (bottleneck, $g_T$, $g_n$) reproduces ~80% of each network's output variance, and adding a small nonlinear model buys only **0.012** more $R^2$. Similarly, interactions between pairs of units are usually tiny (median 0.011 native units), though a small tail reaches 0.28. Fair summary: **mostly additive, not exactly linear.**

## What it doesn't establish

Three limits worth internalizing, all flagged in the report:

- **These are network diagnostics, not plasma experiments.** Zeroing a hidden unit is not a stellarator you can build. It tells you what the head uses; it says nothing directly about causality in the physics.
- **The decoded directions are correlated with each other.** $f_Q$, $\log\langle|\nabla x|\rangle$, and $f_{\rm stab}$ are not independent, so "removing the $f_Q$ direction hurt" is evidence the head uses *that region of the bottleneck*, not that it uniquely computes $f_Q$.
- **Stable/near-floor rows resist variance-based statistics.** When the output is clipped flat, its variance is nearly zero, and anything divided by it goes haywire — the report shows head-fidelity $R^2$ of $-13$ on that stratum. Those numbers were kept as an honest negative rather than cleaned away, and no mechanism claim rests on them.

Two other things worth noting for confidence: this wasn't a "the information is present, therefore the network knows physics" claim, and the failures were kept in the record — an early nonlinear decoder was under-regularized, and two networks with near-dead units blew up to $R^2 \approx -200$ before a guard was added.

## Where this leaves the program

S04's deliverable to S05 is a **short, stable target list**: the handful of units that both Shapley and ablation agree are load-bearing. S05 now asks the question S04 deliberately couldn't — for those specific units, what local density along the field line are they averaging, and does it correspond to a nameable geometric motif (bad curvature, compression, local shear, a magnetic well)?

One caveat the executive summary itself raises: showing that the bottleneck
encodes the paper's feature grammar is confirmation, not discovery, since
$f_Q$'s role was already known. The "what did the network learn that we didn't
already know" question is explicitly deferred to S08.

#####################################################

## Old version of the summary

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
