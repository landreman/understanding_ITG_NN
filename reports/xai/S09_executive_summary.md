# Claude summary

## What step 09 was for

By this point the project had built a list of candidate "concepts" — human-interpretable geometric quantities like bad curvature, geodesic curvature, flux compression, cross-channel co-location, a parallel length scale, plus the paper's own features ($f_Q$, $f_{\rm stab}$, $\log\langle|\nabla x|\rangle$). Earlier steps had shown the networks *encode* some of these (you can read them out of the network's internals) and *use* some of them (perturbing them changes the output). Step 09 asks the completeness question: **taken together, how much of what the network computes do these concepts actually account for?** And secondarily: **does the network's sensitivity to geometry change with the gradient drives $a/L_T$ and $a/L_n$?** — i.e., does geometry matter differently near the stability boundary than well above it?

The headline target was: how much better can you predict the network's output from the concept list than from the paper's baseline trio $\{a/L_T, a/L_n, f_Q\}$? That increment is the candidate answer to "what did the network learn that we didn't already know?"

## How it was done, in plain terms

**The fidelity measurement.** For each of the three canonical ensemble members, fit a deliberately simple stand-in model — a "decoder", here ridge regression (linear least-squares fitting with a penalty that discourages large coefficients, to prevent overfitting) on the concept scores plus their squares and their products with the two drives — and ask how well that simple model reproduces the *network's own prediction* (the native output $\max(\log Q, -2)$, not $Q$ itself). Agreement is scored with $R^2$, the fraction of variance in the network's output that the stand-in captures: 1 is perfect, 0 is no better than predicting the mean.

**Guarding against self-flattery.** The score is always computed on held-out data: the 1,000-row panel (one flux tube from each of 1,000 equilibria) is split into five folds by `equilibrium_files`, the decoder is fit on four folds and scored on the fifth, cycling through — so the score reflects generalization to equilibria the fit never saw, not memorization. Uncertainty on every gain comes from a bootstrap: resampling the 1,000 equilibria with replacement 5,000 times and recomputing the gain each time, giving a 95% interval. If that interval excludes zero, the gain is resolved; if it straddles zero, you can't claim it.

**Nesting.** Concept families were added one at a time — paper baseline → paper geometry features → spatial geometry concepts → everything including local $Q(z)$ concentration and zonal-flow magnitude — so each family's incremental contribution is visible.

**Interactions.** Within low/middle/high bins of each drive, the observed slope of output versus each concept was estimated, separately for stable/near-floor and unstable rows, and mixed derivatives of the fitted decoder (how a concept's effect changes as a drive changes) were computed as a labeled substitute for the plan's full network-Hessian calculation.

## What was found

**The vocabulary is nearly complete.** The full concept set predicts each member's output with held-out $R^2 \approx 0.91$ (0.9080, 0.9087, 0.9174). The paper baseline alone already gets 0.823, so the concepts add about **0.09 of $R^2$**, and that gain is statistically solid in all three members.

**But the honest decomposition deflates the headline.** Most of that 0.09 comes from the last family — local heat-flux concentration along the field line and zonal magnitude — which are *GX output diagnostics*, not geometry inputs. They predict the network well because they correlate with the answer, not because the network computes from them (step 08 had already rejected causal zonal-flow use). The **geometry-only** increment over the paper baseline is small: **0.012, 0.022, 0.019** across the three members, and for one member the 95% interval touches zero. So the genuinely new "geometry the network learned beyond the paper's features" is real but modest — roughly a 1–2% improvement in explained variance — and only two of three members resolve it.

**A curious redundancy.** Adding $f_{\rm stab}$ and $\log\langle|\nabla x|\rangle$ right after the baseline changes essentially nothing — apparently redundant with $f_Q$ in this simple-decoder setting, even though S08 found the networks' hidden layers do use them.

**Drive interactions replicate in sign but not in magnitude.** On unstable rows, all 48 concept-by-drive-by-bin sign patterns agree across all three members — strong qualitative reproducibility. The most interesting single pattern: the slope of output versus **bad curvature flips sign with drive**, from about −0.86 at low $a/L_T$ to +0.36 at high $a/L_T$, in every member. But the report is careful: each endpoint's 95% interval individually straddles zero, and no interval was computed for the difference, so this reversal is **a reproducible hint, not an established result**. Geodesic curvature's slope grows strongly with drive (0.22 → 3.55); co-location's shrinks (1.27 → 0.28). Near the floor (stable rows), everything is murkier — only 36/48 signs agree and almost no slope is individually resolved.

**A methodological trap avoided.** The simple decoder actually fit the small concept set *better* ($R^2$ 0.909) than it fit the network's full internal bottleneck representation (0.870). Read naively that would say the concepts carry *more* information than the network's own internals, which is impossible — the network's trained head reproduces its output from the bottleneck exactly. The resolution is that this is a limitation of the simple decoder, not an information ordering, and the report deliberately uses the exact head (fidelity 1) as the ceiling rather than reporting a misleading ratio above one.

## Conclusions

1. The concept vocabulary is a near-complete description of what these networks compute — ~91% of output variance from interpretable quantities, versus a ceiling of 100%.
2. The paper's existing features already do most of the work. The new geometric concepts add a real but small increment (~0.01–0.02), resolved in two of three members. There is no large hidden geometric feature the concept program missed *or* discovered.
3. Geometry's role plausibly shifts with drive — bad curvature's apparent sign flip and geodesic curvature's growing importance at high drive — with qualitative agreement across members, but the quantitative intervals are too wide to declare any single reversal established.
4. Near-floor (stable) behavior is separately reported and less reliable throughout; the meaningful fidelity conclusions rest on the unstable rows, where MSE improves from 0.69 to 0.37 and $R^2$ from 0.77 to 0.88.

One deferral to know about: the plan's task 3 asked for true mixed second
derivatives of the *network* with respect to its raw geometry inputs and drives;
what was delivered are labeled substitutes (decoder mixed derivatives and binned
observed slopes). Sharpening the bad-curvature reversal — an interval on the
high-minus-low difference itself — is the obvious follow-up the report leaves on
the table.

------


# Codex summary — How complete is the concept vocabulary?

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
was fitted. The frozen panel already has one row per equilibrium, so row and
equilibrium grouping coincide here; a repeated-equilibrium synthetic test
checks that the code remains safe on a less restricted cohort.

The paper's starting vocabulary—temperature gradient, density gradient, and
$f_Q$—already explains a median **82.3%** of the variation in the member
predictions. The full candidate vocabulary reaches held-out $R^2=0.909$. The
exact trained bottleneck head is 1 by construction, so this is a fidelity score
rather than a newly measured “percent of the bottleneck.” Its gain is
about **nine percentage points of $R^2$**, and the uncertainty interval excludes
zero in every member.

Completeness is bounded against the network's invariant bottleneck: the small
set of hidden numbers containing all geometric information used by the network.
The trained head reproduces its own output exactly, so the ceiling is 100% by
definition.
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

The fitted relationships vary with drive, although the clearest apparent
example is not statistically resolved. At low $a/L_T$, increasing the observed
bad-curvature score has median slope **-0.858**; at high $a/L_T$, the point
estimate is **+0.356**. All three members show that direction, but the grouped
95% interval for each endpoint crosses zero in every member. Thus the
step did not compute an interval for the high-minus-low difference, so the
bad-curvature reversal is a hypothesis, not an established result. Geodesic
curvature strengthens sharply with $a/L_T$, while cross-channel co-location
remains positive but weakens.

These patterns reproduce on unstable rows: every one of the 48
concept-by-drive-bin point-estimate signs agree across all three members. Near
the clipped output floor, only 36 of 48 agree, and only 19 of 144 individual
stable-row slopes have 95% intervals excluding zero. That weaker agreement is
itself useful: the network's stiffness behavior well above threshold is much
more coherent than its fine-grained geometry response where the output is
compressed against the floor. The three member outputs are themselves
extremely correlated
($r=0.994$ pairwise), so sign agreement is descriptive evidence and may partly
reflect shared training bias.

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

The drive-dependent point estimates consequently provide focused hypotheses:
shared members appear to reverse or strengthen particular geometry
relationships as turbulence drive changes, but the headline bad-curvature
reversal is not resolved at 95%. Those relationships are observed comparisons,
not physical interventions. They identify focused hypotheses for later compact
formulas and equilibrium-consistent GX tests; they do not yet establish plasma
causality.

The nested completeness result is complete. True mixed derivatives of the
network with respect to geometry inputs and drive, plus paired grouped finite
differences, are deferred; the present interaction diagnostics are fitted-
decoder slopes and mixed terms, not network-input Hessian entries.
