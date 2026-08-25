# Claude summary

## What S12 was trying to do

The object being studied is the trained neural-network ensemble that predicts GX ITG heat flux. It takes seven geometry channels along a flux tube plus two gradient drives ($a/L_T$, $a/L_n$) and outputs `max(log Q, -2)` — the "native" output, a log heat flux with a floor at −2 for stable/quiet cases.

Steps S01–S11 asked *where* the network looks and *what it is sensitive to*. S12 asks the complementary question:

> Can the network's behavior be **replaced** by a short list of named physical quantities and simple curves — i.e. written down as something a physicist can read?

This is called **distillation**: fit a simple, inspectable "student" model to reproduce the outputs of the complicated "teacher" (the network). If the student matches the teacher closely, then whatever the teacher is doing is, at least at the level of its output, expressible in the student's vocabulary. If it doesn't match, the network is doing something outside that vocabulary.

Two vocabularies were planned:

1. **EBM** — an *Explainable Boosting Machine*. Think of it as a generalized additive model: the prediction is a sum of one separate curve per input feature, `prediction = intercept + h₁(x₁) + h₂(x₂) + … `, plus a small, explicitly pre-registered set of two-variable surfaces $h_{ij}(x_i, x_j)$ ("interactions" — terms whose effect can't be split into two independent curves, e.g. where the effect of curvature depends on how strong the density gradient is). Every term is a curve or a surface you can plot and read. It's fitted by gradient boosting (many tiny corrections learned in sequence), but the result is still just a sum of plottable pieces.
2. **PySR** — symbolic regression, which searches over actual algebraic expressions to find compact formulas.

The features fed to the EBM were deliberately built to be **exactly cyclic-invariant**: the flux-tube geometry is a periodic array along the field line, and S02 established the network's canonical form is invariant to shifting that array. Each of the 17 features is some pointwise or derivative operation on the channels followed by a reduction (mean, variance, max over a circular window, Fourier magnitude) that is unchanged if you rotate the whole array. So the student is constrained to the same symmetry the teacher has — it can't cheat by keying on an arbitrary grid position.

The 17 features, in [feature_registry.csv](reports/xai/S12_artifacts/feature_registry.csv), are: the two drives; the paper's three geometry quantities ($\log f_Q$, $f_{\rm stab}$, $\log\langle|\nabla x|\rangle$); several bad-curvature/compression combinations; geodesic-curvature quantities; parallel roughness; the "typical parallel wavenumber" of $B$, $|\nabla x|$ and curvature; mean local shear; and two 25-point-window **peak** features from S05 (the largest value of a running 25-point circular average — a "worst local patch" measure rather than a field-line average).

The cohort is S01's frozen 1,000-row panel, one flux tube from each of 1,000 distinct equilibria; 240 rows stable/near the −2 floor, 760 unstable.

---

## How fidelity was measured (and why the split matters)

Fidelity is reported as **held-out $R^2$**: 1 − (mean squared error)/(variance of the target). $R^2 = 1$ is perfect, $0$ means "no better than always predicting the mean", negative means "worse than the mean".

Crucially it's *out-of-fold*: the rows were divided into five folds **by `equilibrium_files`**, so every row's prediction comes from an EBM fitted on data that contained no tube from that row's equilibrium. Splitting by tube instead would let near-duplicate tubes from the same equilibrium sit on both sides, which inflates the score by letting the model half-memorize. Uncertainties are grouped bootstraps: resample whole equilibria with replacement, 2,000 times, and take the middle 95%.

---

## What was found

### 1. The 17-feature table reproduces the networks well — and it isn't an artifact of averaging

Three separate top-ranked ensemble members were distilled individually:

| target | held-out $R^2$ | 95% interval |
|---|---|---|
| member `2864601_0.437` | 0.8603 | [0.8453, 0.8745] |
| member `2864601_0.371` | 0.8561 | [0.8400, 0.8704] |
| member `2864601_0.409` | 0.8636 | [0.8486, 0.8774] |
| mean of the three | 0.8632 | [0.8482, 0.8765] |
| **true GX** `max(log Q,−2)` | **0.8392** | [0.8210, 0.8561] |

The three members land within a 0.0075-wide band. That matters because an ensemble mean can be reproducible while each individual member is doing something idiosyncratic — here they're not; all three are individually compressible into the same vocabulary.

The last row is deliberately separated: fitting the **same features to the real GX answer** measures physical predictive accuracy of the named-feature model (0.8392), while the member rows measure fidelity to *what the network learned* (0.856–0.864). Keeping these apart is essential — a formula could match the network perfectly while both are wrong about the physics.

**But it is not a complete replacement.** Residual standard deviation is 0.744–0.768 in native log units, against roughly 0.28–0.31 for the networks themselves versus GX. So the compact formula leaves errors about 2.5× larger than the network's own. "Captures most of the variance" is the honest claim, not "explains the network".

### 2. The added fidelity is *interaction structure*, not a new geometry knob — the most interesting result

The nested comparison in [subset_fidelity.csv](reports/xai/S12_artifacts/subset_fidelity.csv) builds the model up one concept family at a time, always on the same folds:

| feature set | $R^2$ range across the five targets |
|---|---|
| the two drives alone | 0.584 – 0.612 |
| \+ the paper's $\log f_Q$ ("baseline trio") | 0.764 – 0.787 |
| \+ $f_{\rm stab}$ and $\log\langle\lvert\nabla x\rvert\rangle$ (all five paper variables) | 0.778 – 0.806 |
| all 17 main-effect curves | 0.780 – 0.807 |
| all 17 **plus the five registered interaction surfaces** | **0.839 – 0.864** |
| baseline trio plus only the $a/L_T \times \log f_Q$ surface | 0.791 – 0.812 |

Read the ladder: the two drives get you ~0.60. The paper's $\log f_Q$ adds a large ~0.17–0.18. Then *fourteen more geometry features add only ~0.02*. Adding the five pairwise surfaces adds 0.075–0.080 (paired gain over the trio, with a bootstrap interval well clear of zero) — **more than triple** what all the extra standalone geometry contributed.

So the network is not using some undiscovered geometry motif as an independent additive term. What it has beyond the paper's picture is **drive-dependent geometry**: how much a given geometry feature matters depends on how hard you're driving the turbulence. And it isn't one single surface — the $a/L_T \times \log f_Q$ surface alone recovers only about a third of the interaction gain; the rest is spread across the other four.

### 3. Which features are stable enough to name

30 bootstrap refits (resampling whole equilibria) record how often each feature lands in the top five by importance ([term_recurrence.csv](reports/xai/S12_artifacts/term_recurrence.csv)):

- **$a/L_T$ and $a/L_n$: 30/30, every target.** Unsurprising, but a sanity check that passed.
- **$f_{\rm stab}$: 0.83–0.90** across members — the most reliably-recurring geometry term.
- **The 25-point windowed peak of the $f_Q$ integrand: 0.83, 0.83, 0.87** (0.70 for the true target). This is an S05-derived feature — the *worst local patch* of bad-curvature-weighted compression rather than its field-line average — and it recurs about as strongly as $f_{\rm stab}$. That's evidence the network attends to localized structure, not only averages.
- **Geodesic-curvature/compression: 0.27–0.60.** Unstable; a feature family, not a nameable term.

There's a trap here the report flags explicitly, and it's worth understanding because it looks like a contradiction. **$\log f_Q$ — which the ladder above showed adds ~0.17 $R^2$, by far the largest geometry contribution — appears in the full model's top five in 0/30, 0/30 and 1/30 refits.** This is not evidence against $f_Q$. The 17 features are strongly correlated with one another, and importance in an additive model is not a unique division of credit: once correlated descendants of $f_Q$ (the co-location feature, the windowed $f_Q$ peak) and the registered $f_Q$ interactions are in the model, they absorb the same information and the credit gets redistributed among them. The nested ladder, which adds features in a fixed order and measures the *change* in held-out score, is the trustworthy attribution; top-five importance rankings within a correlated model are not.

### 4. Individual bottleneck units mostly resist compression

PLAN expected internal units to be the *easiest* target (the network's bottleneck is 64 units across the three members, and a single unit should be simpler than the whole output). The opposite happened:

- 5 of 64 units are **exactly dead** on the panel — constant output, no $R^2$ defined.
- The 59 live units have median held-out $R^2 = 0.5942$.
- Only **13 of 64** reach 0.8. Member medians differ a lot: 0.7821, 0.4528, 0.6037.

So the *output* of these networks is compressible into named physical features, but the *internal representation* largely is not — individual units are not each computing one nameable physical quantity. This is a genuine negative result and it corroborates S05, where most units failed to earn a supported one-name interpretation. It also carries a message about the architecture: the geometry information appears to be distributed across the bottleneck, not factored into physically labeled channels.

### 5. Stable vs. unstable rows behave completely differently

On the 760 unstable rows the fits are good: member $R^2$ 0.8115–0.8268, true-target 0.7965.

On the 240 stable/near-floor rows $R^2$ is wildly negative — −7.4 to −8.5 for members, −3049 for the true target. This is *not* a catastrophic failure; it's what $R^2$ does when the denominator collapses. Nearly all those rows sit at or near the clipped floor of −2, so the target variance is tiny, and dividing a small error by a near-zero variance produces a huge negative number. The corresponding **mean squared errors are ordinary**: 0.643–0.659 for members, 0.852 for the true target. The correct statement is: on stable rows only error-scale claims are meaningful, and the numbers were kept rather than quietly dropped.

### 6. The interaction surfaces themselves

Of the five registered surfaces, the $a/L_n \times$ bad-curvature/compression surface has the largest root-mean-square signed effect for all three members (0.440–0.497 native units). Reading the committed grid for member `2864601_0.437`: at low bad-curvature/compression the surface rises with $a/L_n$ (−0.54 → +0.99), while at high bad-curvature/compression it falls steeply (−0.02 → −2.45). In other words the sign of the density-gradient effect flips depending on the bad-curvature/compression level — which is exactly why it needs a surface and not two curves.

Heavy caveat, stated in the report and worth repeating: **these are descriptive fits over observed geometries, not interventions.** No feature was edited and nothing was held fixed while another was varied. Because the features are correlated, a surface describes how the network's output covaries with these coordinates across the panel; it does not license "increase $a/L_n$ and the flux will drop". The whole feature table carries the `observed-comparison` validity tag for this reason.

### 7. PySR did not run

PySR 1.5.10 requires Julia 1.10.3–1.11; the workstation has Julia 1.12.6, and forcing the older executable failed with an unsatisfiable `OpenSSL_jll ~3.0` dependency. PLAN explicitly designated the EBM half as the minimum deliverable if the Julia toolchain fought back, so **no symbolic expression, complexity/fidelity Pareto frontier, or expression recurrence is claimed at all** — this is recorded as deferred, not worked around. The follow-up is scoped: install a supported Julia channel, then run symbolic fits on the compact feature set, starting the per-unit symbolic work on the 13 units that reached $R^2 \ge 0.8$ and keeping the 46 weaker and 5 dead units as controls.

---

## The conclusions, in one place

1. **The networks' output is largely, but not fully, expressible in named physical language.** 17 exactly cyclic-invariant features reproduce three separate top members at $R^2 \approx 0.86$ across held-out equilibria — but with residuals ~2.5× the networks' own error against GX. Compressible, not solved.
2. **What the networks add beyond the published picture is drive-dependent geometry, not extra geometry.** Fourteen additional standalone features bought ~0.02 $R^2$; five pairwise drive×geometry surfaces bought ~0.077. That is a specific, testable hypothesis about what the networks learned that the paper's additive vocabulary doesn't express.
3. **Localized structure matters.** The windowed-peak $f_Q$ feature — worst local patch, not field-line average — is among the most reliably recurring geometry terms (0.83–0.87), alongside $f_{\rm stab}$ (0.83–0.90).
4. **The internal units are not individually nameable.** Median unit $R^2$ 0.59, only 13/64 above 0.8, 5 dead. The compressibility lives in the output, not in a physically factored internal representation.
5. **Model fidelity and physical accuracy were kept distinct throughout**: 0.856–0.864 against the networks, 0.8392 against real GX — everything in native `max(log Q, −2)`, never exponentiated.
6. **The known limits are real ones.** The panel is interpolation equilibria seen in training, so the held-out score tests whether *the distillation* generalizes across this panel — not whether the networks generalize to unseen equilibrium families. Importance can slide between correlated curvature/compression variants, so recurrence supports a feature *family*, not a unique algebraic identity. And the symbolic half of the step is missing entirely.

The one thing I'd flag as most worth your attention scientifically is #2
combined with #4: the networks appear to encode *how geometry's influence
changes with drive strength*, and they do so in a distributed way that no single
bottleneck unit expresses. That's the concrete handle S13's natural-experiment
work would need to test.

-----


# Codex summary — Can the networks be rewritten in physical language?

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

- 86.03% for member `2864601_0.437` (95% interval 84.53–87.45%);
- 85.61% for member `2864601_0.371` (84.00–87.04%); and
- 86.36% for member `2864601_0.409` (84.86–87.74%).

The three-member mean was 86.32%. Thus the result is not an artifact of averaging
different networks together: all three individual networks are reproduced at
nearly the same level.

The same feature table predicts the true clipped logarithmic GX heat flux at
83.92% (95% interval 82.10–85.61%). These intervals come from resampling whole
equilibria 2,000 times. Physical prediction is a separate question: the
approximately 86% numbers say how well the readable model imitates the neural
networks, while 83.92% says how well it predicts the simulation target. Keeping
those two scores separate prevents good imitation of a network from being
mistaken for new physical validation.

The nested comparison shows where the approximately 86% comes from. The two
drives alone explain 58–61% across the member, mean, and true-target fits. Adding
the paper's $\log f_Q$ raises this to 76–79%. The other 14 geometry main effects
add only about 2 percentage points beyond that baseline. Most of the remaining
gain comes from the five pre-registered drive–geometry interactions: the full
model adds 7.5–8.0 percentage points beyond the drives-plus-$\log f_Q$ baseline.
The readable result is therefore mainly a compact drive-dependent response, not
17 independent geometry effects.

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

That correlation creates one apparent paradox. $\log f_Q$ adds 17–18 percentage
points when it is introduced after the two drives, yet it almost never appears
among the full model's five most important main effects. Its correlated
descendants and interaction terms divide the credit once all are present. Low
importance recurrence in the full model therefore does not erase the clear
nested-model evidence that $\log f_Q$ is predictive.

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
approximation with about 86% fidelity. The nested result says the main compact
story is the two drives, $\log f_Q$, and a few drive-dependent geometry
interactions; recurrence also supports $f_{\rm stab}$ and a windowed
bad-curvature/compression quantity related to $f_Q$. That is a strong
model-mechanistic result: it describes common behavior learned by the networks.
It is not yet a physical causal law, and the weak fit to many individual
bottleneck units shows exactly where the simple description remains incomplete.
