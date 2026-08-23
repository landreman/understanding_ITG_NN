# Claude summary

Steps 1–6 built up a picture of *where along the field line* the network's internals light up. Step 7 asked the obvious next question: **do those internal spatial patterns line up with the actual physics — specifically with GX's heat-flux profile $Q(z)$ and with zonal-flow amplitude?** If they did, you'd have a concrete, physically meaningful hypothesis about what the network learned. The answer was largely no, and the report says so plainly.

---

## What you need to know to read the results

**The objects being compared.** Everything lives on the 96-point grid along the field line (the coordinate $z$). Three different kinds of profile get compared:

1. **Activation density $\rho(z)$** (from S05) — for one internal unit of the network, how strongly it fires at each position $z$. This is just "the unit is active here," nothing more.
2. **Attribution** (from S06b) — a per-position number saying how much each location *contributed to changing the prediction*, relative to some reference input. This is the stronger notion: not "the unit is on here" but "the model used this location to arrive at its answer."
3. **GX $Q(z)$** — the actual simulated heat flux as a function of position. This is a physics diagnostic. **The network never saw it**, not as an input and not as a training target; it was trained only on the single flux-averaged number. So agreement with $Q(z)$ would be genuinely informative, not circular.

Activation and attribution are *not* interchangeable, and the gap between them is the crux of this step.

**Circular Spearman rank correlation.** "Spearman" means you compare *rank orderings* rather than raw values — you ask "are the positions that rank high in $\rho$ also the ones that rank high in $Q$?" This is robust to the fact that the two quantities have completely different units and scales. "Circular" means the field line is periodic, so the profile wraps around. Range is $-1$ to $+1$; $0$ means no ordering agreement.

**Lag.** Because the grid is circular, you can slide one profile relative to the other by $k$ positions and recompute the correlation, for all 96 possible shifts. The reported "lag +22" means the best agreement occurs when the density profile is shifted 22 grid points relative to $Q(z)$. This matters physically: a lag of 0 would say "the unit fires exactly where the heat flows"; a lag of +22 says "the unit fires at a systematically displaced location," which is a much weaker and murkier statement.

**Why picking the best lag is dangerous, and the permutation null.** If you scan 96 shifts and keep the biggest correlation, you will find *something* even in pure noise. So the step includes a **permutation null**: scramble which density goes with which $Q(z)$ (breaking any real association), redo the whole 96-lag search, record the maximum, and repeat 200 times. The 95th percentile of those maxima (`null_q95`) is the bar an honest result must clear. Numbers like 0.043 in the tables are that bar.

**Bootstrap intervals.** "Resample to estimate uncertainty": draw the equilibria with replacement 500 times and recompute. The 95% interval is the spread. Crucially, resampling is done by **equilibrium**, not by flux tube — flux tubes from the same equilibrium are near-duplicates, and resampling them individually would make the data look like it contains more independent evidence than it does, shrinking the intervals artificially.

**Two panels.** "Varied-gradient" = the drive $(a/L_T, a/L_n)$ differs across rows. "Fixed-gradient" = the same geometries re-simulated at a single fixed drive $(3, 0.9)$. The fixed panel isolates geometry effects because drive is held constant.

**The floor.** The model predicts $\max(\log Q, -2)$. About a quarter of the varied rows sit at that $-2$ floor — the model is saying "stable, essentially no transport." Below the floor, the output physically cannot resolve differences, so those rows are analyzed separately throughout.

---

## What was found

### 1. Activation densities do covary with $Q(z)$ — but with a displacement and the wrong sign

The two units S05 had tentatively named — `u001` (the "bad-curvature / flux-compression" candidate, i.e. the region where magnetic curvature drives the instability) and `u008` (the geodesic-curvature / radial-drift candidate) — both show correlations well above their permutation nulls:

| unit | correlation vs $Q(z)$ | 95% interval | lag |
|---|---:|---:|---:|
| `u001` | **−0.361** | [−0.388, −0.333] | +22 |
| `u008` | **−0.268** | [−0.294, −0.241] | +44 |

These are real associations, not lag-search artifacts. But read the signs. **Both are negative**: where these units fire strongly, the heat flux tends to be *low*. The report is scrupulous about this — the headline "1.63 times chance" overlap figure is only obtained *after flipping the sign of $Q(z)$*, so it measures overlap of high activation with **low** heat flux. The straightforward high-activation / high-heat-flux overlap is **0.542 times chance**, i.e. these units systematically *avoid* the high-transport regions.

And the lags are +22 and +44 out of 96 — a quarter and nearly half a field-line period away. This is not "the unit watches where the heat is."

### 2. The negative result that carries the step: attribution shows essentially nothing

This is what the report calls "the central contradiction." When you ask not "is the unit on here" but "did the model *use* this location to make its prediction," the signed correlation with $Q(z)$ collapses to:

**−0.021, −0.013, −0.012** across the three ensemble members.

For scale, those are roughly one twentieth of the density correlations, and they sit right at or below the permutation null. One member's value is below the null threshold outright, one sits essentially exactly on it, and the third ("resolved") is still only 0.021 in magnitude. Additionally the selected lags disagree entirely across members (−36, +47, +48), and one of those lags is unstable — it only reappears in 31% of bootstrap resamples, failing the pre-registered 50% rule.

So: the network's internals *contain* spatial patterns that covary with real physics structure, but the prediction is not being driven by the positions where GX actually transports heat.

**The tempting escape route, and why it was refused.** If you keep only the *positive* contributions and throw away the negative ones, the correlations jump to **+0.266, +0.280, +0.262** — nice, consistent, lag ≈ 0, roughly 7× their null, and overlapping positive $Q(z)$ at ~2.4× chance. That's the number one would want to headline. The report declines to, for two reasons, and both are correct:

- Discarding the negative contributions discards the evidence pointing the other way. The full signed picture is the mechanism; a one-sided view is a resemblance.
- Every attribution number here is tagged **"deliberately off-manifold."** Attribution works by interpolating from a reference input — here a smoothed, low-pass version of the geometry. That reference is *not* asserted to be a valid plasma equilibrium. So the attribution describes how the network extrapolates away from a synthetic starting point. It is a statement about the network, not about the plasma. This is exactly the distinction your project rules insist on, and it's applied even to the flattering number.

### 3. The zonal-flow observable: an effect that isn't candidate-specific

`zonal_phi2_amplitudes` is the natural observable for the geodesic-curvature/zonal-flow hypothesis, so if `u008` were really encoding that, you'd expect a distinctive association. It does show one on the fixed-drive panel: **−0.513** [−0.564, −0.461], much stronger than its −0.122 on the varied panel.

But the amplification is not specific to `u008`. **All nine** selected units land in the range 0.310–0.564 on the fixed panel versus 0.029–0.182 on the varied panel, and the *largest* association belongs to `u003` — an unnamed unit that is silent on 80% of rows. So this is a panel-wide effect of holding drive constant (with drive varying, drive-driven variance swamps a geometry-only association), not evidence for a zonal-flow mechanism in the named candidate.

### 4. Replication failures

Several checks that a genuine mechanism should have passed did not:

- **No cross-member agreement.** Across the nine selected units, signed correlations run from −0.361 to +0.134 and lags from −47 to +48. There is no common sign, no common spatial offset. Different ensemble members, trained on the same problem, are not finding the same spatial story.
- **Sign reversals between panels.** `u021` goes from −0.162 (varied) to +0.155 (fixed); `u027` goes +0.134 → −0.148. The *unit is unchanged*; only the GX field differs. Both reversals clear their own nulls, meaning these are confident contradictions rather than noise.
- **Silent units inflate magnitudes.** Three of nine units are constant (silent) on 42–82% of rows. A constant profile has no spatial ordering, so by convention it contributes zero correlation — which drags pooled numbers toward zero and makes the "top-10% mask" balloon to 50–84 positions instead of 10. The report recomputes over active rows only and discloses both. It doesn't change the conclusion, but it changes what the per-unit numbers *mean*.
- **Only two of nine units were ever actually named.** The other seven were unresolved or uncharacterized in S05. So this isn't even a clean replication test of the two S05 hypotheses.

### 5. Contradicting cases given equal weight

Twenty case-study equilibria, five supporting and five contradicting for each hypothesis. The report makes a point worth noting: the contradictions are not marginal leftovers. For `u001`, the supporting rows reach correlations of −0.95 to −0.93, but the strongest contradicting row is **+0.864** — comparably extreme in the opposite direction. One flux tube contradicts both hypotheses at once.

---

## The conclusions

**S07 promoted no candidate to "physically supported."** That is the verdict, stated up front in the report.

What survives is a modest, honest claim: two internal activation patterns covary with real GX structure, negatively and at a substantial spatial displacement. What does not survive is any inference that the network is using the physics you'd hope it uses. The gap between the activation result (−0.36) and the attribution result (−0.02) is the whole story — the network has spatially structured internals that correlate with physics, but the prediction pathway doesn't route through the locations where heat actually flows.

Three limits bound even the surviving claim:

- **Association, not causality.** $Q(z)$ is a diagnostic output of GX, not a lever. Nothing here shows which geometry cell *controls* flux. Establishing that would need new perturbative GX runs, which the report explicitly defers as outside the step's budget.
- **The attribution evidence is about the network, not the plasma.** The off-manifold reference path means those numbers describe extrapolation behavior.
- **The fixed-panel results are not independent replication.** The same geometry-derived densities are being reused; only the GX field and drive change.

The step also deferred a census of all live units, on the reasonable grounds that widening the search *after* seeing these outcomes would multiply the chances of a spurious hit.

**How I'd summarize it for a reader of the eventual paper:** Step 7 was the step designed to convert the internal maps into physics, and it returned a well-instrumented negative. The most interesting thing it produced is not the −0.361 headline but the contrast between activation and attribution — evidence that "a unit lights up in a physically meaningful pattern" and "the model uses that pattern to predict" are genuinely separable, and that in this network they come apart.

-------------

# Codex summary — Does the network focus where GX transports heat?

## The question

Earlier steps found two kinds of spatial pattern inside the neural network:

1. A **density** is where an internal network unit activates along the 96-point
   periodic field line.
2. An **attribution** is where an input cell receives credit or blame for a
   prediction. Here the primary attribution method is Integrated Gradients: it
   divides the prediction change from a reference geometry among the input
   cells. That reference path is **off-manifold** (not guaranteed to consist of
   valid plasma equilibria), so the result describes how the network behaves
   along a constructed path, not how the plasma responds to a realizable edit.

S07 compares both patterns with `Q_avgs_vs_z`, the heat flux as a function of
position computed by GX. GX $Q(z)$ is a held-out physical diagnostic. The
network never sees that position-resolved curve as an input and was not trained
to reproduce it position by position. Similarity is therefore interesting
independent evidence—but it is still association, not causation.

The analysis uses three independently trained ensemble members and 1,000
equilibria. It keeps stable/near-floor and unstable simulations separate, keeps
signed member results before any summary, and explains the native model output
`max(log Q, -2)` rather than exponentiating it.

## Main answer

Some internal activation densities resemble GX $Q(z)$, but the prediction
attributions do not provide the signed, member-replicated evidence needed to
call this a physical mechanism.

The top member's most important internal unit has a spatial rank correlation of
**-0.361** with GX $Q(z)$ on varied-gradient unstable simulations. Rank
correlation measures whether two curves order positions similarly; -1 is
perfect reverse ordering and +1 is perfect matching ordering. The 95% interval
is **[-0.388, -0.333]**, obtained by resampling whole equilibria to estimate
uncertainty. The best match occurs at lag **+22** of 96 grid points, not at the
same coordinate; 91.2% of resamples return within four positions of that lag.
Its association is negative: high activation goes with low
$Q(z)$. After sign-flipping $Q(z)$ to measure that inverse pattern, their
top-10% regions overlap **1.63 times chance**. Without the sign flip, high
activation and high $Q(z)$ overlap only **0.54 times chance**—they avoid one
another more than chance predicts.

A second unit previously associated with radial drift/geodesic curvature has
correlation **-0.268 [-0.294, -0.241]** at lag **+44**. Its spatial offset is
stable as well: 100% of resamples return within four positions of +44. The two
candidate densities are nearly unchanged in the fixed-gradient comparison
because density depends only on geometry; this is a robustness comparison
against a second GX field, not an
independent replication. These are credible activation-to-physics associations.

A separate limitation affects three of the other seven selected units. They are
silent—constant along all 96 positions—on 42–82% of unstable equilibria. A
constant curve has no spatial ordering, so the registered statistic assigns it
correlation zero; its tie-inclusive top-10% mask also expands to all 96
positions. The published table now records those counts, active-row-only
correlations, and mean mask widths. Restricting descriptively to rows where a
unit varies changes the nine-unit range from **-0.361 to +0.134** to **-0.369
to +0.182**, which remains mixed in sign and lag. The two named candidates are
constant on only 17/760 and 45/760 rows, so their result is essentially
unchanged.

The fixed-panel agreement is not general across the other selected densities:
seven of nine units keep the same association sign, but `.409:u021` reverses
from -0.162 to +0.155 and `.409:u027` from +0.134 to -0.148. Both reversals are
well above their per-comparison lag-search nulls, strengthening the conclusion
that there is no common member-level spatial interpretation.

But the off-manifold attribution diagnostic gives a different picture. Keeping
the full sign of Integrated Gradients, its correlations with $Q(z)$ are only **-0.021,
-0.013, and -0.012** in the three members, with incompatible lags **-36, +47,
and +48** on 760 unstable equilibria. Selecting the largest value from 96 lags
can create a small peak by chance, so we also broke the equilibrium pairing 200
times and repeated the full search. The -0.013 peak lies at the estimated null
threshold, so its binary pass/fail label is not stable; the -0.012 peak is below
the threshold. The -0.021 peak clears it, but is still negligible and does not
replicate.
Moreover, the `.409` +48 offset itself is unstable: only 31.2% of resamples
return within four positions, below the registered 50% lag-stability rule.

The original, non-canonical $f_m$ models tell the same but weaker story. Their
signed correlations are **-0.0133, -0.0071, and -0.0095**; only the first
exceeds its complete 96-lag null (**0.0118, 0.0117, and 0.0115**), and all three
selected lags fail the 50% stability rule. Their positive-only correlations are
**+0.230, +0.248, and +0.231**, still well above nulls of **0.0323, 0.0350, and
0.0352**. Canonical minus original is **-0.00795, -0.00570, and -0.00213** for
signed correlations and **+0.03566, +0.03194, and +0.03036** for positive-only.
Exact shift invariance therefore modestly strengthens the pattern; it does not
create the report's negative conclusion or turn the signed evidence into a
replicated mechanism.

If all negative contributions in the off-manifold diagnostic are thrown away, the correlations rise to
**+0.266, +0.280, and +0.262** at lag 0 or +1. That is a genuine and repeatable
resemblance along that constructed reference path: positively contributing
input cells tend to sit where $Q(z)$ is positive. It is also only half the
network explanation, not plasma evidence. Negative contributions are
part of the model's prediction, so the positive-only result cannot replace the
nearly null signed result.

## The zonal-flow hypothesis

The geodesic-curvature candidate is modestly associated with the GX zonal-flow
observable in the varied-gradient unstable panel: **-0.122 [-0.183, -0.060]**.
In the fixed-gradient panel it becomes much stronger, **-0.513 [-0.564,
-0.461]**. The top bad-curvature candidate changes from an unresolved
**+0.032 [-0.040, +0.099]** to **+0.310 [+0.247, +0.372]**.

This contrast does not establish drive dependence. The fixed panel holds drive
at $(3,0.9)$ while the varied panel does not, so drive-driven variance or
confounding can obscure a geometry-only association in the varied panel. The
fixed result supports an association conditional on constant drive, not a
causal mechanism. It is also not candidate-specific: all nine selected
densities reach absolute fixed-panel correlations of 0.310–0.564, and the
largest is the unnamed, mostly silent `.437:u003` at -0.564 rather than the
geodesic candidate. On active rows alone the nine-unit absolute range is
0.150–0.712; all nine remain same-signed, `.371:u017` weakens most (0.315 to
0.150), and `.437:u003` remains largest.

That pooled `.437:u003` result is itself mostly a silent-versus-active contrast.
The unit is exactly zero on 605/760 varied unstable rows; on the 155 rows where
it fires, its zonal association reverses from **-0.182** pooled to an unresolved
**+0.074 [-0.069, +0.227]**. In the fixed panel it changes from **-0.564**
pooled to **-0.712 [-0.776, -0.619]** over 228 active rows. The two named
candidates move by at most 0.05, so the constant-drive association remains real
but is still panel-wide rather than candidate-specific.

## What the fixed/varied pairs show

For the same geometry, the fixed and varied simulations have substantially
different heat flux. Across all rows, fixed minus varied GX flux is **+1.559
[+1.434, +1.694]** in the native clipped-log units. It is +3.356 when either
simulation is stable/near-floor and +0.957 when both are unstable, so the pooled
value must not be read as a single-regime effect. Each member predicts almost
the same all-row difference. The physical $Q(z)$ curves remain strongly related:
spatial rank correlation **0.736 [0.708, 0.762]** over all rows and **0.874
[0.860, 0.888]** over the 749 both-unstable pairs, both at lag 0.

This is a natural paired comparison, not a constant-drive comparison within
each pair. The fixed panel uses $(a/L_T,a/L_n)=(3,0.9)$ across geometries, while
the paired varied row can have different drives. It controls geometry but does
not by itself isolate the effect of drive.

## Contradictions were kept

For each of the two physical hypotheses, S07 publishes five naturally occurring
equilibria that support the population sign and five that contradict it. Some
contradictions are strong: for the dominant unit, supporting per-row
correlations reach about -0.95, while a contradicting row reaches +0.86. These
are not discarded as outliers. They show why a population association is not a
one-to-one physical definition of the unit.

## Bottom line

The networks contain activation patterns that track real GX structure, especially
for the two bad-curvature/flux-compression and geodesic-curvature candidates
supported in S05. Four other selected units were already unresolved in S05, and
the three units from the third member were not studied there. The cross-member
comparison is therefore conditional on an importance-ranked, mostly unnamed set,
not an independent replication of the two named candidates. Moreover:

- density signs and lags do not replicate across members;
- three secondary densities are silent on many rows, and their pooled
  magnitudes partly reflect the documented zero-correlation convention;
- complete signed off-manifold attributions have almost no spatial association
  with $Q(z)$; one peak is below the 96-lag null, one is at its threshold, and
  the resolved peak remains negligible;
- the stronger attribution result appears only after negative evidence is
  removed;
- stronger fixed-drive zonal associations are panel-wide rather than
  candidate-specific, while the varied panel is drive-confounded and its
  strongest pooled unit is mostly a fired-versus-silent contrast; and
- strong natural contradictions exist for both hypotheses.

The appropriate conclusion is therefore narrower than “the network discovered
the GX transport mechanism.” It discovered internal spatial representations
that covary with GX transport structure, but S07 does not establish where or
how the model uses geometry in the same sense that the plasma produces
$Q(z)$. No candidate is promoted to a physically supported mechanism.

The full numerical record, including all lags and negative results, is in
[the technical report](S07_physics_alignment.md) and
[the registered artifacts](S07_artifacts/manifest.json).
