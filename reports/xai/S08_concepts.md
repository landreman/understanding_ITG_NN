# S08 — Concept probes and TCAV in the hidden layers

## Result

Known geometry concepts are widely encoded in the convolutional stack, but
encoding does not guarantee that the head uses a direction. A sparse probe (a
linear readout constrained to use few hidden features) achieved median
equilibrium-held-out $R^2=\mathbf{0.747}$ over 150 member/layer/concept cells,
versus **-0.00208** after permuting concept labels and **-0.00228** for a
separate random-concept control. Nine of ten concepts pass the preregistered
encoding gate in all 15 member/layer cells. Zonal-flow magnitude does not:
its median $R^2$ is **0.031**, and no zonal cell is called encoded or used.

TCAV-like tests (Testing with Concept Activation Vectors: measuring the native
output derivative along a hidden-layer direction associated with a concept)
are sign-stable across five paired matched-example subsamples in **146/150**
cells. Raw uniform-direction interventions exceed eight isotropic random
directions in **120/150** cells and activation-scale-matched random directions
in **119/150**. The complete claim gate is stricter: held-out encoding,
parent- and subset-level matched-example balance, counterexample sign stability,
a false-discovery-rate-adjusted 500-resample interval, and a scale-matched
intervention/random ratio above one. **83/150** cells pass all conditions.

The most replicated concept is $f_{\mathrm{stab}}$, encoded and used in
**15/15** member/layer cells. Compression, cross-channel co-location, and
$\log f_Q$ each pass in **12/15**; geodesic-curvature magnitude passes in
**10/15**; bad curvature and local $Q(z)$ concentration in **7/15**; parallel
scale in **6/15**; and $\log\langle|\nabla x|\rangle$ in **2/15**. These counts
preserve signed member/layer results rather than averaging members before
testing.

Depth matters. Median TCAV derivatives for bad curvature change from **-0.460**
at layer 1 to **+0.136** at layer 5; geodesic curvature changes from **-0.554**
to **+0.147**; cross-channel co-location changes from **-0.596** to **+0.304**.
In contrast, $f_{\mathrm{stab}}$ is positive at every layer (median **+0.327**
to **+0.678**), while parallel scale is negative early (**-0.699**) and near
zero late (**+0.004**). A single unsigned or depth-averaged concept score would
hide these opposing mechanisms.

## Estimand and cohort

The explained function is each top-three member's S02 canonical exactly
shift-invariant output

$$
\tilde f_m(X,g_T,g_n)=\operatorname{MLP}_m(\bar u_m(X),g_T,g_n),
$$

in native $\max(\log Q,-2)$ units. No result exponentiates the prediction.
Each layer representation is the position mean of its full 96-position
$\grave{\mathrm a}$ trous ReLU/max-pool map, so the representation is exactly
cyclic-invariant and retains a stable feature axis. S08 does not produce a
position-resolved map, so the interpretation contract's canonical/original
map comparison is not invoked; every directional output effect is explicitly
for `invariant_tilde_f`.

The cohort is S01's frozen 1,000-row varied-gradient interpretation panel, one
tube from each of 1,000 `equilibrium_files`: **240 stable/near-floor** and
**760 unstable** rows. S02 supplies the canonical model, and S05 supplies the
concept vocabulary. Production read the external HDF5 dataset, never
`tests/data/review_slice.h5`.

Probe fitting and matched-example construction use all 1,000 equilibria. The
more expensive directional-derivative and hidden-intervention calculations use
a deterministic random subsample of **96 equilibria: 25 stable/near-floor and
71 unstable**. Every artifact row distinguishes these `derivative_*` counts
from the full probe cohort. The number 96 here means equilibria; separately,
each layer map has 96 spatial grid positions.

The registered run is `concept-probes-top3-panel1000`. The published
[manifest](S08_artifacts/manifest.json) records the external dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`,
checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`,
CPU execution, seed 20260823, and **2,203.10 s (36.72 min)** measured directly
by the fresh final production run. No resume replaced that measured wall time
in the published manifest.

## Methods

### Concept scores and matched examples

The ten continuous concepts are $\log f_Q$, $f_{\mathrm{stab}}$, compression,
bad curvature, geodesic-curvature magnitude, robustly scaled parallel Fourier
scale, cross-channel co-location, $\log\langle|\nabla x|\rangle$, local $Q(z)$
concentration, and $\log_{10}$ zonal-flow magnitude. Geometry concepts are
observed comparisons, not interventions. Multichannel parallel scale divides
the seven channels by panel IQRs before combining them.

For each concept, the high/low selection first removes a linear association
with $a/L_T$, $a/L_n$, a simple geometry scale, and equilibrium class, then
selects residual extremes within class and pairs them by nearest nuisance
values. [Matched examples](S08_artifacts/matched_examples.csv) retain source
row and equilibrium IDs. The [balance table](S08_artifacts/matching_balance.csv)
reports standardized mean differences (mean separation in pooled standard-
deviation units). Nine concepts have maximum absolute imbalance below 0.25.
Zonal magnitude fails at **0.352** for $a/L_T$; it is kept but fails the use
claim gate.

### Equilibrium-grouped sparse probes

Each layer/concept/member uses five outer and three inner folds, with complete
`equilibrium_files` assigned to folds. Inner folds select among four L1
penalties using a deterministic near-best-score sparsity rule. Outer-fold
predictions give the reported $R^2$; a final all-panel fit supplies a direction
but never its own training score. Group-level label permutation and a distinct
equilibrium-level Gaussian random concept are null controls. The complete
[probe table](S08_artifacts/probe_scores.csv) contains 150 scientific and 15
random-control rows.

Stable/near-floor and unstable $R^2$ values are computed from the untouched
outer-fold predictions. Most geometry results are similar by regime. The main
exception is local $Q(z)$ concentration: median $R^2$ is **-0.087** on
stable/near-floor rows and **0.362** on unstable rows. Zonal scores are strongly
negative within either stratum because their already-small between-row variance
shrinks further.

### Directional derivatives and interventions

Five CAVs per cell come from 75% subsamples of the matched high/low *pairs*;
sampling pair indices jointly preserves the nearest-neighbor pairing. The
artifact reports the worst nuisance imbalance among those five subsets. Their
five normalized directions are averaged into one declared CAV. That same CAV
produces the reported derivative, interval, signed intervention, and
orthogonal-complement ablation, so the evidence is not combined across two
different vectors. The derivative sums gradients over all 96 spatial positions
because the direction is added uniformly to the layer map. Member/sample signs
are retained. Intervals resample all 96 selected `equilibrium_files` 500 times.

The in-repository TCAV implementation is deliberate: it exposes regression
derivatives and the exact hidden continuation used here directly, whereas
Captum's packaged TCAV workflow is classifier-oriented. Captum is installed in
the environment but does not produce an S08 artifact column.

The intervention shifts the real hidden representation in both directions and
continues the canonical network from that layer. Projecting each centered
representation into the CAV's orthogonal complement provides the registered
direction-removal diagnostic. Both edits are tagged
`deliberately_off_manifold_diagnostic`: a hidden activation edit diagnoses the
network, not the plasma. Eight isotropic equal-norm directions and eight random
directions weighted by each hidden unit's panel interquartile range receive the
same edit magnitude. The latter prevent near-dead units from making the control
artificially weak and supply the claim gate; both ratios are published.
[TCAV/use results](S08_artifacts/tcav_use.csv) retain the raw results even when
the combined claim gate fails.

The pooled scale-matched intervention ratio exceeds one in 119/150 cells. It
does so separately in **116/150 stable/near-floor** cells and **117/150
unstable** cells; these are not the same cells. Nineteen pooled passes fail on
the stable/near-floor stratum, so `use_claim_permitted` is a pooled-cohort claim,
not a claim that the direction is used near the output floor. The artifact
therefore publishes concept-effect RMS, both control RMS medians, and both
ratios separately for the 25 stable/near-floor and 71 unstable derivative rows.

The orthogonal-complement projection ablation is much larger than the small
$\pm0.2$ direction intervention by construction: median RMS **0.609** versus
**0.140**, and larger in **150/150** cells. Its largest concept-family medians
are $f_{\mathrm{stab}}$ (**0.958**) and, contradictorily, zonal magnitude
(**0.896**). Thus the ablation also cannot rescue a zonal-use claim; removing a
row's entire coordinate is disruptive even when that concept is not decodable.
For transparency, this projection ablation is the operational substitute for
PLAN's “intervene along orthogonal complements,” and activation-scale matching
is the operational substitute for “equally decodable random directions.” The
latter is stronger than the old isotropic control but does not assert identical
concept decodability.

### Claim rule

A cell is called encoded and used only when all of the following hold:

1. outer-fold $R^2\ge0.1$ and at least 0.1 above its permuted control;
2. both the parent matched set and every counterexample subset have maximum
   absolute standardized mean difference at most 0.25;
3. at least 80% of counterexample sets give the same derivative sign;
4. the equilibrium-bootstrap 95% interval excludes zero and its p-value passes
   Benjamini-Hochberg false-discovery-rate control at 0.05 across all 150 cells;
   and
5. the concept-direction intervention RMS exceeds the median
   activation-scale-matched random-direction RMS.

This rule is a reporting gate, not a tuned classifier. Every component and the
final Boolean appear in the [encoding/use matrix](S08_artifacts/encoding_use_matrix.csv).
Three new safeguards were conservative but mostly non-binding in this run:
using scale-matched rather than isotropic controls changes **0/83** final
verdicts; FDR adjustment changes **0/83** (one interval-level verdict changes,
but that cell already fails another condition); and subset-balance gating
removes **3** otherwise permitted cells. The 500-resample p-value resolution is
limited: **108/150** cells sit at the minimum $2/501$.

## Negative and contradictory results

- Zonal magnitude is the clearest contradiction. It is not decodable, its
  matching misses the balance threshold, yet all 15 arbitrary zonal directions
  beat the random intervention median. This is evidence that an intervention
  ratio alone can manufacture a use story; **0/15** zonal claims are permitted.
- Direction intervention is not universal: 30/150 cells fail the isotropic
  random comparison, 31/150 fail the stronger activation-scale-matched
  comparison, and only 83/150 pass the complete gate.
- Parallel scale is decodable (median $R^2=0.544$) but used in only **6/15**
  cells. Information available to a probe need not drive the prediction.
- Local $Q(z)$ concentration is weakly encoded overall and fails on
  stable/near-floor rows, although seven member/layer cells pass the full gate.
- Bad-curvature, geodesic-curvature, and co-location directions reverse sign
  with depth. They cannot be summarized as monotone positive mechanisms.
- Hidden interventions are off-manifold. Even a fully gated cell explains the
  network, not a valid equilibrium or causal plasma response.

## Acceptance criteria

| PLAN criterion | Verdict and evidence |
| --- | --- |
| “encoded” and “used” are separate columns | **Pass.** All 150 rows carry separate held-out `encoded_r2`, signed derivative, intervention, component-gate, and final `use_claim_permitted` columns. Only 83/150 pass the complete rule. |
| concept classifiers generalize by equilibrium | **Qualified pass.** Median outer-fold $R^2=0.747$ versus -0.00208 permuted and -0.00228 random; nine concepts pass in all 15 cells. Zonal magnitude fails in all 15 and is retained as a negative result. Both nested levels split by `equilibrium_files`. |
| TCAV is stable across counterexample sets | **Pass with matching failures exposed.** 146/150 cells retain their sign across five paired subsets; parent and subset nuisance balance is published and included in the gate. Zonal parent balance fails at 0.352 and receives no use claim. |
| direction interventions beat matched random controls | **Qualified pass.** 120/150 raw interventions beat the isotropic random median and 119/150 beat activation-scale-matched random controls; $f_{\mathrm{stab}}$ passes the complete gate in 15/15 cells. The full gate permits 83/150; failures and both ratios remain published. |

## Failed checks

- The initial test run failed all intended S08 paths with `NotImplementedError`.
- The first pilot showed a summary-only `KeyError` after adding random-concept
  rows; the artifact summary now selects only rows carrying permutation values.
- A batched random-control implementation was numerically equivalent but slower
  on CPU and was reverted before production.
- Inspection found the uniform layer derivative averaged instead of summed over
  position. A continuation/shift test and the corrected spatial sum now pin the
  intended uniform-edit derivative.
- The first production run failed its post-run nuisance audit: zonal high/low
  sets differed by 1.64 pooled standard deviations in $a/L_T$. A synthetic
  confounding test was written first and failed at 1.99; residualized matching
  made it pass. The original production run was discarded and recomputed.
- The corrected run leaves one explicit failed balance check: zonal $a/L_T$
  imbalance is 0.352 versus the 0.25 threshold. It is gated out, not hidden.
- The first automated review reproduced every checked number but found that the
  reported derivative and intervention used different directions, random
  controls ignored hidden-unit activation scale, and the 96-equilibrium use
  subsample was labeled as if it contained all 1,000 rows. The original use
  statistics were discarded. The corrected production run uses one direction,
  publishes both control families, labels 25/71 regime counts, implements
  orthogonal-complement ablation, and adds multiple-testing control.
- The second automated review closed those blockers but found that intervention
  ratios still pooled the two output regimes, the new script-level assembly and
  gate logic lacked direct tests, and the orthogonal ablation had no numerical
  interpretation. The final run publishes both regime splits; script tests now
  pin paired subsets, direction aggregation, bootstrap/FDR, controls, and
  balance gates; and both reports retain the ablation contradiction.

## Mutation testing

The following deliberate mutations turned the focused suite red and were
reverted:

1. assigning outer folds by row rather than `equilibrium_files` failed the
   repeated-equilibrium disjointness test;
2. removing nuisance residualization failed the strong continuous-confounder
   balance test (standardized difference 1.99); and
3. exponentiating the native scalar in the analytic direction test changed the
   known derivative away from 2.5 and failed the native-output assertion.
4. replacing the uniform-edit spatial gradient sum by a mean failed the finite
   hidden-map intervention comparison;
5. negating that gradient failed the same comparison with the opposite sign;
6. reporting the all-panel in-sample probe fit as held-out $R^2$ failed the
   expanded noisy-feature permutation fixture; and
7. giving isotropic random controls half the concept edit magnitude failed the
   one-dimensional equal-step control fixture.
8. using only the first counterexample CAV instead of all normalized directions
   failed the production-script aggregate-direction test;
9. gating on the isotropic rather than activation-scale-matched ratio failed the
   production-script complete-gate test; and
10. relaxing subset balance from 0.25 to 0.50 failed the same end-to-end gate
    fixture.

## Deferred

Network-dissection IoU, mutual information, and selectivity tables (item 5) are
deferred. The MVD (items 1–3) and item 4's direction interventions are complete;
adding a second mask-downsampling pipeline after a 36.72-minute final run
would exceed S08's one-session budget. S05 already publishes unit/concept mask
overlap at the bottleneck; S08 protects the layerwise encoding/use result.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s08-pilot XDG_CACHE_HOME=/private/tmp/cache-s08-pilot \
  .venv-xai/bin/python scripts/xai_s08_concepts.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s08-prod XDG_CACHE_HOME=/private/tmp/cache-s08-prod \
  .venv-xai/bin/python scripts/xai_s08_concepts.py \
  --output-dir output/xai/S08/concept-probes-top3-panel1000-review3
MPLCONFIGDIR=/private/tmp/mpl-s08-resume XDG_CACHE_HOME=/private/tmp/cache-s08-resume \
  .venv-xai/bin/python scripts/xai_s08_concepts.py --resume --no-publish \
  --output-dir output/xai/S08/concept-probes-top3-panel1000-review3
.venv-xai/bin/python -m pytest tests/xai/test_concepts.py \
  tests/xai/test_concepts_script.py tests/xai/test_concepts_artifacts.py -q
source .venv-xai/bin/activate && make check
```

## Reviewer reproduction

**Recomputable on the slice.** All 1,000 parent row IDs are S01 panel rows in
`tests/data/review_slice.h5`. Translate them with
`load_review_slice_index().slice_rows()` before loading. The reviewer can
recompute the ten concept scores, exact class membership, nuisance balance,
five-layer canonical representations, grouped outer predictions, permutation
and random-concept controls, TCAV signs, and intervention ratios for all three
members. The derivative/intervention row IDs are the deterministic 96-row
subsample recorded by the config and seed. The full run took 2,203.10 s, so the
96-row pilot is the practical proxy; agreement on axes, grouped folds, signs,
and claim gates checks the wiring.

**Checkable from committed artifacts alone.** The 150-cell matrix, 165 probe
rows, 3,940 matched-example rows, balance table, summary, figure, and manifest
are committed under [S08 artifacts](S08_artifacts/). Artifact tests recompute
the headline fractions from the CSVs and verify every manifest output hash.

**Not checkable off the researcher's machine, and why.** No headline number
depends on an off-panel row or a git-ignored scientific array. Exact bytewise
reproduction of the external-data run requires the 678 MB source HDF5 and about
47 CPU minutes. The review slice contains the same 1,000 panel rows and all
required held-out diagnostics, so a full slice rerun is numerically equivalent;
the pilot is the cheaper nearest proxy.
