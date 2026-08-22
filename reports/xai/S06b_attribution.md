# S06b — Scaled input-space attribution

## Status

Complete. S06b scales the S06a-registered methods across the stored-validation
top 10 and a rank-spaced wider-member sample. Every number concerns the native
network output, `max(log Q, -2)`, never `Q` or `exp(prediction)`. The canonical
object is S02's exactly shift-invariant `invariant_tilde_f`; the original
`original_f` is retained beside it. Fixed and varied simulations, and
stable/near-floor and unstable rows, are never pooled.

The registered run is
`scaled-attribution-top10-panel2000-rank10-panel512`. Its ignored full map is
`output/xai/S06b/scaled-attribution-top10-panel2000-rank10-panel512/attribution_maps.h5`
(304 MB), and the verbatim registered manifest is
[`S06b_artifacts/manifest.json`](S06b_artifacts/manifest.json). The initial full
computation took **4,645.802 s**. A hash-validated corrective resume regenerated
postprocessing after two indexing/coverage corrections without recomputing the
full map; its command and wall time are in the final manifest, which also records
the initial wall time and the prior manifest hash.

## Estimand, cohort, and registered methods

For member (m), method (a), function (h\in\{f_m,\tilde f_m\}), gradient set
(g), and S01 panel row (i), the primary artifact is the member-level map

\[
A_{a,m,g,i,c,z}(h),
\]

with explicit member, sample, channel, and periodic-position axes. The top-10
headline run uses all **1,000** S01 panel geometries under varied drives and the
same 1,000 under fixed drives. It contains 240 varied and 23 fixed
stable/near-floor rows. The wider-rank sensitivity uses five rank-spaced members
from stored-validation ranks 11–50 and five from ranks 51–100 on 256 rows per
gradient set. Member selection uses stored validation rank only; no attribution
result or test score selected a member.

S06a registered:

- **Low-pass Integrated Gradients (IG)** as the primary path method. IG (an
  attribution method that averages the prediction's slope along a path) runs for
  64 path points from each input's frequency-8 low-pass version to the original.
  It is signed and contribution-valued. The registered estimator backend is
  Captum 0.9.0's Gauss–Legendre implementation, recorded as
  `integrated_gradients_captum` in each method-bearing artifact. The path is a
  deliberately off-manifold diagnostic: a smoothed input need not be a
  realizable equilibrium.
- **Matched-observed periodic extremal mask** as an explicitly secondary
  perturbation fallback. It is a magnitude-only map optimized for 60 steps. Its
  replacement endpoint is observed, but cellwise interpolation and replacement
  remain off-manifold, and its fixed-background symmetry failure is retained.

Scalar-drive results are signed local sensitivities multiplied by robust S01
reference-cohort scales, 1.91567 for `a/L_T` and 0.904459 for `a/L_n`. They are
not path contributions. Geometry IG already includes its input-minus-baseline
factor, so raw gradients are never compared across the seven differently scaled
geometry channels.

## Results

### The coarse channel ordering is common; the signed position map is less so

For canonical low-pass IG, the median pairwise Spearman channel-rank agreement
(agreement in the ordering of seven mean-absolute channel scores) is **0.9643**
on varied rows and **1.0000** on fixed rows. The mean cellwise sign-agreement
fractions are only **0.7485** and **0.7452**, compared with an exact independent-
sign null of **0.6230** for ten members. Thus the observed varied-row agreement
is only about one third of the way from its null to perfect agreement. The top 10
broadly agree about which channels respond along this path, but they do not
implement one identical signed position-by-position mechanism. With seven
channels, 0.9643 is exactly one adjacent rank transposition and the smallest
possible non-unit Spearman step is 0.0357, so this channel statistic is coarse.
The corresponding original-function
channel-rank agreements are 0.9286 and 0.9643, with sign agreement 0.7122 and
0.7153. Exact values for both methods and both functions are in
[`member_agreement.csv`](S06b_artifacts/member_agreement.csv).

On canonical varied unstable rows, the mean absolute low-pass-path response and
the 95% hierarchical bootstrap interval (resampling both members and whole
`equilibrium_files`) are:

| Rank | Channel | Mean absolute attribution | 95% interval | Mean sign agreement over z |
| ---: | --- | ---: | ---: | ---: |
| 1 | `gbdrift0_over_shat` | 0.002668 | 0.002388–0.002967 | 0.745 |
| 2 | `gbdrift` | 0.002232 | 0.001950–0.002536 | 0.771 |
| 3 | `gds21_over_shat` | 0.001547 | 0.001399–0.001723 | 0.726 |
| 4 | `gds2` | 0.001483 | 0.001121–0.001931 | 0.721 |
| 5 | `cvdrift` | 0.001121 | 0.000965–0.001284 | 0.771 |
| 6 | `gds22_over_shat_squared` | 0.000923 | 0.000818–0.001033 | 0.767 |
| 7 | `bmag` | 0.000584 | 0.000500–0.000672 | 0.719 |

`gbdrift0_over_shat` is therefore the largest *response to restoring short-scale
content from this particular low-pass reference*, not a baseline-free global
importance and not a plasma-causal effect. The signed median over position is
small for every channel because positive and negative locations cancel; signed
and absolute columns remain separate in
[`channel_consensus.csv`](S06b_artifacts/channel_consensus.csv).

The ordering is not universal across equilibrium classes. `gbdrift0_over_shat`
is largest in varied classes 0 and 2, `gds2` in class 1, and
`gds21_over_shat` in classes 3 and 4. This mixed result prevents calling one
geometry channel the shared physical feature of the ensemble. S07 must compare
the position-resolved maps with physics fields and GX diagnostics.

### Response size changes with regime

All six fixed/varied low-, medium-, and high-unstable-flux bins retain
`gbdrift0_over_shat` as their largest low-pass-path response. Its canonical
median absolute value decreases with flux: 0.003484, 0.002306, 0.001900 for
varied low/medium/high unstable flux, and 0.003025, 0.001661, 0.001152 for fixed.
This is a response along a frequency-removal path, not evidence that physical
importance decreases with heat flux.

The same response is larger where the members perform poorly or disagree. The
registered tertile boundaries are fitted on all panel rows, but feature summaries
then exclude every stable/near-floor row. On varied unstable rows, high versus
low mean-member absolute-error tertiles give 0.003628 versus 0.001513 for
`gbdrift0_over_shat` (2.4-fold); high versus low ensemble-spread tertiles give
0.003622 versus 0.001396 (2.6-fold). On fixed unstable rows the corresponding
spread values are 0.003451 versus 0.001063 (3.2-fold). This association is
descriptive: flux and other regimes still vary across those tertiles. All
stability, unstable-flux,
`a/L_T`, `a/L_n`, equilibrium-class, member-error, and ensemble-spread strata
are in [`stratified_consensus.csv`](S06b_artifacts/stratified_consensus.csv).
That artifact reports `sample_count_stable=0` and
`feature_claims_permitted=true` for every feature stratum, alongside its
estimand, validity tag, baseline convention, signed status, and estimator backend.

### Stable/near-floor rows remain report-only

Canonical low-pass absolute attribution summed over the seven channel medians is
only **0.376** times the unstable value on varied stable/near-floor rows and
**0.128** times the unstable value on fixed stable/near-floor rows. S06a already
showed that the low-pass endpoint changes the native output by median 0.0014 on
these rows and that low-pass IG does not beat its network-free displacement
control there. These 240 varied and 23 fixed rows are published separately, but
**no feature-level conclusion in this report uses them**. The artifact column
`feature_claims_permitted` is false for every stable row summary.

### Scalar-drive sensitivities have consistent signs

Across the ten members, the median member-level robust-scaled local sensitivity
of the canonical output to `a/L_T` is **+2.146** on varied rows (member range
2.072–2.276) and **+1.737** on fixed rows (1.640–1.826). The corresponding
`a/L_n` values are **−0.223** (−0.273 to −0.153) and **−0.735** (−0.899 to
−0.600). These are local network derivatives in native clipped-log units per
robust drive scale, not finite physical interventions. Member/function/gradient
set values are in [`scalar_sensitivities.csv`](S06b_artifacts/scalar_sensitivities.csv).

### Validation rank does not predict attribution stability

On a common 256-row-per-gradient-set panel for the top 10 plus five members from
each wider rank band, Spearman correlation between stored validation (R^2) and
canonical low-pass channel-rank agreement with the 20-member median map is
**−0.0738**. This is the expected null: the stored validation-score range is too
narrow and S01 showed its ordering is uncertain. The individual values and the
fixed rank-spaced member registration are in
[`rank_sensitivity.csv`](S06b_artifacts/rank_sensitivity.csv).

### Symmetry distinguishes the canonical and original explanations

For canonical low-pass IG across the top 10, the median co-shifted explanation
equivariance error is **9.12e−7**, at float32 roundoff scale; canonical prediction
invariance error is median 5.49e−8. The research machine has a platform-dependent
upper tail to 8.32e−5, while the automated Linux review recomputed all ten members
at 5.4e−8–7.2e−8. For the original network the corresponding
medians are **0.874** and **0.0435**, consistent with S02's pooling-phase
dependence. Canonical and original maps are not silently substituted, even
though their median member-level channel-rank correlation is high (0.929–0.964).

The secondary periodic mask has canonical median co-shifted error 3.64e−5 but
maximum 0.00464. More importantly, holding its registered matched background
fixed gives median error **0.931**. This reproduces S06a's negative result: the
mask is useful only as a labeled secondary perturbation diagnostic and is not a
fixed-background symmetry-conforming explanation. All member rows are in
[`symmetry_checks.csv`](S06b_artifacts/symmetry_checks.csv).

## Uncertainty

Every channel/stratum headline mean has 500 hierarchical bootstrap replicates:
members are resampled with replacement, and samples are independently resampled
by whole `equilibrium_files`, never by flux tube or row. The primary panel has
one tube per equilibrium, while the test fixture includes sibling rows and pins
whole-group draws exactly. The intervals quantify member and panel-equilibrium
sampling for this frozen interpretation panel; they are not population-weighted
prevalence intervals. The complete table is
[`hierarchical_uncertainty.csv`](S06b_artifacts/hierarchical_uncertainty.csv).

## Failed checks and corrections

1. The first pilot completed attribution but failed in canonical/original
   aggregation because a four-dimensional map was reduced along a nonexistent
   fifth axis. The axis and the analogous sensitivity selection were corrected;
   the rerun completed in 27 s.
2. A rank-correlation toy initially demanded \(\rho>0.8\), but two synthetic
   members intentionally tied. The test now pins the correct average-rank value,
   0.6324555, rather than pretending the tie is absent.
3. Pilot visual inspection found that boolean indexing after integer indices
   moved the row axis ahead of the member axis in the atlas only. The plot now
   slices the member-first tensor before applying its row mask.
4. The same NumPy advanced-indexing behavior affected the first production
   `stratified_consensus.csv`: it treated rows as the agreement axis. Headline
   consensus, uncertainty, raw maps, scalar results, rank sensitivity, and
   symmetry were unaffected. A hash-validated resume reused the exact 304 MB map
   and regenerated the table with member-first slicing. For example, varied
   high-spread mean cell sign agreement corrected from 0.530 to 0.730.
5. A line-by-line acceptance audit found stability strata but no separate
   low/medium/high unstable-flux tertiles. A test was added first, failed, and the
   six fixed/varied flux strata were added before the final registered resume.
6. Automated review found that covariate tertiles still included 240 varied and
   23 fixed stable/near-floor rows. The registered full-panel boundaries are now
   intersected with the unstable mask, changing the varied high/low error ratio
   from 5.0-fold to 2.4-fold. The artifact now pins the stable-row count and
   permission for feature claims on every row.
7. The same review found that the production estimator backend was discarded
   after attribution. The actual `AttributionMap.method` is now threaded into
   every method-bearing artifact; the registered IG backend is
   `integrated_gradients_captum`, and a Captum-path analytic toy test pins it.
8. A full check invoked in the inference-only `20240629-01-ML` environment failed
   that new test because the environment deliberately lacks the XAI extra. The
   required project-local `.venv-xai` check then passed all 152 tests, including
   the real Captum path. No test was skipped or relaxed.

Three deliberate post-run mutations turned the focused suite red and were
reverted: replacing whole-equilibrium draws with row draws failed the exact
bootstrap sequence; pooling stable rows into the unstable mask failed the
disjoint-strata assertion; and averaging members before retaining signed maps
failed the member-axis/opposing-mechanism test.

## Negative results and interpretation limits

- Stable/near-floor attribution remains near-uninformative and supports no
  feature claim, even though its rows and intervals are published.
- No perturbation method passed S06a's complete gate. The mask remains a
  secondary fallback and fails fixed-background equivariance at order one.
- Low-pass IG beat its network-free control only on unstable rows in S06a. Its
  parameter-randomization response was qualified (rank correlation 0.406), and
  its map agreed only 0.432 with robust-constant IG. Scaling reduces member
  uncertainty; it does not remove baseline sensitivity.
- Channel ordering is common, but exact signed position maps are less common;
  equilibrium classes also disagree about the largest channel.
- Every geometry path/edit is tagged deliberately off-manifold. The results
  explain network sensitivity, not plasma causality.
- The 25-row S01 near-threshold varied stratum remains analysis-limited and is
  not promoted to a separate population claim here.
- The mask optimizer is evaluated with the replacement it optimizes, so its
  apparent response is an in-sample diagnostic rather than independent
  faithfulness evidence.

## Acceptance criteria

| PLAN criterion | Verdict and evidence |
| --- | --- |
| Selected methods beat random/control maps on toy recovery and faithfulness | **Qualified/partial under the researcher-approved S06a decision.** Both selected artifacts had perfect analytic-toy recovery. Low-pass IG beat its displacement control on unstable deletion/insertion by 0.00964/0.01105 native units in S06a; stable intervals crossed zero. No mask passed the complete gate, so the matched mask is explicitly secondary. S06b preserves that verdict rather than converting scale into validity. |
| Selected methods respond to parameter randomization | **Qualified pass inherited from S06a.** Absolute-map rank correlation after full parameter reset was 0.406 for low-pass IG and 0.099 for the mask; the low-pass baseline factor retained substantial structure. |
| Baseline sensitivity is understood | **Pass with limitation.** S06a low-pass/robust-constant map correlation was 0.432 and the pilot selected a different baseline. S06b makes only low-pass-path claims and retains the baseline limitation in artifacts and prose. |
| Methods meet symmetry behavior permitted by S02 | **Pass for canonical low-pass IG; fail retained for the secondary mask.** Canonical low-pass co-shift median 9.12e−7 on the research machine; original median 0.874; mask fixed-background median 0.931. Tight tests pin the roundoff-scale canonical median and both order-one failures. See `symmetry_checks.csv`. |
| Uncertainty includes model and equilibrium sampling | **Pass.** All 168 function/method/gradient/stability/channel headline intervals use 500 joint member and whole-`equilibrium_files` resamples. The panel has one tube per equilibrium, so grouping and row resampling coincide here; the grouped implementation is pinned on a sibling-row fixture. |
| Signed and absolute summaries are distinguishable | **Pass.** Full and review HDF5 artifacts carry a method `signed` axis; IG has separate signed and absolute columns, while the mask is marked magnitude-only. |
| No feature is called common without an explicit agreement statistic | **Pass.** Canonical low-pass channel-rank agreement is 0.964/1.000 varied/fixed, while cell sign agreement is 0.749/0.745 against a 0.623 independent-sign null; the equilibrium-class contradiction is retained. |

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s06b-pilot XDG_CACHE_HOME=/private/tmp/cache-s06b-pilot \
  .venv-xai/bin/python scripts/xai_s06b_attribution.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s06b-prod XDG_CACHE_HOME=/private/tmp/cache-s06b-prod \
  .venv-xai/bin/python scripts/xai_s06b_attribution.py
MPLCONFIGDIR=/private/tmp/mpl-s06b-resume XDG_CACHE_HOME=/private/tmp/cache-s06b-resume \
  .venv-xai/bin/python scripts/xai_s06b_attribution.py --resume
.venv-xai/bin/python -m pytest tests/xai/test_attribution_scaled.py \
  tests/xai/test_attribution_scaled_artifacts.py -q
source .venv-xai/bin/activate && make check
```

The checkpoint SHA-256 is
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`;
the external dataset SHA-256 is
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`.

## Reviewer reproduction

**Recomputable on the slice.** All 1,000 parent row IDs are the S01 panel rows
inside `tests/data/review_slice.h5`, under both fixed and varied groups. Translate
the 16 committed `selected_review_maps.h5` parent IDs with
`load_review_slice_index().slice_rows()` before calling `load_hdf5_rows`; never
pass parent IDs directly to the slice reader. The reviewer can recompute both
functions and both methods for those 16 rows, all top-10 members, and compare the
labeled `(function, method, member, gradient_set, sample, channel, z)` arrays.
Canonical low-pass prediction/explanation equivariance and the fixed-background
mask failure are also directly recomputable on those rows. The low-pass wider-
rank statistic uses only panel rows and can be recomputed on the slice.

**Checkable from committed artifacts alone.** All headline channel, sign,
agreement, stratum, scalar, symmetry, validation-rank, and 500-draw interval
numbers are in committed CSV files. `summary.json` records the main verdicts;
the committed manifest hashes every small artifact, the review maps, code,
config inputs, dataset, and checkpoint. Artifact tests pin axes, method signs,
validity tags, exact hashes, row/member counts, resampling units, and the
stable-row no-claim flag.

**Not checkable off the researcher's machine, and why.** The exact 304 MB full
map is intentionally ignored. Recomputing it is possible from the review slice
because all 1,000 panel geometries are present, but the registered CPU run took
4,645.802 s and is too expensive for an ordinary automated review. The 16-row
committed maps are the nearest exact proxy; agreement on axes, signs, canonical
equivariance, and channel ordering would support the full-run wiring. Exact
matched-observed mask backgrounds use 512 off-panel, equilibrium-unique support
rows outside the review slice, so registered mask digits cannot be reconstructed
there. The nearest proxy is a matched background from the slice siblings;
agreement on the mask's magnitude-only status, co-shift behavior, and fixed-
background failure would support—but not reproduce—the registered result.

## Deferred

Nothing from S06b tasks 5–7. Nonselected S06a baselines remain published as
benchmark sensitivity analyses rather than being scaled to the full top-10
panel. LRP remains outside S06 because its layer-rule coverage was not
documented. No physics claim is made; S07 performs the registered comparison
with physics fields and GX diagnostics.
