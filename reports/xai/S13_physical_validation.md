# S13 — Physical validation with natural experiments

## Result

The fixed-gradient panel supplies strong **observational** associations but no
candidate clears the registered confounding gates for an intervention-ready
claim. With the drive held at $(a/L_T,a/L_n)=(3,0.9)$, Spearman rank
correlations with true GX `max(log Q,-2)` are **0.807** for the localized
25-point $f_Q$-integrand peak, **0.808** for bad-curvature/compression variance,
**0.659** for $f_{\rm stab}$, and **0.519** for geodesic-curvature/compression.
All 500 whole-equilibrium bootstrap intervals exclude zero. These correlations
show that the candidates track physical GX outcomes among naturally observed
equilibria; they do not show what changing one candidate would do.

Matching makes that limit sharper rather than removing it. High-versus-low
candidate matches give positive native-output differences of **1.598, 1.486,
1.878, and 1.317** units, respectively, but the largest remaining standardized
nuisance imbalance is **4.151, 2.192, 3.747, and 1.068**. The registered
acceptable threshold was 0.5. The candidate tails therefore occupy different
parts of geometry space even after nearest-neighbour matching. These are
observed comparisons, not causal effects.

The cross-fitted augmented inverse-probability weighting sensitivity analysis
(AIPW: it combines an outcome model with a model of which rows enter the high
candidate tail) reaches the same all-row sign for every candidate, but overlap
is only **0.228–0.478**, below the registered 0.8 requirement. Its most important
contradiction is the localized $f_Q$ peak: the both-unstable matched contrast is
**+1.544 [1.402, 1.693]**, while the adjusted unstable-tail contrast is
**-1.091 [-1.424, -0.806]**. This sign reversal under a confounding adjustment
with poor overlap is evidence that the observational comparison is not robust
enough to specify a physical sign.

Residual validation gives the clearest positive result. Starting from a
cross-fitted EBM using the paper-selected features
$(a/L_T,a/L_n,\log f_Q,f_{\rm stab},\log\langle|\nabla x|\rangle)$, adding
geodesic-curvature/compression raises fixed-panel held-out $R^2$ from **0.8125
to 0.8265** ($\Delta R^2=0.01394$) and improves mean squared error by **0.01880
[0.00617, 0.03252]** native units squared. The windowed $f_Q$ peak and
bad-curvature/compression add only $\Delta R^2=0.00580$ and 0.00589; their MSE
improvement intervals cross zero. Thus geodesic-curvature/compression contains a
small amount of fixed-drive physical predictive information beyond the paper's
selected feature set, but the natural data do not isolate it well enough to call
that information causal.

The resulting order is:

1. **Geodesic-curvature/compression — observational-physical.** The only
   candidate with a resolved fixed-drive gain beyond the full paper-selected
   baseline, and matched/AIPW physical associations of the same sign. Remaining
   imbalance is 1.068 and overlap 0.478, so it is not intervention-ready.
2. **Localized 25-point $f_Q$-integrand peak — observational-physical.** Strong
   raw and matched association, but no resolved gain beyond the full paper
   baseline, severe imbalance (4.151), overlap 0.228, and an unstable-row sign
   reversal under AIPW.
3. **$f_{\rm stab}$ — observational-physical.** Adds information beyond the
   $f_Q$ baseline, as expected because it belongs to the paper-selected set; its
   adjusted native contrast is unresolved and the matching remains imbalanced.
4. **Bad-curvature/compression variance — observational-physical.** Strong raw
   association but no resolved adjusted all-row contrast or gain beyond the full
   paper baseline, with severe remaining imbalance.

No candidate is graded `intervention-ready`. The prospective GX specification
therefore tests the top two competing directions rather than asserting either
one is already validated.

## Estimand, cohort, and provenance

The physical estimand is the association of S12's exactly cyclic-invariant
candidate features with observed GX quantities. The primary outcome is true GX
`max(log Q,-2)`, never `Q` or an exponentiated model prediction. Raw positive
`log Q`, `log10(Q_stds)`, `log10(zonal_phi2_amplitudes)`, parallel-profile
localization, and the fraction of positive $Q(z)$ cells are named physical
comparators. Network outputs are not recomputed in S13; prior model-mechanistic
grades come from S07/S12, and the manifest says `model_outputs_computed=false`.

The cohort is S01's frozen 1,000-equilibrium interpretation panel, one flux
tube per `equilibrium_files`, evaluated on both its fixed- and varied-gradient
GX simulations. The fixed panel has 23 stable/near-floor and 977 unstable rows;
the varied panel has 240 and 760. Fixed rows use the physical drive
$(3,0.9)$, never the legacy off-manifold marker. S12-v1 supplies the invariant
feature formulas and S01 supplies the seven robust channel IQR scales.

The registered run is `physical-validation-panel1000`. It took **73.58 s** on
CPU. The committed [manifest](S13_artifacts/manifest.json) records seed
20260825, all 1,000 parent row IDs, source hashes, exact command, package
versions, output hashes, dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`,
and checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.
The checkpoint is fingerprinted for continuity even though this step does not
run model inference.

The exact production command was:

```bash
MPLCONFIGDIR=/private/tmp/mpl-s13-prod \
XDG_CACHE_HOME=/private/tmp/cache-s13-prod \
.venv-xai/bin/python scripts/xai_s13_physical_validation.py
```

The pilot command was:

```bash
MPLCONFIGDIR=/private/tmp/mpl-s13-pilot \
XDG_CACHE_HOME=/private/tmp/cache-s13-pilot \
.venv-xai/bin/python scripts/xai_s13_physical_validation.py \
  --pilot --no-publish
```

## Methods

### Fixed-drive associations

For each candidate and physical outcome, Spearman correlation measures whether
their ordering across equilibria agrees. Intervals use 500 deterministic
bootstrap draws of complete `equilibrium_files`. All, stable/near-floor, and
unstable strata are separate. The fixed panel has only 23 near-floor rows, so
that stratum is retained in the artifact but not used for a headline claim.

`Q_stds` is a simulation-variability proxy, not an independent transport
measurement. Its candidate correlations almost duplicate those with `log Q`,
consistent with the known scaling of $Q_{\rm std}$ with $Q$. The localized
$f_Q$ peak and bad-curvature/compression correlate with $Q(z)$ localization at
**0.492** and **0.508**, while geodesic-curvature/compression is only **0.102**.
The fraction of positive $Q(z)$ cells is unresolved for every candidate. For
zonal-flow magnitude, $f_{\rm stab}$ has the largest raw correlation,
**0.592**, while the geodesic candidate is **0.315**; this does not isolate a
candidate-specific geodesic/zonal mechanism.

### Equilibrium-disjoint nearest-neighbour matching

Candidate exposure is the upper versus lower quartile. Rows are matched without
replacement and no `equilibrium_files` group can appear in two matched rows.
Distance is the root-mean-square difference after dividing nuisance coordinates
by their panel IQR; the caliper is 2.0 such units. Nuisances are the other paper
and candidate geometry summaries, robustly scaled parallel roughness, `nfp`,
`iota`, `shat`, `aspect`, `rho`, `aspect/rho`, and equilibrium class. Every pair
retains its candidate contrast, distance, outcome differences, and the maximum
standardized mean difference before and after matching.

There are 185–215 disjoint pairs per candidate. Matching does not repair the
main imbalance: three candidates worsen the largest standardized difference,
and the best, geodesic-curvature/compression, improves only from 1.206 to 1.068.
The balance failure is part of the result, not a reason to narrow the cohort.
Uncertainty resamples the disjoint matched-equilibrium pairs; each pair contains
two unique `equilibrium_files` and none is reused within a candidate analysis.

### Doubly robust sensitivity check

The AIPW check uses only upper- and lower-quartile rows. Five-fold cross-fitting
holds out complete `equilibrium_files`; within each training fold, logistic
regression predicts tail assignment and ridge regressions predict each potential
outcome. Propensities are clipped at 0.05 only after the overlap fraction is
recorded. Intervals resample complete equilibria from the cross-fitted influence
values.

“Doubly robust” does not mean immune to confounding. It means the estimator can
remain consistent if either its treatment-assignment model or its outcome model
is correct, under assumptions that include no unmeasured confounding and usable
overlap. Here measured overlap fails badly, so the large adjusted `Q_stds` and
zonal estimates for the localized peak are not credible effect sizes. They are
published as sensitivity failures.

### Residual validation

Five-fold equilibrium-grouped EBM fits predict true GX native output, separately
for fixed and varied panels. The first baseline is
$(a/L_T,a/L_n,\log f_Q)$; the second adds $f_{\rm stab}$ and
$\log\langle|\nabla x|\rangle$. Each remaining candidate is added alone using
the same folds. The primary stable-row measure is MSE, not $R^2$, because target
variance is compressed near the floor. MSE-improvement intervals resample
complete equilibria; signed residual/candidate rank correlations are retained
as a separate diagnostic.

Against the $f_Q$ baseline on fixed rows, all four additions improve all-row
MSE with intervals above zero: $f_{\rm stab}$ **0.07192**, geodesic/compression
**0.06437**, the windowed peak **0.03703**, and bad-curvature variance
**0.01725**. Against the full paper-selected baseline, only the geodesic
candidate stays resolved. On the varied panel the three non-baseline candidates
give small resolved all-row gains ($\Delta R^2=0.00633$–0.01053), showing that
the ranking is panel-dependent rather than a universal feature order.

## Prospective equilibrium-consistent GX test

The machine-readable [GX specification](S13_artifacts/gx_experiment_spec.json)
compares the top two directions:

1. increase/decrease geodesic-curvature/compression while constraining the
   localized $f_Q$ peak, $\log f_Q$, aspect, iota, shat, beta proxy, and `nfp`;
2. increase/decrease the localized 25-point $f_Q$ peak while constraining the
   geodesic candidate and the same global quantities.

For each of three typical unstable anchors, the geometry must be produced by a
VMEC boundary-coefficient continuation that recomputes a force-balanced
equilibrium. No GX channel is edited independently. One positive and one
negative step per candidate are run at drives $(3,0.9)$ and $(4.5,0.9)$: **24
standard GX simulations**. Controls are a zero-step continuation, an
orthogonal/constrained direction, equal boundary-coefficient norms for plus and
minus steps, and a rerun of each original anchor. Six decisive cases receive
doubled spatial resolution and averaging time. A result must keep its sign and
change by no more than 20% at higher resolution, and exceed two combined
`Q_stds` standard errors.

The planning estimate is **32.5 Perlmutter node-hours**: 24 standard runs at
0.5 node-hour (12), six convergence runs at 2 node-hours (12), 2 node-hours for
VMEC searches, and 25% contingency. This is an allocation envelope, **not a
measured Perlmutter pilot**. If approved, one standard and one convergence case
must be benchmarked before the allocation is requested; the estimate is revised
from those timings rather than silently treated as measured.

The expected observational sign is positive for both directions, but the
localized-peak unstable AIPW reversal means its sign is explicitly uncertain.
The decisive outcome is the paired GX response under the equilibrium-consistent
continuation, not agreement with the observational expectation.

## Contradictory cases and negative results

- [Contradictory cases](S13_artifacts/contradictory_cases.csv) gives five
  supporting and five contradicting matched pairs for each candidate. The
  negative cases were selected by the same signed population rule as the
  supporting cases and are given equal space.
- No candidate passes the 0.5 post-match balance threshold or the 0.8 overlap
  threshold; none is intervention-ready.
- The localized $f_Q$ peak changes from a positive both-unstable matched
  contrast to a negative adjusted unstable-tail contrast.
- $f_{\rm stab}$ and bad-curvature/compression have unresolved adjusted all-row
  native contrasts.
- The localized peak and bad-curvature/compression do not give resolved MSE
  gains beyond the full paper-selected fixed-panel baseline.
- The geodesic candidate does not dominate the raw zonal-flow association;
  $f_{\rm stab}$ is stronger.
- Fixed near-floor evidence rests on only 23 rows and 3–7 matched pairs, so no
  stable-regime physical mechanism is claimed.
- AIPW estimates for secondary outcomes can be very large where overlap is
  poor. They are sensitivity warnings, not headline effects.
- These equilibria appeared in network training. The physical GX associations
  are still genuine observations, but prior network selection and the panel do
  not establish generalization to new equilibrium families.

## Acceptance criteria

| PLAN criterion | Verdict | Number or artifact |
| --- | --- | --- |
| “claims are graded as model-mechanistic, observational-physical, or intervention-ready” | **Pass.** | [candidate_ranking.csv](S13_artifacts/candidate_ranking.csv) grades all four candidates `observational-physical`; no candidate is `intervention-ready`. Earlier network evidence remains model-mechanistic and the prospective intervention is separately marked proposal-only. |
| “confounding and invalid perturbations remain visible” | **Pass.** | Post-match maximum imbalance is 1.068–4.151 against a 0.5 gate; AIPW overlap is 0.228–0.478 against 0.8; every natural comparison is tagged `observed-comparison`, every row has `causal_claim_permitted=False`, and `summary.json` records `invalid_perturbations_used=false`. The GX spec requires recomputed VMEC equilibria and is not executed. |

The MVD is complete: task 1 covers fixed-gradient $Q$, $Q(z)$, `Q_stds`, zonal
magnitude, matching, and AIPW sensitivity; task 3 covers residuals beyond both
$f_Q$ and the full paper-selected feature set. Tasks 2 and 4 are also delivered
through the pair table, imbalance diagnostics, contradictory cases, and
prospective GX specification.

## Verification and mutation testing

Tests were written before implementation. The initial focused run failed all
five scientific paths at explicit `NotImplementedError` stubs. The final suite
has an analytic cyclic feature with a known native-output effect, a permuted
null feature, robust-rescaling invariance, sibling-equilibrium grouping,
deterministic grouped uncertainty, an analytic adjusted high/low contrast, GX
proposal budget arithmetic, and exact registered-artifact checks.

Three deliberate mutations turned the focused suite red and were reverted:

1. assigning folds by row instead of `equilibrium_files` split sibling tubes
   between train and test and failed the explicit group-disjoint assertion;
2. dropping IQR scaling from nuisance matching changed the selected pairs after
   a 1,000-fold unit rescaling and failed the robust-distance invariance test;
3. reversing the signed AIPW high-minus-low estimate changed the analytic
   +1.25 effect to -1.25 and failed the signed-effect test.

The first pilot stopped before artifact generation because a summed
15-dimensional distance made the 2-IQR caliper unintentionally tighten with
dimension. The estimator now uses root-mean-square IQR distance per nuisance
coordinate, matching the registered unit; the unchanged 96-row pilot then
completed. A second pilot failure exposed use of a nonexistent convenience
field on the upstream `DistillationResult`; the runner now computes the signed
residual explicitly as target minus prediction, and the pilot was repeated.

The first two full `make check` runs reached 309 passes but failed S07's
bitwise-repeat assertion by $2.8\times10^{-17}$ and $6.9\times10^{-18}$ under
macOS Accelerate FFT. The seeded null now quantizes only its final maxima at
$10^{-15}$ (maximum change $5\times10^{-16}$, far below any scientific or
bootstrap resolution), not a relaxed test tolerance. The exact assertion passed
twice consecutively and the final full check passed **310 tests**.

## Artifacts

- [fixed_associations.csv](S13_artifacts/fixed_associations.csv): 72 candidate ×
  outcome × regime correlations with grouped intervals.
- [matched_pairs.csv](S13_artifacts/matched_pairs.csv): 800 equilibrium-disjoint
  natural pairs with exposure, distance, balance, strata, and signed outcomes.
- [matched_effects.csv](S13_artifacts/matched_effects.csv): 72 matched physical
  effects and intervals.
- [doubly_robust_sensitivity.csv](S13_artifacts/doubly_robust_sensitivity.csv):
  32 AIPW sensitivity rows with overlap disclosed.
- [residual_validation.csv](S13_artifacts/residual_validation.csv): 42 fixed/
  varied, baseline/candidate, regime-specific cross-fitted results.
- [candidate_ranking.csv](S13_artifacts/candidate_ranking.csv) and
  [contradictory_cases.csv](S13_artifacts/contradictory_cases.csv): claim grades,
  ranking evidence, and balanced natural contradictions.
- [gx_experiment_spec.json](S13_artifacts/gx_experiment_spec.json): prospective
  equilibrium and GX protocol, controls, convergence rules, and node-hour
  arithmetic.
- [natural_experiment_atlas.png](S13_artifacts/natural_experiment_atlas.png),
  [summary.json](S13_artifacts/summary.json), and
  [manifest.json](S13_artifacts/manifest.json): visual summary, registered
  headlines, and provenance.

## Reviewer reproduction

### Recomputable on the slice

All 1,000 parent row IDs are in `tests/data/review_slice.h5`. Translate them
with `load_review_slice_index().slice_rows(parent_rows)` before loading.

- Recompute all 17 S12-v1 invariant features from the mapped geometry and S01
  channel IQRs; fixed drives must be exactly $(3,0.9)$.
- Recompute all fixed-panel correlations, the 185–215 deterministic matches,
  balance diagnostics, paired outcome differences, AIPW results, and fixed/
  varied EBM residual fits. The slice contains `Q_avgs`, `Q_avgs_vs_z`,
  `Q_stds`, `zonal_phi2_amplitudes`, scalar nuisances, and equilibrium IDs for
  these rows.
- `tests/xai/test_physical_validation.py` pins the cyclic analytic signal, null
  feature, native estimand, whole-equilibrium folds/bootstrap, robust-distance
  matching, and adjusted estimator. The artifact suite pins the exact geodesic
  headline, localized-peak sign reversal, residual-baseline distinction, and
  claim gates.

### Checkable from committed artifacts alone

- Every headline above is a literal committed CSV or JSON field. Artifact tests
  cross-check the ranking against its source rows, pair uniqueness, 40 balanced
  contradictions, claim flags, budget arithmetic, and every manifest hash.
- The exact geodesic rows are matched **1.3168418645 [1.1499375294,
  1.5023806914]**, AIPW **0.7713531585 [0.5276320327, 1.0128633179]**, and
  paper-baseline MSE improvement **0.0188012346 [0.0061692239,
  0.0325162110]**.
- The localized unstable sign contradiction is fully committed: matched
  **+1.5441856155** versus adjusted **-1.0905590116**.

### Not checkable off the researcher's machine, and why

- Matching the external 678 MB HDF5 source bytes to the manifest SHA-256 cannot
  be done on GitHub. The exact 1,000-row review slice is the nearest proxy and
  contains every row and quantity used for the scientific headlines; agreement
  reproduces the analysis but not the absent source-file fingerprint.
- The 32.5-node-hour GX budget is prospective arithmetic, not a benchmark. No
  new equilibrium or GX output exists to check. The nearest executable proxy is
  the required one-standard/one-convergence pilot after researcher approval;
  agreement would replace the assumed 0.5/2.0 node-hour costs with measured
  values.

## Deferred

Nothing from S13 is deferred. New equilibrium generation and GX simulations are
not part of the step: they are deliberately held at PLAN's researcher decision
gate.
