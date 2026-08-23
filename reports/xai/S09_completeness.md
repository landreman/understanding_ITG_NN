# S09 — Concept completeness and geometry–gradient interactions

## Result

The candidate vocabulary predicts the three canonical member outputs with high
held-out fidelity: the all-candidate decoder reaches member $R^2$ values
**0.9080, 0.9087, and 0.9174** (median **0.9087**) on equilibria excluded from
its fit. This is **90.9% held-out fidelity**; the exact canonical bottleneck
head is a ceiling of 1 by construction, not a measured denominator. The paper baseline
$\{a/L_T,a/L_n,f_Q\}$ already reaches median $R^2=0.8231$; the all-candidate
gain is **0.0849–0.0953** across members (median **0.08954**), and every direct
paired equilibrium-bootstrap 95% interval excludes zero.

That 0.08954 is not a pure “new learned geometry” number. The last family adds
local $Q(z)$ concentration and zonal magnitude, which are observed GX
diagnostics rather than network inputs. The geometry-only set through
compression, curvature, parallel scale, co-location, $f_{\rm stab}$, and
$\log\langle|\nabla x|\rangle$ gains **0.0117, 0.0221, and 0.0193** over the
paper baseline. Its direct 95% intervals are respectively
**[-0.0044, 0.0260]**, **[0.0111, 0.0332]**, and **[0.0085, 0.0300]**. Thus two
members resolve a modest geometry-only increment and one does not. The larger
all-candidate gain shows predictive completeness of the vocabulary, not that
the network causally uses zonal flow; S08 independently rejected all zonal-use
claims.

Drive interaction is nevertheless reproducible on the varied-gradient panel.
All **48/48** concept × drive × drive-bin signs agree across all three members
on the pooled cohort, and **48/48** agree on unstable rows. Stable/near-floor
rows are weaker: **36/48** signs agree. The largest pooled point-estimate change
is bad curvature versus $a/L_T$: its median observed directional slope changes
from **-0.858** in the low-drive bin to **+0.356** in the high-drive bin in all
three members. However, neither endpoint is individually resolved in any
member: the three low-bin 95% intervals span **[-2.569,+0.931]** to
**[-2.355,+0.739]**, and the high-bin intervals span **[-0.206,+0.914]** to
**[-0.111,+0.808]**. Both endpoints straddle zero in every member, and this
step did not compute an interval for their high-minus-low difference. The
apparent reversal is therefore a hypothesis, not an established result.
Geodesic-curvature slope rises from **0.220 to 3.553** across
$a/L_T$ bins, while co-location declines from **1.270 to 0.279** but stays
positive. These are observed comparisons and decoder diagnostics, not valid
single-channel plasma interventions.

## Estimand and cohort

The estimand is each stored-validation top-three member's S02 canonical exactly
shift-invariant function

$$\tilde f_m(X,g_T,g_n)=\operatorname{MLP}_m(\bar u_m(X),g_T,g_n)$$

in native $\max(\log Q,-2)$ units. Signed predictions and residuals are kept by
member and row; no result uses $Q$ or $\exp(\tilde f_m)$. The cohort is S01's
frozen 1,000-row varied-gradient panel, one tube from each of 1,000
`equilibrium_files`: **240 stable/near-floor** and **760 unstable** rows. No
fixed-gradient row and no row loaded from `tests/data/review_slice.h5` was used
for development or production.

The registered run is `completeness-top3-panel1000`. The committed
[manifest](S09_artifacts/manifest.json) records production compute time
**2,423.63 s (40.39 min)** and **585.28 s** for the final hash-validated
post-processing pass. It records dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`
and checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.

## Methods

### Nested completeness decoders

Five outer folds and three inner folds assign complete `equilibrium_files`,
never individual rows. Inner folds select a ridge penalty for a deliberately
small decoder containing main effects, squares, and only the registered
concept-by-$a/L_T$ and concept-by-$a/L_n$ products. Outer-fold predictions are
the sole source of reported fidelity. The nested sets are:

1. paper baseline: $a/L_T$, $a/L_n$, and $\log f_Q$;
2. paper geometry: add $f_{\rm stab}$ and
   $\log\langle|\nabla x|\rangle$;
3. spatial geometry: add compression, bad curvature, geodesic curvature,
   robustly scaled parallel Fourier scale, and cross-channel co-location;
4. all candidates: add local $Q(z)$ concentration and zonal magnitude; and
5. the full invariant bottleneck plus drives.

The simple full-bottleneck decoder reaches median $R^2=0.8695$, below the
lower-dimensional all-candidate decoder. This is a decoder-capacity result, not
an information-ordering result: the trained head consumes the bottleneck and
therefore reproduces its own output exactly. The artifact keeps both
`full_bottleneck_simple_decoder` and `full_bottleneck_exact_head`; completeness
is not formed by dividing by the exact head. The all-candidate/simple-bottleneck
decoder ratio is **1.045**, with the capacity caveat above; the primary quantity
is plain held-out $R^2$. [Completeness rows](S09_artifacts/completeness.csv)
carry out-of-fold $R^2$, MSE, bias, residual standard deviation, incremental
gain, direct gain over baseline, grouped intervals, and selection stability.
[Per-row residuals](S09_artifacts/concept_residuals.csv) make every summary
independently recomputable.

The direct gain intervals resample all 1,000 `equilibrium_files` 5,000 times and
compare each candidate and baseline on the same draw. Bootstrap selection
stability is the fraction of draws with positive gain. This evaluates fixed
out-of-fold predictions; it does not refit the decoder inside every resample.

### Drive interactions

Within low, middle, and high quantile bins of each drive, observed concept
slopes are estimated separately for all, stable/near-floor, and unstable rows.
Uncertainty again resamples `equilibrium_files`. The
[member-level effects](S09_artifacts/interaction_effects.csv) and
[cross-member summary](S09_artifacts/interaction_summary.csv) retain signs and
regimes.

The registered substitute for selected integrated-Hessian terms is a mixed
derivative of the fitted concept decoder (a diagnostic that measures how a
concept's local decoder effect changes with a drive). It uses four held-out decoder evaluations
around each observed row. The declared quadratic decoder makes that four-point
mixed difference exact for the decoder. Contributions are integrated from the
panel-median background and stored in
[integrated_hessian_terms.csv](S09_artifacts/integrated_hessian_terms.csv).
Each row reports the minimum, maximum, and sign agreement across the five
outer-fold coefficients; the row-bootstrap interval is retained only as
evaluation-panel variation and is not used for a mechanism claim. Every row is
tagged `observed-comparison`: correlated concepts and drive
stratification prevent a physical-causal reading.

This is not an entry of the network's 674-input Hessian, and the directional
slopes are observed OLS slopes rather than paired grouped finite differences.
Those PLAN task-3 calculations are explicitly deferred below. No multiplicity
correction is applied across the exploratory interaction table; cross-member
sign replication is descriptive, not a discovery threshold.

## Stable/near-floor versus unstable behavior

Stable-row $R^2$ is not interpretable because member outputs have a compressed
denominator. It is retained (median -11.03 for the baseline and -4.19 for all
candidates) but not used as evidence. MSE gives the useful comparison: the
median stable-row MSE drops from **0.8482** to **0.3847**, while unstable-row
MSE drops from **0.6907** to **0.3708**. On unstable rows, ordinary $R^2$
improves from median **0.7735** to **0.8780**.

Interactions are less reproducible near the floor: 36/48 stable-stratum signs
agree in all three members, versus 48/48 on unstable rows. The mixed terms also
change. For example, median $a/L_T \times
\log\langle|\nabla x|\rangle$ mixed derivative is **-0.189** near the floor but
**+0.319** on unstable rows. However, the stable fold ranges are
**[-0.400,+0.071]**, **[-0.321,+0.045]**, and **[-0.238,-0.064]**: two members
contain an opposite-sign fold. This is a tentative regime contrast, not a
resolved reversal.

## Negative and contradictory results

- Adding $f_{\rm stab}$ and $\log\langle|\nabla x|\rangle$ immediately after
  the paper baseline changes $R^2$ by -0.0059, -0.0023, and +0.0006; every
  interval crosses zero. Those concepts may be redundant with $f_Q$ in this
  decoder even though S08 finds hidden-layer use.
- The geometry-only gain is modest and unresolved for one of three members.
- The strongest increment comes from target-side GX diagnostics. It cannot be
  summarized as geometry the network newly learned, and the zonal negative
  control remains negative evidence.
- A simple decoder can fit the low-dimensional candidate set better than the
  wider sufficient bottleneck. Decoder fidelity is not an information
  ordering; the exact trained head supplies the ceiling.
- Stable-row $R^2$ is strongly negative and stable interaction signs are less
  reproducible. MSE and the unstable stratum carry the meaningful fidelity
  conclusions.

## Failed checks and corrections

- The tests first failed at explicit `NotImplementedError` stubs for all S09
  paths.
- The first 96-row pilot exposed severe overfitting from unrestricted all-pairs
  quadratic features. The decoder was restricted, before production, to main
  effects, squares, and the two registered drive-interaction families; the
  analytic interaction test still recovers its exact coefficient.
- Production validation found the low-dimensional candidate decoder above the
  simple full-bottleneck decoder. Reporting that ratio above one would have
  confused decoder capacity with completeness. The simple comparator is kept,
  and the exact trained bottleneck head is now the bounded ceiling.
- The first post-processing table pooled interaction regimes. It was discarded
  and recomputed separately for all, stable/near-floor, and unstable rows.

## Acceptance criteria

| PLAN criterion | Verdict and evidence |
| --- | --- |
| “high fidelity is demonstrated on held-out equilibria” | **Pass.** All-candidate member $R^2=0.9080$–0.9174 under nested equilibrium-grouped folds; median held-out fidelity is 0.9087 with an exact-head ceiling of 1 by construction. |
| “added complexity has an uncertainty-qualified gain” | **Pass with an important qualification.** All-candidate gain over the paper baseline is 0.0849 [0.0705, 0.0998], 0.0953 [0.0800, 0.1108], and 0.0895 [0.0764, 0.1040]. Geometry-only gain is 0.0117 [-0.0044, 0.0260], 0.0221 [0.0111, 0.0332], and 0.0193 [0.0085, 0.0300]; the larger gain includes target-side diagnostics. |
| “interaction conclusions reproduce across members and do not rest on the fixed-gradient set alone” | **Qualified pass.** The observed-slope calculation uses varied rows only. Pooled and unstable point-estimate signs reproduce in 48/48 cells across all three members; stable/near-floor signs reproduce in 36/48, but only 19/144 stable-row slopes have 95% intervals excluding zero. The bad-curvature $a/L_T$ point estimates reverse in every member, but both endpoint intervals cross zero and this step did not compute an interval for their difference, so the reversal is not established. True network-input mixed derivatives and paired grouped finite differences are deferred. |

## Mutation testing

The following deliberate mutations turned the focused suite red and were
reverted:

1. assigning folds by rows instead of `equilibrium_files` broke the repeated-
   equilibrium disjointness assertion;
2. exponentiating the native output reduced the analytic native-target fidelity
   and failed the native-output assertion; and
3. dropping the drive-by-concept product changed the known mixed derivative
   from 3 to approximately zero and failed the integrated-Hessian control;
4. fitting the outer decoder on all rows, including its held-out fold, failed
   the explicit manual out-of-fold prediction comparison; and
5. replacing the five fold-specific mixed derivatives with their repeated
   panel mean failed the deliberately sign-inconsistent fold-spread test; and
6. resampling individual rows instead of `equilibrium_files` failed the row-
   duplication invariance controls for the completeness, direct-gain,
   directional-effect, and mixed-term bootstraps; and
7. sampling every equilibrium without replacement collapsed all bootstrap
   intervals and failed the material-width controls for all four paths; and
8. changing the reported quantiles from 95% to 90%, or doubling the number of
   equilibria per draw, failed exact seeded-bootstrap reconstruction controls.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s09-pilot XDG_CACHE_HOME=/private/tmp/cache-s09-pilot \
  .venv-xai/bin/python scripts/xai_s09_completeness.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s09-prod XDG_CACHE_HOME=/private/tmp/cache-s09-prod \
  .venv-xai/bin/python scripts/xai_s09_completeness.py
MPLCONFIGDIR=/private/tmp/mpl-s09-resume XDG_CACHE_HOME=/private/tmp/cache-s09-resume \
  .venv-xai/bin/python scripts/xai_s09_completeness.py --resume
source .venv-xai/bin/activate && make check
```

## Reviewer reproduction

**Recomputable on the slice.** All 1,000 parent row IDs are S01 panel rows in
`tests/data/review_slice.h5`. Translate them with
`load_review_slice_index().slice_rows()` before loading. The reviewer can
recompute every concept score, canonical member prediction, grouped fold,
out-of-fold decoder prediction, regime metric, directional slope, and selected
mixed term for the three named members. The practical proxy is the 96-row pilot;
agreement on folds, signed interaction controls, axes, and artifact columns
checks the wiring, while the full slice reproduces the production numbers.
Because the frozen panel deliberately contains one row per equilibrium,
equilibrium-grouped and row-grouped folds coincide for this production cohort;
the repeated-group synthetic test is what exercises the grouping protection.

**Checkable from committed artifacts alone.** Every headline above recomputes
from `completeness.csv`, `concept_residuals.csv`, `interaction_summary.csv`, and
`integrated_hessian_terms.csv`. Artifact tests recompute median completeness,
direct-gain verdicts, regimes, and all seven manifest hashes. The committed
manifest records the exact row/member IDs, config, source hashes, input
fingerprints, and measured production compute time.

**Not checkable off the researcher's machine, and why.** Exact bytewise
reproduction of the external-data run requires the 678 MB source HDF5 and about
40.4 CPU minutes. No scientific headline depends on a row outside the committed
review slice. A full slice rerun is the nearest numerically equivalent proxy;
agreement checks the science but cannot match the source-file bytes to its hash.

## Deferred

The MVD—nested concept completeness and the uncertainty-qualified gain over the
paper baseline—is complete. PLAN task 3's true selected network-input mixed
derivatives (geometry channel × drive) and paired grouped finite differences
are deferred. The delivered concept-decoder mixed derivatives and stratified
OLS slopes are clearly labeled substitutes; implementing a robustly scaled
input-Hessian path would exceed the one-session budget after the complete MVD.
