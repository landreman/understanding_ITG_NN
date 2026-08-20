# S04 — Anatomy of the invariant bottleneck

## Result

The invariant bottleneck carries a compact but nontrivial geometry signal, while
the two scalar drives dominate the canonical head on the frozen varied-gradient
panel. Across the stored-validation top 10, bottleneck units contribute a median
**19.5%** of output variance (member range 18.8%–20.4%) and 31.2% of total mean
absolute Shapley magnitude. Mean-replacement ablation and Shapley unit rankings
agree strongly (median member Spearman **0.890**, range 0.822–0.966), so the
ranking is not an artifact of either one method. Nine members use sampled
Shapley, so the range endpoints carry the reported Monte Carlo and package-version
sensitivity; the median and grouped interval are the more stable summaries.

The bottlenecks strongly encode the paper's geometric feature family. Across all
100 members, median equilibrium-grouped out-of-fold linear/nonlinear decoder
$R^2$ is **0.882/0.891** for $\log f_Q$, **0.828/0.848** for
$\log\langle|\nabla x|\rangle$, and **0.785/0.800** for $f_{\rm stab}$. The
simple controls are less decodable: $\hat s$ 0.372/0.401, `nfp` 0.453/0.497,
and `aspect` 0.283/0.315. Label-permutation controls have median $R^2=-0.050$
and maximum 0.110 over all targets, members, and both decoder families.

Decodability is not use. For the top 10, removing each target's linear decoder
direction changes native output by median RMS **0.668** for $\log f_Q$, **0.403**
for $f_{\rm stab}$, and **0.295** for
$\log\langle|\nabla x|\rangle$, versus **0.150** for a random direction. The
corresponding effects for `nfp` (0.087), $\hat s$ (0.090), and `aspect` (0.143)
use edits of different sizes, so the raw comparison is not magnitude matched.
After division by the RMS removed projection in standardized hidden space, the
effects are 0.479, 0.396, and 0.269 for the three paper features; `nfp` is 0.147,
$\hat s$ is 0.122, `aspect` is 0.171, and the random-direction median is 0.162.
Thus `aspect` is near rather than below the control, while the ordering and the
strong $f_Q$ result survive. These hidden interventions are deliberately
off-manifold diagnostics: they show what the head uses, not a valid plasma
intervention.

## Estimand and cohort

The estimand is each member's canonical S02 function
$\tilde f_m(X,g_T,g_n)=\mathrm{MLP}_m(\bar u_m(X),g_T,g_n)$ in native
$\max(\log Q,-2)$ units. All signed effects are computed per member before any
summary. The cohort is S01's frozen 1,000-row varied-gradient interpretation
panel, one row per `equilibrium_files`; 240 rows are stable or near the floor
($\max(\log Q,-2)\le-1.9$) and 760 are unstable. No fixed-gradient row and no
row was read from the `tests/data/review_slice.h5` file during development or
production. The independently constructed slice happens to contain copies of
all 1,000 registered panel rows for reviewer reproduction.

The production run is recorded verbatim in the committed
[manifest](S04_artifacts/manifest.json), copied from
`output/xai/S04/bottleneck-all100-panel1000-top10-shapley/manifest.json`, with
`run_id=bottleneck-all100-panel1000-top10-shapley`. It records checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`, dataset
SHA-256 `9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`,
the exact row and member IDs, output hashes, and hashes of the dirty-tree runner
and reusable library. The final registered pass took 132.1 s on CPU with Python
3.12.4, torch 2.4.1, numpy 1.26.4, h5py 3.11.0, and Captum 0.9.0.

## Methods and artifacts

`bottlenecks.h5` stores $\bar u_m$, stable unit IDs, predictions, rows, drives,
targets, and equilibrium IDs with explicit `(member, sample, unit)` axes for all
100 members. The top 10 contain 248 registered units, of which 229 are live on
this panel at the preregistered numerical threshold. The committed overview
tables and figures are in [S04_artifacts](S04_artifacts/summary.json).

S04 inherits the real-model cyclic invariance of `InvariantMember` from S02.
Its pre-run toy gate checks an analytic mean-pooling function that is invariant
by construction; it verifies the S04 attribution/intervention plumbing but is
not presented as a new panel-level invariance test.

Shapley uses the panel mean of every bottleneck unit and both scalar drives as a
fixed cohort-conditional reference. The 10-unit top member has 12 players and
was enumerated exactly over all 4,096 coalitions. The other nine top members
have 23–33 players and use Captum `ShapleyValueSampling` with 256 permutations;
16 deterministic permutation blocks provide per-sample standard errors. The
maximum mean feature SE on the overall stratum is 0.0282 native units (0.0341
when the separate stable and unstable summary rows are also considered; the
maximum individual-sample SE is 0.140), and the maximum efficiency error over
all top-member rows is
$8.35\times10^{-7}$. Per-sample signed values and SEs are in `shapley.h5`;
[shapley_global.csv](S04_artifacts/shapley_global.csv) reports signed mean,
mean absolute, RMS, and the signed variance decomposition
$\operatorname{Cov}(\phi_j,\tilde f)/\operatorname{Var}(\tilde f)$, plus the
`captum_shapley_value_sampling` or `exact_enumeration` code path and the
`deliberately_off_manifold_diagnostic` validity tag in every row. The coalition
and PDP/ICE HDF5 writers carry the same tag because both edit hidden states.

Every member received zero, mean-replacement, and fixed-seed cohort-resampling
interventions for every unit and unit pair. Each row records signed output
change, error change, and the discrete pair interaction. The large top-10
member/sample arrays are in `interventions_top10.h5`; the all-member graph source
is the production `intervention_summary.csv` in the hashed run directory. Its
small all-100 summaries are committed in
[summary.json](S04_artifacts/summary.json): the strongest-unit RMS medians for
zero/mean/resample edits are 0.693/0.406/0.582, with median strongest-unit MSE
changes 0.487/0.168/0.346; the corresponding all-unit median MSE changes are
0.00958/0.00598/0.01267. All primary intervention artifacts carry
`deliberately_off_manifold_diagnostic` in code and artifacts. The
[interaction graph](S04_artifacts/interaction_graph.png) shows the top member's
20 strongest mean-replacement edges.

Mean pair-interaction RMS is usually small: median 0.0112, 90th percentile
0.0435, and 99th percentile 0.102 native units across all 30,353 all-member unit
pairs. A sparse tail reaches 0.282, so an additive description is useful but not
exact. Across all 100 members, the strongest unit's mean-replacement RMS has
median 0.406 and range 0.237–0.801. For the exact top member, unit `u001` is
dominant by both methods: mean absolute Shapley 0.418 and mean-replacement RMS
0.634 (95% equilibrium bootstrap interval 0.574–0.696); `u008` is second at
0.170 and 0.273 (0.255–0.290). See
[rank_comparison.csv](S04_artifacts/rank_comparison.csv) and
[grouped_uncertainty.csv](S04_artifacts/grouped_uncertainty.csv).

The [PDP/ICE atlas](S04_artifacts/pdp_ice_atlas.png) covers every live top-10
unit at seven panel quantiles. `pdp_ice_atlas.h5` additionally contains all
sample ICE curves and each unit's 7-by-7 dependence surface against $a/L_T$ and
$a/L_n$, separately summarized for all, stable/near-floor, and unstable rows.
The atlas shows strong ICE heterogeneity even where the global PDP is monotone;
S05 should therefore name units from their local densities, not from a single
global head slope.

Decoders use five folds formed from whole `equilibrium_files`. The linear model
is ridge regression; the small nonlinear model adds 32 fixed ReLU features and
fits a ridge output. Near-dead columns active in at most 1% of a training fold
are excluded from that fold only. This reuses S02's 1% constant on a different
axis—training-fold rows here rather than positions within a sample there; raw
activations remain in all primary artifacts. The known invariant reconstructed
from geometry agrees with the registered field to maximum $2.86\times10^{-7}$
in log units. Complete all-member scores and permutation controls are in
[decodability.csv](S04_artifacts/decodability.csv) and the
[decodability matrix](S04_artifacts/decodability_matrix.png).

Finally, held-out decoding of each top member's canonical $\tilde f_m$ head from
$(\bar u_m,g_T,g_n)$ gives median $R^2=0.799$ for the linear decoder and 0.811
for the nonlinear decoder. The nonlinear increment is only 0.0116 (member range
0.00195–0.0540), so most head behavior is linear over the observed panel even
though the trained head contains two ReLU layers. Full values are in
[head_fidelity.csv](S04_artifacts/head_fidelity.csv), while
[encoded_vs_used.csv](S04_artifacts/encoded_vs_used.csv) keeps encoded and used
statistics in separate columns. This deliberately targets $\tilde f_m$, not the
original $f_m$, to isolate head nonlinearity from the S02 canonicalization
residual; S02 reports that $\tilde f-f$ changed ensemble residual standard
deviation by only $1.68\times10^{-4}$ on the full varied cohort.

## Stable/near-floor versus unstable rows

The regimes are never pooled in the underlying summaries. Geometry carries a
median 22.2% of output variance on unstable rows and 34.3% of total mean
absolute Shapley magnitude. On stable/near-floor rows its signed variance share
is unstable as a statistic: the median is -0.195 with member range -0.415 to
0.104 because the clipped output has very little variance and several geometry
terms act as suppressors. The corresponding mean-absolute fraction, 22.9%, is
descriptive but does not repair the compressed denominator.

Direction-removal effects are consistently larger on unstable rows. Median
stable/unstable RMS is 0.328/0.742 for $\log f_Q$, 0.153/0.454 for
$f_{\rm stab}$, and 0.135/0.330 for
$\log\langle|\nabla x|\rangle$. Linear/nonlinear head-decoder $R^2$ on the
stable stratum is -13.4/-12.3 for the same denominator reason and is retained as
a negative result, not interpreted as a head-fidelity effect; unstable values
are 0.753/0.767.

## Uncertainty

Shapley sampling uncertainty is reported per member, sample, and feature for the
nine sampled members; the exact member has zero sampling error. Unit-level
Shapley and mean-ablation intervals use 1,000 resamples of whole
`equilibrium_files`. The panel has one selected tube per equilibrium, but the
implementation and synthetic leakage test operate on repeated groups so the
resampling unit is not inferred from this accidental one-to-one panel layout.
The top-10 median Shapley/ablation Spearman is 0.890 with grouped-bootstrap 95%
interval 0.857–0.897. For $\log f_Q$, median nonlinear decodability is 0.892
(0.875–0.906) and median direction-removal RMS is 0.668 (0.611–0.722); for
`nfp` the corresponding values are 0.516 (0.473–0.560) and 0.087
(0.082–0.094). Member ranges remain descriptive member spread, not confidence
intervals.

The decoder $R^2$ bootstrap reweights the fixed out-of-fold predictions for
evaluation; it covers equilibrium sampling variation in the evaluation panel,
not variability from refitting the decoder inside every bootstrap replicate.

No multiplicity correction is applied to the 7,200 decoder summary rows, 509
grouped-bootstrap intervals, or 30,353 mean-mode overall pair interactions.
The maximum label-permutation $R^2=0.110$ is retained as an empirical
max-statistic diagnostic across the decoder family; individual pair and decoder
rows are exploratory rather than individually significant discoveries.

## Failed checks and corrections

The first 64-row pilot exposed under-regularization in the nonlinear decoder;
these are the first 64 rows of the final registered panel, not a separate
tuning cohort. The ridge penalty was raised from 0.001 to 1.0 solely because the
pilot control was non-finite/non-null, not to optimize decoder $R^2$; it made
the control finite and null without changing the cohort or target. The first
all-member run
then exposed two non-top members whose near-dead units caused fold-wise
standardization to extrapolate to decoder $R^2\approx-200$. The S02 1%-active
training-fold guard fixed this without consulting labels, dropping rows, or
changing any primary activation; after correction, all-member nonlinear minima
are 0.585 for known-invariant decoding and 0.861 for $\log f_Q$ (the linear
$\log f_Q$ minimum is 0.847).
Both failures and their regression test are retained.

Three deliberate post-run mutations turned the suite red and were reverted:

1. splitting individual rows instead of whole `equilibrium_files` failed the
   repeated-equilibrium isolation test;
2. exponentiating the head output before Shapley failed the analytic native-unit
   decomposition; and
3. treating near-dead fold features as supported reproduced a prediction near
   994 in the synthetic decoder control and failed its boundedness test.

## Negative results

- The head is drive-dominated on this panel; geometry contributes only about a
  fifth of output variance.
- Most head behavior is already captured linearly; the small nonlinear decoder
  adds only 0.0116 median $R^2$.
- `nfp` and $\hat s$ are decodable but their magnitude-normalized targeted
  hidden-direction interventions are below random directions; `aspect` is near
  the control. Decodability alone would have overclaimed their importance.
- Most pair interactions are small, although the nonzero tail prevents calling
  the head exactly additive.
- Stable-stratum variance-normalized quantities can be negative or enormous
  because clipping compresses the denominator; they are retained but not used
  for mechanism claims.

## Interpretation limits

All results describe the trained networks on an interpolation panel whose
equilibria appeared in training; they do not establish equilibrium-level
generalization. Hidden-unit edits do not correspond to realizable stellarator
changes and therefore cannot establish plasma causality. Decoder directions can
mix several correlated concepts, especially $f_Q$,
$\log\langle|\nabla x|\rangle$, and $f_{\rm stab}$; direction-removal effects
are evidence of head use, not unique concept attribution. Wide-member Shapley
rankings retain finite Monte Carlo error, and S05 must connect important unit IDs
to local equivariant densities before any geometric motif is named.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
.venv-xai/bin/python scripts/xai_s04_bottleneck.py --pilot --no-publish
.venv-xai/bin/python scripts/xai_s04_bottleneck.py
conda run -n 20240629-01-ML make check
git diff --check
```

## Acceptance criteria

| PLAN criterion | Verdict | Number or artifact |
| --- | --- | --- |
| “Shapley values are exact (or carry reported sampling error)” | Pass | One 12-player member uses `exact_enumeration`; nine wide members use 256-permutation `captum_shapley_value_sampling` with per-sample SEs. Maximum overall-stratum mean feature SE 0.0282; `shapley.h5` and the method-tagged [shapley_global.csv](S04_artifacts/shapley_global.csv). |
| “ablation and Shapley rankings are compared explicitly” | Pass | Median member Spearman 0.890, grouped-bootstrap 95% CI 0.857–0.897 (member range 0.822–0.966); [rank_comparison.csv](S04_artifacts/rank_comparison.csv) and [grouped_uncertainty.csv](S04_artifacts/grouped_uncertainty.csv). |
| “encoded (decodable) and used (changes the output) are separate columns” | Pass | Grouped-CV decoder $R^2$ and direction-removal RMS are separate in [encoded_vs_used.csv](S04_artifacts/encoded_vs_used.csv); $\log f_Q$ is 0.892 (0.875–0.906) encoded and 0.668 (0.611–0.722) used, while `nfp` is 0.516 (0.473–0.560) encoded and 0.087 (0.082–0.094) used. |
| “random-direction controls are included” | Pass | 16 directions per top member are retained at member/sample resolution in `interventions_top10.h5`; raw median control RMS is 0.150 and magnitude-normalized median is 0.162. [encoded_vs_used.csv](S04_artifacts/encoded_vs_used.csv) records every concept and control edit magnitude so the unmatched raw comparison is explicit. |

## Reviewer reproduction

**Recomputable on the slice.** Load the parent row IDs from S01's varied
`registered_rows`, translate them with
`load_review_slice_index().slice_rows(registered_rows)`, and read those 1,000
rows from `tests/data/review_slice.h5`. They are exactly the S04 panel, so the
240/760 strata, $2.86\times10^{-7}$ invariant reconstruction error, top-member
exact Shapley and ablation values, sampled-Shapley method/efficiency/SE checks,
all-100 decoder scores, grouped intervals,
edit magnitudes, intervention-mode summaries, and direction controls can be
recomputed. `shapley_global.csv` names `exact_enumeration` for the top member and
`captum_shapley_value_sampling` for the other nine; the review environment now
has Captum, scipy, and pandas. The columns to compare are in
[shapley_global.csv](S04_artifacts/shapley_global.csv),
[rank_comparison.csv](S04_artifacts/rank_comparison.csv),
[decodability.csv](S04_artifacts/decodability.csv), and
[encoded_vs_used.csv](S04_artifacts/encoded_vs_used.csv).

**Checkable from committed artifacts alone.** The acceptance values and their
grouped intervals are in [summary.json](S04_artifacts/summary.json) and
[grouped_uncertainty.csv](S04_artifacts/grouped_uncertainty.csv). The all-100
zero/mean/resample RMS and MSE-change summaries are also in `summary.json`.
The committed [manifest](S04_artifacts/manifest.json) is byte-identical to the
registered run manifest and records `run_id`, command, config, source hashes,
packages, 100 member IDs, 1,000 parent row IDs, input fingerprints, wall time,
and 16 output hashes; its recorded commit is an ancestor of this PR head.

**Not checkable off the researcher's machine, and why.** The checkout does not
contain the 678 MB source HDF5 file or the large per-sample `bottlenecks.h5`,
`shapley.h5`, `interventions_top10.h5`, `pdp_ice_atlas.h5`, and full 300,018-row
`intervention_summary.csv` from the run directory, so their bytes cannot be
matched to the manifest's output hashes. The nearest proxy is to rerun the same
deterministic calculations on the mapped 1,000 slice rows and compare the
committed small tables; agreement checks every scientific headline but not the
archived large-file hashes. The slice's `source_sha256` provenance equals the
manifest dataset SHA-256, which checks dataset identity without exposing the
source bytes. No S04 scientific headline depends on the remaining 9,785-row
reference cohort or other rows outside the slice.

## Deferred

Nothing. Tasks 1–3 and 6 were completed for all 100 members; Shapley, PDP/ICE,
drive surfaces, and head-fidelity decoders were completed for the registered top
10.
