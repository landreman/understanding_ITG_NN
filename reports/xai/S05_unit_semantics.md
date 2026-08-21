# S05 — What each bottleneck unit measures

## Result

The important bottleneck units do not admit a simple one-name-per-unit
dictionary. Six of the 29 live units in the two preregistered members pass the
pre-production criteria for a named spatial alignment: absolute lagged
correlation at least 0.20 and recurrence of the winning concept in at least
50% of 500 equilibrium-grouped bootstrap samples. Five of these six align with
the paper's bad-curvature/flux-compression family and one with radial drift /
geodesic curvature. The remaining 23 units are explicitly unresolved.

The strongest result is in the stored-validation leader. Its dominant S04 unit,
`2864601_0.437:u001`, aligns negatively with the 25-point circular mean of the
paper's $f_Q$ integrand at lag +23: correlation **-0.369**, fixed-lag 95%
equilibrium-bootstrap interval **[-0.396, -0.344]**, and concept recurrence
**0.994**. Its second-ranked unit, `u008`, aligns with windowed radial drift /
geodesic curvature at lag -39: **-0.315 [-0.336, -0.292]**, recurrence **0.988**.
Three of this member's five most important S04 units pass the named-alignment
threshold.

Replication is weak. In member `2864601_0.371`, only S04 rank 5 (`u000`) among
the five most important units receives a supported name. The dominant unit
`u017` recurrently prefers radial drift at lag +45, but its correlation is only
**+0.192**, below the preregistered 0.20 threshold. Ranks 2–4 have still weaker
best alignments (**+0.079**, **-0.138**, and **-0.086**). Two less-important
units in this member, `u011` and `u018`, pass for bad-curvature/compression
concepts. Thus S05 supports a recurring concept family across members, but not
a replicated identity for the dominant unit.

Lag is essential rather than cosmetic. The six supported lags are
**+18, +23, -39, -15, -1, and +11** grid points. For example, the dominant top
member's like-for-like zero-lag Pearson correlation with its selected $f_Q$
trace is only **-0.006**, while the lagged Pearson correlation is -0.369 (its
zero-lag Spearman correlation is -0.018). All final receptive fields
wrap the entire 96-point domain (formal spans 180 and 330), so these large lags
are structurally possible. They also mean that a unit's activation maximum
should not be described as the same location as its matched physics feature.

The six observed correlations are **4.09–6.69 times** the 95th percentile of an
eight-draw row-permutation null that repeats the full selection over all 75
concepts and 96 lags. This is a selection-calibration diagnostic, not a
low-resolution permutation $p$-value.

Even supported alignments are incomplete descriptions. A tie-inclusive top-5%
mask selects an average **6.81–36.41** density positions because many ReLU
activations are tied at zero. Its concept overlap is **0.107–0.421**, but the
corresponding row-specific chance baselines are already **0.071–0.379**:
only **1.10–1.51-fold** enrichment. Density-local partial rank correlations
after residualizing the seven channel-magnitude ranks range from **-0.442 to
+0.237**. Four remain at absolute correlation at least 0.33; the sparse
replication unit `u011` falls to **+0.123**. These controls are evaluated at the
density position, not at the lagged concept's source position, so they do not
establish independence from source geometry.

| unit | selected concept | lagged $r$ | defined rows / 1000 | $r$ on defined rows | density-local partial rank $r$ | overlap / chance | selection-null q95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `.437:u000` | bad-curvature compression | -0.397 | 939 | -0.423 | -0.442 | 0.107 / 0.071 | 0.059 |
| `.437:u001` | $f_Q$ integrand | -0.369 | 969 | -0.381 | -0.435 | 0.126 / 0.091 | 0.066 |
| `.437:u008` | radial drift / geodesic curvature | -0.315 | 944 | -0.334 | -0.333 | 0.166 / 0.123 | 0.057 |
| `.371:u000` | bad-curvature compression | -0.386 | 923 | -0.418 | -0.382 | 0.161 / 0.116 | 0.094 |
| `.371:u011` | bad-curvature compression | +0.257 | 728 | +0.353 | +0.123 | 0.383 / 0.348 | 0.054 |
| `.371:u018` | $f_Q$ integrand | +0.266 | 730 | +0.365 | +0.237 | 0.421 / 0.379 | 0.055 |

The primary lagged statistic averages a row correlation of zero whenever the
density or trace is spatially flat. The table exposes both the number of rows
with a defined correlation and the defined-row-only mean; inactive rows are
therefore visible rather than silently dropped.

## Estimand and cohort

The primary object is each live unit's signed full-resolution equivariant
density

$$
\rho_{m,c}(z), \qquad
\tilde f_m(X,g_T,g_n)=\operatorname{MLP}_m
\left(\operatorname{mean}_z\rho_m(X),g_T,g_n\right),
$$

for S02's canonical exactly invariant member. Although the final ReLU makes the
observed densities nonnegative, their axes and unit IDs remain separate; no
member or unit is averaged away. The co-reported original object is the trained
member $f_m$ in native $\max(\log Q,-2)$ units, never $Q$ or
`exp(prediction)`.

The cohort is S01's frozen 1,000-row varied-gradient interpretation panel: one
row from each of 1,000 distinct `equilibrium_files`, with **240** stable or
near-floor rows and **760** unstable rows. S02 supplies the corrected,
spatial-origin-checked $\rho$ implementation; S04 supplies the importance
ranking. Production and development read the external source HDF5 file, not
`tests/data/review_slice.h5`.

The registered run is `unit-semantics-top2-panel1000` at code commit `f0de959`.
Its byte-for-byte published [manifest](S05_artifacts/manifest.json) records the
external dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`
and unchanged checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.
It used CPU, seed 20260821, Python 3.12.4, torch 2.4.1, NumPy 1.26.4, h5py
3.11.0, Captum 0.9.0, and took **196.6 s**. `git_tracked_dirty` is true because
the runner publishes the newly generated small tables before finalizing and
hashing the manifest; its source hashes and `git_commit` identify the clean
committed implementation used for the run. `git_dirty` also includes the
protected untracked `output/` tree.

## Methods

### Density census and support

Both MVD members were evaluated on all 1,000 rows. The top member has **9/10**
live units and the replication member **20/23**. For every live unit and for
overall, stable/near-floor, and unstable rows separately,
[unit_density_summary.csv](S05_artifacts/unit_density_summary.csv) records the
signed mean and distribution, active fraction, median active positions, and
median/90th-percentile half-maximum support. It also joins S04's Shapley and
mean-ablation rankings by stable unit ID.

The densities are usually broad: the median of unit-level median half-maximum
support is **96 positions**, although individual live units range from **20 to
96**. Active fractions range from **0.0122 to 0.9903**. The aligned
[density atlas](S05_artifacts/density_atlas.png) keeps sparse black rows rather
than dropping them.

### Physics concept vocabulary

[Concept traces](S05_artifacts/unit_concept_alignment.csv) use the paper's input
and unary-operation vocabulary:

- bad/good curvature from `Heaviside(cvdrift)`;
- $|\nabla x|$ and powers one through four;
- the $f_Q$ pointwise integrand and bad-curvature/compression family;
- signed and absolute radial drift / geodesic curvature;
- local shear
  $96[\operatorname{roll}_{-1}-\operatorname{roll}_{+1}]
  (\mathrm{gds21}/\mathrm{gds22})/2$;
- $B$, $1/B$, local minima, and extremum strength;
- robustly scaled multichannel local parallel roughness; and
- circular-window expected non-DC Fourier scale for $B$, compression, and
  curvature.

Each pointwise trace is accompanied by 9- and 25-point circular means, giving
**75** concepts. The multichannel derivative scale divides every channel by
S01's IQR robust scale before combination; no raw cross-channel gradient
magnitude is compared. These are `observed-comparison` traces computed from
real geometry, not interventions.

### Circular alignment and uncertainty

For each unit/concept/stratum, the table records:

1. within-tube zero-lag Pearson and Spearman rank correlations;
2. tie-inclusive top-5% overlap after applying the separately recorded best
   lag and sign, along with both mask sizes and the row-specific chance baseline;
3. the circular lag that maximizes the absolute mean within-tube correlation;
4. correlation at that lag, both with flat rows assigned zero and over only
   rows where correlation is defined; and
5. partial within-tube rank association after linearly residualizing the seven
   channel-magnitude ranks at the density position. Because the concept has
   been rolled, this last control is local to the density coordinate rather
   than to the lagged concept's source coordinate.

The point estimate chooses a lag on the full frozen panel. Bootstrap intervals
and concept recurrence then hold each concept's point-estimate lag fixed and
resample complete `equilibrium_files`; they therefore measure equilibrium
sampling stability conditional on the chosen lag, not the extra uncertainty of
reselecting lag on a new panel. The panel happens to have one row per
equilibrium, but the tested implementation accepts repeated rows and assigns
identical bootstrap multiplicity to every tube from the same equilibrium.

The complete member/unit/concept/stratum results are in
[unit_concept_alignment.csv](S05_artifacts/unit_concept_alignment.csv). The
compact [unit motif catalog](S05_artifacts/unit_motifs.csv) applies the two
thresholds that were fixed in the config before the pilot and production run.
A below-threshold winner remains visible as `best_observed_concept` but receives
`claimed_concept=none`.

For every supported winner, [unit_motifs.csv](S05_artifacts/unit_motifs.csv)
also reports eight sample-row permutations. Each draw breaks density/concept
pairing and records the maximum absolute mean Pearson correlation after
searching all **75 concepts × 96 lags**. The observed-to-null-q95 ratio is
4.09–6.69. Eight draws are enough to expose the scale of selection inflation,
but not to estimate a tail probability.

### Natural exemplars and motif clusters

Every unit has **16** maximal natural activations from 16 distinct
`equilibrium_files`. [natural_exemplars.csv](S05_artifacts/natural_exemplars.csv)
records source row, equilibrium, activation position, activation magnitude,
cluster, and the only alignment operation:
`joint_circular_roll_to_activation_center`. Wrapped patches use the exact formal
receptive-field coordinates, -70 through +109 for the top member and -135
through +194 for the replication member, with every source position recoverable
modulo 96. No input was optimized or synthesized.

A deterministic robustly scaled two-cluster summary was applied to each unit's
16 patches. [motif_clusters.csv](S05_artifacts/motif_clusters.csv) reports cluster
size, center statistic (coordinatewise median), dispersion statistic
(coordinatewise MAD), and separation. The result is mostly **15+1** or **14+2**:
the algorithm usually isolates an outlying natural equilibrium rather than two
balanced recurring motifs. The largest minor cluster is 3/16; those 13+3 cases
have center-separation/within-dispersion ratios only 1.08–3.12. This is negative
evidence for clean polysemantic subtypes, not proof that the units are
monosemantic. NMF/dictionary learning was not added after this diagnostic.

### Original member and shift controls

[native_function_comparison.csv](S05_artifacts/native_function_comparison.csv)
reports $\tilde f_m-f_m$ without exponentiating. Overall RMS differences are
**0.0977** and **0.1056** native units for the two members; stable-row RMS is
**0.0616/0.0705**, and unstable-row RMS **0.1066/0.1144**. For the six
S04 top-three units, original strided bottlenecks and `mean_z rho` remain very
highly correlated (**0.9923–0.9993**) but have nonzero RMS differences
**0.0285–0.1463**; see
[native_bottleneck_comparison.csv](S05_artifacts/native_bottleneck_comparison.csv).

On 32 real rows and shifts 1, 17, and 31, maximum density equivariance error is
**1.00e-5**, concept-trace equivariance error is **0.0**, and invariant-output
error is **1.19e-6**. The original members change by RMS **0.0901–0.1277** native
units under the same exact-symmetry shifts. All values and the
`exact-symmetry` tag are in
[shift_consistency.csv](S05_artifacts/shift_consistency.csv).

### First-layer kernels and transfer functions

The signed first-layer weights for every filter and all seven channels are in
[first_layer_kernels.csv](S05_artifacts/first_layer_kernels.csv). The
[transfer catalog](S05_artifacts/first_layer_transfer.csv) records DC amplitude,
peak frequency, spectral centroid, and kernel norms; the
[catalog figure](S05_artifacts/filter_transfer_catalog.png) shows the centroid
matrix. Spectral centroids are descriptive filter properties, not unit-level
semantic claims. For example, median `gds2` centroids are 23.54 and 24.53,
versus 14.21 and 20.73 for $B$, but later layers mix these filters and the raw
kernel amplitudes retain channel-dependent physical units.

## Stable/near-floor versus unstable rows

All density and alignment tables retain both strata. The supported alignments
usually keep their sign and nearly the same lag: stable versus unstable lagged
correlations are -0.333/-0.381 for top `u001`, -0.339/-0.307 for top `u008`,
-0.294/-0.416 for replication `u000`, +0.241/+0.262 for replication `u011`,
and +0.231/+0.277 for replication `u018`. Top `u000` is -0.372/-0.405.

The important caveat is recurrence within the smaller stable stratum. For top
`u000`, the same concept's exact recurrence is only **0.136** on 240 stable
equilibria versus **0.904** on 760 unstable equilibria, even though its
conditional correlation retains the sign. The overall motif name is therefore
not a claim of equally stable mechanism selection at the clipped-log floor.

## Failed checks, negative results, and limits

- The first test run failed all intended S05 paths with explicit
  `NotImplementedError`; after implementation all ten S05 tests pass.
- The initial pilot vocabulary represented parallel roughness but not an actual
  Fourier scale. A closed-form mode-3 test was added, the circular-window
  expected-$k_\parallel$ trace was implemented, and the pilot was rerun before
  production.
- Three deliberate mutations turned the focused suite red and were reverted:
  bootstrapping rows independently instead of whole `equilibrium_files` failed
  the repeated-group multiplicity check; forcing every concept to lag zero
  failed the analytic lag-7 recovery; and omitting S01's robust channel scales
  failed the dimensionless parallel-scale control.
- Automated review exposed that the original `_rank_last` allocated an
  `empty_like` array with non-contiguous strides, so reshaping it could rank a
  temporary copy and leave the returned partial-correlation ranks
  uninitialized. The production run was discarded. A contiguous-allocation
  regression now fails under that exact mutation, and the registered run and
  reports were regenerated. Two further review-driven mutations—excluding
  threshold ties and silently averaging only correlation-defined rows—also
  turned their new tests red and were reverted.
- Only six of 29 live units meet even the deliberately modest named-alignment
  thresholds, and only one of the second member's five most important units
  does. The dominant-unit name does not replicate.
- Tie-inclusive fixed-sparsity overlap enrichment is only 1.10–1.51 over its
  actual chance baseline even for supported alignments. Correlation plus
  recurrence is not purity.
- Sparse units have only 728–730 correlation-defined rows; the primary
  all-panel statistic deliberately gives inactive rows zero weight, while the
  defined-only values are published separately.
- The density-local partial control weakens `2864601_0.371:u011` from +0.257 to
  +0.123 and does not control geometry at the lagged concept source. It is a
  sensitivity check, not proof of a distinct learned feature.
- The selection null uses only eight permutations. Its 4.09–6.69 observed/q95
  ratios are reassuring calibration, not permutation-test significance.
- The second member's strongest unit misses the correlation threshold narrowly
  (+0.192) but is not rounded up or relabeled.
- Natural clustering mostly isolates one or two outliers; no balanced,
  well-separated polysemantic motifs were found.
- All final receptive fields are globally connected. A lagged match can reflect
  distributed computation, not a local convolutional detector.
- The concept family is finite and correlated. Recurrence among the 75 named
  traces does not establish uniqueness, and bootstrap intervals condition on
  the full-panel lag selection.
- These are internal activations on an interpolation panel whose equilibria
  appeared in training. Observed alignment explains the network, not plasma
  causality, and the sign of a density/concept correlation is not by itself the
  sign of the unit's effect on heat flux.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s05 .venv-xai/bin/python \
  scripts/xai_s05_unit_semantics.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s05 .venv-xai/bin/python \
  scripts/xai_s05_unit_semantics.py
MPLCONFIGDIR=/private/tmp/mpl-s05 conda run -n 20240629-01-ML make check
git diff --check
```

Large arrays are in
`output/xai/S05/unit-semantics-top2-panel1000/` and are manifest-hashed but
git-ignored. The production CLI supports `--config`, `--members`, `--rows`,
`--device`, `--batch-size`, `--seed`, `--resume`, and `--output-dir`.

## Acceptance criteria

| PLAN criterion | Verdict | Number or artifact |
| --- | --- | --- |
| “every claimed unit motif has multiple natural exemplars” | Pass | Every one of six claimed motifs has 16 exemplars from 16 distinct equilibria in [natural_exemplars.csv](S05_artifacts/natural_exemplars.csv); unresolved units make no motif claim. |
| “bootstrap recurrence over equilibria” | Pass | Claimed-motif recurrence is 0.950–1.000 over 500 `equilibrium_files` resamples; [unit_motifs.csv](S05_artifacts/unit_motifs.csv). |
| “receptive-field coordinates” | Pass | Exact formal coordinates -70:+109 and -135:+194, unique periodic support 96, in [receptive_fields.csv](S05_artifacts/receptive_fields.csv) and each exemplar row. |
| “shift-consistency” | Pass | Density maximum error 1.00e-5, concept error 0.0, invariant-output error 1.19e-6 under shifts 1/17/31; [shift_consistency.csv](S05_artifacts/shift_consistency.csv). |
| “lag is reported rather than assumed zero” | Pass | All 29 units × 75 concepts × 3 strata have an explicit best lag; supported lags are +18, +23, -39, -15, -1, +11 in [unit_concept_alignment.csv](S05_artifacts/unit_concept_alignment.csv). |
| “no optimized synthetic input is treated as physical evidence” | Pass | Every exemplar is an S01 panel row, `validity_tag=observed-comparison`, `synthetic_optimization_used=False`; no optimization code path exists. |

The requested deliverables are the
[density atlas](S05_artifacts/density_atlas.png),
[alignment tables](S05_artifacts/unit_concept_alignment.csv),
[filter/transfer catalog](S05_artifacts/first_layer_transfer.csv),
[motif clusters](S05_artifacts/motif_clusters.csv), and this report.

## Reviewer reproduction

**Recomputable on the slice.** All scientific headline rows are the 1,000 S01
panel parent row IDs, and every one is present in `tests/data/review_slice.h5`.
The reviewer must translate them rather than pass parent IDs directly:

```python
import json, numpy as np
from itg_nn.xai.review_slice import load_review_slice_index

cohorts = json.load(open("reports/xai/S01_artifacts/cohorts.json"))
parent = np.asarray(cohorts["interpretation_panel"]["varied_row_ids"])
slice_rows = load_review_slice_index().slice_rows(parent)
```

Loading `slice_rows` from the slice reproduces the 240/760 strata, all 29
densities, 75 concept traces, native/invariant outputs, density support,
alignment/lag/density-local-partial/tie-inclusive-overlap rows, counts of
correlation-defined rows, the row-permutation selection null, equilibrium
bootstrap, exemplar source coordinates, and shift checks. Compare to columns in
`unit_density_summary.csv`, `unit_concept_alignment.csv`,
`native_function_comparison.csv`, `natural_exemplars.csv`, and
`shift_consistency.csv`. The analytic mode-3, lag-7, null-concept,
non-contiguous-rank, tied-mask, flat-row, selection-null, grouped-row,
wrapped-coordinate, deterministic-cluster, transfer-axis, and native-signed
tests are independently runnable with `pytest tests/xai/test_unit_semantics.py`.

**Checkable from committed artifacts alone.** The six acceptance verdicts and
all headline numbers above are in [summary.json](S05_artifacts/summary.json),
[unit_motifs.csv](S05_artifacts/unit_motifs.csv), and the linked CSVs. The
committed [manifest](S05_artifacts/manifest.json) records exact config, command,
1000 parent row IDs, two member IDs, source hashes, package versions, source
commit, dataset/checkpoint fingerprints, wall time, and 20 output hashes.

**Not checkable off the researcher's machine, and why.** The checkout lacks the
678 MB source HDF5 bytes and the ignored `densities.h5`, `concept_traces.h5`,
`natural_motifs.h5`, and `first_layer_transfer.h5`, so their bytes cannot be
matched to the manifest hashes. The nearest proxy is a deterministic
recalculation on the mapped 1,000 slice rows and comparison with every committed
small table; agreement checks every S05 scientific headline and coordinate, but
not the archived large-file hashes. The slice provenance carries the same source
SHA-256 as the manifest. No S05 headline depends on a row outside the slice.

## Deferred

- **Direct window surrogate for unresolved units.** Both MVD members' final
  receptive fields cover all 96 positions, so the nominal local surrogate has
  672 raw inputs and is not the small regression anticipated by PLAN. Choosing
  and regularizing a nonlinear surrogate on this same registered panel would
  exceed the one-session MVD and add a method-selection problem. The 23
  unresolved units are retained rather than given post-hoc names.
- **NMF or sparse dictionary learning.** The first natural-exemplar diagnostic
  finds mostly singleton/two-outlier splits, not balanced incoherent mixtures.
  The robust centers and dispersions are published; a higher-capacity
  decomposition was not justified by this result.
