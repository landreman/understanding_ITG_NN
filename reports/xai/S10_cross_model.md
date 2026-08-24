# S10 — Representations and motifs common across networks

## Result

The networks share broad representational structure, but the strict evidence for
shared **causal motifs** is much smaller than activation similarity alone suggests.
Across the stored-validation top 10, globally optimal one-to-one assignment found
**582** above-threshold unit pairs. Of these, **497** initially passed equilibrium-
bootstrap recurrence, flux-residual activation similarity, and a four-number causal
summary. A post-run audit of the retained signed effects found that this summary
could hide opposing effects within an output regime. Requiring cosine similarity
of at least 0.5 separately on the 240 stable/near-floor and 760 unstable rows
reduced the result to **163 final edges**. Motif membership uses a separate,
stricter 0.70 per-regime threshold, now recorded explicitly in the config: 74
final edges are eligible, and 70 remain after the one-unit-per-member constraint
to form **eight motifs**. The
334 edges rejected by the 0.50 final-edge gate are retained in
[unit_matches.csv](S10_artifacts/unit_matches.csv), not dropped.

Five of the eight final motifs occur in at least four top members; their member
counts are **9, 7, 7, 6, and 4**. The other three are two-member correspondences.
Only one motif contains a unit with an independently supported S05 name: the
seven-member `motif_001` contains the top member's `u001`, which S05 associated
with the 25-point circular mean of the paper's $f_Q$ integrand. Because only one
of the seven matched units has that supported name, this is a **tentative anchor**,
not evidence that all seven units measure the $f_Q$ integrand. That anchor is
also the narrowest member and an average-linkage outlier, so it is not a typical
ensemble member under the registered description. S05 screened two units in each of motifs 001–003,
one in motif 004, and **none** in motifs 005–008. Thus the first three unresolved
labels mean “screened without a supported name,” while the last four mean “not
yet screened,” not evidence that the vocabulary failed. These counts are columns in the
[motif catalog](S10_artifacts/motif_catalog.csv).

Representations are broadly similar across all 100 members. Median linear CKA
(Centered Kernel Alignment, a similarity score that compares the geometry of two
activation spaces even when their widths differ) falls from **0.948** in the first
canonical spatial layer to **0.814** at the invariant bottleneck. The intervening
medians are 0.922, 0.894, 0.847, and 0.865. This supports shared representation
geometry, not identical mechanisms. Removing the 5% highest joint-norm probe rows
changes the typical pair score by a median of **0.0064–0.0220**, although the largest
bottleneck change is 0.117. No permutation or chance baseline was registered,
so the raw decline with depth is descriptive and does not establish that the
networks become more individual in deeper layers.

Validation rank does not organize the cross-model results strongly. Average
linkage at a fixed four-cluster cut places **95/100** members in one core cluster,
but complete and Ward linkage instead produce 82/12/5/1 clusters with different
member identities. The cut is therefore descriptive and member-level outlier
claims are not carried forward. Distance from the medoid
member has Spearman rank correlation **0.118** with stored-validation rank
($p=0.243$). Bottleneck CKA medians are 0.796 within the top 10, 0.814 within ranks
11–50, and 0.816 within ranks 51–100; lower-ranked members are not less
representationally similar. The correlation includes the medoid's own zero
distance at stored-validation rank 74.

The four narrow-bottleneck members ($C\le11$) are mixed evidence. Their median
multi-evidence distance to wide members is **3.153**, versus **1.177** between two
wide members. Contradictorily, narrow-wide bottleneck CKA is **0.813**, nearly identical to the
wide-wide median **0.814**. Narrow members therefore differ in the combined
prediction/gradient/causal/concept signatures, not in a simple loss of shared
bottleneck representation geometry.

## Estimand and cohort

The explained function is each member's S02 canonical exactly shift-invariant

$$
\tilde f_m(X,g_T,g_n)=\operatorname{MLP}_m(\bar u_m(X),g_T,g_n)
$$

in native $\max(\log Q,-2)$ units. No prediction or intervention effect is
exponentiated. Member-level bottlenecks, predictions, signed input gradients,
and signed mean-replacement effects are retained before any aggregation.

The cohort is S01's frozen 1,000-row varied-gradient interpretation panel: one
tube from each of 1,000 `equilibrium_files`, with **240 stable/near-floor** and
**760 unstable** rows. Top-10 unit matching uses every panel row. All-100 CKA uses
the same deterministic stratified 64-equilibrium probe, and all-100 input-gradient
profiles use a separate deterministic stratified 64-equilibrium probe. Production
read the external HDF5 dataset, never `tests/data/review_slice.h5`.

The registered run is `cross-model-all100-panel1000`. Its committed
[manifest](S10_artifacts/manifest.json) records dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`,
checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`,
CPU execution, seed 20260823, all 100 member IDs, all 1,000 parent row IDs, and
**3,601.96 s (60.03 min)** measured wall time. The manifest also records the
post-run regime audit and updated hashes of the three affected small artifacts.
After automated review, the audit, S05 motif annotations, cohort CKA columns,
and every derived `summary.json` field were folded into the committed runner;
the small summary and manifest artifacts were regenerated from the retained
tables. The registered runner now reproduces the published schemas directly.
The manifest keeps the registered 60-minute run's original runner/config hashes
(`fae724...` / `2c5fb4...`) at top level. Current post-review code and config are
recorded separately as `postprocessing.reproduction_source_hashes` and
`reproduction_config`; they reproduce the retained arrays and tables but did not
incur the historical wall time. This separation prevents later code from being
misrepresented as the code that executed the registered run.
The manifest also states that its committed `output_hashes` describe the
post-review reproduction artifacts, updated at `2026-08-24T08:17:33Z`, while
the 3,601.96 s wall time applies only to the original all-100 member-signature
and CKA execution before table enrichment. It is intentionally a disclosed
historical-plus-reproduction record, not the output of one command invocation.
The runner now emits all three disclosure keys itself, so a future full run
cannot silently drop the disclosure fields. The committed values remain
hand-maintained because this manifest is a historical-plus-reproduction hybrid;
a new full run would emit values describing that single new invocation.
The production environment used SciPy 1.13.1 for average-linkage clustering.
Its compatibility smoke test imported SciPy and completed the real 100-member
linkage, four-cluster cut, square-distance conversion, and dendrogram calls in
this registered run. SciPy 1.13.1 is consequently pinned in both the `xai`
extra and `requirements/xai.lock`; the command-line module imports it only when
the production clustering path is reached, so core inference and baseline CI
remain independent of optional XAI packages.

## Methods

### Functional unit signatures and assignment

Each top-10 unit has four signature families:

1. its activation over identical panel rows;
2. that activation after linearly removing native target, $a/L_T$, and $a/L_n$;
3. concept selectivity plus density support and six Fourier-band preferences; and
4. signed mean-replacement effects summarized separately by output regime.

The registered score weights raw activation, flux-residual activation,
concept/density, and causal summaries by 0.30/0.30/0.20/0.20. A pure-NumPy
Hungarian assignment finds the global one-to-one maximum and gives every left
unit a private dummy option, so units below 0.65 remain unmatched. Pair recurrence
uses 100 resamples of whole `equilibrium_files`; recurrence therefore has 0.01
resolution. The bootstrap recomputes the two activation components; the
concept/density and causal-summary components, 40% of the score by registered
weight, remain fixed at their full-panel values. The final consensus edge
additionally requires recurrence at least
0.70, flux-residual similarity at least 0.50, four-summary causal similarity at
least 0.70, and signed per-row ablation cosine at least 0.50 separately in both
output regimes.

Because this frozen panel has exactly one tube per equilibrium, grouped and
ordinary row bootstrap draws are arithmetically identical here. Grouping remains
the required implementation contract and matters for any cohort with sibling
tubes, but it does not widen uncertainty on this particular panel.

The matched unit's mean replacement is a
`deliberately_off_manifold_diagnostic`: changing a hidden unit diagnoses the
network, not a realizable equilibrium or a causal plasma intervention. Among the
163 final edges, median flux-residual similarity is **0.892**, median recurrence
is **1.00**, and median stable/unstable causal similarity is **0.708/0.755**.
Direction agreement is not enough for comparable effect size: the ratio of the
larger to smaller all-panel root-mean-square signed effect has median **1.33**,
90th percentile **2.17**, and maximum **4.05** across those edges. The exact
ratio is published on every match row.
Separately by output regime, the stable/near-floor ratio is **1.41/2.50/5.04**
and the unstable ratio is **1.34/2.21/3.98** (median/90th percentile/maximum).

Consensus motifs are connected components of the **74** final edges that also
meet a stricter, config-driven per-regime causal cosine of at least 0.70. This
motif-membership threshold is distinct from the 0.50 final-edge threshold.
Pairwise assignments can form inconsistent cycles, so edges are considered in
descending recurrence-times-causal score; the one-unit-per-member constraint
rejects four unions, leaving 70 catalog edges.

The binding 0.70 motif threshold was already a code literal in the first S10
commit, before the regime audit, and matches the registered pooled-causal gate;
it was not fitted to obtain eight motifs. The catalog is nevertheless sensitive
to it: thresholds 0.50/0.60/0.70/0.80 give **14/12/8/4 motifs**. The committed
[threshold sweep](S10_artifacts/motif_threshold_sensitivity.csv) also publishes
eligible edges, post-constraint edges, and member counts. Eight is therefore a
thresholded catalog description, not a threshold-invariant natural count.

### Cross-model CKA

Linear CKA is computed for every one of the **4,950** member pairs at five full
96-position canonical à-trous layers and at the invariant bottleneck: **29,700**
rows in [cka.csv](S10_artifacts/cka.csv). Spatial maps are flattened only after
featurewise standardization; the bottleneck retains its natural width. CKA is
supporting evidence because similar activation geometry does not prove identical
computation.

Intervals use 20 whole-equilibrium bootstrap draws. This is a budget-limited
sensitivity interval, not a precise 95% confidence interval: median interval
width grows from 0.030 in layer 1 to 0.165 at the bottleneck, and the maximum
bottleneck width is 0.320. The exact point estimates use all 64 probe rows.
Removing the 5% highest joint-norm rows per pair supplies the outlier check.
Because the registered probe has one tube per equilibrium, equilibrium-grouped
CKA resampling is distributionally a row bootstrap here; the grouping remains
explicit so the helper is safe on future multi-tube cohorts.

### Member clustering

Member distance equally combines four robustly standardized blocks:

- all-panel native predictions;
- signed and absolute S01-IQR-scaled canonical input gradients, separately by
  output regime;
- signed/RMS mean-replacement causal profiles, separately by regime; and
- bottleneck concept-selectivity profiles.

Six non-prediction columns are constant across all 100 members and therefore
carry no clustering information: 2 of 20 causal-signature columns and 4 of 150
concept-profile columns. Robust standardization drops those six columns.

The across-member robust standardization is what makes the seven gradient
channels comparable in the distance. The S01 IQR scale is retained for the
archived `scaled_signed_input_gradient` values; a per-channel constant cancels
inside that later standardization, so the distance does not depend on the S01
factors themselves.

Average-linkage hierarchical clustering produces the
[dendrogram](S10_artifacts/member_dendrogram.png); the four-cluster cut was fixed
in the config and is descriptive. Complete and Ward linkage both give
82/12/5/1 rather than average linkage's 95/3/1/1 and change the member identities,
so no individual network is classified as a robust outlier. Architecture and
width are compared after clustering, not used to construct the distance. The
narrow-member conclusion rests on the linkage-free continuous distances.

## Stable/near-floor versus unstable rows

Every causal edge is now required to agree separately in the two regimes. This
was consequential: **334/497** preliminary edges failed the regime-specific
gate. Before correction, preliminary stable causal cosine ranged down to -0.789
and unstable cosine down to -0.353, despite apparently strong four-summary
similarity. After correction every final edge is at least 0.5 in both regimes.

Input-gradient profiles also retain signed stable/unstable blocks. CKA is not a
target-effect statistic, so it uses a single stratified probe containing both
regimes rather than claiming separate causal behavior.

## Uncertainty and sensitivity

- Unit recurrence uses 100 whole-equilibrium resamples. Final-edge recurrence is
  0.78–1.00, with median 1.00.
- CKA uses 20 whole-equilibrium draws because the original all-100 × six-layer ×
  100-draw configuration exceeded the step budget. These intervals are explicitly
  low-resolution sensitivity checks.
- The CKA outlier deletion changes the typical pair score by a median of
  0.0064–0.0220 across layers.
- The clustering cut is not bootstrap-stability evidence. The continuous
  distance matrix and full dendrogram are primary; the four labels are a compact
  description.
- No multiple-testing correction is attached to individual exploratory CKA or
  match rows. Consensus is controlled by recorded effect-size, recurrence,
  flux-residual, and regime gates rather than pairwise $p$-values. The 0.50
  per-regime gate was selected post hoc during the same-panel audit, as disclosed
  above; it was not preregistered.

## Failed checks, corrections, and negative results

- The tests were written first and initially failed all S10 paths with explicit
  `NotImplementedError`.
- The first all-100 scale-up was interrupted during the 200-draw unit-match
  bootstrap after pilot extrapolation proved optimistic. Weighted grouped
  sufficient statistics replaced expanded-row least squares.
- A second scale-up reached CKA but exposed two unnecessary costs: rebuilding
  feature-width Gram matrices inside every draw and using a 3,072-feature cross
  product for each outlier check. Algebraically identical 64×64 Gram operations
  replaced both. The registered budget adaptation uses 100 match draws and 20
  CKA draws; the planned 100-draw CKA intervals are deferred below.
- The preliminary causal-summary gate admitted **334** edges whose per-row signed
  effects failed in at least one regime. They were rejected before reporting.
- The first CSV rewrite of that audit treated serialized text `"False"` as
  truthy. A regression test now prevents promoting a preliminary failure; the
  artifact was regenerated from the preserved preliminary column.
- Seven of eight consensus motifs have no supported S05 name. Functional
  correspondence is not semantic identification.
- CKA is high even where multi-evidence distances say narrow members differ.
  Representation similarity alone would have overstated mechanistic agreement.
- Validation rank is not a useful organizing axis here: rank-versus-medoid
  distance is weak and nonsignificant, and lower-ranked cohort CKA is not lower.
- Hidden interventions are off-manifold and do not establish plasma causality.
- All members share training data and an architecture family. Motif recurrence
  therefore has an unmeasured shared-training floor and is not evidence of
  independent discovery.
- The 0.50 final-edge gate was introduced by an audit on the same 1,000 rows,
  not an independent holdout. Sweeping that nonbinding edge gate does not move
  the catalog because the separate 0.70 motif threshold binds. Sweeping the
  binding motif threshold does move it: 14/12/8/4 motifs at
  0.50/0.60/0.70/0.80. Neither cutoff has independent holdout validation.

## Mutation testing

Three deliberate mutations turned the focused suite red and were reverted:

1. assigning independent bootstrap multiplicities to sibling tubes instead of
   one multiplicity per `equilibrium_files` failed the repeated-equilibrium
   equality check;
2. centering activations without removing the target/drive fit left a
   label-only correspondence at correlation 0.9999 and failed the flux-residual
   null control; and
3. wiring the stable/near-floor causal gate from unstable rows changed a known
   stable cosine from -1 to +1 and failed the regime-specific effect test.

Automated review supplied a fourth mutation: deleting the one-unit-per-member
union constraint initially survived. A new cyclic toy graph with two units from
one member now turns that mutation red; the constraint removes four real unions
in the registered catalog.

A second review found two more surviving production-wiring mutations. Replacing
the residualized activation component inside `functional_similarity` by the raw
correlation, and deleting the S01 robust channel scales inside
`_member_attribution`, both initially stayed green. The label-only null now runs
through the full similarity function, and an analytic gradient toy now requires
the registered 1:1000 channel scaling; both mutations turn red.

A third review found that the new stable/unstable root-mean-square magnitude
columns were not pinned by the original symmetric fixture. The revised fixture
has stable, unstable, and pooled ratios of 3, 1, and $\sqrt{5}$: taking the
stable ratio from unstable rows or replacing both regime ratios by the pooled
ratio now turns the named regime-effect test red.

A fourth review found three untested motif-annotation paths. A temporary S05
table now distinguishes supported, screened-without-support, and unscreened
units; it catches a wrong unit separator and treating every screened unit as
supported. A separate threshold toy catches counting motif-eligible edges with
the 0.50 edge gate instead of the registered 0.70 motif gate. All three
mutations now turn their named science controls red.

A fifth review found the remaining seven untested runner helpers. Direct toys
now pin the stable/unstable effect signature, 5% joint-norm CKA trimming,
regime-stratified row selection, density-spectrum centering, channel-index
ordering, five-quantile member profiles, and removal of constant columns during
standardization. Each corresponding mutation turns its named test red. A zero
RMS in either output regime now raises rather than serializing non-standard
`Infinity` into `summary.json`.

A sixth review found the final untested layer: how `run()` connected those
helpers. Four extracted wiring toys now require the native flux target plus both
drives in the residualization covariates, per-column probe standardization, the
`<= -1.9` stable/near-floor convention, and peak-absolute plus signed-mean
concept profiles. Dropping or reversing any of those connections turns its
named test red.

A seventh review found the remaining inline attribution-profile wiring. It is
now a named helper with an asymmetric toy requiring signed-stable,
signed-unstable, absolute-stable, and absolute-unstable channel means in that
order. Pooling regimes, deleting magnitude blocks, or copying signed values
into the magnitude blocks turns the same named test red.

An eighth review reached the two shared-library paths on which S10 depends.
A two-block 1000× scale toy now requires `member_distance_matrix` to be
invariant to each evidence block's natural scale. A grouped label-only toy now
requires bootstrap recurrence below 0.1 after weighted flux residualization;
skipping that residualization produces recurrence 1.0. Both mutations turn
their named library tests red.

A ninth review found the final two unpinned shared-library lines. A duplicated
200-column block must leave the distance unchanged and a closed-form two-block
toy pins the absolute normalization. A 2×2 score matrix where greedy matching
collides now requires the globally optimal, distinct right-unit assignment.
Deleting column-count normalization or replacing the Hungarian assignment with
row-wise nearest neighbors turns the corresponding test red.

A tenth review found the two final post-loop assemblies. The motif edge helper
now requires the weaker of the stable and unstable similarities, and the member
evidence helper requires predictions, input attributions, causal signatures,
and concept profiles exactly once each. Replacing `min` by `max`, or dropping or
duplicating a family, turns the corresponding named test red.

An eleventh review found correlation centering and the final CKA outlier-check
emission. Adding large per-column offsets must leave concept correlation
unchanged. A paired CKA helper must emit both the untrimmed point and the
independently trimmed result, which differ on a high-norm-row toy. Deleting
centering or copying the point into the trimmed field turns its named test red.

A twelfth review reached the three remaining shared-library similarity
primitives. A lopsided-weight toy now requires weighted correlation to change
from 0 to 9/22; offsetting either activation column must leave ordinary
correlation unchanged; and independently scaling each unit signature must leave
row-wise cosine similarity unchanged. Replacing bootstrap multiplicities by
ones, deleting correlation centering, or deleting cosine normalization turns
the corresponding named test red.

A thirteenth review isolated the constant term in weighted flux/drive
residualization. A nonzero-offset toy with units sharing only constant offsets
has recurrence 0.0 with the registered intercept and 1.0 when the regression is
forced through the origin. Deleting the intercept now turns that named grouped
bootstrap test red.

The manifest source-identity assertions live in a separately named tripwire
test. They force provenance to be refreshed after any runner edit, but are not
counted as evidence that a scientific mutation is caught; the mutation results
above name the specific scientific test that turns red.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s10-pilot XDG_CACHE_HOME=/private/tmp/cache-s10-pilot \
  .venv-xai/bin/python scripts/xai_s10_cross_model.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s10-prod XDG_CACHE_HOME=/private/tmp/cache-s10-prod \
  .venv-xai/bin/python scripts/xai_s10_cross_model.py
.venv-xai/bin/python -m pytest tests/xai/test_cross_model.py \
  tests/xai/test_cross_model_script.py tests/xai/test_cross_model_artifacts.py -q
source .venv-xai/bin/activate && make check
```

The CLI supports `--config`, `--members`, `--rows`, `--device`, `--seed`,
`--batch-size`, `--resume`, `--output-dir`, and dataset/checkpoint overrides.

## Acceptance criteria

| PLAN criterion | Verdict | Number or artifact |
| --- | --- | --- |
| “correspondences survive equilibrium bootstrap and are not driven only by flux labels” | **Pass with strict attrition exposed.** | 163/582 matched pairs pass recurrence $\ge0.70$, flux-residual similarity $\ge0.50$, and both causal-regime gates; final median recurrence 1.00 and flux-residual similarity 0.892. [Unit matches](S10_artifacts/unit_matches.csv). |
| “matched motifs have comparable causal effects” | **Pass after correction.** | All 163 final edges have signed mean-replacement cosine $\ge0.50$ separately on 240 stable/near-floor and 760 unstable rows; medians are 0.708/0.755. Stable effect-size ratios are 1.41/2.50/5.04 and unstable ratios are 1.34/2.21/3.98 (median/p90/max). [Unit matches](S10_artifacts/unit_matches.csv) and [motif catalog](S10_artifacts/motif_catalog.csv). |
| “lower-ranked cohorts provide a registered comparison” | **Pass.** | All 10/40/50 registered members are present. Bottleneck CKA medians are 0.796/0.814/0.816 within top/middle/lower cohorts; rank-versus-medoid distance $\rho=0.118$, $p=0.243$. [Cohort comparison](S10_artifacts/cohort_comparison.csv) and [member clusters](S10_artifacts/member_clusters.csv). |

## Reviewer reproduction

**Recomputable on the slice.** All 1,000 S01 panel parent rows are in
`tests/data/review_slice.h5`. Translate them with
`load_review_slice_index().slice_rows()` before loading. The reviewer can
recompute all 100 invariant bottlenecks/predictions, signed mean-replacement
effects, top-10 assignments, stable/unstable gates, 64-row CKA probe, and member
profiles. The exact production calculation took 60.03 CPU minutes; the practical
nearest proxy is the top-three/96-row pilot, which checks axes, native units,
assignment, grouped recurrence, unmatched controls, both regime gates, and CKA.

**Checkable from committed artifacts alone.** The 582 matches, 163 final edges,
74 motif-eligible edges, the 14/12/8/4 threshold sweep, eight motifs, 29,700 CKA rows, 100 cluster rows, all
cohort counts, figures, and
headline medians are committed under [S10 artifacts](S10_artifacts/).
`test_cross_model_artifacts.py` recomputes counts and gates, enforces one unit per
member per motif, checks every pair/layer, and verifies committed hashes against
the manifest.

**Not checkable off the researcher's machine, and why.** The checkout lacks the
678 MB source HDF5 bytes and the git-ignored `member_signatures.h5`, so those
bytes cannot be matched to the manifest. The nearest proxy is to regenerate the
same member-level arrays on the mapped 1,000 slice rows; agreement with every
committed small table checks the scientific conclusions but not the archived
large-file hash. The 20-draw CKA intervals are exactly reproducible from the
slice but remain low-resolution by design.

## Deferred

- **100-draw all-pair CKA intervals.** The all-100 extrapolation exceeded the
  one-session budget. Exact all-pair point CKA, outlier sensitivity, and 20-draw
  grouped intervals are complete; increasing only the sensitivity draws would
  not change the MVD conclusions.
- Nothing from the MVD was dropped: top-10 functional unit matching and all-100
  prediction/input-attribution/causal/concept member clustering are complete.
