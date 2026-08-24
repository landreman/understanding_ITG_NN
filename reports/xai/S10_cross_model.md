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
reduced the result to **163 edges and eight constrained motifs**. The 334 rejected
edges are retained in [unit_matches.csv](S10_artifacts/unit_matches.csv), not
dropped.

Five of the eight final motifs occur in at least four top members; their member
counts are **9, 7, 7, 6, and 4**. The other three are two-member correspondences.
Only one motif contains a unit with an independently supported S05 name: the
seven-member `motif_001` contains the top member's `u001`, which S05 associated
with the 25-point circular mean of the paper's $f_Q$ integrand. Because only one
of the seven matched units has that supported name, this is a **tentative anchor**,
not evidence that all seven units measure the $f_Q$ integrand. The other seven
motifs remain `unresolved_by_S05_vocabulary` in the
[motif catalog](S10_artifacts/motif_catalog.csv).

Representations are broadly similar across all 100 members. Median linear CKA
(Centered Kernel Alignment, a similarity score that compares the geometry of two
activation spaces even when their widths differ) falls from **0.948** in the first
canonical spatial layer to **0.814** at the invariant bottleneck. The intervening
medians are 0.922, 0.894, 0.847, and 0.865. This supports shared representation
geometry, not identical mechanisms. Removing the 5% highest joint-norm probe rows
changes the median pair score by only **0.0064–0.0220**, although the largest
bottleneck change is 0.117.

Validation rank does not organize the cross-model results strongly. A fixed
four-cluster cut of the multi-evidence dendrogram places **95/100** members in one
core cluster and five in three small outlier clusters. Distance from the medoid
member has Spearman rank correlation **0.118** with stored-validation rank
($p=0.243$). Bottleneck CKA medians are 0.796 within the top 10, 0.814 within ranks
11–50, and 0.816 within ranks 51–100; lower-ranked members are not less
representationally similar.

The four narrow-bottleneck members ($C\le11$) are mixed evidence. Their median
multi-evidence distance to wide members is **3.153**, versus **1.177** between two
wide members, and three of the four sit outside the 95-member core cluster.
Contradictorily, narrow-wide bottleneck CKA is **0.813**, nearly identical to the
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
resolution. The final consensus edge additionally requires recurrence at least
0.70, flux-residual similarity at least 0.50, four-summary causal similarity at
least 0.70, and signed per-row ablation cosine at least 0.50 separately in both
output regimes.

The matched unit's mean replacement is a
`deliberately_off_manifold_diagnostic`: changing a hidden unit diagnoses the
network, not a realizable equilibrium or a causal plasma intervention. Among the
163 final edges, median flux-residual similarity is **0.892**, median recurrence
is **1.00**, and median stable/unstable causal similarity is **0.708/0.755**.

Consensus motifs are connected components of final edges with an extra
one-unit-per-member constraint. Pairwise assignments can form inconsistent
cycles, so edges are considered in descending recurrence-times-causal score and
any union that would add a second unit from the same member is rejected.

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

### Member clustering

Member distance equally combines four robustly standardized blocks:

- all-panel native predictions;
- signed and absolute S01-IQR-scaled canonical input gradients, separately by
  output regime;
- signed/RMS mean-replacement causal profiles, separately by regime; and
- bottleneck concept-selectivity profiles.

Average-linkage hierarchical clustering produces the
[dendrogram](S10_artifacts/member_dendrogram.png); the four-cluster cut was fixed
in the config and is descriptive. Architecture and width are compared after
clustering, not used to construct the distance. The five members outside the
95-member core are stored-validation ranks 1, 58, 59, 86, and 96. Three of the
four narrow members are among them; two wide lower-ranked members are also
outliers, so narrow width is not a complete explanation.

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
  0.70–1.00, with median 1.00.
- CKA uses 20 whole-equilibrium draws because the original all-100 × six-layer ×
  100-draw configuration exceeded the step budget. These intervals are explicitly
  low-resolution sensitivity checks.
- The CKA outlier deletion changes median scores by 0.0064–0.0220 across layers.
- The clustering cut is not bootstrap-stability evidence. The continuous
  distance matrix and full dendrogram are primary; the four labels are a compact
  description.
- No multiple-testing correction is attached to individual exploratory CKA or
  match rows. Consensus is controlled by preregistered effect-size, recurrence,
  flux-residual, and regime gates rather than pairwise $p$-values.

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
| “matched motifs have comparable causal effects” | **Pass after correction.** | All 163 final edges have signed mean-replacement cosine $\ge0.50$ separately on 240 stable/near-floor and 760 unstable rows; medians 0.708/0.755. This correction rejected 334 preliminary edges. [Motif catalog](S10_artifacts/motif_catalog.csv). |
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
eight motifs, 29,700 CKA rows, 100 cluster rows, all cohort counts, figures, and
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
