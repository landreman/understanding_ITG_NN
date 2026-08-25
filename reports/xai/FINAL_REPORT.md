# Final report — what the GX ITG ensemble learned, and what remains physical hypothesis

## Result

The research program supports a coherent but deliberately limited account of the
trained ensemble. The networks use a **feature family** built around the paper's
$f_Q$ and $f_{\rm stab}$ quantities, localized bad-curvature/flux-compression
structure, cross-channel alignment, and drive-dependent interactions. They agree
strongly at the level of channel ordering and broad representations, but not on
one common signed position-by-position calculation. A compact table of 17
cyclic-invariant quantities reproduces three individual members at held-out
$R^2=0.8561$–$0.8636$, yet it leaves substantial residual error and does not
assign one physical meaning to each internal unit.

Several appealing explanations do not survive the full evidence. Signed input
attribution is essentially uncorrelated with where GX transports heat along the
field line: the three member correlations with $Q(z)$ are **-0.0212, -0.0128,
and -0.0116**. The tested hidden representations neither specifically encode nor
use GX zonal-flow magnitude. Ensemble spread ranks likely errors well
(overall Spearman **0.761 [0.730, 0.794]**) but is not a calibrated confidence
interval: absolute error exceeds spread on **44.9%** of rows and there are eight
registered common-mode failures.

The physical evidence is observational. All four leading geometry candidates
track GX output among naturally occurring, force-balanced equilibria, but none
passes the balance and overlap gates needed for a physical causal claim.
Geodesic-curvature/compression is the highest-value next target because its
fixed-drive information gain is fold-robust: its adjusted native-output
association is **+0.5588 [0.3183, 0.7640]**, and adding it to the paper baseline
improves held-out mean squared error by **0.01880 [0.00617, 0.03252]**. Those
numbers are still observed comparisons. Remaining nuisance imbalance is
**1.068** against the registered 0.5 maximum, and overlap is **0.478** against
the registered 0.8 minimum.

The smallest useful next calculation is therefore not a GX campaign. It is the
registered VMEC Jacobian feasibility test at three typical unstable anchors:
search both signed boundary-coefficient directions for geodesic/compression and
bad-curvature/compression while recomputing force-balanced equilibria. It costs
an estimated two Perlmutter node-hours and includes no GX run. Researcher
approval is required before that external computation.

The machine-readable synthesis consists of the
[evidence matrix](S14_artifacts/evidence_matrix.csv),
[64-record evidence ledger](S14_artifacts/evidence_ledger.csv),
[nine-claim register](S14_artifacts/claim_register.csv),
[18-run/19-record provenance index](S14_artifacts/reproducibility_index.csv), and
[prioritized next-experiment list](S14_artifacts/next_experiments.csv).

## What function and outcome each conclusion concerns

The project contains five objects that must not be conflated:

1. **Original member $f_m$.** The prediction from one trained ensemble member
   before any post-hoc symmetry operation. Member-level signed results are
   retained because averaging could hide opposing mechanisms.
2. **Ensemble mean.** The arithmetic mean of the 100 original member outputs.
   It is a predictor, not evidence that every member uses the same mechanism.
3. **Phase average $\bar f$.** A member prediction averaged over common cyclic
   shifts. This removes residual dependence on the arbitrary grid origin but
   does not enforce reflection invariance.
4. **Canonical invariant $\tilde f$.** The phase-averaged prediction additionally
   averaged over the registered physical reflection. S02 selected this as the
   canonical function for model-mechanistic work.
5. **Observed GX.** The simulation output, whose primary estimand is always
   `max(log Q, -2)`. It is never replaced by $Q$, by an exponentiated prediction,
   or by a network output.

Input-attribution, hidden-state, cross-member, and distillation claims below
primarily concern individual $\tilde f_m$ functions. S11's spread concerns the
population of all 100 $\tilde f_m$. S13 concerns observed GX and performs no
network inference. Spatial comparisons retain the original-$f_m$ result beside
the invariant result where applicable. The
[evidence ledger](S14_artifacts/evidence_ledger.csv) records `function_scope`,
the native claim `estimand`, the specifically measured `outcome`, cohort,
regime, validity tag, uncertainty grouping unit, and intervention status for
every piece of evidence.

## Cohorts and regimes

The scientific panel is S01's frozen 1,000-equilibrium sample, one selected flux
tube per `equilibrium_files`. The varied-gradient interpretation panel has
**240 stable/near-floor** and **760 unstable** rows. The fixed-gradient physical
comparison panel has **23 stable/near-floor** and **977 unstable** rows at
$(a/L_T,a/L_n)=(3,0.9)$.

Stable/near-floor rows are never silently pooled into a spatial feature claim.
At the `max(log Q,-2)` floor, ranks and $R^2$ can become uninformative because
the outcome has almost no variance; error-scale quantities remain meaningful.
Every source analysis keeps the stable/near-floor and unstable strata separate,
and the synthesis ledger preserves the source regime. Uncertainty resamples and
splits complete `equilibrium_files`, never individual flux tubes.

## Evidence matrix

The full matrix has the eleven PLAN columns: bottleneck Shapley value (a credit
allocation over hidden units), unit semantics, input attribution, supported
perturbation, hidden encoding, hidden intervention, cross-model consensus,
distillation, $Q(z)$, zonal flow, and natural experiment. Empty cells are not
allowed: a missing test is explicitly named `not_tested`, `not_supported`, or an
equivalent negative value.

| candidate hypothesis | synthesis status | claim grade and decisive limit |
|---|---|---|
| localized $f_Q$-integrand peak | supported | observational-physical; physical causality unresolved |
| $f_{\rm stab}$ | supported | observational-physical; physical causality unresolved |
| bad-curvature/compression | regime-dependent | learned signs vary with depth/drive; physical causality unresolved |
| geodesic-curvature/compression | regime-dependent | highest-value physical target; recurrence and hidden sign vary |
| cross-channel co-location | supported | model-mechanistic only; tested edits are off-manifold |
| parallel order versus low-frequency envelope | unresolved | the family matters, but the two explanations remain inseparable |
| coarse channel consensus | supported | model-mechanistic, not precise signed-cell consensus |
| direct signed focus on $Q(z)$ | contradicted | no common member-level signed spatial mechanism |
| learned zonal-flow mechanism | contradicted | observed association is not specific hidden encoding/use |
| compact invariant formula | supported | about 86% member variance, not an exact formula or physical law |
| spread as an error signal | regime-dependent | useful ranking diagnostic, not an error bar |

This is **5 supported, 3 regime-dependent, 2 contradicted, and 1 unresolved**.
Uncertainty and negative results are columns in the
[machine-readable matrix](S14_artifacts/evidence_matrix.csv), not prose-only
qualifications.

## Headline conclusions and triangulation

Every row below is registered in the
[claim register](S14_artifacts/claim_register.csv). Its evidence IDs resolve to
exact CSV selectors, fields, and verbatim values in the
[evidence ledger](S14_artifacts/evidence_ledger.csv). `Families` counts only
per-claim links marked `corroborates`; it does not read the evidence row's
candidate-level `direction`. Every link also names the conjunct it addresses,
so the two sides of a compound claim remain visible. Every linked evidence row
must belong to a candidate the claim explicitly names. The register separately
retains all method families consulted, including qualifiers, contradictions,
context, and null results.

| ID | conclusion | claim-aligned families | source steps | max families on one conjunct | representative machine-readable evidence |
|---|---|---:|---:|---:|---|
| C01 | The networks encode and use the paper's $f_Q$/$f_{\rm stab}$ family, without one scalar uniquely owning the credit. | 2 | 3 | 1 | exact hidden-direction removal in [S04](S04_artifacts/encoded_vs_used.csv); hidden concept edits in [S08](S08_artifacts/tcav_use.csv); feature recurrence in [S12](S12_artifacts/term_recurrence.csv) |
| C02 | Cross-channel alignment and low-frequency spatial structure matter, but order cannot yet be separated from the low-frequency envelope. | 4 | 3 | 2 | registered perturbation ladder in [S03](S03_artifacts/ladder_summary.csv); low-pass attribution in [S06b](S06b_artifacts/channel_consensus.csv); hidden co-location edits in [S08](S08_artifacts/tcav_use.csv) |
| C03 | Members share a coarse channel/representation story, not one common signed position-by-position mechanism. | 3 | 3 | 1 | member attribution agreement in [S06b](S06b_artifacts/member_agreement.csv); signed spatial null in [S07](S07_artifacts/spatial_alignment.csv); representation similarity and motifs in [S10](S10_artifacts/cohort_comparison.csv) |
| C04 | Broad concept decoding reaches about 91% and a compact 17-feature invariant vocabulary about 86%; neither is an exact formula. | 2 | 2 | 1 | concept completeness in [S09](S09_artifacts/completeness.csv); held-out distillation and the 13/64 unit limit in [S12](S12_artifacts/fidelity.csv), with [nested subsets](S12_artifacts/subset_fidelity.csv) |
| C05 | All four physical candidates are associated with the same native GX output; no physical causal effect has been measured. | 2 | 1 | 1 | grouped-bootstrap native-output rank correlations in [S13 associations](S13_artifacts/fixed_associations.csv); adjusted and matching checks on that same outcome in [S13 rankings](S13_artifacts/candidate_ranking.csv) |
| C06 | Geodesic/compression is the highest-value next physical target, not a causal winner. | 2 | 1 | 1 | native GX rank association plus fold-robust adjusted/residual checks in [S13](S13_artifacts/candidate_ranking.csv); member-variable hidden use and recurrence are retained as qualifiers from [S08](S08_artifacts/tcav_use.csv) and [S12](S12_artifacts/term_recurrence.csv) |
| C07 | The evidence contradicts a specific learned GX zonal-flow mechanism. | 3 | 2 | 1 | hidden probe and use failures in [S08](S08_artifacts/probe_scores.csv) and [S08 use](S08_artifacts/tcav_use.csv); the observed association in [S07](S07_artifacts/zonal_association.csv) corroborates the distinct conjunct that the association is nonspecific |
| C08 | Ensemble spread ranks error but is not a calibrated confidence interval or guaranteed bound. | 2 | 1 | 1 | spread/error association in [S11](S11_artifacts/spread_error_associations.csv); common-mode failure audit in [S11](S11_artifacts/failure_categories.csv) |
| C09 | The evidence contradicts direct shared signed focus where GX transports heat along $z$. | 2 | 1 | 1 | signed density and attribution comparisons in [S07](S07_artifacts/spatial_alignment.csv); positive-but-limited local hidden probes and edits in [S08](S08_artifacts/probe_scores.csv) and [S08 use](S08_artifacts/tcav_use.csv) are qualifiers and do not count toward the rejection |

The named method families show methodological variety, but they do not by
themselves prove independence. Only **5/9** headlines span two or more source
steps; C05/C06 draw both counted families from S13, C08 from S11, and C09 from
one S07 artifact. Only C02 has two counted method families on the same declared
conjunct; the other compound claims combine evidence for distinct
sub-propositions. The claim register publishes the full per-conjunct family
mapping, distinct source steps/artifacts, and `gate_margin`. Under a strict
reading of PLAN's word “independent,” the criterion is therefore not
demonstrated for every headline.

### Direction-label audit after review

The first review response (`8ea8982`) made evidence `direction` feed the
headline gate and revised labels at the same time. That coupled an interpretive
classification to acceptance. This registered run removes the coupling and
derives every disputed label from named source fields under rules stored in the
config and ledger. Relative to that first response, the complete revision is:

| evidence record | first-response label | final candidate-level label | source-derived rule |
|---|---|---|---|
| `E07_FQ_QZ` | supports | mixed | negative correlation and nonzero best lag |
| `E08_FQ_USE` | supports | regime-dependent | 12/15 complete S08 use gates pass |
| `E08_GEO_USE` | supports | regime-dependent | 10/15 complete S08 use gates pass |
| `E08_COLOCATION_USE` | supports | regime-dependent | 12/15 complete S08 use gates pass |
| `E08_LOCAL_QZ_ENCODING` | contradicts | mixed | median decoding gain is positive overall/unstable but not near the floor |
| `E08_LOCAL_QZ_USE` | contradicts | regime-dependent | 7/15 complete S08 use gates pass |
| `E11_SPREAD_ERROR_ASSOCIATION` | supports | regime-dependent | positive stable/unstable intervals are disjoint |
| `E11_COMMON_MODE_FAILURE` | supports | contradicts | eight all-regime common-mode failures occur |
| `E12_BAD_RECURRENCE` | supports | mixed | all three recurrence values are below 0.5 |
| `E12_GEO_RECURRENCE` | supports | regime-dependent | member recurrence spans 0.27–0.60 |
| `E13_FQ_NATURAL` | supports | unresolved | adjusted effect resolves in 0/7 folds |
| `E13_FSTAB_NATURAL` | supports | regime-dependent | adjusted effect resolves in 4/7 folds |
| `E13_BAD_NATURAL` | supports | regime-dependent | adjusted effect resolves in 2/7 folds |

`E13_GEO_NATURAL` remains `supports` because it resolves in 7/7 folds. The
headline counts no longer depend on any of these row labels. C06 counts only the
two S13 physical-observation families; C08 records `error-ranking utility` and
`not a calibrated guarantee` as distinct conjuncts; and C09 counts only the two
S07 families, with both S08 records marked as qualifiers. Thus a future
candidate-level `direction` revision cannot silently make a headline pass;
claim-alignment judgements remain a separate, disclosed input.

None of these nine rows is a physical causal statement—the register names that
column `physical_causal_statement`. S03, S04, S06, and S08
did execute named **model interventions**—respectively shifting/attenuating input
arrays, editing bottleneck coalitions, restoring frequencies along a gradient
path, and editing hidden states along matched concept directions. Those edits
are tagged deliberately off-manifold: they diagnose the network, not the plasma.
S13 performed only observed comparisons between existing equilibria.

## Quantitative synthesis

### A localized, aligned feature family is genuinely used by the networks

For the top member, bottleneck unit `u001` has exact mean absolute Shapley credit
**0.4183** native units. Removing its $f_Q$-aligned hidden direction changes the
output, while the registered hidden concept-edit gate passes in **12/15**
log-$f_Q$ cells and **15/15** $f_{\rm stab}$ cells. Bad-curvature,
geodesic-curvature, and cross-channel co-location pass in **7/15, 10/15, and
12/15** cells. The signs of the curvature edits change with layer and regime;
that is evidence for conditional use, not a single monotone rule.

The input perturbation ladder independently shows that geometry is not treated
as an unordered bag. Independently shifting the seven channels changes top-10
member outputs by a median **2.413 residual standard deviations**. Jointly
permuting the 96 position vectors changes them by **3.258**. Fully attenuating
the low Fourier band changes them by **3.847**; the middle band is **1.223** and
the high-frequency control, at a roughly seven-times smaller robust input dose,
is **0.099**. The exact common circular-shift control is **8.1e-7**, at
round-off scale. Because joint permutation also
destroys the low-frequency envelope, parallel order and that envelope remain
unresolved as separate mechanisms.

### Consensus is coarse, not a shared precise spatial computation

Across the top ten members, median pairwise channel-rank agreement is **0.964**.
Mean signed cell agreement is only **0.7485**, versus an independent-sign null
of **0.6230**. Representation similarity supplies a second family of evidence
for coarse consensus, while the near-zero signed $Q(z)$ alignment supplies a
direct negative control. The ensemble mean must therefore not be described as
revealing a single signed cell-level mechanism shared by every member.

### The compact vocabulary is useful but incomplete

The 17-feature Explainable Boosting Machine (a readable regression made from
one-feature curves and five registered two-feature surfaces) reproduces the
three individual canonical members at held-out $R^2$ **0.8603, 0.8561, and
0.8636**. The registered interactions add **0.0789, 0.0762, and 0.0796**
$R^2$ beyond the drive-plus-$\log f_Q$ baseline. This is strong evidence for
drive-dependent geometry, but residual standard deviations remain
**0.744–0.768** native units, and only 13/64 bottleneck units reach
$R^2\ge0.8$. PySR symbolic regression remains deferred; no algebraic expression
or symbolic Pareto frontier is claimed.

### The attractive spatial and zonal stories are negative results

For unstable rows, signed low-pass Integrated Gradients (an attribution method:
it assigns output change along a path from a smoothed baseline) correlate with
$Q(z)$ at **-0.0212, -0.0128, and -0.0116**, with incompatible best lags.
Positive-only attribution appears more similar, but it discards the negative
credit and cannot rescue a signed mechanism. Local $Q(z)$ concentration is
positively but weakly decodable overall and on unstable rows, not near the
floor, and 7/15 hidden-use cells pass; those results qualify rather than refute
the candidate and do not establish shared signed spatial focus. Zonal magnitude
passes none of the 15 hidden-use cells and has near-zero held-out probe
performance. Natural GX
association with zonal magnitude is therefore not evidence that the network has
direct access to or uses a zonal-flow variable.

### The physical ranking is useful precisely because its causal gate failed

All four fixed-drive comparisons are confounded: high and low candidate values
occupy different regions of geometry space. The adjusted interval resolves in
**0/7** fold assignments for the localized peak, **4/7** for $f_{\rm stab}$,
**2/7** for bad-curvature/compression, and **7/7** for geodesic/compression.
Geodesic's residual improvement is **0.018801 [0.006169, 0.032516]** native
units squared, yet the balance and overlap failures prohibit a causal claim.
The result prioritizes a test; it does not predict the sign of a realizable
equilibrium intervention.

### Spread is an alarm, not an error bar

All-100 member spread and ensemble absolute error have overall Spearman
**0.761 [0.730, 0.794]**. The association is stronger for stable/near-floor rows
(**0.829**) than unstable rows (**0.575**), so even its ranking value is
regime-dependent. Absolute error is larger than spread on **44.9%** of rows.
Eight cases meet the registered common-mode-failure thresholds: all members
agree closely and are collectively wrong. Shared architecture and training data
make spread informative without turning it into independent uncertainty draws.

## Smallest next calculation and decision gate

Priority 1 in the
[next-experiment list](S14_artifacts/next_experiments.csv) is
`N01_VMEC_JACOBIAN_FEASIBILITY`:

- At three typical unstable anchors, compute derivatives of the candidate and
  constraint quantities with respect to VMEC boundary coefficients.
- Search both signed directions for geodesic/compression and the competing
  bad-curvature/compression candidate.
- Recompute a force-balanced equilibrium for every proposed move. Never edit a
  stored geometry channel independently.
- Call the feasibility check successful only if the candidate moves by at least
  **0.5 panel IQR** while every registered constraint drifts by at most
  **0.1 panel IQR**, in both signs at all three anchors.
- Budget **two Perlmutter node-hours** for planning and run **no GX**.

Before the equilibrium checks pass, this would be a plausibly-local
perturbation. A passing, force-balanced result would create an
equilibrium-consistent intervention candidate. It would still require the
separate N02 GX timing/convergence pilot before the 32.5-node-hour competing GX
test. The researcher must decide whether to authorize N01; S14 does not execute
it.

## Reproducibility and provenance

S14 is a synthesis of committed evidence only. It recomputed neither model nor
GX outputs and did not use `tests/data/review_slice.h5` for development,
selection, or reporting. The production run is
`synthesis-registered-evidence-s01-s13`. It read 21 committed source evidence
artifacts and indexed 18 registered production/correction runs from S00–S13,
plus the S03 publication-verification manifest that pins the corrected ladder
CSV by content hash.

The [S14 manifest](S14_artifacts/manifest.json) records exact command, config,
seed, Python and package versions, wall time (**0.768 s**), Git commit/tree,
output hashes, all source hashes, and these immutable fingerprints:

- external dataset SHA-256:
  `9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`;
- checkpoint SHA-256:
  `d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.

The manifest correctly records a dirty worktree: an unrelated pre-existing
tracked executive-summary edit and untracked user files were present. The S14
source files and every input artifact are separately content-hashed, so the
synthesis itself remains pinned. The external dataset and checkpoint were read
only; the checkpoint is fingerprinted for continuity but not loaded by S14.

The pilot command was:

```bash
.venv-xai/bin/python scripts/xai_s14_synthesis.py --pilot --no-publish
```

The production command was:

```bash
.venv-xai/bin/python scripts/xai_s14_synthesis.py \
  --config configs/xai/S14_synthesis.json \
  --output-dir output/xai/S14/synthesis-registered-evidence-s01-s13 \
  --published-dir reports/xai/S14_artifacts
```

The [reproducibility index](S14_artifacts/reproducibility_index.csv) contains 18
run manifests and one publication-verification record. It carries each run ID,
manifest path and SHA-256, exact command, dataset/checkpoint
fingerprints, Git state, row/member counts, and disclosed legacy caveats. Five
early manifests that previously existed only in ignored output directories are
copied verbatim into
[upstream_manifests](S14_artifacts/upstream_manifests/); later manifests remain
at their original committed paths. All 19 provenance hashes resolve in the
committed tree, and together they content-hash pin all 21 evidence artifacts.
Seventeen of the 18 run manifests are independently recreatable. The historical
`S03_PHASE` correction records its command and outputs but not its Git commit;
its index row therefore says `recreates_claims = False` rather than fabricating
provenance. The S03 publication-verification manifest independently pins those
corrected outputs.

## Verification

Tests were written before implementation. The first focused run failed in six
intended places while the synthesis validators were stubs. The final focused
suite passes 28 tests, including a cyclic toy ledger with a known relevant
feature and an explicit null control, exact source-row reproduction for all 64
evidence records, artifact hashes, numerical headline pins, and full manifest
resolution. A real pilot-run test checks finalization and `--no-publish`, and
the ledger rejects unknown validity/direction labels and intervals without a
traceable grouping unit. Fixture tests also require every evidence artifact to
have a manifest pin and require claim alignment/conjunct mappings to cover the
exact linked evidence set.

Five deliberate mutations were each confirmed to turn the suite red and were
then reverted:

1. made the claim gate read candidate-level `direction` instead of per-claim
   alignment; `test_headline_uses_claim_alignment_not_candidate_direction`
   failed;
2. classified a partial TCAV pass fraction as support instead of
   regime-dependent; `test_direction_is_derived_from_declared_source_rule`
   failed;
3. disabled the error for evidence artifacts with no manifest content-hash pin;
   `test_manifest_pin_guard_rejects_unpinned_evidence` failed.
4. disabled the `evidence_alignment` vocabulary whitelist;
   `test_unknown_claim_alignment_is_rejected` failed;
5. disabled the requirement that every declared candidate have a corroborating
   claim-aligned link;
   `test_every_declared_candidate_needs_claim_aligned_evidence` failed.

The earlier production round also turned red when the physical-intervention
guard and reproducibility-index manifest hash check were bypassed. The third
automated review found items 4 and 5 before those tests existed; both exact
mutations are now covered rather than being omitted from this record.

Final local commands and outcomes:

```text
.venv-xai/bin/python -m pytest tests/xai/test_synthesis.py \
  tests/xai/test_synthesis_script.py tests/xai/test_synthesis_artifacts.py -q
28 passed

source .venv-xai/bin/activate && make check
344 passed
```

## Interpretation limits

- All panel equilibria appeared in network training. Equilibrium-held-out fits
  test interpolation across this panel, not generalization to a new equilibrium
  family.
- Hidden and input perturbations are deliberately off-manifold network
  diagnostics. Their sensitivity explains the trained function, not what a
  force-balanced plasma would do.
- The S13 natural comparisons have severe confounding and poor overlap. Their
  adjusted numbers are prioritization evidence, not effect estimates licensed
  for intervention.
- Correlated concepts can exchange Shapley, probe, and distillation credit. The
  evidence supports feature families more strongly than unique scalar owners.
- Multiple ensemble members share data and architecture. Spread is neither a
  frequentist confidence interval nor a Bayesian posterior standard deviation.
- A readable model with $R^2\approx0.86$ is an approximation. Its residuals and
  the many poorly distilled hidden units retain unexplained computation.

## Acceptance criteria

1. **Every headline conclusion links to machine-readable evidence and at least
   two independent method families.** Qualified failure under the strict
   reading of “independent.” All **9/9** headlines have at least two
   claim-aligned method-family labels and committed machine-readable sources,
   but only **5/9** span at least two source steps and only **1/9** has two
   families corroborating the same declared conjunct. C05/C06 share S13, C08
   shares S11, and C09's counted evidence shares one S07 artifact. The
   [claim register](S14_artifacts/claim_register.csv) publishes per-conjunct
   families, distinct source steps/artifacts, and gate margins; record-level
   direction is not an acceptance input. Exact selectors and values are in the
   64-row [evidence ledger](S14_artifacts/evidence_ledger.csv). Whether
   differently named families within one run satisfy PLAN's intended
   independence is a researcher decision, not something S14 silently assumes.
2. **Every causal statement identifies its intervention.** Pass. The claim
   register contains zero physical causal statements because no physical
   intervention was executed. Every model diagnostic in the ledger records its named
   intervention and whether it ran; prospective N01 explicitly names the VMEC
   boundary-coefficient intervention and says it has not run.
3. **All runs can be recreated from manifests.** Qualified historical failure:
   **17/18** indexed run manifests contain the required recreation provenance.
   The `S03_PHASE` correction lacks a recorded Git commit, so its row is
   explicitly `recreates_claims = False`; its exact outputs remain content-hash
   pinned by the S03 publication-verification record. The index resolves and
   SHA-256-verifies all 19 provenance records, every one of the 21 evidence
   artifacts is pinned, and the S14
   [manifest](S14_artifacts/manifest.json) recreates this synthesis. The missing
   historical commit cannot be repaired honestly in S14.
4. **Deliverables.** Pass. This `FINAL_REPORT.md`, the compact reproducibility
   index, the 11-candidate evidence matrix, and the five-item prioritized
   next-experiment list are committed artifacts.

## Deferred

Nothing from S14. Upstream deferrals remain deferrals rather than being silently
filled in: most importantly, S12's PySR symbolic search and true mixed network
derivatives were not performed. The prospective VMEC and GX calculations are
future decision-gated experiments, not incomplete S14 work.

## Reviewer reproduction

**Recomputable on the slice.**

- S14 performs no new row-level model or GX calculation. Its evidence selectors
  can be checked directly against committed artifacts without using the slice.
- For upstream proxy checks, the S01 varied and fixed 1,000-row panels are the
  registered rows present in `tests/data/review_slice.h5`. Use
  `itg_nn.xai.review_slice.load_review_slice_index().slice_rows()` to map them;
  do not pass cohort row IDs directly to the slice reader. The reviewer can
  recompute native-output, symmetry, attribution-control, and registered
  panel-level proxy quantities described in the source reports.

**Checkable from committed artifacts alone.**

- Exact status counts (**11 candidates: 5 supported, 3 regime-dependent,
  2 contradicted, 1 unresolved**), 64 evidence records, nine headline claims,
  18 run manifests plus one publication-verification record, and five prioritized
  experiments.
- Every evidence selector and value, every source-derived direction rule, every
  per-claim alignment and conjunct, the minimum two-family count, all
  negative-result cells, source and output SHA-256 hashes, and immutable
  dataset/checkpoint fingerprints. Seventeen of 18 run manifests are marked
  independently recreatable; the S03 phase exception is explicit. The copied legacy manifests
  themselves are checkable as committed bytes; fidelity to their ignored source
  files is only partially corroborated, as described below.
- The headline values quoted in this report, using the evidence IDs in the
  claim register and `source_values_json` in the ledger.

**Not checkable off the researcher's machine, and why.**

- Re-execution of S01–S13 on the external 100,705-row HDF5 dataset or inspection
  of large ignored run arrays is unavailable on the GitHub runner. The closest
  proxy is exact recomputation on the registered 2,000-row review slice plus
  comparison with the committed selected-row values; agreement validates code
  path and row mapping, but not claims about rows outside the slice.
- Fidelity of the five copied legacy manifests to their original ignored files
  cannot be established off the researcher's machine. The S03 pair is
  independently pinned by `review3_manifest.json`; S01 and S02 each retain
  committed artifacts matching several recorded output hashes; S00 has no such
  independent committed corroboration. Those checks support the copies but do
  not prove their absent originals byte-for-byte.
- The prospective VMEC and GX interventions cannot be checked because they have
  not been authorized or run. The nearest current proxy is S13's committed
  feasibility specification and natural-comparison diagnostics. Agreement only
  verifies the selection logic; it cannot establish a physical causal effect.
