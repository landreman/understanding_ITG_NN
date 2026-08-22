# S07 — Learned spatial importance versus GX physics fields

## Executive result

The strongest learned-density comparison is real but is not enough to identify a
physical mechanism. In the varied-gradient unstable cohort, the top member's
dominant density, `2864601_0.437:u001`, has circular Spearman rank correlation
(agreement of within-tube ordering, from -1 to +1) **-0.361** with the held-out
GX heat-flux profile $Q(z)$, with equilibrium-bootstrap 95% interval
**[-0.388, -0.333]** and lag **+22/96**. Because the association is negative,
the **1.63-times-chance** overlap is computed after sign-flipping $Q(z)$: it is
overlap with *low*, not high, heat-flux regions. The unflipped high-density /
high-$Q(z)$ overlap is only **0.542 times chance**.
The geodesic-curvature candidate `u008` gives **-0.268**
**[-0.294, -0.241]** at lag **+44**. The same geometry-derived densities give
similar comparisons in the fixed-gradient panel, but that is not independent
replication: only the GX field and drive panel change.

That is evidence that two internal activation patterns covary with physical GX
structure. It is not evidence that the network uses the positions where GX
transports heat. Integrated Gradients (an attribution method: it divides the
prediction change from a reference input among input cells) gives nearly zero
*signed* spatial association: **-0.021, -0.013, -0.012** in the three members.
These attribution paths are deliberately off-manifold diagnostics: their
low-pass reference geometries are not asserted to be valid plasma equilibria,
so they explain network extrapolation from that reference, not plasma causality.
Keeping only positive contributions raises the associations to
**0.266, 0.280, 0.262** at lag 0 or +1, but that one-sided view discards
opposing evidence. The member-specific density signs and lags also fail to
replicate. S07 therefore promotes no candidate to “physically supported.”

The network attribution and GX $Q(z)$ remain explicitly different objects
throughout every table: the former explains a model prediction; the latter is a
held-out physical diagnostic that the network never received as an input or
training target at each position.

## Scope and registered estimand

- **Members:** the S01 top three, `2864601_0.437`, `2864601_0.371`, and
  `2864601_0.409`. Results remain member-level; no ensemble averaging is used.
- **Rows:** the 1,000-equilibrium S01 varied-gradient panel and the same
  geometries' 1,000 fixed-gradient simulations. The varied cohort contains 240
  stable/near-floor rows and 760 unstable rows; the fixed cohort contains 23
  and 977. Spatial and zonal tables separate those strata. Paired tables report
  all 1,000 rows plus 251 pairs where either target is stable/near-floor and 749
  pairs where both targets are unstable.
- **Density selection:** the three most important S04-ranked bottleneck units
  per member. This is a preregistered important-map comparison, not a census of
  every live unit.
- **Attributions:** S06b low-pass Integrated Gradients as the primary method,
  for both the original $f_m$ and canonical exactly invariant $\tilde f_m$.
  Every attribution row retains S06b's
  `deliberately_off_manifold_diagnostic` validity tag because the reference path
  is not guaranteed to remain on the manifold of valid equilibria.
  The periodic-mask map is a secondary diagnostic only: S06b found that its
  symmetry check failed, so its rows carry `feature_claims_permitted=False`.
- **Prediction estimand:** native `max(log Q, -2)`. Predictions are never
  exponentiated. GX `Q_avgs_vs_z` and `zonal_phi2_amplitudes` are physical
  comparison quantities, not alternate model outputs.
- **Validity:** fixed/varied GX rows, density comparisons, and case studies are
  naturally observed comparisons. Attribution sources retain the deliberately
  off-manifold S06b reference-path tag even when compared across observed rows.
  `feature_claims_permitted` retains S06b's technical meaning (the explanation
  method passed its own checks in an unstable stratum). The stricter
  `plasma_claims_permitted` additionally requires an observed/on-manifold source;
  it is true for observed GX quantities and observed density/case comparisons,
  and false for every off-manifold attribution row and member-prediction row.

The production command was:

```bash
source .venv-xai/bin/activate
python scripts/xai_s07_physics_alignment.py \
  --config configs/xai/S07_physics_alignment.json
```

Run `physics-alignment-top3-panel1000` took 302.96 seconds on CPU. The committed
[manifest](S07_artifacts/manifest.json) records the exact dataset, checkpoint,
S04/S05 inputs, reused S06b attribution-map hash, package versions, command,
row IDs, and output hashes.

Tests were written before the implementation. Three temporary mutations showed
that they can fail, and each was reverted before the production checks:

1. forcing every selected lag to zero failed 2 focused tests;
2. resampling individual rows instead of whole equilibria failed the explicit
   grouped-bootstrap draw test; and
3. exponentiating predictions before the paired difference failed the native-
   estimand test.

Automated review then exposed two missing test branches. Deleting the
negative-association overlap orientation and changing the tie-inclusive mask
from `>=` to `>` had initially remained green. Inverse-association and tied-mask
fixtures were added; both mutations now turn the focused suite red. A later
review found that replacing average ranks for tied values with arbitrary
ordinal ranks also remained green. An analytic tied-profile fixture now pins
mid-ranks and makes that mutation fail. A mixed active/silent fixture also
separates the active-row association from the pooled value and makes replacing
the former with the latter fail. Its unequal 53-position learned and 10-position
GX mean mask widths pin both overlap normalisations: swapping either the
overlap denominator or the chance baseline now fails.

## Methods

### Spatial comparison

For each learned profile and GX $Q(z)$ profile, the code computes within-tube
Spearman correlation at every one of the 96 circular lags, chooses the largest
absolute mean correlation within that same reported stratum, and reports that
lag without realigning it away. This is in-sample lag selection. The interval
then holds that selected lag fixed. Lag
stability is a separate number: the fraction of bootstrap resamples whose
selected lag lies within four grid positions of the registered lag.

“Bootstrap” here means resampling to estimate uncertainty. Every bootstrap
resamples whole `equilibrium_files`, never individual rows, so related flux
tubes cannot masquerade as independent evidence. There are 500 deterministic
resamples. An association is called stable when its 95% interval excludes zero;
a lag is stable when at least half of resamples select within four positions.
This panel already has one row per equilibrium, so grouped and row resampling
are numerically identical here; preserving `equilibrium_files` grouping keeps
the estimator correct for any later panel containing sibling tubes.

Because selecting the largest absolute value over 96 lags can manufacture a
small peak, every unstable-row comparison also has 200 permutations that break
the learned-profile/GX equilibrium pairing and repeat the full 96-lag maximum.
`selection_null_q95` is the 95th percentile of those maxima. The fixed-lag
bootstrap interval and the lag-search null answer different questions, and both
are reported. `lag_search_null_resolved` applies this 5% threshold separately
to each comparison; across 72 unstable-row comparisons, a handful of borderline
passes are expected and the flag is not a family-wise guarantee.

Overlap uses tie-inclusive top-10% masks. For a negative signed association, the
reported signed overlap sign-flips the aligned GX profile so that it measures
high learned values against *low* $Q(z)$; `overlap_orientation` makes that
operation explicit. The unflipped positive-mode row provides the high/high
comparison for nonnegative densities. Positive-contribution mode clips both
profiles at zero; it answers a deliberately narrower question and is not a
substitute for the signed result. A learned profile that is constant along all
96 positions has no spatial ordering: by convention it contributes rank
correlation zero at every lag, while the tie-inclusive mask contains all 96
positions. Each artifact row therefore records the constant-profile count and
fraction, the association over nonconstant rows alone, and the mean learned and
GX mask widths; the primary association remains the preregistered all-row mean.

### Zonal-flow comparison

The sample-level mean density and four attribution summaries are rank-correlated
with `log10(zonal_phi2_amplitudes)`. This is an across-equilibrium association,
not a claim that either learned signal causes zonal flows. Stable/near-floor and
unstable simulations remain separate.

### Natural paired comparison

The fixed panel holds $(a/L_T,a/L_n)=(3,0.9)$ across its geometries. Each fixed
row is paired to the same geometry's varied-gradient row, whose drive may be
different. The pairing therefore controls geometry, **not** drive within the
pair. Fixed-minus-varied effects use the native clipped-log prediction, and
their intervals resample whole equilibria. Each result is reported for all
pairs, pairs where either physical target is stable/near-floor, and pairs where
both are unstable. The study does not use the off-manifold `-3` marker; that is
separate from the deliberately off-manifold reference path inherited by the
attribution source.

### Cases and symmetry control

For each of the two S05 hypotheses, the table keeps five naturally observed
supporting and five contradicting equilibria, selected at the registered S07
lag. Supporting means that the row-level sign agrees with the population sign;
it does not mean that GX validates a causal mechanism. A joint circular shift
of two already-computed profiles leaves the full lag curve unchanged to exactly
zero by construction. This checks the comparison statistic only; the end-to-end
model and explanation symmetry evidence remains in S05 and S06b.

## Results

### 1. Density versus physical $Q(z)$

The two S05 candidates in the top member remain associated across the two GX
drive panels:

| density | GX panel, unstable rows | circular $r_s$ (95% interval) | lag | permutation-null q95 | overlap high/high; sign-oriented |
|---|---|---:|---:|---:|---:|
| `.437:u001` bad-curvature / flux-compression candidate | varied | -0.361 [-0.388, -0.333] | +22 | 0.043 | 0.542; 1.631 |
| `.437:u001` | fixed | -0.351 [-0.376, -0.328] | +21 | 0.039 | 0.542; 1.624 |
| `.437:u008` geodesic-curvature candidate | varied | -0.268 [-0.294, -0.241] | +44 | 0.041 | 0.742; 1.318 |
| `.437:u008` | fixed | -0.271 [-0.293, -0.250] | +42 | 0.035 | 0.745; 1.325 |

The first overlap number compares actual high-density with high-$Q(z)$ regions;
values below one mean avoidance relative to chance. The second first flips
$Q(z)$ because $r_s<0$ and therefore measures high density against low $Q(z)$.
Both candidate correlations are far above their complete lag-search nulls.
The dominant candidate is constant on 17/760 rows and has an active-row-only
correlation of -0.369; the geodesic candidate is constant on 45/760 rows and
has an active-row-only correlation of -0.285. Thus their headline associations
are not driven by the constant-row convention. Their varied-panel lags are also
stable: +22 recurs within four positions in 91.2% of resamples, and +44 in
100.0% (the fixed-panel +21 and +42 lags recur within four positions in 84.2%
and 100.0%).

Only `.437:u001` and `.437:u008` have S05 `supported_named_motif` status. Four
other selected units in `.437`/`.371` were already unresolved in S05, and the
three `.409` units were not characterised there at all. Conditional on this
deliberately importance-ranked but mostly unnamed set, the varied-unstable signed results
range from **-0.361 to +0.134** and the lags range from **-47 to +48**. The next
members' strongest selected units are `.409:u014` at -0.210 and lag -26, and
`.371:u017` at +0.119 and lag +17. Thus this S07 selection supplies no
cross-member common sign or spatial offset; it is not a clean replication test
of the two S05 names. `spatial_alignment.csv`
contains all members, both drive panels, all/stable/unstable strata, both sign
modes, overlap, intervals, and lag stability. `lag_curves.csv` preserves all
96 registered-lag curves rather than only their maxima.

Three of the nine selected densities are constant on many unstable rows:
`.437:u003` on 605/760 (79.6%), `.371:u004` on 620/760 (81.6%), and
`.371:u019` on 316/760 (41.6%). Their all-row correlations are respectively
+0.034, +0.034, and -0.114, versus +0.168, +0.182, and -0.195 over only rows
where the unit varies. Their tie-inclusive learned masks average 80.8, 84.1,
and 52.0 positions rather than the nominal 10. This changes the meaning of
those per-unit magnitudes but not the negative cross-member conclusion: the
active-row range remains mixed, from -0.369 to +0.182, with no common lag.

### 2. Attribution versus physical $Q(z)$

For canonical $\tilde f_m$, primary low-pass Integrated Gradients on varied
unstable rows gives the following deliberately off-manifold network diagnostic:

| member | signed $r_s$ (95% interval), lag; null q95 | positive-only $r_s$ (95% interval), lag; null q95 | positive overlap / chance |
|---|---:|---:|---:|
| `.437` | -0.021 [-0.029, -0.013], -36 (82.2% within ±4); 0.013 | +0.266 [0.249, 0.285], +1; 0.035 | 2.390 |
| `.371` | -0.013 [-0.022, -0.004], +47 (63.4% within ±4); 0.013 | +0.280 [0.260, 0.298], 0; 0.038 | 2.414 |
| `.409` | -0.012 [-0.020, -0.003], +48 (**unstable**, 31.2% within ±4); 0.013 | +0.262 [0.243, 0.282], 0; 0.037 | 2.439 |

This is the central contradiction. The `.371` signed maximum lies at the
estimated lag-search threshold (0.012764 versus q95 0.012817, too close for a
stable binary verdict), while `.409` is below it; `.437` exceeds it, but its
magnitude is only 0.021. The
positive portions exceed their nulls by factors of 7.0–7.7 and occupy similar
places to positive $Q(z)$, but the complete signed evidence is marginal and its
lags do not agree. Because the reference path is deliberately off-manifold, even
the positive-only result describes network sensitivity to that constructed
path, not a plasma mechanism. It would be misleading to call it simply “where
the model looks.”

The periodic-mask diagnostic is stronger (positive-only $r_s=0.318$–0.345),
but it remains in the tables as a negative methodological result, not as feature
evidence, because its S06b shift-symmetry check failed. Likewise, stable-row
comparisons are retained but carry `feature_claims_permitted=False`: at the
clipped-log floor, the model's native output cannot resolve changes below -2.

### 3. Zonal-flow observable

The proposed geodesic-curvature density `.437:u008` is associated with
`log10(zonal_phi2_amplitudes)` on varied unstable rows at **-0.122**
**[-0.183, -0.060]**, but becomes **-0.513** **[-0.564, -0.461]** on fixed
unstable rows. The bad-curvature candidate `.437:u001` is unresolved on varied
rows, **+0.032 [-0.040, +0.099]**, and positive on fixed rows,
**+0.310 [+0.247, +0.372]**. The large drive dependence prevents a simple
geometry-only interpretation.

For deliberately off-manifold canonical low-pass Integrated Gradients, the signed cell sum has intervals
crossing zero in every member (-0.026, -0.032, -0.054 point estimates). Absolute
attribution mass is negatively associated in all three (-0.144, -0.159,
-0.131, intervals excluding zero), but absolute mass discards sign and measures
how much the model responds, not whether that response raises heat flux.

### 4. Fixed/varied natural pairs

Across all rows, observed fixed-minus-varied clipped-log GX flux is **+1.559**
**[+1.434, +1.694]**; it is **+3.356** **[+3.122, +3.563]** when either member
of the pair is stable/near-floor and **+0.957** **[+0.826, +1.076]** when both
are unstable. All six original/canonical member prediction effects lie
between +1.534 and +1.550 and have intervals excluding zero. This is a check
that the models track the large observed panel difference in their native
estimand, not a controlled estimate of drive causality.

The two physical $Q(z)$ profiles for the same geometry remain related. Across
all rows, signed circular $r_s=0.736$ **[0.710, 0.760]**, lag 0, and overlap
0.588. On the 749 both-unstable pairs it strengthens to **0.874**
**[0.860, 0.888]**, lag 0, and overlap 0.686.

On those both-unstable pairs, deliberately off-manifold canonical attribution's
fixed-minus-varied signed alignment changes remain mixed across members
(+0.006, +0.026, -0.021), while positive-only changes are consistently positive
(+0.034, +0.034, +0.015). The positive-only contrast remains, but it is smaller
than in the pooled table and cannot be promoted to plasma evidence.

### 5. Supporting and contradicting cases

[case_studies.csv](S07_artifacts/case_studies.csv) contains 20 rows from 19
distinct equilibria: 5 supporting and 5 contradicting the population sign for
each of the bad-curvature/flux-compression and geodesic-curvature hypotheses.
One flux tube contradicts both hypotheses. The
contradicting examples are not weaker leftovers: for `.437:u001`, the five
supporting row correlations reach -0.953 to -0.933 while the strongest
contradiction is +0.864. The paired [case figure](S07_artifacts/case_studies.png)
shows both directions with equal space.

## Failed checks, negative results, and interpretation limits

- The signed attribution/Q(z) association is close to zero even though the
  activation-density/Q(z) association is larger. Activation is present in the
  network; attribution asks whether changing the input from a stated reference
  uses that location to change the prediction. They are not interchangeable.
- One signed attribution maximum is below the 200-permutation 96-lag selection
  null, one lies at its estimated threshold, and the remaining 0.021 maximum is
  resolved but negligible.
- Signs and selected lags do not replicate across members. Member-level signed
  results are preserved before any summary, as required.
- The `.409` signed canonical attribution's selected +48 lag fails the stated
  lag-stability rule: only 31.2% of resamples return within four positions.
- Three of nine selected densities are silent on 42–82% of unstable rows, so
  their pooled magnitudes partly reflect the documented zero-correlation
  convention. The active-row range remains mixed, from -0.369 to +0.182.
- The positive-contribution result is stable but incomplete because it clips
  negative evidence. It supports a spatial resemblance, not the full signed
  prediction mechanism.
- Every attribution result is deliberately off-manifold because its S06b
  reference path is not guaranteed to represent valid equilibria.
- The periodic-mask result may not support a feature claim because its upstream
  symmetry check failed.
- The geodesic-density/zonal association changes greatly between drive panels.
- Equal-prominence natural contradictions exist for both hypotheses.
- Circular correlation, even with grouped uncertainty and correct lag, is
  association rather than identity or causality. $Q(z)$ is itself a GX
  diagnostic, not a perturbation showing which geometry cell controls flux.

## Acceptance criteria

1. **Associations are equilibrium-bootstrap stable.** Yes, where claimed.
   Every interval in `spatial_alignment.csv`, `zonal_association.csv`, and
   `paired_analysis.csv` uses 500 resamples of `equilibrium_files`. The headline
   `.437:u001` interval is [-0.388, -0.333]; all three canonical positive-only
   varied-unstable attribution intervals and the two reported candidate-density
   intervals exclude zero and exceed their complete 96-lag permutation-null
   q95. Unresolved and invalid-method rows remain present.
2. **Spatial lag is reported.** Yes. The headline density lags are +22 (91.2%
   of resamples within ±4) and +44 (100.0%); signed canonical attribution lags
   are -36 (82.2%), +47 (63.4%), and +48 (31.2%, failing the registered 50%
   lag-stability rule). Overall, 161/216 spatial rows pass that rule. The
   complete 96-lag curves are in `lag_curves.csv`; 200-permutation lag-search
   q95 values are in `spatial_alignment.csv`; and lag recurrence is separate
   from fixed-lag bootstrap stability and search-null resolution.
3. **Prediction attribution and physical $Q(z)$ are explicitly distinct.** Yes.
   Every spatial row names `source_family`, `method`, `function`, and
   `gx_quantity`, and includes the distinction text. The report never treats
   GX $Q(z)$ as a model attribution or an input to the network.

## Artifacts

- [spatial_alignment.csv](S07_artifacts/spatial_alignment.csv): 216 comparisons
  with member, function, method, drive panel, stability stratum, sign mode,
  uncertainty, lag, overlap orientation, constant-profile and mask-width
  diagnostics, permutation-null calibration, validity, and claim-permission
  columns.
- [lag_curves.csv](S07_artifacts/lag_curves.csv): 20,736 rows preserving every
  circular-lag correlation.
- [zonal_association.csv](S07_artifacts/zonal_association.csv): 306 grouped
  scalar associations.
- [paired_analysis.csv](S07_artifacts/paired_analysis.csv): 138 physical,
  prediction, and learned-signal fixed/varied comparisons (46 quantities times
  all/either-near-floor/both-unstable strata).
- [case_studies.csv](S07_artifacts/case_studies.csv): 20 balanced natural cases.
- [physics_alignment_atlas.png](S07_artifacts/physics_alignment_atlas.png) and
  [case_studies.png](S07_artifacts/case_studies.png): compact visual summaries.
- [summary.json](S07_artifacts/summary.json) and
  [manifest.json](S07_artifacts/manifest.json): registered conclusions and full
  provenance.
- `output/xai/S07/physics-alignment-top3-panel1000/alignment_details.h5`: large,
  git-ignored member/sample/unit/position arrays, with committed hash in the
  manifest.

## Reviewer reproduction

### Recomputable on the slice

All 1,000 S01 parent row IDs used here occur in `tests/data/review_slice.h5`.
Map the manifest's parent IDs with
`itg_nn.xai.review_slice.load_review_slice_index().slice_rows(parent_rows)`;
never send the parent IDs directly to a reader pointed at the slice.

- The nine density-versus-$Q(z)$ rows, zonal associations, member prediction
  effects, physical fixed/varied $Q(z)$ correlation, and symmetry control can be
  recomputed exactly from those mapped rows plus the committed checkpoint and
  config.
- The exact headline table rows are pinned by
  `tests/xai/test_physics_alignment_artifacts.py`, including `.437:u001`
  (-0.360511998599993, lag +22, lag-search null q95 0.0432162), signed canonical `.437` Integrated Gradients
  (-0.02122218224703246, lag -36), positive-only canonical `.437` Integrated
  Gradients (+0.2657020267671119, lag +1), the zonal candidate, physical pair,
  balanced contradictions, and the zero-error shift control.
- The same artifact test pins the dominant density's 17/760 constant rows and
  -0.368761 active-row association, and the mostly silent `.437:u003` density's
  605/760 constant rows, +0.167794 active-row association, and 80.847-position
  mean learned-mask width. A separate analytic test pins mid-ranks for tied
  values and the zero-correlation convention for a constant row.

### Checkable from committed artifacts alone

- All headline numbers above are literal rows in the committed CSVs and summary.
- Every unstable-row permutation-null distribution is reduced to its committed
  q95 and maximum. The density nulls can be recomputed on the slice; the exact
  attribution null reductions are checkable from the committed artifact.
- The manifest hashes every committed artifact, records the dataset and
  checkpoint hashes, keeps the 2,000 gradient-set-tagged row IDs, and identifies
  the exact S06b map (`ab848646...5b8930b`).
- Counts, member/function/stratum axes, validity tags, member-level retention,
  native estimand, case balance, and source distinction are covered by the
  artifact tests without opening the external dataset.

### Not checkable off the researcher's machine, and why

- The exact 304 MB S06b full-panel attribution map is git-ignored. Recomputing
  every S07 attribution number from scratch therefore requires the external
  dataset and the S06b production run. The nearest committed proxy is S06b's
  selected review-map artifact on 16 mapped rows; agreement there checks map
  loading, axes, function/method identity, and sign handling but not the exact
  1,000-row bootstrap intervals or underlying permutation draws. Their q95 and
  maxima are committed in `spatial_alignment.csv`.
- `alignment_details.h5` is git-ignored (about 3.1 MB). Its hash and all headline
  reductions are committed. If the local file is present, the artifact test also
  verifies its member/sample/unit/position axes.
- The external 678 MB HDF5 source cannot be opened on the GitHub runner. The
  review slice is the exact proxy for this registered panel; it is sufficient
  for density, prediction, GX, zonal, and pairing calculations after parent-to-
  slice row mapping.

## Deferred

- An all-live-unit density census. S07 used the preregistered top three
  S04-important units in each of the required top three members; extending the
  search after seeing these outcomes would enlarge the multiple-comparison
  problem and is not needed for the MVD.
- New perturbative GX simulations. Those would be required to turn an observed
  spatial association into a causal plasma statement and are outside S07's
  one-session budget.
