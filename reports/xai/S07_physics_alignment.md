# S07 — Learned spatial importance versus GX physics fields

## Executive result

The strongest learned-density comparison is real but is not enough to identify a
physical mechanism. In the varied-gradient unstable cohort, the top member's
dominant density, `2864601_0.437:u001`, has circular Spearman rank correlation
(agreement of within-tube ordering, from -1 to +1) **-0.361** with the held-out
GX heat-flux profile $Q(z)$, with equilibrium-bootstrap 95% interval
**[-0.388, -0.333]**, lag **+22/96**, and top-10% overlap **1.63 times chance**.
The geodesic-curvature candidate `u008` gives **-0.268**
**[-0.294, -0.241]** at lag **+44**. Both recur in the fixed-gradient panel at
similar strength and lag.

That is evidence that two internal activation patterns covary with physical GX
structure. It is not evidence that the network uses the positions where GX
transports heat. Integrated Gradients (an attribution method: it divides the
prediction change from a reference input among input cells) gives nearly zero
*signed* spatial association: **-0.021, -0.013, -0.012** in the three members.
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
  and 977. Stable and unstable strata are always separate table rows.
- **Density selection:** the three most important S04-ranked bottleneck units
  per member. This is a preregistered important-map comparison, not a census of
  every live unit.
- **Attributions:** S06b low-pass Integrated Gradients as the primary method,
  for both the original $f_m$ and canonical exactly invariant $\tilde f_m$.
  The periodic-mask map is a secondary diagnostic only: S06b found that its
  symmetry check failed, so its rows carry `feature_claims_permitted=False`.
- **Prediction estimand:** native `max(log Q, -2)`. Predictions are never
  exponentiated. GX `Q_avgs_vs_z` and `zonal_phi2_amplitudes` are physical
  comparison quantities, not alternate model outputs.
- **Validity:** all fixed/varied pairs and case studies are naturally observed
  comparisons. No off-manifold edit is used.

The production command was:

```bash
source .venv-xai/bin/activate
python scripts/xai_s07_physics_alignment.py \
  --config configs/xai/S07_physics_alignment.json
```

Run `physics-alignment-top3-panel1000` took 258.11 seconds on CPU. The committed
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

## Methods

### Spatial comparison

For each learned profile and GX $Q(z)$ profile, the code computes within-tube
Spearman correlation at every one of the 96 circular lags, chooses the largest
absolute mean correlation on the full fixed panel, and reports that lag without
realigning it away. The interval then holds that selected lag fixed. Lag
stability is a separate number: the fraction of bootstrap resamples whose
selected lag lies within four grid positions of the registered lag.

“Bootstrap” here means resampling to estimate uncertainty. Every bootstrap
resamples whole `equilibrium_files`, never individual rows, so related flux
tubes cannot masquerade as independent evidence. There are 500 deterministic
resamples. An association is called stable when its 95% interval excludes zero;
a lag is stable when at least half of resamples select within four positions.

Overlap uses tie-inclusive top-10% masks. Signed mode keeps both helpful and
opposing learned contributions. Positive-contribution mode clips both profiles
at zero before comparison; it answers a deliberately narrower question and is
not a substitute for the signed result.

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
their intervals resample whole equilibria. The study does not use the
off-manifold `-3` marker.

### Cases and symmetry control

For each of the two S05 hypotheses, the table keeps five naturally observed
supporting and five contradicting equilibria, selected at the registered S07
lag. Supporting means that the row-level sign agrees with the population sign;
it does not mean that GX validates a causal mechanism. A joint circular shift
of both compared profiles leaves the full lag curve unchanged to exactly zero
in the registered numerical check.

## Results

### 1. Density versus physical $Q(z)$

The two S05 candidates in the top member are reproducible across the two GX
drive panels:

| density | GX panel, unstable rows | circular $r_s$ (95% interval) | lag | lag recurrence within ±4 | overlap / chance |
|---|---|---:|---:|---:|---:|
| `.437:u001` bad-curvature / flux-compression candidate | varied | -0.361 [-0.388, -0.333] | +22 | 0.912 | 1.631 |
| `.437:u001` | fixed | -0.351 [-0.376, -0.328] | +21 | 0.842 | 1.624 |
| `.437:u008` geodesic-curvature candidate | varied | -0.268 [-0.294, -0.241] | +44 | 1.000 | 1.318 |
| `.437:u008` | fixed | -0.271 [-0.293, -0.250] | +42 | 1.000 | 1.325 |

Across all nine selected units, however, the varied-unstable signed results
range from **-0.361 to +0.134** and the lags range from **-47 to +48**. The next
members' strongest selected units are `.409:u014` at -0.210 and lag -26, and
`.371:u017` at +0.119 and lag +17. Thus neither a common sign nor a common
spatial offset identifies an ensemble mechanism. `spatial_alignment.csv`
contains all members, both drive panels, all/stable/unstable strata, both sign
modes, overlap, intervals, and lag stability. `lag_curves.csv` preserves all
96 registered-lag curves rather than only their maxima.

### 2. Attribution versus physical $Q(z)$

For canonical $\tilde f_m$, primary low-pass Integrated Gradients on varied
unstable rows gives:

| member | signed $r_s$ (95% interval), lag | positive-only $r_s$ (95% interval), lag | positive overlap / chance |
|---|---:|---:|---:|
| `.437` | -0.021 [-0.029, -0.013], -36 | +0.266 [0.249, 0.285], +1 | 2.390 |
| `.371` | -0.013 [-0.022, -0.004], +47 | +0.280 [0.260, 0.298], 0 | 2.414 |
| `.409` | -0.012 [-0.020, -0.003], +48 | +0.262 [0.243, 0.282], 0 | 2.439 |

This is the central contradiction. The positive portions occupy similar places
to positive $Q(z)$, but the complete signed evidence is marginal and its lags
do not agree. It would be misleading to describe the positive-only result as
“where the model looks” without that qualification.

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

For canonical low-pass Integrated Gradients, the signed cell sum has intervals
crossing zero in every member (-0.026, -0.032, -0.054 point estimates). Absolute
attribution mass is negatively associated in all three (-0.144, -0.159,
-0.131, intervals excluding zero), but absolute mass discards sign and measures
how much the model responds, not whether that response raises heat flux.

### 4. Fixed/varied natural pairs

Observed fixed-minus-varied clipped-log GX flux is **+1.559**
**[+1.434, +1.694]**. All six original/canonical member prediction effects lie
between +1.534 and +1.550 and have intervals excluding zero. This is a check
that the models track the large observed panel difference in their native
estimand, not a controlled estimate of drive causality.

The two physical $Q(z)$ profiles for the same geometry remain related:
signed circular $r_s=0.736$ **[0.710, 0.760]**, lag 0, overlap 0.588, and lag
recurrence 1.000. Even against that strong physical pairing, canonical signed
attribution's fixed-minus-varied alignment changes are mixed across members
(-0.003, +0.023, -0.021), whereas the positive-only changes are consistently
positive (+0.073, +0.078, +0.053). This again says the apparent replication is
specific to discarding negative contributions.

### 5. Supporting and contradicting cases

[case_studies.csv](S07_artifacts/case_studies.csv) contains 20 distinct
equilibria: 5 supporting and 5 contradicting the population sign for each of
the bad-curvature/flux-compression and geodesic-curvature hypotheses. The
contradicting examples are not weaker leftovers: for `.437:u001`, the five
supporting row correlations reach -0.953 to -0.933 while the strongest
contradiction is +0.864. The paired [case figure](S07_artifacts/case_studies.png)
shows both directions with equal space.

## Failed checks, negative results, and interpretation limits

- The signed attribution/Q(z) association is close to zero even though the
  activation-density/Q(z) association is larger. Activation is present in the
  network; attribution asks whether changing the input from a stated reference
  uses that location to change the prediction. They are not interchangeable.
- Signs and selected lags do not replicate across members. Member-level signed
  results are preserved before any summary, as required.
- The positive-contribution result is stable but incomplete because it clips
  negative evidence. It supports a spatial resemblance, not the full signed
  prediction mechanism.
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
   intervals exclude zero. Unresolved and invalid-method rows remain present.
2. **Spatial lag is reported.** Yes. The headline density lags are +22 and +44;
   signed canonical attribution lags are -36, +47, and +48. The complete 96-lag
   curves are in `lag_curves.csv`, and lag recurrence is separate from
   association stability.
3. **Prediction attribution and physical $Q(z)$ are explicitly distinct.** Yes.
   Every spatial row names `source_family`, `method`, `function`, and
   `gx_quantity`, and includes the distinction text. The report never treats
   GX $Q(z)$ as a model attribution or an input to the network.

## Artifacts

- [spatial_alignment.csv](S07_artifacts/spatial_alignment.csv): 216 comparisons
  with member, function, method, drive panel, stability stratum, sign mode,
  uncertainty, lag, overlap, validity, and claim-permission columns.
- [lag_curves.csv](S07_artifacts/lag_curves.csv): 20,736 rows preserving every
  circular-lag correlation.
- [zonal_association.csv](S07_artifacts/zonal_association.csv): 306 grouped
  scalar associations.
- [paired_analysis.csv](S07_artifacts/paired_analysis.csv): 46 physical,
  prediction, and learned-signal fixed/varied comparisons.
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
  (-0.360511998599993, lag +22), signed canonical `.437` Integrated Gradients
  (-0.02122218224703246, lag -36), positive-only canonical `.437` Integrated
  Gradients (+0.2657020267671119, lag +1), the zonal candidate, physical pair,
  balanced contradictions, and the zero-error shift control.

### Checkable from committed artifacts alone

- All headline numbers above are literal rows in the committed CSVs and summary.
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
  1,000-row bootstrap intervals.
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
