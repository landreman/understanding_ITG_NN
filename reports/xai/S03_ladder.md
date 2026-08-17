# S03 — The structure-destroying counterfactual ladder

## Status and headline result

Complete, including both external-review rounds. Every retained result explains the
trained native output `max(log Q, -2)`, never `Q` or `exp(prediction)`. The
primary estimand is S02's exactly shift-invariant canonical member function
`tilde_f_m = MLP_m(mean_z rho_m, a/L_T, a/L_n)`; the original `f_m` is retained
and compared. Member-level signed predictions and changes are preserved before
aggregation.

**Fixed-gradient results are withdrawn.** The shared loader supplies `a/L_T=-3`,
but the serialized training tensors contain nonnegative `a/L_T` only, and direct
inference shows that `-3` saturates the members at the clipped-log floor whereas
`+3` recovers R² of 0.978–0.985 for the tested top three. This registered-premise
conflict and the required cross-step correction are documented in
[`S03_fixed_gradient_decision.md`](S03_fixed_gradient_decision.md). Every number
interpreted below is from the 1,000-row varied-gradient panel.

The networks use substantially more than the multiset of pointwise
seven-channel vectors. On the frozen 1,000-row varied panel, random joint
permutation changes canonical top-10-member outputs by a median **3.26 panel
member residual standard deviations** (10th–90th member range 3.00–3.54).
Independent channel shifts give 2.41 (2.23–2.59). These edits have similar robust
input displacement, so the result supports sensitivity to both parallel order
and cross-channel co-location, but the off-manifold probes do not form an
additive decomposition.

The corrected common-phase operator rotates each channel's existing Fourier
coefficient by the same random phase and therefore preserves cross-channel phase
differences. Across three replicates, common-phase versus per-channel effects are
2.801/2.984, 2.652/3.114, and 2.672/3.491 panel residual SD. The paired
per-member channel-minus-common median is positive in every replicate
(0.402/0.617/0.584; 8/10, 8/10, and 9/10 members positive). This is consistent
directional evidence that cross-channel phase alignment contributes to the
ordering signal, but independent operator realizations and the one or two member
exceptions preclude a component-size interpretation. The earlier negative
conclusion and the incorrect 4.47-versus-4.05 comparison are withdrawn.

The ignored corrected production run is
`output/xai/S03/ladder-top10-all100-varied1000-paired2000-reviewfix/`. It took
2,809 s (46.8 minutes) on CPU for the uncached inference pass. The final
post-commit cached validation pass refreshed the manifest from a clean tracked
tree without rerunning inference. The manifest hashes the full signed prediction
tensor, full member/function/stratum table, generated compact table, support
paths, figures, operator endpoints, protected checkpoint, and external dataset.
The committed generated headline table is
[`S03_artifacts/ladder_summary.csv`](S03_artifacts/ladder_summary.csv), with
machine-readable checks in [`S03_artifacts/summary.json`](S03_artifacts/summary.json).
Review-derived, source-hash-verified quantities are in
[`S03_artifacts/review2_summary.json`](S03_artifacts/review2_summary.json), and
registered paired control contrasts are in
[`S03_artifacts/contrasts.csv`](S03_artifacts/contrasts.csv).

## Registered cohort, functions, normalization, and uncertainty

The production cohort was the frozen S01 panel: 1,000 varied-gradient rows from
1,000 distinct `equilibrium_files` plus their 1,000 fixed-gradient twins. The
fixed outputs are retained only as an audit trail and are not interpreted. No
row or member was selected after seeing S03 results. The complete ladder was run for
the stored-validation top 10; 13 cheap entries were run for all 100 registered
members. Random operators have three fixed-seed replicates for the top 10, with
replicate zero covering all 100.

For varied rows, RMS output change is divided by the same member's residual
standard deviation measured on this same frozen panel and matching
function/stable/unstable stratum. The prior report divided panel numerators by
S02 full-reference denominators. Panel/S02 denominator ratios across the top 10
are 1.33–1.43 (median 1.39) overall, 1.65–1.84 (median 1.77) for stable rows, and
1.23–1.32 (median 1.29) for unstable rows, so all old normalized magnitudes are
withdrawn. The full table retains S02's denominator in a comparison-only column.
Fixed-row numerical interpretations are withdrawn for the separate
sign-convention reason above.

Top-10 intervals use 1,000 resamples of whole `equilibrium_files`, never flux
tubes. The panel contains one row per equilibrium in each gradient set. Each
bootstrap draw recomputes both the perturbation RMS numerator and the panel
residual-standard-deviation denominator, so denominator uncertainty is included.
Because RMS is nonnegative by construction, whether its interval excludes zero
is not an informative test. Instead, signed paired member contrasts compare each
operator to its registered control. Joint permutation exceeds independent shift
for all 10 members in all three replicates; paired difference medians are 0.883,
0.910, and 0.938 panel residual SD. Joint permutation and independent shift also
exceed the random-joint-shift control for 10/10 members in every replicate. Full
quantiles are retained in `contrasts.csv`.

Every edit also reports an input displacement after division by S01's registered
per-channel IQR/1.349 scales. The scalar dose is RMS over edited channels and
positions; single-channel replacement uses only the edited channel. It is
comparable within the Fourier attenuation family and within the channel-
replacement family, not across operator families: an exact cyclic shift can
have a larger raw RMS dose than a destructive edit because this metric does not
align cyclic phase first.

## Perturbation and baseline API

`itg_nn.xai.perturbations` provides validity-tagged deterministic operators and
the reusable baseline family required downstream:

- per-channel robust constant profiles using the pooled reference median;
- observed backgrounds matched on `(a/L_T, a/L_n)` within equilibrium class;
- nearest-neighbour and observed medoid backgrounds;
- input-specific periodic low-pass backgrounds;
- equilibrium-class/gradient-conditional channel profiles;
- hard wrapped windows with no truncation at index 0, tied to grid scales and
  every member's S02 receptive fields; and
- linear endpoint paths with support diagnostics at doses 0, 0.25, 0.5, 0.75,
  and 1.

An all-zero geometry is explicitly forbidden as a default. Every ladder row is
tagged `exact_symmetry`, `observed_comparison`,
`plausibly_local_not_guaranteed_physical`, or
`deliberately_off_manifold_diagnostic`. Structure-destroying edits and
single-channel replacements are deliberately off manifold: they explain the
network, not the plasma.

The production ladder exercises the exact-symmetry and deliberately-off-manifold
tags. The observed-comparison and plausibly-local tags belong to the reusable
baseline/window API required by task 1; PLAN task 3 does not register a network
intervention using them. Likewise, the wrapped-window acceptance test verifies
the API's periodic support contract, not a claim about a windowed member result.

Block permutations use a seeded random cyclic origin per sample, permute
contiguous blocks, and roll back. Identity and every cyclic rotation of the
block order are rejected, so every endpoint actually destroys block order
rather than reducing to an exact joint shift. Common phase scrambling multiplies
the original spectrum by a shared random unit complex number at each non-DC,
non-Nyquist frequency; independent scrambling replaces each channel's phase
separately. Both preserve each marginal amplitude spectrum, while only the
common rotation preserves relative cross-channel phase.
All seeded random operators now transform the 1,000 unique geometries once and
tile the endpoints, so registered varied/fixed twins receive bit-identical
realizations. The earlier production varied endpoints remain unchanged; only the
now-withdrawn fixed endpoints differed. On the same 64-row registered pilot,
every varied prediction was bit-identical before and after this change (maximum
absolute difference 0), while the fixed predictions changed as expected.

## Exact and near-exact controls

All exact controls pass S02's `atol=rtol=2e-5` standard. Across production,
maximum absolute errors are 9.54e-6 for original `f` under shift 32, 7.63e-6
for `tilde_f` under shift 32, and at most 7.63e-6 for `tilde_f` under arbitrary
per-sample joint shifts. Normalized top-10 canonical shift RMS is below 9e-7.
The corresponding arbitrary-shift effect for original `f` is **0.394 panel
residual SD** (top-10 median), a direct equivariance-error budget for S06. Thus a
position-resolved explanation of uncanonicalized `f` would inherit contamination
comparable to roughly two-fifths of the model's own panel error.

The near-exact stellarator parity changes canonical outputs by median 0.138
panel residual SD, whereas the matched wrong-parity reversal changes them by
1.26. The all-100 medians are 0.126 and 1.28, so this is not a top-10 accident.
Parity is not called a numerical null: S02 showed that channels 3 and 5 obey it
only approximately in observed data.

The larger structure tests agree across functions. Original `f` gives medians
3.16 for joint permutation and 2.35 for independent shifts, versus 3.26 and
2.41 for `tilde_f`. Corrected common/per-channel phase values are 2.85/3.04 for
`f` and 2.80/2.98 for `tilde_f`.

## Ordering, co-location, and length scale

For canonical `tilde_f` on varied rows, replicate-zero top-10 and all-100
member distributions agree:

| Operator | Top-10 median (10%–90%) | All-100 median (10%–90%) |
| --- | ---: | ---: |
| Joint permutation | 3.26 (3.00–3.54) | 3.19 (2.86–3.55) |
| Independent channel shifts | 2.41 (2.23–2.59) | 2.25 (2.08–2.49) |
| Common relative-phase-preserving rotation | 2.80 (1.93–4.74) | 2.36 (1.73–3.56) |
| Per-channel phase scramble | 2.98 (2.32–4.88) | 2.69 (2.28–3.73) |

Joint permutation preserves every pointwise channel vector but destroys its
parallel order, directly rejecting a multiset-only description. Median signed
changes are -0.682 clipped-log units for joint permutation, -0.201 for
independent shifts, -0.050 for common phase rotation, and -0.228 for per-channel
scrambling. Signed member values remain in the full table.

With exact cyclic block orders excluded, the interpretable length-scale spectrum
has medians 2.93, 2.42, 1.83, and 1.26 panel residual SD for lengths 2, 4, 8,
and 16. Across three random replicates, the range of each member-median rung is at
most 0.070. The L=32 endpoint is **not** another generic spectrum rung: with
three blocks, rejecting the three cyclic rotations leaves only a reversal and
its rotations. Its 1.17 effect is therefore reported as a three-block reversal
control, consistent with the wrong-parity reversal effect of 1.26. The prior
claim that L=32 extended the monotone length-scale curve is withdrawn, as is the
older L=32 implementation that mixed in 46.1% exact-shift endpoints.

PLAN's scoping values were approximately 2.6 for joint permutation and 2.3 for
independent shifts on 256–2,000-row unstable varied slices. The frozen S01 panel
gives 3.26 and 2.41 on its own enriched-panel denominators; independent shifts
agree closely, while joint permutation is about 1.25× the scoping value. For the
top-ranked member on unstable original-`f` rows, raw changes are 1.243 and 0.920
clipped-log units, panel-normalized values 3.085 and 2.283, and S02-reference-
normalized values 4.129 and 3.055. The differences follow the registered panel's
error/disagreement enrichment, function/stratum choice, and denominator choice;
they are not an order-of-magnitude discrepancy signaling an operator bug.

## Spectral and channel ladders

![Phase-preserving Fourier dose response](S03_artifacts/dose_response.png)

Raw full-attenuation effects are 3.85, 1.22, and 0.099 panel residual SD for
low (bins 1–4), mid (5–16), and high (17–48) bands. Those edits remove unequal
robust input RMS of 2.64, 2.59, and 0.371. Effect per robust input RMS is therefore
1.46, 0.472, and 0.268. The low band remains 3.1 times more effective than the
mid band and 5.4 times more effective than the high band after explicit dose
normalization; unlike the old raw ranking, this supports prioritizing low
frequency within the registered Fourier attenuation family in later attribution
work. It does not compare Fourier efficiency with permutations, shifts, or other
operator families.

The conclusion is consistent across doses. At doses 0.25/0.5/0.75/1, low-band
effect per robust input RMS is 2.81/2.28/1.84/1.46, mid is
0.649/0.583/0.518/0.472, and high is 0.309/0.324/0.297/0.268. Declining efficiency
with dose shows nonlinearity and is retained rather than summarized as a single
linear sensitivity.

Uniform attenuation of every non-DC amplitude is the registered control that
changes marginal power while preserving all relative phases. At doses
0.25/0.5/0.75/1 its top-10 effects are 1.898/3.118/3.908/4.016 panel residual SD,
with robust input RMS 0.930/1.859/2.789/3.719. Efficiency declines
2.041/1.677/1.401/1.080 across the path. The large full-dose response shows that
marginal spectral power alone matters substantially; the decreasing efficiency
again warns against treating these finite edits as a linear decomposition.

Conditional single-channel replacement gives a different ranking once actual
edit size is shown:

| Channel | Output RMS / panel residual SD | Input displacement (robust RMS) | Effect per robust input RMS |
| --- | ---: | ---: | ---: |
| `gbdrift0_over_shat` | 1.82 | 1.10 | 1.65 |
| `gbdrift` | 2.18 | 2.00 | 1.09 |
| `gds22_over_shat_squared` | 2.02 | 2.06 | 0.976 |
| `cvdrift` | 1.14 | 1.99 | 0.573 |
| `gds21_over_shat` | 1.27 | 2.33 | 0.547 |
| `bmag` | 0.670 | 1.33 | 0.504 |
| `gds2` | 1.99 | 10.84 | 0.183 |

Displacements span 9.8-fold, so the old raw-output ranking was not a fair
cross-channel comparison and is withdrawn. Even the normalized result is only a
coarse prioritization diagnostic: channel correlations, the channel-0/channel-6
target identity, and off-manifold replacement paths prevent causal or
independent-channel interpretation.

![Canonical structure-destroying ladder](S03_artifacts/ladder_overview.png)

## Drive dependence and support warnings

Stable/near-floor and unstable varied rows are never pooled. Top-10 canonical
medians for joint permutation are 0.996 panel residual SD on stable rows and
3.54 on unstable rows; independent shifts are 1.20 and 2.61; common phase
rotation is 1.31 and 3.01; per-channel phase scrambling is 1.72 and 3.19. Fixed
twins are not reported because their shared `a/L_T=-3` input saturates the
checkpoint rather than representing its trained constant-drive convention.

The support fit uses 1,536 equilibrium-unique reference rows and calibrates on
512 held-out equilibria. All 2,048 support/background equilibria are disjoint
from the panel; the old selection leaked 78 sibling tubes and is withdrawn.
Per-channel median/IQR scaling precedes a 24-component PCA. Cyclic phase is
anchored at the largest joint robust-standardized seven-channel excursion, so a
constant channel 0 cannot create a degenerate anchor.

The warning score is the maximum of two two-sided calibrated percentile scores.
Even for two independent calibrated uniforms its structural null median is
sqrt(0.5) = 0.707 and its fraction above 0.95 is 1 - 0.95² = 9.75%, not 5%.
The panel's unperturbed path-dose-zero median 0.719 and tail 11.4% are therefore
close to the structural null. Exact-shift tails are similarly 11.2%–11.3%.
Tail fractions, rather than raw warning medians, show the useful departures:
complete non-DC removal is 82.1% outside, full low-band attenuation 36.9%, and
full mid-band attenuation 26.5%.

An important negative check remains: this coarse PCA warning does not reliably
identify ordering edits. Joint permutation has 10.5% outside and independent
shifts 6.9%. Low warning values are not evidence of physical realizability.

## Toy controls, determinism, and reproducibility

The registered toy gate now runs before any real-member inference and covers all
12 operator families. The permutation toy is invariant to joint and block
permutations and joint shifts but changes under independent shifts. The
co-location toy is invariant to joint/block permutations, joint shifts, and the
correct common phase rotation (RMS 1.78e-8), while independent phase scrambling
changes it by 0.142. The Fourier toy's relevant-band effect is 24.0 versus
5.72e-6 for its high-band control; amplitude scaling changes it by 18.0. Parity
and reversal round trips are exact, replacement touches only its registered
channel, phase scrambling preserves amplitudes to 3.81e-6, and wrapped windows
retain constant support across index 0.

Every full-batch production endpoint is generated twice before inference and
compared bit-for-bit, including channel replacement. Endpoint SHA-256 values are
manifest-hashed in
[`S03_artifacts/operator_endpoint_hashes.json`](S03_artifacts/operator_endpoint_hashes.json).
Two independent fresh-process pilot runs produced byte-identical 46-entry
endpoint-hash JSON files and byte-identical prediction HDF5 files. Determinism is
therefore measured rather than asserted by a literal.
Regression tests additionally require every seeded random operator to give
bit-identical endpoints to registered twins and to reject malformed twin input.

The production manifest records Python 3.12.4, torch 2.4.1, numpy 1.26.4, h5py
3.11.0, and Captum 0.9.0. Dataset SHA-256 is
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`;
protected-checkpoint SHA-256 is
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.
Both were read only.

`predictions.h5` retains `(member, function, perturbation, sample)` axes with
signed native outputs. Resume validates dataset, checkpoint, row IDs, member
IDs, perturbation registry, config, cohort and robust-scale fingerprints, and
code hashes. `ladder_summary.csv` is generated from `ladder.csv` by the
registered CLI and both are manifest-hashed; only the compact table is committed.

Pilot, production, and resume commands:

```bash
MPLCONFIGDIR=/private/tmp/mpl-s03 .venv-xai/bin/python \
  scripts/xai_s03_ladder.py --config configs/xai/S03_ladder.json \
  --pilot --no-publish

MPLCONFIGDIR=/private/tmp/mpl-s03 .venv-xai/bin/python \
  scripts/xai_s03_ladder.py --config configs/xai/S03_ladder.json

MPLCONFIGDIR=/private/tmp/mpl-s03 .venv-xai/bin/python \
  scripts/xai_s03_ladder.py --config configs/xai/S03_ladder.json --resume

MPLCONFIGDIR=/private/tmp/mpl-s03 .venv-xai/bin/python \
  scripts/xai_s03_review2_artifacts.py \
  --source-run output/xai/S03/ladder-top10-all100-varied1000-paired2000-reviewfix
```

The pilot uses a deterministic proportional sample across equilibrium class ×
stable/unstable strata (64 varied rows plus fixed twins), not a sorted row-ID
prefix. It uses one member, two random replicates, and 200 grouped bootstrap
draws. Production uses 1,000 draws.

## External-review disposition

All 14 findings in `review_step03_01.md` were accepted; none was rejected.

1. Common phase now rotates the original complex spectrum; analytic cross-phase
   and co-location controls enforce the intended invariant.
2. Block permutations reject identity and all cyclic block rotations; L=32 was
   rerun on the full cohort.
3. Every ladder row now carries robust input RMS and effect per robust input RMS;
   the spectral conclusion was re-derived on that axis.
4. Channel replacement reports the same robust dose columns and the report ranks
   dose-normalized effects.
5. Headline denominators now come from the panel; the S02 reference value remains
   comparison-only, and bootstrap draws recompute the denominator.
6. Support/background selection excludes panel `equilibrium_files`, with a hard
   zero-overlap runtime assertion.
7. The expanded all-family toy gate runs before model inference.
8. The CLI generates and manifest-hashes `ladder_summary.csv`.
9. The two-sided support-tail column and prose are correctly named.
10. Full-batch repeats, endpoint hashes, and two fresh-process byte comparisons
    replace the hard-coded determinism assertion.
11. Support canonicalization uses joint standardized energy instead of channel
    0 alone; a constant-channel-0 shift-invariance test was added.
12. Production bootstrap resolution is 1,000, and denominator uncertainty is
    included in every varied-row interval.
13. Pilot row reduction is proportional class × stability sampling.
14. Run IDs now distinguish 1,000 varied from 2,000 paired total rows, and the
    baseline acceptance wording distinguishes API coverage from network use.

### Second review (`review_step03_02.md`)

Findings 1–10 and 13–15 were accepted. Findings 11 and 12 were rejected as
stated, with the following concrete evidence; defensive clarifications or checks
were still added.

1. Fixed-gradient interpretations are withdrawn; the serialized training-tensor
   and direct-inference evidence is in the decision memo.
2. L=32 is reclassified as a three-block reversal control and removed from the
   length-scale spectrum.
3. All phase replicates and paired member differences are now reported; the
   earlier negative conclusion is withdrawn.
4. Registered matched controls now produce explicit signed paired contrasts;
   the vacuous RMS-interval statement is removed.
5. The support structural null is corrected to median 0.707 and 9.75% tail.
6. Robust input RMS comparisons are limited to within Fourier or channel-
   replacement sections; the compact table blanks the cross-family dose cells.
7. Random operators apply the same realization to both registered twins, with
   regression coverage.
8. Original-`f` arbitrary-shift error (0.394 panel residual SD) is reported.
9. The non-DC amplitude-scaling control is quantified and interpreted.
10. The PLAN scoping magnitudes are reconciled by cohort, function/stratum, and
    denominator.
11. Rejected as an acceptance defect: S03 task 1 requires all four tags and the
    reusable baseline/window API, while task 3 enumerates the production ladder.
    It does not require a member intervention for every tag or API method. The
    report now states exactly which contracts are API/toy-only.
12. Rejected as a code defect: after reserving one row per stratum, proportional
    allocation uses `remaining <= sum(capacity)`. For `alpha < 1`, each floored
    positive-capacity allocation remains below its capacity and leftover units
    go to distinct positive fractional remainders; for `alpha = 1`, leftover is
    zero. Over-allocation is therefore impossible. A runtime capacity and exact-
    sum assertion was nevertheless added.
13. The varied residual is now branch-local rather than relying on mask order.
14. Conditional profiles now fail loudly on an empty eligible set. Fixed-row
    conditional-profile interpretations are withdrawn with all fixed results.
15. The CLI no longer publishes the full ignored `ladder.csv` under reports;
    that table remains only in the manifest-hashed run directory.

## Acceptance criteria

| Criterion | Evidence | Status |
| --- | --- | --- |
| Deterministic fixed-seed methods | Full-batch repeats, hashes, two byte-identical fresh pilots | Pass |
| Wrapped windows have no boundary artifact | Shifted wraparound support equality | Pass |
| Toy relevant features outrank controls | Pre-inference checks for all 12 families | Pass |
| Exact symmetries null within S02 tolerance | Maximum absolute error 9.54e-6 < 2e-5 | Pass |
| Every perturbation has a validity tag | Machine-readable tag on all 56 entries | Pass |
| Every retained varied endpoint strength has support | 56 entries × five path doses on the unique varied geometries | Pass |
| Top-10 full varied ladder | Both `f` and `tilde_f`, stable/unstable separately | Pass |
| Cheap entries cover all 100 members | 13 entries; signed member predictions retained | Pass |
| Baseline API avoids all-zero default | Six registered API families; conditional profile used on varied rows | Pass |
| Member-level grouped uncertainty | 1,000 equilibrium-file draws with joint denominator resampling | Pass |

Verification commands:

```bash
conda run -n 20240629-01-ML python -m pytest
.venv-xai/bin/python -m pytest
git diff --check
```

## Deferred

The repository-wide fixed-gradient loader correction and refresh of affected
S00–S02 fixed artifacts are deliberately deferred to the decision recorded in
`S03_fixed_gradient_decision.md`; no fixed result may be used downstream until
that correction is made. All S03-local tasks, varied-panel calculations, and
valid review corrections are complete.
