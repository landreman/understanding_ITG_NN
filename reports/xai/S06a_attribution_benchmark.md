# S06a — Input-attribution benchmark

## Result

For the registered top member `2864601_0.437`, the control-aware rerun retains
**64-step Integrated Gradients from the low-pass S03 reference** as the primary
path/gradient method. The matched-observed **periodic extremal mask** is retained
only as a secondary perturbation sensitivity: its robust-constant-background
variant failed the stable insertion and S02 symmetry clauses, while the matched
variant retains its fixed-background symmetry failure. Both explain the member's native
$\max(\log Q,-2)$ output, and both were evaluated for S02's canonical
$\tilde f$ and the original $f$. The path/fallback pair is a benchmark result about
the network, not a physical-causality result: both paths are tagged
`deliberately_off_manifold_diagnostic`.

The researcher-approved rerun adds a control-aware clause to the corrected
stratum rule: every path candidate must beat its own network-free $|X-B|$ map
with a 500-draw whole-`equilibrium_files` paired interval excluding zero in both
unstable directions. Stable rows remain published but are not gated under the
researcher-approved rule. Four path candidates pass the registered rule;
low-pass IG still wins the unchanged normalized-infidelity tie-break at
**0.09697** (robust constant 0.31180, medoid 0.35032, Expected Gradients
0.65198). Matched-observed IG beats its control but remains ineligible because
its unstable deletion margin versus random is -0.263.

This is a **post-run selection rule**: both the per-stratum correction and the
control-aware clause were designed after results on these 128 rows were known.
The 64-row pilot was then regenerated under the corrected rule, so it verifies
the implementation and exposes panel sensitivity but is not an independent
pre-production validation. Selection is therefore a benchmark registration for
S06b, not an unbiased estimate of method-selection performance.

On the analytic wrapped-window toy, the path primary and mask fallback recover the relevant
channel at top 1 and obtain position average precision **1.000**, versus **0.0442**
for the random-map control. Their toy deletion AUC is **0.025** versus **0.557**
for random order, and insertion AUC is **0.975** versus **0.443**. On the
toy, 11 of 12 methods attain the same perfect recovery and AUC values, so this
registered threshold is a sanity-check floor rather than a useful ranking among
the passing methods. On the
128-row canonical panel benchmark, the rule is evaluated separately on the 33
stable/near-floor and 95 unstable rows, not on a pooled normalized ratio.
Low-pass IG deletion/insertion margins are **1.503/0.211** in the stable stratum
and **0.420/0.420** in the unstable stratum. Periodic-mask margins are
**0.431/0.055** and **1.744/1.527**, respectively. The mask is optimized per
sample with the same
replacement operator that its deletion curve scores, so those unusually large
mask margins are in-sample optimization results, not independent faithfulness
evidence. The mask curve also overshoots its baseline at 10% and 25% replacement.
These normalized AUCs are comparable to each method's own random-order control,
not across methods with different baselines.

The former pooled mask ratios, **5.247/5.293**, are retained in the artifact but
are not headline evidence. Their matched-observed endpoint denominators have
opposite signs in the two strata: **-0.4915** stable and **+0.2349** unstable,
leaving only **+0.04765** after pooling. The resulting ratio lies outside both
stratum-specific results. This cancellation also made pooled cyclic occlusion
look worse than random even though its deletion margins are positive in both
strata. The corrected gate therefore requires positive deletion and insertion
margins in each stratum; the stored path/fallback pair is unchanged.

Random ordering is a weak control because it preserves the selected map's cell
magnitudes. The benchmark therefore scores the network-free control map
$|X-B|$ through the same trained-network replacement curves for all five path
candidates and both masks. On stable rows,
low-pass IG does **not** beat that control: paired per-row-oriented gaps are
**-0.00043** (-0.00385–0.00224) deletion and **-0.00202**
(-0.00939–0.00219) insertion. On unstable rows it does: **0.00964**
(0.00446–0.01638) and **0.01105** (0.00454–0.01953). For the matched mask, only unstable
insertion resolves against its displacement control, **0.1374**
(0.0103–0.2593); its other three stratum/direction intervals cross zero.
Robust-constant, medoid, low-pass, and Expected Gradients all clear the added
unstable control clause; low-pass remains selected by the registered tie-break.

The stable control-comparison effect size is baseline-specific, not a generic
near-floor tie. Low-pass's estimates are near zero, while the other four path
candidates' estimates on the same 33 rows are roughly two orders of magnitude
larger:

| Path candidate | Stable deletion method−control gap (95% CI) | Stable insertion gap (95% CI) |
| --- | ---: | ---: |
| robust-constant IG | 0.02931 (0.00155–0.07388) | 0.06622 (0.02305–0.12091) |
| matched-observed IG | 0.16320 (0.04903–0.29224) | 0.14270 (0.05373–0.26515) |
| medoid IG | 0.03735 (0.00058–0.07905) | 0.04915 (0.01599–0.09503) |
| low-pass IG | -0.00043 (-0.00385–0.00224) | -0.00202 (-0.00939–0.00219) |
| Expected Gradients | 0.03871 (0.00048–0.09556) | 0.03673 (0.00228–0.08730) |

The interval endpoints are descriptive, not a reliable way to distinguish the
smallest positive bounds: with 500 resamples the 2.5th percentile is only the
12.5th order statistic, so the approximately 0.0005 medoid and Expected-
Gradients lower bounds are within bootstrap Monte Carlo resolution. The
baseline-specific conclusion rests on the much larger point estimates, not on
which intervals happen to cross zero. Had the registered control clause gated
both strata, low-pass would be ineligible and robust-constant IG would win the
unchanged infidelity tie-break.

This faithfulness-versus-control metric does not establish that clipped rows
carry feature information. The researcher-registered conservative caveat is
therefore unchanged: S06b reports the stable/near-floor stratum but makes no
feature-level claim from any method there. The table remains a baseline-
sensitivity result, not permission to interpret stable-row maps.

Each candidate also faces a different control difficulty because its control is
its own $|X-B|$. In the stable stratum, the low-pass control's normalized
margins are 1.7404/0.7917, versus roughly 0.04–0.23 for the other baseline
families. Passing these candidate-specific controls is therefore not a uniform
bar across baselines.

The earlier reviewer negative result is retained in full: low-pass's $|X-B|$
control itself clears the toy floor (channel top-1 1.000, position AP 1.000) and
the prior per-stratum random-order faithfulness gate in all four cells (stable
1.7404/0.7917, unstable 0.1684/0.1526). The old rule excluded it only because a
model-independent map has parameter-randomization correlation exactly 1.000.
That is why the new paired method-minus-control clause, rather than the old
randomization clause alone, is needed.

Native-unit effect sizes further qualify the unstable pass:

| Eligible path | Method gap, deletion/insertion | Control gap, deletion/insertion | Method−control gap, deletion/insertion |
| --- | ---: | ---: | ---: |
| robust-constant IG | 0.533/0.701 | 0.265/0.200 | 0.268/0.501 |
| medoid IG | 0.549/0.715 | 0.129/0.114 | 0.420/0.601 |
| low-pass IG | 0.0203/0.0207 | 0.0107/0.0097 | 0.0096/0.0111 |
| Expected Gradients | 0.478/0.645 | 0.235/0.221 | 0.243/0.424 |

Low-pass's paired advantage is 25–60 times smaller than the other eligible
paths. It wins on normalized infidelity, a scale-relative ratio that can favour
the shortest baseline path; the selection should not be read as a like-for-like
native-effect contest.

The path primary and mask fallback respond to complete parameter randomization: absolute-map
rank correlation with the randomized member is **0.406** for low-pass IG and
**0.099** for the mask. Low-pass IG has the weakest response among eligible IG
baselines: robust-constant, medoid, and Expected Gradients correlations are
0.235, 0.284, and 0.070. Its input-minus-low-pass magnitude correlates 0.477
with the trained map but 0.816 with the randomized map, showing that baseline
structure explains much of the residual 0.406.

The canonical explanation-equivariance relative RMS with the baseline co-shifted
is $1.03\times10^{-4}$ for low-pass IG and $2.70\times10^{-7}$ for the mask.
Co-shifting is the relevant convention for input-derived low-pass baselines, but
the registered mask uses a fixed matched-observed background. With that
background held fixed, mask equivariance error is **1.009**, not near zero. The
same methods on original $f$ have co-shifted errors **0.821** and **0.820**. This
distinction is explicit in `benchmark_metrics.csv`; S06b must not inherit the
co-shifted mask number as a property of its fixed-background maps.

The new robust-constant mask uses the committed seven-channel z-median profile.
It beats random in stable deletion (**0.774**) but fails stable insertion
(**-0.155**), and its fixed-background equivariance error is
$9.88\times10^{-4}$, above S02's registered $2\times10^{-5}$ tolerance. It does
beat its own control in both unstable directions, but fails the full gate and is
retained as a negative sensitivity result rather than registered as primary.

The $2\times10^{-5}$ ceiling is a conservative **mask-candidate-only** clause,
borrowed from S02's float32 model-output tolerance rather than established as a
universal relative-RMS attribution tolerance. It was added to answer the
decision memo's fixed-background perturbation question. Applying it universally
would also reject fixed-background matched IG (1.122), medoid IG (0.653), and
Expected Gradients (0.737); low-pass's registered co-shifted error is
$1.03\times10^{-4}$. Those path methods retain their documented baseline
conventions and are not claimed to pass the mask-only ceiling.

## Estimand and cohort

The primary estimand is a signed `(member, sample, channel, z)` attribution of

$$
\tilde f_m(X,a/L_T,a/L_n)
=\mathrm{MLP}_m(\bar u_m(X),a/L_T,a/L_n)
$$

in native $\max(\log Q,-2)$ units. Every position-resolved method is also run on
the original $f_m$, and `canonical_minus_original` is stored explicitly. No
prediction is exponentiated. Robust-scaled gradients use S01's per-channel IQR
scales; raw gradients are never compared across the seven channels.

The production cohort is a deterministic class-by-stability-stratified
128-row subset of S01's frozen 1,000-row varied-gradient panel. It contains 33
stable/near-floor and 95 unstable rows, each from a distinct
`equilibrium_files`. The 512-row S03 reference/background cohort excludes every
panel equilibrium; 384 rows fit the robust-scaled PCA support diagnostic and 128
calibrate it. Production and development read the canonical external HDF5 file,
not `tests/data/review_slice.h5`.

The registered run is
`output/xai/S06a/attribution-benchmark-top1-panel128/`, with `run_id` quoted in
the committed [manifest](S06a_artifacts/manifest.json). It records checkpoint
SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`
and dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`.
Production used CPU, Python 3.12.4, torch 2.4.1, numpy 1.26.4, h5py 3.11.0,
and Captum 0.9.0.
The run was generated from a tracked-dirty tree at commit `de7aae6`; the
manifest's hashed script and attribution module match the eventual committed
head byte-for-byte and pin the code used for these post-run diagnostics.

## Methods and artifacts

The benchmark covers 12 registered maps:

- signed robust-scaled gradient;
- Integrated Gradients from robust-constant, matched-observed, medoid, and
  low-pass S03 references;
- Captum GradientSHAP/Expected Gradients over eight observed backgrounds;
- robust-noise VarGrad;
- cyclic channel-by-window occlusion;
- periodic-total-variation extremal masks with matched-observed and
  robust-constant replacement backgrounds; and
- temporal-saliency-rescaled gradient and robust-reference IG.

LRP was not included because this step did not document complete relevance-rule
coverage for circular convolution, max pooling, biases, and signed inputs.

The ignored `attribution_maps.h5` retains all 24 full maps with axes
`(function, method, member, sample, channel, z)`, method-path labels, signed
flags, validity tags, and the canonical-minus-original difference. The committed
[selected review maps](S06a_artifacts/selected_review_maps.h5) retain both
selected methods, both functions, and the first 16 registered rows. The
[benchmark table](S06a_artifacts/benchmark_metrics.csv) distinguishes signed
maps, contribution-valued maps, magnitude-only maps, baseline family, baseline
validity, exact estimator path, Captum batch adapter, optimizer determinism,
co-shifted/fixed-baseline symmetry, and low-pass baseline-structure controls.
For every control-comparison candidate, the `*_margin_vs_control_map` fields are
differences of **normalized AUC ratios**; the selection clause instead uses the
separately named paired per-row-oriented native-unit gaps and intervals. The
exact seven z-median background values are committed in
[`robust_constant_background.csv`](S06a_artifacts/robust_constant_background.csv).
Infidelity is intentionally `NaN` for
dimensionless sensitivity and mask maps rather than pretending they are
additive output contributions.

[Faithfulness curves](S06a_artifacts/faithfulness_curves.csv) carry deletion,
insertion, matched random-order controls, robust input displacement, and S03 PCA
support warning at every registered fraction. The support warning does not
certify physical validity or off-manifold drift: it is a two-sided percentile
diagnostic that can be high for central fit rows as well as tail rows. The
low-pass warning is already 0.707 at the unedited observed geometry and remains
0.687–0.711; the mask warning reaches 0.920 exactly when deletion reaches its
matched-observed support-row endpoint. The useful effect-size diagnostic is the
robust input displacement, which rises from 0 to 0.604 for low-pass IG and to
3.809 robust RMS for the mask. The paths remain tagged off manifold because
their smoothed/intermediate and cell-replaced geometries are not guaranteed to
be observed or physically realizable, not because of the warning score.

The [IG convergence table](S06a_artifacts/ig_convergence.csv) compares 64 with
32 steps. Canonical rank correlation is **0.99925** for low-pass IG and
0.99275–0.99607 for the other references. Low-pass IG completeness residual is
median $7.90\times10^{-5}$, q90 $9.34\times10^{-4}$, and maximum 0.00409 native
units. Its normalized mask infidelity is **0.09697**, the lowest among eligible
path methods.

The normalization denominator is small. On the registered rows, the low-pass
endpoint difference $\tilde f(X)-\tilde f(B)$ has median **0.0014** and mean
**0.0175** native units; the batch-mean prediction moves from 0.4462 to 0.4287
over the full deletion sweep. The median completeness residual
$7.90\times10^{-5}$ is therefore about **5.6%** of the median endpoint
difference. The normalized margins retain the intended point-estimate
orientation, but their uncertainty is not robust to a near-zero denominator;
the native-unit gaps below carry the interval interpretation.

## Baseline sensitivity

Baseline choice materially changes the answer. On canonical $\tilde f$, the
absolute-map rank correlation against robust-constant IG is **0.432** for the
selected low-pass map, **0.536** for matched-observed IG, and **0.749** for
medoid IG. This is an understood sensitivity, not evidence that one baseline is
physically correct. Low-pass was selected because it passed both faithfulness
directions, randomization, toy, and symmetry checks and then had the lowest
infidelity under the fixed tie-break; its endpoint and path remain off manifold.

The regenerated 64-row pilot applies the same control-aware stratum rule and
selects medoid IG. Low-pass misses the unstable deletion control interval there;
the robust-constant mask fails stable insertion, so the documented matched-mask
fallback is used. Production retains low-pass IG by infidelity and uses the same
perturbation fallback. The pilot and production panels overlap by 7 rows (11%
of the pilot).
The committed [pilot selection](S06a_artifacts/pilot_selected_methods.json)
and [candidate rows](S06a_artifacts/pilot_candidate_metrics.csv) record all
five path candidates and both masks in all three strata. This
pilot-to-production instability is a negative result: S06b must retain medoid
and robust-constant IG as baseline sensitivity analyses and must not treat the
primary map as baseline-independent.

## Stable/near-floor versus unstable rows

Stable and unstable results are separate in every benchmark and uncertainty
artifact. The selected maps pass both faithfulness directions in both strata,
but their magnitudes and sharpness differ. Low-pass IG's mean absolute
attribution is **0.000527** on stable rows (500 equilibrium-file bootstrap 95%
interval 0.000161–0.001013) and **0.001395** on unstable rows
(0.001027–0.001769). Its 90%-mass sparsity fraction is 0.387 stable and 0.379
unstable. The mask is much more diffuse on stable rows: 0.732 of cells carry
90% of its mass versus 0.363 on unstable rows.

No mechanism or common feature is named from these maps in S06a. That requires
the top-10 member agreement and equilibrium/member hierarchy in S06b, followed
by the physics comparisons in S07.

Stable/near-floor maps remain near-uninformative for feature claims under the
researcher-registered caveat. Clipping leaves the selected low-pass baseline a
median endpoint difference of only **0.0014 native units**. Other baselines have
larger method-minus-control effect estimates, but that faithfulness comparison
does not demonstrate feature information in clipped rows, and the smallest
positive 500-resample bounds are within bootstrap Monte Carlo resolution. S06b
must report this stratum but may not base a feature-level claim on it.

## Uncertainty

[Grouped uncertainty](S06a_artifacts/grouped_uncertainty.csv) uses 500 resamples
of whole `equilibrium_files` and reports sample-mean absolute attribution for
every method, function, and floor stratum. For the selected canonical maps, the
all-row estimates and 95% intervals are 0.001171 (0.000856–0.001465) for
low-pass IG and 0.1123 (0.1103–0.1143) for the dimensionless mask.
Each of these 128 panel rows has a distinct `equilibrium_files` value, so grouped
and row bootstrap distributions coincide in S06a; grouping becomes substantive
when S06b includes sibling flux tubes.

The same artifact bootstraps deletion and insertion evidence for all five path
candidates and both masks in every floor stratum. It publishes the normalized ratio, its native-unit
numerator, a cohort-mean-oriented gap, a per-row-oriented native-unit gap, and
the endpoint denominator distribution. The old ratio
intervals cross zero for 11 of 12 comparisons, but their width is dominated by
the normalizer rather than cleanly measuring uncertainty in the effect. For
low-pass deletion, the stable denominator is only **0.00273** native units;
**22.6%** of resamples reverse its sign and **70.2%** have magnitude below
0.005. Even all-row low-pass has **3.2%** negative and **8.0%** near-zero
denominators.

The cohort-mean-oriented gaps are an aggregation sensitivity analysis, not the
central interval: 38.3% of low-pass rows have negative endpoint differences, so
orienting every row by the cohort mean still permits cancellation inside each
stratum. Under that convention all four low-pass stratum intervals cross zero.

The primary denominator-free summary instead orients each row by its own
endpoint and then averages. Low-pass deletion gaps are **0.00625**
(0.00086–0.01251) stable and **0.02032** (0.01035–0.03342) unstable; insertion
is **0.00504** (0.00048–0.01094) and **0.02071** (0.00727–0.03298). All four
exclude zero, and 75.8–80.0% of rows favour low-pass within the two strata. Mask
deletion gaps are **0.1296** (-0.0372–0.3287) stable and **0.3268**
(0.1180–0.5515) unstable; insertion is **0.0461** (-0.0155–0.1294) and
**0.3405** (0.1824–0.4994). The two unstable-mask intervals exclude zero; all
mask gaps remain in-sample optimization diagnostics. The reversal between
cohort-mean and per-row orientation is itself a material aggregation
sensitivity, now published rather than collapsed into one conclusion.
The faithfulness and paired control-map intervals are nested, strongly
correlated comparisons on the same candidates and one panel, not independent
hypothesis tests. No multiplicity correction is applied, so their coverage is
descriptive; only the registered two-direction unstable control clause is used
for path eligibility.

S06a has exactly one preregistered member, so these intervals cover equilibrium
sampling only. The full S06 acceptance clause requiring both member and
equilibrium uncertainty is deliberately pending S06b tasks 5–7, which run the
selected methods across the registered top 10 and a wider sensitivity sample.

## Failed checks and corrections

Sixteen failures were found before the final registered run and retained in tests
or method records:

1. Captum batches all IG steps together; the first real-model pilot exposed
   drive vectors that were not repeated in step-major order. The closure was
   corrected and the regression test now exercises the enlarged batch.
2. Captum 0.9 GradientSHAP draws path coefficients with NumPy's global RNG.
   Seeding only PyTorch gave canonical explanation-equivariance error 0.889.
   Seeding and restoring both RNGs reduced it to $7.33\times10^{-8}$ in the
   registered run.
3. The first selection rule checked deletion but not insertion. The pilot showed
   that this admitted a method with worse-than-random insertion, so the rule was
   corrected before production to require both directions.
4. The first production curve normalized by
   $|f(X)-f(B)|$, which reversed faithfulness ordering when stable predictions
   lay below their baselines. The signed analytic control caught the error;
   pilot and production were both regenerated with
   $(f(\cdot)-f(B))/(f(X)-f(B))$.
5. Automated review found that Captum 0.9 `GradientShap` expands panel rows in
   sample-major order, unlike `IntegratedGradients`' step-major expansion. The
   shared drive closure therefore paired some Expected Gradients paths with the
   wrong $a/L_T$ and $a/L_n$. A drive-dependent Captum-versus-fallback test was
   shown to fail, a sample-major adapter was added, and pilot and production
   were regenerated. The corrected canonical Expected Gradients deletion and
   insertion margins are 0.301 and 0.449 (previously 0.294 and 0.395); it stays
   eligible but does not replace either selected method.
6. Review showed that no test constrained absolute-magnitude ordering inside
   deletion/insertion. A mixed-sign fixture now makes signed ordering fail while
   preserving the registered absolute ranking rule.
7. The symmetry artifact silently co-shifted every baseline. The benchmark now
   names that convention, also computes fixed-baseline error, and tests that the
   two differ. This exposed the selected mask's fixed-background error 1.009.
8. Fallback IG was tested only where every quadrature rule is exact. A nonlinear
   analytic path now pins the trapezoid estimate tightly; production remains on
   the explicitly recorded Captum path.
9. The randomization threshold was hard-coded and headline faithfulness margins
   lacked intervals. The 0.95 maximum is now part of the registered rule in
   config, pilot, production selection, and manifest, and grouped margin
   intervals are committed for every control-comparison candidate and all floor
   strata.
10. Review found that pooled faithfulness divided by a denominator in which the
    floor strata cancel. The production gate now requires both directions to be
    positive in each stratum. Native-unit gaps and denominator diagnostics are
    published, and a sibling-tube fixture pins whole-equilibrium resampling.
    The selected production path survives. The regenerated pilot now applies
    the corrected rule directly and records its medoid/fallback-mask result.
11. A second review found the same cancellation within strata because native
    gaps were oriented by the stratum-mean endpoint. The artifact now publishes
    per-row-oriented gaps and whole-equilibrium intervals alongside the
    cohort-mean convention; a negative-denominator analytic fixture pins the
    sign, and the two conventions' different conclusions are retained.
12. A third review showed that random order was not a control map. The selected
    methods are now paired against the network-free $|X-B|$ ranking in every
    stratum and direction. Low-pass adds resolved network-dependent ordering on
    unstable rows but not stable rows; most mask comparisons against its control
    are unresolved.
13. The control statistic was initially unpinned. A negative-endpoint paired
    fixture now fails if the control is not oriented per row, and a mixed-sign
    tensor pins the production control ranking to exactly $|X-B|$.
14. The researcher-approved rerun scores all five path candidates against their
    own baseline-matched controls before selection. An analytic selection
    fixture rejects a candidate when either unstable paired interval touches
    zero, and committed-artifact assertions pin the observed stable-negative and
    unstable-positive low-pass control-gap signs.
15. The robust-constant mask is fixed-background equivariant on the analytic
    cyclic toy, but the real model misses S02's $2\times10^{-5}$ tolerance and
    its stable insertion margin is negative. Both failures are retained; the
    matched-observed mask is the explicitly secondary fallback.
16. Review showed that neither the mask symmetry ceiling nor the two fallback
    branches were exercised by a selection fixture. A paired mask-like fixture
    now straddles the ceiling and asserts both the failing-candidate fallback and
    passing-candidate primary paths.

Eleven deliberate post-run mutations turned the focused suite red and were
reverted: dropping robust channel scales failed the exact scaled-gradient test;
resampling individual rows instead of `equilibrium_files` failed the grouped
bootstrap support test; and restoring the absolute AUC denominator failed the
negative-direction curve test. For the final review response, deleting the
negative-denominator orientation sign failed its analytic control, and removing
the production symmetry co-shift failed the script-level equivariance pair test.
Dropping orientation from the row-favouring fraction also failed the
negative-endpoint control. For the control-aware rerun, bypassing the paired
control clause failed the analytic selection fixture, replacing $|X-B|$ with a
degenerate zero map failed the exact control-map fixture, and wiring the new
mask back to the matched-observed background failed its fixed-background
equivariance fixture (error 0.413 versus the $2\times10^{-5}$ ceiling). Finally,
bypassing the mask symmetry clause and making fallback unconditional both fail
the new two-branch selection fixture.

## Negative results and interpretation limits

- Cyclic grouped occlusion fails the toy position threshold: average precision
  is 0.6125 versus the registered 0.75 minimum. Its real deletion margins are
  actually positive in both strata, **0.641** stable and **0.074** unstable;
  the former pooled value -1.620 was a denominator-cancellation artifact and is
  not a reason for exclusion. The control-aware rerun removed it from the
  perturbation candidate list because the toy failure already makes it
  ineligible; it remains fully published as a sensitivity.
- Matched-observed IG passes deletion in the stable stratum (**0.500**) but fails
  it in the unstable stratum (**-0.263**). Its pooled -2.070 is likewise not
  interpreted. An observed endpoint does not make the interpolation path
  physical.
- The post-run stratum gate changes `scaled_gradient` and `vargrad` from eligible
  under the pooled rule to ineligible: their stable insertion margins are
  **-0.0742** and **-0.0823**. Neither was a selected candidate, but both verdict
  flips are retained.
- Low-pass IG does not beat its network-free displacement control on stable
  rows, although it does in both unstable directions. The matched mask beats its
  control conclusively only for unstable insertion. Passing a random-order
  control is therefore insufficient evidence that an attribution adds
  learned-network information. The control itself clears the toy and random-
  order gates and is excluded by randomization correlation 1.000. Low-pass's
  near-floor endpoint is small, but that alone cannot explain the effect-size
  contrast: the other four estimates are roughly two orders of magnitude
  larger on the same rows. The control comparison is baseline-specific because
  $|X-B|$ is also IG's displacement factor for low-pass. The smallest positive
  500-resample bounds are within bootstrap Monte Carlo resolution, and the
  metric does not establish feature information in clipped rows. S06b reports
  these rows but makes no feature claim from any method there.
- The robust-constant mask fails stable insertion (-0.155) and fixed-background
  equivariance ($9.88\times10^{-4}$ versus the $2\times10^{-5}$ tolerance), even
  though both unstable control intervals favour it. A shift-invariant
  replacement background was necessary but not sufficient for a real-model
  mask to pass the complete gate.
- Baseline agreement is only moderate; selected low-pass IG correlates 0.432
  with robust-reference IG.
- Low-pass IG has the weakest parameter-randomization response among eligible IG
  baselines (0.406 versus 0.235 robust-constant, 0.284 medoid, and 0.070 Expected
  Gradients). Its baseline factor is more correlated with the randomized map
  (0.816) than the trained map (0.477), so 0.406 is a qualified pass, not clean
  evidence that the explanation is dominated by learned parameters.
- Ratio and cohort-mean-oriented intervals are unstable because endpoints can
  approach, cross, and cancel around zero. Per-row orientation reverses the
  low-pass conclusion: all four stratum-specific intervals are positive.
  Stable-mask deletion and insertion remain unresolved under per-row
  orientation, while both unstable-mask intervals are positive, subject to the
  mask's in-sample optimization caveat.
- Canonical and original maps are not interchangeable: their rank correlation
  is 0.875 for low-pass IG and 0.605 for the mask, and original-$f$ maps fail
  exact equivariance as S02 predicts.
- The periodic mask's strong AUC margins are scored with the same replacement
  operator used to optimize the mask and include baseline overshoot. Its edits
  reach robust displacement 3.809. This is an in-sample network diagnostic, not
  independent faithfulness evidence or a realizable plasma intervention.
- The mask is equivariant only when its matched-observed background is co-shifted
  with the input (error $2.70\times10^{-7}$). Holding the registered background
  fixed gives error 1.009, so S06b cannot claim fixed-background map
  equivariance. This does not disqualify the mask as a perturbation sensitivity
  method, but S06b must treat it as secondary to the symmetry-conforming
  low-pass map and label the fixed-background convention explicitly.
- Parameter randomization is a full reset rather than the more granular
  layer-by-layer cascade; it establishes response, not where that response
  begins.

## Acceptance criteria

| PLAN criterion | S06a verdict and evidence |
| --- | --- |
| Selected methods beat random/control maps on toy recovery and faithfulness | **Partial.** The control-aware path primary passes toy recovery, both random-order directions in both strata, and both unstable paired control intervals (0.00964, CI 0.00446–0.01638 deletion; 0.01105, 0.00454–0.01953 insertion). Low-pass alone fails the published, non-gating stable intervals; gating them would select robust-constant IG. No perturbation candidate meets the complete gate: the robust-constant mask fails stable insertion and symmetry, so the matched mask remains a secondary sensitivity and resolves against control only for unstable insertion. |
| Selected methods respond to parameter randomization | **Pass, qualified.** Canonical absolute-map rank correlation is 0.406 for low-pass IG and 0.099 for the mask. Low-pass is weakest among eligible IG baselines, and its input-baseline factor correlates 0.816 with the randomized map. |
| Baseline sensitivity is understood | **Pass, with a strong limitation.** Low-pass/robust map correlation is 0.432; low-pass's unstable native method−control effects (0.0096/0.0111) are 25–60× smaller than the other eligible paths, and normalized infidelity is scale-relative. All four IG baselines and Expected Gradients remain published sensitivity analyses. |
| Methods meet symmetry behavior permitted by S02 | **Pass for the path's registered convention; fail for both fixed-background perturbation variants.** Low-pass's input-derived co-shift error is $1.03\times10^{-4}$. The matched mask's fixed error is 1.009; the robust-constant mask improves it to $9.88\times10^{-4}$ but still misses the mask-only $2\times10^{-5}$ ceiling. That borrowed ceiling is not asserted as a universal attribution tolerance. |
| Uncertainty includes model and equilibrium sampling | **Pending S06b by the explicit S06a/S06b split.** S06a provides 500-draw equilibrium-file intervals for its one registered member; model sampling begins in S06b. |
| Signed and absolute summaries are distinguishable | **Pass.** Full HDF5 maps retain signs; every metrics row has `signed` and `contribution_valued`; VarGrad and masks are marked magnitude-only. |
| No feature is called common without agreement | **Pass.** S06a names no common feature; member agreement is deferred to S06b. |

**Resolved 2026-08-22.** The researcher approved the control-aware selection
rerun across all five path candidates, plus a robust-constant (z-median)
background variant of the periodic mask to resolve the fixed-background
symmetry failure by construction. The decision, its evidence, and the
instructions to the implementer are recorded in
[S06a_control_aware_selection_decision.md](S06a_control_aware_selection_decision.md).
The rerun is complete. Low-pass remains the path primary; the new mask fails two
clauses, so the matched mask is retained only as the memo's documented secondary
fallback. There is no remaining S06a decision gate.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s06a XDG_CACHE_HOME=/private/tmp/cache-s06a \
  .venv-xai/bin/python scripts/xai_s06a_attribution.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s06a XDG_CACHE_HOME=/private/tmp/cache-s06a \
  .venv-xai/bin/python scripts/xai_s06a_attribution.py
.venv-xai/bin/python -m pytest tests/xai/test_attribution.py \
  tests/xai/test_attribution_artifacts.py -q
```

## Reviewer reproduction

**Recomputable on the slice.** All 128 production parent row IDs are S01 panel
rows in `tests/data/review_slice.h5`. Translate them with
`load_review_slice_index().slice_rows(manifest["row_ids"])`; never pass parent
IDs directly to `load_hdf5_rows`. Low-pass IG uses only each analyzed geometry,
its low-pass transform, the checkpoint, and the stored drives, so the reviewer
can recompute its 128-row toy, completeness, symmetry, randomization, and
faithfulness numbers, including all low-pass displacement-control gaps, within
ordinary platform tolerance. Near-zero canonical
co-shifted equivariance is roundoff-limited: the production artifact records
$1.03\times10^{-4}$ for low-pass IG while reviewer recomputations are many orders
below the fixed-baseline value, so only its effectively-zero character is
expected to agree.
The robust-constant mask background is exactly reconstructable from the seven
committed values, so its toy, randomization, fixed-background equivariance, and
review-slice faithfulness/control diagnostics are also recomputable. Agreement
on its two production failure verdicts, rather than exact off-platform digits,
is the expected check.
The first 16 maps to compare are in `selected_review_maps.h5`, with explicit axes
and row IDs. The same production path gives the low-pass endpoint denominator
median 0.0014 and mean 0.0175 native units and the full-sweep batch means
0.4462→0.4287.

**Checkable from committed artifacts alone.** Method selection and its fixed
rule are in `selected_methods.json`; all benchmark/stratum numbers are in
`benchmark_metrics.csv`; all deletion/insertion doses and support warnings are
in `faithfulness_curves.csv`; 500-draw equilibrium intervals are in
`grouped_uncertainty.csv`, including cohort-mean and per-row-oriented native-unit
gaps, row-favouring fractions, endpoint-denominator diagnostics, control-map
gaps, and paired method-minus-control intervals; convergence is in
`ig_convergence.csv`; analytic
controls are in `toy_controls.json`; the pilot medoid selection is in
`pilot_selected_methods.json`, with its 21 candidate/stratum rows in
`pilot_candidate_metrics.csv`; both pilot exports are hash-pinned in the
manifest. Hashes for the CLI and estimator module, package
versions, rows, member, checkpoint, and dataset fingerprints are in the
committed manifest. The artifact tests independently pin the production schemas
and hashes.

**Not checkable off the researcher's machine, and why.** The exact periodic-mask
map uses matched backgrounds selected from 512 equilibrium-unique S01 reference
rows outside the panel and therefore outside the review slice. The PCA support
warnings use those same rows, with a 384/128 fit/calibration split. The nearest
reviewer proxy is to select non-panel slice rows as an alternative observed
background, rerun the mask, and compare toy recovery, canonical equivariance,
randomization response, and the sign of deletion/insertion margins; agreement
would show the selection is not peculiar to the unavailable background, but it
cannot reproduce the registered digits. The medoid, matched-observed, and
Expected-Gradients baselines also depend on that 512-row
support cohort, as do the **0.432** low-pass/robust baseline-sensitivity
correlation and all PCA-warning digits. The nearest slice proxy is to form each
baseline from non-panel slice rows and compare the selected method and
correlation ordering; qualitative agreement would support, but not reproduce,
the registered sensitivity result. Full 24-method maps are about 10.5 MB in the
ignored run; only the selected 16-row maps are committed.
The mask's four displacement-control comparisons likewise depend on the
off-slice matched-observed background and cannot be independently recomputed on
the review runner.

## Deferred

- S06b tasks 5–7: top-10 signed maps, member/equilibrium hierarchical
  uncertainty, fixed/varied and covariate strata, ranks 11–100 sensitivity, and
  attribution-stability versus validation-$R^2$.
- LRP, because rule coverage was not documented.
- No feature-level scientific claim; S06a selects estimators, while S06b and S07
  establish member agreement and physics alignment.
