# S06a — Input-attribution benchmark

## Result

For the registered top member `2864601_0.437`, the preregistered quantitative
selection rule chose **64-step Integrated Gradients from the low-pass S03
reference** as the primary path/gradient method and a **periodic extremal mask**
as the primary perturbation method. Both explain the member's native
$\max(\log Q,-2)$ output, and both were evaluated for S02's canonical
$\tilde f$ and the original $f$. The selected pair is a benchmark result about
the network, not a physical-causality result: both paths are tagged
`deliberately_off_manifold_diagnostic`.

On the analytic wrapped-window toy, both selected methods recover the relevant
channel at top 1 and obtain position average precision **1.000**, versus **0.0442**
for the random-map control. Their toy deletion AUC is **0.025** versus **0.557**
for random order, and insertion AUC is **0.975** versus **0.443**. On the
128-row canonical panel benchmark, low-pass IG beats random order by **0.572**
deletion-AUC units and **0.416** insertion-AUC units; the periodic mask's margins
are **5.247** and **5.293**. These normalized AUCs are comparable to each
method's own random-order control, not across methods with different baselines.

The result survives the required floor split. Low-pass IG deletion/insertion
margins are **1.503/0.211** on 33 stable or near-floor rows and **0.420/0.420**
on 95 unstable rows. Periodic-mask margins are **0.431/0.055** and
**1.744/1.527**, respectively. The smaller stable-mask insertion margin is kept
as a weakness, not pooled away.

Both selected methods respond to complete parameter randomization: absolute-map
rank correlation with the randomized member is **0.406** for low-pass IG and
**0.099** for the mask. The canonical explanation-equivariance relative RMS is
$1.03\times10^{-4}$ and $2.70\times10^{-7}$, respectively, while the same
methods on original $f$ have errors **0.821** and **0.820**. This is the symmetry
behavior S02 permits: $\tilde f$ is exactly invariant, whereas the pooling phase
in $f$ contaminates a position-resolved map.

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

## Methods and artifacts

The benchmark covers 11 registered maps:

- signed robust-scaled gradient;
- Integrated Gradients from robust-constant, matched-observed, medoid, and
  low-pass S03 references;
- Captum GradientSHAP/Expected Gradients over eight observed backgrounds;
- robust-noise VarGrad;
- cyclic channel-by-window occlusion;
- a periodic-total-variation extremal mask; and
- temporal-saliency-rescaled gradient and robust-reference IG.

LRP was not included because this step did not document complete relevance-rule
coverage for circular convolution, max pooling, biases, and signed inputs.

The ignored `attribution_maps.h5` retains all 22 full maps with axes
`(function, method, member, sample, channel, z)`, method-path labels, signed
flags, validity tags, and the canonical-minus-original difference. The committed
[selected review maps](S06a_artifacts/selected_review_maps.h5) retain both
selected methods, both functions, and the first 16 registered rows. The
[benchmark table](S06a_artifacts/benchmark_metrics.csv) distinguishes signed
maps, contribution-valued maps, magnitude-only maps, baseline family, baseline
validity, and exact estimator path. Infidelity is intentionally `NaN` for
dimensionless sensitivity and mask maps rather than pretending they are
additive output contributions.

[Faithfulness curves](S06a_artifacts/faithfulness_curves.csv) carry deletion,
insertion, matched random-order controls, robust input displacement, and S03 PCA
support warning at every registered fraction. The support warning does not
certify physical validity. For low-pass IG, robust displacement rises from 0 to
0.604 at full replacement and the warning remains 0.687–0.711 along the
reported path. For the mask path, displacement reaches 3.809 robust RMS and the
warning reaches 0.920. Those values reinforce the off-manifold tag.

The [IG convergence table](S06a_artifacts/ig_convergence.csv) compares 64 with
32 steps. Canonical rank correlation is **0.99925** for low-pass IG and
0.99275–0.99607 for the other references. Low-pass IG completeness residual is
median $7.90\times10^{-5}$, q90 $9.34\times10^{-4}$, and maximum 0.00409 native
units. Its normalized mask infidelity is **0.09697**, the lowest among eligible
path methods.

## Baseline sensitivity

Baseline choice materially changes the answer. On canonical $\tilde f$, the
absolute-map rank correlation against robust-constant IG is **0.432** for the
selected low-pass map, **0.536** for matched-observed IG, and **0.749** for
medoid IG. This is an understood sensitivity, not evidence that one baseline is
physically correct. Low-pass was selected because it passed both faithfulness
directions, randomization, toy, and symmetry checks and then had the lowest
infidelity under the fixed tie-break; its endpoint and path remain off manifold.

The 64-row pilot selected medoid IG because low-pass insertion did not beat its
random control there. The same fixed rule selected low-pass IG on the registered
128 rows. This pilot-to-production instability is a negative result: S06b must
retain medoid and robust-constant IG as baseline sensitivity analyses and must
not treat the primary map as baseline-independent.

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

## Uncertainty

[Grouped uncertainty](S06a_artifacts/grouped_uncertainty.csv) uses 500 resamples
of whole `equilibrium_files` and reports sample-mean absolute attribution for
every method, function, and floor stratum. For the selected canonical maps, the
all-row estimates and 95% intervals are 0.001171 (0.000856–0.001465) for
low-pass IG and 0.1123 (0.1103–0.1143) for the dimensionless mask.

S06a has exactly one preregistered member, so these intervals cover equilibrium
sampling only. The full S06 acceptance clause requiring both member and
equilibrium uncertainty is deliberately pending S06b tasks 5–7, which run the
selected methods across the registered top 10 and a wider sensitivity sample.

## Failed checks and corrections

Four failures were found before the final registered run and retained in tests
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

Three deliberate post-run mutations turned the focused suite red and were
reverted: dropping robust channel scales failed the exact scaled-gradient test;
resampling individual rows instead of `equilibrium_files` failed the grouped
bootstrap support test; and restoring the absolute AUC denominator failed the
negative-direction curve test.

## Negative results and interpretation limits

- Cyclic grouped occlusion fails the toy position threshold: average precision
  is 0.6125 versus the registered 0.75 minimum. On the real canonical panel its
  deletion margin is -1.620, despite a positive insertion margin, so it is not
  selected.
- Matched-observed IG also fails real deletion faithfulness (margin -2.070), even
  though its endpoint is an observed geometry. An observed endpoint does not
  make the interpolation path physical.
- Baseline agreement is only moderate; selected low-pass IG correlates 0.432
  with robust-reference IG.
- Canonical and original maps are not interchangeable: their rank correlation
  is 0.875 for low-pass IG and 0.605 for the mask, and original-$f$ maps fail
  exact equivariance as S02 predicts.
- The periodic mask's strong AUC margins occur on edits with robust displacement
  up to 3.809 and high support warnings. They diagnose the trained function,
  not a realizable plasma intervention.
- Parameter randomization is a full reset rather than the more granular
  layer-by-layer cascade; it establishes response, not where that response
  begins.

## Acceptance criteria

| PLAN criterion | S06a verdict and evidence |
| --- | --- |
| Selected methods beat random/control maps on toy recovery and faithfulness | **Pass.** Both selected methods have toy channel top-1 1.0 and position AP 1.0 versus random AP 0.0442; canonical all/stable/unstable deletion and insertion margins are positive, with exact values above and in `benchmark_metrics.csv`. |
| Selected methods respond to parameter randomization | **Pass.** Canonical absolute-map rank correlation is 0.406 for low-pass IG and 0.099 for the mask. |
| Baseline sensitivity is understood | **Pass, with a strong limitation.** Low-pass/robust map correlation is 0.432; all four IG baselines and Expected Gradients remain published sensitivity analyses. |
| Methods meet symmetry behavior permitted by S02 | **Pass.** Canonical equivariance relative RMS is $1.03\times10^{-4}$ and $2.70\times10^{-7}$ for the selected pair; original-$f$ errors are 0.821 and 0.820 and are retained. |
| Uncertainty includes model and equilibrium sampling | **Pending S06b by the explicit S06a/S06b split.** S06a provides 500-draw equilibrium-file intervals for its one registered member; model sampling begins in S06b. |
| Signed and absolute summaries are distinguishable | **Pass.** Full HDF5 maps retain signs; every metrics row has `signed` and `contribution_valued`; VarGrad and masks are marked magnitude-only. |
| No feature is called common without agreement | **Pass.** S06a names no common feature; member agreement is deferred to S06b. |

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
faithfulness numbers exactly (within platform float tolerance). The first 16
maps to compare are in `selected_review_maps.h5`, with explicit axes and row IDs.

**Checkable from committed artifacts alone.** Method selection and its fixed
rule are in `selected_methods.json`; all benchmark/stratum numbers are in
`benchmark_metrics.csv`; all deletion/insertion doses and support warnings are
in `faithfulness_curves.csv`; 500-draw equilibrium intervals are in
`grouped_uncertainty.csv`; convergence is in `ig_convergence.csv`; analytic
controls are in `toy_controls.json`; hashes, package versions, rows, member,
checkpoint, and dataset fingerprints are in the committed manifest. The
artifact tests independently pin these schemas and hashes.

**Not checkable off the researcher's machine, and why.** The exact periodic-mask
map uses matched backgrounds selected from 512 equilibrium-unique S01 reference
rows outside the panel and therefore outside the review slice. The PCA support
warnings use those same rows, with a 384/128 fit/calibration split. The nearest
reviewer proxy is to select non-panel slice rows as an alternative observed
background, rerun the mask, and compare toy recovery, canonical equivariance,
randomization response, and the sign of deletion/insertion margins; agreement
would show the selection is not peculiar to the unavailable background, but it
cannot reproduce the registered digits. Full 22-method maps are 9.6 MB in the
ignored run; only the selected 16-row maps are committed.

## Deferred

- S06b tasks 5–7: top-10 signed maps, member/equilibrium hierarchical
  uncertainty, fixed/varied and covariate strata, ranks 11–100 sensitivity, and
  attribution-stability versus validation-$R^2$.
- LRP, because rule coverage was not documented.
- No feature-level scientific claim; S06a selects estimators, while S06b and S07
  establish member agreement and physics alignment.
