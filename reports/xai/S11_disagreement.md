# S11 — Ensemble disagreement and failure modes

## Result

Model-to-model disagreement is structured, but it is a substantially better
diagnostic of itself than of held-out prediction error. On S01's frozen
1,000-equilibrium varied-gradient panel, the median all-100-member population
standard deviation was **0.1049 native units**, while the median absolute error
of the ensemble mean was **0.0942**. A five-fold equilibrium-held-out ridge
model using every feature frozen in the S11 config explained **42.5%** of
spread variation but only **9.68%** of absolute-error variation. The latter is
the important negative result: these diagnostics do not make ensemble spread a
calibrated error bar.

The fixed pre-run thresholds identify **8/1,000 (0.8%) common-mode failures**:
rows with all-member spread below 0.15 but ensemble absolute error at least 0.5
native units. Two are stable/near-floor rows and six are unstable. The other
categories contain 240 high-spread/low-error rows, 76 high-spread/high-error
rows, and 676 unanimous-success rows. These are diagnostic bins at fixed native-
unit cutoffs, not estimated confidence classes.

![Disagreement and common-mode failure atlas](S11_artifacts/failure_atlas.png)

The strongest frozen rank diagnostic was the original model's residual one-
shift symmetry error: Spearman correlation was **0.617** with all-row spread
(500 whole-equilibrium resample interval 0.572–0.664) and **0.482** with absolute
error (0.427–0.533). This does not mean broken symmetry causes prediction error.
It says that geometries where the unsymmetrized network is least invariant also
tend to be geometries where members disagree or err. The canonical S02 function
remained exactly invariant: a random joint circular shift changed all-100 spread
by only **2.13e-8 RMS native units**.

Simulation time variability `Q_stds` is contradictory across output regimes. It
correlates positively with spread on stable/near-floor rows (**0.636**, interval
0.564–0.707) but negatively on unstable rows (**-0.347**, -0.414 to -0.287).
The all-row correlation is 0.240. The sign reversal rules out a single pooled
interpretation such as “noisier simulations always make the models disagree.”

S10-derived concept-selective activation dispersion has a modest all-row
association with spread (**0.241**, 0.182–0.307) and a weaker association with
absolute error (**0.0889**, 0.0314–0.150). In contrast, dispersion within S10's
matched motifs is null for both spread (**0.0028**, -0.0587–0.0599) and error
(**-0.0178**, -0.0783–0.0373). S03's support-warning score is likewise null for
spread (**0.0130**) and error (**-0.0132**). These nulls are retained rather than
replacing the frozen features after seeing residuals.

## Estimand and cohort

For every registered member (m), the explained function is S02's exactly
shift-invariant canonical model

\[
\tilde f_m(X,g_T,g_n)=\operatorname{MLP}_m(\bar u_m(X),g_T,g_n)
\]

in native \(\max(\log Q,-2)\) units. The disagreement estimand is the population
standard deviation

\[
s_f(X)=\operatorname{std}_{m=1}^{100}\tilde f_m(X,g_T,g_n),
\]

with `ddof=0`. It is member dispersion, not a confidence interval. Member
residuals are \(r_m=\tilde f_m-y\) against held-out GX targets. Their input
gradient equals the native member-prediction gradient because the observed
target is constant with respect to an input derivative; the full signed member
axis is retained before any summary.

The cohort is S01's frozen 1,000-row varied-gradient interpretation panel: one
tube from each of 1,000 `equilibrium_files`, including 240 stable/near-floor and
760 unstable rows. All 100 members were fixed by stored validation rank before
the panel residuals were examined. The 14 continuous diagnostic features and
equilibrium class were also frozen in
[the config](../../configs/xai/S11_disagreement.json) before production; every
feature is reported, with no selection on these residuals.

The registered run is `disagreement-all100-panel1000`. Its committed
[manifest](S11_artifacts/manifest.json) records CPU execution, seed 20260824,
all 100 member IDs, all 1,000 parent row IDs, and **2,409.80 s (40.16 min)** wall
time. The dataset SHA-256 is
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`;
the checkpoint SHA-256 is
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.
The run began from commit `fc9581a`; `git_tracked_dirty` is false. The manifest's
broader `git_dirty` flag is true only because the worktree already contained the
researcher's ignored/untracked `output/`, `scratch/`, and notes.

## Methods

### Direct spread and residual gradients

The all-100 tensor of canonical member predictions was differentiated through
the population standard deviation directly. This is a variance attribution: it
asks which local input changes increase or decrease disagreement. It is not the
mean of member-prediction gradients, because standard deviation is nonlinear.

For comparison, member residual gradients were computed for the stored-
validation top 10 on a deterministic 128-row stable/unstable-stratified subset.
All signed `(member, sample, z, channel)` values remain in the ignored registered
HDF5 file, and 16 exact rows are in the committed
[review proxy](S11_artifacts/selected_review_diagnostics.h5). Channel summaries
multiply gradients by S01's robust per-channel IQR scale before comparison.

Spread sensitivity is largest for channel 1, then channels 3 and 4: all-row mean
absolute robust-scaled gradients are **0.001943**, **0.001668**, and **0.001320**.
The top-10 mean-prediction gradient has the same first three channels but is a
different object and larger: **0.009583**, **0.007237**, and **0.005858**. The
member-residual signed means agree across all ten members on channels 0
(10/10 negative), 1 (10/10 positive), 2 (10/10 positive), 4 (10/10 negative),
and 6 (10/10 positive), but split on channels 3 (6/10 positive) and 5 (5/10
positive). This is evidence of opposing local strategies, not yet the task-4
cancellation test deferred below.

### Supported perturbations

Two S03 operators were evaluated for every member and row, retaining signed
member changes:

- `random_joint_shift` is tagged `exact_symmetry`. Median member RMS change is
  **2.08e-7** native units (10th–90th member range 1.73e-7–2.49e-7), and the
  all-member spread RMS change is **2.13e-8**.
- `independent_channel_shifts` is tagged
  `deliberately_off_manifold_diagnostic`. It destroys cross-channel alignment,
  gives median member RMS prediction change **0.815** (0.769–0.890), and raises
  spread by **0.185** on average (spread-change RMS 0.288). Its median signed
  member prediction change is -0.191. This explains the network under an
  unrealizable edit; it is not a causal plasma claim.

### Frozen diagnostic relationships

The frozen features are support-warning score, equilibrium class, \(a/L_T\),
\(a/L_n\), `Q_stds`, original-function symmetry error, S10 motif-activation
dispersion, concept-selective activation dispersion, `nfp`, `iota`, `shat`,
`d_pressure_d_s`, `aspect`, `rho`, and `aspect/rho`.

Support warning uses S03's robust channel scaling and 16-component ordinary PCA
(a compressed representation of geometry) fitted on 384 off-panel equilibria
and calibrated on another 128; no panel equilibrium enters that fit. Motif
dispersion standardizes each matched unit across panel rows and averages the
within-motif cross-member standard deviation. Concept dispersion projects each
top-10 bottleneck onto the complete S05 concept vocabulary without choosing a
concept from residual performance, then measures cross-member activation
dispersion.

Every feature/outcome pair is reported separately for all, stable/near-floor,
and unstable rows in
[diagnostic associations](S11_artifacts/diagnostic_associations.csv). Point
estimates are Spearman rank correlations. The 500-draw ranges resample whole
`equilibrium_files`; they are sampling-sensitivity intervals, not confidence
intervals for model uncertainty. Because this panel has one tube per
equilibrium, grouped and row resampling are numerically identical here, but the
grouped implementation is tested on sibling tubes.

The multivariable diagnostic is a five-fold ridge regression (a linear model
whose coefficients are shrunk to reduce instability), with feature scaling
learned inside each training fold and whole equilibria held out. It uses every
frozen feature plus equilibrium-class indicators and fixed penalty 1.0. Its
held-out results are spread \(R^2=0.425\), MAE 0.0559 native units; absolute-
error \(R^2=0.0968\), MAE 0.156. No model or feature was chosen using these
held-out residuals.

## Output-regime and class differences

Stable/near-floor and unstable rows are never pooled without also publishing
their separate values. Common-mode failure counts are 2/240 and 6/760. High-
spread/high-error counts are 13/240 and 63/760.

Equilibrium class 0 has the largest all-row mean spread (**0.151**) and error
(**0.212**); class 3 has mean spread 0.135 and the smallest mean error 0.151.
These are descriptive class means, not adjusted causal effects. Full class and
regime values are in
[equilibrium-class diagnostics](S11_artifacts/equilibrium_class_diagnostics.csv).

## Uncertainty and interpretation limits

- The member distribution reflects networks trained on the same data and
  architecture family. It is not an independent sample of plausible worlds.
- Model spread is never called a confidence interval. The grouped resample
  ranges quantify how a descriptive association moves when equilibria are
  resampled; they do not calibrate predictive coverage.
- Diagnostic correlations are exploratory and no multiplicity-adjusted claim is
  made for individual rows. The fixed full table prevents residual-driven
  feature selection but does not turn correlation into causality.
- Original-function symmetry error is a diagnostic covariate. The explained
  canonical function has exact shift invariance, and correlation does not show
  that unsymmetrized shift error causes GX error.
- `Q_stds` is an observed simulation-variability proxy, not label-free input;
  its opposing stable/unstable signs prohibit a pooled mechanism claim.
- S10 motif and concept summaries describe internal network evidence. They do
  not establish that an edited equilibrium is physically realizable.
- Fixed failure thresholds provide transparent case bins but are not optimized
  operating points or calibrated warning probabilities.

## Failed checks, negative results, and corrections

- Tests were written first and initially failed all scientific paths with
  explicit `NotImplementedError` stubs.
- The first pilot completed gradients but failed when its 64 rows contained one
  equilibrium class and class-indicator construction received an empty column
  list. The design now accepts a zero-column indicator block; a regression test
  pins the case. No cohort, threshold, or result was changed.
- The pilot's multivariable held-out \(R^2\) values were negative (-0.983 spread,
  -0.387 error), correctly showing that 64 rows were insufficient for scientific
  conclusions. Production was launched only after the numerical and symmetry
  checks passed, not because the pilot relationship looked favorable.
- The first post-production `make check` attempt aborted while importing
  PyTorch, before test collection. A direct `.venv-xai` PyTorch import then
  succeeded, and the unchanged full retry passed all 270 tests. The same
  one-off import abort occurred before one mutation check; its unchanged retry
  reached the intended assertion failure. This is retained as an environment
  startup transient rather than hidden from the verification record.
- S03 support warning is null for both spread and error.
- S10 matched-motif activation dispersion is null for both outcomes.
- The frozen multivariable diagnostics explain only 9.68% of held-out absolute-
  error variation.
- `Q_stds` reverses association sign between output regimes.
- Common-mode failures exist despite low all-member spread.

## Mutation testing

Three deliberate mutations were checked after implementation and reverted:

1. changing the registered all-member population spread from `ddof=0` to the
   sample estimator `ddof=1` failed the native-unit spread test (0.707/0.354
   instead of 0.500/0.250 on the two-member fixture);
2. dropping S01's robust channel scales and comparing raw gradients failed all
   eight values in the signed scaling fixture; and
3. assigning folds by tube row instead of `equilibrium_files` split every
   sibling pair across two folds and failed the grouped-split check.

## Acceptance criteria

| PLAN criterion | Verdict | Number or artifact |
| --- | --- | --- |
| “residual analyses use held-out targets without selecting models or features on those same residuals” | **Pass.** | All 100 members were fixed by stored validation rank; all 15 diagnostics were frozen in config and every continuous feature appears in all outcome/regime combinations. Five-fold splits hold out whole `equilibrium_files`. [Config](../../configs/xai/S11_disagreement.json), [associations](S11_artifacts/diagnostic_associations.csv), and [cross-fit table](S11_artifacts/crossfit_diagnostics.csv). |
| “model uncertainty is not called a confidence interval” | **Pass.** | `summary.json` says “member dispersion, not a confidence interval”; artifact columns label 500-draw ranges `grouped_resample_sensitivity_interval`. [Summary](S11_artifacts/summary.json). |
| “common-mode failure is reported” | **Pass.** | 8/1,000 rows at fixed thresholds, split 2 stable/near-floor and 6 unstable. [Failure categories](S11_artifacts/failure_categories.csv) and [row diagnostics](S11_artifacts/row_diagnostics.csv). |

## Deferred

- **Task 3 detailed case-study narratives.** The MVD protects tasks 1–2. The
  fixed category atlas and every row are published, but selecting and narrating
  individual equilibria would extend the step beyond its one-session budget.
- **Task 4 a formal test of cancellation among opposing but individually
  faithful strategies.** Signed member gradients expose candidate disagreement
  on channels 3 and 5, but faithfulness-conditioned cancellation needs a
  separately tested analysis. No cancellation conclusion is made here.

Nothing from the MVD was dropped.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s11-pilot XDG_CACHE_HOME=/private/tmp/cache-s11-pilot \
  .venv-xai/bin/python scripts/xai_s11_disagreement.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s11-prod XDG_CACHE_HOME=/private/tmp/cache-s11-prod \
  .venv-xai/bin/python scripts/xai_s11_disagreement.py
MPLCONFIGDIR=/private/tmp/mpl-s11-resume XDG_CACHE_HOME=/private/tmp/cache-s11-resume \
  .venv-xai/bin/python scripts/xai_s11_disagreement.py --resume
.venv-xai/bin/python -m pytest tests/xai/test_disagreement.py \
  tests/xai/test_disagreement_script.py tests/xai/test_disagreement_artifacts.py -q
source .venv-xai/bin/activate && make check
```

## Reviewer reproduction

**Recomputable on the slice.** All 1,000 parent row IDs are S01 panel rows in
`tests/data/review_slice.h5`. Translate them with
`load_review_slice_index().slice_rows()` before loading. The reviewer can
recompute all 100 canonical predictions, native residuals, spread/error bins,
`Q_stds`, scalar covariates, original-function symmetry error, motif/concept
dispersion, grouped associations, cross-fit results, and both perturbations.
The exact 16 committed parent IDs in
`selected_review_diagnostics.h5` provide the practical gradient proxy: compare
all-100 predictions and spread gradients plus top-10 signed member-residual
gradients with axes `(member, sample)`, `(sample, z, channel)`, and
`(gradient_member, sample, z, channel)`.

**Checkable from committed artifacts alone.** The 8 common-mode failures, all
category/regime counts, 84 feature/outcome/regime associations, two held-out
ridge results, 252 gradient summaries, 606 perturbation summaries, class table,
atlas, row-level diagnostics, exact hashes, member/row IDs, and 40.16-minute
wall time are committed under [S11 artifacts](S11_artifacts/). Artifact tests
recompute counts, schemas, validity tags, exact-shift null behavior, axes, and
hashes without the external dataset.

**Not checkable off the researcher's machine, and why.** The exact S03 support-
warning scores use 512 off-panel, equilibrium-unique rows outside the review
slice. The nearest slice proxy is to fit the same 16-component support model on
non-panel sibling rows, then check whether its warning remains null against
spread and error; agreement would show the null is not peculiar to the
unavailable support cohort, but cannot reproduce the registered digits. The
full signed `member_level_diagnostics.h5` and 200,000 signed perturbation rows
are git-ignored; the 16-row HDF5 proxy checks their axes, signs, native units,
and direct-autograd wiring but not every archived byte. Recomputing the full
all-100/1,000-row gradient took 40.16 CPU minutes and is too expensive for an
ordinary automated review.
