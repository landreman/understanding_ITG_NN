# S12 — Distil the networks into invariant physical formulas

## Result

A compact table of 17 physically named, exactly cyclic-invariant features can
reproduce the outputs of three top stored-validation members with similar
equilibrium-held-out fidelity. Explainable Boosting Machine (EBM; a regression
model that learns one inspectable curve per feature plus a small registered set
of pairwise interactions) $R^2$ is **0.8603, 0.8561, and 0.8636** for members
`2864601_0.437`, `2864601_0.371`, and `2864601_0.409`. Their median is
**0.8603**. The same feature table gives $R^2=0.8632$ for the three-member mean.
Grouped-bootstrap 95% intervals are **[0.8453, 0.8745]**, **[0.8400, 0.8704]**,
**[0.8486, 0.8774]**, and **[0.8482, 0.8765]**, respectively. These are
out-of-fold predictions: each row was predicted by a model fitted without its
`equilibrium_files` group.

The fidelity does not belong only to the ensemble mean. All three members land
in a narrow 0.0075-wide $R^2$ range, so the compact vocabulary reproduces
multiple high-ranked members. It is nevertheless incomplete: residual standard
deviations are **0.744–0.768** native clipped-log units, materially larger than
the original members' roughly 0.28–0.31 residual standard deviation against GX
reported in earlier steps.

Individual bottleneck units are much less uniformly compressible. Of **64**
units across the three members, five are exactly dead on the panel and therefore
have zero error but undefined $R^2$. The 59 live units have median held-out
$R^2=0.5942$; only **13/64** reach 0.8. Member medians are **0.7821, 0.4528,
and 0.6037**. The compact vocabulary therefore captures important shared output
behavior without giving a simple physical formula to every internal unit. This
negative result agrees with S05's finding that most units did not receive a
supported one-name interpretation.

The model-output and physical-prediction questions are separated. Fitting the
same EBM directly to true native clipped $log Q$ gives held-out
$R^2=0.8392$, below the three member fits but still high. This number measures
physical predictive accuracy of the named feature model. The member-output
numbers measure fidelity to what the neural networks learned. Neither is formed
from $Q$ or `exp(prediction)`. Its grouped-bootstrap 95% interval is **[0.8210,
0.8561]**.

A nested feature-set comparison changes the interpretation of that headline.
Across the three members, member mean, and true target, the two drives alone
give $R^2=0.5839$–0.6124. Adding the paper's $\log f_Q$ raises this to
**0.7639–0.7868**. Adding $f_{\rm stab}$ and
$\log\langle|\nabla x|\rangle$ raises it only to **0.7783–0.8055**, and all 17
main effects reach **0.7798–0.8069**. The full five-interaction model reaches
**0.8392–0.8636**, a paired gain of **0.0753–0.0796** over the baseline trio.
Thus most of the added fidelity is not a new standalone geometry motif: it is
the registered drive-dependent interaction structure. The single
$a/L_T\times\log f_Q$ interaction accounts for part, but not all, of that gain
($R^2=0.7906$–0.8116). Exact scores and grouped-bootstrap intervals for all six
models and all five primary targets are in
[subset_fidelity.csv](S12_artifacts/subset_fidelity.csv).

The most recurrent terms are the two drives. Both $a/L_T$ and $a/L_n$ occur in
the top five terms in **30/30 equilibrium-bootstrap refits** for every member,
the member mean, and true clipped $log Q$. Among geometry terms, $f_{\rm stab}$
recurs in **0.83–0.90** of the member bootstrap refits, and the S05-inspired
25-point-window peak of the $f_Q$ integrand recurs in **0.83, 0.83, and 0.87**.
For the true target those recurrences are 0.90 and 0.70. The geodesic-curvature/
compression feature is less stable: member recurrence is **0.27–0.60**. Because
the feature table is correlated, EBM term importance is not a unique division
of credit. A particularly important example is the paper's $\log f_Q$: despite
adding roughly 0.17–0.18 $R^2$ beyond the two drives in the nested fits, it is
in the full model's top five in **0/30, 0/30, and 1/30** member refits, and 0/30
for the mean and true target. Correlated descendants and registered interactions
redistribute the full model's importance credit; low top-five recurrence there
does not mean $\log f_Q$ carries no predictive information.

Exactly five pairwise interactions were registered before the pilot. On the
downsampled committed effect grids, the $a/L_n$ × bad-curvature/compression
surface has the largest root-mean-square signed effect for all three members,
**0.440–0.497** native units. This is an EBM description of observed rows, not
a single-channel intervention or a causal plasma result. The complete signed
main-effect curves and five signed interaction surfaces are in
[ebm_effects.csv](S12_artifacts/ebm_effects.csv), with the visual atlas in
[ebm_effects.png](S12_artifacts/ebm_effects.png).

PySR did not produce a symbolic-expression frontier. PySR 1.5.10 requires Julia
1.10.3–1.11, while the installed Julia is 1.12.6. Forcing that executable failed
before fitting with an unsatisfiable `OpenSSL_jll ~3.0` constraint. PLAN makes
the EBM half the minimum viable deliverable when the Julia toolchain fights
back, so installing a second Julia channel and rerunning PySR is deferred rather
than changing the workstation toolchain inside this step.

## Estimand and cohort

The model estimand is each stored-validation top-three member's S02 canonical
exactly shift-invariant function

$$
\tilde f_m(X,g_T,g_n)=\operatorname{MLP}_m(\bar u_m(X),g_T,g_n)
$$

in native $\max(\log Q,-2)$ units. Member outputs, individual bottleneck units,
and signed residuals are retained before any mean. The ensemble mean is a
separate target. The physical comparator is the true GX
$\max(\log Q,-2)$ on the same rows.

The cohort is S01's frozen 1,000-row varied-gradient interpretation panel: one
tube from each of 1,000 distinct `equilibrium_files`, with **240** stable or
near-floor rows and **760** unstable rows. S02 supplies the canonical invariant
member. S05 supplies the named windowed motif vocabulary, S09 supplies the
compact completeness concepts and drive interactions, and S10 motivates testing
multiple members. The three members were selected deterministically as the first
three of S01's registered stored-validation top-ten list to cap production cost
at 64 bottleneck-unit targets times five folds, plus five primary targets; they
were not selected by distillation fidelity. Development and production used the
external HDF5 source, never `tests/data/review_slice.h5`.

The registered run is `distillation-top3-panel1000`. Its committed
[manifest](S12_artifacts/manifest.json) records CPU execution, seed 20260824,
all 1,000 parent row IDs, the three member IDs, **342.97 s** wall time,
dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`,
and unchanged checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.
The production path used Python 3.12.4, torch 2.4.1, NumPy 1.26.4, h5py
3.11.0, SciPy 1.13.1, Captum 0.9.0, and `interpret-core` 0.7.8.
The manifest reports `git_tracked_dirty=true` because this registered run was
made before its review-fix commit. To make the exact executed state checkable,
it hashes the runner and distillation module as well as the upstream canonical
symmetry and bottleneck modules; artifact tests match all four hashes to the
committed files.

## Methods

### Versioned invariant feature table

[feature_registry.csv](S12_artifacts/feature_registry.csv) is version `S12-v1`.
It contains 17 scalar features:

- $a/L_T$, $a/L_n$, $\log f_Q$, $f_{\rm stab}$, and
  $\log\langle|\nabla x|\rangle$;
- bad-curvature/compression variance, mean square, and co-location;
- absolute geodesic curvature and geodesic-curvature/compression;
- robustly scaled parallel roughness, with every geometry channel divided by
  S01's interquartile range before derivatives are combined;
- expected non-DC parallel Fourier mode for $B$, compression, and curvature;
- mean absolute local shear; and
- 25-point circular-window peaks for the $f_Q$ integrand and absolute geodesic
  curvature.

Every feature is an invariant reduction of an equivariant or pointwise
operation. Shifting all channels together by any grid offset leaves the table
unchanged to floating-point tolerance. All rows carry the
`observed-comparison` validity tag: they summarize real geometries and do not
represent interventions.

### Grouped EBM fitting

Five folds assign complete `equilibrium_files`; no equilibrium occurs in both
train and test within a fold. Each EBM uses 128 main-effect bins, 32 interaction
bins, four outer bags, learning rate 0.03, 300 maximum rounds, and minimum leaf
size 3. The five fixed interactions are:

1. $a/L_T$ × $\log f_Q$;
2. $a/L_T$ × bad-curvature/compression;
3. $a/L_T$ × absolute geodesic curvature;
4. $a/L_n$ × bad-curvature/compression; and
5. $\log f_Q$ × absolute geodesic curvature.

This particular frozen panel has exactly one tube per equilibrium, so a grouped
split or bootstrap has the same row arithmetic as a row-level operation here.
The implementation nevertheless groups explicitly by `equilibrium_files`, and
a behavioral test with two tubes per equilibrium verifies that every resampled
fit contains complete equilibrium blocks.

Bottleneck units were attempted first as PLAN requested. The runner then fit
each member output, the member mean, and true clipped $\log Q$. All target and
regime metrics are in [fidelity.csv](S12_artifacts/fidelity.csv). The committed
[primary residuals](S12_artifacts/primary_residuals.csv) retain every row's
target, out-of-fold prediction, signed residual, fold, and equilibrium for the
member, mean, and true-target fits.

Six nested fits quantify what each concept family adds: drives only; the drives
plus $\log f_Q$ baseline trio; the five paper variables; all 17 main effects;
all 17 plus the five registered interactions; and the baseline trio plus only
$a/L_T\times\log f_Q$. They reuse the same grouped folds for paired comparison.
[subset_fidelity.csv](S12_artifacts/subset_fidelity.csv) reports point scores,
2,000-draw grouped-bootstrap 95% intervals, and paired gains over the trio.

### Stability

For primary targets, 30 bootstrap refits resample whole
`equilibrium_files` with replacement. [term_recurrence.csv](S12_artifacts/term_recurrence.csv)
records how often each feature is among the five most important main effects,
plus its median and 95% bootstrap importance interval. For each bottleneck unit,
[term_importance.csv](S12_artifacts/term_importance.csv) records mean and
standard deviation across the five held-out folds and top-five fold recurrence.
Thirty draws are a stability screen with 0.033 resolution, not a precise tail
probability. Separately, every primary held-out $R^2$ and nested-subset score has
a 2,000-draw grouped-bootstrap 95% interval.

## Stable/near-floor versus unstable rows

Stable-row $R^2$ is not meaningful because the target variance is highly
compressed. It is strongly negative for the member fits (-7.37 to -8.50) and
-3048.7 for true clipped $\log Q$, while the corresponding stable-row MSEs are
finite: **0.643–0.659** for members and **0.852** for the true target. These
negative $R^2$ values are retained, not hidden.

On 760 unstable rows, member fidelity is $R^2=0.8115$–0.8268 with MSE
0.521–0.573. True-target fidelity is $R^2=0.7965$ with MSE 0.615. The formulas
therefore work materially better than a constant on unstable rows, while the
stable stratum supports only error-scale statements.

## Failed checks, negative results, and limits

- Tests were written first. The initial focused run failed all four EBM paths
  with explicit `NotImplementedError`; the symbolic Pareto/recurrence test also
  failed at its explicit stub before implementation.
- The first pilot reached all ten bottleneck fits but failed while exporting an
  interaction grid: InterpretML supplies bin-edge names that can be one element
  longer than the score axis. A regression test now pins score-cell indexing,
  and the pilot and production runs were repeated.
- Five bottleneck units are dead; 46/64 units do not reach held-out $R^2=0.8$.
  Compact output fidelity does not imply a one-feature formula for every unit.
- Stable-row $R^2$ is negative because the denominator is compressed. MSE is
  primary there.
- EBM importance can move among correlated variants of curvature/compression.
  Recurrence supports a feature family, not a unique algebraic identity.
- The five interaction surfaces are descriptive fits on observed geometries.
  No feature was edited, so no perturbation is being called physical.
- The panel consists of interpolation equilibria that appeared in training.
  Grouped held-out EBM fidelity tests distillation generalization across this
  panel, not neural-network generalization to unseen equilibrium families.
- PySR 1.5.10 could not initialize under Julia 1.12.6. No symbolic expression,
  complexity frontier, or expression recurrence is claimed.

## Mutation testing

Four deliberate mutations turned the named focused test red and were reverted:

1. assigning folds by row rather than `equilibrium_files` separated sibling
   tubes and failed the repeated-group equality assertion;
2. dropping S01's per-channel IQR scales before combining parallel derivatives
   failed the 1,000× channel-rescaling control; and
3. exponentiating each member's native clipped-log output failed the signed
   native-output wiring test; and
4. replacing whole-equilibrium bootstrap draws with raw row draws failed the
   recorded-fit complete-block assertion.

## Acceptance criteria

| PLAN criterion | Verdict | Number or artifact |
| --- | --- | --- |
| “formulas generalize by equilibrium” | **Pass for the EBM MVD.** | Five `equilibrium_files`-grouped held-out folds give member $R^2=0.8561$–0.8636 (grouped-bootstrap 95% bounds across members: 0.8400–0.8774) and true-target $R^2=0.8392$ [0.8210, 0.8561]; [fidelity.csv](S12_artifacts/fidelity.csv), [subset_fidelity.csv](S12_artifacts/subset_fidelity.csv), and [primary residuals](S12_artifacts/primary_residuals.csv). |
| “reproduce multiple top members rather than only the mean” | **Pass.** | Three separate member fits are 0.8603, 0.8561, and 0.8636; the mean fit is separately 0.8632. |
| “are stable enough to interpret” | **Qualified pass for replicated feature families and drive-dependent interactions, not every unit.** | Both drives recur 1.00; $f_{\rm stab}$ 0.83–0.90; windowed $f_Q$ integrand 0.83–0.87. The full interactions add 0.0753–0.0796 $R^2$ over the trio, while $\log f_Q$ itself has 0/30, 0/30, and 1/30 top-five member recurrence because correlated terms redistribute credit. Only 13/64 bottleneck units reach $R^2\ge0.8$; [term_recurrence.csv](S12_artifacts/term_recurrence.csv), [subset_fidelity.csv](S12_artifacts/subset_fidelity.csv), and [term_importance.csv](S12_artifacts/term_importance.csv). |
| “clearly separate model fidelity from physical predictive accuracy” | **Pass.** | Separate target kinds and residuals give member fidelity 0.8561–0.8636 versus true clipped-$\log Q$ accuracy 0.8392; no exponentiation. |

The complete EBM MVD deliverables are the versioned feature registry, fidelity
and residual tables, signed main/interactions atlas, term stability tables, and
this report. The PySR-specific deliverables are deferred below under PLAN's
explicit toolchain fallback.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s12-pilot XDG_CACHE_HOME=/private/tmp/cache-s12-pilot \
  .venv-xai/bin/python scripts/xai_s12_distillation.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s12-prod XDG_CACHE_HOME=/private/tmp/cache-s12-prod \
  .venv-xai/bin/python scripts/xai_s12_distillation.py
.venv-xai/bin/python scripts/xai_s12_distillation.py --resume
conda run -n 20240629-01-ML make check
```

The CLI supports `--config`, `--members`, `--rows`, `--device`, `--seed`,
`--batch-size`, `--resume`, `--output-dir`, and dataset/checkpoint overrides.

## Reviewer reproduction

**Recomputable on the slice.** All 1,000 parent row IDs are S01 panel rows in
`tests/data/review_slice.h5`. Translate them with
`load_review_slice_index().slice_rows()` before loading. The reviewer can
recompute the 17 invariant features, three canonical member predictions and 64
bottleneck units, grouped folds, EBM out-of-fold predictions, all six nested
fits, grouped fidelity intervals, regime metrics, and term recurrence. The
analytic cyclic, null-feature, repeated-equilibrium, complete-block bootstrap,
native-output, live-InterpretML term-order, interaction-grid, and robust-scale
controls run in `test_distillation.py` and `test_distillation_script.py` (the
live-InterpretML check skips only when that optional package is absent).

**Checkable from committed artifacts alone.** Every headline $R^2$, MSE, unit
count, signed residual, feature recurrence, effect curve, interaction surface,
source hash, package version, and PySR deferral reason is committed under
[S12 artifacts](S12_artifacts/). The member and true-target scores recompute
exactly from `primary_residuals.csv`; bottleneck scores are in `fidelity.csv`.
The nested scores and their paired gains are in `subset_fidelity.csv`. The
manifest hashes all nine registered outputs and records source/config/input
identity.

**Not checkable off the researcher's machine, and why.** Matching the external
source HDF5 bytes to its SHA-256 requires the 678 MB file. The nearest proxy is
the mapped 1,000-row review slice: agreement with the committed features,
canonical outputs, and out-of-fold predictions checks every scientific EBM
headline but not the absent source-file bytes. Reproducing the PySR failure also
requires Julia 1.12.6; it is a setup failure, not a scientific result.

## Deferred

- **PySR symbolic-expression Pareto frontier and expression bootstrap.** PySR
  1.5.10 requires Julia 1.10.3–1.11; the workstation has 1.12.6, and forcing it
  fails on `OpenSSL_jll ~3.0`. Follow-up cost is installing a supported Julia
  channel, validating PySR against this checkpoint, then running compact-feature
  symbolic fits with equilibrium-held-out scoring and grouped bootstrap.
- **Equivariant-operation symbolic reduction of each $\bar u_{m,c}$.** This is
  part of the same deferred PySR follow-up. The completed EBM unit fits establish
  which units are plausible targets: start with the 13 units at $R^2\ge0.8$ and
  retain the 46 weaker live units and five dead units as controls.

Nothing from PLAN's EBM MVD was dropped.
