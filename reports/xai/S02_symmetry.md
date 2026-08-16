# S02 — Symmetry, the canonical invariant model, and the equivariant density

## Status and decision

Complete. The researcher confirmed **`invariant_tilde_f` (`tilde_f`) as the
canonical model on 2026-08-16**. `InvariantMember.forward()` now evaluates

`tilde_f_m(X, g_T, g_n) = MLP_m(mean_z rho_m(X), g_T, g_n)`.

Later position-resolved studies must report this exactly shift-invariant model
and the original member `f_m`, with their difference; they must not silently
substitute one for the other. The registered API also exposes the 32-phase
output average `bar_f_m`. Every result here concerns the native output
`max(log Q, -2)`, never `Q` or `exp(prediction)`.

The ignored production run is
`output/xai/S02/symmetry-all100-panel2000-reference9785/`. Its manifest hashes
the read-only external dataset, protected checkpoint, predictions, tables, and
figures. The compact machine-readable results are in
[`S02_artifacts/summary.json`](S02_artifacts/summary.json), with all published
tables in the same directory.

## Registered methods and cohorts

The shift analysis used all 100 members and every shift `S_k`, `k=0,...,95`, on
the frozen S01 interpretation panel: 1,000 varied-gradient rows from distinct
`equilibrium_files` and their 1,000 fixed-gradient twins. Signed predictions
were retained with member, sample, shift, and transform axes in the ignored
HDF5 artifact. Shift error is `f_m(S_k X)-f_m(X)` and is normalized by that
member's residual standard deviation on the full varied-gradient reference
cohort.

Accuracy and cost were measured for `f_m`,
`bar_f_m = mean_{k=0,...,31} f_m(S_k X)`, and `tilde_f_m` on all 9,785 varied
reference rows. Results retain all members and separately report the 3,170
stable/near-floor rows (`target <= -1.9`) and 6,615 unstable rows. Paired
uncertainty resampled the 8,113 `equilibrium_files` 500 times, never individual
flux tubes.

The full-resolution density uses a stride-one à trous version of the trained
five-block convolution/max-pooling chain. At block `j`, both convolution and
two-point max pooling use dilation `2^(j-1)`, so all 32 pooling phases remain on
the 96-point grid. The resulting `(sample, bottleneck_unit, z)` tensor is
`rho_m`; its position mean is `bar_u_m`, the input to the unchanged trained MLP
head in `tilde_f_m`.

Parity is the exact diagnostic operator `P`: grid-anchored `z -> -z`, with sign
flips on channels 3 and 5. Plain reversal without those sign flips is the
matched wrong-parity control. Both are exact-symmetry diagnostics of the model,
but parity is only approximate in the observed data; its data mismatch is
therefore reported independently.

## Cyclic-shift symmetry

The pooling subgroup at shifts 0, 32, and 64 is invariant to a maximum absolute
member-output change of **9.54e-6**, passing the registered `atol=rtol=2e-5`
tolerance. Averaging all 96 shifts and the 32 distinct pooling phases agrees to
**1.07e-6** maximum absolute error. All later symmetrization therefore uses and
must be labeled as **32-phase**, not 96-phase.

For non-subgroup shifts, the pooled member/shift distribution of RMS output
change divided by each member's own reference residual standard deviation has
10th/median/90th percentiles **0.264 / 0.366 / 0.446**. The ensemble mean hides
much of this member sensitivity: across non-subgroup shifts its median RMS
change is 0.01030 native-output units (0.0453 residual standard deviations),
with maximum 0.01202 (0.0528). The ensemble spread itself changes by a median
RMS of 0.01041. These are distinct estimands and are retained separately in
[`shift_symmetry.csv`](S02_artifacts/shift_symmetry.csv).

![All-shift member symmetry](S02_artifacts/shift_symmetry.png)

The repeated troughs are a real consequence of the five stride-two pooling
blocks, not evidence of continuous shift invariance. Exact subgroup points are
marked at 0, 32, and 64.

## Original and invariant model fidelity

All three ensemble functions remain highly accurate on the varied reference
cohort:

| Function | R2, all | Residual std, all | Residual std, stable | Residual std, unstable | Cost (us/sample) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f` | 0.9893107 | 0.227456 | 0.148565 | 0.255641 | 396 |
| `bar_f` | 0.9893036 | 0.227529 | 0.148337 | 0.255805 | 11,164 |
| `tilde_f` | 0.9892924 | 0.227614 | 0.148469 | 0.256062 | 417 |

Stable-stratum R2 is retained in
[`accuracy.csv`](S02_artifacts/accuracy.csv) but is not interpretable because
that target is nearly constant; residual standard deviation and bias are the
useful summaries there. `tilde_f` costs only 1.054 times `f`, whereas explicit
32-phase `bar_f` costs 28.2 times `f` in this CPU run.

The grouped bootstrap does not resolve an ensemble-accuracy difference. For
`tilde_f - f`, median delta R2 is -1.90e-5 with 95% interval
[-6.93e-5, +3.03e-5], and median delta residual standard deviation is +1.68e-4
with interval [-3.49e-4, +6.84e-4]. For `bar_f - f`, the corresponding values
are -8.27e-6 [-4.99e-5, +2.88e-5] and +8.61e-5
[-3.09e-4, +5.30e-4]. These intervals cross zero.

At member level, however, symmetrization improves residual standard deviation
for **all 100 members**. The 10th/median/90th member deltas are
-0.01372/-0.01013/-0.00656 for `bar_f - f` and
-0.01216/-0.00910/-0.00573 for `tilde_f - f`. The small opposite change in the
ensemble mean is a negative result caused by altered cancellation among member
errors, not evidence that individual models worsened.

![Member fidelity comparison](S02_artifacts/accuracy_comparison.png)

The canonical choice is consequently structural rather than a claim of a
resolved ensemble accuracy gain: `tilde_f` is exactly invariant, exposes the
exact density identity below, improves every individual member, has no resolved
ensemble penalty, and is nearly as cheap as `f`.

## Equivariant density

Across all 100 members, the maximum discrepancy in
`mean_z rho_m(X) = bar_u_m(X)` is **1.91e-6**. The maximum discrepancy in
`rho_m(S_k X) = S_k rho_m(X)` is **2.86e-6**. Both pass the registered 2e-5
absolute/relative tolerance. The unit-level signed density is preserved before
any averaging and is the primary position-resolved object for S05 and S07.
Per-member errors are in
[`density_exactness.csv`](S02_artifacts/density_exactness.csv).

## Parity control

Observed data obey the declared parity exactly to numerical precision in even
channels. The two odd channels are approximate: normalized mismatch MSE is
**0.0407** for channel 3 and **0.0774** for channel 5. The wrong plain-reversal
control gives **3.959** and **3.923**, respectively. Full channel results are in
[`parity_data_mismatch.csv`](S02_artifacts/parity_data_mismatch.csv).

For members, correct-parity RMS output change divided by reference residual
standard deviation has 10th/median/90th percentiles
**0.137 / 0.160 / 0.188**. The wrong control is much larger at
**1.135 / 1.271 / 1.432**. At ensemble level the correct transformation changes
the output by RMS 0.00705 (0.0310 residual standard deviations), versus 0.2521
(1.108) for the wrong control. Thus the trained functions strongly prefer the
physical parity rule, but parity is neither exact in the data nor exact in the
model. The signed member results and ensemble spread are retained in
[`parity_symmetry.csv`](S02_artifacts/parity_symmetry.csv).

## Receptive fields and bottleneck census

[`receptive_fields.csv`](S02_artifacts/receptive_fields.csv) records the formal
span, left/right extents, half-grid center offset induced by even kernels,
periodic unique-position support, and global-connectivity flag for every member
after every block. Formal spans range from 3–17, 8–48, 20–105, 58–210, and
126–392 through blocks 1–5. No member is global after blocks 1–2; 3 are global
after block 3, 89 after block 4, and all 100 after block 5. Periodic unique
support, rather than an unwrapped formal span greater than 96, is the effective
structural receptive field.

The 100 bottlenecks have widths 7–32 (median 26). A dead unit is one whose
maximum absolute `rho` on the S01 panel is at most 1e-12; a near-dead unit is
non-dead but active on less than 1% of panel positions. There are **293 dead**
and **23 near-dead** units among 2,449 total units. Spearman correlations with
stored validation R2 are small: +0.0228 for width, -0.1056 for dead count, and
+0.0555 for near-dead count. The full signed activation census, with stable unit
IDs and quantiles, is in
[`bottleneck_units.csv`](S02_artifacts/bottleneck_units.csv).

## Failed checks, negative results, and limits

- No registered exactness check failed.
- Native arbitrary-shift sensitivity is substantial at member level despite
  the pooling-subgroup exactness; ensemble averaging conceals most of it.
- Neither invariant construction has a statistically resolved ensemble-level
  accuracy advantage. The canonical decision must not be cited as such a gain.
- The fact that every member improves while the ensemble is marginally worse
  shows that member-error cancellation is changed by symmetrization. Signed
  member results must remain available downstream.
- Correct stellarator parity is much closer than the wrong control, but is
  approximate in both data and predictions; it is not a numerical null.
- The receptive-field table is structural support, not the gradient-magnitude
  distribution sometimes called an empirical effective receptive field.
- The S01 reference score is still interpolation at the equilibrium level:
  every reference equilibrium occurs in training. S02 does not repair that
  split limitation or make a physical-causal claim.
- Cost timings are CPU measurements from one run and are implementation and
  batch-size dependent.

## Reproduction and artifacts

Pilot, production, and cached post-processing commands:

```bash
MPLCONFIGDIR=/private/tmp/mpl-s02 .venv-xai/bin/python \
  scripts/xai_s02_symmetry.py --config configs/xai/S02_symmetry.json \
  --pilot --no-publish

MPLCONFIGDIR=/private/tmp/mpl-s02 .venv-xai/bin/python \
  scripts/xai_s02_symmetry.py --config configs/xai/S02_symmetry.json

MPLCONFIGDIR=/private/tmp/mpl-s02 .venv-xai/bin/python \
  scripts/xai_s02_symmetry.py --config configs/xai/S02_symmetry.json --resume
```

The pilot used the top five members, 128 panel rows, and 128 reference rows; it
passed before the all-member run. The production computation used all 100
members, all 2,000 panel samples, all 9,785 reference rows, CPU, seed 20260816,
and about 5.20 wall-clock hours; the stored member timing components total 3.26
hours. Checkpointed predictions make the run resumable member by member. The
post-decision `--resume` pass took 117.8 s and regenerated all small artifacts
and the manifest against code commit `a308da8` without reusing data or checkpoint
contents from a mismatched fingerprint.

The manifest records Python 3.12.4, torch 2.4.1, numpy 1.26.4, h5py 3.11.0,
Captum 0.9.0, the 678,040,404-byte dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`,
and unchanged checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.
`git_tracked_dirty` is false at manifest capture; `git_dirty` is true only
because the required untracked `output/` and not-yet-committed report artifacts
exist. The canonical external dataset and `models/cyclic_ensemble_pre2.pt` were
read only. Large HDF5 output remains uncommitted.

Verification commands:

```bash
conda run -n 20240629-01-ML python -m pytest
.venv-xai/bin/python -m pytest
git diff --check
```

## Acceptance criteria

| Criterion | Evidence | Status |
| --- | --- | --- |
| All 96 shifts, all member/ensemble estimands | 100 members x 2,000 panel rows; signed member predictions retained | Pass |
| Exact subgroup verified | Maximum absolute change 9.54e-6 at shifts 0/32/64 | Pass |
| 32 phases replace 96 phases | Maximum average discrepancy 1.07e-6 | Pass |
| `f`, `bar_f`, and `tilde_f` registered | Public `InvariantMember` API; all-member fidelity/cost table | Pass |
| Stable and unstable rows separate | 3,170 / 6,615 rows in `accuracy.csv` | Pass |
| Grouped uncertainty | 500 paired resamples of 8,113 `equilibrium_files` | Pass |
| Full-resolution `rho` exactness | Mean identity 1.91e-6; equivariance 2.86e-6 | Pass |
| Data and model parity reported | Correct transform plus wrong-reversal control | Pass |
| Receptive fields cover every member/block | Formal and periodic support, asymmetry, and global flags | Pass |
| Bottleneck census covers all members | 2,449 stable unit IDs with signed statistics | Pass |
| Arbitrary shift error scaled fairly | Member error divided by own reference residual std | Pass |
| Canonical function fixed for later steps | Researcher-confirmed `invariant_tilde_f` | Pass |

## Deferred

Nothing. All six S02 tasks and the registered production cohort were completed.
