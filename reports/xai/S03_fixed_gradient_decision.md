# S03 fixed-gradient premise conflict

## Decision

**Approved by the researcher on 2026-08-20.** The shared loader now supplies
fixed-gradient rows at the checkpoint's demonstrated training convention,
`a/L_T = +3`, and the affected S00–S02 fixed-row artifacts have been refreshed.
The work was done as a focused prerequisite task on its own branch, before S05,
and is not folded into any numbered step.

The original question, recorded before the gate was answered, was: *approve
changing the shared fixed-gradient loader from `a/L_T = -3` to the checkpoint's
demonstrated `+3` training convention and refreshing affected S00–S02 fixed-row
artifacts?* The recommendation was yes. It was accepted without change.

## What changed

| Where | Change |
| --- | --- |
| `itg_nn/data.py` | `load_hdf5_rows` no longer negates `a/L_T` for fixed rows. The old behaviour is reachable as `legacy_fixed_marker=True` and documented as **off-manifold**. |
| `scripts/xai_s01_audit.py` | The panel-metadata builder records the value the network is actually given. |
| `reports/xai/S01_artifacts/panel_metadata.csv` | `a_over_LT_model` for the 1,000 fixed twins: `-3.0` → `3.0`. The run asserts no other cell and no varied row moved. |
| `reports/xai/S02_artifacts/` | Fixed-gradient strata in `shift_symmetry_summary.csv`, `phase_average_exactness.csv` and `parity_symmetry.csv` recomputed; `summary.json` exactness checks updated. Varied rows carried through byte-identical. |
| `tests/data/review_slice.h5` | Rebuilt. The stored `reference_fixed_*` predictions had been computed at `-3`, so a reviewer checking a fixed-row number against the slice would have been confirming it against saturation. |
| `PLAN.md` | The contested-premise paragraph is resolved; S07 task 3 no longer instructs retaining the marker as a learned interaction. |

S03's own fixed-gradient results stay **withdrawn**. They are not restored by
this correction, because restoring them would mean rerunning S03's perturbation
ladder on fixed rows, which is a step's worth of work and not a repair. The
varied-gradient S03 ladder was never affected.

## Evidence

Three independent lines agree, and the third resolves a conflict the original
memo left open.

### 1. The serialized training inputs

The trusted serialized `TensorDataset` at
`neural_networks/cyclic_invariant_models/train_val_test_dataset_5_pre_2.pth`
holds 199,637 rows and **no negative `a/L_T` anywhere** — train, validation, or
test. Its first 100,705 rows are the fixed-gradient half and are all exactly
`+3`; the remainder carry the varied distribution, minimum 0.000129, maximum
14.43.

### 2. The checkpoint's own behaviour

On the 1,000 fixed-gradient rows of S01's registered panel, all 100 members
([`fixed_gradient_convention.csv`](S03_fixed_gradient_artifacts/fixed_gradient_convention.csv)):

| Fixed input | Ensemble mean | Ensemble std | Prediction range | Ensemble R² | Member R² range |
| --- | --- | --- | --- | --- | --- |
| Legacy marker, `a/L_T = -3` | -2.033 | 0.014 | [-2.101, -2.004] | **-12.08** | [-19.74, -8.15] |
| Training convention, `a/L_T = +3` | 2.006 | 1.145 | [-2.000, 5.590] | **0.9883** | [0.9726, 0.9867] |

The panel's fixed targets have mean 2.005 and standard deviation 1.161, with
2.2% at the clipped-log floor. At `-3`, **100%** of predictions land within 0.05
of the floor and 0 of 100 members reach R² > 0.9; at `+3`, all 100 do.

This is the decisive argument, and it is an argument about training, not about
fit quality. Fixed-gradient rows are roughly half the training set. A network
trained with those rows marked `-3` would have had to learn to predict them at
`-3`, and would fit them there. These members instead collapse to a nearly
constant floor value — the signature of an input they never saw.

### 3. Provenance of the legacy marker — why the source comment is not evidence

The original memo dismissed a legacy source comment describing the negative
marker. That was the right conclusion but not yet a demonstration, because the
comment describes real code: `Cyclic_net.py:152` does execute
`out_data1["tprims"] = -1 * out_data1["tprims"]`, and calls it a deliberate
trick. Two artifacts date it after training.

- The saved `.pth` above records `test_dataset` with **19,965** rows, the full
  unfiltered test split. That file is written under `if not os.path.exists(...)`,
  so it comes from a run in which the `if dataset[i][2] < 0` filter dropped
  nothing — that is, before the negation existed.
- The published inference log `out-35793251.log` records
  `New test_dataset: 19965 9785`, so the negation *was* active in the run that
  produced `pred_vs_actual_plot_pre2.pdf` and the R² of 0.98931.

The negation was therefore added after the ensemble was trained, as a
bookkeeping device so the paper's test score would be computed on
varied-gradient rows only. It never described the inputs the members learned
from. It is preserved in this repository only as `legacy_fixed_marker=True`, an
explicitly off-manifold probe.

`VALIDATION.md` carries the same finding in the reference-validation record,
where a reader checking the legacy pipeline will meet it.

## What this changes scientifically

The earlier fixed-twin sensitivity measurements were made where the members are
saturated, so they measured floor wobble rather than the trained function. The
earlier practice of classifying fixed rows as stable or unstable from their
*target* while their predictions sat at the floor was incoherent for the same
reason, and is now meaningful again.

With `+3`, fixed-gradient rows stop being a marked class and become ordinary
in-distribution points at $(a/L_T, a/L_n) = (3, 0.9)$. The fixed/varied twin
pair is now the clean constant-drive comparison that S07 task 3 and S13 task 1
were written to use. That is a capability the correction restores, not only
damage it repairs.

## Blast radius across S00–S04

Traced row by row, not assumed.

| Step | Ran the model on fixed rows? | Effect |
| --- | --- | --- |
| S00 | No — 64 varied rows | None |
| S01 | No | One committed metadata column was wrong. The `fixed_pair_split` strata are *varied*-row predictions stratified by the twin's split, so no S01 number moves. |
| S02 | Yes | Real. The fixed-row shift and parity sensitivity claims were measured at the floor and are corrected here. Varied-row results, accuracy, the canonical $\tilde f$, the density-rotation fix and the bottleneck census are unaffected — the bottleneck is computed from geometry alone and never sees $a/L_T$. |
| S03 | Yes | Already self-withdrawn before this correction. |
| S04 | No — explicitly varied-only | None |
| `VALIDATION.md` | No | Untouched. The R² = 0.989 figure reproduction uses varied test rows only, so the reference validation is silent about the fixed convention and remains valid. |

## What the S02 refresh found

Recomputed on the 1,000 fixed panel rows with all 100 members, unnormalized RMS
output change over the 93 non-subgroup shifts:

| Stratum | q10 | median | q90 | Withdrawn median |
| --- | --- | --- | --- | --- |
| Fixed, all (n = 1,000) | 0.0607 | **0.0818** | 0.0979 | 0.01167 |
| Fixed, stable/near-floor (n = 23) | 0.0123 | 0.0269 | 0.0490 | — |
| Fixed, unstable (n = 977) | 0.0612 | 0.0826 | 0.0988 | — |
| Varied, all (n = 1,000), unchanged | 0.1002 | 0.1382 | 0.1681 | 0.1382 |

The fixed/varied median ratio moves from **0.084** to **0.59**. The earlier
claim — repeated as a headline finding in S02's executive summary — that fixed
rows are about ten times less shift-sensitive was an artifact of the floor. The
real drive dependence is a factor of about two, which is a result worth keeping
but an ordinary one. Correct-parity RMS change on fixed rows moves the same way,
from 0.00067 / 0.00455 / 0.01009 to **0.0429 / 0.0516 / 0.0640** against
0.0519 / 0.0607 / 0.0708 on varied rows.

Two of S02's exactness checks are computed as a maximum over the whole panel and
therefore moved. The subgroup maximum rose from 9.54e-6 to **1.33e-5** and the
32-versus-96-phase maximum fell from 1.07e-6 to **9.54e-7**; both still pass the
registered `atol=rtol=2e-5`. The first is the meaningful one: under the old
convention half the panel was constant, and a constant output is trivially
shift-invariant, so the check was partly vacuous. It is now a stricter test on
rows that actually respond, and S02's exact-subgroup claim survives it.

Two self-checks establish that these differences are the convention and not the
refresh pipeline. Re-predicting sampled varied members reproduced the registered
S02 run's stored values with maximum absolute difference **exactly 0.0**; and
rebuilding all 1,782 committed varied rows across the three CSVs through the
same statistic builders reproduced them with maximum absolute difference
**exactly 0.0**. The published diff is 1,782 replaced fixed rows and no changed
varied byte.

## Cost

As estimated: one focused session, no GX rerun and no retraining. The convention
run is 64 seconds on CPU for 100 members × 1,000 rows × 2 conventions. The S02
fixed-stratum refresh recomputes 96 shifts and two parity transforms for 100
members on 1,000 fixed rows — **44 minutes of member compute**, summed from its
per-member log — while the varied half is read back from the registered S02 run
rather than recomputed, which is what keeps it under an hour instead of over
five.

Note that `s02_fixed_refresh_manifest.json` reports a wall time of 53 seconds.
That is the resumed re-run which reused the cached per-member predictions and
added the varied-rebuild check; it is not the cost of the computation. The
44-minute figure above is the honest one.

## Reviewer reproduction

**Recomputable on the slice.** All 1,000 panel fixed rows are inside
`tests/data/review_slice.h5`.

- The convention result itself: `tests/xai/test_fixed_gradient_convention.py`
  recomputes it on the first 128 slice rows with the committed checkpoint, and
  asserts R² > 0.9 at `+3`, collapse to within 0.5 of a constant at `-3`, and a
  mean absolute separation above 1.0 between the two. No dataset needed.
- The slice's own `reference_fixed_*` predictions are now made at `+3`;
  `tests/xai/test_review_slice.py` checks fresh predictions against them, and
  refuses any slice that does not carry the `fixed_gradient_convention`
  attribute.
- The full-panel numbers in the table above are over 1,000 rows rather than 128,
  so a reviewer recomputing on the slice should expect the same qualitative
  separation and R² within a few thousandths, not the identical digits.

**Checkable from committed artifacts alone.**

- Per-member and ensemble statistics under both conventions:
  [`fixed_gradient_convention.csv`](S03_fixed_gradient_artifacts/fixed_gradient_convention.csv),
  200 rows.
- Run provenance, hashes, package versions and wall time:
  [`manifest.json`](S03_fixed_gradient_artifacts/manifest.json) and
  [`s02_fixed_refresh_manifest.json`](S03_fixed_gradient_artifacts/s02_fixed_refresh_manifest.json).
- The serialized-training-tensor statistics and the S01 refresh's
  changed-cell census:
  [`summary.json`](S03_fixed_gradient_artifacts/summary.json).
- That the S01 refresh moved only `a_over_LT_model` on fixed rows: the
  `s01_panel_metadata_refresh.changed_cells` block, and the git diff of
  `panel_metadata.csv`, which is 1,000 changed lines for 2,000 rows.
- The refresh's row-replacement rule:
  `tests/xai/test_fixed_gradient_refresh.py` pins that varied rows come through
  byte-identical and that a changed stratum size is refused.

**Not checkable off the researcher's machine, and why.**

- The legacy provenance in section 3 rests on
  `train_val_test_dataset_5_pre_2.pth` (515 MB) and `out-35793251.log`, which
  live in the external legacy directory and are not in this repository. Their
  fingerprints are recorded in `summary.json`. The nearest proxy a reviewer can
  compute is section 2 on the slice: if the members fit fixed rows at `+3` and
  collapse at `-3`, the training convention is settled regardless of what the
  legacy files say, and section 3 only explains how the wrong comment arose.
- The S02 fixed-stratum refresh runs on the 1,000-row panel and reads back the
  registered S02 run's 87 MB `predictions.h5` from a git-ignored directory. The
  run asserts, and records in its summary, that re-predicting sampled varied
  members reproduces the stored values with maximum absolute difference exactly
  0.0 — so any change in the fixed rows is the convention and not the pipeline.
  A reviewer can recompute a single member's fixed-row shift RMS on slice rows
  and compare it with that member's row in `shift_symmetry_summary.csv`.

## Deferred

- S03's withdrawn fixed-gradient perturbation results are not regenerated. They
  need S03's ladder rerun on fixed rows, which is step-sized work; S03's report
  and executive summary continue to mark them withdrawn.
- S02's report text is updated with the corrected fixed-row numbers, but S02's
  registered run is not re-executed end to end. Nothing outside the three
  refreshed CSVs and the two exactness checks depends on fixed-row predictions.
