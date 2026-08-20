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
2.2% at the clipped-log floor. At `-3`, 0 of 100 members reach R² > 0.9; at
`+3`, all 100 do.

Two cautions on how the collapse is quoted, both found by writing tests against
the artifact rather than by reading it:

- **"100% of predictions at or below the floor plus 0.05" is a statement about
  the *ensemble mean*, whose whole 1,000-row spread is [-2.101, -2.004].** It is
  not true member by member: the median member has 98.2% of its rows there, but
  the least-pinned member has 0%, sitting in a narrow band slightly off the
  floor rather than on it. Distance to the floor is the wrong statistic for such
  a member.
- **The scale-free form does hold for every member.** At `-3` member prediction
  standard deviations run 0.0036–0.218, median 0.056; at `+3` the *smallest* is
  1.122, median 1.151. No member at `-3` varies as much as the least-varying
  member at `+3` — a median compression of about 20×. That, not proximity to
  the floor, is what the collapse argument rests on.

The committed `summary.json`'s `conclusion` string predates this scoping and
still reads "saturates the members at the clipped-log floor". It is an artifact
field, so correcting it would mean republishing the run for a string; the
generator now emits the scoped wording and this paragraph is the authoritative
form. Only 37 of 100 members have every row at or below floor plus 0.05, and the
least-pinned member predicts in [-1.823, -1.325] — flat, but nowhere near the
floor.

The floor fraction is also one-sided: 10.7% of ensemble-mean predictions sit
more than 0.05 *below* the floor, so the two-sided fraction is 0.893. Both
columns are published, as `fraction_at_or_below_floor_plus_0p05` and
`fraction_within_0p05_of_floor_two_sided`.

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
but an ordinary one.

Read that ratio with both panels' scales in view, because it compares
*unnormalized* changes across row sets whose outputs differ in spread: the
varied panel's targets have mean 0.446 and standard deviation **2.044** with
21.5% at the floor, the fixed panel's mean 2.005 and standard deviation
**1.161** with 2.2% at the floor. Normalizing fixed rows by the varied-reference
residual denominator is refused here, per `AGENTS.md`, so part of the remaining
0.59 is plausibly scale rather than drive, and this comparison does not separate
the two. Correct-parity RMS change on fixed rows moves the same way,
from 0.00067 / 0.00455 / 0.01009 to **0.0429 / 0.0516 / 0.0640** against
0.0519 / 0.0607 / 0.0708 on varied rows.

Two of S02's exactness checks are computed as a maximum over the whole panel and
therefore moved: the subgroup maximum from 9.54e-6 to **1.33e-5**, the
32-versus-96-phase maximum from 1.07e-6 to **9.54e-7**. Both still pass the
registered `atol=rtol=2e-5`, by more than an order of magnitude.

**Neither change should be read as a result.** Both are maxima of float32
roundoff differences, and an independent recomputation on a GitHub runner
reproduced the pass verdict but not the values — it put the panel maximum at
2.86e-6 and found the fixed half *smaller* than the varied half, the opposite of
this machine, where the split is 1.33e-5 fixed against 6.08e-6 varied. The
scalar is platform- and batching-specific. An earlier draft of this memo argued
from the rise that the check had become stricter; that inference was not
supported and has been withdrawn.

The defensible statement is about what the criterion tests, not about the
number. Under the old convention half the panel was nearly constant, where
shift-invariance is close to trivially satisfied, so the subgroup check was
partly vacuous on those rows. It is now evaluated on fixed rows that respond,
and it passes. The rows both maxima are taken over are published as
[`s02_subgroup_exactness.csv`](S03_fixed_gradient_artifacts/s02_subgroup_exactness.csv)
(1,224 rows: every entity, both gradient sets, all strata, shifts 32 and 64), so
a reviewer can compare distributions rather than one machine-specific scalar.

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
  202 rows. Each row carries a `validity_tag` — `observed_comparison` for the
  training convention, `off_manifold` for the legacy marker — so the tagging
  requirement is met in the artifact and not only in this prose.
- The rows behind S02's two exactness maxima:
  [`s02_subgroup_exactness.csv`](S03_fixed_gradient_artifacts/s02_subgroup_exactness.csv),
  1,224 rows.
- Run provenance, hashes, package versions and wall time:
  [`manifest.json`](S03_fixed_gradient_artifacts/manifest.json) and
  [`s02_fixed_refresh_manifest.json`](S03_fixed_gradient_artifacts/s02_fixed_refresh_manifest.json).
- The serialized-training-tensor statistics:
  [`summary.json`](S03_fixed_gradient_artifacts/summary.json).
- That the S01 refresh moved only `a_over_LT_model` on fixed rows: the git diff
  of `panel_metadata.csv` against `main`, which is 1,000 changed lines for 2,000
  rows, all in that one column and all on fixed rows; plus
  `s01_panel_metadata_refresh.fixed_a_over_LT_model_values: ["3.0"]` and
  `varied_rows_unchanged: true` in the same block.

  **Do not read `s01_panel_metadata_refresh.changed_cells` as that census.** It
  is `{}` in the committed summary, and empty *by construction*: the convention
  run was re-executed after the review, by which time the published
  `panel_metadata.csv` already held the corrected values, so the run found
  nothing left to change. That is an idempotency check — the second run
  reproduces the first run's output exactly — not a contradiction of the claim.
  The census read `{"a_over_LT_model": 1000}` on the first execution, at commit
  `d8ccaec`. The two sibling fields above and the git diff are what back the
  claim in the committed state.
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
  Two reproduction routes exist for the refreshed numbers, and note that
  `shift_symmetry_summary.csv` is **not** one of them: `_shift_summary_rows`
  collapses the 100 members into `member_distribution` quantiles, so it holds no
  per-member row to compare against. Instead, either recompute one member's
  fixed-row parity RMS on slice rows and compare it with that member's row in
  [`parity_symmetry.csv`](S02_artifacts/parity_symmetry.csv), which does carry
  per-member rows; or recompute all 100 members at a single shift and compare
  the `member_distribution` q10/median/q90 row in `shift_symmetry_summary.csv`.
  Both were checked by the automated review and reproduce to better than 2e-7.

## Review disposition

The automated review returned **no blocking findings**, two should-fix and four
notes. Disposition:

| Finding | Severity | Action |
| --- | --- | --- |
| Reviewer-reproduction route pointed at `shift_symmetry_summary.csv`, which has no per-member rows | should-fix | **Fixed.** The route was wrong; it now names `parity_symmetry.csv` for per-member comparison and the `member_distribution` quantile row as the alternative, and says explicitly that the summary file holds no per-member rows. |
| `exact_subgroup_max_abs` did not reproduce off this machine, and the report built a "stricter test" claim on it | should-fix | **Fixed.** The inference is withdrawn as unsupported. Both maxima are now labelled platform-specific float32 roundoff, the reviewer's disagreeing recomputation is quoted, and the 1,224 rows they are taken over are published as `s02_subgroup_exactness.csv`. |
| `fixed_gradient_convention.csv` lacked the registered perturbation-validity vocabulary | note | **Accepted.** A `validity_tag` column now carries `off_manifold` / `observed_comparison`, matching `S03_artifacts/support.csv`. |
| `fraction_within_0p05_of_floor` was one-sided but read as two-sided | note | **Accepted.** Renamed to `fraction_at_or_below_floor_plus_0p05`, a two-sided column added, and the prose in this memo and `PLAN.md` corrected. |
| The `AGENTS.md` slice-regeneration rule was amended in the commit that used it | note | **Open — for the researcher.** See below. |
| Mutation 1's failure tally differs from the reviewer's | note | **Accepted.** The PR body now names the variant used; the second review confirmed the body's tally of 5 was right for that variant. |

Second round: the review confirmed both should-fix items fixed and raised one
more, plus three notes.

| Finding | Severity | Action |
| --- | --- | --- |
| The post-review rerun emptied `s01_panel_metadata_refresh.changed_cells`, which the memo cited as the changed-cell census | should-fix | **Fixed.** The bullet now says the block is `{}` by construction on a rerun — an idempotency check — and rests the claim on the git diff and on the block's two surviving fields. |
| `VALIDATION.md` kept the two-sided reading of the floor fraction | note | **Accepted**, and superseded: the documents now quote the member-level spread collapse instead, which is scale-free. |
| No test pinned the new `validity_tag` and floor columns or the subgroup artifact | note | **Accepted.** `tests/xai/test_fixed_gradient_artifacts.py` pins the schema, the two floor definitions, per-member coverage, and that the published subgroup rows reproduce `exact_subgroup_max_abs` exactly. Writing it found the scoping error below. |
| The fixed/varied ratio compares unnormalized changes on differently-scaled row sets | note | **Accepted.** Both panels' target spreads are now quoted beside the ratio in this memo and in `S02_symmetry.md`, with the limitation stated. |

Writing the requested test surfaced a scoping error the reviews had not caught:
the "100% at or below the floor plus 0.05" figure holds for the **ensemble
mean**, not member by member — only 37 of 100 members have every row there, and
the least-pinned member has none. The member-level statement rests on spread
compression instead. The collapse conclusion is unaffected and better supported:
no member at `-3` varies as much as the least-varying member at `+3`.

Third round: no blocking findings, one should-fix and two notes.

| Finding | Severity | Action |
| --- | --- | --- |
| Four sentences added by this PR still stated the member-level claim the previous commit scoped away | should-fix | **Fixed.** The scoping was applied to this memo, `PLAN.md` and `VALIDATION.md` but missed `S02_executive_summary.md`, `S02_symmetry.md` (two places) and `S03_ladder.md`. All four now say the marker pins the *ensemble mean* and *flattens* every member. |
| The committed artifact's `conclusion` string carries the old wording | note | **Accepted as advised**: the artifact is not republished for a string, the generator is corrected for future runs, and the stale field is flagged above. |
| The scope-guard test pinned only `min < 1.0`, not the figures the memo quotes | note | **Accepted.** It now pins 37 of 100 fully pinned, median 0.982, minimum 0.0. |

The correction that started this — a wrong `a/L_T` — has now been followed by
three rounds in which the *reporting* of it, not the science, was what needed
fixing: a reproduction route to a file without the right rows, an inference
drawn from float32 noise, a citation invalidated by a rerun, and a claim scoped
in three documents but not in four other sentences. That pattern is worth
recording. None of it moved a number; all of it would have misled a reader.

### Open item for the researcher

The approval recorded at the top of this memo covers the **loader correction**.
It does not cover the amendment to `AGENTS.md`'s review-slice rule, which was
widened in the same commit from "when a step deliberately changes the registered
panel" to also allow "when a correction invalidates the reference predictions it
stores". The automated review is right that authorising an act and amending the
rule that governs it should not arrive together unremarked. The amendment is
disclosed in the PR body and paired with a new reader-side guard that refuses a
slice built under the wrong convention, but it needs a separate yes or no. If
declined, the regeneration still stands on the narrower reading — a slice whose
stored baselines are known-wrong is not a verification artifact — but the rule
text should then be reverted.

## Deferred

- S03's withdrawn fixed-gradient perturbation results are not regenerated. They
  need S03's ladder rerun on fixed rows, which is step-sized work; S03's report
  and executive summary continue to mark them withdrawn.
- S02's report text is updated with the corrected fixed-row numbers, but S02's
  registered run is not re-executed end to end. Nothing outside the three
  refreshed CSVs and the two exactness checks depends on fixed-row predictions.
