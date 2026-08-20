---
description: Adversarial review of a PLAN.md step PR against its acceptance criteria and the interpretation contract
argument-hint: "[PR number, or blank for the current branch]"
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(git status:*), Bash(gh pr:*), Bash(make*), Bash(pytest*), Bash(python*), Read, Grep, Glob
---

Review $ARGUMENTS as an adversarial reviewer. Assume the implementation is
plausible, well-formatted, passing CI, and written by a capable agent, and that if
its scientific conclusion is wrong then its tests are wrong in the same direction.
Your job is to find that. Style is not your job.

Read the diff, then the step's subsection of `PLAN.md` for its acceptance criteria
and MVD, `PLAN.md`'s `## Interpretation contract` and `## What not to conclude`,
`AGENTS.md`'s non-negotiables, and the merged prerequisite reports in
`reports/xai/`. Read those sources yourself; do not take the PR body's or the
report's characterization of them.

You are running without the external HDF5 dataset and without `.venv-xai`, so you
cannot rerun the pilot or the production run. You have three things instead.

`pytest`, which runs on synthetic fixtures and the committed checkpoint. Every
committed artifact, manifest, and summary JSON. And **`tests/data/review_slice.h5`**
— 2,000 real rows: the whole 1,000-row S01 interpretation panel plus 1,000 sibling
flux tubes of the same equilibria, with `scripts/build_review_slice.py` recording
how they were chosen. Use it to recompute, not merely to re-read. If the PR reports
a correlation, an R², an attribution ranking, a bootstrap interval, or a
per-channel statistic on panel rows, compute it yourself on the slice and compare.
A number that does not reproduce is a blocking finding.

Load it through `itg_nn.xai.review_slice`:

```python
from itg_nn.xai.review_slice import REVIEW_SLICE_PATH, load_review_slice_index
index = load_review_slice_index()
rows = index.slice_rows(parent_row_ids)   # raises if a row is not in the slice
```

**Slice row IDs are not parent row IDs.** The cohort registry in
`reports/xai/S01_artifacts/cohorts.json` is written in parent row IDs; passing one
straight to a reader pointed at the slice silently reads a different flux tube.
Always go through `slice_rows`, which raises instead. Never edit or regenerate the
slice during a review.

The slice covers the panel and its sibling tubes, not the 9,785-row reference
cohort and not the full 100,705 rows. Where a claim can only be checked by
rerunning a dataset-backed calculation on rows the slice does not hold, say that
explicitly in the finding rather than assuming it holds or assuming it fails.

Answer these in order, each with a verdict and the evidence you checked.

**1. Is the explained function the registered one?**
The native output is `max(log Q, -2)`. Check every place the code takes a
gradient, a difference, or a ratio: is it on that quantity, or has `Q`,
`exp(prediction)`, or an unclipped log crept in? Check the baseline and reference
values too — a cohort-conditional mean computed in the wrong units is the same
bug wearing a different hat. Confirm the function matches what S02's decision gate
settled on, not what was convenient.

**2. Would the test fail if the science were wrong?**
For each new test: construct an implementation that passes it and is nevertheless
wrong. If you can, that is a finding. Check specifically that the suite would go
red if the bootstrap grouped by flux tube instead of `equilibrium_files`, if the
robust per-channel scale were dropped, if a sign were flipped, if members were
averaged before the signed statistic, or if a control that must be null were fed
real labels. The PR body claims certain mutations were verified — apply one
yourself, run the affected test, and confirm it goes red. Report what you ran.

**3. Leakage, pseudoreplication, and cohort integrity.**
Every split, fold, bootstrap resample, and permutation must be grouped by
`equilibrium_files`. Flux tubes from one equilibrium on both sides of a split is a
blocking finding no matter how good the reported score is. The slice carries 780
equilibria with more than one tube precisely so you can test this rather than read
it: run the PR's resampling code on those rows both ways and check the interval
widths actually differ. If they do not, the grouping is not doing anything. Check that the cohort
and panel are the ones S01 registered, that no hyperparameter was tuned on the
same rows used for the final claim, and that near-floor and unstable rows are
reported separately rather than pooled.

**4. Symmetry handled, not assumed.**
Cyclic shifts: is the quantity equivariant where it should be, invariant where it
should be, and is that demonstrated rather than asserted? Check any aggregation
over positions for an implicit choice of origin. Check that a claimed exact
symmetry is exact to roundoff in a test, not merely small.

**5. Are the perturbations honestly tagged?**
Every intervention must be labelled exact-symmetry, observed-comparison,
plausibly-local, or off-manifold, in the artifact and in the report. An
off-manifold edit that the report discusses as if it told us about the plasma is a
blocking finding. So is a feature replacement described as a physical
counterfactual without an equilibrium-consistent construction.

**6. Do the numbers in the report exist in the artifacts?**
Recompute at least two headline numbers on the review slice where the claim
concerns panel rows, and spot-check the rest against the committed summary JSON or
figure data. Check that `manifest.json` is complete — commit, dirty status,
command, config, seeds, versions, device, dataset fingerprint, checkpoint hash,
member and row IDs, wall time, output hashes — and that the checkpoint hash
matches the committed model. A conclusion that cannot be traced to a manifest is a
finding regardless of how reasonable it sounds.

**7. Was an acceptance criterion quietly weakened?**
Diff tolerances, sample counts, member counts, `xfail`/`skip` markers, and config
values against `main`. Compare the report's `## Acceptance criteria` section, line
by line, with `PLAN.md`'s `Accept when:` text. Any loosening, silent narrowing of
the cohort, or criterion answered by restating the method instead of a number is a
blocking finding regardless of how reasonable the justification sounds. If the PR
uses the MVD escape, check that `## Deferred` names what was dropped and that what
remains is genuinely complete rather than a shallow version of everything.

**8. Statistics.**
Uncertainty on every headline number, computed the grouped way. Multiple
comparisons acknowledged where many units, channels, or concepts were tested.
Controls present and reported even when they came out the wrong way — check the
diff and the run directory index for a control that was computed and then not
discussed. Negative and contradictory results kept, per `AGENTS.md`.

**9. Overclaiming.**
Compare the executive summary and the report's headline against `PLAN.md`'s
`## What not to conclude`. Predictiveness stated as physical causality,
decodability stated as use, an ensemble mean presented as a mechanism, or a
network fact presented as a plasma fact — each is a finding, and the executive
summary is where they hide.

**10. Repository hygiene and undocumented decisions.**
`models/cyclic_ensemble_pre2.pt`, the external dataset, `paper/`, and unrelated
untracked files must be untouched — check the diff, not the PR body's claim. No
large generated artifact committed. Any ambiguity in `PLAN.md` that the
implementer resolved by choosing, where the choice changes the estimand, the
cohort, or the baseline family, needs a decision memo, not a silent pick.

Output a table of findings: severity (**blocking** / **should-fix** / **note**),
location as `file:line`, what is wrong, and what to do about it. Then one line:
**merge / fix first / needs researcher decision**.

Post the table even when you find nothing — say plainly that there are no blocking
findings. Do not invent findings to seem thorough, and do not soften a blocking
finding into a suggestion. Do not edit code, and do not merge.
