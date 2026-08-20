# Agent instructions for this repository

This file is the canonical instruction file for both Codex and Claude Code.
`CLAUDE.md` is a symlink to it, so there is only one copy to keep current.

## What this repository is

An inference-only copy of the trained cyclic-convolutional ensemble that predicts
GX ITG heat flux, plus a program of explainable-AI experiments to work out what
geometric features those networks extract. The scientific goal, the numbered
steps, and the acceptance criteria live in `PLAN.md`. The dispatch/review/merge
process lives in `WORKFLOW.md`. Read both before starting a step, along with the
merged reports in `reports/xai/` for the steps you depend on.

Work is dispatched one `PLAN.md` step at a time. If you are implementing, the
sequence is `.agents/skills/step/SKILL.md`, invoked as `/step SNN` or `/step next`.
If you are reviewing, it is `.claude/commands/review-step.md`, which the
`Claude Code Review` GitHub Actions workflow runs automatically on every pull
request. One step, one branch, one PR.

## Environments

- The conda environment `20240629-01-ML` has an editable install of this package
  and its inference dependencies. Use it for plain inference and for
  `python -m pytest`.
- **Do not install anything into `20240629-01-ML`.** XAI extras go in the
  project-local `.venv-xai`, created by `bash scripts/setup_xai_env.sh`. Create
  one per worktree; never share an editable-install venv across worktrees.
- New dependencies go in the `xai` extra of `pyproject.toml` and in
  `requirements/xai.lock`, with a compatibility smoke test and the tested version
  recorded in the step report.

## Do not touch

- `models/cyclic_ensemble_pre2.pt` — hash it in manifests; never rewrite it.
- The external HDF5 dataset at the canonical path registered in `PLAN.md` — read
  only, never modified or copied into the repository.
- `paper/` — the published manuscript and its figures.
- Unrelated untracked user files, including the existing `output/` directory.
- `git reset --hard`, forced branch deletion, and recursive deletion are off
  limits in research worktrees.

## Non-negotiables for every experiment

- Explain the native output `max(log Q, -2)` — not `Q`, not `exp(prediction)`.
- Keep member-level signed results before any aggregation. An ensemble mean can
  hide opposing mechanisms.
- Bootstrap and split by `equilibrium_files`, not by flux tube.
- Report stable/near-floor rows separately from unstable rows; a third of the
  varied-gradient set sits at the clipped-log floor.
- Never compare raw gradients across the seven geometry channels without an
  explicit robust scale; their magnitudes differ by three orders of magnitude.
- Tag every perturbation as exact-symmetry, observed-comparison, plausibly-local,
  or off-manifold. Model sensitivity to an off-manifold edit explains the network,
  not the plasma.
- Write large output to `output/xai/<step>/<run_id>/` (git-ignored) with a
  complete `manifest.json`; commit only small reports and figures. **Commit a
  verbatim copy of the registered run's `manifest.json` to
  `reports/xai/SNN_artifacts/manifest.json`** — pass `published_dir=` to
  `RunArtifacts.finalize`. It is a few kilobytes, and without it the reviewer
  can only confirm that the code *would* write a complete manifest, never that
  the run actually did. Pilot manifests stay git-ignored.
- `tests/data/review_slice.h5` is a **verification artifact, not a development
  set**. The automated review recomputes reported numbers on it. Do not develop
  against it, tune explanation hyperparameters on it, select a method with it, or
  report a result from it: doing so would make the reviewer's independent check a
  restatement of your own choices. Use the real dataset through `.venv-xai` for
  all scientific work. Its row IDs are slice-local — go through
  `itg_nn.xai.review_slice.load_review_slice_index().slice_rows()`, never pass
  cohort row IDs straight to a reader pointed at the slice. Regenerate it only
  when a step deliberately changes the registered panel, with
  `scripts/build_review_slice.py`, and say so in the PR body.
- Negative and contradictory results are kept, not dropped.
- Every reported number must say which code path produced it when more than one
  exists. Where an optional dependency selects the path — the Captum estimator
  versus the in-repo fallback, say — record the method in the artifact column,
  not only in prose, so a reviewer can tell which one it is holding.
- Say what a reviewer cannot check. See `## Reviewer reproduction` below.

## Definition of done for a step

A step is done when all of the following hold. Not most of them.

- Every acceptance criterion in the step's `PLAN.md` subsection is answered by a
  number or a named artifact, one by one, in the report's `## Acceptance criteria`
  section — not by a restatement of the method.
- Tests were written before the implementation, and include an analytic cyclic toy
  function with known relevant features and at least one control that must come
  out null. They pass without the external HDF5 dataset, because CI and the
  automated review both run without it; gate anything that truly needs the dataset
  behind a skip on its absence.
- The tests were shown to be able to fail. Two or three mutations that matter for
  this step were applied, confirmed to turn the suite red, and reverted, and they
  are named in the PR body. A mutation that does not turn the suite red is a
  finding to report, not a mutation to quietly swap out.
- `make test` is green locally and CI is green on the pushed branch.
- The run directory under `output/xai/<step>/<run_id>/` carries a complete
  `manifest.json`, and every conclusion in the report traces to it.
- `reports/xai/SNN_<name>.md` and `reports/xai/SNN_executive_summary.md` are
  committed, with negative results, failed checks, and interpretation limits kept,
  a `## Deferred` section that says what was dropped or says "nothing", and a
  `## Reviewer reproduction` section as specified below.
- `reports/xai/SNN_artifacts/manifest.json` is committed and is the registered
  production run's manifest, matching the `run_id` the report quotes.
- `git status --short` is clean of unrelated files, no large generated artifact is
  staged, and the model file, the external dataset, and `paper/` are untouched.

## Reviewer reproduction

The automated review runs on a GitHub runner with no external dataset and no
`.venv-xai`. It has the checkout, `pytest`, `tests/data/review_slice.h5`, and
captum, scipy and pandas. Anything outside that it cannot check, and the review's
"what I could not check" list is a standing cost of the workflow, not a
formality. Shrink it deliberately.

Every step report carries a `## Reviewer reproduction` section with three lists:

- **Recomputable on the slice.** Headline numbers whose rows are inside the
  2,000-row slice, each with the artifact column or the few-line calculation
  that reproduces it. Say which rows, so the reviewer does not have to infer the
  mapping. These are the numbers a reviewer is expected to reproduce exactly.
- **Checkable from committed artifacts alone.** Numbers that live in a committed
  CSV, `summary.json`, or `manifest.json` and need no dataset access.
- **Not checkable off the researcher's machine, and why.** Claims resting on
  rows outside the slice — the 9,785-row reference cohort, the full 100,705
  rows — or on a run too expensive to repeat. Name the nearest proxy the
  reviewer *can* compute on the slice and what agreement would mean, so an
  unverifiable claim still gets a partial check instead of a shrug.

If a headline number lands in the third list and could have landed in the first
by publishing one more small artifact, publish the artifact. A number reachable
only from a git-ignored run directory is not traceable, whatever the report says.

## Git and pull requests

Branch from up-to-date `main`, one step per branch, named `codex/xai-sNN-name` or
`claude/xai-sNN-name`. Commit code, tests, configs, and small report artifacts;
never large output. Push, get CI green, then `gh pr create --fill` with the body
structure the `step` skill specifies.

Never merge your own step PR. Never `git reset --hard`, force-delete a branch, or
delete recursively in a research worktree.

## Responding to the automated review

The `Claude Code Review` workflow posts a findings table on every push. Fix every
finding marked **blocking** or **should-fix**, push, and iterate until it returns
neither; **note** items are your judgement. Disagreeing is allowed and sometimes
correct — reply on the PR with the artifact, line, or test that shows the finding
is wrong. Silently ignoring a finding is not. When the table comes back clean,
stop: do not merge, and do not start the next step.

Never satisfy a review by relaxing a tolerance, narrowing the cohort, marking a
test skipped, or dropping a control. If a finding can only be answered that way,
it is a decision for the researcher.

## Reviewing, when you are the reviewer rather than the implementer

Follow `.claude/commands/review-step.md`. Assume the work is plausible,
well-documented, and passing, and that if its conclusion is wrong its tests are
wrong in the same direction. Do not edit code and do not merge. Say plainly when
you find nothing blocking, and say plainly when a claim can only be checked by
rerunning a dataset-backed calculation you cannot run here.

## When to stop and ask

Proceed autonomously through ordinary numerical and coding choices. Stop and ask
only at the decision gates listed in `PLAN.md` and `WORKFLOW.md`, or if a step
would materially change the registered estimand, cohort, or baseline family. Keep
the question short and include evidence, estimated cost, and a recommendation.

If a step exceeds its `PLAN.md` effort budget, deliver the step's minimum viable
deliverable completely, record what was dropped under a `## Deferred` heading in
the step report, and stop. Do not deliver a shallow version of everything.
