---
name: step
description: Execute one numbered research step from PLAN.md end to end — branch, tests first, pilot, production run, step report, PR, and iteration on the automated Claude review. Use whenever the user asks to work on a step, do the next step, continue the plan, or names a step number like S04 or S06a.
---

# Execute one PLAN.md step

The user will either name a step (`S04`, `S06a`) or say "next". If they say next,
take the first step in `PLAN.md`'s dependency and concurrency map whose
prerequisites all have merged reports in `reports/xai/` on `main` and whose own
report is absent. Say which step you picked and why before doing anything else.

Work on exactly one step. `AGENTS.md` holds the environment rules, the
non-negotiables, and the stop conditions; `PLAN.md` holds the science; this skill
holds the sequence. Read all three plus the merged reports for the steps you
depend on. Do not restate the plan back to the user.

## 1. Orient

Read, in this order: `AGENTS.md`, the step's subsection of `PLAN.md`, the
`## Interpretation contract` and `## Standard experiment contract and artifacts`
sections of `PLAN.md`, and the `reports/xai/` reports for every prerequisite step.

Before writing code, state in five sentences or fewer:

- the **estimand** — which function of which model on which rows, in native
  `max(log Q, -2)` units;
- the **cohort and panel** you will use, and where they were registered (S01 for
  cohorts, S02 for the canonical function);
- the step's **acceptance criteria**, quoted from `PLAN.md`, and the number or
  artifact each one will be answered with;
- the step's **MVD**, so you know what to protect if you run out of budget; and
- which of `AGENTS.md`'s non-negotiables actually bite in this step.

If a prerequisite report is missing, or its acceptance criteria are not all
answered, stop and say so. Do not start a step whose inputs are not merged.

## 2. Branch and environment

```bash
git switch main && git pull
git switch -c codex/xai-sNN-short-name
bash scripts/setup_xai_env.sh
```

One `.venv-xai` per worktree, never shared. If the step needs a new package, add
it to the `xai` extra in `pyproject.toml` and to `requirements/xai.lock`, run a
real call against this repository's model with it, and record the tested version
in the report.

## 3. Write the tests first

Derive them from `PLAN.md`'s acceptance criteria and interpretation contract, not
from the implementation you are about to write. A useful step's test set almost
always includes:

- an **analytic cyclic toy function** with known relevant features, where the
  method's correct answer is known in closed form;
- a **control that must come out null** — label permutation, random direction,
  scrambled member, or a channel the toy function ignores;
- a check that the code operates on the native output, not `Q` or
  `exp(prediction)`;
- a check that grouping is by `equilibrium_files`, not by flux tube, wherever
  a split, bootstrap, or cross-validation fold is formed; and
- a determinism check: same seed and config, same numbers.

Tests use synthetic fixtures and the committed checkpoint; they must pass without
the external HDF5 dataset, because CI and the review both run without it. Gate any
test that genuinely needs the dataset behind a skip on its absence.

Then run the new tests against the unimplemented code and confirm they fail for
the reason you intend — the missing science, not an import error.

## 4. Implement

The minimum that satisfies the step. Reusable code in `itg_nn/xai/`, a thin CLI in
`scripts/xai_*.py` supporting `--config`, `--members`, `--rows`, `--device`,
`--seed`, `--resume`, and an output directory where applicable; small committed
config in `configs/xai/`. Keep member-level signed results as the primary artifact
and derive any aggregate from them. Tag every perturbation exact-symmetry,
observed-comparison, plausibly-local, or off-manifold, in the code and in the
artifact, not only in prose.

## 5. Pilot, then produce

Pilot first — top member, 64–128 rows — and inspect the numeric and visual
artifacts yourself before scaling. Do not launch the registered production
configuration until the pilot's acceptance tests pass.

Write everything large to `output/xai/<step>/<run_id>/` with a complete
`manifest.json`: git commit and dirty status, exact command, full config, seeds,
Python and package versions, device, dataset path and fingerprint, checkpoint
hash, member IDs, row IDs, gradient set, wall time, and output hashes. Commit only
the report and the small figures and JSON summaries the conclusions rest on.

Monitor a launched run to completion. Do not assume a detached command succeeded.

## 6. Verify

```bash
make check
```

Then verify the tests can fail. Pick the two or three mutations that matter for
this step, apply each, confirm the suite goes red, and revert. Candidates that
have historically mattered here:

- bootstrap or split by flux tube instead of `equilibrium_files`;
- drop the robust per-channel scale and compare raw gradients across the seven
  geometry channels;
- explain `exp(prediction)` instead of the native clipped log;
- average over members before the signed member-level statistic;
- pool near-floor rows in with unstable rows.

Record which mutations you checked; this goes in the PR body. A mutation that does
*not* turn the suite red is itself a finding — say so rather than quietly picking
a different one.

Also confirm, every time:

```bash
git status --short
```

`models/cyclic_ensemble_pre2.pt` unchanged, the external HDF5 dataset unchanged,
`paper/` untouched, no unrelated untracked user files absorbed, and no large
generated artifact staged.

## 7. Write the report

`reports/xai/SNN_<name>.md`, with methods, cohort, estimand, uncertainty, failed
checks, negative results, interpretation limits, and the exact commands to
reproduce every number. Two sections are mandatory:

- `## Acceptance criteria` — the `PLAN.md` criteria one by one, each with a
  verdict and the number or artifact that answers it;
- `## Deferred` — what you dropped and why, or "nothing".

Add `reports/xai/SNN_executive_summary.md` in plain language for the researcher,
following the tone of the existing S01–S03 summaries. Keep contradictory and
negative results in both.

## 8. Push and get CI green

Push the branch and let GitHub Actions run. CI installs the inference
dependencies and runs `pytest` with no external dataset. If it fails, fix it,
push, and iterate until it is green — do not open the PR on a red branch unless
you are stopping at a decision gate.

## 9. Open the PR

```bash
gh pr create --fill
```

The PR body must contain, in this order:

- step number and a one-line summary;
- the estimand and cohort, in one sentence each;
- the acceptance criteria, one per line, each with its verdict and measured
  number;
- `run_id`, the manifest path, and the checkpoint and dataset fingerprints the run
  recorded;
- the mutations you verified turn the suite red, and any that did not;
- tests added, and what each one would catch;
- what is in `## Deferred`, and why;
- any decision-gate question you are raising, or "none";
- confirmation that the model file, the dataset, and `paper/` are untouched; and
- what you were least sure about and want the reviewer to attack hardest.

## 10. Answer the automated review

The `Claude Code Review` workflow reviews the PR on open and on every push. Watch
for its run and read the findings table it posts.

Fix everything flagged **blocking** or **should-fix**, push, and iterate until the
review posts neither. **note** items are your judgement. If you disagree with a
finding, say so in a PR comment with concrete evidence — the artifact, the line,
or the test that shows it is wrong — rather than silently ignoring it.

When the review comes back with no blocking and no should-fix findings, stop. Do
not merge. Do not start the next step.

## If you hit a stop condition

`AGENTS.md` and `PLAN.md` list the decision gates. When you reach one, or when a
step would materially change the registered estimand, cohort, or baseline family:
write a short decision memo at `reports/xai/SNN_<topic>_decision.md` containing the
evidence, the options, the estimated cost of each, and your recommendation; commit
it; push; open the PR as a draft naming the memo; and end your turn. Do not choose
and proceed.

If the step runs past its `PLAN.md` budget, deliver the MVD completely, record
what you dropped under `## Deferred`, and stop. Relaxing an acceptance criterion,
widening a tolerance, or dropping a control to reach green is never the answer.
