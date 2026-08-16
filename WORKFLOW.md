# Workflow for carrying out the XAI research plan with Codex or Claude

## The simple default

Use one fresh AI task per numbered step in `PLAN.md`. Complete S00 and S01 first.
After that, use the concurrency map in the plan, with one git worktree and branch
per concurrent step. Have the other AI system review important branches before
merging: for example, Codex implements and Claude reviews, then reverse the roles
on the next step.

The only prompt you normally need is:

```text
Execute PLAN.md step SNN end-to-end. Follow WORKFLOW.md. Implement, test, run the
registered pilot, write the step report, and commit the finished work on the
current branch.
```

Replace `SNN` with the step number. The plan contains the scientific question,
tasks, deliverables, and acceptance tests, so there is no need to restate them in
the prompt.

## One-time setup

No dataset environment variable is needed. `PLAN.md` registers the canonical
dataset path as
`/Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/20250102-01_GX_stellarator_dataset.h5`.
Agents and XAI scripts should use it by default, with an optional `--dataset`
override for portability. Keep the large HDF5 file outside git and do not copy
the legacy 518 MB serialized dataset into this repository.

The existing conda environment `20240629-01-ML` has the package and its inference
dependencies. S00 should create a project-local XAI virtual environment for any
additional packages, because `AGENTS.md` says not to modify the conda environment.
A suitable starting pattern is:

```bash
conda run -n 20240629-01-ML python -m venv --system-site-packages .venv-xai
.venv-xai/bin/python -m pip install -e .
```

The S00 agent should then add and lock only the extra dependencies that pass a
compatibility smoke test. If `--system-site-packages` does not expose the expected
PyTorch build, the agent should create a standalone venv and document the exact
install instead of changing conda.

Create `.venv-xai` separately inside every active worktree. Do not share an
editable-install venv across worktrees: its import path can silently point at the
wrong branch. S00 should make venv creation reproducible and add `.venv-xai/` to
gitignore; each later agent can then run the same setup command in its checkout.

Before starting, preserve the user's existing changes:

```bash
git status --short
git branch --show-current
```

Do not delete, reset, or absorb unrelated untracked files such as the existing
`output/` directory.

## Branches and worktrees

### Sequential steps

For S00 and S01, a normal feature branch is simplest:

```bash
git switch -c codex/xai-s00-scaffold
```

After the AI completes and commits S00, review and merge it. Then create and merge
S01 from the updated base. Keeping these foundation steps sequential avoids every
later branch inventing its own cohort or file format.

### Concurrent steps

Use git worktrees whenever two AIs may edit or run code at the same time. A
worktree is an isolated checkout on its own branch, so one agent cannot overwrite
another agent's files or switch its branch.

In Codex Desktop, the easiest route is to start a new task, select **Worktree** as
the environment, and choose the fully merged prerequisite branch. Codex creates
the isolated checkout automatically. Its worktrees begin on a detached `HEAD`, so
have the task create a named `codex/xai-sNN-short-name` branch and commit before
integration (or use **Handoff to Local**). See the official
[Codex worktree documentation](https://learn.chatgpt.com/docs/environments/git-worktrees).

Manual git worktrees are the universal alternative:

Create worktrees from the same fully merged base. For example, after S01:

```bash
git worktree add ../understanding_ITG_NN-xai-s02 -b codex/xai-s02-baselines main
git worktree add ../understanding_ITG_NN-xai-s03 -b claude/xai-s03-symmetry main
```

Open each directory as a separate Codex task or start a separate Claude Code
session there. Claude Code also officially supports isolated sessions directly:

```bash
claude --worktree xai-s03
```

Its [parallel-agent documentation](https://code.claude.com/docs/en/agents) and
[common-workflows guide](https://code.claude.com/docs/en/common-workflows) explain
the current worktree behavior.

Use these branch prefixes so ownership is obvious:

- `codex/xai-sNN-short-name`
- `claude/xai-sNN-short-name`

Do not have concurrent steps edit `PLAN.md`, `WORKFLOW.md`, the same report, or a
shared progress file. Shared infrastructure belongs in S00; if a later step truly
needs a shared change, make that change a small separate commit and tell the
integrating agent.

### Cleanup

Only after a branch has been merged and its useful output has been copied or is
reproducible from a manifest:

```bash
git worktree remove ../understanding_ITG_NN-xai-s02
git branch -d codex/xai-s02-baselines
git worktree prune
```

These operations remove the checkout/merged branch, not the external dataset.
Always run `git worktree list` first. Never use forced branch deletion for an
unmerged research result.

## What can run concurrently

Use the dependency map in `PLAN.md` as authoritative. The useful waves are:

| Wave | Steps | Notes |
|---|---|---|
| 0 | S00 | Alone; creates shared infrastructure. |
| 1 | S01 | Alone; freezes the cohort and audit. |
| 2 | S02 and S03 | Safe in parallel after S01. |
| 3 | S04 | Alone; chooses trustworthy explainers. |
| 4 | S05, S06, and S08 | Best main parallel wave; input attribution, counterfactuals, and activation catalog write separate artifacts. |
| 5 | S07 | Integrates the input-level results from S05 and S06. |
| 6 | S09 and S10 | Hidden interventions and concept tests can proceed in parallel after S07/S08. |
| 7 | S11 | Requires the intervention and concept results. |
| 8 | S12 | Compares members after S08--S11. |
| 9 | S13 | Analyzes disagreement after S05, S06, and S12. |
| 10 | S14 | Distills after S07, S11, and S12. |
| 11 | S15 | Optional/new-simulation decision after S07, S13, and S14. |
| 12 | S16 | Final synthesis after all required predecessor reports. |

Concurrency saves agent time, but simultaneous numerical jobs can make all of
them slower. On one laptop, default to:

- up to three concurrent code-development/pilot tasks;
- only one full-HDF5, all-shifts, or all-100-member production run at a time; and
- top-member/64-row pilots before top-10 or full-panel jobs.

If GPUs or a batch scheduler are available later, an AI should add explicit job
scripts and resource estimates rather than assuming local concurrency scales.

## The task lifecycle

### 1. Start from a clean, current base

The step branch must contain every merged prerequisite. Ask the agent to begin by
reading `AGENTS.md`, `PLAN.md`, `WORKFLOW.md`, the prerequisite reports, and the
relevant code. The standard prompt already tells it to follow these documents.

### 2. Let the agent implement and pilot

The agent should do the whole bounded step:

- implement reusable code and a thin CLI;
- add unit tests and analytic/control tests;
- run a small pilot;
- inspect numeric and visual artifacts;
- fix failures rather than merely describing them;
- write the registered step report; and
- commit code, configs, tests, and small report artifacts.

Large arrays and routine figures belong under ignored `output/xai/`, not in git.
Every conclusion in a committed report must be reproducible from a manifest.

### 3. Review with the other AI

Use a fresh read-only review task when a branch claims scientific results:

```text
Review branch <branch> against PLAN.md step SNN and its acceptance criteria.
Check scientific estimands, cyclic symmetry, leakage, controls, statistics,
reproducibility, and code correctness. Do not edit. Return only actionable issues,
ordered by severity, with file and line references.
```

Then send the implementation agent the review:

```text
Address every valid review issue, rerun affected tests and pilot calculations,
update the SNN report, and commit the fixes. Explain any issue you reject with
concrete evidence.
```

Alternating Codex and Claude between implementation and review is a cheap way to
reduce tool-specific blind spots. There is no need to decide that one system owns
all numerical work.

### 4. Verify before merging

The integrating agent, or the researcher with the agent watching, should run:

```bash
git status --short
git diff main...HEAD --check
conda run -n 20240629-01-ML python -m pytest
```

It should also run the step's documented smoke/pilot command using `.venv-xai`
when the step uses extra dependencies. Confirm that:

- the report names the exact estimand and cohort;
- acceptance criteria are answered one by one;
- controls and negative results are present;
- generated large files are not staged;
- existing inference results remain unchanged; and
- no unrelated user files were modified.

### 5. Merge one branch at a time

Even if several steps finish together, merge them serially. After each merge, run
tests. If two branches require the same shared helper, merge the cleaner helper
change first, rebase or merge the new base into the other worktree, and ask that
agent to resolve/test. Do not manually paste files between worktrees.

### 6. Record the state

The committed report and run manifest are the progress record. Avoid a single
checkbox file edited by every parallel agent because it becomes a merge-conflict
hotspot. When starting a new task, the agent can infer readiness from merged
reports and the dependency map.

## Prompts you can copy

### Implement one step

```text
Execute PLAN.md step SNN end-to-end. Follow WORKFLOW.md. Implement, test, run the
registered pilot, write the step report, and commit the finished work on the
current branch.
```

### Continue a step that stopped

```text
Continue PLAN.md step SNN from the current branch. Inspect existing changes and
the latest manifest/report, finish unmet acceptance criteria, rerun affected
checks, and commit the completed step. Preserve unrelated work.
```

### Run the production calculation after a pilot passes

```text
The SNN pilot is accepted. Run its registered production configuration, monitor
it to completion, validate the artifacts against the report's controls, and
update and commit only the small report/index files. Keep large outputs ignored.
```

### Review a step

```text
Review branch <branch> against PLAN.md step SNN and its acceptance criteria.
Check scientific estimands, cyclic symmetry, leakage, controls, statistics,
reproducibility, and code correctness. Do not edit. Return actionable issues with
file and line references, ordered by severity.
```

### Integrate a completed wave

```text
Integrate the completed Wave N branches one at a time. Inspect each diff and
report, resolve conflicts without dropping scientific controls, run tests after
each merge, and summarize which PLAN.md acceptance criteria are satisfied. Do not
merge a branch that lacks a reproducible manifest or has unresolved high-severity
review findings.
```

### Produce the final synthesis

```text
Execute PLAN.md step S16. Treat prior reports and machine-readable artifacts as
evidence, audit their manifests, preserve negative and contradictory results, and
separate model-mechanistic from physical-causal claims.
```

## How much context to give the AI

Do not paste the paper, plan, or old reports into the prompt. Point the agent to
the files and let it read what is relevant. A fresh task per step prevents a long
conversation from accumulating stale assumptions. Resume the same task only for
fixes or a production run of that same step.

If an agent proposes a material change to the scientific estimand, cohort,
baseline family, or intervention validity, it should stop before the expensive
run and put a short decision memo in its response. Routine implementation choices
do not require researcher input.

## Handling long calculations

- Require the CLI to support `--config`, `--members`, `--rows`, `--device`,
  `--batch-size`, `--seed`, `--resume`, and an output directory where applicable.
- Make writes atomic and checkpoint by member/sample block so an interrupted run
  can resume without corrupting completed output.
- Print progress, estimated remaining work, and the final manifest path.
- Keep raw member-level data. Never retain only an ensemble average.
- On failure, save the exception and completed-block index in the run directory.
- Ask the agent to monitor a launched job to completion; do not assume a detached
  command succeeded.

## Decisions that should come back to you

Agents should proceed autonomously through ordinary numerical and coding choices.
They should ask you only at the decision gates in `PLAN.md`, chiefly:

- whether to retrain an exactly invariant model;
- whether to spend substantially more compute on all 100 members;
- whether to restore training code for remove-and-retrain experiments;
- whether to generate new equilibria or launch GX simulations; and
- whether publication should wait for those simulations.

The agent's question should be short and include evidence, estimated cost, and a
recommended option. You should not have to reconstruct the issue from logs.

## Safety and scientific hygiene

- Never modify or overwrite `models/cyclic_ensemble_pre2.pt` or the external HDF5
  dataset. Hash them in manifests.
- Never use `git reset --hard`, forced branch deletion, or recursive deletion to
  clean research worktrees.
- Do not install XAI libraries into `20240629-01-ML`; use `.venv-xai`.
- Do not commit large generated artifacts, environment directories, caches, or
  copied data.
- Do not tune explanation hyperparameters on the same cases used for the final
  scientific claim. Use pilot/development and registered evaluation panels.
- Do not call feature replacement a physical counterfactual unless it preserves a
  realizable equilibrium and is propagated through GX.
- Preserve member-to-member variation and failures; an ensemble mean can hide
  opposing mechanisms.

## Tool documentation

- The official [Codex worktree documentation](https://learn.chatgpt.com/docs/environments/git-worktrees)
  describes creating parallel Desktop tasks, detached `HEAD` behavior, and
  handing a worktree back to the local checkout.
- Claude Code's [parallel-agent guide](https://code.claude.com/docs/en/agents),
  [worktree workflow](https://code.claude.com/docs/en/common-workflows), and
  [session documentation](https://code.claude.com/docs/en/sessions) describe its
  current parallel-session behavior. Worktrees isolate files, but parallel agents
  consume quota and compute independently.
