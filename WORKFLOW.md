# Workflow for carrying out the XAI research plan with Codex or Claude

## What you actually do

For each step in `PLAN.md`, in dependency order:

1. Make a branch (or worktree — see below) from up-to-date `main`.
2. Paste **the one prompt** below, with `SNN` replaced by the step number.
3. When the agent says it is done, run the review prompt in a *fresh* task of the
   *other* AI system.
4. Paste the review back to the implementing agent with the fix prompt.
5. Merge, then delete the branch.

That is the whole loop. Everything else in this document is reference material
for when something goes sideways.

### The one prompt

```text
Execute PLAN.md step SNN end-to-end. Follow WORKFLOW.md and AGENTS.md.
Implement, test, run the registered pilot, write the step report, and commit the
finished work on the current branch.
```

`PLAN.md` contains the scientific question, tasks, effort budget, deliverables,
and acceptance tests for every step, so there is nothing to restate in the
prompt. Do not paste the paper, the plan, or old reports into the prompt — point
the agent at the files and let it read what it needs.

### The review prompt

```text
Review branch <branch> against PLAN.md step SNN and its acceptance criteria.
Check scientific estimands, cyclic symmetry, leakage, controls, statistics,
reproducibility, and code correctness. Do not edit. Return actionable issues with
file and line references, ordered by severity.
```

### The fix prompt

```text
Address every valid review issue, rerun affected tests and pilot calculations,
update the SNN report, and commit the fixes. Explain any issue you reject with
concrete evidence.
```

Alternating Codex and Claude between implementation and review is a cheap way to
reduce tool-specific blind spots. There is no need to decide that one system owns
all numerical work.

## Setup

### Dataset

No environment variable is needed. `PLAN.md` registers the canonical dataset path
as
`/Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/20250102-01_GX_stellarator_dataset.h5`.
Agents and XAI scripts use it by default, with an optional `--dataset` override
for portability. Keep the large HDF5 file outside git and do not copy the legacy
518 MB serialized dataset into this repository.

### Python environment

The conda environment `20240629-01-ML` has the package and its inference
dependencies. `AGENTS.md` forbids installing into it, so XAI extras live in a
project-local `.venv-xai`. One command creates it in any fresh checkout or
worktree:

```bash
bash scripts/setup_xai_env.sh
```

Create `.venv-xai` separately inside **every** active worktree. Do not share an
editable-install venv across worktrees: its import path can silently point at the
wrong branch. The venv inherits torch/numpy/h5py from conda via
`--system-site-packages`, so it costs a few megabytes, not gigabytes.

If a step needs a new package, the agent adds it to `pyproject.toml`'s `xai`
extra and `requirements/xai.lock`, runs a compatibility smoke test, and records
the tested version in its report.

### Before starting anything

```bash
git status --short
git branch --show-current
```

Do not delete, reset, or absorb unrelated untracked files such as the existing
`output/` directory.

## Branches and worktrees

### Sequential steps

For S01, S02 and S03 — which are deliberately sequential because each one changes
what the next should measure — a normal feature branch is simplest:

```bash
git switch -c codex/xai-s01-audit
```

Review and merge before starting the next. Keeping these foundation steps
sequential avoids every later branch inventing its own cohort or file format.

### Concurrent steps

Use git worktrees whenever two AIs may edit or run code at the same time. A
worktree is an isolated checkout on its own branch, so one agent cannot overwrite
another agent's files or switch its branch.

In Codex Desktop, start a new task, select **Worktree** as the environment, and
choose the fully merged prerequisite branch; Codex creates the isolated checkout
automatically. Its worktrees begin on a detached `HEAD`, so have the task create a
named `codex/xai-sNN-short-name` branch and commit before integration (or use
**Handoff to Local**). See the official
[Codex worktree documentation](https://learn.chatgpt.com/docs/environments/git-worktrees).

Manual git worktrees are the universal alternative. Create them from the same
fully merged base:

```bash
git worktree add ../understanding_ITG_NN-xai-s04 -b codex/xai-s04-bottleneck main
git worktree add ../understanding_ITG_NN-xai-s06 -b claude/xai-s06-attribution main
```

Open each directory as a separate Codex task or start a separate Claude Code
session there. Claude Code also supports isolated sessions directly:

```bash
claude --worktree xai-s04
```

See its [parallel-agent documentation](https://code.claude.com/docs/en/agents) and
[common-workflows guide](https://code.claude.com/docs/en/common-workflows).

Use these branch prefixes so ownership is obvious:

- `codex/xai-sNN-short-name`
- `claude/xai-sNN-short-name`

Do not have concurrent steps edit `PLAN.md`, `WORKFLOW.md`, `AGENTS.md`, the same
report, or a shared progress file. Shared infrastructure belongs in the earliest
step that needs it; if a later step truly needs a shared change, make that change
a small separate commit and tell the integrating agent.

### Cleanup

Only after a branch has been merged and its useful output has been copied or is
reproducible from a manifest:

```bash
git worktree remove ../understanding_ITG_NN-xai-s04
git branch -d codex/xai-s04-bottleneck
git worktree prune
```

These operations remove the checkout and merged branch, not the external dataset.
Always run `git worktree list` first. Never use forced branch deletion for an
unmerged research result.

## What can run concurrently

`PLAN.md`'s dependency map is authoritative. The useful waves are:

| Wave | Steps | Notes |
|---|---|---|
| 0 | S01 | Alone; freezes cohorts, panel, and member re-ranking. |
| 1 | S02 | Alone; fixes the canonical explained function and the equivariant density. |
| 2 | S03 | Alone; the counterfactual ladder tells later steps where to spend effort. |
| 3 | S04 and S06a | Bottleneck anatomy and the attribution-method benchmark are independent. |
| 4 | S05 and S06b | Unit semantics and the scaled attribution run. |
| 5 | S07 and S08 | Physics alignment and concept probes. |
| 6 | S09 | Completeness; needs the concept results. |
| 7 | S10 | Cross-member comparison after S04, S05, S08. |
| 8 | S11 and S12 | Disagreement analysis and distillation. |
| 9 | S13 | Natural experiments and the GX proposal. |
| 10 | S14 | Final synthesis. |

Concurrency saves agent time, but simultaneous numerical jobs can make all of
them slower. On this laptop, default to:

- up to three concurrent code-development/pilot tasks;
- only one full-HDF5, all-shifts, or all-100-member production run at a time; and
- top-member/64-row pilots before top-10 or full-panel jobs.

### Perlmutter

The plan's default target is this laptop. If a step wants NERSC, it must first
run at pilot scale locally and produce a measured extrapolation (wall time,
memory, node-hours) — then ask. An agent should not port a calculation it has
never run. When a step is approved for Perlmutter, it should add explicit job
scripts and a resource estimate to the repository rather than assuming local
concurrency scales.

## The task lifecycle

### 1. Start from a clean, current base

The step branch must contain every merged prerequisite. The standard prompt
already tells the agent to read `AGENTS.md`, `PLAN.md`, `WORKFLOW.md`, the
prerequisite reports, and the relevant code.

### 2. Let the agent implement and pilot

The agent does the whole bounded step:

- implement reusable code and a thin CLI;
- add unit tests and analytic/control tests;
- run a small pilot;
- inspect numeric and visual artifacts;
- fix failures rather than merely describing them;
- write the registered step report; and
- commit code, configs, tests, and small report artifacts.

Large arrays and routine figures belong under ignored `output/xai/`, not in git.
Every conclusion in a committed report must be reproducible from a manifest.

If the step is running past its `PLAN.md` budget, the agent delivers the step's
**minimum viable deliverable** completely, records what it dropped in a
`## Deferred` section of the report, and stops — rather than delivering a shallow
version of everything.

### 3. Review with the other AI

Use the review prompt above in a fresh, read-only task, then send the
implementation agent the fix prompt.

### 4. Verify before merging

The integrating agent, or you with the agent watching, should run:

```bash
git status --short
```

```bash
conda run -n 20240629-01-ML python -m pytest
```

It should also run the step's documented smoke/pilot command using `.venv-xai`.
Confirm that:

- the report names the exact estimand and cohort;
- acceptance criteria are answered one by one;
- controls and negative results are present;
- generated large files are not staged;
- existing inference results remain unchanged; and
- no unrelated user files were modified.

### 5. Merge one branch at a time

Even if several steps finish together, merge them serially. After each merge, run
tests. If two branches require the same shared helper, merge the cleaner helper
change first, merge the new base into the other worktree, and ask that agent to
resolve and test. Do not manually paste files between worktrees.

### 6. Record the state

The committed report and run manifest are the progress record. Avoid a single
checkbox file edited by every parallel agent because it becomes a merge-conflict
hotspot. When starting a new task, the agent infers readiness from merged reports
and the dependency map.

## Other prompts worth keeping

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
Execute PLAN.md step S14. Treat prior reports and machine-readable artifacts as
evidence, audit their manifests, preserve negative and contradictory results, and
separate model-mechanistic from physical-causal claims.
```

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

Agents proceed autonomously through ordinary numerical and coding choices. They
should stop and ask only at the `PLAN.md` decision gates:

1. which function is canonical after S02 (original, shift-averaged, or the
   exactly invariant bottleneck model);
2. whether to move a calculation to Perlmutter;
3. whether to restore training code for remove-and-retrain experiments — you have
   said you would rather not, so this needs a genuine scientific reason;
4. whether to generate new equilibria or launch GX simulations; and
5. whether publication should wait for those simulations.

The agent's question should be short and include evidence, estimated cost, and a
recommended option. You should not have to reconstruct the issue from logs.

If an agent proposes a material change to the scientific estimand, cohort,
baseline family, or intervention validity, it should stop before the expensive
run and put a short decision memo in its response. Routine implementation choices
do not require your input.

### Where it is worth actually reading the output

You do not need to read every report closely. The three that change the shape of
the program are **S02** (which function is canonical, and how badly arbitrary
shifts break invariance), **S03** (how much of the network can possibly be
spatial or cross-channel), and **S09** (what the network knows beyond
$\{a/L_T, a/L_n, f_Q\}$). Skim the rest; read those.

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

- [Codex worktree documentation](https://learn.chatgpt.com/docs/environments/git-worktrees):
  parallel Desktop tasks, detached `HEAD` behavior, and handing a worktree back to
  the local checkout.
- Claude Code's [parallel-agent guide](https://code.claude.com/docs/en/agents),
  [worktree workflow](https://code.claude.com/docs/en/common-workflows), and
  [session documentation](https://code.claude.com/docs/en/sessions). Worktrees
  isolate files, but parallel agents consume quota and compute independently.
