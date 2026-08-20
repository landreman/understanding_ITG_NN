# Workflow for carrying out the XAI research plan

Codex implements a step and opens a PR. GitHub Actions runs the tests and then
runs Claude as an adversarial reviewer, which posts a findings table on the PR.
Codex reads that table, fixes what it flags, and pushes again. You type one line
at the start, answer decision-gate questions when they come up, and click merge.

Setup is once, and takes under an hour. `PLAN.md` holds the science, `AGENTS.md`
holds the contract both agents obey, and this file is for you.

## The loop

**You**, in Codex, in this repository:

```text
/step next
```

or `/step S04` to pick one. That is the whole prompt. `PLAN.md` already contains
the question, the tasks, the budget, the deliverables, and the acceptance tests,
so there is nothing to restate and nothing to paste.

**Codex** branches, writes the tests first, implements, pilots, runs the
registered configuration, writes `reports/xai/SNN_<name>.md` and its executive
summary, verifies that the tests can fail, pushes, and opens a PR. Or it stops at
a decision gate, commits a decision memo, and opens the PR as a draft.

**CI** installs the inference dependencies and runs `pytest` on Linux and macOS.
It never sees the external HDF5 dataset, so the suite must stand on synthetic
fixtures and the committed checkpoint.

**Claude** reviews the PR on open and on every push, against
`.claude/commands/review-step.md`, and posts a findings table with severities and
a verdict line.

**Codex** fixes every **blocking** and **should-fix** finding and pushes, until
the review returns neither. It does not merge.

**You** read the final review and the executive summary, then merge. If a decision
memo is waiting, that one is yours: write the answer into the memo, commit it, and
tell Codex to continue.

**You never** write a bespoke prompt for a sub-task, approve individual file
writes, copy a review between two chat windows, or explain `PLAN.md` to an agent
again. When you find yourself giving the same review note twice, put it in
`AGENTS.md` and it stops recurring.

## One-time setup

### 1. Files in this repository

| Path | Whose | What it does |
|---|---|---|
| `AGENTS.md` (`CLAUDE.md` → symlink) | both | The contract. One copy, one place to correct it. |
| `.agents/skills/step/SKILL.md` | Codex | The `/step` skill: the whole implementation sequence. |
| `.claude/commands/review-step.md` | Claude | The adversarial review, ten questions with verdicts. |
| `.github/workflows/ci.yml` | CI | pytest on Linux and macOS, no dataset. |
| `.github/workflows/claude-code-review.yml` | CI | Runs the review on every PR push. |
| `.codex/config.toml` | Codex | Sandboxed, no approvals, no web search. |
| `Makefile` | both | `make test`, `make check`, `make smoke`, `make env`. |

### 2. Codex

Sandboxed full-auto inside the workspace is what makes this safe to leave
unattended. `.codex/config.toml` sets `approval_policy = "never"` and
`sandbox_mode = "workspace-write"` with network access, which the `.venv-xai`
install and `gh` need. Do not use `--yolo`; it removes the sandbox that makes the
rest of this acceptable.

Web search is disabled on purpose. An agent that can search will find a blog post
about saliency maps and quietly follow it instead of `PLAN.md`'s interpretation
contract.

Take whatever model sits at the top of your `/model` picker. Effort matters more
than the name: use high for anything numerical, and xhigh for the steps where a
wrong answer looks right — S02 (the canonical function everything downstream
inherits), S03 (the ladder that decides where the rest of the effort goes), S04
(exact Shapley at the bottleneck), S09 (completeness), and S12 (distillation).

### 3. Claude Code PR review

You need the `CLAUDE_CODE_OAUTH_TOKEN` secret on the GitHub repository. From this
repository:

```bash
claude
```

then, in that session, `/install-github-app` and pick this repository. It installs
the GitHub app and stores the secret. It will also offer to write its own
`claude-code-review.yml`; the one in this repository is already adapted to point
at `.claude/commands/review-step.md`, so keep this one if asked to overwrite.

Check the secret afterwards:

```bash
gh secret list
```

The review is the one place to spend on the strongest model — it is what stands
between a subtly wrong estimand and a merged subtly wrong estimand — so
`claude_args: --model opus` stays. If you would rather not pay on every push,
change the trigger in `claude-code-review.yml` to `types: [opened]`, or drop the
workflow and run `claude` with `/review-step` locally on the branch instead.

### 4. Dataset

No environment variable is needed. `PLAN.md` registers the canonical dataset path
as
`/Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/20250102-01_GX_stellarator_dataset.h5`.
Scripts use it by default with a `--dataset` override. It stays outside git; the
518 MB legacy serialized dataset is never copied in.

### 5. Python environment

The conda environment `20240629-01-ML` has the package and its inference
dependencies. `AGENTS.md` forbids installing into it, so XAI extras live in a
project-local `.venv-xai`:

```bash
bash scripts/setup_xai_env.sh
```

Create `.venv-xai` separately inside **every** active worktree. Do not share an
editable-install venv across worktrees: its import path can silently point at the
wrong branch, and you will lose an afternoon watching an agent fix code it is not
running. The venv inherits torch/numpy/h5py from conda via
`--system-site-packages`, so it costs megabytes, not gigabytes.

A step that needs a new package adds it to `pyproject.toml`'s `xai` extra and
`requirements/xai.lock`, runs a real call against this repository's model with it,
and records the tested version in its report.

## Branches and worktrees

Branch per step, always — `PLAN.md` sizes each step for one reviewable PR, and
that maps onto one PR. Prefixes so ownership is obvious:

- `codex/xai-sNN-short-name`
- `claude/xai-sNN-short-name`

Worktrees only when two agents run at once. In Codex Desktop, start a task with
**Worktree** as the environment and choose the fully merged prerequisite branch.
Its worktrees begin on a detached `HEAD`, so have the task create the named branch
and commit before integration. Manual worktrees are the universal alternative:

```bash
git worktree add ../understanding_ITG_NN-xai-s04 -b codex/xai-s04-bottleneck main
cd ../understanding_ITG_NN-xai-s04 && bash scripts/setup_xai_env.sh
```

Do not have concurrent steps edit `PLAN.md`, `AGENTS.md`, `WORKFLOW.md`, the same
report, or a shared progress file. Shared infrastructure belongs in the earliest
step that needs it.

Cleanup, only after a branch is merged and its output is reproducible from a
manifest:

```bash
git worktree list
git worktree remove ../understanding_ITG_NN-xai-s04
git branch -d codex/xai-s04-bottleneck
git worktree prune
```

Never force-delete an unmerged research branch, and never `git reset --hard` a
research worktree.

## What can run concurrently

`PLAN.md`'s dependency map is authoritative. The useful waves:

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

Concurrency saves agent time; simultaneous numerical jobs can make all of them
slower. On this laptop, default to up to three concurrent development/pilot tasks,
only one full-HDF5 or all-100-member production run at a time, and pilots before
full-panel jobs. More than two or three PRs in flight and you become the
bottleneck at merge, which defeats the purpose.

**Perlmutter.** A step that wants NERSC must first run at pilot scale locally and
produce a measured extrapolation — wall time, memory, node-hours — and then ask.
An agent should not port a calculation it has never run.

## Merging

Merge one branch at a time even when several finish together, and run the tests
after each merge. If two branches need the same shared helper, merge the cleaner
helper change first, merge the new base into the other worktree, and ask that
agent to resolve and test. Do not paste files between worktrees by hand.

Before merging, confirm from the PR and the review that:

- the report names the exact estimand and cohort;
- the acceptance criteria are answered one by one with numbers;
- controls and negative results are present, including the ones that came out
  awkwardly;
- the run manifest is complete and the conclusions trace to it;
- no large generated file is staged, and the model, dataset, and `paper/` are
  untouched.

The committed report and its manifest are the progress record. There is no
checkbox file, deliberately: it would be a merge-conflict hotspot for parallel
agents. `/step next` infers readiness from the dependency map and the merged
reports.

## Other prompts worth keeping

Continue a step that stopped:

```text
Continue PLAN.md step SNN from the current branch. Inspect existing changes and
the latest manifest/report, finish unmet acceptance criteria, rerun affected
checks, and commit the completed step. Preserve unrelated work.
```

Run production after a pilot passes, if the step stopped there:

```text
The SNN pilot is accepted. Run its registered production configuration, monitor
it to completion, validate the artifacts against the report's controls, and
update and commit only the small report/index files. Keep large outputs ignored.
```

Answer a review by hand, if you are not using the Action:

```text
Address every blocking and should-fix finding in the review comment, rerun
affected tests and pilot calculations, update the SNN report, and push. Explain
any finding you reject with concrete evidence.
```

## Handling long calculations

- The CLI supports `--config`, `--members`, `--rows`, `--device`, `--batch-size`,
  `--seed`, `--resume`, and an output directory where applicable.
- Writes are atomic and checkpointed by member/sample block, so an interrupted run
  resumes without corrupting completed output.
- Progress, remaining work, and the final manifest path are printed.
- Raw member-level data is kept. Never retain only an ensemble average.
- On failure, the exception and completed-block index are saved in the run
  directory.
- The agent monitors a launched job to completion; a detached command is not
  assumed to have succeeded.

## Decisions that should come back to you

Agents proceed autonomously through ordinary numerical and coding choices, and
stop only at the `PLAN.md` decision gates:

1. which function is canonical after S02 (original, shift-averaged, or the exactly
   invariant bottleneck model);
2. whether to move a calculation to Perlmutter;
3. whether to restore training code for remove-and-retrain experiments — you have
   said you would rather not, so this needs a genuine scientific reason;
4. whether to generate new equilibria or launch GX simulations; and
5. whether publication should wait for those simulations.

A gate arrives as a committed memo at `reports/xai/SNN_<topic>_decision.md` on a
draft PR: evidence, options, estimated cost, recommendation. Those memos are your
queue — if none is open, nothing wants you. You should not have to reconstruct the
issue from logs.

### Where it is worth actually reading the output

You do not need to read every report closely. The three that change the shape of
the program are **S02** (which function is canonical, and how badly arbitrary
shifts break invariance), **S03** (how much of the network can possibly be spatial
or cross-channel), and **S09** (what the network knows beyond
$\{a/L_T, a/L_n, f_Q\}$). Skim the rest; read those, slowly, alongside Claude's
review.

## Safety and scientific hygiene

- Never modify or overwrite `models/cyclic_ensemble_pre2.pt` or the external HDF5
  dataset. Hash them in manifests.
- Never use `git reset --hard`, forced branch deletion, or recursive deletion to
  clean research worktrees.
- Do not install XAI libraries into `20240629-01-ML`; use `.venv-xai`.
- Do not commit large generated artifacts, environment directories, caches, or
  copied data.
- Do not tune explanation hyperparameters on the same cases used for the final
  scientific claim.
- Do not call feature replacement a physical counterfactual unless it preserves a
  realizable equilibrium and is propagated through GX.
- Preserve member-to-member variation and failures; an ensemble mean can hide
  opposing mechanisms.

## Honest limits

Agents on interpretability code fail in a specific way: they produce something
that runs, is well documented, and passes tests that do not constrain the
science. Green CI is weak evidence here — the suite cannot see the dataset, and
the expensive claims live in artifacts no test asserts on. What makes this
tractable is the analytic toy functions with known relevant features, the controls
that must come out null, and the mutation checks. If you drop one thing from this
workflow, do not let it be the mutation checks.

Expect the review to be most valuable on estimand and grouping questions, and
least valuable on anything that would require rerunning a dataset-backed
calculation — it says so when it cannot check something, and those lines are the
ones worth your own attention.

## Tool documentation

- [Codex worktree documentation](https://learn.chatgpt.com/docs/environments/git-worktrees):
  parallel Desktop tasks, detached `HEAD` behavior, and handing a worktree back to
  the local checkout.
- Claude Code's [parallel-agent guide](https://code.claude.com/docs/en/agents),
  [common-workflows guide](https://code.claude.com/docs/en/common-workflows), and
  [GitHub Actions documentation](https://code.claude.com/docs/en/github-actions).
