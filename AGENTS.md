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
  complete `manifest.json`; commit only small reports and figures.
- Negative and contradictory results are kept, not dropped.

## When to stop and ask

Proceed autonomously through ordinary numerical and coding choices. Stop and ask
only at the decision gates listed in `PLAN.md` and `WORKFLOW.md`, or if a step
would materially change the registered estimand, cohort, or baseline family. Keep
the question short and include evidence, estimated cost, and a recommendation.

If a step exceeds its `PLAN.md` effort budget, deliver the step's minimum viable
deliverable completely, record what was dropped under a `## Deferred` heading in
the step report, and stop. Do not deliver a shallow version of everything.
