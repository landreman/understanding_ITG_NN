# S00 — Reproducible XAI scaffold

## Status

Completed. This step establishes infrastructure only; it makes no claim about a
learned geometry feature or a plasma mechanism.

### Post-step corrections

Applied after the step was written, outside its numbered scope:

- The Captum pin was raised from `0.7.0` to `0.9.0` after verifying imports and a
  live `IntegratedGradients` / `LayerIntegratedGradients` call against
  torch 2.4.1, numpy 1.26.4, Python 3.12.4. The pilot below used `0.7.0`; nothing
  in it depends on Captum, and the 12-test suite passes on `0.9.0`.
- `scripts/setup_xai_env.sh` replaces the multi-command venv recipe, so a fresh
  worktree is one command away from a working `.venv-xai`.

## Delivered contract

- `itg_nn/xai/` provides validated JSON configuration, deterministic seed setup,
  source-addressable batching, validation-score-only member selection,
  individual-member native-target predictions, activation capture, atomic HDF5/
  JSON artifacts, and full manifests.
- `ModuleCyclicInvariantNet` replaces only functional ReLUs with named `nn.ReLU`
  modules for attribution tooling. Its convolution, pooling, dense, and output
  arithmetic is otherwise the original inference architecture.
- `PeriodicWindowToy`, `ColocationToy`, and `FourierBandToy` are analytic cyclic
  controls with known relevant channel/window/co-location/Fourier features.
- Captum 0.7.0 is the sole optional XAI dependency, locked in
  `requirements/xai.lock` and installed only in the project-local `.venv-xai`.

## Registered CPU pilot

Command:

```bash
.venv-xai/bin/python scripts/xai_smoke.py --config configs/xai/S00_smoke.json
```

The committed configuration selects the one member with the highest stored
validation $R^2$ (`2864601_0.437`; no test metric is consulted), uses CPU and
seed `20260816`, and evaluates 64 explicitly listed positive-target
varied-gradient row IDs. The small, intentionally non-stratified source fixture
is a smoke control, not an interpretation panel.

The resulting ignored artifact directory is
`output/xai/S00/cpu-top-member-64positive-rows/`. Its `predictions.h5` retains
native clipped-log predictions with axes `(member, sample)`, stable source row
IDs, target values, and member IDs. `module_equivalence.json` reports that
`torch.equal` passed for all 100 original/module-form members on the 64-row
fixture. The manifest has hashes for both output files plus the full config,
command, seed, device, data/checkpoint fingerprints, package versions, member
IDs, row IDs, elapsed time, and git state.

The top member's 64 native predictions ranged from -2.033 to 5.935 (mean 0.489);
the fixture targets ranged from -2.000 to 5.922 (mean 0.491). These values are a
pipeline sanity check only and are not a performance estimate.

## Verification

Both the baseline and XAI-environment suites passed:

```text
conda run -n 20240629-01-ML python -m pytest  # 12 passed
.venv-xai/bin/python -m pytest                # 12 passed
```

The test suite includes the all-100-member bit-for-bit conversion check, member
selection tie/control behavior, individual-member prediction ordering, named
activation capture, toy calculations, and manifest completeness. The XAI virtual
environment compatibility smoke imported Captum 0.7.0 with PyTorch 2.4.1.

## Checks, uncertainty, and limits

An initial eight-row draft was rejected because raw row 3 has non-positive heat
flux, and the established clipped-log target correctly raises on it. The
registered config instead lists 64 positive rows; no target transform or reader
was weakened. No other checks failed.

There is no sampling uncertainty estimate in S00 because it reports no
scientific quantity. The 64 rows do not represent stability regimes,
equilibria, or the downstream interpretation cohort. The module conversion is
validated for forward values only; LRP conservation/rule coverage and the
faithfulness, symmetry, baseline, perturbation, and physical-plausibility tests
remain prerequisites for later claims. Outputs are native
`max(log(Q), -2)` values; neither `exp(prediction)` nor a causal intervention is
explained here.

## Acceptance criteria

| Criterion | Evidence | Status |
| --- | --- | --- |
| Existing tests pass | Both test environments: 12 passed | Pass |
| All 100 original/wrapped outputs match | `torch.equal` pilot check and regression test | Pass |
| Smoke artifact reproducible from manifest | Versioned config/command and complete ignored run manifest | Pass |
| No scientific result claimed | Status and limits above | Pass |
