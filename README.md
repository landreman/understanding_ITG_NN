# Inference for the ITG heat-flux neural-network ensemble

This repository contains the inference-only form of the cyclic convolutional
ensemble used to predict GX ion-temperature-gradient heat flux. Training,
hyperparameter search, pickle readers, and unused architecture branches have
been removed. The inference pipeline reads the metadata-rich HDF5 dataset
directly.

## Contents

- `itg_nn/`: the model, HDF5 reader, batched ensemble inference, and plotting
  code.
- `models/cyclic_ensemble_pre2.pt`: the 100 best legacy state dictionaries,
  consolidated with only the architecture metadata required for inference.
- `scripts/build_ensemble_checkpoint.py`: the reproducible one-time converter
  from the legacy DeepHyper results and state dictionaries.
- `tests/`: lightweight unit tests.

The HDF5 dataset is intentionally external because it is about 659 MB. Its
canonical location for this repository is
`/Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/20250102-01_GX_stellarator_dataset.h5`;
`*.h5` is ignored by git.

## Environment

The existing conda environment has all required runtime packages, so no
package changes are needed:

```bash
conda run -n 20240629-01-ML python -m pytest
```

## XAI environment and S00 smoke pilot

Interpretation code is optional so normal inference keeps its established
environment. One command creates a project-local virtual environment from the
existing conda runtime; this does not modify `20240629-01-ML`:

```bash
bash scripts/setup_xai_env.sh
```

`captum==0.9.0` is the intentionally small locked XAI addition; it is also
declared as the `xai` optional dependency in `pyproject.toml` and pinned in
`requirements/xai.lock`. Make a separate `.venv-xai` in each worktree because an
editable install is worktree-specific.

Core CI installs `.[dev]` only, so the suite keeps proving that inference and the
tests stand alone without the XAI extras. The automated PR review installs
`.[dev,xai]` plus `scipy` and `pandas`, so it exercises the same attribution code
path production did and can cross-check statistics with libraries this package
does not depend on.

The registered S00 CPU pilot selects the highest stored validation-$R^2$ member,
uses 64 positive-target varied-gradient rows, verifies the module-form conversion against all
100 original members, and writes labeled member-level native-target predictions:

```bash
.venv-xai/bin/python scripts/xai_smoke.py \
  --config configs/xai/S00_smoke.json
```

The default dataset path is the canonical HDF5 path registered in `PLAN.md`.
For a portable checkout, override it without changing the committed config:

```bash
.venv-xai/bin/python scripts/xai_smoke.py --dataset /path/to/dataset.h5
```

Its ignored run directory is `output/xai/S00/cpu-top-member-64positive-rows/` and contains
`predictions.h5`, `module_equivalence.json`, and `manifest.json`. The manifest
records the fully resolved config, data/checkpoint content hashes, source rows,
members, environment, git state, command, elapsed time, and hashes of every
output artifact.

## Run inference

This example predicts the first 1,000 varied-gradient rows and writes a
compressed NumPy archive containing the mean, model-to-model standard
deviation, one-standard-deviation flux interval, row indices, and member IDs:

```bash
conda run -n 20240629-01-ML python -m itg_nn.infer \
  /Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/20250102-01_GX_stellarator_dataset.h5 \
  output/predictions_rows_0_1000.npz \
  --start 0 --stop 1000 --gradient-set varied
```

Omit `--stop` to process every row. Use `--gradient-set fixed` for the fixed
temperature- and density-gradient simulations. `--device auto` prefers CUDA,
then Apple MPS, then CPU; pass `--device cpu` for deterministic CPU inference.

The networks predict `max(log(Q), -2)`. The output archive includes both this
native target and `exp(prediction)` in gyro-Bohm units. Its lower and upper
bounds describe ensemble spread, not calibrated confidence intervals.

For varied-gradient rows, the model receives the physical `a_over_LT` value
stored in HDF5. During legacy training, fixed-gradient samples were marked by
negating this feature (so their value was -3 rather than +3). The fixed-set
reader preserves that learned convention and records the actual model inputs
as `model_a_over_LT` and `model_a_over_Ln` in the output archive.

## Reproduce the reference figure

```bash
conda run -n 20240629-01-ML python -m itg_nn.reference_figure \
  /Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/20250102-01_GX_stellarator_dataset.h5 \
  --device cpu \
  --output output/pdf/pred_vs_actual_plot_pre2.pdf
```

The script reconstructs the seeded 80/10/test split and selects only the
varied-gradient test rows, matching the legacy evaluation cohort without a
serialized dataset cache.

## Data-field mapping

The model inputs are HDF5 `raw_feature_tensor`,
`*/a_over_LT`, and `*/a_over_Ln`. The target is `*/Q_avgs`. Direct comparison
with the old pickle files shows that `Q_avgs` is exactly equal to the legacy
field named `Q_avgs_without_FSA_grad_x`; the old field name was misleading,
while the HDF5 name and description correctly record the normalization.

## Large files

If the HDF5 dataset is later added with Git LFS, remove or narrow the `*.h5`
ignore rule first and track the intended filename explicitly with LFS.
