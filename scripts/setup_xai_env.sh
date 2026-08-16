#!/usr/bin/env bash
# Create the project-local XAI virtual environment for this checkout or worktree.
#
# The venv inherits torch/numpy/h5py from conda environment 20240629-01-ML via
# --system-site-packages, so it costs a few megabytes rather than gigabytes, and
# AGENTS.md's rule against modifying that conda environment is respected.
#
# Run once per worktree, from the repository root:
#
#     bash scripts/setup_xai_env.sh
#
set -euo pipefail

CONDA_ENV="${ITG_CONDA_ENV:-20240629-01-ML}"
VENV_DIR="${ITG_VENV_DIR:-.venv-xai}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f pyproject.toml ]]; then
  echo "error: not at the repository root (no pyproject.toml)" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "creating $VENV_DIR from conda environment $CONDA_ENV"
  conda run -n "$CONDA_ENV" python -m venv --system-site-packages "$VENV_DIR"
else
  echo "reusing existing $VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e '.[xai]'

"$PY" - <<'PYCHECK'
import importlib.metadata as md
import torch
names = ("itg-nn", "torch", "numpy", "h5py", "captum")
for name in names:
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        version = "MISSING"
    print(f"  {name:>8s} {version}")
if not torch.__version__:
    raise SystemExit("torch is not importable from the venv")
import itg_nn  # noqa: F401  - proves the editable install points at this worktree
print(f"  itg_nn resolved from {itg_nn.__file__}")
PYCHECK

echo
echo "ready. run step commands with: $VENV_DIR/bin/python scripts/xai_*.py ..."
