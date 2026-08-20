# List available recipes when you type `make`
.PHONY: default test check smoke env review-slice

default:
	@grep -E '^[A-Za-z][A-Za-z0-9_-]*:' $(MAKEFILE_LIST) | cut -d: -f1

# The test suite. Runs without the external HDF5 dataset: synthetic fixtures plus
# the committed checkpoint. This is what CI and the PR review both run.
test:
	python -m pytest -q --durations=10

# The gate. This is what "done" means for code, before the pilot and the report.
check: test

# Registered S00 CPU smoke calculation. Needs the external dataset and .venv-xai,
# so it is a local check only, never a CI one.
smoke:
	.venv-xai/bin/python scripts/xai_smoke.py

# Create or refresh this worktree's .venv-xai.
env:
	bash scripts/setup_xai_env.sh

# Rebuild the committed review slice. Needs the external dataset. This is a
# deliberate act: the slice is frozen, and regenerating it invalidates every
# review that ran against the old one. See AGENTS.md.
review-slice:
	.venv-xai/bin/python scripts/build_review_slice.py
