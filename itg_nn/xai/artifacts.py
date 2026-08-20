"""Atomic artifact writing and machine-readable run provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a content hash without loading a large artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Describe a source file by path, byte size, modification time, and SHA-256."""

    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def _atomic_json(path: Path, content: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_output(arguments: Sequence[str], cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_success(arguments: Sequence[str], cwd: Path) -> bool:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, text=True, capture_output=True, check=False
    )
    return result.returncode == 0


def _package_versions() -> dict[str, str]:
    names = ("itg-nn", "torch", "numpy", "h5py", "captum")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


class RunArtifacts:
    """Write a run directory and finalise its manifest only after success."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started = time.monotonic()
        self._outputs: list[Path] = []

    def _output_path(self, name: str) -> Path:
        path = (self.output_dir / name).resolve()
        if self.output_dir not in path.parents:
            raise ValueError("artifact name must remain within the run directory")
        return path

    def write_json(self, name: str, content: Mapping[str, Any]) -> Path:
        path = self._output_path(name)
        _atomic_json(path, content)
        self._outputs.append(path)
        return path

    def write_text(self, name: str, content: str) -> Path:
        """Write a UTF-8 text artifact atomically and include it in the manifest."""

        path = self._output_path(name)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
        self._outputs.append(path)
        return path

    def register_existing(self, name: str) -> Path:
        """Register an already-written file (for example, a plot) for hashing."""

        path = self._output_path(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        self._outputs.append(path)
        return path

    def write_hdf5(
        self,
        name: str,
        arrays: Mapping[str, np.ndarray],
        *,
        axes: Mapping[str, Sequence[str]],
        attributes: Mapping[str, Any] | None = None,
        compression: str | None = None,
    ) -> Path:
        """Write labeled arrays atomically in a self-describing HDF5 artifact."""

        path = self._output_path(name)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with h5py.File(temporary, "w") as h5_file:
            h5_file.attrs["axes_json"] = json.dumps({key: list(value) for key, value in axes.items()})
            for key, value in (attributes or {}).items():
                h5_file.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
            for key, value in arrays.items():
                array = np.asarray(value)
                compress = compression if array.ndim > 0 and array.dtype.kind != "O" else None
                dataset = h5_file.create_dataset(key, data=array, compression=compress)
                dataset.attrs["axes"] = json.dumps(list(axes.get(key, ())))
        temporary.replace(path)
        self._outputs.append(path)
        return path

    def finalize(
        self,
        *,
        config: Mapping[str, Any],
        dataset: str | Path,
        checkpoint: str | Path,
        member_ids: Sequence[str],
        row_ids: Sequence[int],
        gradient_set: str,
        device: str | torch.device,
        repository: str | Path,
        command: Sequence[str] | None = None,
        published_dir: str | Path | None = None,
    ) -> Path:
        """Record all registered provenance and hashes in ``manifest.json``.

        The run directory under ``output/xai/`` is git-ignored, so an automated
        reviewer working from a checkout cannot see the manifest there and has
        to take the report's word that it exists and is complete. Pass
        ``published_dir`` — normally ``reports/xai/SNN_artifacts`` — to copy the
        manifest to a committed location. It is a few kilobytes.
        """

        repository_path = Path(repository).resolve()
        manifest = {
            "command": list(command if command is not None else sys.argv),
            "config": dict(config),
            "checkpoint": file_fingerprint(checkpoint),
            "dataset": file_fingerprint(dataset),
            "device": str(device),
            "git_commit": _git_output(["rev-parse", "HEAD"], repository_path),
            "git_tree": _git_output(["rev-parse", "HEAD^{tree}"], repository_path),
            "git_dirty": bool(_git_output(["status", "--porcelain"], repository_path)),
            "git_tracked_dirty": not (
                _git_success(["diff", "--quiet"], repository_path)
                and _git_success(["diff", "--cached", "--quiet"], repository_path)
            ),
            "gradient_set": gradient_set,
            "member_ids": list(member_ids),
            "output_hashes": {
                path.name: sha256_file(path) for path in sorted(set(self._outputs))
            },
            "package_versions": _package_versions(),
            "python": {
                "implementation": platform.python_implementation(),
                "version": sys.version,
            },
            "row_ids": [int(row) for row in row_ids],
            "seed": config.get("seed"),
            "wall_time_seconds": time.monotonic() - self._started,
        }
        path = self._output_path("manifest.json")
        _atomic_json(path, manifest)
        if published_dir is not None:
            published = Path(published_dir)
            published.mkdir(parents=True, exist_ok=True)
            _atomic_json(published / "manifest.json", manifest)
        return path
