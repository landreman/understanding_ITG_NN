"""Reference plotting and metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ensemble import EnsemblePrediction


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return the coefficient of determination in the trained log space."""

    actual_64 = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted_64 = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual_64.shape != predicted_64.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    residual_sum = np.sum((actual_64 - predicted_64) ** 2)
    total_sum = np.sum((actual_64 - actual_64.mean()) ** 2)
    return float(1.0 - residual_sum / total_sum)


def save_prediction_comparison(
    actual_log_heat_flux: np.ndarray,
    prediction: EnsemblePrediction,
    output_path: str | Path,
) -> float:
    """Reproduce the legacy prediction-versus-actual PDF."""

    actual_log = np.asarray(actual_log_heat_flux).reshape(-1)
    predicted_log = prediction.mean_log_heat_flux.reshape(-1)
    predicted_std = prediction.std_log_heat_flux.reshape(-1)
    if not (len(actual_log) == len(predicted_log) == len(predicted_std)):
        raise ValueError("Actual, predicted, and uncertainty lengths do not match")

    score = r2_score(actual_log, predicted_log)
    actual = np.exp(actual_log)
    predicted = np.exp(predicted_log)
    errorbar_lower = np.exp(predicted_log - predicted_std)
    errorbar_upper = np.exp(predicted_log + predicted_std)

    plt.rcParams.update({"font.size": 14})
    figure, axes = plt.subplots(figsize=(6, 6))
    axes.vlines(
        actual,
        errorbar_lower,
        errorbar_upper,
        alpha=0.5,
        linewidth=0.5,
        color="gray",
    )
    axes.scatter(actual, predicted, s=1.5, alpha=0.5, color="r", zorder=10)
    axes.plot(
        [1e-1, 1e3],
        [1e-1, 1e3],
        "k--",
        linewidth=1,
        label="Perfect Prediction",
        zorder=10,
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim([1e-1, 1e3])
    axes.set_ylim([1e-1, 1e3])
    axes.set_aspect("equal", "box")
    axes.set_xlabel(r"True heat flux $Q/Q_{gB}$ from GX")
    axes.set_ylabel(r"Predicted heat flux $Q/Q_{gB}$")
    axes.set_title(
        "Comparison of Ensemble Predictions with Actual Values", fontsize=12
    )
    axes.text(0.04, 0.92, rf"$R^2$ = {score:0.3f}", transform=axes.transAxes)
    axes.grid(True, linewidth=0.5, alpha=0.5)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return score
