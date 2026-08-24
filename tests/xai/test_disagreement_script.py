import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "xai_s11_disagreement.py"
CONFIG = ROOT / "configs" / "xai" / "S11_disagreement.json"


def _module():
    spec = importlib.util.spec_from_file_location("xai_s11_disagreement", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s11_cli_exposes_required_reproducibility_controls():
    parser = _module().build_parser()
    destinations = {action.dest for action in parser._actions}
    assert {"config", "members", "rows", "device", "seed", "resume", "output_dir"} <= destinations


def test_s11_config_freezes_native_thresholds_features_and_grouping():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["canonical_function"] == "invariant_tilde_f"
    assert config["estimand"] == "native max(log Q, -2)"
    assert config["resampling_unit"] == "equilibrium_files"
    assert config["high_error_threshold_native"] > 0
    assert config["high_spread_threshold_native"] > 0
    assert config["diagnostic_features"] == [
        "support_warning_score",
        "equilibrium_class",
        "a_over_lt",
        "a_over_ln",
        "q_stds",
        "symmetry_error",
        "motif_activation_dispersion",
        "concept_activation_dispersion",
        "nfp",
        "iota",
        "shat",
        "d_pressure_d_s",
        "aspect",
        "rho",
        "aspect_over_rho",
    ]
    assert config["feature_selection"] == "none_frozen_before_residual_analysis"


def test_crossfit_table_accepts_a_single_equilibrium_class_pilot():
    rows = _module()._crossfit_rows(
        {"support_warning_score": np.arange(8.0), "q_stds": np.arange(8.0) ** 2},
        np.zeros(8, dtype=int),
        {"ensemble_spread": np.linspace(0.1, 0.8, 8)},
        np.asarray([f"eq{index}" for index in range(8)]),
        folds=4,
        alpha=1.0,
        seed=3,
    )
    assert len(rows) == 1
    assert rows[0]["split_unit"] == "equilibrium_files"


def test_runner_spread_gradient_uses_canonical_native_outputs():
    class FakeMember:
        def __init__(self, sign):
            self.sign = sign

        def invariant(self, geometry, a_lt, a_ln):
            means = geometry.mean(dim=1)
            return means[:, 0] + self.sign * means[:, 1]

        def original(self, geometry, a_lt, a_ln):
            return torch.exp(self.invariant(geometry, a_lt, a_ln)) + 100.0

    geometry = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[2.0, -3.0], [4.0, -1.0], [6.0, -2.0]]]
    )
    predictions, gradient = _module()._spread_gradients(
        {"m0": FakeMember(1.0), "m1": FakeMember(-1.0)},
        ("m0", "m1"),
        geometry,
        torch.zeros(2),
        torch.zeros(2),
        batch_size=2,
        device=torch.device("cpu"),
    )
    means = geometry.mean(dim=1)
    np.testing.assert_allclose(
        predictions, torch.stack((means[:, 0] + means[:, 1], means[:, 0] - means[:, 1])).numpy()
    )
    expected = np.zeros_like(geometry.numpy())
    expected[:, :, 1] = means[:, 1].sign().numpy()[:, None] / 3
    np.testing.assert_allclose(gradient, expected)


def test_runner_gradient_summary_applies_scale_and_keeps_regimes_separate():
    spread_gradient = np.zeros((2, 2, 7))
    spread_gradient[0, :, 0] = 1.0
    spread_gradient[1, :, 0] = 3.0
    member_gradient = np.zeros((2, 2, 2, 7))
    rows = _module()._gradient_summary(
        spread_gradient,
        member_gradient,
        np.asarray([True, False]),
        np.asarray([0, 1]),
        np.asarray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ("m0", "m1"),
    )
    selected = {
        row["regime"]: row
        for row in rows
        if row["outcome"] == "ensemble_spread" and row["channel"] == 0
    }
    assert selected["stable_or_near_floor"]["mean_absolute_robust_scaled_gradient"] == 2.0
    assert selected["unstable"]["mean_absolute_robust_scaled_gradient"] == 6.0


def test_runner_keeps_spread_native_and_member_symmetry_before_cancellation():
    predictions = np.asarray([[0.0, 2.0], [2.0, 4.0]])
    signed = np.asarray([[1.0, -3.0], [-1.0, 1.0]])
    spread = _module().ensemble_spread(predictions)
    member_absolute = np.abs(signed).mean(axis=0)
    np.testing.assert_allclose(spread, [1.0, 1.0])
    np.testing.assert_allclose(member_absolute, [1.0, 2.0])
    assert member_absolute[0] > abs(signed[:, 0].mean())
    source = SCRIPT.read_text(encoding="utf-8")
    assert "spread = ensemble_spread(predictions)" in source
    assert "symmetry_member_mean_absolute = np.abs(original_shift_signed).mean(axis=0)" in source


def test_runner_perturbation_summary_keeps_exact_symmetry_null():
    reference = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    rows = _module()._perturbation_summary(
        reference,
        {"joint_shift": reference.copy()},
        {"joint_shift": _module().ValidityTag.EXACT_SYMMETRY},
        np.asarray([True, False]),
        ("m0", "m1"),
    )
    assert all(row["rms_change_native"] == 0.0 for row in rows)
    assert {row["validity"] for row in rows} == {"exact_symmetry"}
    edited = reference + np.asarray([[1.0, -1.0], [2.0, -2.0]])
    changed = _module()._perturbation_summary(
        reference,
        {"known": edited},
        {"known": _module().ValidityTag.OFF_MANIFOLD},
        np.asarray([True, False]),
        ("m0", "m1"),
    )
    member_all = [row for row in changed if row["outcome"] == "member_native_prediction" and row["regime"] == "all"]
    assert [row["signed_mean_change_native"] for row in member_all] == [0.0, 0.0]
    np.testing.assert_allclose([row["rms_change_native"] for row in member_all], [1.0, 2.0])
    stable_m0 = next(row for row in changed if row["member_id"] == "m0" and row["regime"] == "stable_or_near_floor")
    assert stable_m0["signed_mean_change_native"] == 1.0


def test_runner_motif_and_concept_dispersion_helpers_are_finite(tmp_path):
    motif_path = tmp_path / "motifs.csv"
    motif_path.write_text("motif_id,unit_ids\nmotif_001,m0:u000|m1:u000\n", encoding="utf-8")
    bottlenecks = [
        np.column_stack((np.arange(4.0), np.asarray([0.0, 1.0, 0.0, 1.0]))),
        np.column_stack((np.arange(4.0)[::-1], np.asarray([1.0, 0.0, 1.0, 0.0]))),
    ]
    motif = _module()._motif_dispersion(motif_path, ("m0", "m1"), bottlenecks)
    assert motif.shape == (4,)
    assert np.all(motif > 0)

    z = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    geometry = np.zeros((4, 96, 7))
    for sample in range(4):
        phase = sample * 0.3
        geometry[sample, :, 0] = 1.2 + 0.1 * np.cos(z + phase)
        geometry[sample, :, 1] = np.sin(z + phase)
        geometry[sample, :, 2] = np.cos(z - phase)
        geometry[sample, :, 3] = 0.2 * np.sin(2 * z + phase)
        geometry[sample, :, 4] = 2.0 + 0.1 * np.cos(z)
        geometry[sample, :, 5] = 0.1 * np.sin(z)
        geometry[sample, :, 6] = np.square(1.0 + 0.1 * np.sin(z + phase))
    concept = _module()._concept_dispersion(geometry, np.ones(7), bottlenecks)
    assert concept.shape == (4,)
    assert np.isfinite(concept).all()
    assert np.all(concept >= 0)
