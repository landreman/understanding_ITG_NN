from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from itg_nn.xai.completeness import (
    _expand,
    _folds,
    _nested_prediction,
    _predict,
    _ridge_fit,
    _slope,
    grouped_completeness,
    grouped_integrated_hessian,
    stratified_directional_effects,
)


def _test_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - float(np.sum((target - prediction) ** 2)) / denominator


def _cyclic_fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(24), 3)
    phase = rng.integers(0, 12, len(groups))
    geometry = rng.normal(size=(len(groups), 12, 2))
    signal = geometry[:, :, 0].mean(1)
    null = geometry[:, :, 1].mean(1)
    drive = rng.uniform(-1, 1, len(groups))
    native = 0.4 + 2.0 * signal + 1.5 * drive + 3.0 * signal * drive
    # A joint cyclic roll leaves the analytic concepts and output unchanged.
    rolled = np.stack([np.roll(row, shift, axis=0) for row, shift in zip(geometry, phase)])
    np.testing.assert_allclose(rolled[:, :, 0].mean(1), signal, atol=1e-12)
    return groups, signal, null, drive, native


def test_grouped_completeness_recovers_increment_and_null_control():
    groups, signal, null, drive, native = _cyclic_fixture()
    second_drive = np.zeros_like(drive)
    sets = {
        "drive_only": np.column_stack([drive, second_drive]),
        "known_signal": np.column_stack([drive, second_drive, signal]),
        "null_added": np.column_stack([drive, second_drive, signal, null]),
    }
    result = grouped_completeness(
        sets, native, groups, outer_folds=4, inner_folds=3,
        penalties=(1e-6, 1e-3, 0.1), seed=19, bootstrap_replicates=100,
    )
    by_name = {row.name: row for row in result}
    assert by_name["known_signal"].held_out_r2 > 0.98
    assert by_name["known_signal"].increment_r2 > 0.5
    assert abs(by_name["null_added"].increment_r2) < 0.03
    assert np.array_equal(by_name["known_signal"].fold, by_name["null_added"].fold)
    for fold in np.unique(by_name["known_signal"].fold):
        test_groups = set(groups[by_name["known_signal"].fold == fold])
        train_groups = set(groups[by_name["known_signal"].fold != fold])
        assert test_groups.isdisjoint(train_groups)


def test_completeness_is_deterministic_and_native_not_exponentiated():
    groups, signal, _, drive, native = _cyclic_fixture()
    sets = {"base": np.column_stack([drive, np.zeros_like(drive)]), "full": np.column_stack([drive, np.zeros_like(drive), signal])}
    kwargs = dict(outer_folds=4, inner_folds=3, penalties=(1e-6, 1e-3), seed=3, bootstrap_replicates=50)
    first = grouped_completeness(sets, native, groups, **kwargs)
    second = grouped_completeness(sets, native, groups, **kwargs)
    np.testing.assert_allclose(first[-1].prediction, second[-1].prediction)
    assert first[-1].held_out_r2 > grouped_completeness(sets, np.exp(native), groups, **kwargs)[-1].held_out_r2


def test_completeness_bootstrap_is_invariant_to_row_duplication():
    groups, signal, _, drive, native = _cyclic_fixture()
    values = np.column_stack([drive, np.zeros_like(drive), signal])
    kwargs = dict(
        outer_folds=4,
        inner_folds=3,
        penalties=(1e-8,),
        seed=43,
        bootstrap_replicates=50,
    )
    original = grouped_completeness({"known": values}, native, groups, **kwargs)[0]
    duplicated = grouped_completeness(
        {"known": np.repeat(values, 3, axis=0)},
        np.repeat(native, 3),
        np.repeat(groups, 3),
        **kwargs,
    )[0]
    np.testing.assert_allclose(
        (original.increment_ci95_lower, original.increment_ci95_upper),
        (duplicated.increment_ci95_lower, duplicated.increment_ci95_upper),
        atol=1e-8,
    )
    assert original.increment_ci95_upper - original.increment_ci95_lower > 0.01
    unique = np.unique(groups)
    positions_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(kwargs["seed"] + 65537)
    baseline = np.full_like(native, native.mean())
    draws = []
    for _ in range(kwargs["bootstrap_replicates"]):
        positions = np.concatenate([
            positions_by_group[group]
            for group in rng.choice(unique, len(unique), replace=True)
        ])
        draws.append(
            _test_r2(native[positions], original.prediction[positions])
            - _test_r2(native[positions], baseline[positions])
        )
    np.testing.assert_allclose(
        (original.increment_ci95_lower, original.increment_ci95_upper),
        np.quantile(draws, (0.025, 0.975)),
        atol=1e-12,
    )


def test_paired_gain_bootstrap_is_invariant_to_row_duplication(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    paired_gain = importlib.import_module("xai_s09_completeness")._paired_gain
    rng = np.random.default_rng(47)
    groups = np.repeat(np.arange(30), 4)
    target = rng.normal(size=len(groups))
    group_error = rng.normal(scale=0.8, size=30)
    baseline = target + np.repeat(group_error, 4)
    candidate = target + 0.5 * np.repeat(group_error, 4)
    original = paired_gain(target, candidate, baseline, groups, 100, 53)
    duplicated = paired_gain(
        np.repeat(target, 3),
        np.repeat(candidate, 3),
        np.repeat(baseline, 3),
        np.repeat(groups, 3),
        100,
        53,
    )
    np.testing.assert_allclose(original, duplicated, atol=1e-12)
    assert original[1] - original[0] > 0.01
    unique = np.unique(groups)
    positions_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    draw_rng = np.random.default_rng(53)
    draws = []
    for _ in range(100):
        positions = np.concatenate([
            positions_by_group[group]
            for group in draw_rng.choice(unique, len(unique), replace=True)
        ])
        draws.append(
            _test_r2(target[positions], candidate[positions])
            - _test_r2(target[positions], baseline[positions])
        )
    np.testing.assert_allclose(original[:2], np.quantile(draws, (0.025, 0.975)))


def test_stratified_interaction_recovers_signed_drive_change_and_null():
    groups, signal, null, drive, native = _cyclic_fixture()
    rows = stratified_directional_effects(
        {"signal": signal, "null": null}, drive, native, groups,
        bins=3, bootstrap_replicates=100, seed=11,
    )
    signal_rows = [row for row in rows if row.concept == "signal"]
    null_rows = [row for row in rows if row.concept == "null"]
    assert signal_rows[-1].slope - signal_rows[0].slope > 3.0
    assert max(abs(row.slope) for row in null_rows) < 1.0
    assert all(row.validity_tag == "observed-comparison" for row in rows)


def test_stratified_interaction_bootstrap_is_invariant_to_row_duplication():
    groups, signal, _, drive, native = _cyclic_fixture()
    kwargs = dict(bins=3, bootstrap_replicates=100, seed=37)
    original = stratified_directional_effects(
        {"signal": signal}, drive, native, groups, **kwargs
    )
    duplicated = stratified_directional_effects(
        {"signal": np.repeat(signal, 3)},
        np.repeat(drive, 3),
        np.repeat(native, 3),
        np.repeat(groups, 3),
        **kwargs,
    )
    np.testing.assert_allclose(
        [(row.ci95_lower, row.ci95_upper) for row in original],
        [(row.ci95_lower, row.ci95_upper) for row in duplicated],
        atol=1e-12,
    )
    assert all(row.ci95_upper - row.ci95_lower > 0.05 for row in original)
    edges = np.quantile(drive, np.linspace(0, 1, 4))
    positions_by_group = {
        group: np.flatnonzero(groups == group) for group in np.unique(groups)
    }
    for bin_index, row in enumerate(original):
        mask = (drive >= edges[bin_index]) & (
            (drive <= edges[bin_index + 1])
            if bin_index == 2 else (drive < edges[bin_index + 1])
        )
        present = np.unique(groups[mask])
        draw_rng = np.random.default_rng(37 + bin_index)
        draws = []
        for _ in range(100):
            positions = np.concatenate([
                positions_by_group[group]
                for group in draw_rng.choice(present, len(present), replace=True)
            ])
            positions = positions[mask[positions]]
            draws.append(_slope(signal[positions], native[positions]))
        np.testing.assert_allclose(
            (row.ci95_lower, row.ci95_upper),
            np.quantile(draws, (0.025, 0.975)),
        )


def test_grouped_integrated_hessian_recovers_selected_mixed_term():
    groups, signal, null, drive, native = _cyclic_fixture()
    values = np.column_stack([drive, np.zeros_like(drive), signal, null])
    rows = grouped_integrated_hessian(
        values, native, groups, concept_names=("signal", "null"),
        outer_folds=4, inner_folds=3, penalties=(1e-6, 1e-3, 0.1),
        bootstrap_replicates=100, seed=17,
    )
    lookup = {(row.drive_index, row.concept): row for row in rows}
    assert lookup[(0, "signal")].mixed_derivative == pytest.approx(3.0, abs=0.15)
    assert lookup[(0, "signal")].fold_minimum_mixed_derivative < 3.1
    assert lookup[(0, "signal")].fold_maximum_mixed_derivative > 2.9
    assert lookup[(0, "signal")].fold_sign_agreement == 1.0
    assert abs(lookup[(0, "null")].mixed_derivative) < 0.3
    assert all(row.validity_tag == "observed-comparison" for row in rows)


def test_grouped_integrated_hessian_reports_fold_sign_disagreement():
    rng = np.random.default_rng(23)
    groups = np.repeat(np.arange(40), 4)
    drive = rng.uniform(-1, 1, len(groups))
    signal = rng.normal(size=len(groups))
    seed = 41
    fold = _folds(groups, 4, seed)
    # Holding out fold 0 fits three positive-interaction cohorts; holding out
    # any other fold includes the deliberately dominant negative cohort.
    coefficient = np.where(fold == 0, -30.0, 10.0)
    native = coefficient * drive * signal
    rows = grouped_integrated_hessian(
        np.column_stack([drive, np.zeros_like(drive), signal]),
        native,
        groups,
        concept_names=("signal",),
        outer_folds=4,
        inner_folds=3,
        penalties=(1e-8,),
        bootstrap_replicates=20,
        seed=seed,
    )
    term = next(row for row in rows if row.drive_index == 0)
    assert term.fold_minimum_mixed_derivative < -1.0
    assert term.fold_maximum_mixed_derivative > 1.0
    assert term.fold_sign_agreement < 1.0
    duplicated = grouped_integrated_hessian(
        np.repeat(np.column_stack([drive, np.zeros_like(drive), signal]), 3, axis=0),
        np.repeat(native, 3),
        np.repeat(groups, 3),
        concept_names=("signal",),
        outer_folds=4,
        inner_folds=3,
        penalties=(1e-8,),
        bootstrap_replicates=20,
        seed=seed,
    )
    duplicated_term = next(row for row in duplicated if row.drive_index == 0)
    np.testing.assert_allclose(
        (term.ci95_lower, term.ci95_upper),
        (duplicated_term.ci95_lower, duplicated_term.ci95_upper),
        atol=1e-8,
    )
    assert term.ci95_upper - term.ci95_lower > 0.1
    raw = np.column_stack([drive, np.zeros_like(drive), signal])
    expanded = _expand(raw)
    _, selected = _nested_prediction(
        expanded,
        native,
        groups,
        outer_fold=fold,
        inner_folds=3,
        penalties=(1e-8,),
        seed=seed,
    )
    scale = np.subtract(*np.quantile(raw, (0.75, 0.25), axis=0))
    scale[scale < 1e-8] = 1.0
    h_drive, h_signal = 0.1 * scale[0], 0.1 * scale[2]
    mixed = np.empty(len(native))
    for outer in np.unique(fold):
        train, test = fold != outer, fold == outer
        model = _ridge_fit(expanded[train], native[train], float(selected[int(outer)]))
        evaluations = []
        for drive_sign, signal_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            changed = raw[test].copy()
            changed[:, 0] += drive_sign * h_drive
            changed[:, 2] += signal_sign * h_signal
            evaluations.append(_predict(model, _expand(changed)))
        mixed[test] = (
            evaluations[0] - evaluations[1] - evaluations[2] + evaluations[3]
        ) / (4 * h_drive * h_signal)
    unique = np.unique(groups)
    positions_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    draw_rng = np.random.default_rng(seed)
    draws = []
    for _ in range(20):
        positions = np.concatenate([
            positions_by_group[group]
            for group in draw_rng.choice(unique, len(unique), replace=True)
        ])
        draws.append(float(np.mean(mixed[positions])))
    np.testing.assert_allclose(
        (term.ci95_lower, term.ci95_upper),
        np.quantile(draws, (0.025, 0.975)),
        atol=1e-12,
    )


def test_repeated_rows_with_too_few_equilibria_are_rejected():
    groups, signal, _, drive, native = _cyclic_fixture()
    with pytest.raises(ValueError, match="groups"):
        grouped_completeness(
                {"base": np.column_stack([drive, signal])}, native,
            np.zeros_like(groups), outer_folds=4, inner_folds=3,
            penalties=(0.1,), seed=0, bootstrap_replicates=10,
        )


def test_outer_predictions_are_fit_without_the_held_out_fold():
    groups, signal, _, drive, native = _cyclic_fixture()
    values = np.column_stack([drive, np.zeros_like(drive), signal])
    noisy_native = native + np.random.default_rng(31).normal(0, 0.25, len(native))
    noisy_native[groups == groups[-1]] += 2.0
    result = grouped_completeness(
        {"known": values}, noisy_native, groups, outer_folds=4, inner_folds=3,
        penalties=(1e-6,), seed=29, bootstrap_replicates=20,
    )[0]
    expanded = _expand(values)
    leaky = _predict(_ridge_fit(expanded, noisy_native, 1e-6), expanded)
    assert not np.allclose(result.prediction, leaky)
    for fold in np.unique(result.fold):
        train = result.fold != fold
        expected = _predict(
            _ridge_fit(expanded[train], noisy_native[train], 1e-6), expanded[~train]
        )
        np.testing.assert_allclose(result.prediction[~train], expected, atol=1e-10)
