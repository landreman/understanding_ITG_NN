from __future__ import annotations

import numpy as np
import pytest

from itg_nn.xai.completeness import (
    _expand,
    _folds,
    _predict,
    _ridge_fit,
    grouped_completeness,
    grouped_integrated_hessian,
    stratified_directional_effects,
)


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
