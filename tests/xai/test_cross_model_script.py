from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "xai_s10_cross_model.py"
SPEC = importlib.util.spec_from_file_location("xai_s10_cross_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_registered_rank_cohorts_are_not_collapsed():
    assert MODULE._member_cohort(1) == "stored_validation_top_10"
    assert MODULE._member_cohort(10) == "stored_validation_top_10"
    assert MODULE._member_cohort(11) == "stored_validation_ranks_11_50"
    assert MODULE._member_cohort(50) == "stored_validation_ranks_11_50"
    assert MODULE._member_cohort(51) == "stored_validation_ranks_51_100"
    assert MODULE._member_cohort(100) == "stored_validation_ranks_51_100"


def test_consensus_requires_recurrence_and_comparable_causal_effects():
    units = ("m0:u000", "m1:u002", "m2:u001", "m3:u000")
    edges = [
        {"left": units[0], "right": units[1], "recurrence": 0.9, "causal_similarity": 0.8},
        {"left": units[1], "right": units[2], "recurrence": 0.85, "causal_similarity": 0.75},
        {"left": units[2], "right": units[3], "recurrence": 0.95, "causal_similarity": 0.2},
    ]
    motifs = MODULE._consensus_components(
        units, edges, minimum_recurrence=0.7, minimum_causal_similarity=0.7
    )
    assert len(motifs) == 1
    assert motifs[0]["unit_ids"] == ["m0:u000", "m1:u002", "m2:u001"]
    assert motifs[0]["member_count"] == 3
    assert units[3] not in motifs[0]["unit_ids"]


def test_consensus_never_places_two_units_from_one_member_in_a_motif():
    units = ("m0:u000", "m0:u001", "m1:u000", "m2:u000")
    edges = [
        {"left": units[0], "right": units[2], "recurrence": 1.0, "causal_similarity": 1.0},
        {"left": units[1], "right": units[3], "recurrence": 0.9, "causal_similarity": 1.0},
        {"left": units[2], "right": units[3], "recurrence": 0.8, "causal_similarity": 1.0},
    ]
    motifs = MODULE._consensus_components(
        units, edges, minimum_recurrence=0.7, minimum_causal_similarity=0.7
    )
    assert len(motifs) == 2
    for motif in motifs:
        member_ids = [unit.rsplit(":u", 1)[0] for unit in motif["unit_ids"]]
        assert len(member_ids) == len(set(member_ids))


def test_catalog_reads_the_distinct_motif_threshold_from_config():
    units = ("m0:u000", "m1:u000", "m2:u000")
    edges = [
        {"left": units[0], "right": units[1], "recurrence": 1.0, "causal_similarity": 0.9},
        {"left": units[1], "right": units[2], "recurrence": 1.0, "causal_similarity": 0.6},
    ]
    resolved = {
        "minimum_match_recurrence": 0.7,
        "minimum_regime_causal_similarity": 0.5,
        "minimum_motif_regime_causal_similarity": 0.7,
    }
    strict = MODULE._catalog_motifs(units, edges, resolved)
    assert strict[0]["member_count"] == 2
    resolved["minimum_motif_regime_causal_similarity"] = 0.5
    loose = MODULE._catalog_motifs(units, edges, resolved)
    assert loose[0]["member_count"] == 3


def test_annotate_motifs_counts_only_independently_supported_s05_names(tmp_path):
    table = tmp_path / "unit_motifs.csv"
    table.write_text(
        "unit_id,motif_status,claimed_concept\n"
        "m0:u001,supported_named_motif,concept_a\n"
        "m0:u002,screened_without_support,\n",
        encoding="utf-8",
    )
    rows = [
        {"unit_ids": "m0:u001|m0:u002|m1:u003"},
        {"unit_ids": "m1:u004|m2:u005"},
    ]
    MODULE._annotate_motifs(rows, s05_unit_motifs=table)
    assert rows[0]["s05_screened_unit_count"] == 2
    assert rows[0]["s05_supported_unit_count"] == 1
    assert rows[0]["s05_supported_concepts"] == "concept_a"
    assert rows[0]["interpretive_label"] == "concept_a"
    assert rows[1]["s05_screened_unit_count"] == 0
    assert rows[1]["s05_supported_unit_count"] == 0
    assert rows[1]["interpretive_label"] == "unresolved_by_S05_vocabulary"


def test_motif_eligible_count_reads_the_stricter_motif_threshold():
    rows = [
        {
            "equilibrium_bootstrap_recurrence": 0.9,
            "causal_effect_similarity_stable_or_near_floor": 0.8,
            "causal_effect_similarity_unstable": 0.8,
        },
        {
            "equilibrium_bootstrap_recurrence": 0.9,
            "causal_effect_similarity_stable_or_near_floor": 0.6,
            "causal_effect_similarity_unstable": 0.6,
        },
    ]
    resolved = {
        "minimum_match_recurrence": 0.7,
        "minimum_regime_causal_similarity": 0.5,
        "minimum_motif_regime_causal_similarity": 0.7,
    }
    assert MODULE._count_motif_eligible_edges(rows, resolved) == 1


def test_effect_signature_keeps_the_two_output_regimes_separate():
    import numpy as np

    effects = np.asarray([[2.0], [2.0], [1.0], [-1.0]])
    stable = np.asarray([True, True, False, False])
    signature = MODULE._effect_signature(effects, stable)
    assert signature[0] == pytest.approx([2.0, 0.0, 2.0, 1.0])


def test_stratified_indices_are_unique_sorted_and_regime_balanced():
    import numpy as np

    stable = np.asarray([True] * 4 + [False] * 6)
    chosen = MODULE._stratified_indices(stable, count=5, seed=4)
    assert chosen.tolist() == sorted(set(chosen.tolist()))
    assert len(chosen) == 5
    assert int(stable[chosen].sum()) == 2


def test_channel_scales_follow_explicit_channel_indices(tmp_path):
    table = tmp_path / "scales.csv"
    table.write_text(
        "channel,iqr\n"
        "6,7\n0,1\n4,5\n2,3\n5,6\n1,2\n3,4\n",
        encoding="utf-8",
    )
    assert MODULE._channel_scales(table).tolist() == pytest.approx([1, 2, 3, 4, 5, 6, 7])


def test_fixed_member_profile_retains_registered_quantiles():
    import numpy as np

    values = np.asarray([[0, 10], [1, 20], [2, 30], [3, 40], [4, 50]], dtype=float)
    assert MODULE._fixed_member_profile(values) == pytest.approx(
        [0, 1, 2, 3, 4, 10, 20, 30, 40, 50]
    )


def test_standardize_removes_constant_columns_and_scales_the_rest():
    import numpy as np

    values = np.asarray([[4.0, 1.0], [4.0, 2.0], [4.0, 3.0]])
    standardized = MODULE._standardize(values)
    assert standardized.shape == (3, 1)
    assert standardized.mean() == pytest.approx(0.0)
    assert standardized.std() == pytest.approx(1.0)


def test_density_signature_spectrum_ignores_density_offsets():
    import numpy as np

    phase = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    density = np.stack((2.0 + np.sin(phase), 3.0 + np.cos(2 * phase)), axis=0)
    density = np.stack((density, 1.5 * density), axis=0)
    shifted = density + np.asarray([5.0, 11.0])[None, :, None]
    original = MODULE._density_signature(density)
    offset = MODULE._density_signature(shifted)
    assert original[:, :6] == pytest.approx(offset[:, :6])
    assert np.any(original[:, :6] > 0)


def test_outlier_trimmed_cka_removes_the_high_joint_norm_row():
    import numpy as np

    left = np.column_stack((np.arange(20.0), np.arange(20.0) ** 2))
    right = left.copy()
    right[-1] = np.asarray([-1000.0, 1000.0])
    trimmed, retained = MODULE._outlier_trimmed_cka(left, right)
    assert retained == 19
    assert trimmed != pytest.approx(MODULE.linear_cka(left, right))


def test_regime_mask_assigns_floor_and_threshold_rows_to_stable():
    import numpy as np

    target = np.asarray([-2.0, -1.9, -1.8])
    assert MODULE._regime_mask(target, -1.9).tolist() == [True, True, False]


def test_panel_covariates_include_native_target_and_both_drives():
    import numpy as np

    target = np.asarray([-2.0, 0.5])
    a_lt = np.asarray([1.0, 2.0])
    a_ln = np.asarray([3.0, 4.0])
    covariates = MODULE._panel_covariates(target, a_lt, a_ln)
    assert np.allclose(covariates, [[-2.0, 1.0, 3.0], [0.5, 2.0, 4.0]])


def test_concept_profile_keeps_peak_absolute_and_signed_mean_blocks():
    import numpy as np

    selectivity = np.asarray([[-3.0, 2.0, 1.0], [1.0, -4.0, 3.0]])
    assert MODULE._concept_profile(selectivity) == pytest.approx([3, 4, 3, -1, -1, 2])


def test_probe_representation_flattens_and_standardizes_each_column():
    import numpy as np

    values = np.asarray([[[4.0, 1.0]], [[4.0, 2.0]], [[4.0, 3.0]]])
    probe = MODULE._probe_representation(values)
    assert probe.shape == (3, 1)
    assert probe.mean(axis=0) == pytest.approx([0.0])
    assert probe.std(axis=0) == pytest.approx([1.0])


def test_acceptance_summary_refuses_missing_lower_rank_comparison():
    summary = {
        "stable_rows": 20,
        "unstable_rows": 40,
        "match_bootstrap_group": "equilibrium_files",
        "flux_residualized_matching": True,
        "causal_effect_validity": "deliberately_off_manifold_diagnostic",
        "cohort_member_counts": {
            "stored_validation_top_10": 10,
            "stored_validation_ranks_11_50": 0,
            "stored_validation_ranks_51_100": 0,
        },
    }
    with pytest.raises(ValueError, match="lower-ranked"):
        MODULE._validate_summary(summary)


def test_regime_causal_gate_rejects_stable_opposition():
    import numpy as np

    stable = np.asarray([True, True, False, False])
    rows = [{
        "left_member_id": "m0", "right_member_id": "m1",
        "left_unit_id": "m0:u000", "right_unit_id": "m1:u000",
        "consensus_gate": True,
    }]
    effects = (
        np.asarray([[1.0], [1.0], [1.0], [1.0]]),
        np.asarray([[-3.0], [-3.0], [1.0], [1.0]]),
    )
    MODULE._apply_regime_causal_gate(
        rows, effects=effects, stable=stable, member_ids=("m0", "m1"),
        minimum_similarity=0.5,
    )
    assert rows[0]["causal_effect_similarity_stable_or_near_floor"] == pytest.approx(-1.0)
    assert rows[0]["causal_effect_similarity_unstable"] == pytest.approx(1.0)
    assert rows[0]["causal_effect_rms_magnitude_ratio_stable_or_near_floor"] == pytest.approx(3.0)
    assert rows[0]["causal_effect_rms_magnitude_ratio_unstable"] == pytest.approx(1.0)
    assert rows[0]["causal_effect_rms_magnitude_ratio"] == pytest.approx(5.0 ** 0.5)
    assert rows[0]["pre_regime_consensus_gate"] is True
    assert rows[0]["consensus_gate"] is False


def test_regime_causal_gate_does_not_promote_serialized_false():
    import numpy as np

    rows = [{
        "left_member_id": "m0", "right_member_id": "m1",
        "left_unit_id": "m0:u000", "right_unit_id": "m1:u000",
        "consensus_gate": "False",
    }]
    effects = (np.ones((4, 1)), np.ones((4, 1)))
    MODULE._apply_regime_causal_gate(
        rows, effects=effects, stable=np.asarray([True, True, False, False]),
        member_ids=("m0", "m1"), minimum_similarity=0.5,
    )
    assert rows[0]["causal_regime_gate"] is True
    assert rows[0]["pre_regime_consensus_gate"] is False
    assert rows[0]["consensus_gate"] is False


def test_regime_causal_gate_rejects_zero_rms_in_either_regime():
    import numpy as np

    rows = [{
        "left_member_id": "m0", "right_member_id": "m1",
        "left_unit_id": "m0:u000", "right_unit_id": "m1:u000",
        "consensus_gate": True,
    }]
    effects = (
        np.asarray([[0.0], [0.0], [1.0], [1.0]]),
        np.asarray([[1.0], [1.0], [1.0], [1.0]]),
    )
    with pytest.raises(ValueError, match="near-zero RMS"):
        MODULE._apply_regime_causal_gate(
            rows, effects=effects,
            stable=np.asarray([True, True, False, False]),
            member_ids=("m0", "m1"), minimum_similarity=0.5,
        )


def test_member_attribution_applies_registered_channel_scales():
    import numpy as np
    import torch

    class MeanInvariant:
        def invariant(self, geometry, a_lt, a_ln):
            del a_lt, a_ln
            return geometry.mean(dim=1)

    geometry = torch.arange(4 * 6 * 2, dtype=torch.float32).reshape(4, 6, 2)
    attribution = MODULE._member_attribution(
        MeanInvariant(),
        geometry,
        torch.zeros(4),
        torch.zeros(4),
        scales=np.asarray([1.0, 1000.0]),
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert attribution.shape == (4, 2)
    assert np.allclose(attribution[:, 1], 1000.0 * attribution[:, 0])


def test_resume_rejects_changed_or_corrupted_output(tmp_path):
    import hashlib

    dataset = tmp_path / "dataset.h5"
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "run"
    output.mkdir()
    dataset.write_bytes(b"dataset")
    checkpoint.write_bytes(b"checkpoint")
    artifact = output / "summary.json"
    artifact.write_bytes(b"{}\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    resolved = {"run_id": "synthetic", "source_hashes": {"runner": "abc"}}
    manifest = {
        "config": resolved,
        "dataset": {"sha256": digest(dataset)},
        "checkpoint": {"sha256": digest(checkpoint)},
        "output_hashes": {"summary.json": digest(artifact)},
    }
    MODULE._validate_resume_manifest(
        manifest, resolved=resolved, output_dir=output,
        dataset=dataset, checkpoint=checkpoint,
    )
    artifact.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="output hash mismatch"):
        MODULE._validate_resume_manifest(
            manifest, resolved=resolved, output_dir=output,
            dataset=dataset, checkpoint=checkpoint,
        )
