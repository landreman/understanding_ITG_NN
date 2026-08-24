from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
ARTIFACTS = ROOT / "reports" / "xai" / "S10_artifacts"


def _rows(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s10_headlines_recompute_from_committed_tables():
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    matches = _rows("unit_matches.csv")
    motifs = _rows("motif_catalog.csv")
    cka = _rows("cka.csv")
    members = _rows("member_clusters.csv")
    final = [row for row in matches if row["consensus_gate"] == "True"]
    preliminary = [row for row in matches if row["pre_regime_consensus_gate"] == "True"]

    assert len(matches) == summary["matched_pairs"] == 582
    assert len(final) == summary["consensus_edges"] == 163
    assert len(preliminary) == summary["pre_regime_consensus_edges"] == 497
    assert len(motifs) == summary["consensus_motifs"] == 8
    assert len(cka) == summary["cka_pair_layer_rows"] == 29_700
    assert len(members) == summary["members"] == 100
    assert summary["stable_rows"] == 240 and summary["unstable_rows"] == 760
    motif_eligible = [
        row for row in final
        if float(row["equilibrium_bootstrap_recurrence"])
        >= summary["motif_minimum_recurrence"]
        and min(
            float(row["causal_effect_similarity_stable_or_near_floor"]),
            float(row["causal_effect_similarity_unstable"]),
        ) >= summary["motif_minimum_regime_causal_similarity"]
    ]
    assert len(motif_eligible) == summary["motif_eligible_edges"] == 74
    assert sum(int(row["edge_count"]) for row in motifs) == 70
    assert {row["cohort"] for row in members} == {
        "stored_validation_top_10",
        "stored_validation_ranks_11_50",
        "stored_validation_ranks_51_100",
    }


def test_s10_consensus_is_one_to_one_and_agrees_in_both_regimes():
    matches = _rows("unit_matches.csv")
    final = [row for row in matches if row["consensus_gate"] == "True"]
    assert final
    for row in final:
        assert float(row["equilibrium_bootstrap_recurrence"]) >= 0.7
        assert float(row["activation_flux_residual_similarity"]) >= 0.5
        assert float(row["causal_effect_similarity_stable_or_near_floor"]) >= 0.5
        assert float(row["causal_effect_similarity_unstable"]) >= 0.5
        assert 1.0 <= float(row["causal_effect_rms_magnitude_ratio"]) <= 4.1
        assert row["causal_effect_validity"] == "deliberately_off_manifold_diagnostic"

    for motif in _rows("motif_catalog.csv"):
        units = motif["unit_ids"].split("|")
        member_ids = [unit.rsplit(":u", 1)[0] for unit in units]
        assert len(member_ids) == len(set(member_ids)) == int(motif["member_count"])
        assert float(motif["minimum_recurrence"]) >= 0.7
        assert float(motif["minimum_causal_similarity"]) >= 0.7
        assert int(motif["s05_screened_unit_count"]) >= int(
            motif["s05_supported_unit_count"]
        )


def test_s10_cka_covers_every_pair_and_layer():
    rows = _rows("cka.csv")
    layers = {row["layer_name"] for row in rows}
    assert layers == {
        "canonical_atrous_layer_1",
        "canonical_atrous_layer_2",
        "canonical_atrous_layer_3",
        "canonical_atrous_layer_4",
        "canonical_atrous_layer_5",
        "invariant_bottleneck",
    }
    for layer in layers:
        subset = [row for row in rows if row["layer_name"] == layer]
        assert len(subset) == 4_950
        assert all(row["bootstrap_group"] == "equilibrium_files" for row in subset)
        assert all(0.0 <= float(row["cka"]) <= 1.0 for row in subset)


def test_motif_threshold_sensitivity_pins_the_binding_gate():
    rows = _rows("motif_threshold_sensitivity.csv")
    assert [float(row["minimum_regime_causal_similarity"]) for row in rows] == [
        0.5, 0.6, 0.7, 0.8
    ]
    assert [int(row["motif_count"]) for row in rows] == [14, 12, 8, 4]


def test_s10_manifest_hashes_every_committed_headline_artifact():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    for name in (
        "unit_matches.csv",
        "motif_catalog.csv",
        "motif_threshold_sensitivity.csv",
        "cka.csv",
        "member_clusters.csv",
        "member_distances.csv",
        "cohort_comparison.csv",
        "architecture.csv",
        "summary.json",
        "cross_model_cka.png",
        "member_dendrogram.png",
    ):
        assert _sha256(ARTIFACTS / name) == manifest["output_hashes"][name]
    assert manifest["postprocessing"]["source_artifact"] == "member_signatures.h5"
    assert manifest["source_hashes"]["runner"] == (
        "fae7242276544d55179377441e4a67dafbf9f4df5eb9b9d2e9243b524afb8dfc"
    )
    reproduction = manifest["postprocessing"]["reproduction_source_hashes"]
    assert reproduction["runner"] == _sha256(ROOT / "scripts" / "xai_s10_cross_model.py")
    assert reproduction["config"] == _sha256(ROOT / "configs" / "xai" / "S10_cross_model.json")
    assert manifest["postprocessing"]["reproduction_config"]["source_hashes"] == reproduction
