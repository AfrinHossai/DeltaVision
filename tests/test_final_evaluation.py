"""Tests for prespecified Task 17 evaluation and episode-level statistics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from deltavision.final_evaluation import (
    FINAL_CONFIG_PATH,
    handcrafted_metric_view,
    load_final_config,
    validate_final_config,
    validate_final_manifest_provenance,
    validate_result_provenance,
    write_evaluation_artifacts,
)
from deltavision.final_statistics import (
    bootstrap_episode_indices,
    paired_bootstrap_contrast,
)


def test_handcrafted_metric_calculation() -> None:
    sufficient = {
        "true_positives": 6,
        "false_positives": 2,
        "true_negatives": 8,
        "false_negatives": 4,
        "transition_patch_count": 20,
        "transition_recomputed_patches": 11,
        "total_patch_count": 40,
        "recomputed_patches": 21,
        "reused_patches": 19,
        "feature_absolute_error_sum": 2.0,
        "feature_value_count": 100,
        "maximum_absolute_feature_error": 0.25,
    }

    result = handcrafted_metric_view(sufficient)

    assert result["precision"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(0.6)
    assert result["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert result["missed_change_rate"] == pytest.approx(0.4)
    assert result["false_refresh_rate"] == pytest.approx(0.2)
    assert result["transition_refresh_fraction"] == pytest.approx(0.55)
    assert result["all_frame_refresh_fraction"] == pytest.approx(0.525)
    assert result["feature_mean_absolute_error"] == pytest.approx(0.02)


def test_final_manifest_partition_and_seed_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "test" / "manifest.jsonl"
    manifest.parent.mkdir()
    rows = (
        {"episode_id": "test_0", "random_seed": 500},
        {"episode_id": "test_1", "random_seed": 501},
    )
    validate_final_manifest_provenance(
        manifest,
        rows,
        expected_partition="test",
        expected_count=2,
        expected_seed_start=500,
    )
    with pytest.raises(ValueError, match="test manifest"):
        validate_final_manifest_provenance(
            tmp_path / "validation" / "manifest.jsonl",
            rows,
            expected_partition="test",
            expected_count=2,
            expected_seed_start=500,
        )
    with pytest.raises(ValueError, match="seed range"):
        validate_final_manifest_provenance(
            manifest,
            ({"episode_id": "test_0", "random_seed": 501}, rows[1]),
            expected_partition="test",
            expected_count=2,
            expected_seed_start=500,
        )


def test_final_manifest_rejects_duplicate_episode_or_seed(tmp_path: Path) -> None:
    manifest = tmp_path / "validation" / "condition" / "manifest.jsonl"
    rows = (
        {"episode_id": "episode_a", "random_seed": 10},
        {"episode_id": "episode_b", "random_seed": 10},
    )
    with pytest.raises(ValueError, match="duplicate random seeds"):
        validate_final_manifest_provenance(
            manifest,
            rows,
            expected_partition="validation",
            expected_count=2,
            expected_seed_start=10,
        )


def test_bootstrap_episode_indices_are_reproducible() -> None:
    first = bootstrap_episode_indices(7, 40, 912)
    second = bootstrap_episode_indices(7, 40, 912)

    assert first == second
    assert len(first) == 40
    assert all(
        len(sample) == 7 and all(0 <= index < 7 for index in sample) for sample in first
    )


def test_paired_bootstrap_preserves_episode_pairing() -> None:
    result = paired_bootstrap_contrast(
        [5.0, 2.0, 8.0, 4.0],
        [4.0, 1.0, 7.0, 3.0],
        replicates=200,
        seed=17,
    )
    identical = paired_bootstrap_contrast(
        [1.0, 3.0, 9.0],
        [1.0, 3.0, 9.0],
        replicates=100,
        seed=18,
    )

    assert result["point_estimate"] == pytest.approx(1.0)
    assert result["confidence_interval"] == {"lower": 1.0, "upper": 1.0}
    assert identical["confidence_interval"] == {"lower": 0.0, "upper": 0.0}
    assert result["sampling_unit"] == "paired_episode"


def test_prespecified_configuration_loads_and_rejects_policy_drift() -> None:
    config = load_final_config(FINAL_CONFIG_PATH)
    assert len(config["_sha256"]) == 64
    validate_final_config(config)
    config["pretrained_dinov2"]["cache_capacity_per_episode"] = 4

    with pytest.raises(ValueError, match="capacity"):
        validate_final_config(config)


def test_result_provenance_requires_previous_heldout_exposure() -> None:
    result = {
        "data_provenance": {
            "handcrafted_final_evaluation": {
                "partition": "previously_inspected_held_out_test"
            },
            "pretrained_final_evaluation": {"partition": "synthetic_validation"},
        },
        "research_integrity": {
            "held_out_is_blind": False,
            "known_prior_held_out_inspection": True,
            "pretrained_used_validation_only": True,
        },
    }
    validate_result_provenance(result)
    result["research_integrity"]["held_out_is_blind"] = True

    with pytest.raises(ValueError, match="must not be described as blind"):
        validate_result_provenance(result)


def test_evaluation_json_and_csv_serialization(tmp_path: Path) -> None:
    result = {
        "handcrafted": {
            "methods": {
                "test_method": {
                    "per_episode": [
                        {
                            "episode_id": "episode_01",
                            "metrics": {"f1": 0.5, "precision": 0.6},
                        }
                    ]
                }
            }
        },
        "pretrained_dinov2": {"conditions": {}},
    }
    json_path = tmp_path / "nested" / "evaluation.json"
    csv_path = tmp_path / "nested" / "evaluation.csv"

    write_evaluation_artifacts(result, json_path, csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == result
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["representation_family"] == "handcrafted_rgb_statistics"
    assert rows[0]["episode_id"] == "episode_01"
    assert rows[0]["f1"] == "0.5"


def test_evaluation_serialization_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        write_evaluation_artifacts(
            {"invalid": float("nan")}, tmp_path / "result.json", tmp_path / "result.csv"
        )
