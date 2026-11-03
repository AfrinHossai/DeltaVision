"""Deterministic unit tests for saved pretrained representation validation."""

from __future__ import annotations

import math

import pytest

from deltavision.pretrained_validation import (
    count_non_finite_values,
    expected_feature_shape,
    reproducibility_metrics,
    validate_feature_shape,
    validate_manifest_records,
    validate_metadata,
)

MOCK_MODEL_CONFIG = {
    "model_type": "mock-vit",
    "patch_size": 2,
    "hidden_size": 3,
    "num_register_tokens": 2,
}


def test_shape_validation_uses_patch_grid_and_special_tokens() -> None:
    assert expected_feature_shape("spatial_tokens", MOCK_MODEL_CONFIG, 8, 6) == (
        4,
        3,
        3,
    )
    assert expected_feature_shape("global_embedding", MOCK_MODEL_CONFIG, 8, 6) == (3,)
    assert expected_feature_shape("token_sequence", MOCK_MODEL_CONFIG, 8, 6) == (15, 3)
    validate_feature_shape((4, 3, 3), (4, 3, 3))
    with pytest.raises(ValueError, match="does not match expected"):
        validate_feature_shape((4, 3, 2), (4, 3, 3))


def test_finite_value_check_counts_nan_and_both_infinities() -> None:
    assert count_non_finite_values([0.0, 1.0, math.nan, math.inf, -math.inf]) == 3
    assert count_non_finite_values([0.0, 1.0, -2.5]) == 0


def test_manifest_matching_checks_episode_frame_and_source_path() -> None:
    expected = [
        ("episode_0000", 0, "episode_0000/frame_0000.ppm"),
        ("episode_0000", 1, "episode_0000/frame_0001.ppm"),
    ]
    records = [
        {
            "episode_id": "episode_0000",
            "frame_index": 0,
            "source_frame_path": "episode_0000/frame_0000.ppm",
            "feature_array_path": "features/0.f32le",
        },
        {
            "episode_id": "episode_0000",
            "frame_index": 1,
            "source_frame_path": "episode_0000/frame_0001.ppm",
            "feature_array_path": "features/1.f32le",
        },
    ]

    report = validate_manifest_records(expected, records)

    assert report["missing_records"] == []
    assert report["unexpected_records"] == []
    assert report["source_path_mismatches"] == []


def test_manifest_matching_reports_missing_and_wrong_source_path() -> None:
    expected = [
        ("episode_0000", 0, "episode_0000/frame_0000.ppm"),
        ("episode_0000", 1, "episode_0000/frame_0001.ppm"),
    ]
    records = [
        {
            "episode_id": "episode_0000",
            "frame_index": 0,
            "source_frame_path": "episode_0000/wrong.ppm",
            "feature_array_path": "features/0.f32le",
        }
    ]

    report = validate_manifest_records(expected, records)

    assert report["missing_records"] == [
        {"episode_id": "episode_0000", "frame_index": 1}
    ]
    assert report["source_path_mismatches"] == [
        {
            "episode_id": "episode_0000",
            "frame_index": 0,
            "expected_source_frame_path": "episode_0000/frame_0000.ppm",
            "recorded_source_frame_path": "episode_0000/wrong.ppm",
        }
    ]


def test_duplicate_frame_records_and_feature_paths_are_detected() -> None:
    expected = [("episode_0000", 0, "episode_0000/frame_0000.ppm")]
    record = {
        "episode_id": "episode_0000",
        "frame_index": 0,
        "source_frame_path": "episode_0000/frame_0000.ppm",
        "feature_array_path": "features/0.f32le",
    }

    report = validate_manifest_records(expected, [record, record.copy()])

    assert report["duplicate_frame_records"] == [
        {"episode_id": "episode_0000", "frame_index": 0, "count": 2}
    ]
    assert report["duplicate_feature_paths"] == [
        {"feature_array_path": "features/0.f32le", "count": 2}
    ]


def test_reproducibility_metrics_are_deterministic_for_mock_representations() -> None:
    exact = reproducibility_metrics([1.0, -2.0, 3.0], [1.0, -2.0, 3.0])
    assert exact["exact_match"] is True
    assert exact["maximum_absolute_difference"] == 0.0
    assert exact["mean_absolute_difference"] == 0.0
    assert exact["cosine_similarity"] == pytest.approx(1.0)

    changed = reproducibility_metrics([1.0, 0.0], [1.1, 0.0])
    assert changed["exact_match"] is False
    assert changed["maximum_absolute_difference"] == pytest.approx(0.1)
    assert changed["mean_absolute_difference"] == pytest.approx(0.05)
    assert changed["cosine_similarity"] == pytest.approx(1.0)


def test_metadata_validation_checks_recorded_inference_settings() -> None:
    expected = {
        "model_identifier": "facebook/dinov2-small",
        "model_revision": "a" * 40,
        "device": "cpu",
        "preprocessing": {"crop_height": 224, "image_mean": [0.485, 0.456, 0.406]},
    }
    assert validate_metadata(expected, expected) == []

    mismatched = {**expected, "device": "cuda"}
    errors = validate_metadata(mismatched, expected)
    assert len(errors) == 1
    assert "metadata.device" in errors[0]
