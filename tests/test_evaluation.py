"""Deterministic patch-level evaluation tests against generated pixel masks."""

import json
import struct
from pathlib import Path

import pytest

from deltavision.approximate import SUPPORTED_THRESHOLDS, ThresholdPatchCache
from deltavision.encoder import RGBImage
from deltavision.evaluation import (
    ConfusionCounts,
    count_patch_decisions,
    detection_metrics,
    evaluate_change_detection,
    ground_truth_mask_to_patch_labels,
)
from deltavision.synthetic import generate_sequences


def _solid_image(width: int, height: int, color: tuple[int, int, int]) -> RGBImage:
    return RGBImage(width, height, bytes(color) * width * height)


def test_ground_truth_mask_maps_any_changed_pixel_to_its_patch() -> None:
    mask = bytearray(4 * 4)
    mask[2 * 4 + 1] = 255
    mask[3 * 4 + 2] = 1

    labels = ground_truth_mask_to_patch_labels(bytes(mask), 4, 4, patch_size=2)

    assert labels == (False, False, True, True)


def test_single_pixel_change_marks_one_edge_patch() -> None:
    mask = bytearray(3 * 3)
    mask[2 * 3 + 2] = 255

    labels = ground_truth_mask_to_patch_labels(bytes(mask), 3, 3, patch_size=2)

    assert labels == (False, False, False, True)


def test_no_change_and_fully_changed_masks() -> None:
    unchanged = ground_truth_mask_to_patch_labels(bytes(16), 4, 4, patch_size=2)
    changed = ground_truth_mask_to_patch_labels(bytes([255]) * 16, 4, 4, patch_size=2)

    assert unchanged == (False, False, False, False)
    assert changed == (True, True, True, True)


def test_confusion_matrix_counts_all_four_outcomes() -> None:
    counts = count_patch_decisions(
        (True, True, False, False), (True, False, True, False)
    )

    assert counts == ConfusionCounts(
        true_positives=1,
        false_positives=1,
        true_negatives=1,
        false_negatives=1,
    )
    metrics = detection_metrics(counts)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["missed_change_rate"] == 0.5
    assert metrics["false_refresh_rate"] == 0.5


def test_no_change_transition_has_only_true_negatives(tmp_path: Path) -> None:
    manifest = generate_sequences(
        tmp_path / "dataset",
        width=16,
        height=16,
        sequence_length=2,
        num_episodes=1,
        seed=62,
        transformations=("unchanged",),
    )
    summary = evaluate_change_detection(
        manifest,
        tmp_path / "result.json",
        tmp_path / "plot.png",
        patch_size=8,
        thresholds=(0.10,),
    )
    detection = summary["results"][0]["change_detection"]

    assert detection["true_positives"] == 0
    assert detection["false_positives"] == 0
    assert detection["true_negatives"] == 4
    assert detection["false_negatives"] == 0
    assert detection["precision"] == 0.0
    assert detection["recall"] == 0.0
    assert detection["f1"] == 0.0
    assert detection["missed_change_rate"] == 0.0
    assert detection["false_refresh_rate"] == 0.0


def test_threshold_zero_has_no_false_positive_or_false_negative(tmp_path: Path) -> None:
    manifest = generate_sequences(
        tmp_path / "dataset",
        width=16,
        height=16,
        sequence_length=2,
        seed=17,
        transformations=("global_brightness",),
    )
    summary = evaluate_change_detection(
        manifest,
        tmp_path / "result.json",
        tmp_path / "plot.png",
        patch_size=8,
        thresholds=(0.0,),
    )
    detection = summary["results"][0]["change_detection"]

    assert detection["false_positives"] == 0
    assert detection["false_negatives"] == 0
    assert detection["precision"] == 1.0
    assert detection["recall"] == 1.0
    assert detection["missed_change_rate"] == 0.0


def test_actual_threshold_gate_can_produce_a_false_negative() -> None:
    cache = ThresholdPatchCache(threshold=0.10, patch_size=1)
    before = _solid_image(1, 1, (100, 100, 100))
    after = _solid_image(1, 1, (110, 110, 110))
    cache.process_frame(before)

    gate_result = cache.process_frame(after)
    truth = ground_truth_mask_to_patch_labels(bytes([255]), 1, 1, patch_size=1)
    counts = count_patch_decisions(truth, gate_result.refresh_decisions)

    assert gate_result.refresh_decisions == (False,)
    assert counts.false_negatives == 1
    assert detection_metrics(counts)["missed_change_rate"] == 1.0


def test_division_by_zero_metrics_are_defined_as_zero() -> None:
    metrics = detection_metrics(ConfusionCounts(0, 0, 0, 0))

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["missed_change_rate"] == 0.0
    assert metrics["false_refresh_rate"] == 0.0


def test_evaluation_serializes_distinct_detection_and_feature_metrics(
    tmp_path: Path,
) -> None:
    manifest = generate_sequences(
        tmp_path / "dataset",
        width=18,
        height=17,
        sequence_length=3,
        num_episodes=2,
        seed=75,
        transformations=("movement", "recoloring"),
    )
    output = tmp_path / "results" / "change.json"
    figure = tmp_path / "figures" / "change.png"

    summary = evaluate_change_detection(manifest, output, figure, patch_size=16)
    serialized = json.loads(output.read_text(encoding="utf-8"))

    assert summary == serialized
    assert summary["experiment_label"] == "Synthetic-benchmark findings"
    assert summary["episode_count"] == 2
    assert summary["frame_count"] == 6
    assert summary["thresholds"] == list(SUPPORTED_THRESHOLDS)
    assert "rendered RGB pixel changes" in summary["ground_truth_definition"]
    assert "not semantic changes" in summary["ground_truth_definition"]
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">2I", figure.read_bytes()[16:24]) == (1100, 760)
    for row in summary["results"]:
        assert set(row) == {
            "threshold",
            "change_detection",
            "feature_approximation",
            "computation",
        }
        assert "true_positives" in row["change_detection"]
        assert "mean_absolute_feature_error" in row["feature_approximation"]
        assert "refresh_fraction" in row["computation"]


def test_invalid_mask_inputs_and_patch_decisions_are_rejected() -> None:
    with pytest.raises(ValueError, match="byte count"):
        ground_truth_mask_to_patch_labels(b"short", 4, 4, patch_size=2)
    with pytest.raises(ValueError, match="patch_size"):
        ground_truth_mask_to_patch_labels(bytes(4), 2, 2, patch_size=0)
    with pytest.raises(ValueError, match="counts must match"):
        count_patch_decisions((True,), ())
