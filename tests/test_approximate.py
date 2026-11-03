"""Tests for the deterministic threshold-based approximate reuse policy."""

import json
import math
import struct
from pathlib import Path

import pytest

from deltavision.approximate import (
    SUPPORTED_THRESHOLDS,
    ThresholdPatchCache,
    compare_feature_vectors,
    mean_absolute_rgb_difference,
    run_threshold_sweep,
)
from deltavision.encoder import RGBImage, encode_image
from deltavision.state import StatefulPatchCache
from deltavision.synthetic import generate_sequences


def _solid_image(width: int, height: int, color: tuple[int, int, int]) -> RGBImage:
    return RGBImage(width, height, bytes(color) * width * height)


def _replace_pixel(
    image: RGBImage, x: int, y: int, color: tuple[int, int, int]
) -> RGBImage:
    pixels = bytearray(image.pixels)
    offset = (y * image.width + x) * 3
    pixels[offset : offset + 3] = bytes(color)
    return RGBImage(image.width, image.height, bytes(pixels))


def _assert_features_equal(actual: tuple, expected: tuple) -> None:
    assert len(actual) == len(expected)
    for actual_vector, expected_vector in zip(actual, expected, strict=True):
        assert actual_vector == pytest.approx(expected_vector, abs=1e-12)


def test_supported_threshold_values_are_exact() -> None:
    assert SUPPORTED_THRESHOLDS == (0.0, 0.01, 0.03, 0.05, 0.10)


def test_threshold_zero_matches_exact_reuse_baseline() -> None:
    initial = _solid_image(4, 4, (30, 70, 110))
    local_change = _replace_pixel(initial, 0, 0, (31, 71, 111))
    larger_change = _replace_pixel(local_change, 3, 3, (240, 200, 160))
    approximate = ThresholdPatchCache(threshold=0.0, patch_size=2)
    exact = StatefulPatchCache(patch_size=2)

    for frame in (initial, local_change, larger_change, larger_change):
        approximate_result = approximate.process_frame(frame)
        exact_result = exact.process_frame(frame)
        _assert_features_equal(approximate_result.features, exact_result.features)
        assert approximate_result.recomputed_patches == exact_result.recomputed_patches
        assert approximate_result.reused_patches == exact_result.reused_patches
        assert approximate_result.refresh_fraction == exact_result.refresh_fraction


def test_reuse_rule_includes_exactly_equal_threshold_boundary() -> None:
    initial = _solid_image(1, 1, (100, 100, 100))
    changed = _solid_image(1, 1, (110, 110, 110))
    score = mean_absolute_rgb_difference(changed.pixels, initial.pixels)
    equal_cache = ThresholdPatchCache(threshold=score, patch_size=1)
    equal_cache.process_frame(initial)
    below_cache = ThresholdPatchCache(threshold=score - 1e-9, patch_size=1)
    below_cache.process_frame(initial)

    equal_result = equal_cache.process_frame(changed)
    below_result = below_cache.process_frame(changed)

    assert equal_result.mean_difference_scores == (score,)
    assert equal_result.recomputed_patches == 0
    assert equal_result.reused_patches == 1
    assert below_result.recomputed_patches == 1


def test_raw_patch_observation_updates_when_feature_is_reused() -> None:
    initial = _solid_image(1, 1, (100, 100, 100))
    first_small_change = _solid_image(1, 1, (103, 103, 103))
    second_small_change = _solid_image(1, 1, (106, 106, 106))
    cache = ThresholdPatchCache(threshold=0.02, patch_size=1)
    cache.process_frame(initial)

    first_result = cache.process_frame(first_small_change)
    second_result = cache.process_frame(second_small_change)

    assert first_result.recomputed_patches == 0
    assert second_result.recomputed_patches == 0
    assert second_result.mean_difference_scores[0] == pytest.approx(3 / 255)
    # The feature remains based on the initial frame even though the raw
    # observation advances each frame, demonstrating possible staleness.
    _assert_features_equal(
        second_result.features, encode_image(initial, patch_size=1).features
    )


def test_cache_reset_forces_full_recomputation() -> None:
    image = _solid_image(4, 4, (45, 90, 135))
    cache = ThresholdPatchCache(threshold=0.10, patch_size=2)
    cache.process_frame(image)
    cache.process_frame(image)

    cache.reset()
    result = cache.process_frame(image)

    assert result.recomputed_patches == result.patch_count == 4
    assert result.reused_patches == 0
    assert result.refresh_fraction == 1.0


@pytest.mark.parametrize("threshold", SUPPORTED_THRESHOLDS)
def test_constant_frames_reuse_all_patches(threshold: float) -> None:
    image = _solid_image(5, 3, (22, 88, 144))
    cache = ThresholdPatchCache(threshold=threshold, patch_size=2)
    cache.process_frame(image)

    result = cache.process_frame(image)

    assert result.recomputed_patches == 0
    assert result.reused_patches == result.patch_count
    assert result.refresh_fraction == 0.0


def test_small_rgb_change_is_reused_but_large_change_is_recomputed() -> None:
    initial = _solid_image(2, 2, (100, 100, 100))
    small = _replace_pixel(initial, 0, 0, (102, 102, 102))
    large = _replace_pixel(small, 1, 1, (180, 180, 180))
    cache = ThresholdPatchCache(threshold=0.05, patch_size=2)
    cache.process_frame(initial)

    small_result = cache.process_frame(small)
    large_result = cache.process_frame(large)

    assert small_result.mean_difference_scores[0] == pytest.approx(2 / (4 * 255))
    assert small_result.recomputed_patches == 0
    assert large_result.mean_difference_scores[0] > 0.05
    assert large_result.recomputed_patches == 1


def test_edge_patch_scores_use_only_actual_rgb_values() -> None:
    initial = _solid_image(3, 3, (10, 40, 70))
    changed = _replace_pixel(initial, 2, 2, (15, 45, 75))
    cache = ThresholdPatchCache(threshold=0.01, patch_size=2)
    first = cache.process_frame(initial)
    result = cache.process_frame(changed)

    assert (first.grid_rows, first.grid_columns) == (2, 2)
    assert result.recomputed_patches == 1
    assert result.mean_difference_scores[-1] == pytest.approx(5 / 255)


def test_feature_error_and_cosine_metrics() -> None:
    actual = ((1.0, 0.0), (0.5, 0.5))
    expected = ((1.0, 0.0), (0.0, 1.0))

    metrics = compare_feature_vectors(actual, expected)

    assert metrics.mean_absolute_error == pytest.approx(0.25)
    assert metrics.maximum_absolute_error == pytest.approx(0.5)
    assert metrics.mean_cosine_similarity == pytest.approx((1 + 1 / math.sqrt(2)) / 2)
    assert metrics.feature_value_count == 4
    assert metrics.feature_vector_count == 2


def test_sweep_serializes_all_metrics_and_resets_cache_per_episode(
    tmp_path: Path,
) -> None:
    manifest = generate_sequences(
        tmp_path / "dataset",
        width=18,
        height=17,
        sequence_length=2,
        num_episodes=2,
        seed=53,
        transformations=("unchanged",),
    )
    output_path = tmp_path / "results" / "sweep.json"
    figure_path = tmp_path / "figures" / "tradeoff.png"

    summary = run_threshold_sweep(
        manifest,
        output_path,
        figure_path,
        patch_size=16,
    )
    serialized = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary == serialized
    assert summary["experiment_label"] == "Synthetic-benchmark findings"
    assert len(summary["results"]) == 5
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">2I", figure_path.read_bytes()[16:24])
    assert (width, height) == (1100, 760)
    for row in summary["results"]:
        assert row["total_patch_count"] == 16
        assert row["recomputed_patches"] == 8
        assert row["reused_patches"] == 8
        assert row["refresh_fraction"] == 0.5
        assert row["mean_absolute_feature_error"] == 0.0
        assert row["maximum_absolute_feature_error"] == 0.0
        assert row["mean_feature_vector_cosine_similarity"] == 1.0
        assert row["patch_computation_proxy"]["value"] == 8
        assert row["patch_computation_proxy"]["value"] == row["recomputed_patches"]
        assert row["wall_clock_latency_seconds"] >= 0
        assert row["comparison_overhead_seconds"] >= 0
        assert row["feature_compute_seconds"] >= 0


def test_invalid_threshold_and_feature_shapes_are_rejected(tmp_path: Path) -> None:
    for threshold in (-0.01, 1.01, math.inf, math.nan):
        with pytest.raises(ValueError, match="threshold"):
            ThresholdPatchCache(threshold=threshold)
    with pytest.raises(ValueError, match="patch_size"):
        ThresholdPatchCache(threshold=0.0, patch_size=0)
    with pytest.raises(ValueError, match="equal non-zero RGB lengths"):
        mean_absolute_rgb_difference(b"\x00", b"\x00")
    with pytest.raises(ValueError, match="equal non-zero"):
        compare_feature_vectors((), ())
    with pytest.raises(ValueError, match="equal non-zero dimensions"):
        compare_feature_vectors(((1.0,),), ((1.0, 2.0),))

    invalid_manifest = tmp_path / "manifest.jsonl"
    invalid_manifest.write_text(
        json.dumps({"episode_id": "none", "frame_paths": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty frame_paths"):
        run_threshold_sweep(invalid_manifest)
