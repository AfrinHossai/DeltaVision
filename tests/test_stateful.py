"""Deterministic tests for exact RGB-patch cache reuse."""

import json
from pathlib import Path

import pytest

from deltavision.encoder import RGBImage, encode_image
from deltavision.state import StatefulPatchCache, run_stateful
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


def _shift_all_channels(image: RGBImage, amount: int) -> RGBImage:
    pixels = bytes(min(255, value + amount) for value in image.pixels)
    return RGBImage(image.width, image.height, pixels)


def _assert_features_match(actual: tuple, expected: tuple) -> None:
    assert len(actual) == len(expected)
    for actual_patch, expected_patch in zip(actual, expected, strict=True):
        assert actual_patch == pytest.approx(expected_patch, abs=1e-12)


def test_first_frame_recomputes_every_patch() -> None:
    cache = StatefulPatchCache(patch_size=2)

    result = cache.process_frame(_solid_image(5, 3, (20, 80, 140)))

    assert result.patch_count == 6
    assert result.recomputed_patches == 6
    assert result.reused_patches == 0
    assert result.refresh_fraction == 1.0
    assert result.feature_dimensions == 6


def test_identical_frame_reuses_every_patch() -> None:
    image = _solid_image(4, 4, (30, 60, 90))
    cache = StatefulPatchCache(patch_size=2)
    cache.process_frame(image)

    result = cache.process_frame(image)

    assert result.patch_count == 4
    assert result.recomputed_patches == 0
    assert result.reused_patches == 4
    assert result.refresh_fraction == 0.0
    assert result.feature_compute_seconds == 0.0


def test_local_change_recomputes_only_its_patch() -> None:
    initial = _solid_image(4, 4, (10, 30, 50))
    changed = _replace_pixel(initial, 1, 0, (200, 180, 160))
    cache = StatefulPatchCache(patch_size=2)
    cache.process_frame(initial)

    result = cache.process_frame(changed)

    assert result.recomputed_patches == 1
    assert result.reused_patches == 3
    assert result.refresh_fraction == 0.25


def test_global_pixel_change_recomputes_all_patches() -> None:
    initial = _solid_image(4, 4, (20, 80, 140))
    changed = _shift_all_channels(initial, 5)
    cache = StatefulPatchCache(patch_size=2)
    cache.process_frame(initial)

    result = cache.process_frame(changed)

    assert result.recomputed_patches == result.patch_count == 4
    assert result.reused_patches == 0
    assert result.refresh_fraction == 1.0


def test_partial_edge_patch_is_compared_using_in_bounds_pixels() -> None:
    initial = _solid_image(3, 3, (15, 35, 55))
    changed = _replace_pixel(initial, 2, 2, (215, 195, 175))
    cache = StatefulPatchCache(patch_size=2)
    cache.process_frame(initial)

    result = cache.process_frame(changed)
    baseline = encode_image(changed, patch_size=2)

    assert (result.grid_rows, result.grid_columns) == (2, 2)
    assert result.recomputed_patches == 1
    assert result.reused_patches == 3
    _assert_features_match(result.features, baseline.features)


def test_cache_reset_forces_full_refresh_for_the_next_episode() -> None:
    image = _solid_image(4, 4, (45, 90, 135))
    cache = StatefulPatchCache(patch_size=2)
    first_episode = cache.process_frame(image)
    within_episode = cache.process_frame(image)
    cache.reset()
    second_episode = cache.process_frame(image)

    assert first_episode.recomputed_patches == 4
    assert within_episode.recomputed_patches == 0
    assert second_episode.recomputed_patches == 4


def test_stateful_features_match_frame_independent_baseline() -> None:
    initial = _solid_image(5, 3, (20, 80, 140))
    frames = (
        initial,
        _replace_pixel(initial, 0, 0, (200, 30, 10)),
        _shift_all_channels(initial, 4),
        initial,
    )
    cache = StatefulPatchCache(patch_size=2)

    for frame in frames:
        stateful = cache.process_frame(frame)
        baseline = encode_image(frame, patch_size=2)
        _assert_features_match(stateful.features, baseline.features)


def test_refresh_fraction_is_recomputed_over_total_patches() -> None:
    initial = _solid_image(4, 4, (25, 50, 75))
    changed = _replace_pixel(initial, 3, 3, (180, 160, 140))
    cache = StatefulPatchCache(patch_size=2)
    first = cache.process_frame(initial)
    partial = cache.process_frame(changed)
    unchanged = cache.process_frame(changed)

    assert first.refresh_fraction == first.recomputed_patches / first.patch_count
    assert partial.refresh_fraction == partial.recomputed_patches / partial.patch_count
    assert partial.refresh_fraction == 0.25
    assert unchanged.refresh_fraction == 0.0


def test_stateful_runner_serializes_records_and_resets_per_episode(
    tmp_path: Path,
) -> None:
    manifest_path = generate_sequences(
        tmp_path / "dataset",
        width=18,
        height=17,
        sequence_length=3,
        num_episodes=2,
        seed=33,
        transformations=("unchanged",),
    )
    output_path = run_stateful(
        manifest_path, tmp_path / "stateful.jsonl", patch_size=16
    )
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 6
    for record in records:
        assert record["patch_count"] == 4
        assert record["feature_dimensions"] == 6
        assert len(record["features"]) == 4
        assert all(len(vector) == 6 for vector in record["features"])
        assert record["recomputed_patches"] + record["reused_patches"] == 4
        assert record["refresh_fraction"] == (
            record["recomputed_patches"] / record["patch_count"]
        )
        assert record["processing_latency_seconds"] >= 0
        assert record["comparison_overhead_seconds"] >= 0
        assert record["processing_cost_proxy"]["value"] == record["recomputed_patches"]

    for episode_id in ("episode_0000", "episode_0001"):
        first_frame = next(
            record
            for record in records
            if record["episode_id"] == episode_id and record["frame_index"] == 0
        )
        assert first_frame["recomputed_patches"] == first_frame["patch_count"]


def test_invalid_patch_size_and_manifest_path_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="patch_size"):
        StatefulPatchCache(patch_size=0)

    invalid_manifest = tmp_path / "manifest.jsonl"
    invalid_manifest.write_text(
        json.dumps({"episode_id": "missing", "frame_paths": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty frame_paths"):
        run_stateful(invalid_manifest, tmp_path / "out.jsonl")
