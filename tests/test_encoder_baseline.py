"""Coverage for patch-statistic encoding and the independent-frame runner."""

import json
from pathlib import Path

import pytest

from deltavision.baseline import load_manifest, run_baseline
from deltavision.encoder import RGBImage, encode_image, read_ppm
from deltavision.synthetic import generate_sequences


def _solid_image(width: int, height: int, color: tuple[int, int, int]) -> RGBImage:
    return RGBImage(width, height, bytes(color) * width * height)


def test_feature_grid_and_vector_shapes() -> None:
    image = _solid_image(7, 5, (30, 90, 180))

    result = encode_image(image, patch_size=3)

    assert (result.grid_rows, result.grid_columns) == (2, 3)
    assert result.feature_dimensions == 6
    assert result.processed_patches == 6
    assert all(len(vector) == 6 for vector in result.features)


def test_constant_color_features_have_correct_means_and_zero_stddev() -> None:
    image = _solid_image(4, 3, (32, 128, 255))
    expected = (32 / 255, 128 / 255, 1.0, 0.0, 0.0, 0.0)

    result = encode_image(image, patch_size=2)

    assert all(vector == pytest.approx(expected) for vector in result.features)


def test_feature_extraction_is_deterministic() -> None:
    image = _solid_image(5, 4, (71, 132, 19))

    first = encode_image(image, patch_size=3)
    second = encode_image(image, patch_size=3)

    assert first == second


def test_frames_are_encoded_independently_without_temporal_state() -> None:
    first_image = _solid_image(4, 4, (20, 80, 140))
    second_image = _solid_image(4, 4, (200, 100, 40))

    first_before = encode_image(first_image, patch_size=2)
    encode_image(second_image, patch_size=2)
    first_after = encode_image(first_image, patch_size=2)

    assert first_after == first_before
    assert first_after != encode_image(second_image, patch_size=2)


def test_partial_edge_patches_use_only_in_bounds_pixels() -> None:
    image = _solid_image(3, 3, (51, 102, 153))
    expected = (51 / 255, 102 / 255, 153 / 255, 0.0, 0.0, 0.0)

    result = encode_image(image, patch_size=2)

    assert (result.grid_rows, result.grid_columns) == (2, 2)
    assert all(vector == pytest.approx(expected) for vector in result.features)


def test_manifest_loader_resolves_generated_relative_paths(tmp_path: Path) -> None:
    manifest_path = generate_sequences(
        tmp_path / "dataset",
        width=24,
        height=20,
        sequence_length=2,
        num_episodes=2,
        seed=8,
    )

    episodes = load_manifest(manifest_path)

    assert len(episodes) == 2
    assert [episode.episode_id for episode in episodes] == [
        "episode_0000",
        "episode_0001",
    ]
    assert all(len(episode.frames) == 2 for episode in episodes)
    assert all(frame.path.is_file() for episode in episodes for frame in episode.frames)
    assert all(
        not Path(frame.relative_path).is_absolute()
        for episode in episodes
        for frame in episode.frames
    )


def test_runner_will_not_overwrite_its_manifest_or_frames(tmp_path: Path) -> None:
    manifest_path = generate_sequences(
        tmp_path / "dataset",
        width=24,
        height=20,
        sequence_length=1,
        seed=9,
    )
    frame_path = load_manifest(manifest_path)[0].frames[0].path
    original_manifest = manifest_path.read_bytes()
    original_frame = frame_path.read_bytes()

    with pytest.raises(ValueError, match="input manifest"):
        run_baseline(manifest_path, manifest_path)
    with pytest.raises(ValueError, match="frame image"):
        run_baseline(manifest_path, frame_path)

    assert manifest_path.read_bytes() == original_manifest
    assert frame_path.read_bytes() == original_frame


def test_runner_serializes_features_timing_and_patch_cost(tmp_path: Path) -> None:
    manifest_path = generate_sequences(
        tmp_path / "dataset",
        width=18,
        height=17,
        sequence_length=2,
        num_episodes=2,
        seed=11,
    )
    output_path = tmp_path / "outputs" / "baseline.jsonl"

    result_path = run_baseline(manifest_path, output_path, patch_size=16)

    records = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    for record in records:
        assert record["patch_grid_dimensions"] == {"rows": 2, "columns": 2}
        assert record["patch_size"] == 16
        assert record["feature_dimensions"] == 6
        assert len(record["features"]) == 4
        assert all(len(vector) == 6 for vector in record["features"])
        assert record["processing_time_seconds"] >= 0
        assert record["processed_patches"] == 4
        assert record["processing_cost_proxy"]["name"] == "patch_operations"
        assert record["processing_cost_proxy"]["value"] == 4
        assert (
            "not a measured FLOP count" in record["processing_cost_proxy"]["definition"]
        )


def test_invalid_images_patch_sizes_and_manifest_paths_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="byte count"):
        RGBImage(2, 2, b"too short")
    with pytest.raises(ValueError, match="patch_size"):
        encode_image(_solid_image(2, 2, (0, 0, 0)), patch_size=0)

    invalid_ppm = tmp_path / "invalid.ppm"
    invalid_ppm.write_bytes(b"P3\n1 1\n255\n0 0 0\n")
    with pytest.raises(ValueError, match="expected P6"):
        read_ppm(invalid_ppm)

    invalid_manifest = tmp_path / "manifest.jsonl"
    invalid_manifest.write_text(
        json.dumps({"episode_id": "escape", "frame_paths": ["../outside.ppm"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relative and stay within"):
        load_manifest(invalid_manifest)

    with pytest.raises(ValueError, match="relative and stay within"):
        invalid_manifest.write_text(
            json.dumps({"episode_id": "drive", "frame_paths": ["C:/outside.ppm"]})
            + "\n",
            encoding="utf-8",
        )
        load_manifest(invalid_manifest)
