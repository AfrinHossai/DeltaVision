"""Tests for deterministic synthetic RGB sequences and pixel masks."""

import json
from pathlib import Path, PurePosixPath

import pytest

from deltavision.synthetic import TRANSFORMATION_TYPES, generate_sequences


def _read_netpbm(path: Path, magic: bytes) -> tuple[int, int, bytes]:
    with path.open("rb") as image_file:
        assert image_file.readline().strip() == magic
        width, height = (int(value) for value in image_file.readline().split())
        assert image_file.readline().strip() == b"255"
        pixels = image_file.read()
    return width, height, pixels


def _read_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_mask_matches_frames(
    previous_frame_path: Path, current_frame_path: Path, mask_path: Path
) -> None:
    width, height, previous = _read_netpbm(previous_frame_path, b"P6")
    current_width, current_height, current = _read_netpbm(current_frame_path, b"P6")
    mask_width, mask_height, mask = _read_netpbm(mask_path, b"P5")
    assert (current_width, current_height) == (width, height)
    assert (mask_width, mask_height) == (width, height)
    expected = bytes(
        255
        if previous[index * 3 : index * 3 + 3] != current[index * 3 : index * 3 + 3]
        else 0
        for index in range(width * height)
    )
    assert mask == expected


def test_same_seed_produces_identical_files(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_sequences(
        first, width=64, height=48, sequence_length=5, num_episodes=3, seed=12
    )
    generate_sequences(
        second, width=64, height=48, sequence_length=5, num_episodes=3, seed=12
    )

    first_files = sorted(
        path.relative_to(first).as_posix()
        for path in first.rglob("*")
        if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second).as_posix()
        for path in second.rglob("*")
        if path.is_file()
    )
    assert first_files == second_files
    assert all(
        (first / path).read_bytes() == (second / path).read_bytes()
        for path in first_files
    )


def test_different_seeds_can_produce_different_scenes(tmp_path: Path) -> None:
    first = tmp_path / "seed_one"
    second = tmp_path / "seed_two"
    generate_sequences(first, width=64, height=64, sequence_length=1, seed=1)
    generate_sequences(second, width=64, height=64, sequence_length=1, seed=2)

    first_frame = first / "episode_0000" / "frame_0000.ppm"
    second_frame = second / "episode_0000" / "frame_0000.ppm"
    assert first_frame.read_bytes() != second_frame.read_bytes()


def test_unchanged_frames_have_empty_masks(tmp_path: Path) -> None:
    manifest_path = generate_sequences(
        tmp_path,
        width=48,
        height=48,
        sequence_length=4,
        seed=4,
        transformations=("unchanged",),
    )
    episode = _read_manifest(manifest_path)[0]
    frame_paths = episode["frame_paths"]
    mask_paths = episode["ground_truth_mask_paths"]
    assert isinstance(frame_paths, list)
    assert isinstance(mask_paths, list)

    for index, relative_mask in enumerate(mask_paths):
        mask_width, mask_height, mask = _read_netpbm(tmp_path / relative_mask, b"P5")
        assert (mask_width, mask_height) == (48, 48)
        assert not any(mask)
        _assert_mask_matches_frames(
            tmp_path / frame_paths[index],
            tmp_path / frame_paths[index + 1],
            tmp_path / relative_mask,
        )


@pytest.mark.parametrize("transformation", TRANSFORMATION_TYPES)
def test_transformation_masks_match_rendered_pixel_differences(
    tmp_path: Path, transformation: str
) -> None:
    manifest_path = generate_sequences(
        tmp_path,
        width=64,
        height=64,
        sequence_length=2,
        seed=21,
        transformations=(transformation,),  # type: ignore[arg-type]
    )
    episode = _read_manifest(manifest_path)[0]
    frame_paths = episode["frame_paths"]
    mask_paths = episode["ground_truth_mask_paths"]
    assert isinstance(frame_paths, list)
    assert isinstance(mask_paths, list)
    _assert_mask_matches_frames(
        tmp_path / frame_paths[0],
        tmp_path / frame_paths[1],
        tmp_path / mask_paths[0],
    )
    if transformation != "unchanged":
        _, _, mask = _read_netpbm(tmp_path / mask_paths[0], b"P5")
        assert any(mask)


def test_repeated_partial_occlusions_change_visible_pixels(tmp_path: Path) -> None:
    manifest_path = generate_sequences(
        tmp_path,
        width=64,
        height=64,
        sequence_length=8,
        seed=43,
        transformations=("partial_occlusion",),
    )
    episode = _read_manifest(manifest_path)[0]
    mask_paths = episode["ground_truth_mask_paths"]
    assert isinstance(mask_paths, list)
    for relative_mask in mask_paths:
        _, _, mask = _read_netpbm(tmp_path / relative_mask, b"P5")
        assert any(mask)


def test_image_and_mask_dimensions_and_jsonl_paths_are_portable(
    tmp_path: Path,
) -> None:
    width, height = 72, 40
    manifest_path = generate_sequences(
        tmp_path,
        width=width,
        height=height,
        sequence_length=3,
        num_episodes=2,
        seed=31,
    )
    episodes = _read_manifest(manifest_path)
    assert len(episodes) == 2

    for episode in episodes:
        assert set(episode) >= {
            "episode_id",
            "random_seed",
            "ground_truth_mask_definition",
            "transitions",
            "frame_paths",
            "ground_truth_mask_paths",
        }
        frame_paths = episode["frame_paths"]
        mask_paths = episode["ground_truth_mask_paths"]
        transitions = episode["transitions"]
        assert isinstance(frame_paths, list)
        assert isinstance(mask_paths, list)
        assert isinstance(transitions, list)
        assert len(frame_paths) == 3
        assert len(mask_paths) == 2
        assert len(transitions) == 2

        for relative_path in [*frame_paths, *mask_paths]:
            portable_path = PurePosixPath(relative_path)
            assert not portable_path.is_absolute()
            assert ".." not in portable_path.parts
            assert "\\" not in relative_path
            assert (tmp_path / Path(*portable_path.parts)).is_file()

        for relative_frame in frame_paths:
            frame_width, frame_height, _ = _read_netpbm(
                tmp_path / relative_frame, b"P6"
            )
            assert (frame_width, frame_height) == (width, height)
        for relative_mask in mask_paths:
            mask_width, mask_height, _ = _read_netpbm(tmp_path / relative_mask, b"P5")
            assert (mask_width, mask_height) == (width, height)


def test_generation_options_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 16"):
        generate_sequences(tmp_path / "small", width=8, height=32)
    with pytest.raises(ValueError, match="positive"):
        generate_sequences(tmp_path / "empty", sequence_length=0)
    with pytest.raises(ValueError, match="unsupported"):
        generate_sequences(tmp_path / "invalid", transformations=("warp",))  # type: ignore[arg-type]
