"""Manifest loading and output serialization for the independent-frame baseline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TextIO

from .encoder import FEATURE_DEFINITION, encode_image, read_ppm

COST_PROXY_DEFINITION = (
    "One patch operation is one evaluation of the six-component RGB-statistics "
    "feature vector for one image patch. This count is a patch-operation proxy, "
    "not a measured FLOP count."
)


@dataclass(frozen=True, slots=True)
class ManifestFrame:
    """A validated manifest path and its resolved local image file."""

    relative_path: str
    path: Path


@dataclass(frozen=True, slots=True)
class ManifestEpisode:
    """Frame paths for one episode, preserving manifest order."""

    episode_id: str
    frames: tuple[ManifestFrame, ...]


def load_manifest(manifest_path: str | Path) -> tuple[ManifestEpisode, ...]:
    """Load episodes and validate that each frame path stays in the dataset.

    Manifest frame paths must be portable, relative POSIX paths. Resolved paths
    are also checked to prevent a symlink inside the dataset from escaping its
    root directory.

    Args:
        manifest_path: Existing synthetic generator JSONL manifest.

    Returns:
        Validated episodes in manifest order.

    Raises:
        ValueError: If the manifest is malformed or a frame path is invalid.
    """
    path = Path(manifest_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read manifest {path}: {error}") from error

    dataset_root = path.parent.resolve()
    episodes: list[ManifestEpisode] = []
    seen_episode_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON on manifest line {line_number}: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")

        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"manifest line {line_number} has an invalid episode_id")
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate episode_id {episode_id!r} in manifest")
        seen_episode_ids.add(episode_id)

        frame_paths = row.get("frame_paths")
        if not isinstance(frame_paths, list) or not frame_paths:
            raise ValueError(
                f"manifest episode {episode_id!r} must have non-empty frame_paths"
            )
        frames = tuple(
            _validate_frame_path(
                dataset_root, frame_path, episode_id=episode_id, frame_index=index
            )
            for index, frame_path in enumerate(frame_paths)
        )
        episodes.append(ManifestEpisode(episode_id=episode_id, frames=frames))

    if not episodes:
        raise ValueError(f"manifest {path} contains no episode records")
    return tuple(episodes)


def run_baseline(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    patch_size: int = 16,
) -> Path:
    """Independently encode every manifest frame and write results as JSONL.

    The measured per-frame time includes PPM file reading and feature
    calculation. It excludes JSON serialization and output writing. Feature
    values depend only on input pixels and ``patch_size``; processing times
    naturally vary between runs.

    Args:
        manifest_path: Synthetic generator manifest to load.
        output_path: JSONL path for one output record per frame.
        patch_size: Positive square patch width and height in pixels.

    Returns:
        The output JSONL path.

    Raises:
        ValueError: If ``patch_size`` or the input manifest/images are invalid.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")

    episodes = load_manifest(manifest_path)
    destination = Path(output_path)
    resolved_destination = destination.resolve()
    manifest_resolved = Path(manifest_path).resolve()
    if resolved_destination == manifest_resolved:
        raise ValueError("output_path must not overwrite the input manifest")
    frame_paths = {frame.path for episode in episodes for frame in episode.frames}
    if resolved_destination in frame_paths:
        raise ValueError("output_path must not overwrite a frame image")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8", newline="\n") as output_file:
        for episode in episodes:
            for frame_index, frame in enumerate(episode.frames):
                started = time.perf_counter()
                image = read_ppm(frame.path)
                encoded = encode_image(image, patch_size=patch_size)
                processing_time = time.perf_counter() - started
                patch_operations = encoded.processed_patches

                record = {
                    "episode_id": episode.episode_id,
                    "frame_index": frame_index,
                    "frame_path": frame.relative_path,
                    "patch_size": patch_size,
                    "patch_grid_dimensions": {
                        "rows": encoded.grid_rows,
                        "columns": encoded.grid_columns,
                    },
                    "feature_dimensions": encoded.feature_dimensions,
                    "feature_definition": FEATURE_DEFINITION,
                    "features": encoded.features,
                    "processing_time_seconds": processing_time,
                    "processed_patches": patch_operations,
                    "processing_cost_proxy": {
                        "name": "patch_operations",
                        "value": patch_operations,
                        "unit": "patches",
                        "definition": COST_PROXY_DEFINITION,
                    },
                }
                _write_jsonl_record(output_file, record)
    return destination


def _validate_frame_path(
    dataset_root: Path, raw_path: object, *, episode_id: str, frame_index: int
) -> ManifestFrame:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            f"episode {episode_id!r} frame {frame_index} has an invalid path"
        )
    if "\\" in raw_path:
        raise ValueError(
            f"episode {episode_id!r} frame {frame_index} path must use POSIX separators"
        )
    relative = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if (
        relative.is_absolute()
        or windows_path.drive
        or ".." in relative.parts
        or not relative.parts
    ):
        raise ValueError(
            f"episode {episode_id!r} frame {frame_index} path must be relative "
            "and stay within the dataset"
        )

    resolved = (dataset_root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(dataset_root):
        raise ValueError(
            f"episode {episode_id!r} frame {frame_index} path escapes the dataset"
        )
    if not resolved.is_file():
        raise ValueError(
            f"episode {episode_id!r} frame {frame_index} image does not exist: "
            f"{raw_path}"
        )
    return ManifestFrame(relative_path=relative.as_posix(), path=resolved)


def _write_jsonl_record(output_file: TextIO, record: dict[str, object]) -> None:
    output_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
    output_file.write("\n")
