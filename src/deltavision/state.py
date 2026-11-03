"""Exact raw-patch feature reuse and JSONL output for a stateful baseline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .baseline import load_manifest
from .encoder import FEATURE_DIMENSIONS, RGBImage, encode_image, encode_patch, read_ppm

PATCH_OPERATION_DEFINITION = (
    "One patch operation is one actual six-component RGB-statistics feature "
    "computation for a patch. Reused patches and raw-pixel comparisons are "
    "excluded. This proxy is not a neural FLOP count."
)


@dataclass(frozen=True, slots=True)
class StatefulEncodingResult:
    """Features and per-frame cache accounting for one image."""

    grid_rows: int
    grid_columns: int
    feature_dimensions: int
    features: tuple[tuple[float, ...], ...]
    patch_count: int
    recomputed_patches: int
    reused_patches: int
    refresh_fraction: float
    refresh_decisions: tuple[bool, ...]
    processing_latency_seconds: float
    comparison_overhead_seconds: float
    feature_compute_seconds: float
    cache_update_seconds: float


class StatefulPatchCache:
    """Reuse exact features for patches whose raw RGB bytes have not changed.

    A cache instance belongs to one episode. Create a new instance or call
    :meth:`reset` before processing another episode. The cache stores the last
    raw bytes and six-value feature vector for each row-major patch.
    """

    def __init__(self, patch_size: int = 16) -> None:
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self._patch_size = patch_size
        self._image_shape: tuple[int, int] | None = None
        self._raw_patches: tuple[bytes, ...] | None = None
        self._features: tuple[tuple[float, ...], ...] | None = None

    @property
    def patch_size(self) -> int:
        """Configured square patch width and height in pixels."""
        return self._patch_size

    def reset(self) -> None:
        """Clear cached pixels and features, starting a new episode."""
        self._image_shape = None
        self._raw_patches = None
        self._features = None

    def process_frame(self, image: RGBImage) -> StatefulEncodingResult:
        """Encode one frame, reusing only byte-identical previous patches.

        The first frame after construction or reset is fully encoded. If image
        dimensions change within an episode, the cache is reset and that frame
        is also fully encoded because patch positions and boundaries changed.

        Args:
            image: Validated RGB image for the current episode and frame.

        Returns:
            Features in row-major patch order with recomputation and timing
            accounting.
        """
        processing_started = time.perf_counter()
        image_shape = (image.width, image.height)
        if self._image_shape is not None and self._image_shape != image_shape:
            self.reset()

        grid_columns = (image.width + self._patch_size - 1) // self._patch_size
        grid_rows = (image.height + self._patch_size - 1) // self._patch_size
        patch_count = grid_rows * grid_columns

        if self._features is None or self._raw_patches is None:
            feature_started = time.perf_counter()
            encoded = encode_image(image, patch_size=self._patch_size)
            feature_compute_seconds = time.perf_counter() - feature_started

            cache_started = time.perf_counter()
            raw_patches = _extract_raw_patches(
                image, self._patch_size, grid_rows, grid_columns
            )
            self._image_shape = image_shape
            self._raw_patches = raw_patches
            self._features = encoded.features
            cache_update_seconds = time.perf_counter() - cache_started
            processing_latency_seconds = time.perf_counter() - processing_started
            return StatefulEncodingResult(
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                feature_dimensions=FEATURE_DIMENSIONS,
                features=encoded.features,
                patch_count=patch_count,
                recomputed_patches=patch_count,
                reused_patches=0,
                refresh_fraction=1.0,
                refresh_decisions=tuple(True for _ in range(patch_count)),
                processing_latency_seconds=processing_latency_seconds,
                comparison_overhead_seconds=0.0,
                feature_compute_seconds=feature_compute_seconds,
                cache_update_seconds=cache_update_seconds,
            )

        comparison_started = time.perf_counter()
        raw_patches = _extract_raw_patches(
            image, self._patch_size, grid_rows, grid_columns
        )
        refresh_decisions = tuple(
            current != previous
            for current, previous in zip(raw_patches, self._raw_patches, strict=True)
        )
        changed_indices = [
            index
            for index, should_refresh in enumerate(refresh_decisions)
            if should_refresh
        ]
        comparison_overhead_seconds = time.perf_counter() - comparison_started

        next_features = list(self._features)
        if changed_indices:
            feature_started = time.perf_counter()
            for index in changed_indices:
                patch_row, patch_column = divmod(index, grid_columns)
                next_features[index] = encode_patch(
                    image, patch_row, patch_column, self._patch_size
                )
            feature_compute_seconds = time.perf_counter() - feature_started
        else:
            feature_compute_seconds = 0.0

        cache_started = time.perf_counter()
        self._image_shape = image_shape
        self._raw_patches = raw_patches
        self._features = tuple(next_features)
        cache_update_seconds = time.perf_counter() - cache_started
        recomputed_patches = len(changed_indices)
        reused_patches = patch_count - recomputed_patches
        processing_latency_seconds = time.perf_counter() - processing_started
        return StatefulEncodingResult(
            grid_rows=grid_rows,
            grid_columns=grid_columns,
            feature_dimensions=FEATURE_DIMENSIONS,
            features=self._features,
            patch_count=patch_count,
            recomputed_patches=recomputed_patches,
            reused_patches=reused_patches,
            refresh_fraction=recomputed_patches / patch_count,
            refresh_decisions=refresh_decisions,
            processing_latency_seconds=processing_latency_seconds,
            comparison_overhead_seconds=comparison_overhead_seconds,
            feature_compute_seconds=feature_compute_seconds,
            cache_update_seconds=cache_update_seconds,
        )


def run_stateful(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    patch_size: int = 16,
) -> Path:
    """Process manifest episodes with isolated caches and write frame JSONL.

    Processing latency includes reading the PPM, exact raw-patch comparison,
    feature recomputation, and cache update. Output serialization is excluded.
    Each episode receives a new :class:`StatefulPatchCache` instance.

    Args:
        manifest_path: Synthetic generator JSONL manifest.
        output_path: Destination JSONL path, one record per frame.
        patch_size: Positive square patch width and height in pixels.

    Returns:
        The output JSONL path.

    Raises:
        ValueError: If patch size, manifest, or input frame data are invalid.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    episodes = load_manifest(manifest_path)
    destination = Path(output_path)
    resolved_destination = destination.resolve()
    if resolved_destination == Path(manifest_path).resolve():
        raise ValueError("output_path must not overwrite the input manifest")
    frame_paths = {frame.path for episode in episodes for frame in episode.frames}
    if resolved_destination in frame_paths:
        raise ValueError("output_path must not overwrite a frame image")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8", newline="\n") as output_file:
        for episode in episodes:
            cache = StatefulPatchCache(patch_size=patch_size)
            for frame_index, frame in enumerate(episode.frames):
                processing_started = time.perf_counter()
                image = read_ppm(frame.path)
                encoded = cache.process_frame(image)
                processing_latency_seconds = time.perf_counter() - processing_started
                record = {
                    "episode_id": episode.episode_id,
                    "frame_index": frame_index,
                    "frame_path": frame.relative_path,
                    "patch_grid_dimensions": {
                        "rows": encoded.grid_rows,
                        "columns": encoded.grid_columns,
                    },
                    "patch_count": encoded.patch_count,
                    "recomputed_patches": encoded.recomputed_patches,
                    "reused_patches": encoded.reused_patches,
                    "refresh_fraction": encoded.refresh_fraction,
                    "processing_latency_seconds": processing_latency_seconds,
                    "comparison_overhead_seconds": encoded.comparison_overhead_seconds,
                    "feature_compute_seconds": encoded.feature_compute_seconds,
                    "cache_update_seconds": encoded.cache_update_seconds,
                    "feature_dimensions": encoded.feature_dimensions,
                    "features": encoded.features,
                    "processing_cost_proxy": {
                        "name": "recomputed_patch_feature_operations",
                        "value": encoded.recomputed_patches,
                        "unit": "patch_operations",
                        "definition": PATCH_OPERATION_DEFINITION,
                    },
                }
                _write_jsonl_record(output_file, record)
    return destination


def _extract_raw_patches(
    image: RGBImage, patch_size: int, grid_rows: int, grid_columns: int
) -> tuple[bytes, ...]:
    """Copy valid RGB bytes for each patch, excluding any conceptual padding."""
    patches: list[bytes] = []
    for patch_row in range(grid_rows):
        y_start = patch_row * patch_size
        y_stop = min(y_start + patch_size, image.height)
        for patch_column in range(grid_columns):
            x_start = patch_column * patch_size
            x_stop = min(x_start + patch_size, image.width)
            patch = bytearray()
            for y in range(y_start, y_stop):
                row_start = (y * image.width + x_start) * 3
                row_stop = (y * image.width + x_stop) * 3
                patch.extend(image.pixels[row_start:row_stop])
            patches.append(bytes(patch))
    return tuple(patches)


def _write_jsonl_record(output_file: TextIO, record: dict[str, object]) -> None:
    output_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
    output_file.write("\n")
