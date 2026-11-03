"""Threshold-based approximate reuse using normalized raw RGB patch differences."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

from .baseline import load_manifest
from .encoder import (
    FEATURE_DEFINITION,
    FEATURE_DIMENSIONS,
    RGBImage,
    encode_image,
    encode_patch,
    read_ppm,
)
from .plotting import write_threshold_tradeoff_png

FeatureVectors = tuple[tuple[float, ...], ...]
PreparedFrame = tuple[RGBImage, FeatureVectors]
SUPPORTED_THRESHOLDS: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.10)
EXPERIMENT_LABEL = "Synthetic-benchmark findings"
DIFFERENCE_DEFINITION = (
    "Mean absolute RGB difference over all channel bytes in a patch, divided "
    "by 255; scores are in [0, 1]. Reuse if score <= threshold; recompute if "
    "score > threshold."
)
PATCH_OPERATION_DEFINITION = (
    "One patch operation is one actual six-component RGB-statistics feature "
    "computation for a patch. Reused patches and difference-score calculations "
    "are excluded. Independent reference encodings used only for error metrics "
    "are also excluded. This proxy is not a neural FLOP count."
)


@dataclass(frozen=True, slots=True)
class ThresholdEncodingResult:
    """Approximate features and computation accounting for one frame."""

    grid_rows: int
    grid_columns: int
    feature_dimensions: int
    features: tuple[tuple[float, ...], ...]
    patch_count: int
    recomputed_patches: int
    reused_patches: int
    refresh_fraction: float
    mean_difference_scores: tuple[float, ...]
    refresh_decisions: tuple[bool, ...]
    processing_latency_seconds: float
    comparison_overhead_seconds: float
    feature_compute_seconds: float
    cache_update_seconds: float


@dataclass(frozen=True, slots=True)
class FeatureErrorMetrics:
    """Elementwise error and per-vector cosine similarity against a reference."""

    mean_absolute_error: float
    maximum_absolute_error: float
    mean_cosine_similarity: float
    feature_value_count: int
    feature_vector_count: int


class ThresholdPatchCache:
    """Reuse features when per-patch normalized RGB change is small enough.

    Raw-patch observations are replaced after every frame, including patches
    whose feature vectors are reused. Thus repeated small changes can accrue
    staleness: the gate always compares with the immediately previous raw
    frame, while the cached feature may describe an earlier frame.
    """

    def __init__(self, threshold: float, patch_size: int = 16) -> None:
        _validate_threshold(threshold)
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.threshold = float(threshold)
        self._patch_size = patch_size
        self._image_shape: tuple[int, int] | None = None
        self._raw_patches: tuple[bytes, ...] | None = None
        self._features: tuple[tuple[float, ...], ...] | None = None

    @property
    def patch_size(self) -> int:
        """Configured square patch width and height in pixels."""
        return self._patch_size

    def reset(self) -> None:
        """Clear raw observations and cached features for a new episode."""
        self._image_shape = None
        self._raw_patches = None
        self._features = None

    def process_frame(self, image: RGBImage) -> ThresholdEncodingResult:
        """Encode a frame, reusing cached features under the threshold rule.

        The first frame after construction or reset is fully encoded. For each
        later patch, calculate mean absolute RGB difference against the raw
        patch from the previous frame, normalized by 255. Reuse when the score
        is less than or equal to ``threshold``; recompute only when greater.
        The current raw patch bytes replace the previous observation every
        frame, even when its feature vector is reused.
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
            return ThresholdEncodingResult(
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                feature_dimensions=FEATURE_DIMENSIONS,
                features=encoded.features,
                patch_count=patch_count,
                recomputed_patches=patch_count,
                reused_patches=0,
                refresh_fraction=1.0,
                mean_difference_scores=tuple(0.0 for _ in range(patch_count)),
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
        difference_scores = tuple(
            mean_absolute_rgb_difference(current, previous)
            for current, previous in zip(raw_patches, self._raw_patches, strict=True)
        )
        refresh_decisions = tuple(score > self.threshold for score in difference_scores)
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
        return ThresholdEncodingResult(
            grid_rows=grid_rows,
            grid_columns=grid_columns,
            feature_dimensions=FEATURE_DIMENSIONS,
            features=self._features,
            patch_count=patch_count,
            recomputed_patches=recomputed_patches,
            reused_patches=reused_patches,
            refresh_fraction=recomputed_patches / patch_count,
            mean_difference_scores=difference_scores,
            refresh_decisions=refresh_decisions,
            processing_latency_seconds=processing_latency_seconds,
            comparison_overhead_seconds=comparison_overhead_seconds,
            feature_compute_seconds=feature_compute_seconds,
            cache_update_seconds=cache_update_seconds,
        )


def mean_absolute_rgb_difference(current: bytes, previous: bytes) -> float:
    """Return mean absolute RGB-byte difference normalized to ``[0, 1]``.

    Both inputs must describe the same non-empty number of RGB pixels. The
    score averages absolute differences over every channel value and divides
    by 255.
    """
    if not current or len(current) != len(previous) or len(current) % 3:
        raise ValueError("patch byte arrays must have equal non-zero RGB lengths")
    absolute_difference_sum = sum(
        abs(current_value - previous_value)
        for current_value, previous_value in zip(current, previous, strict=True)
    )
    return absolute_difference_sum / (len(current) * 255)


def compare_feature_vectors(
    actual: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]
) -> FeatureErrorMetrics:
    """Measure elementwise absolute error and cosine similarity per patch.

    A pair of zero vectors has cosine similarity 1.0; if only one vector has
    zero norm, its similarity to the non-zero vector is 0.0.
    """
    if not actual or len(actual) != len(expected):
        raise ValueError("feature batches must have equal non-zero patch counts")

    absolute_error_sum = 0.0
    maximum_absolute_error = 0.0
    feature_value_count = 0
    cosine_sum = 0.0
    for actual_vector, expected_vector in zip(actual, expected, strict=True):
        if not actual_vector or len(actual_vector) != len(expected_vector):
            raise ValueError("feature vectors must have equal non-zero dimensions")
        for actual_value, expected_value in zip(
            actual_vector, expected_vector, strict=True
        ):
            absolute_error = abs(actual_value - expected_value)
            absolute_error_sum += absolute_error
            maximum_absolute_error = max(maximum_absolute_error, absolute_error)
            feature_value_count += 1
        cosine_sum += _cosine_similarity(actual_vector, expected_vector)

    return FeatureErrorMetrics(
        mean_absolute_error=absolute_error_sum / feature_value_count,
        maximum_absolute_error=maximum_absolute_error,
        mean_cosine_similarity=cosine_sum / len(actual),
        feature_value_count=feature_value_count,
        feature_vector_count=len(actual),
    )


def run_threshold_sweep(
    manifest_path: str | Path,
    output_path: str | Path = "results/threshold_sweep.json",
    figure_path: str | Path = "docs/figures/threshold_tradeoff.png",
    *,
    patch_size: int = 16,
    thresholds: Sequence[float] = SUPPORTED_THRESHOLDS,
) -> dict[str, object]:
    """Evaluate supported thresholds on the same episodes and write artifacts.

    Input frames and independent baseline features are prepared once before
    sweeping. Per-threshold wall-clock latency sums only each stateful
    ``process_frame`` call; baseline encoding, feature-error calculation,
    manifest loading, image loading, and JSON/PNG output are excluded.

    Args:
        manifest_path: Synthetic generator manifest to evaluate.
        output_path: JSON path for the sweep summary.
        figure_path: PNG path for the refresh/error trade-off plot.
        patch_size: Square patch size, shared with the frame-independent
            encoder.
        thresholds: Unique supported threshold values to evaluate.

    Returns:
        The JSON-serializable sweep result dictionary.

    Raises:
        ValueError: If inputs or threshold configuration are invalid.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    selected_thresholds = tuple(float(value) for value in thresholds)
    if not selected_thresholds or len(set(selected_thresholds)) != len(
        selected_thresholds
    ):
        raise ValueError("thresholds must be a non-empty sequence without duplicates")
    for threshold in selected_thresholds:
        _validate_threshold(threshold)

    episodes = load_manifest(manifest_path)
    destination = Path(output_path)
    figure_destination = Path(figure_path)
    input_path = Path(manifest_path).resolve()
    resolved_destination = destination.resolve()
    resolved_figure = figure_destination.resolve()
    if resolved_destination == input_path or resolved_figure == input_path:
        raise ValueError("output paths must not overwrite the input manifest")
    source_frames = {frame.path for episode in episodes for frame in episode.frames}
    if resolved_destination in source_frames or resolved_figure in source_frames:
        raise ValueError("output paths must not overwrite input frame images")
    if resolved_destination == resolved_figure:
        raise ValueError("JSON output and figure paths must be different")

    prepared_episodes: list[tuple[str, tuple[PreparedFrame, ...]]] = []
    for episode in episodes:
        prepared_frames = []
        for frame in episode.frames:
            image = read_ppm(frame.path)
            baseline = encode_image(image, patch_size=patch_size)
            prepared_frames.append((image, baseline.features))
        prepared_episodes.append((episode.episode_id, tuple(prepared_frames)))

    result_rows: list[dict[str, object]] = []
    for threshold in selected_thresholds:
        total_patch_count = 0
        recomputed_patches = 0
        reused_patches = 0
        wall_clock_latency_seconds = 0.0
        comparison_overhead_seconds = 0.0
        feature_compute_seconds = 0.0
        cache_update_seconds = 0.0
        error_accumulator = _FeatureErrorAccumulator()
        frame_count = 0

        for _, frames in prepared_episodes:
            cache = ThresholdPatchCache(threshold, patch_size=patch_size)
            for image, baseline_features in frames:
                started = time.perf_counter()
                encoded = cache.process_frame(image)
                wall_clock_latency_seconds += time.perf_counter() - started
                frame_count += 1
                total_patch_count += encoded.patch_count
                recomputed_patches += encoded.recomputed_patches
                reused_patches += encoded.reused_patches
                comparison_overhead_seconds += encoded.comparison_overhead_seconds
                feature_compute_seconds += encoded.feature_compute_seconds
                cache_update_seconds += encoded.cache_update_seconds
                error_accumulator.add(encoded.features, baseline_features)

        refresh_fraction = recomputed_patches / total_patch_count
        result_rows.append(
            {
                "threshold": threshold,
                "total_patch_count": total_patch_count,
                "recomputed_patches": recomputed_patches,
                "reused_patches": reused_patches,
                "refresh_fraction": refresh_fraction,
                "mean_absolute_feature_error": error_accumulator.mean_absolute_error,
                "maximum_absolute_feature_error": (
                    error_accumulator.maximum_absolute_error
                ),
                "mean_feature_vector_cosine_similarity": (
                    error_accumulator.mean_cosine_similarity
                ),
                "wall_clock_latency_seconds": wall_clock_latency_seconds,
                "mean_frame_latency_seconds": wall_clock_latency_seconds / frame_count,
                "comparison_overhead_seconds": comparison_overhead_seconds,
                "feature_compute_seconds": feature_compute_seconds,
                "cache_update_seconds": cache_update_seconds,
                "patch_computation_proxy": {
                    "name": "recomputed_patch_feature_operations",
                    "value": recomputed_patches,
                    "unit": "patch_operations",
                    "definition": PATCH_OPERATION_DEFINITION,
                },
            }
        )

    summary: dict[str, object] = {
        "experiment_label": EXPERIMENT_LABEL,
        "method": "Threshold-based approximate raw RGB patch reuse",
        "manifest_path": Path(manifest_path).as_posix(),
        "patch_size": patch_size,
        "episode_count": len(prepared_episodes),
        "frame_count": sum(len(frames) for _, frames in prepared_episodes),
        "image_dimensions": [
            {"width": width, "height": height}
            for width, height in sorted(
                {
                    (image.width, image.height)
                    for _, frames in prepared_episodes
                    for image, _ in frames
                }
            )
        ],
        "threshold_definition": DIFFERENCE_DEFINITION,
        "feature_definition": FEATURE_DEFINITION,
        "feature_dimensions": FEATURE_DIMENSIONS,
        "refresh_fraction_definition": "recomputed_patches / total_patch_count",
        "feature_error_aggregation_definition": (
            "Mean absolute error averages scalar feature errors across every "
            "patch and frame; maximum absolute error is the largest scalar "
            "feature error."
        ),
        "cosine_aggregation_definition": (
            "Mean cosine similarity averages one cosine score per patch feature "
            "vector across all frames. Two zero vectors have similarity 1; one "
            "zero and one non-zero vector have similarity 0."
        ),
        "latency_definition": (
            "Sum of in-memory stateful process_frame wall-clock times across "
            "all frames; baseline computation, metric calculation, I/O, and "
            "output serialization are excluded."
        ),
        "comparison_overhead_definition": (
            "Includes extracting current patch bytes and calculating their "
            "normalized mean absolute RGB difference scores. It is reported "
            "separately and excluded from the patch-computation proxy."
        ),
        "patch_computation_proxy_definition": (
            "Counts only feature recomputations performed by the threshold "
            "policy. Independent reference encodings used for error metrics "
            "are excluded."
        ),
        "results": result_rows,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure_destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    write_threshold_tradeoff_png(result_rows, figure_destination)
    return summary


class _FeatureErrorAccumulator:
    """Accumulate global feature errors without retaining every vector."""

    def __init__(self) -> None:
        self.absolute_error_sum = 0.0
        self.maximum_absolute_error = 0.0
        self.feature_value_count = 0
        self.cosine_sum = 0.0
        self.feature_vector_count = 0

    def add(
        self, actual: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]
    ) -> None:
        metrics = compare_feature_vectors(actual, expected)
        self.absolute_error_sum += (
            metrics.mean_absolute_error * metrics.feature_value_count
        )
        self.maximum_absolute_error = max(
            self.maximum_absolute_error, metrics.maximum_absolute_error
        )
        self.feature_value_count += metrics.feature_value_count
        self.cosine_sum += metrics.mean_cosine_similarity * metrics.feature_vector_count
        self.feature_vector_count += metrics.feature_vector_count

    @property
    def mean_absolute_error(self) -> float:
        return self.absolute_error_sum / self.feature_value_count

    @property
    def mean_cosine_similarity(self) -> float:
        return self.cosine_sum / self.feature_vector_count


def _validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value between 0 and 1")


def _extract_raw_patches(
    image: RGBImage, patch_size: int, grid_rows: int, grid_columns: int
) -> tuple[bytes, ...]:
    """Copy actual RGB bytes in row-major patch order without edge padding."""
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


def _cosine_similarity(actual: Sequence[float], expected: Sequence[float]) -> float:
    if all(a == b for a, b in zip(actual, expected, strict=True)):
        return 1.0
    dot_product = sum(a * b for a, b in zip(actual, expected, strict=True))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    expected_norm = math.sqrt(sum(value * value for value in expected))
    if actual_norm == 0.0 and expected_norm == 0.0:
        return 1.0
    if actual_norm == 0.0 or expected_norm == 0.0:
        return 0.0
    similarity = dot_product / (actual_norm * expected_norm)
    return min(1.0, max(-1.0, similarity))


def _write_jsonl_record(output_file: TextIO, record: dict[str, object]) -> None:
    output_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
    output_file.write("\n")
