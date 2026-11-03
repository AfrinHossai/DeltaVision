"""Validation-only comparison of local reuse and a global-change fallback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .approximate import (
    ThresholdEncodingResult,
    ThresholdPatchCache,
    compare_feature_vectors,
)
from .baseline import ManifestEpisode, load_manifest
from .encoder import FEATURE_DEFINITION, RGBImage, encode_image, read_ppm
from .evaluation import (
    ConfusionCounts,
    count_patch_decisions,
    detection_metrics,
    ground_truth_mask_to_patch_labels,
)
from .global_fallback import GlobalFallbackPatchCache, GlobalFallbackResult
from .plotting import write_global_change_comparison_png
from .state import StatefulEncodingResult, StatefulPatchCache
from .synthetic import TRANSFORMATION_TYPES

LOCAL_THRESHOLDS = (0.01, 0.03, 0.05)
GLOBAL_LOCAL_THRESHOLD = 0.05
EXPERIMENT_LABEL = "Validation-only synthetic-benchmark findings"


@dataclass(slots=True)
class _Stats:
    counts: ConfusionCounts = field(default_factory=lambda: ConfusionCounts(0, 0, 0, 0))
    transition_patch_count: int = 0
    transition_recomputed_patches: int = 0
    total_patch_count: int = 0
    recomputed_patches: int = 0
    reused_patches: int = 0
    full_frame_refresh_count: int = 0
    global_full_refresh_count: int = 0
    transition_count: int = 0
    feature_error_sum: float = 0.0
    feature_value_count: int = 0
    maximum_feature_error: float = 0.0
    global_detector_runtime_seconds: float = 0.0
    global_detector_comparison_count: int = 0
    processing_runtime_seconds: float = 0.0
    full_refresh_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class _EpisodeStats:
    aggregate: _Stats = field(default_factory=_Stats)
    by_transformation: dict[str, _Stats] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _EpisodeData:
    episode_id: str
    frames: tuple[RGBImage, ...]
    baseline_features: tuple[tuple[tuple[float, ...], ...], ...]
    transition_labels: tuple[tuple[bool, ...], ...]
    transformation_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FrameResult:
    features: tuple[tuple[float, ...], ...]
    refresh_decisions: tuple[bool, ...]
    patch_count: int
    recomputed_patches: int
    reused_patches: int
    processing_latency_seconds: float
    global_change_score: float | None
    global_change_detected: bool
    full_refresh_reason: str | None
    global_detector_runtime_seconds: float
    global_detector_was_run: bool


def evaluate_global_change_validation(
    manifest_path: str | Path,
    output_path: str | Path = "results/global_change_validation.json",
    figure_path: str | Path = "docs/figures/global_change_comparison.png",
    *,
    patch_size: int = 16,
    global_change_threshold: float = 0.50,
) -> dict[str, object]:
    """Compare exact/local policies with global fallback on validation only.

    The input must contain split-prefixed ``validation_`` episode IDs. Only
    files referenced by that manifest are read. No benchmark root discovery
    or held-out split loading is performed by this function.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if (
        isinstance(global_change_threshold, bool)
        or not isinstance(global_change_threshold, (int, float))
        or not math.isfinite(global_change_threshold)
        or not 0.0 <= global_change_threshold <= 1.0
    ):
        raise ValueError("global_change_threshold must be between 0 and 1")

    manifest = Path(manifest_path)
    episodes = load_manifest(manifest)
    rows = _read_manifest_rows(manifest)
    if len(episodes) != len(rows):
        raise ValueError("manifest episode records changed while loading")
    for episode in episodes:
        if not episode.episode_id.startswith("validation_"):
            raise ValueError(
                "global fallback evaluation accepts validation-prefixed episodes only"
            )

    dataset_root = manifest.parent.resolve()
    prepared = tuple(
        _prepare_episode(dataset_root, episode, row, patch_size)
        for episode, row in zip(episodes, rows, strict=True)
    )
    method_specs = (
        ("exact_reuse", "exact", None),
        *(
            (f"local_threshold_{threshold:.2f}", "local", threshold)
            for threshold in LOCAL_THRESHOLDS
        ),
        (
            "local_threshold_0.05_plus_global_fallback",
            "global_fallback",
            global_change_threshold,
        ),
    )
    method_stats: dict[str, list[_EpisodeStats]] = {
        method_name: [] for method_name, _kind, _value in method_specs
    }

    for episode in prepared:
        for method_name, method_kind, method_value in method_specs:
            method_stats[method_name].append(
                _evaluate_episode(
                    episode,
                    method_kind,
                    method_value,
                    patch_size=patch_size,
                )
            )

    result_rows: list[dict[str, object]] = []
    for method_name, method_kind, method_value in method_specs:
        per_episode = method_stats[method_name]
        aggregate = _sum_stats([item.aggregate for item in per_episode])
        method_result: dict[str, object] = {
            "method": method_name,
            "configuration": _method_configuration(
                method_kind, method_value, patch_size
            ),
            "aggregate": _format_stats(aggregate, include_all_frame=True),
            "per_transformation": {},
        }
        transformation_results: dict[str, object] = {}
        for transformation in TRANSFORMATION_TYPES:
            transformation_stats = _sum_stats(
                [
                    item.by_transformation.get(transformation, _Stats())
                    for item in per_episode
                ]
            )
            transformation_results[transformation] = _format_stats(
                transformation_stats, include_all_frame=False
            )
        method_result["per_transformation"] = transformation_results
        result_rows.append(method_result)

    summary: dict[str, object] = {
        "experiment_label": EXPERIMENT_LABEL,
        "partition": "validation",
        "manifest_path": manifest.as_posix(),
        "episode_count": len(prepared),
        "frame_count": sum(len(episode.frames) for episode in prepared),
        "transition_count": sum(len(episode.transition_labels) for episode in prepared),
        "patch_size": patch_size,
        "methods": result_rows,
        "global_detector_definition": {
            "statistic": (
                "Fraction of pixel locations whose RGB triplet differs from the "
                "immediately preceding frame, in [0, 1]."
            ),
            "trigger_rule": (
                "Trigger a full refresh when changed_pixel_fraction exceeds "
                "global_change_threshold."
            ),
            "global_change_threshold": global_change_threshold,
            "scope": "Validation partition only; no held-out split is loaded.",
        },
        "metric_definitions": {
            "ground_truth": (
                "A patch is changed when its exact rendered RGB difference mask "
                "contains at least one non-zero pixel; this is not semantic "
                "ground truth."
            ),
            "detection_scope": (
                "Transition frames only; the first frame of each episode fully "
                "initializes its cache and is excluded from classification."
            ),
            "refresh_fraction": (
                "Transition recomputations divided by transition patch count."
            ),
            "full_frame_refresh_frequency": (
                "Transition frames refreshing every patch divided by transition "
                "count; initial episode frames are excluded."
            ),
            "feature_mean_absolute_error": (
                "Mean absolute difference from independent RGB-statistics features; "
                "aggregate includes all frames, per-transformation uses "
                "destination frames."
            ),
            "zero_denominator": "Undefined rates are returned as 0.",
            "runtime": (
                "Global detector runtime measures only the changed-pixel-fraction "
                "comparison, separately from cache processing. It is validation CPU "
                "timing and not a speedup claim."
            ),
        },
        "feature_definition": FEATURE_DEFINITION,
        "threshold_selection": {
            "held_out_test_accessed": False,
            "preferred_threshold_selected": False,
        },
    }

    _write_validation_result(summary, output_path, figure_path)
    return summary


def _prepare_episode(
    dataset_root: Path,
    episode: ManifestEpisode,
    row: dict[str, Any],
    patch_size: int,
) -> _EpisodeData:
    if row.get("episode_id") != episode.episode_id:
        raise ValueError("manifest episode ordering is inconsistent")
    frames = tuple(read_ppm(frame.path) for frame in episode.frames)
    if any(
        (frame.width, frame.height) != (frames[0].width, frames[0].height)
        for frame in frames
    ):
        raise ValueError(f"frame dimensions differ in episode {episode.episode_id!r}")
    baseline_features = tuple(
        encode_image(frame, patch_size=patch_size).features for frame in frames
    )
    mask_paths = row.get("ground_truth_mask_paths")
    transitions = row.get("transitions")
    if (
        not isinstance(mask_paths, list)
        or not isinstance(transitions, list)
        or len(mask_paths) != len(frames) - 1
        or len(transitions) != len(frames) - 1
    ):
        raise ValueError(f"episode {episode.episode_id!r} has invalid transition data")

    labels: list[tuple[bool, ...]] = []
    transformations: list[str] = []
    for index, (raw_mask_path, transition) in enumerate(
        zip(mask_paths, transitions, strict=True)
    ):
        if not isinstance(transition, dict):
            raise ValueError("manifest transition entries must be objects")
        if transition.get("transformation_type") not in TRANSFORMATION_TYPES:
            raise ValueError("manifest has an unsupported transformation type")
        mask_path = _resolve_mask_path(dataset_root, raw_mask_path)
        mask_width, mask_height, mask_pixels = _read_pgm_mask(mask_path)
        if (mask_width, mask_height) != (frames[0].width, frames[0].height):
            raise ValueError(
                f"mask dimensions differ in episode {episode.episode_id!r}"
            )
        labels.append(
            ground_truth_mask_to_patch_labels(
                mask_pixels, mask_width, mask_height, patch_size
            )
        )
        transformations.append(transition["transformation_type"])
        if (
            transition.get("from_frame") != index
            or transition.get("to_frame") != index + 1
        ):
            raise ValueError("transition frame indices are not consecutive")
    return _EpisodeData(
        episode_id=episode.episode_id,
        frames=frames,
        baseline_features=baseline_features,
        transition_labels=tuple(labels),
        transformation_types=tuple(transformations),
    )


def _evaluate_episode(
    episode: _EpisodeData,
    method_kind: str,
    method_value: float | None,
    *,
    patch_size: int,
) -> _EpisodeStats:
    if method_kind == "exact":
        cache: StatefulPatchCache | ThresholdPatchCache | GlobalFallbackPatchCache = (
            StatefulPatchCache(patch_size=patch_size)
        )
    elif method_kind == "local":
        if method_value is None:
            raise ValueError("local threshold method is missing its threshold")
        cache = ThresholdPatchCache(method_value, patch_size=patch_size)
    elif method_kind == "global_fallback":
        if method_value is None:
            raise ValueError("global fallback method is missing its threshold")
        cache = GlobalFallbackPatchCache(
            local_threshold=GLOBAL_LOCAL_THRESHOLD,
            global_change_threshold=method_value,
            patch_size=patch_size,
        )
    else:
        raise ValueError(f"unknown method kind: {method_kind}")

    episode_stats = _EpisodeStats()
    for frame_index, (frame, baseline) in enumerate(
        zip(episode.frames, episode.baseline_features, strict=True)
    ):
        frame_result = _process_frame(cache, frame, method_kind, frame_index)
        feature_error = compare_feature_vectors(frame_result.features, baseline)
        aggregate = episode_stats.aggregate
        aggregate.total_patch_count += frame_result.patch_count
        aggregate.recomputed_patches += frame_result.recomputed_patches
        aggregate.reused_patches += frame_result.reused_patches
        aggregate.feature_error_sum += (
            feature_error.mean_absolute_error * feature_error.feature_value_count
        )
        aggregate.feature_value_count += feature_error.feature_value_count
        aggregate.maximum_feature_error = max(
            aggregate.maximum_feature_error, feature_error.maximum_absolute_error
        )
        aggregate.processing_runtime_seconds += frame_result.processing_latency_seconds
        aggregate.global_detector_runtime_seconds += (
            frame_result.global_detector_runtime_seconds
        )
        aggregate.global_detector_comparison_count += int(
            frame_result.global_detector_was_run
        )
        if frame_result.full_refresh_reason is not None:
            aggregate.full_refresh_reason_counts[frame_result.full_refresh_reason] = (
                aggregate.full_refresh_reason_counts.get(
                    frame_result.full_refresh_reason, 0
                )
                + 1
            )
        if frame_index == 0:
            continue

        transition_index = frame_index - 1
        transformation = episode.transformation_types[transition_index]
        transition_counts = count_patch_decisions(
            episode.transition_labels[transition_index], frame_result.refresh_decisions
        )
        _add_transition(aggregate, transition_counts, frame_result)
        transformed = episode_stats.by_transformation.setdefault(
            transformation, _Stats()
        )
        _add_transition(transformed, transition_counts, frame_result)
        transformed.feature_error_sum += (
            feature_error.mean_absolute_error * feature_error.feature_value_count
        )
        transformed.feature_value_count += feature_error.feature_value_count
        transformed.maximum_feature_error = max(
            transformed.maximum_feature_error, feature_error.maximum_absolute_error
        )
        transformed.global_detector_runtime_seconds += (
            frame_result.global_detector_runtime_seconds
        )
        transformed.global_detector_comparison_count += int(
            frame_result.global_detector_was_run
        )
        transformed.processing_runtime_seconds += (
            frame_result.processing_latency_seconds
        )
        if frame_result.full_refresh_reason is not None:
            transformed.full_refresh_reason_counts[frame_result.full_refresh_reason] = (
                transformed.full_refresh_reason_counts.get(
                    frame_result.full_refresh_reason, 0
                )
                + 1
            )
    return episode_stats


def _process_frame(
    cache: StatefulPatchCache | ThresholdPatchCache | GlobalFallbackPatchCache,
    frame: RGBImage,
    method_kind: str,
    frame_index: int,
) -> _FrameResult:
    if method_kind == "global_fallback":
        result: GlobalFallbackResult = cache.process_frame(frame)  # type: ignore[assignment]
        encoding = result.encoding
        return _FrameResult(
            encoding.features,
            encoding.refresh_decisions,
            encoding.patch_count,
            encoding.recomputed_patches,
            encoding.reused_patches,
            result.processing_latency_seconds,
            result.global_change_score,
            result.global_change_detected,
            result.full_refresh_reason,
            result.global_detector_runtime_seconds,
            result.global_change_score is not None,
        )

    result: StatefulEncodingResult | ThresholdEncodingResult = cache.process_frame(
        frame
    )  # type: ignore[assignment]
    reason = None
    if frame_index == 0:
        reason = "episode_initialization"
    elif result.recomputed_patches == result.patch_count:
        reason = "local_threshold_all_patches"
    return _FrameResult(
        result.features,
        result.refresh_decisions,
        result.patch_count,
        result.recomputed_patches,
        result.reused_patches,
        result.processing_latency_seconds,
        None,
        False,
        reason,
        0.0,
        False,
    )


def _add_transition(
    stats: _Stats, counts: ConfusionCounts, result: _FrameResult
) -> None:
    stats.counts = _sum_counts(stats.counts, counts)
    stats.transition_patch_count += result.patch_count
    stats.transition_recomputed_patches += result.recomputed_patches
    stats.transition_count += 1
    stats.full_frame_refresh_count += int(
        result.recomputed_patches == result.patch_count
    )
    stats.global_full_refresh_count += int(result.global_change_detected)


def _format_stats(stats: _Stats, *, include_all_frame: bool) -> dict[str, object]:
    detection = detection_metrics(stats.counts)
    detection.update(
        {
            "transition_patch_count": stats.transition_patch_count,
            "transition_recomputed_patches": stats.transition_recomputed_patches,
            "refresh_fraction": (
                stats.transition_recomputed_patches / stats.transition_patch_count
                if stats.transition_patch_count
                else 0.0
            ),
            "full_frame_refresh_count": stats.full_frame_refresh_count,
            "full_frame_refresh_frequency": (
                stats.full_frame_refresh_count / stats.transition_count
                if stats.transition_count
                else 0.0
            ),
            "global_full_refresh_count": stats.global_full_refresh_count,
            "global_full_refresh_frequency": (
                stats.global_full_refresh_count / stats.transition_count
                if stats.transition_count
                else 0.0
            ),
        }
    )
    summary: dict[str, object] = {
        "change_detection": detection,
        "feature_approximation": {
            "mean_absolute_feature_error": (
                stats.feature_error_sum / stats.feature_value_count
                if stats.feature_value_count
                else 0.0
            ),
            "maximum_absolute_feature_error": stats.maximum_feature_error,
            "scope": (
                "All frames including episode initial frames."
                if include_all_frame
                else "Transition destination frames of this transformation type."
            ),
        },
        "runtime": {
            "global_detector_runtime_seconds": stats.global_detector_runtime_seconds,
            "global_detector_comparison_count": stats.global_detector_comparison_count,
            "mean_global_detector_runtime_seconds": (
                stats.global_detector_runtime_seconds
                / stats.global_detector_comparison_count
                if stats.global_detector_comparison_count
                else 0.0
            ),
            "processing_runtime_seconds": stats.processing_runtime_seconds,
        },
        "full_refresh_reason_counts": dict(
            sorted(stats.full_refresh_reason_counts.items())
        ),
    }
    if include_all_frame:
        summary["computation"] = {
            "all_frame_patch_count": stats.total_patch_count,
            "all_frame_recomputed_patches": stats.recomputed_patches,
            "all_frame_reused_patches": stats.reused_patches,
            "all_frame_refresh_fraction": (
                stats.recomputed_patches / stats.total_patch_count
                if stats.total_patch_count
                else 0.0
            ),
            "patch_computation_proxy": {
                "name": "recomputed_patch_feature_operations",
                "value": stats.recomputed_patches,
                "unit": "patch_operations",
                "definition": (
                    "Actual RGB-statistics feature computations only; not neural "
                    "FLOPs and excludes image-comparison overhead."
                ),
            },
        }
    return summary


def _method_configuration(
    method_kind: str, method_value: float | None, patch_size: int
) -> dict[str, object]:
    if method_kind == "exact":
        return {"policy": "exact raw-patch equality", "patch_size": patch_size}
    if method_kind == "local":
        return {
            "policy": "local mean absolute RGB difference",
            "local_threshold": method_value,
            "patch_size": patch_size,
        }
    return {
        "policy": "global changed-pixel fraction then local threshold reuse",
        "local_threshold": GLOBAL_LOCAL_THRESHOLD,
        "global_change_threshold": method_value,
        "patch_size": patch_size,
    }


def _write_validation_result(
    summary: dict[str, object], output_path: str | Path, figure_path: str | Path
) -> None:
    output = Path(output_path)
    figure = Path(figure_path)
    if output.resolve() == figure.resolve():
        raise ValueError("JSON output and figure paths must be different")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_global_change_comparison_png(summary["methods"], figure)


def _read_manifest_rows(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(
            f"could not read validation manifest {path}: {error}"
        ) from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on manifest line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        rows.append(row)
    return tuple(rows)


def _resolve_mask_path(dataset_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError("ground-truth mask path is invalid or not portable")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or PureWindowsPath(raw_path).drive
        or ".." in relative.parts
        or not relative.parts
    ):
        raise ValueError("ground-truth mask path must stay inside validation data")
    resolved = (dataset_root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(dataset_root) or not resolved.is_file():
        raise ValueError("ground-truth mask path is missing or escapes validation data")
    return resolved


def _read_pgm_mask(path: Path) -> tuple[int, int, bytes]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read PGM mask {path}: {error}") from error
    header = data.split(b"\n", 3)
    if len(header) != 4 or header[0].strip() != b"P5":
        raise ValueError(f"invalid binary PGM mask header: {path}")
    try:
        width, height = (int(value) for value in header[1].split())
        maximum = int(header[2].strip())
    except ValueError as error:
        raise ValueError(f"invalid PGM mask dimensions or maximum: {path}") from error
    pixels = header[3]
    if width < 1 or height < 1 or maximum != 255 or len(pixels) != width * height:
        raise ValueError(f"invalid PGM mask dimensions or pixel count: {path}")
    return width, height, pixels


def _sum_stats(stats: list[_Stats]) -> _Stats:
    combined = _Stats()
    for item in stats:
        combined.counts = _sum_counts(combined.counts, item.counts)
        combined.transition_patch_count += item.transition_patch_count
        combined.transition_recomputed_patches += item.transition_recomputed_patches
        combined.total_patch_count += item.total_patch_count
        combined.recomputed_patches += item.recomputed_patches
        combined.reused_patches += item.reused_patches
        combined.full_frame_refresh_count += item.full_frame_refresh_count
        combined.global_full_refresh_count += item.global_full_refresh_count
        combined.transition_count += item.transition_count
        combined.feature_error_sum += item.feature_error_sum
        combined.feature_value_count += item.feature_value_count
        combined.maximum_feature_error = max(
            combined.maximum_feature_error, item.maximum_feature_error
        )
        combined.global_detector_runtime_seconds += item.global_detector_runtime_seconds
        combined.global_detector_comparison_count += (
            item.global_detector_comparison_count
        )
        combined.processing_runtime_seconds += item.processing_runtime_seconds
        for reason, count in item.full_refresh_reason_counts.items():
            combined.full_refresh_reason_counts[reason] = (
                combined.full_refresh_reason_counts.get(reason, 0) + count
            )
    return combined


def _sum_counts(first: ConfusionCounts, second: ConfusionCounts) -> ConfusionCounts:
    return ConfusionCounts(
        first.true_positives + second.true_positives,
        first.false_positives + second.false_positives,
        first.true_negatives + second.true_negatives,
        first.false_negatives + second.false_negatives,
    )
