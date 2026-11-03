"""Reproducible in-memory CPU latency benchmark for encoding policies."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
import sys
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .approximate import ThresholdPatchCache, compare_feature_vectors
from .baseline import load_manifest
from .encoder import FEATURE_DIMENSIONS, RGBImage, encode_image, read_ppm
from .global_fallback import GlobalFallbackPatchCache
from .plotting import write_latency_comparison_png
from .state import StatefulPatchCache

POLICY_NAMES = (
    "independent_frame_encoding",
    "exact_patch_reuse",
    "approximate_reuse_0.01",
    "approximate_reuse_0.05",
    "approximate_reuse_0.05_global_fallback",
)
BASELINE_POLICY = "independent_frame_encoding"


@dataclass(frozen=True, slots=True)
class EpisodeWorkload:
    """One preloaded episode and its independent reference features."""

    episode_id: str
    frames: tuple[RGBImage, ...]
    reference_features: tuple[tuple[tuple[float, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class PolicyAudit:
    """Untimed correctness and work-accounting results for one policy."""

    total_patch_count: int
    recomputed_patches: int
    reused_patches: int
    feature_mean_absolute_error: float
    feature_maximum_absolute_error: float
    peak_cache_memory_estimate_bytes: int


def load_validation_workload(
    manifest_path: str | Path,
    *,
    patch_size: int = 16,
    expected_episode_count: int | None = None,
    expected_frames_per_episode: int | None = None,
) -> tuple[EpisodeWorkload, ...]:
    """Preload validation frames and independent reference features.

    Manifest parsing, path validation, PPM reads, and reference feature
    extraction all happen before any benchmark timing starts.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    episodes = load_manifest(manifest_path)
    if expected_episode_count is not None and len(episodes) != expected_episode_count:
        raise ValueError(
            f"expected {expected_episode_count} validation episodes, "
            f"found {len(episodes)}"
        )
    workload: list[EpisodeWorkload] = []
    expected_dimensions: tuple[int, int] | None = None
    for episode in episodes:
        if not episode.episode_id.startswith("validation_"):
            raise ValueError(
                "latency benchmark accepts validation-prefixed episodes only"
            )
        if (
            expected_frames_per_episode is not None
            and len(episode.frames) != expected_frames_per_episode
        ):
            raise ValueError(
                f"episode {episode.episode_id!r} has {len(episode.frames)} frames; "
                f"expected {expected_frames_per_episode}"
            )
        frames = tuple(read_ppm(frame.path) for frame in episode.frames)
        for frame in frames:
            dimensions = (frame.width, frame.height)
            if expected_dimensions is None:
                expected_dimensions = dimensions
            elif dimensions != expected_dimensions:
                raise ValueError("validation workload images must share dimensions")
        references = tuple(
            encode_image(frame, patch_size=patch_size).features for frame in frames
        )
        workload.append(
            EpisodeWorkload(
                episode_id=episode.episode_id,
                frames=frames,
                reference_features=references,
            )
        )
    return tuple(workload)


def deterministic_policy_orders(
    policy_names: Sequence[str], rounds: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Return shuffled, reproducible policy orders for benchmark rounds."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not policy_names or len(set(policy_names)) != len(policy_names):
        raise ValueError("policy_names must be non-empty and unique")
    rng = random.Random(seed)
    orders: list[tuple[str, ...]] = []
    for _ in range(rounds):
        order = list(policy_names)
        rng.shuffle(order)
        orders.append(tuple(order))
    return tuple(orders)


def summarize_timings(
    latency_seconds_by_repetition: dict[str, Sequence[float]],
    *,
    baseline_policy: str = BASELINE_POLICY,
) -> dict[str, dict[str, float]]:
    """Summarize repeated workload timings and paired baseline speedups.

    Speedup samples are paired within repetition as baseline latency divided by
    policy latency. Standard deviation is the sample standard deviation. P95
    uses linear interpolation between ordered observations.
    """
    if baseline_policy not in latency_seconds_by_repetition:
        raise ValueError("baseline policy is missing from timing samples")
    baseline = tuple(latency_seconds_by_repetition[baseline_policy])
    if not baseline or any(
        value <= 0 or not math.isfinite(value) for value in baseline
    ):
        raise ValueError("baseline timings must be finite and positive")

    summaries: dict[str, dict[str, float]] = {}
    for policy_name, raw_values in latency_seconds_by_repetition.items():
        values = tuple(raw_values)
        if len(values) != len(baseline):
            raise ValueError("all policies must have the same repetition count")
        if any(value <= 0 or not math.isfinite(value) for value in values):
            raise ValueError("policy timings must be finite and positive")
        paired_speedups = tuple(
            base / value for base, value in zip(baseline, values, strict=True)
        )
        p95 = statistics.quantiles(values, n=100, method="inclusive")[94]
        summaries[policy_name] = {
            "median_latency_seconds": statistics.median(values),
            "mean_latency_seconds": statistics.fmean(values),
            "standard_deviation_seconds": statistics.stdev(values)
            if len(values) > 1
            else 0.0,
            "p95_latency_seconds": p95,
            "median_speedup_relative_to_independent": statistics.median(
                paired_speedups
            ),
            "ratio_of_median_latencies_relative_to_independent": (
                statistics.median(baseline) / statistics.median(values)
            ),
        }
    return summaries


def audit_policy(
    workload: Sequence[EpisodeWorkload],
    policy_name: str,
    *,
    patch_size: int = 16,
    global_change_threshold: float = 0.50,
) -> PolicyAudit:
    """Run one untimed policy pass to measure feature error and cache state."""
    _validate_policy_name(policy_name)
    total_patches = recomputed_patches = reused_patches = 0
    absolute_error_sum = 0.0
    feature_value_count = 0
    maximum_error = 0.0
    peak_cache_bytes = 0

    for episode in workload:
        cache = _make_cache(policy_name, patch_size, global_change_threshold)
        for frame, reference in zip(
            episode.frames, episode.reference_features, strict=True
        ):
            result = _process_frame(policy_name, cache, frame, patch_size)
            if policy_name == BASELINE_POLICY:
                features = result.features
                patch_count = result.processed_patches
                recomputed = patch_count
                reused = 0
            elif policy_name == "exact_patch_reuse":
                features = result.features
                patch_count = result.patch_count
                recomputed = result.recomputed_patches
                reused = result.reused_patches
            elif policy_name == "approximate_reuse_0.05_global_fallback":
                features = result.encoding.features
                patch_count = result.encoding.patch_count
                recomputed = result.encoding.recomputed_patches
                reused = result.encoding.reused_patches
            else:
                features = result.features
                patch_count = result.patch_count
                recomputed = result.recomputed_patches
                reused = result.reused_patches

            error = compare_feature_vectors(features, reference)
            absolute_error_sum += error.mean_absolute_error * error.feature_value_count
            feature_value_count += error.feature_value_count
            maximum_error = max(maximum_error, error.maximum_absolute_error)
            total_patches += patch_count
            recomputed_patches += recomputed
            reused_patches += reused
            peak_cache_bytes = max(peak_cache_bytes, _estimate_cache_bytes(cache))

    if total_patches == 0 or feature_value_count == 0:
        raise ValueError("validation workload must contain at least one frame")
    return PolicyAudit(
        total_patch_count=total_patches,
        recomputed_patches=recomputed_patches,
        reused_patches=reused_patches,
        feature_mean_absolute_error=absolute_error_sum / feature_value_count,
        feature_maximum_absolute_error=maximum_error,
        peak_cache_memory_estimate_bytes=peak_cache_bytes,
    )


def run_cpu_latency_benchmark(
    manifest_path: str | Path,
    output_json: str | Path = "results/latency_benchmark.json",
    figure_path: str | Path = "docs/figures/latency_comparison.png",
    *,
    feature_vectors_path: str | Path | None = None,
    patch_size: int = 16,
    warmup_repetitions: int = 5,
    timed_repetitions: int = 30,
    policy_order_seed: int = 20261102,
    global_change_threshold: float = 0.50,
    expected_episode_count: int | None = None,
    expected_frames_per_episode: int | None = None,
) -> dict[str, object]:
    """Measure all policies on a preloaded validation-only workload.

    Each repetition processes every episode in manifest order, creating a
    fresh cache for each episode. Warm-ups and timed runs use the same complete
    in-memory workload; only ``perf_counter`` around policy processing is
    included. Dataset loading, baseline reference generation, feature-error
    auditing, serialization, plotting, and interpreter startup are excluded.
    """
    if warmup_repetitions < 5:
        raise ValueError("at least five warm-up repetitions are required")
    if timed_repetitions < 30:
        raise ValueError("at least thirty timed repetitions are required")
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if (
        isinstance(global_change_threshold, bool)
        or not isinstance(global_change_threshold, (int, float))
        or not math.isfinite(global_change_threshold)
        or not 0.0 <= global_change_threshold <= 1.0
    ):
        raise ValueError("global_change_threshold must be between 0 and 1")

    workload = load_validation_workload(
        manifest_path,
        patch_size=patch_size,
        expected_episode_count=expected_episode_count,
        expected_frames_per_episode=expected_frames_per_episode,
    )
    json_path = Path(output_json)
    png_path = Path(figure_path)
    features_path = (
        Path(feature_vectors_path)
        if feature_vectors_path is not None
        else json_path.with_name(f"{json_path.stem}_features.jsonl")
    )
    output_paths = (json_path.resolve(), png_path.resolve(), features_path.resolve())
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("JSON, figure, and feature-vector output paths must differ")
    input_paths = {Path(manifest_path).resolve()}
    input_paths.update(
        frame.path.resolve()
        for episode in load_manifest(manifest_path)
        for frame in episode.frames
    )
    if any(path in input_paths for path in output_paths):
        raise ValueError("benchmark output paths must not overwrite input files")
    total_frames = sum(len(episode.frames) for episode in workload)
    orders_rng = random.Random(policy_order_seed)

    warmup_orders: list[list[str]] = []
    for _warmup_round in range(warmup_repetitions):
        order = list(POLICY_NAMES)
        orders_rng.shuffle(order)
        warmup_orders.append(order)
        for policy_name in order:
            _execute_policy_workload(
                workload, policy_name, patch_size, global_change_threshold
            )

    audits = {
        policy_name: audit_policy(
            workload,
            policy_name,
            patch_size=patch_size,
            global_change_threshold=global_change_threshold,
        )
        for policy_name in POLICY_NAMES
    }

    raw_repetitions: list[dict[str, object]] = []
    timing_values: dict[str, list[float]] = {
        policy_name: [] for policy_name in POLICY_NAMES
    }
    for repetition in range(timed_repetitions):
        order = list(POLICY_NAMES)
        orders_rng.shuffle(order)
        latencies: dict[str, float] = {}
        for policy_name in order:
            latencies[policy_name] = _time_policy_workload(
                workload, policy_name, patch_size, global_change_threshold
            )
            timing_values[policy_name].append(latencies[policy_name])
        raw_repetitions.append(
            {
                "repetition": repetition + 1,
                "policy_order": order,
                "latency_seconds_by_policy": latencies,
            }
        )

    timing_columns = {
        policy_name: [
            float(row["latency_seconds_by_policy"][policy_name])
            for row in raw_repetitions
        ]
        for policy_name in POLICY_NAMES
    }
    timing_summaries = summarize_timings(timing_columns)
    for repetition in raw_repetitions:
        latencies = repetition["latency_seconds_by_policy"]
        baseline_latency = latencies[BASELINE_POLICY]
        repetition["paired_speedup_by_policy"] = {
            policy_name: baseline_latency / latencies[policy_name]
            for policy_name in POLICY_NAMES
        }

    policy_summaries: dict[str, object] = {}
    for policy_name in POLICY_NAMES:
        audit = audits[policy_name]
        summary = timing_summaries[policy_name]
        policy_summaries[policy_name] = {
            **summary,
            "all_frame_patch_count": audit.total_patch_count,
            "all_frame_recomputed_patches": audit.recomputed_patches,
            "all_frame_reused_patches": audit.reused_patches,
            "all_frame_patch_recomputation_fraction": (
                audit.recomputed_patches / audit.total_patch_count
            ),
            "feature_mean_absolute_error": audit.feature_mean_absolute_error,
            "feature_maximum_absolute_error": audit.feature_maximum_absolute_error,
            "peak_cache_memory_estimate_bytes": audit.peak_cache_memory_estimate_bytes,
            "cache_memory_estimate_scope": (
                "Deep sys.getsizeof estimate of cache-owned containers, copied "
                "patch bytes, feature tuples, and scalar metadata. Global fallback's "
                "prior-frame reference is excluded because its RGB bytes belong to "
                "the preloaded workload, not an additional cache copy."
            ),
        }

    system = {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_build": sysconfig.get_config_var("CONFIG_ARGS") or "unknown",
        "operating_system": platform.platform(),
        "cpu_identifier": platform.processor() or "unknown",
        "cpu_architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    summary: dict[str, object] = {
        "experiment_label": (
            "Synthetic-benchmark findings: validation-only CPU latency benchmark"
        ),
        "partition": "validation",
        "manifest_path": Path(manifest_path).as_posix(),
        "episode_count": len(workload),
        "frame_count": total_frames,
        "patch_size": patch_size,
        "feature_dimensions": FEATURE_DIMENSIONS,
        "warmup_repetitions_per_policy": warmup_repetitions,
        "timed_repetitions_per_policy": timed_repetitions,
        "policy_order_seed": policy_order_seed,
        "warmup_policy_orders": warmup_orders,
        "global_change_threshold": global_change_threshold,
        "processing_cost_proxy": {
            "name": "actual_patch_feature_computations",
            "unit": "patch_operations",
            "count_field": "all_frame_recomputed_patches",
            "definition": (
                "Counts patches whose six-component RGB-statistics feature vector "
                "was computed. Raw-pixel comparisons, cache management, and global "
                "change detection are excluded; this is not a neural FLOP count."
            ),
        },
        "policies": policy_summaries,
        "raw_timing_repetitions": raw_repetitions,
        "timing_methodology": {
            "measured_scope": (
                "In-memory processing of every preloaded validation frame in its "
                "episode order, including cache creation/reset, comparisons, cache "
                "updates, and feature extraction."
            ),
            "excluded_scope": (
                "Manifest parsing, image loading, independent reference feature "
                "generation, feature-error audit, JSON serialization, plot generation, "
                "and Python startup."
            ),
            "repetition_interpretation": (
                "Repeated runs measure the same fixed workload, not independent "
                "dataset samples."
            ),
            "execution_order": (
                "Each warm-up and timed round uses a fresh deterministic shuffle of "
                "the policy order, generated from policy_order_seed."
            ),
            "median_speedup": (
                "Median of within-repetition ratios: independent-frame latency / "
                "policy latency. Ratio of median latencies is also reported."
            ),
            "standard_deviation": "Sample standard deviation across timed repetitions.",
            "p95": "95th percentile using inclusive linear interpolation.",
            "patch_operation_proxy": (
                "All-frame patch feature computations are reported separately from "
                "end-to-end latency, which includes comparisons and cache work."
            ),
            "latency_claim": (
                "Single-host CPU measurements; not GPU throughput, neural FLOPs, "
                "or a general speedup claim."
            ),
        },
        "system": system,
        "test_partition_accessed": False,
    }
    summary["feature_vectors"] = {
        "relative_path": Path(
            os.path.relpath(features_path.resolve(), start=json_path.resolve().parent)
        ).as_posix(),
        "format": "JSONL",
        "record_count": total_frames,
        "feature_dimensions": FEATURE_DIMENSIONS,
        "policies": list(POLICY_NAMES),
    }
    _write_feature_vectors(
        workload,
        features_path,
        patch_size=patch_size,
        global_change_threshold=global_change_threshold,
    )
    _write_latency_outputs(summary, json_path, png_path)
    return summary


def _time_policy_workload(
    workload: Sequence[EpisodeWorkload],
    policy_name: str,
    patch_size: int,
    global_change_threshold: float,
) -> float:
    _validate_policy_name(policy_name)
    started = time.perf_counter()
    _execute_policy_workload(workload, policy_name, patch_size, global_change_threshold)
    return time.perf_counter() - started


def _execute_policy_workload(
    workload: Sequence[EpisodeWorkload],
    policy_name: str,
    patch_size: int,
    global_change_threshold: float,
) -> None:
    _validate_policy_name(policy_name)
    for episode in workload:
        cache = _make_cache(policy_name, patch_size, global_change_threshold)
        for frame in episode.frames:
            _process_frame(policy_name, cache, frame, patch_size)


def _write_feature_vectors(
    workload: Sequence[EpisodeWorkload],
    path: str | Path,
    *,
    patch_size: int,
    global_change_threshold: float,
) -> None:
    """Save one JSONL record per frame with all policy feature vectors.

    This export pass is outside the timed region. Each stateful policy gets a
    fresh cache for each episode, matching the timed workload's reset policy.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output_file:
        for episode in workload:
            caches = {
                policy_name: _make_cache(
                    policy_name, patch_size, global_change_threshold
                )
                for policy_name in POLICY_NAMES
                if policy_name != BASELINE_POLICY
            }
            for frame_index, frame in enumerate(episode.frames):
                features_by_policy: dict[str, tuple[tuple[float, ...], ...]] = {}
                baseline_result = encode_image(frame, patch_size=patch_size)
                features_by_policy[BASELINE_POLICY] = baseline_result.features
                for policy_name in POLICY_NAMES:
                    if policy_name == BASELINE_POLICY:
                        continue
                    result = _process_frame(
                        policy_name, caches[policy_name], frame, patch_size
                    )
                    features_by_policy[policy_name] = _result_features(
                        policy_name, result
                    )
                record = {
                    "episode_id": episode.episode_id,
                    "frame_index": frame_index,
                    "patch_grid_dimensions": {
                        "rows": baseline_result.grid_rows,
                        "columns": baseline_result.grid_columns,
                    },
                    "feature_dimensions": baseline_result.feature_dimensions,
                    "patch_count": baseline_result.processed_patches,
                    "features_by_policy": features_by_policy,
                }
                output_file.write(json.dumps(record, separators=(",", ":")) + "\n")


def _result_features(
    policy_name: str,
    result: Any,
) -> tuple[tuple[float, ...], ...]:
    """Return the feature batch from the corresponding policy result object."""
    if policy_name == "approximate_reuse_0.05_global_fallback":
        return result.encoding.features
    return result.features


def _process_frame(
    policy_name: str,
    cache: StatefulPatchCache | ThresholdPatchCache | GlobalFallbackPatchCache | None,
    frame: RGBImage,
    patch_size: int,
) -> Any:
    if policy_name == BASELINE_POLICY:
        return encode_image(frame, patch_size=patch_size)
    if cache is None:
        raise ValueError(f"policy {policy_name!r} requires a cache")
    return cache.process_frame(frame)


def _make_cache(
    policy_name: str, patch_size: int, global_change_threshold: float
) -> StatefulPatchCache | ThresholdPatchCache | GlobalFallbackPatchCache | None:
    if policy_name == BASELINE_POLICY:
        return None
    if policy_name == "exact_patch_reuse":
        return StatefulPatchCache(patch_size=patch_size)
    if policy_name == "approximate_reuse_0.01":
        return ThresholdPatchCache(threshold=0.01, patch_size=patch_size)
    if policy_name == "approximate_reuse_0.05":
        return ThresholdPatchCache(threshold=0.05, patch_size=patch_size)
    if policy_name == "approximate_reuse_0.05_global_fallback":
        return GlobalFallbackPatchCache(
            local_threshold=0.05,
            global_change_threshold=global_change_threshold,
            patch_size=patch_size,
        )
    raise ValueError(f"unknown policy: {policy_name}")


def _estimate_cache_bytes(
    cache: StatefulPatchCache | ThresholdPatchCache | GlobalFallbackPatchCache | None,
) -> int:
    """Estimate reachable Python cache state without walking module objects."""
    if cache is None:
        return 0
    seen: set[int] = set()

    def deep_size(value: object) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(value)
        if isinstance(value, (tuple, list, set, frozenset)):
            size += sum(deep_size(item) for item in value)
        elif isinstance(value, RGBImage):
            size += deep_size(value.pixels)
        return size

    total = deep_size(cache)
    if isinstance(cache, GlobalFallbackPatchCache):
        inner = cache._cache
        total += deep_size(inner)
        total += deep_size(inner._image_shape)
        total += deep_size(inner._raw_patches)
        total += deep_size(inner._features)
        total += deep_size(inner._patch_size)
        total += deep_size(inner.threshold)
        total += deep_size(cache._previous_dimensions)
        total += deep_size(cache._global_change_threshold)
    else:
        total += deep_size(cache._image_shape)
        total += deep_size(cache._raw_patches)
        total += deep_size(cache._features)
        total += deep_size(cache._patch_size)
        if isinstance(cache, ThresholdPatchCache):
            total += deep_size(cache.threshold)
    return total


def _write_latency_outputs(
    summary: dict[str, object], output_json: str | Path, figure_path: str | Path
) -> None:
    json_path = Path(output_json)
    png_path = Path(figure_path)
    if json_path.resolve() == png_path.resolve():
        raise ValueError("JSON and figure output paths must differ")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_latency_comparison_png(summary["policies"], png_path)


def _validate_policy_name(policy_name: str) -> None:
    if policy_name not in POLICY_NAMES:
        raise ValueError(f"unknown policy: {policy_name}")
