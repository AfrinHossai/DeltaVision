"""Controlled validation-only evaluation of independent and exact DINOv2 passes."""

from __future__ import annotations

import array
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .baseline import ManifestEpisode, load_manifest
from .encoder import RGBImage, read_ppm
from .pretrained import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    FrozenVisionEncoderAdapter,
    ImagePreprocessingConfig,
    RepresentationKind,
)
from .pretrained_baseline import _flatten_values
from .pretrained_stateful import ExactWholeFrameCache, _reproducibility_metrics
from .synthetic import generate_sequences

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_CONFIG = REPOSITORY_ROOT / "configs/pretrained_cache_evaluation.json"
DEFAULT_BASELINE_CONFIG = REPOSITORY_ROOT / "configs/pretrained_baseline.json"
DEFAULT_VALIDATION_ROOT = (
    REPOSITORY_ROOT / "data/generated/pretrained_cache_evaluation/validation"
)
DEFAULT_OUTPUT_JSON = REPOSITORY_ROOT / "results/pretrained_cache_evaluation.json"
DEFAULT_OUTPUT_CSV = REPOSITORY_ROOT / "results/pretrained_cache_evaluation.csv"
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "docs/figures"
CSV_FIELDS = (
    "condition",
    "condition_label",
    "repetition",
    "execution_order",
    "policy",
    "total_frames",
    "model_forward_calls",
    "cached_frame_hits",
    "cache_misses",
    "cache_hit_rate",
    "avoided_forward_percentage",
    "processing_latency_seconds",
    "model_inference_seconds",
    "preprocessing_seconds",
    "cache_lookup_overhead_seconds",
    "cache_retrieval_overhead_seconds",
    "cache_storage_overhead_seconds",
    "peak_cache_memory_estimate_bytes",
)
EpisodeImages = tuple[tuple[str, tuple[RGBImage, ...]], ...]


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable experiment settings."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read evaluation config {path}: {error}") from error
    validate_evaluation_config(config)
    return config


def validate_evaluation_config(config: Mapping[str, Any]) -> None:
    """Reject incomplete or non-reproducible scenario and timing settings."""
    positive_integer_fields = (
        "width",
        "height",
        "sequence_length",
        "num_episodes",
        "warmup_repetitions",
        "timed_repetitions",
    )
    for field in positive_integer_fields:
        value = config.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config["warmup_repetitions"] < 3:
        raise ValueError("at least three untimed warm-up repetitions are required")
    if config["timed_repetitions"] < 5:
        raise ValueError("at least five timed repetitions are required")
    for field in ("source_seed", "widespread_seed", "policy_order_seed"):
        if not isinstance(config.get(field), int):
            raise ValueError(f"{field} must be an integer")
    if (
        not isinstance(config.get("cache_capacity"), int)
        or config["cache_capacity"] < 0
    ):
        raise ValueError("cache_capacity must be a non-negative integer")
    if len(config.get("conditions", ())) != 4:
        raise ValueError("exactly four temporal-redundancy conditions are required")
    names = [condition.get("name") for condition in config["conditions"]]
    if names != [
        "identical_consecutive_frames",
        "sparse_visual_changes",
        "moderate_visual_changes",
        "frequent_widespread_changes",
    ]:
        raise ValueError("conditions must appear in the documented fixed order")
    source_indices = [
        condition.get("source_frame_indices")
        for condition in config["conditions"]
        if condition.get("kind") == "replay"
    ]
    for indices in source_indices:
        if len(indices) != config["sequence_length"]:
            raise ValueError("replay source_frame_indices must match sequence_length")
        if any(not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError(
                "replay source frame indices must be non-negative integers"
            )
    brightness_factors = config.get("brightness_sweep_factors", ())
    if len(brightness_factors) != config["sequence_length"]:
        raise ValueError("brightness_sweep_factors must match sequence_length")
    factors = [float(value) for value in brightness_factors]
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in factors):
        raise ValueError("brightness sweep factors must be finite values in (0, 1]")
    if any(right <= left for left, right in zip(factors, factors[1:], strict=False)):
        raise ValueError("brightness sweep factors must be strictly increasing")
    if any(
        (right - left) * 18.0 <= 1.0
        for left, right in zip(factors, factors[1:], strict=False)
    ):
        raise ValueError("brightness sweep steps must change every generated pixel")
    if any(
        not math.isfinite(float(config.get(field, float("nan"))))
        for field in (
            "maximum_absolute_difference_tolerance",
            "minimum_cosine_similarity_tolerance",
        )
    ):
        raise ValueError("feature comparison tolerances must be finite numbers")
    if config["maximum_absolute_difference_tolerance"] < 0.0:
        raise ValueError("maximum absolute difference tolerance cannot be negative")
    if not -1.0 <= config["minimum_cosine_similarity_tolerance"] <= 1.0:
        raise ValueError("minimum cosine similarity tolerance must be in [-1, 1]")


def generate_validation_scenarios(
    validation_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Path]:
    """Generate four deterministic validation manifests from synthetic episodes."""
    validate_evaluation_config(config)
    root = Path(validation_root).resolve()
    if "test" in {part.lower() for part in root.parts}:
        raise ValueError("held-out test paths are excluded from scenario generation")
    if "validation" not in {part.lower() for part in root.parts}:
        raise ValueError("scenario output must be inside a validation directory")
    source_scene_manifest = generate_sequences(
        root / "source_scenes",
        width=config["width"],
        height=config["height"],
        sequence_length=1,
        num_episodes=config["num_episodes"],
        seed=config["source_seed"],
        transformations=("global_brightness",),
        episode_id_prefix="cache_source_scene",
    )
    source_scene_episodes = load_manifest(source_scene_manifest)
    source_sweep_manifest = _write_brightness_sweep(
        source_scene_episodes,
        root / "source_brightness_sweep",
        condition_name="source_brightness_sweep",
        factors=config["brightness_sweep_factors"],
        seed=config["source_seed"],
    )
    source_episodes = load_manifest(source_sweep_manifest)
    manifests: dict[str, Path] = {}
    for condition in config["conditions"]:
        name = condition["name"]
        condition_root = root / name
        if condition["kind"] == "replay":
            manifest = _write_replay_condition(
                source_episodes,
                condition_root,
                name=name,
                source_indices=condition["source_frame_indices"],
                seed=config["source_seed"],
            )
        elif condition["kind"] == "brightness_sweep":
            widespread_scene_manifest = generate_sequences(
                root / "widespread_source_scenes",
                width=config["width"],
                height=config["height"],
                sequence_length=1,
                num_episodes=config["num_episodes"],
                seed=config["widespread_seed"],
                transformations=("global_brightness",),
                episode_id_prefix="widespread_source_scene",
            )
            manifest = _write_brightness_sweep(
                load_manifest(widespread_scene_manifest),
                condition_root,
                condition_name=name,
                factors=config["brightness_sweep_factors"],
                seed=config["widespread_seed"],
            )
        else:
            raise ValueError(f"unsupported condition kind for {name}")
        manifests[name] = manifest
    return manifests


def _write_brightness_sweep(
    scene_episodes: Sequence[ManifestEpisode],
    output_root: Path,
    *,
    condition_name: str,
    factors: Sequence[float],
    seed: int,
) -> Path:
    """Render controlled global intensity variants of generated base scenes."""
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode_index, scene_episode in enumerate(scene_episodes):
        base = read_ppm(scene_episode.frames[0].path)
        episode_id = f"{condition_name}_episode_{episode_index:04d}"
        episode_dir = output_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[str] = []
        for frame_index, factor in enumerate(factors):
            pixels = bytes(round(channel * float(factor)) for channel in base.pixels)
            frame = RGBImage(base.width, base.height, pixels)
            relative = Path(episode_id) / f"frame_{frame_index:04d}.ppm"
            destination = output_root / relative
            destination.write_bytes(
                f"P6\n{frame.width} {frame.height}\n255\n".encode("ascii")
                + frame.pixels
            )
            frame_paths.append(relative.as_posix())
        transitions = [
            {
                "from_frame": frame_index - 1,
                "to_frame": frame_index,
                "transformation_type": "global_brightness_sweep",
                "previous_factor": float(factors[frame_index - 1]),
                "factor": float(factors[frame_index]),
            }
            for frame_index in range(1, len(factors))
        ]
        rows.append(
            {
                "episode_id": episode_id,
                "random_seed": seed + episode_index,
                "frame_paths": frame_paths,
                "transitions": transitions,
                "source_episode_id": scene_episode.episode_id,
                "brightness_factors": [float(value) for value in factors],
                "condition": condition_name,
            }
        )
    manifest = output_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            output.write("\n")
    return manifest


def _write_replay_condition(
    source_episodes: Sequence[ManifestEpisode],
    output_root: Path,
    *,
    name: str,
    source_indices: Sequence[int],
    seed: int,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode_index, source_episode in enumerate(source_episodes):
        if max(source_indices, default=0) >= len(source_episode.frames):
            raise ValueError("replay source index exceeds the generated source episode")
        episode_id = f"{name}_episode_{episode_index:04d}"
        episode_dir = output_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for frame_index, source_index in enumerate(source_indices):
            source = source_episode.frames[source_index].path
            relative = Path(episode_id) / f"frame_{frame_index:04d}.ppm"
            destination = output_root / relative
            shutil.copyfile(source, destination)
            paths.append(relative.as_posix())
        rows.append(
            {
                "episode_id": episode_id,
                "random_seed": seed + episode_index,
                "frame_paths": paths,
                "source_frame_indices": list(source_indices),
                "source_episode_id": source_episode.episode_id,
                "condition": name,
            }
        )
    manifest = output_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            output.write("\n")
    return manifest


def preload_validation_episodes(manifest_path: str | Path) -> EpisodeImages:
    """Load every image before timed policy passes and require validation paths."""
    manifest = Path(manifest_path).resolve()
    path_parts = {part.lower() for part in manifest.parts}
    if "test" in path_parts:
        raise ValueError("held-out test split paths are excluded from this evaluation")
    if "validation" not in path_parts:
        raise ValueError("input manifests must be inside the validation partition")
    return tuple(
        (
            episode.episode_id,
            tuple(read_ppm(frame.path) for frame in episode.frames),
        )
        for episode in load_manifest(manifest)
    )


def summarize_visual_changes(episodes: EpisodeImages) -> dict[str, Any]:
    """Describe exact rendered RGB differences without semantic interpretation."""
    fractions: list[float] = []
    for _, images in episodes:
        for previous, current in zip(images, images[1:], strict=False):
            pixel_count = previous.width * previous.height
            if (previous.width, previous.height) != (current.width, current.height):
                raise ValueError("frame dimensions changed inside an episode")
            changed_pixels = sum(
                previous.pixels[offset : offset + 3]
                != current.pixels[offset : offset + 3]
                for offset in range(0, len(previous.pixels), 3)
            )
            fractions.append(changed_pixels / pixel_count)
    changed_transitions = sum(fraction > 0.0 for fraction in fractions)
    return {
        "transition_count": len(fractions),
        "changed_transition_count": changed_transitions,
        "changed_transition_fraction": (
            changed_transitions / len(fractions) if fractions else 0.0
        ),
        "mean_changed_pixel_fraction_per_transition": (
            statistics.fmean(fractions) if fractions else 0.0
        ),
        "interpretation": (
            "Pixel differences in generated RGB frames; this is not semantic "
            "change detection."
        ),
    }


def run_policy_pass(
    adapter: FrozenVisionEncoderAdapter,
    episodes: EpisodeImages,
    *,
    policy: str,
    representation: RepresentationKind,
    cache_capacity: int,
) -> dict[str, Any]:
    """Run one preloaded episode set and measure the complete processing path."""
    if policy not in ("independent", "exact_cache"):
        raise ValueError("policy must be 'independent' or 'exact_cache'")
    total_frames = sum(len(images) for _, images in episodes)
    if total_frames < 1:
        raise ValueError("episodes must contain at least one frame")
    start = time.perf_counter()
    feature_shapes: set[tuple[int, ...]] = set()
    model_inference_seconds = 0.0
    preprocessing_seconds = 0.0
    if policy == "independent":
        for _, images in episodes:
            for image in images:
                result = adapter.encode_frame(image, representation=representation)
                feature_shapes.add(result.feature_shape)
                model_inference_seconds += result.model_forward_latency_seconds
                preprocessing_seconds += max(
                    0.0,
                    result.inference_latency_seconds
                    - result.model_forward_latency_seconds,
                )
        processing_seconds = time.perf_counter() - start
        return {
            "policy": policy,
            "cache_enabled": False,
            "total_frames": total_frames,
            "model_forward_calls": total_frames,
            "cached_frame_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
            "avoided_forward_percentage": 0.0,
            "processing_latency_seconds": processing_seconds,
            "model_inference_seconds": model_inference_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "cache_lookup_overhead_seconds": 0.0,
            "cache_retrieval_overhead_seconds": 0.0,
            "cache_storage_overhead_seconds": 0.0,
            "peak_cache_memory_estimate_bytes": 0,
            "cache_evictions": 0,
            "episode_cache_resets": 0,
            "feature_shape": _common_feature_shape(feature_shapes),
        }

    cache = ExactWholeFrameCache(adapter, capacity=cache_capacity)
    for _, images in episodes:
        cache.reset_episode()
        for image in images:
            result = cache.encode_frame(image, representation=representation)
            feature_shapes.add(result.encoding.feature_shape)
    processing_seconds = time.perf_counter() - start
    if cache.total_frames != total_frames:
        raise ValueError("cache frame accounting does not match preloaded inputs")
    if cache.cache_hits + cache.cache_misses != total_frames:
        raise ValueError("cache hits and misses do not account for every input frame")
    if cache.actual_model_forward_calls != cache.cache_misses:
        raise ValueError(
            "each exact cache miss must trigger one complete model forward"
        )
    cache_metrics = cache.metrics(total_processing_time_seconds=processing_seconds)
    return {
        "policy": policy,
        "cache_enabled": True,
        "total_frames": cache.total_frames,
        "model_forward_calls": cache.actual_model_forward_calls,
        "cached_frame_hits": cache.cache_hits,
        "cache_misses": cache.cache_misses,
        "cache_hit_rate": cache_metrics["cache_hit_rate"],
        "avoided_forward_percentage": (
            100.0 * cache.cache_hits / cache.total_frames if cache.total_frames else 0.0
        ),
        "processing_latency_seconds": processing_seconds,
        "model_inference_seconds": cache.model_inference_time_seconds,
        "preprocessing_seconds": cache.preprocessing_seconds,
        "cache_lookup_overhead_seconds": cache.cache_lookup_overhead_seconds,
        "cache_retrieval_overhead_seconds": cache.cache_retrieval_overhead_seconds,
        "cache_storage_overhead_seconds": cache.cache_storage_overhead_seconds,
        "peak_cache_memory_estimate_bytes": cache.peak_cache_payload_bytes,
        "cache_evictions": cache.cache_evictions,
        "episode_cache_resets": cache.episode_resets,
        "feature_shape": _common_feature_shape(feature_shapes),
    }


def _common_feature_shape(shapes: set[tuple[int, ...]]) -> list[int] | None:
    if len(shapes) > 1:
        raise ValueError(
            f"representation feature shape changed during a pass: {shapes}"
        )
    return list(next(iter(shapes))) if shapes else None


def compare_policy_features(
    adapter: FrozenVisionEncoderAdapter,
    episodes: EpisodeImages,
    *,
    representation: RepresentationKind,
    cache_capacity: int,
    maximum_absolute_difference_tolerance: float,
    minimum_cosine_similarity_tolerance: float,
) -> dict[str, Any]:
    """Run a separate untimed output audit against independent full inference."""
    independent_values = array.array("f")
    cached_values = array.array("f")
    independent_calls = 0
    stateful_calls = 0
    cache = ExactWholeFrameCache(adapter, capacity=cache_capacity)
    for _, images in episodes:
        cache.reset_episode()
        for image in images:
            independent = adapter.encode_frame(image, representation=representation)
            cached = cache.encode_frame(image, representation=representation)
            independent_values.extend(_flatten_values(independent.features))
            cached_values.extend(_flatten_values(cached.encoding.features))
            independent_calls += 1
    stateful_calls = cache.actual_model_forward_calls
    metrics = _reproducibility_metrics(independent_values, cached_values)
    within_tolerance = (
        metrics["maximum_absolute_difference"] <= maximum_absolute_difference_tolerance
        and metrics["cosine_similarity"] >= minimum_cosine_similarity_tolerance
    )
    return {
        **metrics,
        "independent_comparison_forward_calls": independent_calls,
        "stateful_comparison_forward_calls": stateful_calls,
        "maximum_absolute_difference_tolerance": (
            maximum_absolute_difference_tolerance
        ),
        "minimum_cosine_similarity_tolerance": minimum_cosine_similarity_tolerance,
        "within_documented_tolerance": within_tolerance,
        "comparison_timing_included_in_policy_latency": False,
    }


def timing_summary(samples: Sequence[float]) -> dict[str, float]:
    """Summarize raw wall-clock samples with sample standard deviation."""
    if not samples:
        raise ValueError("at least one timing sample is required")
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("timing samples must be finite and non-negative")
    return {
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "sample_standard_deviation_seconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "repetition_count": len(values),
    }


def evaluate_conditions(
    adapter: FrozenVisionEncoderAdapter,
    scenarios: Mapping[str, tuple[Path, EpisodeImages]],
    config: Mapping[str, Any],
    *,
    representation: RepresentationKind,
) -> dict[str, Any]:
    """Run warmups, randomized paired timings, and a separate feature audit."""
    if set(scenarios) != {condition["name"] for condition in config["conditions"]}:
        raise ValueError("scenario names do not match the configured conditions")
    rng = random.Random(config["policy_order_seed"])
    raw_timings: list[dict[str, Any]] = []
    condition_results: dict[str, Any] = {}
    schedule: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        name = condition["name"]
        manifest, episodes = scenarios[name]
        policy = {"independent": [], "exact_cache": []}
        warmup_trials: list[dict[str, Any]] = []
        for repetition in range(config["warmup_repetitions"]):
            order = ["independent", "exact_cache"]
            rng.shuffle(order)
            schedule.append(
                {
                    "condition": name,
                    "phase": "warmup",
                    "repetition": repetition + 1,
                    "policy_order": order,
                }
            )
            for ordered_policy in order:
                result = run_policy_pass(
                    adapter,
                    episodes,
                    policy=ordered_policy,
                    representation=representation,
                    cache_capacity=config["cache_capacity"],
                )
                warmup_trials.append(
                    {
                        "repetition": repetition + 1,
                        "execution_order": order.index(ordered_policy) + 1,
                        **result,
                        "processing_latency_seconds": None,
                    }
                )

        for repetition in range(config["timed_repetitions"]):
            order = ["independent", "exact_cache"]
            rng.shuffle(order)
            schedule.append(
                {
                    "condition": name,
                    "phase": "timed",
                    "repetition": repetition + 1,
                    "policy_order": order,
                }
            )
            for order_index, ordered_policy in enumerate(order, start=1):
                result = run_policy_pass(
                    adapter,
                    episodes,
                    policy=ordered_policy,
                    representation=representation,
                    cache_capacity=config["cache_capacity"],
                )
                policy[ordered_policy].append(result)
                raw_timings.append(
                    {
                        "condition": name,
                        "condition_label": condition["label"],
                        "repetition": repetition + 1,
                        "execution_order": order_index,
                        **result,
                    }
                )

        independent = policy["independent"]
        cached = policy["exact_cache"]
        if not independent or not cached:
            raise ValueError("each policy must have at least one timed repetition")
        independent_times = [item["processing_latency_seconds"] for item in independent]
        cached_times = [item["processing_latency_seconds"] for item in cached]
        independent_mean = statistics.fmean(independent_times)
        cached_mean = statistics.fmean(cached_times)
        speedup_ratio = independent_mean / cached_mean if cached_mean else None
        if speedup_ratio is None:
            latency_outcome = "undefined_zero_cached_latency"
        elif math.isclose(speedup_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            latency_outcome = "approximately_equal"
        elif speedup_ratio > 1.0:
            latency_outcome = "measured_speedup"
        else:
            latency_outcome = "measured_slowdown"
        representative_independent = independent[0]
        representative_cached = cached[0]
        if any(
            item["model_forward_calls"]
            != representative_independent["model_forward_calls"]
            for item in independent + cached
            if item["policy"] == "independent"
        ):
            raise ValueError(
                "independent forward call count changed between repetitions"
            )
        if any(
            item["model_forward_calls"] != representative_cached["model_forward_calls"]
            for item in cached
        ):
            raise ValueError("cache forward call count changed between repetitions")
        equivalence = compare_policy_features(
            adapter,
            episodes,
            representation=representation,
            cache_capacity=config["cache_capacity"],
            maximum_absolute_difference_tolerance=float(
                config["maximum_absolute_difference_tolerance"]
            ),
            minimum_cosine_similarity_tolerance=float(
                config["minimum_cosine_similarity_tolerance"]
            ),
        )
        visual = summarize_visual_changes(episodes)
        condition_results[name] = {
            "label": condition["label"],
            "manifest": str(manifest),
            "total_frames_per_pass": representative_independent["total_frames"],
            "visual_change_summary": visual,
            "warmup_repetitions_per_policy": config["warmup_repetitions"],
            "timed_repetitions_per_policy": config["timed_repetitions"],
            "warmup_trials": warmup_trials,
            "independent": {
                "model_forward_calls_per_pass": (
                    representative_independent["model_forward_calls"]
                ),
                "timed_model_forward_calls_total": sum(
                    item["model_forward_calls"] for item in independent
                ),
                "warmup_model_forward_calls_total": sum(
                    item["model_forward_calls"]
                    for item in warmup_trials
                    if item["policy"] == "independent"
                ),
                "mean_processing_latency": timing_summary(independent_times),
                "mean_model_inference_seconds": statistics.fmean(
                    item["model_inference_seconds"] for item in independent
                ),
                "mean_preprocessing_seconds": statistics.fmean(
                    item["preprocessing_seconds"] for item in independent
                ),
                "feature_shape": representative_independent["feature_shape"],
            },
            "exact_cache": {
                "model_forward_calls_per_pass": (
                    representative_cached["model_forward_calls"]
                ),
                "timed_model_forward_calls_total": sum(
                    item["model_forward_calls"] for item in cached
                ),
                "warmup_model_forward_calls_total": sum(
                    item["model_forward_calls"]
                    for item in warmup_trials
                    if item["policy"] == "exact_cache"
                ),
                "cached_frame_hits_per_pass": (
                    representative_cached["cached_frame_hits"]
                ),
                "cache_misses_per_pass": representative_cached["cache_misses"],
                "cache_hit_rate": representative_cached["cache_hit_rate"],
                "avoided_forward_percentage": (
                    representative_cached["avoided_forward_percentage"]
                ),
                "mean_processing_latency": timing_summary(cached_times),
                "mean_model_inference_seconds": statistics.fmean(
                    item["model_inference_seconds"] for item in cached
                ),
                "mean_preprocessing_seconds": statistics.fmean(
                    item["preprocessing_seconds"] for item in cached
                ),
                "mean_cache_lookup_overhead_seconds": statistics.fmean(
                    item["cache_lookup_overhead_seconds"] for item in cached
                ),
                "mean_cache_retrieval_overhead_seconds": statistics.fmean(
                    item["cache_retrieval_overhead_seconds"] for item in cached
                ),
                "mean_cache_storage_overhead_seconds": statistics.fmean(
                    item["cache_storage_overhead_seconds"] for item in cached
                ),
                "mean_cache_comparison_and_management_seconds": statistics.fmean(
                    item["cache_lookup_overhead_seconds"]
                    + item["cache_retrieval_overhead_seconds"]
                    + item["cache_storage_overhead_seconds"]
                    for item in cached
                ),
                "mean_peak_cache_memory_estimate_bytes": statistics.fmean(
                    item["peak_cache_memory_estimate_bytes"] for item in cached
                ),
                "maximum_peak_cache_memory_estimate_bytes": max(
                    item["peak_cache_memory_estimate_bytes"] for item in cached
                ),
                "cache_capacity_per_episode": config["cache_capacity"],
                "feature_shape": representative_cached["feature_shape"],
            },
            "latency_comparison": {
                "independent_mean_seconds": independent_mean,
                "exact_cache_mean_seconds": cached_mean,
                "independent_over_exact_cache_speedup_ratio": speedup_ratio,
                "latency_change_percent_vs_independent": (
                    (cached_mean - independent_mean) / independent_mean * 100.0
                    if independent_mean
                    else 0.0
                ),
                "outcome": latency_outcome,
            },
            "feature_equivalence": equivalence,
        }

    return {
        "conditions": condition_results,
        "raw_timed_trials": raw_timings,
        "execution_schedule": schedule,
        "raw_timing_policy": (
            "Preloaded images; wall time includes adapter preprocessing, cache "
            "lookup and exact comparison, complete DINOv2 inference on misses, "
            "cache storage/retrieval, and loop overhead. Excludes disk reads, "
            "model loading, feature audit, result serialization, and plotting."
        ),
    }


def build_evaluation_report(
    evaluation: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    adapter: FrozenVisionEncoderAdapter,
    config_path: str | Path,
    baseline_config_path: str | Path,
) -> dict[str, Any]:
    """Add exact model settings, runtime versions, and interpretation limits."""
    condition_results = evaluation["conditions"]
    timed_independent_calls = sum(
        result["independent"]["timed_model_forward_calls_total"]
        for result in condition_results.values()
    )
    timed_cached_calls = sum(
        result["exact_cache"]["timed_model_forward_calls_total"]
        for result in condition_results.values()
    )
    warmup_independent_calls = sum(
        result["independent"]["warmup_model_forward_calls_total"]
        for result in condition_results.values()
    )
    warmup_cached_calls = sum(
        result["exact_cache"]["warmup_model_forward_calls_total"]
        for result in condition_results.values()
    )
    audit_independent_calls = sum(
        result["feature_equivalence"]["independent_comparison_forward_calls"]
        for result in condition_results.values()
    )
    audit_cached_calls = sum(
        result["feature_equivalence"]["stateful_comparison_forward_calls"]
        for result in condition_results.values()
    )
    timed_calls_avoided = timed_independent_calls - timed_cached_calls
    forward_call_accounting = {
        "timed_passes": {
            "independent_full_model_calls": timed_independent_calls,
            "exact_cache_full_model_calls": timed_cached_calls,
            "full_model_calls_avoided": timed_calls_avoided,
            "avoided_call_fraction": (
                timed_calls_avoided / timed_independent_calls
                if timed_independent_calls
                else 0.0
            ),
            "scope": "all five timed repetitions, four conditions, both policies",
        },
        "untimed_warmups": {
            "independent_full_model_calls": warmup_independent_calls,
            "exact_cache_full_model_calls": warmup_cached_calls,
        },
        "separate_feature_equivalence_audits": {
            "independent_full_model_calls": audit_independent_calls,
            "exact_cache_full_model_calls": audit_cached_calls,
            "included_in_policy_latency": False,
        },
        "all_evaluation_forward_calls": (
            timed_independent_calls
            + timed_cached_calls
            + warmup_independent_calls
            + warmup_cached_calls
            + audit_independent_calls
            + audit_cached_calls
        ),
    }
    hardware = runtime_hardware(adapter)
    model_config = getattr(adapter.model, "config", None)
    previous_task14 = _previous_task14_runtime()
    current_versions = {
        key: hardware.get(key)
        for key in ("python", "torch_version", "transformers_version")
    }
    version_changes = {
        key: {"task14": previous_task14.get(key), "task15": value}
        for key, value in current_versions.items()
        if previous_task14.get(key) != value
    }
    cache_design = {
        "key": "SHA-256 of every preprocessed model input",
        "collision_safety": "candidate reuse is conditional on exact tensor equality",
        "cached_object": "complete selected representation and complete inputs",
        "capacity_scope": "per episode",
        "eviction_policy": "deterministic least-recently-used",
        "partial_token_computation": False,
        "approximate_reuse": False,
    }
    return {
        "experiment": "frozen DINOv2 independent frames vs exact whole-frame cache",
        "status": "completed",
        "validation_only": True,
        "held_out_test_partition_accessed": False,
        "evaluation_config_path": str(Path(config_path).resolve()),
        "baseline_config_path": str(Path(baseline_config_path).resolve()),
        "evaluation_config": dict(config),
        "model": {
            "model_identifier": adapter.model_id,
            "model_revision": adapter.revision,
            "model_config": {
                "model_type": getattr(model_config, "model_type", None),
                "patch_size": getattr(model_config, "patch_size", None),
                "hidden_size": getattr(model_config, "hidden_size", None),
                "num_register_tokens": getattr(model_config, "num_register_tokens", 0),
            },
            "representation": baseline_config.get("representation", "spatial_tokens"),
            "preprocessing": adapter.preprocessing.to_dict(),
            "device": adapter.device,
            "frozen_inference": not any(
                bool(parameter.requires_grad)
                for parameter in adapter.model.parameters()
            ),
            "evaluation_mode": not bool(getattr(adapter.model, "training", False)),
            "deterministic_inference_requested": adapter.deterministic,
            "checkpoint_downloads_allowed": False,
            "checkpoint_cache_dir": baseline_config.get("cache_dir"),
        },
        "hardware_and_software": hardware,
        "task14_runtime_comparison": {
            "previous_task14_runtime": previous_task14,
            "version_changes": version_changes,
            "note": (
                "Task 15 was run with the currently installed local environment; "
                "these runtime versions differ from Task 14 where shown."
            ),
        },
        "cache_design": cache_design,
        "experiment_protocol": {
            "image_loading": "all frames read before any policy timing",
            "timing_repetitions": config["timed_repetitions"],
            "untimed_warmup_repetitions": config["warmup_repetitions"],
            "policy_order_seed": config["policy_order_seed"],
            "policy_order": (
                "deterministically shuffled within each condition/repetition"
            ),
            "cache_reset": "before each episode in every stateful pass",
            "timing_scope": evaluation["raw_timing_policy"],
            "independent_calls_per_pass": "one full-model call per input frame",
            "stateful_calls_per_pass": "one full-model call per exact cache miss",
            "latency_interpretation": (
                "Forward-call reductions are reported separately from measured "
                "end-to-end latency; cache hits do not imply speedup."
            ),
        },
        "forward_call_accounting": forward_call_accounting,
        "condition_results": condition_results,
        "raw_timed_trials": evaluation["raw_timed_trials"],
        "execution_schedule": evaluation["execution_schedule"],
        "research_limits": [
            (
                "Synthetic generated validation episodes only; no natural-video "
                "generalization is measured."
            ),
            (
                "Feature similarity is numerical equivalence evidence, not "
                "semantic accuracy."
            ),
            (
                "DINOv2 contextual spatial tokens are cached only as complete "
                "frame outputs; no token is independently recomputed."
            ),
            (
                "Cache memory estimates include tensor payloads and exclude "
                "Python, hash, and mapping overhead."
            ),
            (
                "Measured latency is specific to this CPU, software stack, "
                "preprocessing, and workload."
            ),
            "No neural-network acceleration or semantic-performance claim is made.",
        ],
    }


def runtime_hardware(adapter: FrozenVisionEncoderAdapter) -> dict[str, Any]:
    """Return local runtime and CPU metadata without probing external services."""
    metadata: dict[str, Any] = {
        "operating_system": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "cuda_available": bool(adapter._torch.cuda.is_available()),
    }
    for package, key in (
        ("torch", "torch_version"),
        ("transformers", "transformers_version"),
        ("Pillow", "pillow_version"),
    ):
        try:
            metadata[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata[key] = None
    return metadata


def _previous_task14_runtime() -> dict[str, Any]:
    path = REPOSITORY_ROOT / "results/pretrained_stateful/run_metadata.json"
    metrics_path = REPOSITORY_ROOT / "results/pretrained_stateful/stateful_metrics.json"
    result: dict[str, Any] = {"status": "unavailable"}
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        result = {
            "python": metadata.get("hardware", {}).get("python"),
            "torch_version": metadata.get("hardware", {}).get("torch_version"),
            "transformers_version": metadata.get("hardware", {}).get(
                "transformers_version"
            ),
            "model_revision": metadata.get("model_revision"),
        }
        if metrics_path.is_file():
            result["status"] = "available"
    except (OSError, json.JSONDecodeError):
        return result
    return result


def write_evaluation_outputs(
    report: Mapping[str, Any],
    output_json: str | Path,
    output_csv: str | Path,
    figure_dir: str | Path,
) -> tuple[Path, Path, tuple[Path, Path]]:
    """Serialize report and raw timing trials, then draw dependency-light PNGs."""
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    figures = Path(figure_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for trial in report["raw_timed_trials"]:
            writer.writerow({field: trial.get(field) for field in CSV_FIELDS})
    latency_path = figures / "pretrained_cache_latency.png"
    hit_rate_path = figures / "pretrained_cache_hit_rate.png"
    draw_latency_figure(report, latency_path)
    draw_hit_rate_figure(report, hit_rate_path)
    return json_path, csv_path, (latency_path, hit_rate_path)


def draw_latency_figure(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Create mean latency bars with standard-deviation whiskers using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    conditions = report["condition_results"]
    items = list(conditions.items())
    width, height = 1400, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 100, 100, width - 70, height - 170
    draw.text(
        (left, 35), "Pretrained DINOv2 processing latency", fill="#111111", font=font
    )
    maximum = max(
        1e-9,
        max(
            max(
                entry[policy]["mean_processing_latency"]["maximum_seconds"]
                for policy in ("independent", "exact_cache")
            )
            for _, entry in items
        ),
    )
    plot_height = bottom - top
    for tick in range(6):
        y = bottom - round(plot_height * tick / 5)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text(
            (left - 70, y - 5), f"{maximum * tick / 5:.2f}s", fill="#333333", font=font
        )
    group_width = (right - left) / len(items)
    bar_width = 56
    colors = {"independent": "#4C78A8", "exact_cache": "#F58518"}
    for group_index, (name, entry) in enumerate(items):
        center = left + group_width * (group_index + 0.5)
        for policy_index, policy in enumerate(("independent", "exact_cache")):
            summary = entry[policy]["mean_processing_latency"]
            mean = summary["mean_seconds"]
            deviation = summary["sample_standard_deviation_seconds"]
            x = center + (-bar_width if policy_index == 0 else 0)
            y = bottom - round(plot_height * mean / maximum)
            draw.rectangle((x, y, x + bar_width - 8, bottom), fill=colors[policy])
            whisker_top = bottom - round(
                plot_height * min(maximum, mean + deviation) / maximum
            )
            whisker_bottom = bottom - round(
                plot_height * max(0.0, mean - deviation) / maximum
            )
            whisker_x = x + (bar_width - 8) // 2
            draw.line(
                (whisker_x, whisker_top, whisker_x, whisker_bottom),
                fill="#222222",
                width=2,
            )
            draw.line(
                (whisker_x - 8, whisker_top, whisker_x + 8, whisker_top),
                fill="#222222",
                width=2,
            )
            draw.line(
                (whisker_x - 8, whisker_bottom, whisker_x + 8, whisker_bottom),
                fill="#222222",
                width=2,
            )
        label = entry["label"]
        label_box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (center - (label_box[2] - label_box[0]) / 2, bottom + 20),
            label,
            fill="#222222",
            font=font,
        )
    legend_y = height - 75
    draw.rectangle(
        (left + 300, legend_y, left + 320, legend_y + 14), fill=colors["independent"]
    )
    draw.text((left + 328, legend_y), "Independent", fill="#222222", font=font)
    draw.rectangle(
        (left + 470, legend_y, left + 490, legend_y + 14), fill=colors["exact_cache"]
    )
    draw.text(
        (left + 498, legend_y), "Exact whole-frame cache", fill="#222222", font=font
    )
    draw.text(
        (left, height - 40),
        "Bars show mean ± sample SD; image reads excluded",
        fill="#444444",
        font=font,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def draw_hit_rate_figure(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Create exact-cache hit-rate bars; hit rate is not wall-clock speedup."""
    from PIL import Image, ImageDraw, ImageFont

    conditions = list(report["condition_results"].items())
    width, height = 1400, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 100, 100, width - 70, height - 145
    draw.text((left, 35), "Exact whole-frame cache hit rate", fill="#111111", font=font)
    for tick in range(6):
        value = tick / 5
        y = bottom - round((bottom - top) * value)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((left - 48, y - 5), f"{value:.0%}", fill="#333333", font=font)
    group_width = (right - left) / len(conditions)
    for index, (_name, entry) in enumerate(conditions):
        hit_rate = entry["exact_cache"]["cache_hit_rate"]
        center = left + group_width * (index + 0.5)
        bar_width = 80
        y = bottom - round((bottom - top) * hit_rate)
        draw.rectangle(
            (center - bar_width / 2, y, center + bar_width / 2, bottom), fill="#54A24B"
        )
        draw.text((center - 22, y - 20), f"{hit_rate:.1%}", fill="#222222", font=font)
        label = entry["label"]
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (center - (box[2] - box[0]) / 2, bottom + 18),
            label,
            fill="#222222",
            font=font,
        )
    draw.text(
        (left, height - 35),
        (
            "Cache hits avoid complete model forwards; they do not imply "
            "latency reduction."
        ),
        fill="#444444",
        font=font,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def load_baseline_settings(path: str | Path) -> dict[str, Any]:
    """Read the existing pinned adapter configuration and require its revision."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read baseline config {path}: {error}") from error
    if config.get("model_id") != DEFAULT_MODEL_ID:
        raise ValueError("evaluation requires the existing pinned DINOv2 model id")
    if config.get("revision") != DEFAULT_MODEL_REVISION:
        raise ValueError("evaluation requires the existing pinned DINOv2 revision")
    if config.get("device") != "cpu":
        raise ValueError("Task 15 evaluation is configured for CPU inference")
    return config


def make_adapter(
    baseline_config: Mapping[str, Any], *, repository_root: str | Path = REPOSITORY_ROOT
) -> FrozenVisionEncoderAdapter:
    """Load the configured checkpoint from local cache only."""
    preprocessing_values = dict(baseline_config.get("preprocessing", {}))
    for key in ("image_mean", "image_std"):
        if key in preprocessing_values:
            preprocessing_values[key] = tuple(preprocessing_values[key])
    cache_dir = baseline_config.get("cache_dir")
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = Path(repository_root) / cache_path
        cache_dir = str(cache_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return FrozenVisionEncoderAdapter.from_pretrained(
        model_id=baseline_config["model_id"],
        revision=baseline_config["revision"],
        device="cpu",
        deterministic=bool(baseline_config.get("deterministic", True)),
        preprocessing=ImagePreprocessingConfig(**preprocessing_values),
        cache_dir=cache_dir,
        allow_download=False,
    )
