"""Controlled CPU ablation of patch granularity and temporal redundancy."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import sys
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .approximate import ThresholdPatchCache
from .baseline import ManifestEpisode, load_manifest
from .encoder import RGBImage, encode_image, read_ppm
from .global_fallback import changed_pixel_fraction
from .latency import (
    EpisodeWorkload,
    audit_policy,
    deterministic_policy_orders,
    summarize_timings,
)
from .plotting import (
    write_ablation_change_frequency_png,
    write_ablation_patch_size_png,
)
from .state import StatefulPatchCache
from .synthetic import TRANSFORMATION_TYPES, TransformationType, generate_sequences

METHOD_NAMES = (
    "independent_frame_encoding",
    "exact_patch_reuse",
    "approximate_reuse_0.01",
)
PATCH_SIZES = (8, 16, 32)
CONDITION_NAMES = (
    "no_visual_changes",
    "sparse_local_changes",
    "moderate_local_changes",
    "frequent_widespread_changes",
)
EQUALITY_TOLERANCE_DEFAULT = 0.01
CSV_FIELDS = (
    "condition",
    "condition_label",
    "patch_size",
    "method",
    "number_of_frames",
    "total_patches",
    "recomputed_patches",
    "refresh_fraction",
    "feature_mae",
    "median_processing_latency_seconds",
    "mean_processing_latency_seconds",
    "processing_latency_standard_deviation_seconds",
    "p95_processing_latency_seconds",
    "relative_speedup",
    "median_latency_ratio_to_independent",
    "speed_category",
    "cache_memory_estimate_bytes",
)


@dataclass(frozen=True, slots=True)
class ScenarioEpisode:
    """Preloaded frames for one deterministic episode."""

    episode_id: str
    frames: tuple[RGBImage, ...]


@dataclass(frozen=True, slots=True)
class AblationScenario:
    """A generated validation condition and its preloaded episodes."""

    name: str
    label: str
    base_seed: int
    episode_id_prefix: str
    transformations: tuple[TransformationType, ...]
    manifest_path: Path
    episodes: tuple[ScenarioEpisode, ...]
    profile: dict[str, object]


def load_ablation_config(config_path: str | Path) -> dict[str, object]:
    """Read and validate an ablation JSON configuration."""
    path = Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read ablation config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("ablation config must be a JSON object")
    validate_ablation_config(config)
    return config


def validate_ablation_config(config: Mapping[str, object]) -> None:
    """Validate required settings and the fixed patch and method comparisons."""
    if config.get("partition") != "validation":
        raise ValueError("ablation study must use the validation partition")
    for key in ("image_width", "image_height"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 16:
            raise ValueError(f"{key} must be an integer of at least 16")
    for key in ("sequence_length", "episodes_per_condition"):
        value = config.get(key)
        minimum = 2 if key == "sequence_length" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")

    patch_sizes = config.get("patch_sizes")
    if not isinstance(patch_sizes, list) or tuple(patch_sizes) != PATCH_SIZES:
        raise ValueError(f"patch_sizes must be {list(PATCH_SIZES)} in this ablation")
    if tuple(config.get("methods", ())) != METHOD_NAMES:
        raise ValueError("method list must match the fixed ablation methods")
    _positive_int(config, "warmup_repetitions", minimum=5)
    _positive_int(config, "timed_repetitions", minimum=20)
    _positive_int(config, "policy_order_seed", minimum=0)

    threshold = config.get("approximate_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("approximate_threshold must be between 0 and 1")
    tolerance = config.get(
        "equal_latency_relative_tolerance", EQUALITY_TOLERANCE_DEFAULT
    )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or not 0.0 <= tolerance < 1.0
    ):
        raise ValueError("equal latency tolerance must be in [0, 1)")

    conditions = config.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(CONDITION_NAMES):
        raise ValueError("exactly four controlled change conditions are required")
    names: list[str] = []
    prefixes: set[str] = set()
    seed_ranges: list[range] = []
    episode_count = int(config["episodes_per_condition"])
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("each condition must be a JSON object")
        name = condition.get("name")
        if not isinstance(name, str) or name not in CONDITION_NAMES:
            raise ValueError(f"unknown ablation condition: {name!r}")
        names.append(name)
        label = condition.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"condition {name!r} requires a non-empty label")
        seed = condition.get("base_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"condition {name!r} requires a non-negative base_seed")
        prefix = condition.get("episode_id_prefix")
        if not isinstance(prefix, str) or not prefix.isascii() or not prefix:
            raise ValueError(f"condition {name!r} requires a portable ID prefix")
        if any(not (char.isalnum() or char in "_-") for char in prefix):
            raise ValueError(f"condition {name!r} has a non-portable ID prefix")
        if prefix in prefixes:
            raise ValueError("condition episode ID prefixes must be unique")
        prefixes.add(prefix)
        seed_ranges.append(range(seed, seed + episode_count))

        transformations = condition.get("transformations")
        if (
            not isinstance(transformations, list)
            or not transformations
            or any(item not in TRANSFORMATION_TYPES for item in transformations)
        ):
            raise ValueError(f"condition {name!r} has invalid transformations")

    if tuple(names) != CONDITION_NAMES:
        raise ValueError("conditions must appear in the configured canonical order")
    for index, seed_range in enumerate(seed_ranges):
        if any(
            max(seed_range.start, other.start) < min(seed_range.stop, other.stop)
            for other in seed_ranges[:index]
        ):
            raise ValueError("condition episode seed ranges must not overlap")


def generate_ablation_scenarios(
    output_dir: str | Path, config: Mapping[str, object]
) -> tuple[AblationScenario, ...]:
    """Generate separate seeded validation episodes for each change condition.

    Frame and mask creation delegates to the existing synthetic generator.
    All frames are read into memory here, before any policy timing begins.
    """
    validate_ablation_config(config)
    root = Path(output_dir)
    width = int(config["image_width"])
    height = int(config["image_height"])
    sequence_length = int(config["sequence_length"])
    episode_count = int(config["episodes_per_condition"])
    raw_conditions = config["conditions"]
    if not isinstance(raw_conditions, list):
        raise ValueError("conditions must be a list")

    scenarios: list[AblationScenario] = []
    seen_ids: set[str] = set()
    for condition in raw_conditions:
        assert isinstance(condition, dict)
        name = str(condition["name"])
        transformations = tuple(condition["transformations"])
        manifest_path = generate_sequences(
            root / name,
            width=width,
            height=height,
            sequence_length=sequence_length,
            num_episodes=episode_count,
            seed=int(condition["base_seed"]),
            transformations=transformations,
            episode_id_prefix=str(condition["episode_id_prefix"]),
        )
        episodes, transition_types, pixel_fractions = _load_scenario_manifest(
            manifest_path,
            expected_prefix=str(condition["episode_id_prefix"]),
            expected_episode_count=episode_count,
            expected_frames=sequence_length,
        )
        scenario_ids = {episode.episode_id for episode in episodes}
        if seen_ids.intersection(scenario_ids):
            raise ValueError("episode identifiers overlap across change conditions")
        seen_ids.update(scenario_ids)
        profile = _scenario_profile(
            name=name,
            transition_types=transition_types,
            changed_pixel_fractions=pixel_fractions,
        )
        scenarios.append(
            AblationScenario(
                name=name,
                label=str(condition["label"]),
                base_seed=int(condition["base_seed"]),
                episode_id_prefix=str(condition["episode_id_prefix"]),
                transformations=transformations,
                manifest_path=manifest_path,
                episodes=episodes,
                profile=profile,
            )
        )

    _validate_generated_condition_profiles(scenarios)
    return tuple(scenarios)


def run_ablation_study(
    config_path: str | Path,
    *,
    data_dir: str | Path = "data/generated/ablation_study",
    output_json: str | Path = "results/ablation_study.json",
    output_csv: str | Path = "results/ablation_study.csv",
    patch_size_figure: str | Path = "docs/figures/ablation_patch_size.png",
    change_frequency_figure: str | Path = "docs/figures/ablation_change_frequency.png",
) -> dict[str, object]:
    """Run the paired patch-size by temporal-change CPU ablation."""
    config = load_ablation_config(config_path)
    scenarios = generate_ablation_scenarios(data_dir, config)
    all_results: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    raw_groups: list[dict[str, object]] = []
    condition_index = {name: index for index, name in enumerate(CONDITION_NAMES)}

    for scenario in scenarios:
        for patch_size in PATCH_SIZES:
            workload = _build_feature_reference_workload(scenario, patch_size)
            audits = {
                method: audit_policy(workload, method, patch_size=patch_size)
                for method in METHOD_NAMES
            }
            order_seed = (
                int(config["policy_order_seed"])
                + condition_index[scenario.name] * 1_000
                + patch_size
            )
            warmups = int(config["warmup_repetitions"])
            timed_count = int(config["timed_repetitions"])
            all_orders = deterministic_policy_orders(
                METHOD_NAMES, warmups + timed_count, seed=order_seed
            )
            warmup_orders = all_orders[:warmups]
            timed_orders = all_orders[warmups:]

            for order in warmup_orders:
                for method in order:
                    _execute_policy(
                        scenario,
                        method,
                        patch_size,
                        float(config["approximate_threshold"]),
                    )

            timing_values: dict[str, list[float]] = {
                method: [] for method in METHOD_NAMES
            }
            raw_repetitions: list[dict[str, object]] = []
            for repetition_index, order in enumerate(timed_orders, start=1):
                latencies: dict[str, float] = {}
                for method in order:
                    started = time.perf_counter()
                    _execute_policy(
                        scenario,
                        method,
                        patch_size,
                        float(config["approximate_threshold"]),
                    )
                    latency = time.perf_counter() - started
                    latencies[method] = latency
                    timing_values[method].append(latency)
                raw_repetitions.append(
                    {
                        "repetition": repetition_index,
                        "policy_order": list(order),
                        "latency_seconds_by_method": latencies,
                    }
                )

            timing_summary = summarize_timings(timing_values)
            independent_median = timing_summary["independent_frame_encoding"][
                "median_latency_seconds"
            ]
            method_results: dict[str, dict[str, object]] = {}
            for method in METHOD_NAMES:
                audit = audits[method]
                timing = timing_summary[method]
                median_latency = timing["median_latency_seconds"]
                median_ratio = median_latency / independent_median
                speed_category = (
                    "reference"
                    if method == "independent_frame_encoding"
                    else _classify_speed(
                        median_ratio,
                        float(
                            config.get(
                                "equal_latency_relative_tolerance",
                                EQUALITY_TOLERANCE_DEFAULT,
                            )
                        ),
                    )
                )
                method_row: dict[str, object] = {
                    **timing,
                    "median_processing_latency_seconds": median_latency,
                    "mean_processing_latency_seconds": timing["mean_latency_seconds"],
                    "processing_latency_standard_deviation_seconds": timing[
                        "standard_deviation_seconds"
                    ],
                    "p95_processing_latency_seconds": timing["p95_latency_seconds"],
                    "number_of_frames": sum(
                        len(episode.frames) for episode in scenario.episodes
                    ),
                    "total_patches": audit.total_patch_count,
                    "recomputed_patches": audit.recomputed_patches,
                    "refresh_fraction": (
                        audit.recomputed_patches / audit.total_patch_count
                    ),
                    "feature_mae": audit.feature_mean_absolute_error,
                    "relative_speedup": timing[
                        "median_speedup_relative_to_independent"
                    ],
                    "median_latency_ratio_to_independent": median_ratio,
                    "speed_category": speed_category,
                    "cache_memory_estimate_bytes": (
                        audit.peak_cache_memory_estimate_bytes
                    ),
                }
                method_results[method] = method_row
                all_results.append(
                    {
                        "condition": scenario.name,
                        "condition_label": scenario.label,
                        "patch_size": patch_size,
                        "method": method,
                        **method_row,
                    }
                )
                if method != "independent_frame_encoding":
                    comparison_rows.append(
                        {
                            "condition": scenario.name,
                            "patch_size": patch_size,
                            "method": method,
                            "median_latency_ratio_to_independent": median_ratio,
                            "relative_speedup": method_row["relative_speedup"],
                            "speed_category": speed_category,
                        }
                    )

            raw_groups.append(
                {
                    "condition": scenario.name,
                    "patch_size": patch_size,
                    "policy_order_seed": order_seed,
                    "warmup_policy_orders": [list(order) for order in warmup_orders],
                    "raw_timing_repetitions": raw_repetitions,
                    "methods": method_results,
                }
            )

    summary_counts = _comparison_category_counts(comparison_rows)
    result: dict[str, object] = {
        "experiment_label": (
            "CPU synthetic benchmark findings: patch-size and temporal-change ablation"
        ),
        "partition": "validation",
        "test_partition_accessed": False,
        "data_directory": Path(data_dir).as_posix(),
        "configuration": config,
        "scenario_profiles": [scenario.profile for scenario in scenarios],
        "patch_granularity_definition": (
            "Each feature vector summarizes one square image patch. Smaller "
            "patches create more, more spatially localized feature vectors and "
            "more patch operations per frame; larger patches summarize coarser "
            "regions with fewer feature vectors."
        ),
        "cost_proxy_definition": (
            "Recomputed patches count actual six-value RGB-statistics feature "
            "computations. Patch comparisons and cache work are excluded from "
            "this operation count but included in measured CPU latency. This is "
            "not a neural-network FLOP estimate."
        ),
        "refresh_fraction_definition": (
            "All-frame recomputed patch count divided by all-frame total patches; "
            "the initial fully encoded frame of every episode is included."
        ),
        "feature_mae_definition": (
            "Elementwise mean absolute error across all patch feature values and "
            "frames, compared against independent encoding at the same patch size."
        ),
        "speed_category_definition": (
            "Reuse is faster if its median latency is more than the configured "
            "tolerance below independent encoding, slower if more than the "
            "tolerance above, and equal otherwise. Equal is descriptive within "
            "the tolerance, not a statistical equivalence claim."
        ),
        "comparison_summary": {
            "per_condition": comparison_rows,
            "category_counts_by_patch_size": summary_counts,
        },
        "results": all_results,
        "configuration_raw_timings": raw_groups,
        "methodology": {
            "same_input_within_condition": True,
            "all_frames_preloaded_before_timing": True,
            "timed_scope": (
                "Full in-memory policy pass over all frames of each condition, "
                "including cache creation/reset, raw-patch comparisons, cache "
                "updates, and feature extraction."
            ),
            "untimed_scope": (
                "Synthetic data generation, PPM loading, independent reference "
                "feature extraction, feature-error audits, serialization, CSV "
                "writing, plot generation, and interpreter startup."
            ),
            "repeat_interpretation": (
                "Warm-ups and timed repetitions reuse the same deterministic "
                "episodes and are not independent dataset samples."
            ),
            "relative_speedup": (
                "Median of within-repetition ratios: independent latency divided "
                "by the compared method latency."
            ),
        },
        "system": {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "operating_system": platform.platform(),
            "cpu_identifier": platform.processor() or "unknown",
            "cpu_architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "python_build": sysconfig.get_config_var("CONFIG_ARGS") or "unknown",
        },
    }
    _write_json(result, output_json)
    _write_csv(all_results, output_csv)
    write_ablation_patch_size_png(all_results, patch_size_figure)
    write_ablation_change_frequency_png(all_results, change_frequency_figure)
    return result


def _load_scenario_manifest(
    manifest_path: Path,
    *,
    expected_prefix: str,
    expected_episode_count: int,
    expected_frames: int,
) -> tuple[
    tuple[ScenarioEpisode, ...],
    tuple[str, ...],
    tuple[float, ...],
]:
    """Load and validate one generated manifest and profile rendered changes."""
    episodes: tuple[ManifestEpisode, ...] = load_manifest(manifest_path)
    if len(episodes) != expected_episode_count:
        raise ValueError("generated condition episode count does not match config")
    try:
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read generated manifest: {error}") from error
    if len(rows) != len(episodes):
        raise ValueError("generated manifest rows do not match validated episodes")

    loaded: list[ScenarioEpisode] = []
    transition_types: list[str] = []
    changed_fractions: list[float] = []
    for manifest_episode, row in zip(episodes, rows, strict=True):
        if not manifest_episode.episode_id.startswith(f"{expected_prefix}_episode_"):
            raise ValueError("generated episode ID does not match its condition")
        if len(manifest_episode.frames) != expected_frames:
            raise ValueError("generated episode frame count does not match config")
        frames = tuple(read_ppm(frame.path) for frame in manifest_episode.frames)
        transitions = row.get("transitions")
        if not isinstance(transitions, list) or len(transitions) != expected_frames - 1:
            raise ValueError("generated transition records do not match frame count")
        previous = frames[0]
        for transition, current in zip(transitions, frames[1:], strict=True):
            kind = transition.get("transformation_type")
            if not isinstance(kind, str):
                raise ValueError("generated transition lacks a transformation type")
            transition_types.append(kind)
            changed_fractions.append(changed_pixel_fraction(current, previous))
            previous = current
        loaded.append(
            ScenarioEpisode(episode_id=manifest_episode.episode_id, frames=frames)
        )
    return tuple(loaded), tuple(transition_types), tuple(changed_fractions)


def _scenario_profile(
    *,
    name: str,
    transition_types: Sequence[str],
    changed_pixel_fractions: Sequence[float],
) -> dict[str, object]:
    """Summarize the generated event schedule and realized pixel changes."""
    if not transition_types or len(transition_types) != len(changed_pixel_fractions):
        raise ValueError("condition profile requires transition observations")
    changed_count = sum(fraction > 0.0 for fraction in changed_pixel_fractions)
    active_fractions = [
        fraction for fraction in changed_pixel_fractions if fraction > 0
    ]
    counts: dict[str, int] = {}
    for kind in transition_types:
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "condition": name,
        "transition_count": len(transition_types),
        "changed_transition_count": changed_count,
        "active_transition_fraction": changed_count / len(transition_types),
        "mean_changed_pixel_fraction": statistics.fmean(changed_pixel_fractions),
        "mean_changed_pixel_fraction_when_active": (
            statistics.fmean(active_fractions) if active_fractions else 0.0
        ),
        "transformation_counts": counts,
    }


def _validate_generated_condition_profiles(
    scenarios: Sequence[AblationScenario],
) -> None:
    """Reject generated samples that fail their intended temporal profile."""
    profiles = {scenario.name: scenario.profile for scenario in scenarios}
    no_change = profiles["no_visual_changes"]
    sparse = profiles["sparse_local_changes"]
    moderate = profiles["moderate_local_changes"]
    widespread = profiles["frequent_widespread_changes"]
    if no_change["changed_transition_count"] != 0:
        raise ValueError("no-change condition rendered one or more changed frames")
    sparse_count = int(sparse["changed_transition_count"])
    sparse_total = int(sparse["transition_count"])
    if not 0 < sparse_count < sparse_total:
        raise ValueError(
            "sparse-local condition must include changed and unchanged transitions"
        )
    for profile, name in ((moderate, "moderate-local"), (widespread, "widespread")):
        if profile["changed_transition_count"] != profile["transition_count"]:
            raise ValueError(f"{name} condition must change every transition")
    if float(moderate["mean_changed_pixel_fraction"]) >= 0.5:
        raise ValueError("moderate-local changes must remain spatially local")
    if float(widespread["mean_changed_pixel_fraction"]) < 0.5:
        raise ValueError(
            "widespread condition must alter at least half of frame pixels"
        )


def _build_feature_reference_workload(
    scenario: AblationScenario, patch_size: int
) -> tuple[EpisodeWorkload, ...]:
    """Precompute independent reference features before method timing."""
    return tuple(
        EpisodeWorkload(
            episode_id=episode.episode_id,
            frames=episode.frames,
            reference_features=tuple(
                encode_image(frame, patch_size=patch_size).features
                for frame in episode.frames
            ),
        )
        for episode in scenario.episodes
    )


def _execute_policy(
    scenario: AblationScenario,
    method: str,
    patch_size: int,
    approximate_threshold: float,
) -> None:
    """Process one condition pass, resetting each stateful cache per episode."""
    if method not in METHOD_NAMES:
        raise ValueError(f"unknown ablation method: {method}")
    for episode in scenario.episodes:
        cache: StatefulPatchCache | ThresholdPatchCache | None
        if method == "exact_patch_reuse":
            cache = StatefulPatchCache(patch_size=patch_size)
        elif method == "approximate_reuse_0.01":
            cache = ThresholdPatchCache(
                threshold=approximate_threshold, patch_size=patch_size
            )
        else:
            cache = None
        for frame in episode.frames:
            if method == "independent_frame_encoding":
                encode_image(frame, patch_size=patch_size)
            else:
                if cache is None:
                    raise RuntimeError("stateful ablation method has no cache")
                cache.process_frame(frame)


def _classify_speed(relative_latency: float, tolerance: float) -> str:
    """Classify speed relative to independent encoding with a fixed band."""
    if relative_latency < 1.0 - tolerance:
        return "faster"
    if relative_latency > 1.0 + tolerance:
        return "slower"
    return "equal"


def _comparison_category_counts(
    comparisons: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Count faster/equal/slower cases by patch size and cache policy."""
    counts: list[dict[str, object]] = []
    for patch_size in PATCH_SIZES:
        for method in METHOD_NAMES[1:]:
            relevant = [
                row
                for row in comparisons
                if row["patch_size"] == patch_size and row["method"] == method
            ]
            counts.append(
                {
                    "patch_size": patch_size,
                    "method": method,
                    "faster": sum(
                        row["speed_category"] == "faster" for row in relevant
                    ),
                    "equal": sum(row["speed_category"] == "equal" for row in relevant),
                    "slower": sum(
                        row["speed_category"] == "slower" for row in relevant
                    ),
                    "condition_count": len(relevant),
                }
            )
    return counts


def _write_json(result: Mapping[str, object], output_path: str | Path) -> None:
    """Write deterministic sorted JSON output."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> None:
    """Write one row per policy and patch-size/condition configuration."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def _positive_int(config: Mapping[str, object], name: str, *, minimum: int) -> int:
    """Read an integer configuration value with a minimum constraint."""
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value
