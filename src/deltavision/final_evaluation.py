"""Prespecified final evaluation of handcrafted and pretrained reuse policies."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .approximate import ThresholdPatchCache, compare_feature_vectors
from .baseline import load_manifest
from .encoder import RGBImage, encode_image
from .evaluation import ConfusionCounts, count_patch_decisions, detection_metrics
from .global_fallback import GlobalFallbackPatchCache
from .pretrained import RepresentationKind
from .pretrained_cache_evaluation import (
    compare_policy_features,
    load_baseline_settings,
    load_evaluation_config,
    make_adapter,
    preload_validation_episodes,
    run_policy_pass,
    runtime_hardware,
    timing_summary,
)
from .research_benchmark import (
    _EpisodeInput,
    _load_episode_input,
    _read_jsonl_rows,
    load_research_config,
)
from .state import StatefulPatchCache

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FINAL_CONFIG_PATH = REPOSITORY_ROOT / "configs/final_evaluation.json"
FINAL_OUTPUT_PATH = REPOSITORY_ROOT / "results/final_evaluation.json"


def load_final_config(path: str | Path = FINAL_CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen Task 17 settings and reject policy or provenance drift."""
    config_path = Path(path)
    try:
        raw_bytes = config_path.read_bytes()
        config = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read final evaluation config: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("final evaluation config must be a JSON object")
    validate_final_config(config)
    config["_path"] = str(config_path.resolve())
    config["_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    return config


def validate_final_config(config: Mapping[str, Any]) -> None:
    """Require the validation-derived policies frozen before final evaluation."""
    if config.get("status") != "prespecified_before_task17_final_evaluation":
        raise ValueError("final evaluation config is not marked as prespecified")
    if config.get("final_partition") != "previously_inspected_held_out_test":
        raise ValueError("final split must be identified as previously inspected")
    exposure = config.get("prior_test_exposure", {})
    if exposure.get("status") != "known_previous_inspection":
        raise ValueError("known prior held-out inspection must be documented")
    handcrafted = config.get("handcrafted", {})
    methods = handcrafted.get("methods", [])
    expected = [
        ("independent_encoding", "independent_frame_encoding", None),
        ("exact_patch_reuse", "exact_raw_patch_equality", None),
        (
            "approximate_patch_reuse_0.01",
            "normalized_mean_absolute_rgb_difference",
            0.01,
        ),
        (
            "global_change_fallback",
            "local_threshold_with_global_changed_pixel_fraction_fallback",
            None,
        ),
    ]
    observed = [
        (item.get("name"), item.get("policy"), item.get("threshold"))
        for item in methods
    ]
    if observed != expected:
        raise ValueError("handcrafted policies differ from frozen Task 17 methods")
    fallback = methods[-1]
    if (
        fallback.get("local_threshold") != 0.05
        or fallback.get("global_change_threshold") != 0.5
    ):
        raise ValueError("global fallback thresholds differ from validation settings")
    if handcrafted.get("patch_size") != 16:
        raise ValueError("handcrafted patch size must remain 16 pixels")
    pretrained = config.get("pretrained_dinov2", {})
    if pretrained.get("partition") != "validation":
        raise ValueError("pretrained workload must use validation data only")
    if pretrained.get("cache_capacity_per_episode") != 8:
        raise ValueError("DINOv2 capacity must remain at the prespecified value 8")
    if pretrained.get("allow_checkpoint_downloads") is not False:
        raise ValueError("DINOv2 checkpoint downloads must be disabled")
    stats = config.get("statistics", {})
    if (
        stats.get("sampling_unit") != "episode"
        or stats.get("method") != "percentile_cluster_bootstrap"
        or stats.get("replicates", 0) < 1
        or not 0.0 < stats.get("confidence_level", 0.0) < 1.0
        or stats.get("paired_comparisons")
        != (
            "Use identical resampled episode indices for every method within a "
            "partition or DINOv2 condition."
        )
    ):
        raise ValueError("episode-level paired bootstrap settings are invalid")
    for section in (handcrafted, pretrained):
        if (
            section.get(
                "warmup_repetitions_per_policy",
                section.get("warmup_repetitions_per_policy_condition_episode", 3),
            )
            < 0
        ):
            raise ValueError("warmup repetitions cannot be negative")
        if (
            section.get(
                "timed_repetitions_per_policy",
                section.get("timed_repetitions_per_policy_condition_episode", 5),
            )
            < 1
        ):
            raise ValueError("at least one timed repetition is required")


def validate_final_manifest_provenance(
    manifest_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_partition: str,
    expected_count: int,
    expected_seed_start: int,
) -> None:
    """Validate only the requested manifest's partition, IDs, and seed range."""
    manifest = Path(manifest_path).resolve()
    if manifest.name != "manifest.jsonl":
        raise ValueError("evaluation input must be a manifest.jsonl file")
    if expected_partition == "test":
        if manifest.parent.name != "test":
            raise ValueError("handcrafted final evaluation requires the test manifest")
    elif expected_partition == "validation":
        if "validation" not in {part.lower() for part in manifest.parts}:
            raise ValueError(
                "pretrained final evaluation requires validation manifests"
            )
        if "test" in {part.lower() for part in manifest.parts}:
            raise ValueError("held-out test paths are excluded from pretrained runs")
    else:
        raise ValueError("unsupported evaluation partition")
    if len(rows) != expected_count:
        raise ValueError(
            f"{expected_partition} manifest has {len(rows)} episodes; "
            f"expected {expected_count}"
        )
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for index, row in enumerate(rows):
        episode_id = row.get("episode_id")
        seed = row.get("random_seed")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("manifest contains an invalid episode_id")
        if episode_id in seen_ids:
            raise ValueError("manifest contains duplicate episode IDs")
        seen_ids.add(episode_id)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("manifest contains an invalid random_seed")
        if seed in seen_seeds:
            raise ValueError("manifest contains duplicate random seeds")
        seen_seeds.add(seed)
        if seed != expected_seed_start + index:
            raise ValueError("manifest seed range differs from frozen provenance")
        prefix = f"{expected_partition}_" if expected_partition == "test" else None
        if prefix is not None and not episode_id.startswith(prefix):
            raise ValueError("test manifest contains a non-test episode ID")


def _runtime_metadata() -> dict[str, Any]:
    data: dict[str, Any] = {
        "operating_system": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    for package, key in (
        ("torch", "torch_version"),
        ("transformers", "transformers_version"),
        ("Pillow", "pillow_version"),
    ):
        try:
            data[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            data[key] = None
    return data


def _load_handcrafted_episodes(
    config: Mapping[str, Any],
) -> tuple[tuple[_EpisodeInput, ...], dict[str, Any]]:
    research_config_path = _resolve(config["research_config"])
    research_config = load_research_config(research_config_path)
    manifest_path = _resolve(config["final_manifest"])
    expected_manifest = (
        _resolve(config["research_data_root"]) / "test" / "manifest.jsonl"
    )
    if manifest_path.resolve() != expected_manifest.resolve():
        raise ValueError(
            "final handcrafted manifest differs from the frozen test partition"
        )
    expected = research_config.splits["test"]
    rows = _read_jsonl_rows(manifest_path)
    validate_final_manifest_provenance(
        manifest_path,
        rows,
        expected_partition="test",
        expected_count=expected.episode_count,
        expected_seed_start=expected.base_seed,
    )
    episodes = load_manifest(manifest_path)
    if len(episodes) != len(rows):
        raise ValueError("test manifest frame records and metadata do not match")
    split_root = manifest_path.parent
    prepared = tuple(
        _load_episode_input(
            split_root,
            episode,
            dict(row),
            width=research_config.width,
            height=research_config.height,
            sequence_length=research_config.sequence_length,
            patch_size=research_config.patch_size,
        )
        for episode, row in zip(episodes, rows, strict=True)
    )
    return prepared, {
        "manifest": str(manifest_path),
        "partition": "previously_inspected_held_out_test",
        "episode_count": len(prepared),
        "frame_count": sum(len(item.frames) for item in prepared),
        "transition_count": sum(len(item.transition_labels) for item in prepared),
        "random_seed_range": [
            expected.base_seed,
            expected.base_seed + expected.episode_count - 1,
        ],
        "episode_random_seeds": [
            {"episode_id": row["episode_id"], "random_seed": row["random_seed"]}
            for row in rows
        ],
        "image_dimensions": [research_config.width, research_config.height],
        "patch_size": research_config.patch_size,
        "sequence_length": research_config.sequence_length,
        "benchmark_config_sha256": research_config.sha256,
        "prior_test_exposure": config["prior_test_exposure"],
    }


def _handcrafted_process_episode(
    episode: _EpisodeInput, method: Mapping[str, Any], patch_size: int
) -> tuple[
    float, list[tuple[tuple[tuple[float, ...], ...], tuple[bool, ...], int, int, int]]
]:
    name = method["name"]
    if name == "independent_encoding":
        cache: Any = None
    elif name == "exact_patch_reuse":
        cache = StatefulPatchCache(patch_size=patch_size)
    elif name == "approximate_patch_reuse_0.01":
        cache = ThresholdPatchCache(float(method["threshold"]), patch_size=patch_size)
    elif name == "global_change_fallback":
        cache = GlobalFallbackPatchCache(
            local_threshold=float(method["local_threshold"]),
            global_change_threshold=float(method["global_change_threshold"]),
            patch_size=patch_size,
        )
    else:
        raise ValueError(f"unknown handcrafted policy: {name}")

    outputs = []
    started = time.perf_counter()
    for frame in episode.frames:
        if name == "independent_encoding":
            encoded = encode_image(frame, patch_size=patch_size)
            outputs.append(
                (
                    encoded.features,
                    tuple(True for _ in encoded.features),
                    len(encoded.features),
                    len(encoded.features),
                    0,
                )
            )
            continue
        frame_result = cache.process_frame(frame)
        encoding = getattr(frame_result, "encoding", frame_result)
        outputs.append(
            (
                encoding.features,
                encoding.refresh_decisions,
                encoding.patch_count,
                encoding.recomputed_patches,
                encoding.reused_patches,
            )
        )
    return time.perf_counter() - started, outputs


def _empty_sufficient() -> dict[str, Any]:
    return {
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "transition_patch_count": 0,
        "transition_recomputed_patches": 0,
        "total_patch_count": 0,
        "recomputed_patches": 0,
        "reused_patches": 0,
        "feature_absolute_error_sum": 0.0,
        "feature_value_count": 0,
        "maximum_absolute_feature_error": 0.0,
    }


def _accumulate_episode_metrics(
    episode: _EpisodeInput,
    outputs: Sequence[
        tuple[tuple[tuple[float, ...], ...], tuple[bool, ...], int, int, int]
    ],
    patch_size: int,
) -> dict[str, Any]:
    aggregate = _empty_sufficient()
    by_transformation: dict[str, dict[str, Any]] = {}
    for frame_index, (output, baseline) in enumerate(
        zip(outputs, episode.baseline_features, strict=True)
    ):
        features, decisions, patch_count, recomputed, reused = output
        feature_error = compare_feature_vectors(features, baseline)
        aggregate["total_patch_count"] += patch_count
        aggregate["recomputed_patches"] += recomputed
        aggregate["reused_patches"] += reused
        aggregate["feature_absolute_error_sum"] += (
            feature_error.mean_absolute_error * feature_error.feature_value_count
        )
        aggregate["feature_value_count"] += feature_error.feature_value_count
        aggregate["maximum_absolute_feature_error"] = max(
            aggregate["maximum_absolute_feature_error"],
            feature_error.maximum_absolute_error,
        )
        if frame_index == 0:
            continue
        transition_index = frame_index - 1
        transformation = episode.transformation_types[transition_index]
        actual = episode.transition_labels[transition_index]
        counts = count_patch_decisions(actual, decisions)
        target = by_transformation.setdefault(transformation, _empty_sufficient())
        for key in (
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        ):
            aggregate[key] += getattr(counts, key)
            target[key] += getattr(counts, key)
        aggregate["transition_patch_count"] += counts.total
        aggregate["transition_recomputed_patches"] += recomputed
        target["transition_patch_count"] += counts.total
        target["transition_recomputed_patches"] += recomputed
        destination_features = compare_feature_vectors(features, baseline)
        target["feature_absolute_error_sum"] += (
            destination_features.mean_absolute_error
            * destination_features.feature_value_count
        )
        target["feature_value_count"] += destination_features.feature_value_count
        target["maximum_absolute_feature_error"] = max(
            target["maximum_absolute_feature_error"],
            destination_features.maximum_absolute_error,
        )
    return {
        "episode_id": episode.episode_id,
        "aggregate": aggregate,
        "by_transformation": by_transformation,
    }


def _sum_sufficient(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = _empty_sufficient()
    for record in records:
        for key in result:
            if key == "maximum_absolute_feature_error":
                result[key] = max(result[key], record.get(key, 0.0))
            else:
                result[key] += record.get(key, 0)
    return result


def handcrafted_metric_view(sufficient: Mapping[str, Any]) -> dict[str, Any]:
    """Convert episode sufficient statistics to the requested metric set."""
    counts = ConfusionCounts(
        int(sufficient["true_positives"]),
        int(sufficient["false_positives"]),
        int(sufficient["true_negatives"]),
        int(sufficient["false_negatives"]),
    )
    detection = detection_metrics(counts)
    transition_count = int(sufficient["transition_patch_count"])
    all_patch_count = int(sufficient["total_patch_count"])
    positives = counts.true_positives + counts.false_negatives
    result = {
        **detection,
        "missed_change_rate": detection["missed_change_rate"],
        "false_refresh_rate": detection["false_refresh_rate"],
        "transition_refresh_fraction": (
            sufficient["transition_recomputed_patches"] / transition_count
            if transition_count
            else 0.0
        ),
        "all_frame_refresh_fraction": (
            sufficient["recomputed_patches"] / all_patch_count
            if all_patch_count
            else 0.0
        ),
        "feature_mean_absolute_error": (
            sufficient["feature_absolute_error_sum"] / sufficient["feature_value_count"]
            if sufficient["feature_value_count"]
            else 0.0
        ),
        "maximum_absolute_feature_error": sufficient["maximum_absolute_feature_error"],
        "transition_patch_count": transition_count,
        "positive_transition_patches": positives,
        "transition_recomputed_patches": sufficient["transition_recomputed_patches"],
        "total_patch_count": all_patch_count,
        "recomputed_patches": sufficient["recomputed_patches"],
        "reused_patches": sufficient["reused_patches"],
        "feature_value_count": sufficient["feature_value_count"],
    }
    return result


def _run_handcrafted(
    episodes: Sequence[_EpisodeInput], config: Mapping[str, Any]
) -> dict[str, Any]:
    section = config["handcrafted"]
    method_specs = section["methods"]
    warmups = int(section["warmup_repetitions_per_policy"])
    repetitions = int(section["timed_repetitions_per_policy"])
    rng = random.Random(int(section["policy_order_seed"]))
    raw_trials: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    method_episode_records: dict[str, list[dict[str, Any]]] = {
        method["name"]: [] for method in method_specs
    }
    latency_by_method: dict[str, list[float]] = {
        method["name"]: [] for method in method_specs
    }

    for phase, count in (("warmup", warmups), ("timed", repetitions)):
        for repetition in range(1, count + 1):
            order = [method["name"] for method in method_specs]
            rng.shuffle(order)
            schedule.append(
                {"phase": phase, "repetition": repetition, "policy_order": order}
            )
            for order_index, name in enumerate(order, start=1):
                method = next(item for item in method_specs if item["name"] == name)
                workload_seconds = 0.0
                for episode in episodes:
                    elapsed, outputs = _handcrafted_process_episode(
                        episode, method, int(section["patch_size"])
                    )
                    workload_seconds += elapsed
                    if phase == "timed":
                        raw_trials.append(
                            {
                                "phase": phase,
                                "method": name,
                                "repetition": repetition,
                                "execution_order": order_index,
                                "episode_id": episode.episode_id,
                                "processing_latency_seconds": elapsed,
                            }
                        )
                        if repetition == 1:
                            record = _accumulate_episode_metrics(
                                episode, outputs, int(section["patch_size"])
                            )
                            record["metrics"] = handcrafted_metric_view(
                                record["aggregate"]
                            )
                            record["by_transformation_metrics"] = {
                                category: handcrafted_metric_view(sufficient)
                                for category, sufficient in record[
                                    "by_transformation"
                                ].items()
                            }
                            method_episode_records[name].append(record)
                if phase == "timed":
                    latency_by_method[name].append(workload_seconds)
                    raw_trials.append(
                        {
                            "phase": "timed_workload_total",
                            "method": name,
                            "repetition": repetition,
                            "execution_order": order_index,
                            "episode_id": None,
                            "processing_latency_seconds": workload_seconds,
                        }
                    )

    methods: dict[str, Any] = {}
    for method in method_specs:
        name = method["name"]
        records = method_episode_records[name]
        aggregate = _sum_sufficient([record["aggregate"] for record in records])
        methods[name] = {
            "configuration": {
                key: value
                for key, value in method.items()
                if key != "selection_provenance"
            },
            "aggregate": handcrafted_metric_view(aggregate),
            "per_episode": records,
            "timing": timing_summary(latency_by_method[name]),
        }
    return {
        "episodes": len(episodes),
        "frames": sum(len(episode.frames) for episode in episodes),
        "transitions": sum(len(episode.transition_labels) for episode in episodes),
        "methods": methods,
        "raw_timed_trials": raw_trials,
        "execution_schedule": schedule,
        "timing_scope": section["timing_scope"],
        "metric_scope": {
            "classification": (
                "Transition-only patch labels; first episode frame is excluded."
            ),
            "computation_and_feature_error": (
                "All frames including episode initialization; RGB-statistics "
                "patch operations are not neural forward calls."
            ),
            "ground_truth": (
                "Rendered pixel-change masks aggregated to a patch if any "
                "in-bounds pixel changed; not semantic labels."
            ),
            "independent_encoding": (
                "Every transition patch is predicted refreshed, so recall is 1 "
                "when positives exist and false-refresh behavior reflects "
                "unchanged patches."
            ),
        },
    }


def _run_pretrained(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    section = config["pretrained_dinov2"]
    evaluation_config_path = _resolve(section["evaluation_config"])
    evaluation_config = load_evaluation_config(evaluation_config_path)
    baseline_config_path = _resolve(section["baseline_config"])
    baseline_config = load_baseline_settings(baseline_config_path)
    if evaluation_config["cache_capacity"] != section["cache_capacity_per_episode"]:
        raise ValueError(
            "Task 17 DINO capacity differs from the existing validation config"
        )
    if (
        evaluation_config["warmup_repetitions"]
        != section["warmup_repetitions_per_policy_condition_episode"]
    ):
        raise ValueError("DINO warmup count differs from its frozen evaluation config")
    if (
        evaluation_config["timed_repetitions"]
        != section["timed_repetitions_per_policy_condition_episode"]
    ):
        raise ValueError(
            "DINO timed repetition count differs from its frozen evaluation config"
        )
    validation_root = _resolve(evaluation_config["validation_root"])
    scenarios: dict[str, tuple[Path, tuple[tuple[str, tuple[RGBImage, ...]], ...]]] = {}
    scenario_seeds: dict[str, dict[str, int]] = {}
    for condition in evaluation_config["conditions"]:
        manifest = validation_root / condition["name"] / "manifest.jsonl"
        rows = _read_jsonl_rows(manifest)
        validate_final_manifest_provenance(
            manifest,
            rows,
            expected_partition="validation",
            expected_count=evaluation_config["num_episodes"],
            expected_seed_start=evaluation_config["widespread_seed"]
            if condition["kind"] == "brightness_sweep"
            else evaluation_config["source_seed"],
        )
        episodes = preload_validation_episodes(manifest)
        if any(
            len(images) != evaluation_config["sequence_length"]
            for _, images in episodes
        ):
            raise ValueError("DINO validation episode frame count differs from config")
        if [episode_id for episode_id, _ in episodes] != [
            row["episode_id"] for row in rows
        ]:
            raise ValueError("DINO validation frame IDs and manifest metadata differ")
        scenarios[condition["name"]] = (manifest, episodes)
        scenario_seeds[condition["name"]] = {
            row["episode_id"]: row["random_seed"] for row in rows
        }

    adapter = make_adapter(baseline_config, repository_root=REPOSITORY_ROOT)
    model = adapter.model
    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    if getattr(model, "training", True) or any(parameter_flags):
        raise ValueError(
            "DINOv2 model is not in evaluation mode with frozen parameters"
        )
    representation: RepresentationKind = baseline_config["representation"]
    rng = random.Random(int(section["policy_order_seed"]))
    warmups = int(section["warmup_repetitions_per_policy_condition_episode"])
    repetitions = int(section["timed_repetitions_per_policy_condition_episode"])
    cache_capacity = int(section["cache_capacity_per_episode"])
    raw_trials: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    conditions: dict[str, Any] = {}

    for condition in evaluation_config["conditions"]:
        name = condition["name"]
        manifest, all_episodes = scenarios[name]
        per_episode: list[dict[str, Any]] = []
        episode_times: dict[str, dict[str, list[float]]] = {}
        episode_passes: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for episode_id, images in all_episodes:
            episode_times[episode_id] = {"independent": [], "exact_cache": []}
            episode_passes[episode_id] = {"independent": [], "exact_cache": []}

        for phase, count in (("warmup", warmups), ("timed", repetitions)):
            for repetition in range(1, count + 1):
                for episode_id, images in all_episodes:
                    order = ["independent", "exact_cache"]
                    rng.shuffle(order)
                    schedule.append(
                        {
                            "condition": name,
                            "episode_id": episode_id,
                            "phase": phase,
                            "repetition": repetition,
                            "policy_order": order,
                        }
                    )
                    for order_index, policy in enumerate(order, start=1):
                        result = run_policy_pass(
                            adapter,
                            ((episode_id, images),),
                            policy=policy,
                            representation=representation,
                            cache_capacity=cache_capacity,
                        )
                        if phase == "timed":
                            episode_times[episode_id][policy].append(
                                result["processing_latency_seconds"]
                            )
                            episode_passes[episode_id][policy].append(result)
                            raw_trials.append(
                                {
                                    "condition": name,
                                    "condition_label": condition["label"],
                                    "episode_id": episode_id,
                                    "phase": phase,
                                    "repetition": repetition,
                                    "execution_order": order_index,
                                    **result,
                                }
                            )
        for episode_id, images in all_episodes:
            independent = episode_passes[episode_id]["independent"][0]
            cached = episode_passes[episode_id]["exact_cache"][0]
            equivalence = compare_policy_features(
                adapter,
                ((episode_id, images),),
                representation=representation,
                cache_capacity=cache_capacity,
                maximum_absolute_difference_tolerance=float(
                    evaluation_config["maximum_absolute_difference_tolerance"]
                ),
                minimum_cosine_similarity_tolerance=float(
                    evaluation_config["minimum_cosine_similarity_tolerance"]
                ),
            )
            if not equivalence["within_documented_tolerance"]:
                raise ValueError(
                    f"DINOv2 cache output failed feature tolerance: {episode_id}"
                )
            per_episode.append(
                {
                    "episode_id": episode_id,
                    "random_seed": scenario_seeds[name][episode_id],
                    "total_frames": independent["total_frames"],
                    "independent_forward_calls": independent["model_forward_calls"],
                    "stateful_forward_calls": cached["model_forward_calls"],
                    "cache_hits": cached["cached_frame_hits"],
                    "cache_misses": cached["cache_misses"],
                    "cache_hit_rate": cached["cache_hit_rate"],
                    "forward_call_reduction": (
                        independent["model_forward_calls"]
                        - cached["model_forward_calls"]
                    )
                    / independent["model_forward_calls"],
                    "independent_timing": timing_summary(
                        episode_times[episode_id]["independent"]
                    ),
                    "cached_timing": timing_summary(
                        episode_times[episode_id]["exact_cache"]
                    ),
                    "peak_cache_memory_estimate_bytes": cached[
                        "peak_cache_memory_estimate_bytes"
                    ],
                    "feature_shape": cached["feature_shape"],
                    "feature_equivalence": equivalence,
                }
            )

        condition_raw = [item for item in raw_trials if item["condition"] == name]
        raw_by_policy_rep: dict[str, dict[int, float]] = {
            "independent": {},
            "exact_cache": {},
        }
        for row in condition_raw:
            raw_by_policy_rep[row["policy"]].setdefault(row["repetition"], 0.0)
            raw_by_policy_rep[row["policy"]][row["repetition"]] += row[
                "processing_latency_seconds"
            ]
        independent_totals = [
            raw_by_policy_rep["independent"][repetition]
            for repetition in range(1, repetitions + 1)
        ]
        cached_totals = [
            raw_by_policy_rep["exact_cache"][repetition]
            for repetition in range(1, repetitions + 1)
        ]
        total_frames = sum(item["total_frames"] for item in per_episode)
        independent_calls = sum(
            item["independent_forward_calls"] for item in per_episode
        )
        cached_calls = sum(item["stateful_forward_calls"] for item in per_episode)
        cache_hits = sum(item["cache_hits"] for item in per_episode)
        conditions[name] = {
            "label": condition["label"],
            "manifest": str(manifest),
            "episodes": len(per_episode),
            "random_seeds": [item["random_seed"] for item in per_episode],
            "total_frames_per_pass": total_frames,
            "independent_forward_calls_per_pass": independent_calls,
            "stateful_forward_calls_per_pass": cached_calls,
            "cached_frame_hits_per_pass": cache_hits,
            "cache_misses_per_pass": cached_calls,
            "cache_hit_rate": cache_hits / total_frames if total_frames else 0.0,
            "forward_call_reduction": (independent_calls - cached_calls)
            / independent_calls
            if independent_calls
            else 0.0,
            "independent_processing_latency": timing_summary(independent_totals),
            "cached_processing_latency": timing_summary(cached_totals),
            "median_speedup": statistics.median(independent_totals)
            / statistics.median(cached_totals)
            if statistics.median(cached_totals)
            else None,
            "cache_memory_peak_bytes": max(
                item["peak_cache_memory_estimate_bytes"] for item in per_episode
            ),
            "per_episode": per_episode,
            "feature_equivalence": {
                "all_within_documented_tolerance": all(
                    item["feature_equivalence"]["within_documented_tolerance"]
                    for item in per_episode
                ),
                "maximum_absolute_difference": max(
                    item["feature_equivalence"]["maximum_absolute_difference"]
                    for item in per_episode
                ),
                "mean_absolute_difference": statistics.fmean(
                    item["feature_equivalence"]["mean_absolute_difference"]
                    for item in per_episode
                ),
                "minimum_cosine_similarity": min(
                    item["feature_equivalence"]["cosine_similarity"]
                    for item in per_episode
                ),
            },
        }

    model_metadata = {
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "device": adapter.device,
        "deterministic": adapter.deterministic,
        "representation": representation,
        "preprocessing": adapter.preprocessing.to_dict(),
        "model_evaluation_mode": not bool(model.training),
        "all_parameters_frozen": not any(parameter_flags),
        "feature_dimensions_by_condition": {
            name: result["per_episode"][0]["feature_shape"]
            for name, result in conditions.items()
        },
        "checkpoint_downloads_allowed": False,
    }
    report = {
        "status": "complete",
        "partition": "validation",
        "evaluation_config": str(evaluation_config_path),
        "baseline_config": str(baseline_config_path),
        "configurations": {
            "evaluation": {
                "path": str(evaluation_config_path),
                "sha256": hashlib.sha256(
                    evaluation_config_path.read_bytes()
                ).hexdigest(),
                "values": evaluation_config,
            },
            "baseline": {
                "path": str(baseline_config_path),
                "sha256": hashlib.sha256(baseline_config_path.read_bytes()).hexdigest(),
                "values": baseline_config,
            },
        },
        "model": model_metadata,
        "episode_count_per_condition": evaluation_config["num_episodes"],
        "frames_per_episode": evaluation_config["sequence_length"],
        "capacity_per_episode": cache_capacity,
        "warmup_repetitions_per_policy_episode": warmups,
        "timed_repetitions_per_policy_episode": repetitions,
        "timing_scope": section["timing_scope"],
        "conditions": conditions,
        "raw_timed_trials": raw_trials,
        "execution_schedule": schedule,
        "limitations": [
            (
                "Synthetic validation workload only; does not establish "
                "natural-video generalization."
            ),
            (
                "Exact whole-frame caching reuses complete representations and "
                "performs no selective DINOv2 token computation."
            ),
            "Representation equality does not measure semantic accuracy.",
            (
                "When all preprocessed inputs differ, every frame requires a "
                "full model forward; preprocessing, hashing, comparison, and "
                "storage remain overhead."
            ),
            (
                "Cache payload estimates exclude Python objects, allocator "
                "overhead, and mapping overhead."
            ),
        ],
    }
    return report, runtime_hardware(adapter)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _write_csv(result: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "representation_family",
        "partition",
        "condition",
        "method",
        "episode_id",
        "policy",
        "precision",
        "recall",
        "f1",
        "missed_change_rate",
        "false_refresh_rate",
        "transition_refresh_fraction",
        "all_frame_refresh_fraction",
        "feature_mean_absolute_error",
        "maximum_absolute_feature_error",
        "total_frames",
        "independent_forward_calls",
        "stateful_forward_calls",
        "cache_hits",
        "cache_hit_rate",
        "forward_call_reduction",
        "processing_latency_seconds_median",
        "peak_cache_memory_estimate_bytes",
        "feature_maximum_absolute_difference",
        "feature_cosine_similarity",
    )
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        handcrafted = result["handcrafted"]
        for method, summary in handcrafted["methods"].items():
            for episode in summary["per_episode"]:
                metrics = episode["metrics"]
                writer.writerow(
                    {
                        "representation_family": "handcrafted_rgb_statistics",
                        "partition": "previously_inspected_held_out_test",
                        "condition": "all_transformations",
                        "method": method,
                        "episode_id": episode["episode_id"],
                        **{
                            key: metrics.get(key, "")
                            for key in fields
                            if key in metrics
                        },
                    }
                )
        pretrained = result["pretrained_dinov2"]
        for condition, condition_result in pretrained["conditions"].items():
            for episode in condition_result["per_episode"]:
                for policy in ("independent", "exact_cache"):
                    is_cache = policy == "exact_cache"
                    equivalence = episode["feature_equivalence"]
                    writer.writerow(
                        {
                            "representation_family": "pretrained_dinov2_spatial_tokens",
                            "partition": "validation",
                            "condition": condition,
                            "method": policy,
                            "episode_id": episode["episode_id"],
                            "policy": policy,
                            "total_frames": episode["total_frames"],
                            "independent_forward_calls": episode[
                                "independent_forward_calls"
                            ],
                            "stateful_forward_calls": episode["stateful_forward_calls"]
                            if is_cache
                            else episode["independent_forward_calls"],
                            "cache_hits": episode["cache_hits"] if is_cache else 0,
                            "cache_hit_rate": episode["cache_hit_rate"]
                            if is_cache
                            else 0.0,
                            "forward_call_reduction": episode["forward_call_reduction"]
                            if is_cache
                            else 0.0,
                            "processing_latency_seconds_median": (
                                episode["cached_timing"]["median_seconds"]
                                if is_cache
                                else episode["independent_timing"]["median_seconds"]
                            ),
                            "peak_cache_memory_estimate_bytes": episode[
                                "peak_cache_memory_estimate_bytes"
                            ]
                            if is_cache
                            else 0,
                            "feature_maximum_absolute_difference": equivalence[
                                "maximum_absolute_difference"
                            ],
                            "feature_cosine_similarity": equivalence[
                                "cosine_similarity"
                            ],
                        }
                    )


def write_evaluation_artifacts(
    result: Mapping[str, Any], json_path: str | Path, csv_path: str | Path
) -> None:
    """Serialize the machine-readable evaluation record and its episode CSV."""
    output_json = Path(json_path)
    output_csv = Path(csv_path)
    if output_json.resolve() == output_csv.resolve():
        raise ValueError("evaluation JSON and CSV paths must differ")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(result, output_csv)


def run_final_evaluation(
    config_path: str | Path = FINAL_CONFIG_PATH,
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Run Task 17 with frozen settings and write actual raw evaluation outputs."""
    config = load_final_config(config_path)
    research_config = load_research_config(_resolve(config["research_config"]))
    handcrafted_episodes, handcrafted_provenance = _load_handcrafted_episodes(config)
    handcrafted = _run_handcrafted(handcrafted_episodes, config)
    try:
        pretrained, pretrained_runtime = _run_pretrained(config)
    except Exception as error:
        # Keep completed handcrafted outputs usable if the local checkpoint or
        # optional pretrained runtime is unavailable. The artifact records the
        # failure rather than substituting mocked values.
        pretrained = {
            "status": "blocked",
            "partition": "validation",
            "conditions": {},
            "error_type": type(error).__name__,
            "error_message": str(error),
            "actual_pretrained_inference_completed": False,
        }
        pretrained_runtime = _runtime_metadata()
    result = {
        "schema_version": 1,
        "experiment": "DeltaVision Task 17 final evaluation",
        "config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "config_path": config["_path"],
        "config_sha256": config["_sha256"],
        "data_provenance": {
            "development": {
                "feature_family": "handcrafted_rgb_statistics_and_policy_development",
                "partition": "synthetic_train",
                "episode_count": research_config.splits["train"].episode_count,
                "random_seed_range": [
                    research_config.splits["train"].base_seed,
                    research_config.splits["train"].base_seed
                    + research_config.splits["train"].episode_count
                    - 1,
                ],
                "used_for_task17_final_metrics": False,
            },
            "policy_validation": {
                "feature_family": (
                    "handcrafted_threshold_and_pretrained_capacity_selection"
                ),
                "partition": "synthetic_validation",
                "episode_count": research_config.splits["validation"].episode_count,
                "random_seed_range": [
                    research_config.splits["validation"].base_seed,
                    research_config.splits["validation"].base_seed
                    + research_config.splits["validation"].episode_count
                    - 1,
                ],
                "used_for_handcrafted_final_metrics": False,
                "used_for_pretrained_final_workload": True,
            },
            "handcrafted_final_evaluation": handcrafted_provenance,
            "pretrained_final_evaluation": {
                "feature_family": "pretrained_dinov2_spatial_tokens",
                "partition": "synthetic_validation",
                "conditions": list(pretrained["conditions"]),
                "random_seeds_by_condition": {
                    name: condition["random_seeds"]
                    for name, condition in pretrained["conditions"].items()
                },
                "episode_count_per_condition": pretrained.get(
                    "episode_count_per_condition"
                ),
                "status": pretrained.get("status", "complete"),
                "used_for_capacity_or_policy_selection_in_task17": False,
            },
            "policy_selection_sources": {
                "handcrafted": {
                    "partition": "synthetic_validation",
                    "fixed_benchmark_config": str(_resolve(config["research_config"])),
                    "validation_reports": ["results/global_change_validation.json"],
                    "approximate_threshold": 0.01,
                    "global_fallback_local_threshold": 0.05,
                    "global_fallback_changed_pixel_fraction_threshold": 0.50,
                },
                "pretrained_dinov2": {
                    "partition": "synthetic_validation",
                    "validation_reports": [
                        "results/pretrained_cache_evaluation.json",
                        "results/pretrained_cache_ablation.json",
                    ],
                    "cache_capacity_per_episode": config["pretrained_dinov2"][
                        "cache_capacity_per_episode"
                    ],
                    "task17_final_data_used_for_selection": False,
                },
            },
        },
        "runtime": {
            "handcrafted": _runtime_metadata(),
            "pretrained": pretrained_runtime,
        },
        "handcrafted": handcrafted,
        "pretrained_dinov2": pretrained,
        "research_integrity": {
            "held_out_is_blind": False,
            "known_prior_held_out_inspection": True,
            "held_out_metrics_used_for_policy_selection": False,
            "handcrafted_test_manifest_only": str(_resolve(config["final_manifest"])),
            "pretrained_used_validation_only": True,
            "pretrained_inference_completed": pretrained.get(
                "actual_pretrained_inference_completed",
                pretrained.get("status") != "blocked",
            ),
            "semantic_accuracy_claimed": False,
            "selective_dinov2_token_computation_claimed": False,
        },
    }
    json_path = (
        Path(output_json)
        if output_json is not None
        else _resolve(config["outputs"]["evaluation_json"])
    )
    csv_path = (
        Path(output_csv)
        if output_csv is not None
        else _resolve(config["outputs"]["evaluation_csv"])
    )
    write_evaluation_artifacts(result, json_path, csv_path)
    return result


def validate_result_provenance(result: Mapping[str, Any]) -> None:
    """Reject result files with partition leakage or missing source provenance."""
    provenance = result.get("data_provenance")
    integrity = result.get("research_integrity")
    if not isinstance(provenance, Mapping) or not isinstance(integrity, Mapping):
        raise ValueError("result is missing machine-readable provenance")
    if integrity.get("held_out_is_blind") is not False:
        raise ValueError("previous held-out inspection must not be described as blind")
    if integrity.get("known_prior_held_out_inspection") is not True:
        raise ValueError("result must declare known prior held-out inspection")
    handcrafted = provenance.get("handcrafted_final_evaluation", {})
    if handcrafted.get("partition") != "previously_inspected_held_out_test":
        raise ValueError("handcrafted final result has invalid partition provenance")
    pretrained = provenance.get("pretrained_final_evaluation", {})
    if pretrained.get("partition") != "synthetic_validation":
        raise ValueError("pretrained result must use validation data")
    if integrity.get("pretrained_used_validation_only") is not True:
        raise ValueError("pretrained validation-only declaration is missing")
