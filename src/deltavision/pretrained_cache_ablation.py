"""DINOv2 exact-cache capacity ablation on deterministic validation traces."""

from __future__ import annotations

import array
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .baseline import load_manifest
from .encoder import RGBImage
from .pretrained import FrozenVisionEncoderAdapter, RepresentationKind
from .pretrained_cache_evaluation import (
    EpisodeImages,
    run_policy_pass,
    runtime_hardware,
    summarize_visual_changes,
)
from .pretrained_stateful import _inputs_equal, stable_model_input_hash
from .synthetic import generate_sequences

CAPACITY_VALUES = (1, 2, 4, 8, 16, 32)
CSV_FIELDS = (
    "condition",
    "condition_label",
    "repetition",
    "execution_order",
    "policy",
    "cache_capacity",
    "total_frames",
    "model_forward_calls",
    "cached_frame_hits",
    "cache_misses",
    "cache_hit_rate",
    "cache_evictions",
    "avoided_forward_percentage",
    "processing_latency_seconds",
    "model_inference_seconds",
    "preprocessing_seconds",
    "cache_lookup_overhead_seconds",
    "cache_retrieval_overhead_seconds",
    "cache_storage_overhead_seconds",
    "peak_cache_memory_estimate_bytes",
)
POLICY_NAMES = (
    "independent",
    *(f"exact_cache_capacity_{value}" for value in CAPACITY_VALUES),
)


def load_ablation_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the fixed capacity-sweep configuration."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read cache ablation config {path}: {error}"
        ) from error
    validate_ablation_config(config)
    return config


def validate_ablation_config(config: Mapping[str, Any]) -> None:
    """Enforce deterministic validation-only capacity and workload settings."""
    if config.get("cache_capacities") != list(CAPACITY_VALUES):
        raise ValueError("cache_capacities must be [1, 2, 4, 8, 16, 32]")
    for field in ("warmup_repetitions", "timed_repetitions", "width", "height"):
        value = config.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config["warmup_repetitions"] < 3:
        raise ValueError("at least three warmup repetitions are required")
    if config["timed_repetitions"] < 5:
        raise ValueError("at least five timed repetitions are required")
    if config.get("global_rgb_offset_modulus") != 256:
        raise ValueError("global_rgb_offset_modulus must be 256 for RGB inputs")
    if config["width"] < 16 or config["height"] < 16:
        raise ValueError("synthetic image dimensions must be at least 16 pixels")
    if not isinstance(config.get("base_seed"), int) or not isinstance(
        config.get("policy_order_seed"), int
    ):
        raise ValueError("base_seed and policy_order_seed must be integers")
    conditions = config.get("conditions")
    expected = (
        "repeated_identical_frames",
        "recurring_nonconsecutive_frames",
        "sparse_changes",
        "frequent_widespread_changes",
    )
    if (
        not isinstance(conditions, list)
        or tuple(condition.get("name") for condition in conditions) != expected
    ):
        raise ValueError("conditions must use the four documented ordered traces")
    trace_names = (
        "repeat_single",
        "ascending_repeat_zero_descending",
        "four_equal_blocks",
        "sequential",
    )
    for condition, expected_trace in zip(conditions, trace_names, strict=True):
        if condition.get("trace") != expected_trace:
            raise ValueError("condition trace does not match its configured role")
        bank_size = condition.get("bank_size")
        offset_step = condition.get("offset_step")
        if not isinstance(bank_size, int) or bank_size < 1:
            raise ValueError("bank_size must be a positive integer")
        if not isinstance(offset_step, int) or offset_step < 1:
            raise ValueError("offset_step must be a positive integer")
    for field in (
        "maximum_absolute_difference_tolerance",
        "minimum_cosine_similarity_tolerance",
    ):
        value = float(config.get(field, float("nan")))
        if not math.isfinite(value):
            raise ValueError("feature comparison tolerances must be finite")
    if config["maximum_absolute_difference_tolerance"] < 0.0:
        raise ValueError("maximum absolute difference tolerance cannot be negative")
    if not -1.0 <= config["minimum_cosine_similarity_tolerance"] <= 1.0:
        raise ValueError("cosine similarity tolerance must be in [-1, 1]")


def _trace_indices(trace: str) -> list[int]:
    if trace == "repeat_single":
        return [0] * 64
    if trace == "ascending_repeat_zero_descending":
        return [*range(32), 0, *range(31, -1, -1)]
    if trace == "four_equal_blocks":
        return [bank_index for bank_index in range(4) for _ in range(16)]
    if trace == "sequential":
        return list(range(64))
    raise ValueError(f"unsupported deterministic trace: {trace}")


def generate_capacity_scenarios(
    validation_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Path]:
    """Create existing-manifest-compatible RGB sequences under validation only."""
    validate_ablation_config(config)
    root = Path(validation_root).resolve()
    parts = {part.lower() for part in root.parts}
    if "test" in parts:
        raise ValueError("held-out test paths are excluded from capacity evaluation")
    if "validation" not in parts:
        raise ValueError("capacity data must be generated under validation")
    generated: dict[str, Path] = {}
    for condition_index, condition in enumerate(config["conditions"]):
        name = condition["name"]
        seed = config["base_seed"] + condition_index * 100
        base_manifest = generate_sequences(
            root / "source_scenes" / name,
            width=config["width"],
            height=config["height"],
            sequence_length=1,
            num_episodes=1,
            seed=seed,
            transformations=("unchanged",),
            episode_id_prefix=f"validation_capacity_{name}",
        )
        base_episode = load_manifest(base_manifest)[0]
        base_image = _read_image(base_episode.frames[0].path)
        offsets = [
            (index * condition["offset_step"]) % config["global_rgb_offset_modulus"]
            for index in range(condition["bank_size"])
        ]
        bank = [
            _offset_rgb(base_image, offset, config["global_rgb_offset_modulus"])
            for offset in offsets
        ]
        trace_indices = _trace_indices(condition["trace"])
        if max(trace_indices, default=0) >= len(bank):
            raise ValueError(f"trace index exceeds image bank for condition {name}")
        scenario_root = root / name
        scenario_root.mkdir(parents=True, exist_ok=True)
        episode_id = f"validation_capacity_{name}_episode_0000"
        episode_dir = scenario_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[str] = []
        for frame_index, bank_index in enumerate(trace_indices):
            image = bank[bank_index]
            relative = Path(episode_id) / f"frame_{frame_index:04d}.ppm"
            destination = scenario_root / relative
            destination.write_bytes(_encode_ppm(image))
            frame_paths.append(relative.as_posix())
        manifest = scenario_root / "manifest.jsonl"
        row = {
            "episode_id": episode_id,
            "random_seed": seed,
            "frame_paths": frame_paths,
            "condition": name,
            "trace": condition["trace"],
            "source_episode_id": base_episode.episode_id,
            "global_rgb_offsets": offsets,
            "frame_bank_indices": trace_indices,
            "expected_unique_raw_frames": len(set(trace_indices)),
        }
        manifest.write_text(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        generated[name] = manifest
    return generated


def _read_image(path: Path) -> RGBImage:
    from .encoder import read_ppm

    return read_ppm(path)


def _offset_rgb(image: RGBImage, offset: int, modulus: int) -> RGBImage:
    return RGBImage(
        width=image.width,
        height=image.height,
        pixels=bytes((channel + offset) % modulus for channel in image.pixels),
    )


def _encode_ppm(image: RGBImage) -> bytes:
    header = f"P6\n{image.width} {image.height}\n255\n".encode("ascii")
    return header + image.pixels


def count_unique_preprocessed_inputs(
    adapter: FrozenVisionEncoderAdapter, episodes: EpisodeImages
) -> dict[str, Any]:
    """Count processor-output identities with tensor equality checks."""
    total = 0
    unique = 0
    duplicates = 0
    for _episode_id, images in episodes:
        seen: dict[str, list[Mapping[str, Any]]] = {}
        for image in images:
            total += 1
            inputs = adapter.prepare_frame(image).model_inputs
            digest = stable_model_input_hash(inputs, adapter._torch)
            candidates = seen.setdefault(digest, [])
            if any(_inputs_equal(old, inputs, adapter._torch) for old in candidates):
                duplicates += 1
            else:
                unique += 1
                candidates.append(inputs)
    return {
        "frame_count": total,
        "unique_preprocessed_input_count": unique,
        "duplicate_preprocessed_input_count": duplicates,
    }


def _timing_summary(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("at least one latency sample is required")
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.fmean(values),
        "sample_standard_deviation_seconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "repetition_count": len(values),
    }


def _audit_cached_features(
    adapter: FrozenVisionEncoderAdapter,
    episodes: EpisodeImages,
    *,
    representation: RepresentationKind,
    cache_capacity: int,
    maximum_absolute_difference_tolerance: float,
    minimum_cosine_similarity_tolerance: float,
) -> dict[str, Any]:
    """Compare cached features with a separate independent feature reference."""
    from .pretrained_stateful import ExactWholeFrameCache

    reference = array.array("f")
    independent_calls = 0
    for _episode_id, images in episodes:
        for image in images:
            result = adapter.encode_frame(image, representation=representation)
            reference.extend(_flatten_values(result.features))
            independent_calls += 1
    cached = array.array("f")
    cache = ExactWholeFrameCache(adapter, capacity=cache_capacity)
    for _episode_id, images in episodes:
        cache.reset_episode()
        for image in images:
            result = cache.encode_frame(image, representation=representation)
            cached.extend(_flatten_values(result.encoding.features))
    metrics = _array_feature_metrics(adapter._torch, reference, cached)
    equivalent = (
        metrics["maximum_absolute_difference"] <= maximum_absolute_difference_tolerance
        and metrics["cosine_similarity"] >= minimum_cosine_similarity_tolerance
    )
    return {
        **metrics,
        "independent_comparison_forward_calls": independent_calls,
        "stateful_comparison_forward_calls": cache.actual_model_forward_calls,
        "within_documented_tolerance": equivalent,
        "comparison_timing_included_in_policy_latency": False,
        "maximum_absolute_difference_tolerance": (
            maximum_absolute_difference_tolerance
        ),
        "minimum_cosine_similarity_tolerance": minimum_cosine_similarity_tolerance,
    }


def _array_feature_metrics(
    torch_module: Any, reference: array.array[float], cached: array.array[float]
) -> dict[str, Any]:
    """Compute finite, float32-aware feature error metrics with tensor kernels."""
    if len(reference) != len(cached):
        raise ValueError("independent and cached feature arrays have different lengths")
    if not reference:
        raise ValueError("feature arrays must be non-empty")
    reference_tensor = torch_module.tensor(reference, dtype=torch_module.float32)
    cached_tensor = torch_module.tensor(cached, dtype=torch_module.float32)
    if not bool(torch_module.isfinite(reference_tensor).all()) or not bool(
        torch_module.isfinite(cached_tensor).all()
    ):
        raise ValueError("feature comparison contains NaN or infinity")
    absolute_difference = torch_module.abs(reference_tensor - cached_tensor)
    reference_norm = torch_module.linalg.vector_norm(reference_tensor)
    cached_norm = torch_module.linalg.vector_norm(cached_tensor)
    if float(reference_norm) == 0.0 and float(cached_norm) == 0.0:
        cosine = 1.0
    elif float(reference_norm) == 0.0 or float(cached_norm) == 0.0:
        cosine = 0.0
    else:
        cosine = float(
            torch_module.dot(reference_tensor, cached_tensor)
            / (reference_norm * cached_norm)
        )
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "element_count": len(reference),
        "different_value_count": int(torch_module.count_nonzero(absolute_difference)),
        "maximum_absolute_difference": float(absolute_difference.max()),
        "mean_absolute_difference": float(absolute_difference.mean()),
        "cosine_similarity": cosine,
        "exact_match": bool(torch_module.equal(reference_tensor, cached_tensor)),
    }


def evaluate_capacity_ablation(
    adapter: FrozenVisionEncoderAdapter,
    scenarios: Mapping[str, tuple[Path, EpisodeImages]],
    config: Mapping[str, Any],
    *,
    representation: RepresentationKind,
) -> dict[str, Any]:
    """Run deterministic warmups, timed capacity policies, and feature audits."""
    if set(scenarios) != {condition["name"] for condition in config["conditions"]}:
        raise ValueError("scenario names do not match ablation configuration")
    rng = random.Random(config["policy_order_seed"])
    raw_trials: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    scenario_details = {
        condition["name"]: {
            "unique_inputs": count_unique_preprocessed_inputs(
                adapter, scenarios[condition["name"]][1]
            ),
            "visual": summarize_visual_changes(scenarios[condition["name"]][1]),
        }
        for condition in config["conditions"]
    }
    for condition in config["conditions"]:
        name = condition["name"]
        manifest, episodes = scenarios[name]
        unique_inputs = scenario_details[name]["unique_inputs"]
        visual = scenario_details[name]["visual"]
        pass_results: dict[str, list[dict[str, Any]]] = {
            policy: [] for policy in POLICY_NAMES
        }
        warmup_counts: dict[str, int] = dict.fromkeys(POLICY_NAMES, 0)
        for repetition in range(config["warmup_repetitions"]):
            order = list(POLICY_NAMES)
            rng.shuffle(order)
            schedules.append(
                {
                    "condition": name,
                    "phase": "warmup",
                    "repetition": repetition + 1,
                    "policy_order": order,
                }
            )
            for position, policy in enumerate(order, start=1):
                result = _run_named_policy(
                    adapter,
                    episodes,
                    policy,
                    representation=representation,
                    capacities=CAPACITY_VALUES,
                )
                warmup_counts[policy] += result["model_forward_calls"]
                result["processing_latency_seconds"] = None
                result.update(
                    {
                        "condition": name,
                        "condition_label": condition["label"],
                        "repetition": repetition + 1,
                        "execution_order": position,
                    }
                )
        for repetition in range(config["timed_repetitions"]):
            order = list(POLICY_NAMES)
            rng.shuffle(order)
            schedules.append(
                {
                    "condition": name,
                    "phase": "timed",
                    "repetition": repetition + 1,
                    "policy_order": order,
                }
            )
            for position, policy in enumerate(order, start=1):
                result = _run_named_policy(
                    adapter,
                    episodes,
                    policy,
                    representation=representation,
                    capacities=CAPACITY_VALUES,
                )
                pass_results[policy].append(result)
                raw_trials.append(
                    {
                        "condition": name,
                        "condition_label": condition["label"],
                        "repetition": repetition + 1,
                        "execution_order": position,
                        **result,
                    }
                )

        independent_trials = pass_results["independent"]
        independent_latencies = [
            result["processing_latency_seconds"] for result in independent_trials
        ]
        independent_median = statistics.median(independent_latencies)
        capacity_results: list[dict[str, Any]] = []
        for capacity in CAPACITY_VALUES:
            policy_name = f"exact_cache_capacity_{capacity}"
            trials = pass_results[policy_name]
            if len(trials) != config["timed_repetitions"]:
                raise ValueError("timed repetitions are incomplete")
            first = trials[0]
            for result in trials[1:]:
                for metric in (
                    "total_frames",
                    "model_forward_calls",
                    "cached_frame_hits",
                    "cache_misses",
                    "cache_evictions",
                    "peak_cache_memory_estimate_bytes",
                ):
                    if result[metric] != first[metric]:
                        raise ValueError(
                            f"non-deterministic {metric} for {name}, "
                            f"capacity {capacity}"
                        )
            latency_values = [result["processing_latency_seconds"] for result in trials]
            cache_median = statistics.median(latency_values)
            equivalence = _audit_cached_features(
                adapter,
                episodes,
                representation=representation,
                cache_capacity=capacity,
                maximum_absolute_difference_tolerance=float(
                    config["maximum_absolute_difference_tolerance"]
                ),
                minimum_cosine_similarity_tolerance=float(
                    config["minimum_cosine_similarity_tolerance"]
                ),
            )
            capacity_results.append(
                {
                    "capacity": capacity,
                    "total_frames_per_pass": first["total_frames"],
                    "actual_model_forward_calls_per_pass": first["model_forward_calls"],
                    "cached_frame_hits_per_pass": first["cached_frame_hits"],
                    "cache_misses_per_pass": first["cache_misses"],
                    "cache_hit_rate": first["cache_hit_rate"],
                    "avoided_forward_percentage": first["avoided_forward_percentage"],
                    "cache_evictions_per_pass": first["cache_evictions"],
                    "peak_cache_memory_estimate_bytes": first[
                        "peak_cache_memory_estimate_bytes"
                    ],
                    "peak_cache_memory_estimate_mib": (
                        first["peak_cache_memory_estimate_bytes"] / (1024**2)
                    ),
                    "median_processing_latency": _timing_summary(latency_values),
                    "independent_median_processing_latency_seconds": (
                        independent_median
                    ),
                    "independent_over_cache_median_speedup_ratio": (
                        independent_median / cache_median if cache_median else None
                    ),
                    "latency_change_percent_vs_independent": (
                        (cache_median - independent_median) / independent_median * 100.0
                        if independent_median
                        else 0.0
                    ),
                    "mean_model_inference_seconds": statistics.fmean(
                        result["model_inference_seconds"] for result in trials
                    ),
                    "mean_preprocessing_seconds": statistics.fmean(
                        result["preprocessing_seconds"] for result in trials
                    ),
                    "mean_cache_lookup_overhead_seconds": statistics.fmean(
                        result["cache_lookup_overhead_seconds"] for result in trials
                    ),
                    "mean_cache_retrieval_overhead_seconds": statistics.fmean(
                        result["cache_retrieval_overhead_seconds"] for result in trials
                    ),
                    "mean_cache_storage_overhead_seconds": statistics.fmean(
                        result["cache_storage_overhead_seconds"] for result in trials
                    ),
                    "feature_equivalence": equivalence,
                }
            )
        summaries[name] = {
            "label": condition["label"],
            "manifest": manifest.as_posix(),
            "seed": config["base_seed"] + config["conditions"].index(condition) * 100,
            "frame_count_per_pass": unique_inputs["frame_count"],
            "unique_preprocessed_input_count": unique_inputs[
                "unique_preprocessed_input_count"
            ],
            "duplicate_preprocessed_input_count": unique_inputs[
                "duplicate_preprocessed_input_count"
            ],
            "visual_change_summary": visual,
            "independent": {
                "model_forward_calls_per_pass": independent_trials[0][
                    "model_forward_calls"
                ],
                "median_processing_latency": _timing_summary(independent_latencies),
                "warmup_model_forward_calls_total": warmup_counts["independent"],
                "feature_shape": independent_trials[0]["feature_shape"],
            },
            "warmup_model_forward_calls_by_policy": warmup_counts,
            "capacities": capacity_results,
        }
    return {
        "condition_results": summaries,
        "raw_timed_trials": raw_trials,
        "execution_schedule": schedules,
        "timing_scope": (
            "Preloaded validation frames; includes preprocessing, exact cache "
            "hash/equality lookup, full-model forwards on misses, cache storage "
            "and retrieval. Excludes disk I/O, model loading, feature audits, "
            "result serialization, and plotting."
        ),
    }


def _run_named_policy(
    adapter: FrozenVisionEncoderAdapter,
    episodes: EpisodeImages,
    policy_name: str,
    *,
    representation: RepresentationKind,
    capacities: Sequence[int],
) -> dict[str, Any]:
    if policy_name == "independent":
        result = run_policy_pass(
            adapter,
            episodes,
            policy="independent",
            representation=representation,
            cache_capacity=0,
        )
        return {**result, "cache_capacity": None}
    expected = {f"exact_cache_capacity_{capacity}": capacity for capacity in capacities}
    try:
        capacity = expected[policy_name]
    except KeyError as error:
        raise ValueError(f"unknown policy name {policy_name}") from error
    result = run_policy_pass(
        adapter,
        episodes,
        policy="exact_cache",
        representation=representation,
        cache_capacity=capacity,
    )
    return {**result, "cache_capacity": capacity}


def build_ablation_report(
    evaluation: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    adapter: FrozenVisionEncoderAdapter,
    config_path: str | Path,
    baseline_config_path: str | Path,
) -> dict[str, Any]:
    """Record model/runtime provenance, workload accounting, and limits."""
    all_timings = evaluation["raw_timed_trials"]
    timed_independent_calls = sum(
        trial["model_forward_calls"]
        for trial in all_timings
        if trial["policy"] == "independent"
    )
    timed_cache_calls = sum(
        trial["model_forward_calls"]
        for trial in all_timings
        if trial["policy"] == "exact_cache"
    )
    capacity_trials = [
        trial for trial in all_timings if trial["policy"] == "exact_cache"
    ]
    matched_independent_calls = sum(trial["total_frames"] for trial in capacity_trials)
    timed_calls_avoided = matched_independent_calls - timed_cache_calls
    condition_results = evaluation["condition_results"]
    comparison_audit = {
        "independent_full_model_calls": sum(
            capacity["feature_equivalence"]["independent_comparison_forward_calls"]
            for condition in condition_results.values()
            for capacity in condition["capacities"]
        ),
        "exact_cache_full_model_calls": sum(
            capacity["feature_equivalence"]["stateful_comparison_forward_calls"]
            for condition in condition_results.values()
            for capacity in condition["capacities"]
        ),
        "included_in_timed_latency": False,
    }
    return {
        "experiment": "DINOv2 exact whole-frame cache capacity ablation",
        "status": "completed",
        "partition": "validation",
        "held_out_test_partition_accessed": False,
        "config_path": Path(config_path).resolve().as_posix(),
        "baseline_config_path": Path(baseline_config_path).resolve().as_posix(),
        "configuration": dict(config),
        "model": {
            "model_identifier": adapter.model_id,
            "model_revision": adapter.revision,
            "representation": baseline_config.get("representation", "spatial_tokens"),
            "preprocessing": adapter.preprocessing.to_dict(),
            "device": adapter.device,
            "evaluation_mode": not bool(getattr(adapter.model, "training", False)),
            "frozen_inference": not any(
                bool(parameter.requires_grad)
                for parameter in adapter.model.parameters()
            ),
            "checkpoint_downloads_allowed": False,
            "model_config": {
                "model_type": getattr(adapter.model.config, "model_type", None),
                "patch_size": getattr(adapter.model.config, "patch_size", None),
                "hidden_size": getattr(adapter.model.config, "hidden_size", None),
                "num_register_tokens": getattr(
                    adapter.model.config, "num_register_tokens", 0
                ),
            },
        },
        "hardware_and_software": runtime_hardware(adapter),
        "cache_design": {
            "type": "exact complete-frame representation reuse",
            "hash": "SHA-256 of complete preprocessed model inputs",
            "collision_safety": "hash candidate is verified by exact tensor equality",
            "eviction_policy": "deterministic per-episode LRU",
            "cache_capacity_values": list(CAPACITY_VALUES),
            "partial_token_computation": False,
        },
        "timing_protocol": {
            "warmup_repetitions_per_policy_condition": config["warmup_repetitions"],
            "timed_repetitions_per_policy_condition": config["timed_repetitions"],
            "policy_order_seed": config["policy_order_seed"],
            "policy_order": (
                "Deterministic shuffle of independent plus six capacities per round."
            ),
            "image_reads_preloaded_before_timing": True,
            "scope": evaluation["timing_scope"],
            "repetitions_are_independent_datasets": False,
        },
        "forward_call_accounting": {
            "timed_independent_full_model_calls_measured": timed_independent_calls,
            "timed_exact_cache_full_model_calls": timed_cache_calls,
            "capacity_matched_independent_reference_calls": matched_independent_calls,
            "timed_full_model_calls_avoided_across_capacity_pairs": (
                timed_calls_avoided
            ),
            "baseline_reference_scope": (
                "One measured independent pass per condition and repetition is "
                "shared across six cache-capacity comparisons; matched calls "
                "are derived from each cache trial's frame count."
            ),
            "untimed_warmup_full_model_calls_by_policy": {
                policy: sum(
                    condition["warmup_model_forward_calls_by_policy"][policy]
                    for condition in condition_results.values()
                )
                for policy in POLICY_NAMES
            },
            "separate_feature_equivalence_audits": comparison_audit,
        },
        "condition_results": condition_results,
        "raw_timed_trials": all_timings,
        "execution_schedule": evaluation["execution_schedule"],
        "research_limits": [
            "All inputs are deterministic synthetic RGB scenes in the validation "
            "partition.",
            "Feature equality is numerical evidence and does not imply semantic "
            "accuracy.",
            "Only complete representations are reused; no DINOv2 token is "
            "independently recomputed.",
            "Without repeated preprocessed inputs, lookup and storage add overhead "
            "without avoiding full-model forwards.",
            "Tensor-payload memory excludes Python, container, allocator, and "
            "in-flight memory.",
            "Five timed repetitions reuse the same trace and are not independent "
            "samples.",
        ],
    }


def write_ablation_outputs(
    report: Mapping[str, Any],
    output_json: str | Path,
    output_csv: str | Path,
    figure_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Write complete report, raw timed rows, and the capacity-latency figure."""
    json_path, csv_path, png_path = map(Path, (output_json, output_csv, figure_path))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for trial in report["raw_timed_trials"]:
            writer.writerow({field: trial.get(field) for field in CSV_FIELDS})
    draw_capacity_figure(report, png_path)
    return json_path, csv_path, png_path


def draw_capacity_figure(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Plot validation latency by capacity against independent references."""
    from PIL import Image, ImageDraw, ImageFont

    colors = {
        "repeated_identical_frames": "#4C78A8",
        "recurring_nonconsecutive_frames": "#F58518",
        "sparse_changes": "#54A24B",
        "frequent_widespread_changes": "#E45756",
    }
    results = report["condition_results"]
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 110, 105, width - 70, height - 215
    draw.text(
        (left, 32),
        "DINOv2 exact whole-frame cache capacity: synthetic validation",
        fill="#111111",
        font=font,
    )
    all_medians = [
        entry["median_processing_latency"]["median_seconds"]
        for condition in results.values()
        for entry in condition["capacities"]
    ]
    all_medians += [
        condition["independent"]["median_processing_latency"]["median_seconds"]
        for condition in results.values()
    ]
    maximum = max(all_medians, default=1.0) * 1.08
    plot_height = bottom - top
    for tick in range(6):
        fraction = tick / 5
        y = bottom - round(plot_height * fraction)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text(
            (left - 70, y - 5), f"{maximum * fraction:.2f}s", fill="#333333", font=font
        )
    x_positions = [
        left + (right - left) * index / (len(CAPACITY_VALUES) - 1)
        for index in range(len(CAPACITY_VALUES))
    ]
    for x, capacity in zip(x_positions, CAPACITY_VALUES, strict=True):
        draw.line((x, bottom, x, bottom + 6), fill="#333333", width=1)
        label = str(capacity)
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x - (bounds[2] - bounds[0]) / 2, bottom + 12),
            label,
            fill="#222222",
            font=font,
        )
    for name, condition in results.items():
        color = colors[name]
        values = [
            item["median_processing_latency"]["median_seconds"]
            for item in condition["capacities"]
        ]
        points = [
            (x, bottom - round(plot_height * value / maximum))
            for x, value in zip(x_positions, values, strict=True)
        ]
        draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        baseline = condition["independent"]["median_processing_latency"][
            "median_seconds"
        ]
        baseline_y = bottom - round(plot_height * baseline / maximum)
        draw.line(
            (left, baseline_y, right, baseline_y),
            fill=color,
            width=1,
            joint="curve",
        )
    legend_y = height - 105
    legend_x = left
    for name, condition in results.items():
        color = colors[name]
        draw.line(
            (legend_x, legend_y + 7, legend_x + 25, legend_y + 7), fill=color, width=4
        )
        draw.text(
            (legend_x + 32, legend_y), condition["label"], fill="#222222", font=font
        )
        legend_x += 310
    draw.text(
        (left, bottom + 35),
        "Cache capacity (entries/episode; log2-spaced positions)",
        fill="#333333",
        font=font,
    )
    draw.text(
        (left, height - 52),
        "Median seconds per trace pass; thin lines are independent references. "
        "Payload memory is in the report. Synthetic validation only.",
        fill="#444444",
        font=font,
    )
    path = Path(output_path)
    image.save(path)
    return path


def _flatten_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for child in value:
            flattened.extend(_flatten_values(child))
        return flattened
    return [float(value)]
