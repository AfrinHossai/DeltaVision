"""Reproducible split generation and evaluation for the synthetic benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Sequence

from .approximate import (
    SUPPORTED_THRESHOLDS,
    FeatureErrorMetrics,
    ThresholdPatchCache,
    compare_feature_vectors,
)
from .baseline import ManifestEpisode, load_manifest
from .encoder import RGBImage, encode_image, read_ppm
from .evaluation import (
    ConfusionCounts,
    count_patch_decisions,
    detection_metrics,
    ground_truth_mask_to_patch_labels,
)
from .state import StatefulPatchCache
from .synthetic import (
    MASK_DEFINITION,
    TRANSFORMATION_TYPES,
    TransformationType,
    generate_sequences,
)

SPLIT_NAMES = ("train", "validation", "test")
EXPERIMENT_LABEL = "Synthetic multi-seed benchmark"
BOOTSTRAP_METRICS = (
    "precision",
    "recall",
    "f1",
    "missed_change_rate",
    "false_refresh_rate",
    "transition_refresh_fraction",
    "feature_mean_absolute_error",
)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Episode count, seed range, and ID namespace for one partition."""

    episode_count: int
    base_seed: int
    episode_id_prefix: str


@dataclass(frozen=True, slots=True)
class ResearchBenchmarkConfig:
    """Validated configuration values used to build and evaluate the dataset."""

    path: Path
    raw: dict[str, Any]
    benchmark_name: str
    width: int
    height: int
    sequence_length: int
    patch_size: int
    transformations: tuple[TransformationType, ...]
    thresholds: tuple[float, ...]
    splits: dict[str, SplitConfig]
    require_all_transformations_per_split: bool
    bootstrap_replicates: int
    bootstrap_confidence_level: float
    bootstrap_seed: int
    threshold_selection_policy: str
    sha256: str


@dataclass(slots=True)
class _Stats:
    """Sufficient statistics for pooled detection and feature metrics."""

    counts: ConfusionCounts = field(default_factory=lambda: ConfusionCounts(0, 0, 0, 0))
    transition_patch_count: int = 0
    transition_recomputed_patches: int = 0
    total_patch_count: int = 0
    recomputed_patches: int = 0
    reused_patches: int = 0
    feature_error_sum: float = 0.0
    feature_value_count: int = 0
    maximum_feature_error: float = 0.0

    def add_feature_metrics(self, metrics: FeatureErrorMetrics) -> None:
        self.feature_error_sum += (
            metrics.mean_absolute_error * metrics.feature_value_count
        )
        self.feature_value_count += metrics.feature_value_count
        self.maximum_feature_error = max(
            self.maximum_feature_error, metrics.maximum_absolute_error
        )


@dataclass(slots=True)
class _EpisodeStats:
    """All-frame and transformation-conditioned statistics for one episode."""

    aggregate: _Stats = field(default_factory=_Stats)
    by_transformation: dict[str, _Stats] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _EpisodeInput:
    """Loaded pixels, reference features, masks, and transition labels."""

    episode_id: str
    frames: tuple[RGBImage, ...]
    baseline_features: tuple[tuple[tuple[float, ...], ...], ...]
    transition_labels: tuple[tuple[bool, ...], ...]
    transformation_types: tuple[str, ...]


def load_research_config(config_path: str | Path) -> ResearchBenchmarkConfig:
    """Read and validate the fixed multi-split benchmark configuration."""
    path = Path(config_path)
    try:
        config_bytes = path.read_bytes()
        raw = json.loads(config_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read benchmark config {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("benchmark configuration must be a JSON object")

    benchmark_name = raw.get("benchmark_name")
    if not isinstance(benchmark_name, str) or not benchmark_name:
        raise ValueError("benchmark_name must be a non-empty string")
    width = _positive_int(raw.get("image_width"), "image_width", minimum=16)
    height = _positive_int(raw.get("image_height"), "image_height", minimum=16)
    sequence_length = _positive_int(
        raw.get("sequence_length"), "sequence_length", minimum=2
    )
    patch_size = _positive_int(raw.get("patch_size"), "patch_size")

    raw_transformations = raw.get("transformations")
    if (
        not isinstance(raw_transformations, list)
        or not raw_transformations
        or any(not isinstance(value, str) for value in raw_transformations)
    ):
        raise ValueError("transformations must be a non-empty JSON array")
    if len(set(raw_transformations)) != len(raw_transformations) or set(
        raw_transformations
    ) != set(TRANSFORMATION_TYPES):
        raise ValueError("transformations must list each supported type exactly once")
    transformations = tuple(raw_transformations)

    raw_thresholds = raw.get("thresholds")
    expected_thresholds = SUPPORTED_THRESHOLDS[1:]
    if (
        not isinstance(raw_thresholds, list)
        or tuple(raw_thresholds) != expected_thresholds
    ):
        raise ValueError(
            "thresholds must be the fixed values 0.01, 0.03, 0.05, and 0.10"
        )
    thresholds = tuple(float(value) for value in raw_thresholds)

    raw_splits = raw.get("splits")
    if not isinstance(raw_splits, dict) or set(raw_splits) != set(SPLIT_NAMES):
        raise ValueError("splits must contain train, validation, and test")
    splits: dict[str, SplitConfig] = {}
    for split_name in SPLIT_NAMES:
        split_raw = raw_splits[split_name]
        if not isinstance(split_raw, dict):
            raise ValueError(f"split {split_name!r} must be an object")
        episode_count = _positive_int(
            split_raw.get("episode_count"), f"{split_name}.episode_count"
        )
        base_seed = _nonnegative_int(
            split_raw.get("base_seed"), f"{split_name}.base_seed"
        )
        prefix = split_raw.get("episode_id_prefix")
        if not isinstance(prefix, str) or not prefix or not _is_portable_prefix(prefix):
            raise ValueError(f"{split_name}.episode_id_prefix is not portable")
        splits[split_name] = SplitConfig(episode_count, base_seed, prefix)

    seed_ranges: list[set[int]] = []
    prefixes: set[str] = set()
    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        seed_range = set(range(split.base_seed, split.base_seed + split.episode_count))
        if any(seed_range & previous for previous in seed_ranges):
            raise ValueError(
                "episode random-seed ranges must not overlap across splits"
            )
        seed_ranges.append(seed_range)
        if split.episode_id_prefix in prefixes:
            raise ValueError("episode ID prefixes must be unique across splits")
        prefixes.add(split.episode_id_prefix)

    coverage = raw.get("require_all_transformations_per_split")
    if not isinstance(coverage, bool):
        raise ValueError("require_all_transformations_per_split must be boolean")
    raw_bootstrap = raw.get("bootstrap")
    if not isinstance(raw_bootstrap, dict):
        raise ValueError("bootstrap must be an object")
    bootstrap_replicates = _positive_int(
        raw_bootstrap.get("replicates"), "bootstrap.replicates"
    )
    confidence_level = raw_bootstrap.get("confidence_level")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("bootstrap.confidence_level must be between 0 and 1")
    bootstrap_seed = _nonnegative_int(raw_bootstrap.get("seed"), "bootstrap.seed")

    selection = raw.get("threshold_selection")
    if not isinstance(selection, dict) or selection.get("method") != "none":
        raise ValueError("threshold_selection.method must be 'none'")
    selection_policy = selection.get("policy")
    if not isinstance(selection_policy, str) or not selection_policy:
        raise ValueError("threshold_selection.policy must be a non-empty string")

    return ResearchBenchmarkConfig(
        path=path,
        raw=raw,
        benchmark_name=benchmark_name,
        width=width,
        height=height,
        sequence_length=sequence_length,
        patch_size=patch_size,
        transformations=transformations,  # type: ignore[arg-type]
        thresholds=thresholds,
        splits=splits,
        require_all_transformations_per_split=coverage,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_confidence_level=float(confidence_level),
        bootstrap_seed=bootstrap_seed,
        threshold_selection_policy=selection_policy,
        sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def build_research_benchmark(
    config_path: str | Path,
    output_dir: str | Path = "data/generated/research_benchmark",
) -> dict[str, object]:
    """Generate all configured splits and write a portable top-level manifest."""
    config = load_research_config(config_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        split = config.splits[split_name]
        generate_sequences(
            root / split_name,
            width=config.width,
            height=config.height,
            sequence_length=config.sequence_length,
            num_episodes=split.episode_count,
            seed=split.base_seed,
            transformations=config.transformations,
            episode_id_prefix=split.episode_id_prefix,
        )

    validation = validate_research_benchmark(config, root)
    index: dict[str, object] = {
        "benchmark_name": config.benchmark_name,
        "config_path": config.path.as_posix(),
        "config_sha256": config.sha256,
        "split_manifests": {
            split_name: f"{split_name}/manifest.jsonl" for split_name in SPLIT_NAMES
        },
        "split_validation": validation,
        "ground_truth_mask_definition": MASK_DEFINITION,
    }
    index_path = root / "benchmark_manifest.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def validate_research_benchmark(
    config: ResearchBenchmarkConfig, data_root: str | Path
) -> dict[str, object]:
    """Validate split sizes, paths, IDs, seeds, transformations, and overlap."""
    root = Path(data_root).resolve()
    seen_episode_ids: set[str] = set()
    seen_seeds: set[int] = set()
    seen_frame_hashes: dict[str, str] = {}
    seen_frame_paths: dict[Path, str] = {}
    split_reports: dict[str, object] = {}

    for split_name in SPLIT_NAMES:
        split = config.splits[split_name]
        split_root = root / split_name
        manifest_path = split_root / "manifest.jsonl"
        episodes = load_manifest(manifest_path)
        rows = _read_jsonl_rows(manifest_path)
        if len(episodes) != split.episode_count or len(rows) != split.episode_count:
            raise ValueError(
                f"split {split_name!r} contains {len(episodes)} episodes; "
                f"expected {split.episode_count}"
            )

        transformation_counts = {name: 0 for name in config.transformations}
        for episode_index, (episode, row) in enumerate(
            zip(episodes, rows, strict=True)
        ):
            expected_id = f"{split.episode_id_prefix}_episode_{episode_index:04d}"
            if (
                episode.episode_id != expected_id
                or row.get("episode_id") != expected_id
            ):
                raise ValueError(
                    f"split {split_name!r} episode {episode_index} has an invalid ID"
                )
            if episode.episode_id in seen_episode_ids:
                raise ValueError(
                    f"episode ID overlaps across splits: {episode.episode_id}"
                )
            seen_episode_ids.add(episode.episode_id)

            seed = row.get("random_seed")
            expected_seed = split.base_seed + episode_index
            if isinstance(seed, bool) or seed != expected_seed:
                raise ValueError(
                    f"episode {episode.episode_id!r} has seed {seed!r}; "
                    f"expected {expected_seed}"
                )
            if seed in seen_seeds:
                raise ValueError(f"random seed overlaps across splits: {seed}")
            seen_seeds.add(seed)

            frame_paths = row.get("frame_paths")
            mask_paths = row.get("ground_truth_mask_paths")
            transitions = row.get("transitions")
            if (
                not isinstance(frame_paths, list)
                or len(frame_paths) != config.sequence_length
                or not isinstance(mask_paths, list)
                or len(mask_paths) != config.sequence_length - 1
                or not isinstance(transitions, list)
                or len(transitions) != config.sequence_length - 1
            ):
                raise ValueError(f"episode {episode.episode_id!r} has invalid lengths")

            for frame in episode.frames:
                if (frame.path.stat().st_size < 0) or not frame.path.is_file():
                    raise ValueError(f"invalid frame sample path: {frame.path}")
                prior_split = seen_frame_paths.get(frame.path)
                if prior_split is not None and prior_split != split_name:
                    raise ValueError("frame paths overlap across benchmark splits")
                seen_frame_paths[frame.path] = split_name
                frame_digest = hashlib.sha256(frame.path.read_bytes()).hexdigest()
                prior_digest_split = seen_frame_hashes.get(frame_digest)
                if prior_digest_split is not None and prior_digest_split != split_name:
                    raise ValueError(
                        "identical rendered frame samples overlap between "
                        f"{prior_digest_split!r} and {split_name!r}"
                    )
                seen_frame_hashes[frame_digest] = split_name

            for transition_index, transition in enumerate(transitions):
                if not isinstance(transition, dict):
                    raise ValueError("manifest transitions must be JSON objects")
                transformation = transition.get("transformation_type")
                parameters = transition.get("transformation_parameters")
                if transformation not in config.transformations:
                    raise ValueError(
                        f"episode {episode.episode_id!r} has unknown transformation"
                    )
                if not isinstance(parameters, dict):
                    raise ValueError("transformation_parameters must be a JSON object")
                if (
                    transition.get("from_frame") != transition_index
                    or transition.get("to_frame") != transition_index + 1
                ):
                    raise ValueError("transition frame indices are not consecutive")
                mask_path = _resolve_dataset_path(
                    split_root,
                    mask_paths[transition_index],
                    f"episode {episode.episode_id!r} mask {transition_index}",
                )
                transition_mask = transition.get("ground_truth_mask_path")
                if transition_mask != mask_paths[transition_index]:
                    raise ValueError("transition and episode mask paths do not match")
                _read_pgm_mask(mask_path)
                transformation_counts[transformation] += 1

        missing_transformations = [
            name for name, count in transformation_counts.items() if count == 0
        ]
        if config.require_all_transformations_per_split and missing_transformations:
            raise ValueError(
                f"split {split_name!r} is missing transformations: "
                f"{', '.join(missing_transformations)}"
            )
        split_reports[split_name] = {
            "episode_count": len(episodes),
            "base_seed": split.base_seed,
            "last_seed": split.base_seed + split.episode_count - 1,
            "episode_id_prefix": split.episode_id_prefix,
            "frame_count": len(episodes) * config.sequence_length,
            "transition_count": len(episodes) * (config.sequence_length - 1),
            "transformation_transition_counts": transformation_counts,
        }

    return {
        "episode_ids_unique_across_splits": True,
        "random_seed_ranges_disjoint": True,
        "rendered_frames_have_no_cross_split_duplicates": True,
        "portable_frame_and_mask_paths_validated": True,
        "splits": split_reports,
    }


def evaluate_research_benchmark(
    config_path: str | Path,
    data_root: str | Path = "data/generated/research_benchmark",
    output_json: str | Path = "results/research_benchmark.json",
    output_csv: str | Path = "results/research_benchmark.csv",
) -> dict[str, object]:
    """Evaluate exact and fixed-threshold reuse separately on each split.

    Thresholds are a fixed grid supplied by the config. This function does not
    select or recommend a threshold; held-out test metrics are reported
    separately and are never used to select a policy.
    """
    config = load_research_config(config_path)
    validation = validate_research_benchmark(config, data_root)
    root = Path(data_root).resolve()
    bootstrap_samples = {
        split_name: _bootstrap_episode_indices(
            config.splits[split_name].episode_count,
            config.bootstrap_replicates,
            config.bootstrap_seed + split_index * 1_000_003,
        )
        for split_index, split_name in enumerate(SPLIT_NAMES)
    }

    partitions: dict[str, object] = {}
    for split_name in SPLIT_NAMES:
        split_root = root / split_name
        episodes = load_manifest(split_root / "manifest.jsonl")
        rows = _read_jsonl_rows(split_root / "manifest.jsonl")
        method_episode_stats: dict[str, list[_EpisodeStats]] = {
            method_name: [] for method_name in _method_names(config.thresholds)
        }
        for episode, row in zip(episodes, rows, strict=True):
            episode_input = _load_episode_input(
                split_root,
                episode,
                row,
                width=config.width,
                height=config.height,
                sequence_length=config.sequence_length,
                patch_size=config.patch_size,
            )
            for method_name, threshold in _methods(config.thresholds):
                method_episode_stats[method_name].append(
                    _evaluate_episode(
                        episode_input, method_name, threshold, config.patch_size
                    )
                )

        method_summaries: dict[str, object] = {}
        for method_name, _threshold in _methods(config.thresholds):
            episode_stats = method_episode_stats[method_name]
            method_summaries[method_name] = _summarize_method(
                episode_stats,
                config.transformations,
                bootstrap_samples[split_name],
                confidence_level=config.bootstrap_confidence_level,
            )
        partitions[split_name] = {
            "episode_count": len(episodes),
            "frame_count": len(episodes) * config.sequence_length,
            "transition_count": len(episodes) * (config.sequence_length - 1),
            "methods": method_summaries,
        }

    result: dict[str, object] = {
        "experiment_label": EXPERIMENT_LABEL,
        "benchmark_name": config.benchmark_name,
        "config_path": config.path.as_posix(),
        "config_sha256": config.sha256,
        "benchmark_manifest_path": (
            Path(data_root) / "benchmark_manifest.json"
        ).as_posix(),
        "patch_size": config.patch_size,
        "thresholds": list(config.thresholds),
        "methods": list(_method_names(config.thresholds)),
        "ground_truth_definition": (
            "A patch is changed when its exact rendered RGB difference mask "
            "contains at least one non-zero pixel; this is pixel-level, not "
            "semantic scene-change ground truth."
        ),
        "aggregation_definitions": {
            "detection": (
                "Pooled patch-level confusion metrics over transitions only; "
                "each episode's initial frame is excluded."
            ),
            "transition_refresh_fraction": (
                "Recomputed patches over patches in transition frames only."
            ),
            "all_frame_computation": (
                "Patch totals, recomputations, and reuse include each episode's "
                "initial frame."
            ),
            "feature_mean_absolute_error": (
                "Mean elementwise error against independent frame encodings "
                "over all frames, patches, and six features."
            ),
            "per_transformation_feature_error": (
                "Feature error on each transition's destination frame, grouped "
                "by its recorded transformation type."
            ),
            "undefined_rates": (
                "A metric with a zero denominator is reported as 0; F1 is 0 "
                "when undefined."
            ),
            "bootstrap": (
                "Percentile cluster bootstrap resampling whole episodes with "
                "replacement; no statistical-significance claim is made."
            ),
            "patch_operation_proxy": (
                "Count of actual six-value patch feature computations; not "
                "neural FLOPs and excludes comparison overhead."
            ),
        },
        "bootstrap": {
            "replicates": config.bootstrap_replicates,
            "confidence_level": config.bootstrap_confidence_level,
            "sampling_unit": "episode",
            "seed": config.bootstrap_seed,
        },
        "threshold_selection": {
            "method": "none",
            "policy": config.threshold_selection_policy,
            "preferred_threshold": None,
            "test_results_used_for_selection": False,
        },
        "split_validation": validation,
        "partitions": partitions,
    }
    _write_results(result, output_json, output_csv)
    return result


def _load_episode_input(
    split_root: Path,
    episode: ManifestEpisode,
    row: dict[str, Any],
    *,
    width: int,
    height: int,
    sequence_length: int,
    patch_size: int,
) -> _EpisodeInput:
    frames = tuple(read_ppm(frame.path) for frame in episode.frames)
    if len(frames) != sequence_length:
        raise ValueError(f"episode {episode.episode_id!r} has an invalid frame count")
    if any((frame.width, frame.height) != (width, height) for frame in frames):
        raise ValueError(
            f"episode {episode.episode_id!r} has unexpected frame dimensions"
        )
    baseline_features = tuple(
        encode_image(frame, patch_size=patch_size).features for frame in frames
    )
    mask_paths = row.get("ground_truth_mask_paths")
    transitions = row.get("transitions")
    if not isinstance(mask_paths, list) or not isinstance(transitions, list):
        raise ValueError(f"episode {episode.episode_id!r} has invalid transitions")
    if (
        len(mask_paths) != sequence_length - 1
        or len(transitions) != sequence_length - 1
    ):
        raise ValueError(f"episode {episode.episode_id!r} has invalid transition count")

    labels: list[tuple[bool, ...]] = []
    transformation_types: list[str] = []
    for transition_index, (raw_mask_path, transition) in enumerate(
        zip(mask_paths, transitions, strict=True)
    ):
        mask_path = _resolve_dataset_path(
            split_root, raw_mask_path, f"episode {episode.episode_id!r} mask"
        )
        mask_width, mask_height, mask_pixels = _read_pgm_mask(mask_path)
        if (mask_width, mask_height) != (width, height):
            raise ValueError(
                f"mask dimensions differ in episode {episode.episode_id!r}"
            )
        labels.append(
            ground_truth_mask_to_patch_labels(mask_pixels, width, height, patch_size)
        )
        transformation_types.append(transition["transformation_type"])
        if (
            transition.get("from_frame") != transition_index
            or transition.get("to_frame") != transition_index + 1
        ):
            raise ValueError("transition frame indices are not consecutive")
    return _EpisodeInput(
        episode_id=episode.episode_id,
        frames=frames,
        baseline_features=baseline_features,
        transition_labels=tuple(labels),
        transformation_types=tuple(transformation_types),
    )


def _evaluate_episode(
    episode: _EpisodeInput,
    method_name: str,
    threshold: float | None,
    patch_size: int,
) -> _EpisodeStats:
    cache: StatefulPatchCache | ThresholdPatchCache
    if method_name == "exact_reuse":
        cache = StatefulPatchCache(patch_size=patch_size)
    else:
        if threshold is None:
            raise ValueError("threshold method requires a threshold")
        cache = ThresholdPatchCache(threshold, patch_size=patch_size)

    result = _EpisodeStats()
    for frame_index, (frame, baseline_features) in enumerate(
        zip(episode.frames, episode.baseline_features, strict=True)
    ):
        encoded = cache.process_frame(frame)
        feature_error = compare_feature_vectors(encoded.features, baseline_features)
        result.aggregate.total_patch_count += encoded.patch_count
        result.aggregate.recomputed_patches += encoded.recomputed_patches
        result.aggregate.reused_patches += encoded.reused_patches
        result.aggregate.add_feature_metrics(feature_error)
        if frame_index == 0:
            continue

        transition_index = frame_index - 1
        transformation = episode.transformation_types[transition_index]
        actual_labels = episode.transition_labels[transition_index]
        if len(actual_labels) != encoded.patch_count:
            raise ValueError(f"patch grid mismatch for episode {episode.episode_id!r}")
        transition_counts = count_patch_decisions(
            actual_labels, encoded.refresh_decisions
        )
        _add_transition(result.aggregate, transition_counts, encoded.recomputed_patches)
        transformation_stats = result.by_transformation.setdefault(
            transformation, _Stats()
        )
        _add_transition(
            transformation_stats, transition_counts, encoded.recomputed_patches
        )
        transformation_stats.add_feature_metrics(feature_error)
    return result


def _add_transition(
    stats: _Stats, counts: ConfusionCounts, recomputed_patches: int
) -> None:
    stats.counts = _sum_counts(stats.counts, counts)
    stats.transition_patch_count += counts.total
    stats.transition_recomputed_patches += recomputed_patches


def _summarize_method(
    episode_stats: Sequence[_EpisodeStats],
    transformations: Sequence[str],
    bootstrap_samples: Sequence[Sequence[int]],
    *,
    confidence_level: float,
) -> dict[str, object]:
    overall_stats = _sum_episode_stats(episode_stats)
    aggregate = _format_stats(overall_stats, include_all_frame=True)
    aggregate["bootstrap_confidence_intervals"] = _bootstrap_intervals(
        episode_stats,
        None,
        bootstrap_samples,
        confidence_level=confidence_level,
    )

    per_transformation: dict[str, object] = {}
    for transformation in transformations:
        transformation_stats = _sum_episode_stats(episode_stats, transformation)
        summary = _format_stats(transformation_stats, include_all_frame=False)
        summary["bootstrap_confidence_intervals"] = _bootstrap_intervals(
            episode_stats,
            transformation,
            bootstrap_samples,
            confidence_level=confidence_level,
        )
        per_transformation[transformation] = summary
    return {"aggregate": aggregate, "per_transformation": per_transformation}


def _format_stats(stats: _Stats, *, include_all_frame: bool) -> dict[str, object]:
    detection = detection_metrics(stats.counts)
    detection.update(
        {
            "transition_patch_count": stats.transition_patch_count,
            "transition_recomputed_patches": stats.transition_recomputed_patches,
            "transition_refresh_fraction": (
                stats.transition_recomputed_patches / stats.transition_patch_count
                if stats.transition_patch_count
                else 0.0
            ),
        }
    )
    feature_approximation = {
        "mean_absolute_feature_error": (
            stats.feature_error_sum / stats.feature_value_count
            if stats.feature_value_count
            else 0.0
        ),
        "maximum_absolute_feature_error": stats.maximum_feature_error,
        "feature_value_count": stats.feature_value_count,
        "scope": (
            "All frames, including each episode's first frame."
            if include_all_frame
            else "Transition destination frames grouped by transformation type."
        ),
    }
    summary: dict[str, object] = {
        "change_detection": detection,
        "feature_approximation": feature_approximation,
    }
    if include_all_frame:
        summary["computation"] = {
            "total_patch_count": stats.total_patch_count,
            "recomputed_patches": stats.recomputed_patches,
            "reused_patches": stats.reused_patches,
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
                    "One actual RGB-statistics patch-feature computation; not "
                    "neural FLOPs and excludes comparison overhead."
                ),
            },
        }
    return summary


def _sum_episode_stats(
    episode_stats: Sequence[_EpisodeStats], transformation: str | None = None
) -> _Stats:
    combined = _Stats()
    for episode in episode_stats:
        stats = (
            episode.aggregate
            if transformation is None
            else episode.by_transformation.get(transformation)
        )
        if stats is None:
            continue
        combined.counts = _sum_counts(combined.counts, stats.counts)
        combined.transition_patch_count += stats.transition_patch_count
        combined.transition_recomputed_patches += stats.transition_recomputed_patches
        combined.total_patch_count += stats.total_patch_count
        combined.recomputed_patches += stats.recomputed_patches
        combined.reused_patches += stats.reused_patches
        combined.feature_error_sum += stats.feature_error_sum
        combined.feature_value_count += stats.feature_value_count
        combined.maximum_feature_error = max(
            combined.maximum_feature_error, stats.maximum_feature_error
        )
    return combined


def _bootstrap_intervals(
    episode_stats: Sequence[_EpisodeStats],
    transformation: str | None,
    bootstrap_samples: Sequence[Sequence[int]],
    *,
    confidence_level: float,
) -> dict[str, dict[str, float]]:
    if len(episode_stats) < 2:
        return {}
    sampled_metrics: dict[str, list[float]] = {
        metric: [] for metric in BOOTSTRAP_METRICS
    }
    if transformation is None:
        sampled_metrics["all_frame_refresh_fraction"] = []
    for indices in bootstrap_samples:
        sampled_stats = [
            episode_stats[index].aggregate
            if transformation is None
            else episode_stats[index].by_transformation.get(transformation, _Stats())
            for index in indices
        ]
        pooled = _sum_stats(sampled_stats)
        metrics = _format_stats(pooled, include_all_frame=transformation is None)
        for metric in BOOTSTRAP_METRICS:
            if metric == "feature_mean_absolute_error":
                value = metrics["feature_approximation"]["mean_absolute_feature_error"]
            elif metric == "transition_refresh_fraction":
                value = metrics["change_detection"]["transition_refresh_fraction"]
            else:
                value = metrics["change_detection"][metric]
            sampled_metrics[metric].append(float(value))
        if transformation is None:
            sampled_metrics["all_frame_refresh_fraction"].append(
                float(metrics["computation"]["all_frame_refresh_fraction"])
            )

    alpha = (1.0 - confidence_level) / 2.0
    return {
        metric: {
            "lower": _percentile(values, alpha),
            "upper": _percentile(values, 1.0 - alpha),
        }
        for metric, values in sampled_metrics.items()
    }


def _sum_stats(stats_values: Sequence[_Stats]) -> _Stats:
    combined = _Stats()
    for stats in stats_values:
        combined.counts = _sum_counts(combined.counts, stats.counts)
        combined.transition_patch_count += stats.transition_patch_count
        combined.transition_recomputed_patches += stats.transition_recomputed_patches
        combined.total_patch_count += stats.total_patch_count
        combined.recomputed_patches += stats.recomputed_patches
        combined.reused_patches += stats.reused_patches
        combined.feature_error_sum += stats.feature_error_sum
        combined.feature_value_count += stats.feature_value_count
        combined.maximum_feature_error = max(
            combined.maximum_feature_error, stats.maximum_feature_error
        )
    return combined


def _bootstrap_episode_indices(
    episode_count: int, replicates: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Create reproducible episode-cluster bootstrap samples."""
    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(episode_count) for _ in range(episode_count))
        for _ in range(replicates)
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _write_results(
    result: dict[str, object], output_json: str | Path, output_csv: str | Path
) -> None:
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    if json_path.resolve() == csv_path.resolve():
        raise ValueError("JSON and CSV result paths must differ")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    columns = (
        "partition",
        "method",
        "scope",
        "transformation_type",
        "episode_count",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "missed_change_rate",
        "false_refresh_rate",
        "transition_refresh_fraction",
        "all_frame_refresh_fraction",
        "feature_mean_absolute_error",
        "maximum_absolute_feature_error",
        "bootstrap_confidence_intervals_json",
    )
    partitions = result["partitions"]
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for partition_name, partition in partitions.items():
            for method_name, method_result in partition["methods"].items():
                aggregate = method_result["aggregate"]
                writer.writerow(
                    _csv_row(
                        partition_name,
                        method_name,
                        "aggregate",
                        "",
                        partition["episode_count"],
                        aggregate,
                    )
                )
                for transformation, transformation_result in method_result[
                    "per_transformation"
                ].items():
                    writer.writerow(
                        _csv_row(
                            partition_name,
                            method_name,
                            "transformation",
                            transformation,
                            partition["episode_count"],
                            transformation_result,
                        )
                    )


def _csv_row(
    partition: str,
    method: str,
    scope: str,
    transformation: str,
    episode_count: int,
    metrics: dict[str, object],
) -> dict[str, object]:
    detection = metrics["change_detection"]
    features = metrics["feature_approximation"]
    computation = metrics.get("computation", {})
    return {
        "partition": partition,
        "method": method,
        "scope": scope,
        "transformation_type": transformation,
        "episode_count": episode_count,
        "true_positives": detection["true_positives"],
        "false_positives": detection["false_positives"],
        "true_negatives": detection["true_negatives"],
        "false_negatives": detection["false_negatives"],
        "precision": detection["precision"],
        "recall": detection["recall"],
        "f1": detection["f1"],
        "missed_change_rate": detection["missed_change_rate"],
        "false_refresh_rate": detection["false_refresh_rate"],
        "transition_refresh_fraction": detection["transition_refresh_fraction"],
        "all_frame_refresh_fraction": computation.get("all_frame_refresh_fraction", ""),
        "feature_mean_absolute_error": features["mean_absolute_feature_error"],
        "maximum_absolute_feature_error": features["maximum_absolute_feature_error"],
        "bootstrap_confidence_intervals_json": json.dumps(
            metrics["bootstrap_confidence_intervals"], sort_keys=True
        ),
    }


def _read_jsonl_rows(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read manifest {path}: {error}") from error
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


def _resolve_dataset_path(root: Path, raw_path: object, context: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(f"{context} has an invalid non-portable path")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or PureWindowsPath(raw_path).drive
        or ".." in relative.parts
        or not relative.parts
    ):
        raise ValueError(f"{context} path must be relative and stay in its split")
    resolved = (root / Path(*relative.parts)).resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"{context} path is missing or escapes its split: {raw_path}")
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


def _methods(thresholds: Sequence[float]) -> tuple[tuple[str, float | None], ...]:
    return (
        ("exact_reuse", None),
        *((f"threshold_{threshold:.2f}", threshold) for threshold in thresholds),
    )


def _method_names(thresholds: Sequence[float]) -> tuple[str, ...]:
    return tuple(name for name, _threshold in _methods(thresholds))


def _sum_counts(first: ConfusionCounts, second: ConfusionCounts) -> ConfusionCounts:
    return ConfusionCounts(
        first.true_positives + second.true_positives,
        first.false_positives + second.false_positives,
        first.true_negatives + second.true_negatives,
        first.false_negatives + second.false_negatives,
    )


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _is_portable_prefix(prefix: str) -> bool:
    return all(
        character.isascii() and (character.isalnum() or character in "_-")
        for character in prefix
    )
