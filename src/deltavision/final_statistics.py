"""Episode-cluster bootstrap analysis for Task 17 raw evaluation artifacts."""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .final_evaluation import (
    REPOSITORY_ROOT,
    _empty_sufficient,
    _resolve,
    _sum_sufficient,
    handcrafted_metric_view,
    load_final_config,
    validate_result_provenance,
)
from .synthetic import TRANSFORMATION_TYPES

HANDCRAFTED_METRICS = (
    "precision",
    "recall",
    "f1",
    "missed_change_rate",
    "false_refresh_rate",
    "transition_refresh_fraction",
    "all_frame_refresh_fraction",
    "feature_mean_absolute_error",
)


def bootstrap_episode_indices(
    episode_count: int, replicates: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Return seeded, with-replacement samples of episode indices."""
    if episode_count < 1 or replicates < 1 or seed < 0:
        raise ValueError("bootstrap episode count, repetitions, and seed must be valid")
    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(episode_count) for _ in range(episode_count))
        for _ in range(replicates)
    )


def percentile(values: Sequence[float], probability: float) -> float:
    """Calculate a linearly interpolated percentile without NumPy."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_interval(
    samples: Sequence[float], confidence_level: float
) -> dict[str, float]:
    """Summarize percentile interval endpoints for a bootstrap distribution."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between 0 and 1")
    tail = (1.0 - confidence_level) / 2.0
    return {
        "lower": percentile(samples, tail),
        "upper": percentile(samples, 1.0 - tail),
    }


def paired_bootstrap_contrast(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap the paired mean difference ``left - right`` by episode."""
    if not left or len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have equal non-zero length")
    if any(not math.isfinite(float(value)) for value in (*left, *right)):
        raise ValueError("paired bootstrap values must be finite")
    indices = bootstrap_episode_indices(len(left), replicates, seed)
    distribution = [
        statistics.fmean(float(left[index]) - float(right[index]) for index in row)
        for row in indices
    ]
    return {
        "point_estimate": statistics.fmean(
            float(a) - float(b) for a, b in zip(left, right, strict=True)
        ),
        "confidence_interval": bootstrap_interval(distribution, confidence_level),
        "replicates": replicates,
        "sampling_unit": "paired_episode",
    }


def _pool_handcrafted(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    transformation: str | None,
) -> dict[str, Any]:
    selected: list[Mapping[str, Any]] = []
    for index in indices:
        record = records[index]
        if transformation is None:
            selected.append(record["aggregate"])
        else:
            selected.append(
                record.get("by_transformation", {}).get(
                    transformation, _empty_sufficient()
                )
            )
    return _sum_sufficient(selected)


def _metric_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    transformation: str | None,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if transformation is None:
        population = list(range(len(records)))
    else:
        population = [
            index
            for index, record in enumerate(records)
            if transformation in record.get("by_transformation", {})
        ]
    if not population:
        return {"episode_count": 0, "metrics": {}}
    samples = bootstrap_episode_indices(len(population), replicates, seed)
    mapped_samples = tuple(
        tuple(population[index] for index in draw) for draw in samples
    )
    distributions = {name: [] for name in HANDCRAFTED_METRICS}
    for indices in mapped_samples:
        pooled = _pool_handcrafted(records, indices, transformation)
        values = handcrafted_metric_view(pooled)
        for metric in HANDCRAFTED_METRICS:
            distributions[metric].append(float(values[metric]))
    return {
        "episode_count": len(population),
        "metrics": {
            name: bootstrap_interval(values, confidence_level)
            for name, values in distributions.items()
        },
    }


def _handcrafted_statistics(
    evaluation: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    settings = config["statistics"]
    replicates = int(settings["replicates"])
    confidence_level = float(settings["confidence_level"])
    seed = int(settings["handcrafted_seed"])
    methods = evaluation["handcrafted"]["methods"]
    method_records = {name: item["per_episode"] for name, item in methods.items()}
    method_ids = {
        name: tuple(record["episode_id"] for record in records)
        for name, records in method_records.items()
    }
    reference_ids = next(iter(method_ids.values()))
    if any(ids != reference_ids for ids in method_ids.values()):
        raise ValueError("handcrafted methods do not contain the same paired episodes")

    summaries: dict[str, Any] = {}
    for method_index, (method, records) in enumerate(method_records.items()):
        point = methods[method]["aggregate"]
        summaries[method] = {
            "episode_count": len(records),
            "observation_count": evaluation["handcrafted"]["transitions"],
            "point_estimates": {name: point[name] for name in HANDCRAFTED_METRICS},
            "overall_confidence_intervals": _metric_bootstrap(
                records,
                transformation=None,
                replicates=replicates,
                seed=seed + method_index,
                confidence_level=confidence_level,
            )["metrics"],
            "by_transformation": {},
        }
        for category_index, transformation in enumerate(TRANSFORMATION_TYPES):
            category_records = [
                record
                for record in records
                if transformation in record.get("by_transformation", {})
            ]
            pooled = _sum_sufficient(
                [
                    record["by_transformation"][transformation]
                    for record in category_records
                ]
            )
            summaries[method]["by_transformation"][transformation] = {
                "episode_count": len(category_records),
                "point_estimates": handcrafted_metric_view(pooled),
                "confidence_intervals": _metric_bootstrap(
                    records,
                    transformation=transformation,
                    replicates=replicates,
                    seed=seed + method_index * 100 + category_index + 1,
                    confidence_level=confidence_level,
                )["metrics"],
            }

    reference = "independent_encoding"
    paired_contrasts: dict[str, Any] = {}
    for method_index, method in enumerate(method_records):
        if method == reference:
            continue
        paired_contrasts[method] = {}
        for metric_index, metric in enumerate(HANDCRAFTED_METRICS):
            # Each draw uses one episode index vector for both policies and pools
            # the selected episodes before forming a metric contrast.
            rng_seed = seed + 10000 + method_index * 100 + metric_index
            draws = bootstrap_episode_indices(len(reference_ids), replicates, rng_seed)
            samples: list[float] = []
            for draw in draws:
                left = handcrafted_metric_view(
                    _pool_handcrafted(method_records[method], draw, None)
                )[metric]
                right = handcrafted_metric_view(
                    _pool_handcrafted(method_records[reference], draw, None)
                )[metric]
                samples.append(float(left) - float(right))
            left_point = methods[method]["aggregate"][metric]
            right_point = methods[reference]["aggregate"][metric]
            paired_contrasts[method][metric] = {
                "contrast": f"{method} - {reference}",
                "point_estimate": left_point - right_point,
                "confidence_interval": bootstrap_interval(samples, confidence_level),
                "replicates": replicates,
                "sampling_unit": "paired_episode",
            }

    return {
        "episode_count": len(reference_ids),
        "transition_observation_count": evaluation["handcrafted"]["transitions"],
        "transition_patch_observation_count": next(iter(methods.values()))["aggregate"][
            "transition_patch_count"
        ],
        "methods": summaries,
        "paired_contrasts_vs_independent": paired_contrasts,
        "failure_analysis": _failure_analysis(summaries),
        "bootstrap": {
            "method": "percentile cluster bootstrap",
            "sampling_unit": "episode",
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
            "paired_comparisons_share_episode_indices": True,
            "interpretation": (
                "Intervals describe episode-sampling uncertainty for this "
                "synthetic partition. They are not significance tests and do "
                "not include run-to-run timing variability."
            ),
        },
    }


def _failure_analysis(method_summaries: Mapping[str, Any]) -> dict[str, Any]:
    by_transformation: dict[str, Any] = {}
    categories = tuple(next(iter(method_summaries.values()))["by_transformation"])
    for transformation in categories:
        methods = {
            method: summary["by_transformation"][transformation]
            for method, summary in method_summaries.items()
        }
        observed = [
            (method, item) for method, item in methods.items() if item["episode_count"]
        ]
        by_transformation[transformation] = {
            "method_metrics": {
                method: {
                    "episode_count": item["episode_count"],
                    "f1": item["point_estimates"].get("f1", 0.0),
                    "recall": item["point_estimates"].get("recall", 0.0),
                    "missed_change_rate": item["point_estimates"].get(
                        "missed_change_rate", 0.0
                    ),
                    "false_refresh_rate": item["point_estimates"].get(
                        "false_refresh_rate", 0.0
                    ),
                    "transition_refresh_fraction": item["point_estimates"].get(
                        "transition_refresh_fraction", 0.0
                    ),
                    "confidence_intervals": item["confidence_intervals"],
                }
                for method, item in methods.items()
            },
            "highest_observed_missed_change_rate_method": (
                max(
                    observed,
                    key=lambda item: item[1]["point_estimates"]["missed_change_rate"],
                )[0]
                if observed
                else None
            ),
        }
    return {
        "by_transformation": by_transformation,
        "interpretation": (
            "Transformation strata are generated synthetic categories with "
            "rendered-pixel masks. Differences describe these configured "
            "traces and are not semantic recognition results."
        ),
        "global_fallback_limitations": [
            (
                "The global detector uses changed-pixel area fraction and does "
                "not measure change magnitude, semantic importance, or feature "
                "error."
            ),
            (
                "A spatially small but locally important change can remain "
                "below the 0.50 global-area trigger; local patch thresholding "
                "still governs that case."
            ),
            (
                "A widespread small intensity change can exceed the global "
                "threshold and force a full handcrafted feature refresh."
            ),
        ],
    }


def _pretrained_episode_metric(
    episodes: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> dict[str, float]:
    selected = [episodes[index] for index in indices]
    frames = sum(item["total_frames"] for item in selected)
    independent_calls = sum(item["independent_forward_calls"] for item in selected)
    cache_calls = sum(item["stateful_forward_calls"] for item in selected)
    hits = sum(item["cache_hits"] for item in selected)
    independent_latency = sum(
        item["independent_timing"]["median_seconds"] for item in selected
    )
    cached_latency = sum(item["cached_timing"]["median_seconds"] for item in selected)
    weights = [item["total_frames"] for item in selected]
    weight_sum = sum(weights)
    return {
        "forward_call_reduction": (independent_calls - cache_calls) / independent_calls
        if independent_calls
        else 0.0,
        "cache_hit_rate": hits / frames if frames else 0.0,
        "episode_median_latency_speedup": independent_latency / cached_latency
        if cached_latency
        else 0.0,
        "feature_mean_absolute_difference": sum(
            item["feature_equivalence"]["mean_absolute_difference"] * weight
            for item, weight in zip(selected, weights, strict=True)
        )
        / weight_sum
        if weight_sum
        else 0.0,
        "feature_maximum_absolute_difference": max(
            item["feature_equivalence"]["maximum_absolute_difference"]
            for item in selected
        ),
        "feature_minimum_cosine_similarity": min(
            item["feature_equivalence"]["cosine_similarity"] for item in selected
        ),
    }


def _pretrained_statistics(
    evaluation: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    section = config["statistics"]
    replicates = int(section["replicates"])
    confidence = float(section["confidence_level"])
    seed_base = int(section["pretrained_seed"])
    output: dict[str, Any] = {}
    for condition_index, (condition, summary) in enumerate(
        evaluation["pretrained_dinov2"]["conditions"].items()
    ):
        episodes = summary["per_episode"]
        draws = bootstrap_episode_indices(
            len(episodes), replicates, seed_base + condition_index
        )
        distributions: dict[str, list[float]] = {
            name: []
            for name in (
                "forward_call_reduction",
                "cache_hit_rate",
                "episode_median_latency_speedup",
                "feature_mean_absolute_difference",
                "feature_maximum_absolute_difference",
                "feature_minimum_cosine_similarity",
            )
        }
        for draw in draws:
            metrics = _pretrained_episode_metric(episodes, draw)
            for metric, value in metrics.items():
                distributions[metric].append(value)
        point = _pretrained_episode_metric(episodes, tuple(range(len(episodes))))
        independent_episode_sds = [
            item["independent_timing"]["sample_standard_deviation_seconds"]
            for item in episodes
        ]
        cached_episode_sds = [
            item["cached_timing"]["sample_standard_deviation_seconds"]
            for item in episodes
        ]
        output[condition] = {
            "episode_count": len(episodes),
            "transition_observation_count": sum(
                max(0, item["total_frames"] - 1) for item in episodes
            ),
            "point_estimates": point,
            "measured_end_to_end_speedup": summary["median_speedup"],
            "confidence_intervals": {
                metric: bootstrap_interval(values, confidence)
                for metric, values in distributions.items()
            },
            "measurement_variability": {
                "unit": "timed repetition",
                "timed_repetitions_per_policy_episode": evaluation["pretrained_dinov2"][
                    "timed_repetitions_per_policy_episode"
                ],
                (
                    "independent_full_condition_latency_sample_standard_deviation_seconds"
                ): summary["independent_processing_latency"][
                    "sample_standard_deviation_seconds"
                ],
                (
                    "cached_full_condition_latency_sample_standard_deviation_seconds"
                ): summary["cached_processing_latency"][
                    "sample_standard_deviation_seconds"
                ],
                (
                    "mean_within_episode_independent_latency_sample_sd_seconds"
                ): statistics.fmean(independent_episode_sds),
                (
                    "mean_within_episode_cached_latency_sample_sd_seconds"
                ): statistics.fmean(cached_episode_sds),
                "meaning": (
                    "Full-condition standard deviations describe repeated "
                    "end-to-end workload measurements. Mean within-episode "
                    "standard deviations summarize timing variation on fixed "
                    "episode inputs; neither is the bootstrap interval."
                ),
            },
        }
    return {
        "partition": "synthetic_validation",
        "conditions": output,
        "bootstrap": {
            "method": "percentile paired episode-cluster bootstrap",
            "sampling_unit": "episode within condition",
            "replicates": replicates,
            "confidence_level": confidence,
            "seed": seed_base,
            "independent_and_cache_comparison_uses_same_episodes": True,
            "interpretation": (
                "Intervals describe episode-sampling uncertainty among four "
                "synthetic episodes per condition. Repeated timing variation "
                "is reported separately."
            ),
        },
    }


def analyze_final_results(
    input_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path | None = None,
    figure_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read the actual runner artifact, validate provenance, and write analyses."""
    try:
        evaluation = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read evaluation result: {error}") from error
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation result must be a JSON object")
    validate_result_provenance(evaluation)
    config = load_final_config(config_path)
    if evaluation.get("config_sha256") != config["_sha256"]:
        raise ValueError("evaluation result config hash does not match analysis config")
    if evaluation.get("handcrafted", {}).get("methods") is None:
        raise ValueError("evaluation result is missing handcrafted experiment output")
    handcrafted = _handcrafted_statistics(evaluation, config)
    pretrained = _pretrained_statistics(evaluation, config)
    statistics_result = {
        "schema_version": 1,
        "source_evaluation": str(Path(input_path).resolve()),
        "source_config": config["_path"],
        "source_config_sha256": config["_sha256"],
        "data_provenance": evaluation["data_provenance"],
        "handcrafted": handcrafted,
        "pretrained_dinov2": pretrained,
        "uncertainty_statement": (
            "Confidence intervals resample episodes, preserving frame "
            "dependence within each episode and pairing method comparisons by "
            "episode. They do not establish statistical significance. Timing "
            "repeat standard deviations describe measurement variability, a "
            "different source of uncertainty."
        ),
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else _resolve(config["outputs"]["statistics_json"])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(statistics_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_figures = (
        Path(figure_dir) if figure_dir is not None else REPOSITORY_ROOT / "docs/figures"
    )
    write_accuracy_figure(
        statistics_result, output_figures / "final_accuracy_comparison.png"
    )
    write_efficiency_figure(
        evaluation,
        statistics_result,
        output_figures / "final_efficiency_comparison.png",
    )
    return statistics_result


def write_accuracy_figure(
    statistics_result: Mapping[str, Any], path: str | Path
) -> Path:
    """Create an overall transition F1 figure with episode-bootstrap intervals."""
    from PIL import Image, ImageDraw

    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = "Held-out synthetic patch-change detection (previously inspected)"
    draw.text((42, 28), title, fill="#17212b")
    draw.text(
        (42, 58),
        "Transition-only F1; pixel-change masks are not semantic labels. "
        "Bars show 95% episode-bootstrap intervals.",
        fill="#424b55",
    )
    methods = statistics_result["handcrafted"]["methods"]
    colors = ("#566573", "#2471a3", "#d68910", "#7d3c98")
    left, right, top, bottom = 300, 1100, 125, 610
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    for tick in range(0, 11):
        value = tick / 10
        x = left + int((right - left) * value)
        draw.line((x, top, x, bottom), fill="#e5e7e9", width=1)
        draw.text((x - 8, bottom + 9), f"{value:.1f}", fill="#424b55")
    method_names = list(methods)
    band = (bottom - top) / max(1, len(method_names))
    for index, name in enumerate(method_names):
        y_center = top + int(band * (index + 0.5))
        item = methods[name]
        point = float(item["point_estimates"]["f1"])
        interval = item["overall_confidence_intervals"]["f1"]
        x0 = left + int((right - left) * max(0.0, interval["lower"]))
        x1 = left + int((right - left) * min(1.0, interval["upper"]))
        xp = left + int((right - left) * point)
        draw.text((42, y_center - 8), name.replace("_", " "), fill="#17212b")
        draw.rectangle(
            (left, y_center - 17, xp, y_center + 17), fill=colors[index % len(colors)]
        )
        draw.line((x0, y_center, x1, y_center), fill="#111111", width=3)
        draw.line((x0, y_center - 8, x0, y_center + 8), fill="#111111", width=2)
        draw.line((x1, y_center - 8, x1, y_center + 8), fill="#111111", width=2)
        draw.text((min(xp + 10, 1040), y_center - 8), f"{point:.3f}", fill="#17212b")
    draw.text((left, 660), "F1", fill="#17212b")
    draw.text(
        (42, 690),
        "Thirty fixed synthetic test episodes; earlier benchmark work inspected "
        "this partition. No policy was tuned from this Task 17 run.",
        fill="#424b55",
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def write_efficiency_figure(
    evaluation: Mapping[str, Any],
    statistics_result: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Plot handcrafted patch-operation ratios separately from DINOv2 forwards."""
    from PIL import Image, ImageDraw

    width, height = 1320, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (40, 26), "Computation and latency are separate quantities", fill="#17212b"
    )
    draw.text(
        (40, 56),
        "RGB bars: handcrafted patch-feature operations / independent. DINO "
        "bars: complete neural forward calls avoided.",
        fill="#424b55",
    )
    hand = evaluation["handcrafted"]["methods"]
    hand_names = [name for name in hand if name != "independent_encoding"]
    colors = ("#2471a3", "#d68910", "#7d3c98")
    draw.text(
        (42, 104), "Handcrafted RGB-statistics - patch operation proxy", fill="#17212b"
    )
    left, right, top, bottom = 370, 1180, 145, 400
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    for tick in range(0, 11):
        value = tick / 10
        x = left + int((right - left) * value)
        draw.line((x, top, x, bottom), fill="#e5e7e9", width=1)
        draw.text((x - 8, bottom + 5), f"{value:.1f}", fill="#424b55")
    band = (bottom - top) / max(1, len(hand_names))
    for index, name in enumerate(hand_names):
        y = top + int(band * (index + 0.5))
        ratio = float(hand[name]["aggregate"]["all_frame_refresh_fraction"])
        draw.text((42, y - 8), name.replace("_", " "), fill="#17212b")
        draw.rectangle(
            (left, y - 15, left + int((right - left) * min(1.0, ratio)), y + 15),
            fill=colors[index],
        )
        draw.text((right + 10, y - 8), f"{ratio:.3f}", fill="#17212b")
    draw.text(
        (left, bottom + 27),
        "Fraction of independent patch-feature operations",
        fill="#17212b",
    )

    draw.text((42, 465), "Pretrained DINOv2 - whole-frame cache", fill="#17212b")
    pretrained = evaluation["pretrained_dinov2"]["conditions"]
    conditions = list(pretrained)
    left2, right2, top2, bottom2 = 370, 1180, 505, 760
    draw.line((left2, bottom2, right2, bottom2), fill="#333333", width=2)
    for tick in range(0, 11):
        value = tick / 10
        x = left2 + int((right2 - left2) * value)
        draw.line((x, top2, x, bottom2), fill="#e5e7e9", width=1)
        draw.text((x - 8, bottom2 + 5), f"{value:.1f}", fill="#424b55")
    band2 = (bottom2 - top2) / max(1, len(conditions))
    for index, name in enumerate(conditions):
        item = pretrained[name]
        y = top2 + int(band2 * (index + 0.5))
        reduction = float(item["forward_call_reduction"])
        draw.text((42, y - 10), item["label"], fill="#17212b")
        draw.rectangle(
            (left2, y - 18, left2 + int((right2 - left2) * min(1.0, reduction)), y - 2),
            fill="#117864",
        )
        speedup = item["median_speedup"]
        draw.text(
            (left2, y + 3),
            f"calls avoided {reduction:.1%}; measured latency speedup {speedup:.3f}x",
            fill="#17212b",
        )
    draw.text(
        (left2, bottom2 + 27),
        "Fraction of complete model forwards avoided; speedup text is measured "
        "CPU wall time",
        fill="#17212b",
    )
    draw.text(
        (42, 820),
        "Synthetic data only. RGB operation proxies and DINOv2 forward calls "
        "are not directly comparable; cache hits are not speedup.",
        fill="#424b55",
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination
