"""Provenance-aware summary of validation-only DeltaVision experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAFE_JSON_SOURCES = {
    "handcrafted_latency": "results/latency_benchmark.json",
    "handcrafted_global_change": "results/global_change_validation.json",
    "handcrafted_patch_ablation": "results/ablation_study.json",
    "legacy_threshold_demo": "results/threshold_sweep.json",
    "legacy_change_detection_demo": "results/change_detection.json",
    "pretrained_task15": "results/pretrained_cache_evaluation.json",
    "pretrained_capacity_task16": "results/pretrained_cache_ablation.json",
    "pretrained_task14_metadata": "results/pretrained_stateful/run_metadata.json",
    "pretrained_task14_metrics": "results/pretrained_stateful/stateful_metrics.json",
}
HELD_OUT_REPORTS_NOT_OPENED = (
    "results/research_benchmark.json",
    "results/research_benchmark.csv",
)


def load_comparison_sources(
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Read allowlisted validation or demo reports, never held-out reports."""
    root = Path(repository_root).resolve()
    sources: dict[str, Any] = {}
    for source_id, relative_path in SAFE_JSON_SOURCES.items():
        path = root / relative_path
        try:
            sources[source_id] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"could not read required source {relative_path}: {error}"
            ) from error
    return sources


def validate_validation_provenance(
    report: Mapping[str, Any], source_path: str, *, source_id: str
) -> None:
    """Reject any evidence not explicitly tied to validation-only inputs."""
    if report.get("partition") == "validation":
        if report.get("test_partition_accessed") is True:
            raise ValueError(f"{source_id} reports held-out test access")
        if report.get("held_out_test_partition_accessed") is True:
            raise ValueError(f"{source_id} reports held-out test access")
        manifest = report.get("manifest_path")
        if isinstance(manifest, str) and "validation" not in Path(manifest).parts:
            raise ValueError(f"{source_id} manifest is not validation-prefixed")
        return
    if (
        report.get("validation_only") is True
        and report.get("held_out_test_partition_accessed") is False
    ):
        return
    raise ValueError(f"{source_id} does not establish validation-only provenance")


def _method_row(
    policy: str, metrics: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    return {
        "method": policy,
        "median_latency_seconds_per_240_frame_pass": metrics["median_latency_seconds"],
        "median_speedup_relative_to_independent": metrics[
            "median_speedup_relative_to_independent"
        ],
        "patch_operations_recomputed": metrics["all_frame_recomputed_patches"],
        "patch_operation_fraction": metrics["all_frame_patch_recomputation_fraction"],
        "feature_mean_absolute_error": metrics["feature_mean_absolute_error"],
        "feature_maximum_absolute_error": metrics["feature_maximum_absolute_error"],
        "cache_memory_estimate_bytes": metrics["peak_cache_memory_estimate_bytes"],
        "source_report": source,
    }


def _change_detection_table(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_validation_provenance(
        report,
        "results/global_change_validation.json",
        source_id="handcrafted_global_change",
    )
    rows = []
    for method in report["methods"]:
        aggregate = method["aggregate"]
        detection = aggregate["change_detection"]
        feature = aggregate["feature_approximation"]
        rows.append(
            {
                "method": method["method"],
                "configuration": method["configuration"],
                "precision": detection["precision"],
                "recall": detection["recall"],
                "f1": detection["f1"],
                "missed_change_rate": detection["missed_change_rate"],
                "transition_refresh_fraction": detection["refresh_fraction"],
                "feature_mean_absolute_error": feature["mean_absolute_feature_error"],
                "feature_maximum_absolute_error": feature[
                    "maximum_absolute_feature_error"
                ],
                "detector_runtime_seconds": aggregate["runtime"][
                    "global_detector_runtime_seconds"
                ],
                "processing_runtime_seconds": aggregate["runtime"][
                    "processing_runtime_seconds"
                ],
                "source_report": "results/global_change_validation.json",
            }
        )
    return rows


def _patch_ablation_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain per-condition and per-patch-size handcrafted measurements."""
    return [
        {
            "condition": result["condition"],
            "condition_label": result["condition_label"],
            "patch_size": result["patch_size"],
            "method": result["method"],
            "frames": result["number_of_frames"],
            "median_latency_seconds": result["median_latency_seconds"],
            "median_speedup_relative_to_independent": result[
                "median_speedup_relative_to_independent"
            ],
            "recomputed_patches": result["recomputed_patches"],
            "total_patches": result["total_patches"],
            "feature_mean_absolute_error": result["feature_mae"],
            "cache_memory_estimate_bytes": result["cache_memory_estimate_bytes"],
            "source_report": "results/ablation_study.json",
        }
        for result in report["results"]
    ]


def _pretrained_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_validation_provenance(
        report,
        "results/pretrained_cache_evaluation.json",
        source_id="pretrained_task15",
    )
    rows = []
    for name, condition in report["condition_results"].items():
        rows.append(
            {
                "condition": name,
                "label": condition["label"],
                "partition": "validation",
                "frames_per_pass": condition["total_frames_per_pass"],
                "independent_full_model_calls": condition["independent"][
                    "model_forward_calls_per_pass"
                ],
                "exact_cache_capacity": condition["exact_cache"][
                    "cache_capacity_per_episode"
                ],
                "cached_full_model_calls": condition["exact_cache"][
                    "model_forward_calls_per_pass"
                ],
                "cached_frame_hits": condition["exact_cache"][
                    "cached_frame_hits_per_pass"
                ],
                "cache_hit_rate": condition["exact_cache"]["cache_hit_rate"],
                "independent_mean_latency_seconds": condition["independent"][
                    "mean_processing_latency"
                ]["mean_seconds"],
                "cached_mean_latency_seconds": condition["exact_cache"][
                    "mean_processing_latency"
                ]["mean_seconds"],
                "independent_over_cached_speedup_ratio": condition[
                    "latency_comparison"
                ]["independent_over_exact_cache_speedup_ratio"],
                "feature_shape": condition["exact_cache"]["feature_shape"],
                "feature_maximum_absolute_difference": condition["feature_equivalence"][
                    "maximum_absolute_difference"
                ],
                "feature_mean_absolute_difference": condition["feature_equivalence"][
                    "mean_absolute_difference"
                ],
                "feature_cosine_similarity": condition["feature_equivalence"][
                    "cosine_similarity"
                ],
                "feature_equivalent_within_tolerance": condition["feature_equivalence"][
                    "within_documented_tolerance"
                ],
                "cache_peak_tensor_payload_bytes": condition["exact_cache"][
                    "maximum_peak_cache_memory_estimate_bytes"
                ],
                "source_report": "results/pretrained_cache_evaluation.json",
            }
        )
    return rows


def build_final_comparison(
    sources: Mapping[str, Any], *, repository_root: str | Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    """Create separate handcrafted and learned-representation validation tables."""
    root = Path(repository_root).resolve()
    latency = sources["handcrafted_latency"]
    patch_ablation = sources["handcrafted_patch_ablation"]
    global_change = sources["handcrafted_global_change"]
    task15 = sources["pretrained_task15"]
    capacity = sources["pretrained_capacity_task16"]
    validate_validation_provenance(
        latency, "results/latency_benchmark.json", source_id="handcrafted_latency"
    )
    validate_validation_provenance(
        patch_ablation,
        "results/ablation_study.json",
        source_id="handcrafted_patch_ablation",
    )
    validate_validation_provenance(
        global_change,
        "results/global_change_validation.json",
        source_id="handcrafted_global_change",
    )
    validate_validation_provenance(
        capacity,
        "results/pretrained_cache_ablation.json",
        source_id="pretrained_capacity_task16",
    )
    policies = latency["policies"]
    handcrafted_rows = [
        _method_row(name, metrics, source="results/latency_benchmark.json")
        for name, metrics in policies.items()
    ]
    validation_pretrained_rows = _pretrained_rows(task15)
    dino_config = task15["evaluation_config"]
    latency_config = json.loads(
        (root / "configs/cpu_latency_benchmark.json").read_text(encoding="utf-8")
    )
    ablation_config = patch_ablation["configuration"]
    global_transition_table = _change_detection_table(global_change)
    prior_metadata = sources["pretrained_task14_metadata"]
    prior_metrics = sources["pretrained_task14_metrics"]
    task14_hardware = prior_metadata.get("hardware", {})
    threshold_demo = sources["legacy_threshold_demo"]
    change_demo = sources["legacy_change_detection_demo"]
    capacity_conditions = capacity["condition_results"]

    inventory = [
        {
            "source_report": "results/latency_benchmark.json",
            "configuration": "configs/cpu_latency_benchmark.json",
            "source_script": "scripts/benchmark_cpu_latency.py",
            "partition": "validation",
            "episodes": latency["episode_count"],
            "frames": latency["frame_count"],
            "image_dimensions": [128, 128],
            "patch_size": latency["patch_size"],
            "timed_repetitions_per_policy": latency["timed_repetitions_per_policy"],
            "warmup_repetitions_per_policy": latency["warmup_repetitions_per_policy"],
            "hardware": latency["system"],
            "timing_methodology": latency["timing_methodology"],
            "benchmark_configuration": latency_config,
            "used_in_table": (
                "A: handcrafted RGB-statistics latency and patch-operation metrics"
            ),
        },
        {
            "source_report": "results/ablation_study.json",
            "configuration": "configs/ablation_study.json",
            "source_script": "scripts/run_ablation_study.py",
            "partition": patch_ablation["partition"],
            "episodes_per_condition": ablation_config["episodes_per_condition"],
            "frames_per_episode": ablation_config["sequence_length"],
            "image_dimensions": [
                ablation_config["image_width"],
                ablation_config["image_height"],
            ],
            "patch_sizes": ablation_config["patch_sizes"],
            "warmup_repetitions": ablation_config["warmup_repetitions"],
            "timed_repetitions": ablation_config["timed_repetitions"],
            "hardware": patch_ablation["system"],
            "timing_methodology": patch_ablation["methodology"],
            "used_in_table": "A: patch-size and redundancy sensitivity context",
        },
        {
            "source_report": "results/global_change_validation.json",
            "configuration": "configs/global_change_validation.json",
            "source_script": "scripts/evaluate_global_change.py",
            "partition": global_change["partition"],
            "episodes": global_change["episode_count"],
            "frames": global_change["frame_count"],
            "transitions": global_change["transition_count"],
            "image_dimensions": [128, 128],
            "hardware": "Not recorded in the source report.",
            "timing_methodology": {
                "summary": "One instrumented validation processing pass per method.",
                "detector_runtime": "Recorded separately from processing runtime.",
                "warmups_or_repetitions": "Not recorded in the source report.",
            },
            "patch_size": global_change["patch_size"],
            "methods": [row["method"] for row in global_transition_table],
            "used_in_table": (
                "Separate change-detection comparison; not merged with latency values"
            ),
        },
        {
            "source_report": "results/pretrained_cache_evaluation.json",
            "configuration": "configs/pretrained_cache_evaluation.json",
            "source_script": "scripts/evaluate_pretrained_caching.py",
            "partition": "validation",
            "conditions": len(validation_pretrained_rows),
            "frames_per_condition_pass": dino_config["num_episodes"]
            * dino_config["sequence_length"],
            "image_dimensions": [dino_config["width"], dino_config["height"]],
            "model": task15["model"],
            "hardware": task15["hardware_and_software"],
            "warmups": task15["experiment_protocol"]["untimed_warmup_repetitions"],
            "timed_repetitions": task15["experiment_protocol"]["timing_repetitions"],
            "timing_methodology": task15["experiment_protocol"],
            "used_in_table": "B: controlled pretrained independent-vs-cache results",
        },
        {
            "source_report": "results/pretrained_cache_ablation.json",
            "configuration": "configs/pretrained_cache_ablation.json",
            "source_script": "scripts/ablate_pretrained_cache.py",
            "partition": capacity["partition"],
            "conditions": len(capacity_conditions),
            "cache_capacities": capacity["configuration"]["cache_capacities"],
            "frames_per_condition": {
                name: condition["frame_count_per_pass"]
                for name, condition in capacity_conditions.items()
            },
            "image_dimensions": [
                capacity["configuration"]["width"],
                capacity["configuration"]["height"],
            ],
            "patch_size": capacity["model"]["model_config"]["patch_size"],
            "timing_methodology": capacity["timing_protocol"],
            "model": capacity["model"],
            "hardware": capacity["hardware_and_software"],
            "used_in_table": "B: capacity-sensitivity table and Figure 1",
        },
        {
            "source_report": "results/pretrained_stateful/stateful_metrics.json",
            "configuration": "configs/pretrained_baseline.json",
            "source_script": "scripts/run_pretrained_stateful.py",
            "partition": "synthetic demo; not used in validation-only tables",
            "frames": prior_metrics["cache_metrics"]["total_frames"],
            "image_dimensions": "Not recorded in the Task 14 metadata.",
            "patch_size": prior_metadata["model_config"]["patch_size"],
            "model_identifier": prior_metadata["model_identifier"],
            "model_revision": prior_metrics["model_revision"],
            "preprocessing": prior_metadata["preprocessing"],
            "hardware": task14_hardware,
            "timing_methodology": prior_metadata["timing_scope"],
            "latency_note": (
                "Task 14 includes PPM reads and output writes and used older runtime "
                "versions; not compared with Task 15/16 latency."
            ),
            "used_in_table": "historical implementation provenance only",
        },
        {
            "source_report": "results/threshold_sweep.json",
            "configuration": (
                "Seed-17 synthetic demo manifest; no validation split label"
            ),
            "source_script": "scripts/run_threshold_sweep.py",
            "partition": "synthetic demo; excluded from validation comparison",
            "frames": threshold_demo["frame_count"],
            "image_dimensions": threshold_demo["image_dimensions"],
            "patch_size": threshold_demo["patch_size"],
            "timing_methodology": threshold_demo["latency_definition"],
            "hardware": "Not recorded in the source report.",
            "used_in_table": "provenance inventory only",
        },
        {
            "source_report": "results/change_detection.json",
            "configuration": (
                "Seed-17 synthetic demo manifest; no validation split label"
            ),
            "source_script": "scripts/evaluate_change_detection.py",
            "partition": "synthetic demo; excluded from validation comparison",
            "frames": change_demo["frame_count"],
            "image_dimensions": change_demo.get("image_dimensions", [128, 128]),
            "patch_size": change_demo["patch_size"],
            "timing_methodology": "No timing measurements used in this comparison.",
            "hardware": "Not recorded in the source report.",
            "used_in_table": "provenance inventory only",
        },
    ]
    excluded = [
        {
            "source_report": path,
            "reason": (
                "Repository documentation identifies this aggregate as including "
                "the held-out test partition; it was not opened or used."
            ),
        }
        for path in HELD_OUT_REPORTS_NOT_OPENED
    ]
    return {
        "experiment": (
            "DeltaVision final baseline comparison and cache-capacity ablation"
        ),
        "status": "completed",
        "partition_scope": (
            "Validation-only comparison; demo experiments are separately labeled "
            "provenance context."
        ),
        "held_out_test_partition_accessed": False,
        "comparison_table_A_handcrafted_rgb_statistics": {
            "representation": (
                "Handcrafted six-value RGB means and population standard deviations "
                "per patch; not a learned representation."
            ),
            "dataset": {
                "partition": "validation",
                "episodes": latency["episode_count"],
                "frames": latency["frame_count"],
                "image_dimensions": [128, 128],
                "patch_size": latency["patch_size"],
            },
            "timing_methodology": latency["timing_methodology"],
            "rows": handcrafted_rows,
            "change_detection_metrics_separate_source": {
                "dataset": {
                    "partition": "validation",
                    "episodes": global_change["episode_count"],
                    "frames": global_change["frame_count"],
                    "transitions": global_change["transition_count"],
                    "patch_size": global_change["patch_size"],
                },
                "rows": global_transition_table,
            },
            "patch_size_ablation_provenance": {
                "report": "results/ablation_study.json",
                "patch_sizes": ablation_config["patch_sizes"],
                "conditions": [item["name"] for item in ablation_config["conditions"]],
                "timing_repetitions": ablation_config["timed_repetitions"],
                "metrics_are_patch_operation_proxies": True,
                "rows": _patch_ablation_rows(patch_ablation),
            },
            "observed_reuse_tradeoffs": {
                "exact_patch_reuse": (
                    "Preserved exact RGB-statistics feature values in recorded "
                    "validation runs while reducing patch feature computations."
                ),
                "approximate_patch_reuse": (
                    "Can reduce recomputed patch operations further, but adds "
                    "feature error and similarity-comparison cost."
                ),
                "global_change_fallback": (
                    "Recovered change-detection recall on this validation set at "
                    "additional detector runtime and refresh work."
                ),
            },
        },
        "comparison_table_B_pretrained_dinov2": {
            "representation": (
                "Frozen DINOv2 contextual spatial tokens; semantically distinct "
                "from RGB-statistics patch features."
            ),
            "model": task15["model"],
            "hardware_and_software": task15["hardware_and_software"],
            "dataset": {
                "partition": "validation",
                "conditions": 4,
                "frames_per_condition_pass": dino_config["num_episodes"]
                * dino_config["sequence_length"],
                "image_dimensions": [dino_config["width"], dino_config["height"]],
            },
            "timing_methodology": task15["experiment_protocol"],
            "capacity_8_independent_vs_cache_rows": validation_pretrained_rows,
            "capacity_ablation_source": "results/pretrained_cache_ablation.json",
            "capacity_ablation_protocol": capacity["timing_protocol"],
            "capacity_ablation_dataset": {
                "partition": capacity["partition"],
                "image_dimensions": [
                    capacity["configuration"]["width"],
                    capacity["configuration"]["height"],
                ],
                "conditions": list(capacity_conditions),
            },
            "capacity_ablation_conditions": capacity_conditions,
        },
        "cross_experiment_interpretation": {
            "absolute_latency_comparison_across_tables": False,
            "relative_latency_ratios": (
                "Ratios are within their named workload and source experiment; "
                "they do not make workloads or representations equivalent."
            ),
            "patch_operation_proxy": (
                "Counts handcrafted RGB-statistics feature-vector recomputations; "
                "not neural FLOPs or DINOv2 forward calls."
            ),
            "neural_forward_calls": (
                "Counts complete frozen DINOv2 model forwards; one per independent "
                "frame or exact-cache miss."
            ),
            "feature_equivalence": (
                "Numerical agreement only; it is not semantic accuracy."
            ),
            "change_detection": (
                "Reported separately from feature computation and latency; labels "
                "are rendered RGB pixel differences."
            ),
        },
        "global_fallback_limitations": [
            "The global detector uses changed-pixel area fraction, not change "
            "magnitude or semantics.",
            "A change below the configured area threshold does not trigger a "
            "full refresh, even if its local color difference is large.",
            "Widespread flicker or other broad pixel changes can trigger "
            "unnecessary full refreshes.",
            "Local threshold reuse can preserve stale handcrafted features "
            "outside globally triggered frames.",
            "Detector scans add comparison work; refresh and detector metrics are "
            "not end-to-end speedup claims.",
        ],
        "experiment_inventory": inventory,
        "excluded_reports": excluded,
    }


def draw_comparison_figure(
    comparison: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Draw within-study ratios in separate labeled panels, with scope caveats."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1500, 850), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (70, 35),
        "Research baselines: separate validation experiments",
        fill="#111111",
        font=font,
    )
    panels = [
        (
            (70, 100, 710, 690),
            "Handcrafted RGB statistics | 240-frame validation run",
            [
                row
                for row in comparison["comparison_table_A_handcrafted_rgb_statistics"][
                    "rows"
                ]
                if row["method"] != "independent_frame_encoding"
            ],
            "method",
            "median_speedup_relative_to_independent",
        ),
        (
            (790, 100, 1430, 690),
            "Frozen DINOv2 | 32-frame validation conditions",
            comparison["comparison_table_B_pretrained_dinov2"][
                "capacity_8_independent_vs_cache_rows"
            ],
            "label",
            "independent_over_cached_speedup_ratio",
        ),
    ]
    for panel, title, rows, label_key, metric_key in panels:
        left, top, right, bottom = panel
        draw.rectangle(panel, outline="#999999", width=1)
        draw.text((left + 14, top + 12), title, fill="#111111", font=font)
        plot_left, plot_top = left + 65, top + 65
        plot_right, plot_bottom = right - 20, bottom - 95
        values = [float(row[metric_key]) for row in rows]
        maximum = max(2.0, max(values, default=1.0) * 1.12)
        for tick in range(5):
            fraction = tick / 4
            y = plot_bottom - round((plot_bottom - plot_top) * fraction)
            draw.line((plot_left, y, plot_right, y), fill="#e0e0e0", width=1)
            draw.text(
                (left + 7, y - 5),
                f"{maximum * fraction:.1f}x",
                fill="#333333",
                font=font,
            )
        independent_y = plot_bottom - round((plot_bottom - plot_top) / maximum)
        for x in range(plot_left, plot_right, 12):
            draw.line(
                (x, independent_y, min(x + 6, plot_right), independent_y),
                fill="#333333",
                width=1,
            )
        group_height = (plot_bottom - plot_top) / max(1, len(rows))
        for index, row in enumerate(rows):
            y_center = plot_top + group_height * (index + 0.5)
            value = float(row[metric_key])
            bar_end = plot_left + round((plot_right - plot_left) * value / maximum)
            color = "#4C78A8" if label_key == "method" else "#F58518"
            draw.rectangle(
                (plot_left, y_center - 11, bar_end, y_center + 11), fill=color
            )
            label = str(row[label_key]).replace("_", " ")
            draw.text((plot_left, y_center - 26), label[:36], fill="#333333", font=font)
            draw.text(
                (min(bar_end + 5, plot_right - 35), y_center - 6),
                f"{value:.2f}x",
                fill="#222222",
                font=font,
            )
        draw.text(
            (plot_left, bottom - 35),
            "Within-study independent latency / reuse latency",
            fill="#333333",
            font=font,
        )
    draw.text(
        (70, 755),
        "Panels use different inputs and timing protocols. Do not compare absolute "
        "latency or infer semantic equivalence.",
        fill="#444444",
        font=font,
    )
    draw.text(
        (70, 780),
        "Synthetic validation only. Ratios are not natural-video or neural "
        "acceleration claims.",
        fill="#444444",
        font=font,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def write_comparison_outputs(
    report: Mapping[str, Any], output_json: str | Path, figure_path: str | Path
) -> tuple[Path, Path]:
    """Write baseline comparison JSON and its scoped ratio figure."""
    json_path = Path(output_json)
    png_path = Path(figure_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    draw_comparison_figure(report, png_path)
    return json_path, png_path
