"""Tests for the controlled patch-size and temporal-change ablation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import pytest

from deltavision.ablation import (
    CONDITION_NAMES,
    METHOD_NAMES,
    PATCH_SIZES,
    generate_ablation_scenarios,
    run_ablation_study,
    validate_ablation_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _default_config() -> dict[str, object]:
    return json.loads(
        (REPOSITORY_ROOT / "configs" / "ablation_study.json").read_text(
            encoding="utf-8"
        )
    )


def _small_config() -> dict[str, object]:
    config = _default_config()
    config["image_width"] = 16
    config["image_height"] = 16
    config["sequence_length"] = 8
    config["episodes_per_condition"] = 4
    return config


def _write_config(path: Path, config: dict[str, object]) -> Path:
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_scenario_generation_is_reproducible_and_condition_specific(
    tmp_path: Path,
) -> None:
    config = _default_config()
    first = generate_ablation_scenarios(tmp_path / "first", config)
    second = generate_ablation_scenarios(tmp_path / "second", config)

    assert tuple(scenario.name for scenario in first) == CONDITION_NAMES
    assert tuple(scenario.profile for scenario in first) == tuple(
        scenario.profile for scenario in second
    )
    assert len(
        {episode.episode_id for scenario in first for episode in scenario.episodes}
    ) == sum(len(scenario.episodes) for scenario in first)
    for first_scenario, second_scenario in zip(first, second, strict=True):
        assert first_scenario.manifest_path.read_bytes() == (
            second_scenario.manifest_path.read_bytes()
        )
        assert [
            frame.pixels
            for episode in first_scenario.episodes
            for frame in episode.frames
        ] == [
            frame.pixels
            for episode in second_scenario.episodes
            for frame in episode.frames
        ]

    profiles = {scenario.name: scenario.profile for scenario in first}
    no_change = profiles["no_visual_changes"]
    sparse = profiles["sparse_local_changes"]
    moderate = profiles["moderate_local_changes"]
    widespread = profiles["frequent_widespread_changes"]
    assert no_change["changed_transition_count"] == 0
    assert 0 < sparse["changed_transition_count"] < sparse["transition_count"]
    assert moderate["changed_transition_count"] == moderate["transition_count"]
    assert moderate["mean_changed_pixel_fraction"] < 0.5
    assert widespread["changed_transition_count"] == widespread["transition_count"]
    assert widespread["mean_changed_pixel_fraction"] >= 0.5


def test_patch_size_and_method_configurations_are_fixed() -> None:
    config = _default_config()
    validate_ablation_config(config)

    assert tuple(config["patch_sizes"]) == PATCH_SIZES
    assert tuple(config["methods"]) == METHOD_NAMES
    config["patch_sizes"] = [8, 16]
    with pytest.raises(ValueError, match="patch_sizes"):
        validate_ablation_config(config)


def test_timing_aggregation_exact_reuse_and_serialization(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "ablation.json", _small_config())
    json_path = tmp_path / "results" / "ablation.json"
    csv_path = tmp_path / "results" / "ablation.csv"
    patch_figure = tmp_path / "figures" / "patch.png"
    change_figure = tmp_path / "figures" / "change.png"

    result = run_ablation_study(
        config_path,
        data_dir=tmp_path / "data",
        output_json=json_path,
        output_csv=csv_path,
        patch_size_figure=patch_figure,
        change_frequency_figure=change_figure,
    )
    serialized = json.loads(json_path.read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))

    assert result == serialized
    assert result["partition"] == "validation"
    assert result["test_partition_accessed"] is False
    assert "initial fully encoded" in result["refresh_fraction_definition"]
    assert "same patch size" in result["feature_mae_definition"]
    assert len(result["results"]) == 4 * len(PATCH_SIZES) * len(METHOD_NAMES)
    assert len(csv_rows) == len(result["results"])
    assert {int(row["patch_size"]) for row in csv_rows} == set(PATCH_SIZES)
    assert {row["condition"] for row in csv_rows} == set(CONDITION_NAMES)
    assert patch_figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert change_figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    for group in result["configuration_raw_timings"]:
        raw = group["raw_timing_repetitions"]
        assert len(group["warmup_policy_orders"]) == 5
        assert len(raw) == 20
        baseline_samples = [
            row["latency_seconds_by_method"]["independent_frame_encoding"]
            for row in raw
        ]
        baseline_median = statistics.median(baseline_samples)
        for method in METHOD_NAMES:
            row = group["methods"][method]
            samples = [item["latency_seconds_by_method"][method] for item in raw]
            assert row["median_processing_latency_seconds"] == statistics.median(
                samples
            )
            assert row["mean_processing_latency_seconds"] == pytest.approx(
                statistics.fmean(samples)
            )
            assert row[
                "processing_latency_standard_deviation_seconds"
            ] == pytest.approx(statistics.stdev(samples))
            assert row["p95_processing_latency_seconds"] > 0
            assert row["number_of_frames"] == 32
            assert row["total_patches"] > 0
            assert row["refresh_fraction"] == pytest.approx(
                row["recomputed_patches"] / row["total_patches"]
            )
            if method == "exact_patch_reuse":
                assert row["feature_mae"] == 0.0
            if method == "independent_frame_encoding":
                assert row["median_latency_ratio_to_independent"] == 1.0
                assert row["cache_memory_estimate_bytes"] == 0
            else:
                assert row["relative_speedup"] == pytest.approx(
                    statistics.median(
                        baseline / current
                        for baseline, current in zip(
                            baseline_samples, samples, strict=True
                        )
                    )
                )
                assert row["median_latency_ratio_to_independent"] == pytest.approx(
                    row["median_processing_latency_seconds"] / baseline_median
                )
                assert row["speed_category"] in {"faster", "equal", "slower"}


def test_validation_config_rejects_test_partition_and_insufficient_repetitions() -> (
    None
):
    config = _default_config()
    config["partition"] = "test"
    with pytest.raises(ValueError, match="validation partition"):
        validate_ablation_config(config)

    config = _default_config()
    config["timed_repetitions"] = 19
    with pytest.raises(ValueError, match="timed_repetitions"):
        validate_ablation_config(config)
