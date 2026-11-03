"""Tests for split construction and multi-seed benchmark evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from deltavision.research_benchmark import (
    SPLIT_NAMES,
    build_research_benchmark,
    evaluate_research_benchmark,
    load_research_config,
    validate_research_benchmark,
)
from deltavision.synthetic import TRANSFORMATION_TYPES


def _write_tiny_config(path: Path) -> Path:
    config: dict[str, Any] = {
        "benchmark_name": "test_synthetic_benchmark",
        "image_width": 32,
        "image_height": 32,
        "sequence_length": 8,
        "patch_size": 16,
        "transformations": list(TRANSFORMATION_TYPES),
        "thresholds": [0.01, 0.03, 0.05, 0.1],
        "splits": {
            "train": {
                "episode_count": 8,
                "base_seed": 100,
                "episode_id_prefix": "train",
            },
            "validation": {
                "episode_count": 6,
                "base_seed": 1000,
                "episode_id_prefix": "validation",
            },
            "test": {
                "episode_count": 6,
                "base_seed": 2000,
                "episode_id_prefix": "test",
            },
        },
        "require_all_transformations_per_split": False,
        "bootstrap": {
            "replicates": 40,
            "confidence_level": 0.95,
            "seed": 321,
        },
        "threshold_selection": {
            "method": "none",
            "policy": "Fixed grid; test is not used to select a threshold.",
        },
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_fixed_research_configuration_documents_requested_splits() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/research_benchmark.json"
    )

    config = load_research_config(config_path)

    assert (config.width, config.height, config.sequence_length, config.patch_size) == (
        128,
        128,
        8,
        16,
    )
    assert [config.splits[name].episode_count for name in SPLIT_NAMES] == [100, 30, 30]
    assert [config.splits[name].base_seed for name in SPLIT_NAMES] == [
        17000,
        27100,
        37900,
    ]
    assert config.require_all_transformations_per_split is True


def test_research_benchmark_reproducible_and_split_separated(
    tmp_path: Path,
) -> None:
    config = _write_tiny_config(tmp_path / "config.json")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = build_research_benchmark(config, first_root)
    second = build_research_benchmark(config, second_root)

    first_files = sorted(
        path.relative_to(first_root).as_posix()
        for path in first_root.rglob("*")
        if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second_root).as_posix()
        for path in second_root.rglob("*")
        if path.is_file()
    )
    assert first_files == second_files
    assert all(
        (first_root / name).read_bytes() == (second_root / name).read_bytes()
        for name in first_files
    )
    assert first == second
    assert first["split_validation"]["episode_ids_unique_across_splits"] is True
    assert first["split_validation"]["random_seed_ranges_disjoint"] is True
    assert (
        first["split_validation"]["rendered_frames_have_no_cross_split_duplicates"]
        is True
    )

    ids: set[str] = set()
    seeds: set[int] = set()
    for split_name in SPLIT_NAMES:
        rows = _read_jsonl(first_root / split_name / "manifest.jsonl")
        for row in rows:
            assert row["episode_id"].startswith(f"{split_name}_")
            assert row["episode_id"] not in ids
            assert row["random_seed"] not in seeds
            ids.add(row["episode_id"])
            seeds.add(row["random_seed"])


def test_manifest_records_transformations_and_portable_existing_paths(
    tmp_path: Path,
) -> None:
    config_path = _write_tiny_config(tmp_path / "config.json")
    root = tmp_path / "dataset"
    build_research_benchmark(config_path, root)
    config = load_research_config(config_path)
    validation = validate_research_benchmark(config, root)

    assert validation["portable_frame_and_mask_paths_validated"] is True
    for split_name in SPLIT_NAMES:
        rows = _read_jsonl(root / split_name / "manifest.jsonl")
        assert len(rows) == config.splits[split_name].episode_count
        for row in rows:
            assert len(row["transitions"]) == 7
            for transition in row["transitions"]:
                assert transition["transformation_type"] in TRANSFORMATION_TYPES
                assert isinstance(transition["transformation_parameters"], dict)
                assert (
                    root / split_name / transition["ground_truth_mask_path"]
                ).is_file()


def test_overlap_in_configured_random_seed_ranges_is_rejected(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path / "config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["splits"]["validation"]["base_seed"] = 105
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="seed ranges must not overlap"):
        load_research_config(config_path)


def test_per_transformation_metrics_and_initial_frame_exclusion(
    tmp_path: Path,
) -> None:
    config_path = _write_tiny_config(tmp_path / "config.json")
    root = tmp_path / "dataset"
    build_research_benchmark(config_path, root)
    output_json = tmp_path / "results.json"
    output_csv = tmp_path / "results.csv"

    result = evaluate_research_benchmark(config_path, root, output_json, output_csv)
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert result == loaded
    assert output_csv.read_text(encoding="utf-8").startswith("partition,method,scope,")
    assert result["threshold_selection"]["preferred_threshold"] is None
    assert result["threshold_selection"]["test_results_used_for_selection"] is False

    validation = result["partitions"]["validation"]
    exact = validation["methods"]["exact_reuse"]["aggregate"]
    detection = exact["change_detection"]
    computation = exact["computation"]
    assert detection["transition_patch_count"] == 6 * 7 * 4
    assert computation["total_patch_count"] == 6 * 8 * 4
    assert computation["total_patch_count"] > detection["transition_patch_count"]

    threshold_method = validation["methods"]["threshold_0.05"]
    approximate_detection = threshold_method["aggregate"]["change_detection"]
    per_transformation = threshold_method["per_transformation"]
    assert set(per_transformation) == set(TRANSFORMATION_TYPES)
    for metric_name in (
        "transition_patch_count",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ):
        assert (
            sum(
                metrics["change_detection"][metric_name]
                for metrics in per_transformation.values()
            )
            == approximate_detection[metric_name]
        )
    assert all(
        "bootstrap_confidence_intervals" in metrics
        for metrics in per_transformation.values()
    )


def test_episode_bootstrap_is_reproducible_and_partitions_are_separate(
    tmp_path: Path,
) -> None:
    config_path = _write_tiny_config(tmp_path / "config.json")
    root = tmp_path / "dataset"
    build_research_benchmark(config_path, root)

    first = evaluate_research_benchmark(
        config_path, root, tmp_path / "one.json", tmp_path / "one.csv"
    )
    second = evaluate_research_benchmark(
        config_path, root, tmp_path / "two.json", tmp_path / "two.csv"
    )

    assert first == second
    assert set(first["partitions"]) == set(SPLIT_NAMES)
    assert first["bootstrap"]["sampling_unit"] == "episode"
    for partition in first["partitions"].values():
        intervals = partition["methods"]["threshold_0.03"]["aggregate"][
            "bootstrap_confidence_intervals"
        ]
        assert intervals["recall"]["lower"] <= intervals["recall"]["upper"]


def test_validation_rejects_nonportable_mask_paths(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path / "config.json")
    root = tmp_path / "dataset"
    build_research_benchmark(config_path, root)
    manifest = root / "train" / "manifest.jsonl"
    rows = _read_jsonl(manifest)
    rows[0]["ground_truth_mask_paths"][0] = "../validation/escape.pgm"
    manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative and stay in its split"):
        validate_research_benchmark(load_research_config(config_path), root)
