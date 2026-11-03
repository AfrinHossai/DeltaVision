"""Tests for policy execution and CPU latency result aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltavision.encoder import RGBImage, encode_image
from deltavision.latency import (
    POLICY_NAMES,
    EpisodeWorkload,
    audit_policy,
    deterministic_policy_orders,
    load_validation_workload,
    run_cpu_latency_benchmark,
    summarize_timings,
)
from deltavision.synthetic import generate_sequences


def _solid_image(rgb: tuple[int, int, int]) -> RGBImage:
    return RGBImage(16, 16, bytes(rgb) * (16 * 16))


def test_deterministic_policy_ordering_rotates_full_method_set() -> None:
    first = deterministic_policy_orders(POLICY_NAMES, 8, seed=492)
    second = deterministic_policy_orders(POLICY_NAMES, 8, seed=492)

    assert first == second
    assert all(set(order) == set(POLICY_NAMES) for order in first)
    assert len(set(first)) > 1


def test_policy_audits_handle_identical_episodes_and_cache_resets() -> None:
    image = _solid_image((30, 80, 140))
    reference = encode_image(image, patch_size=16).features
    episode_a = EpisodeWorkload("validation_a", (image, image), (reference, reference))
    episode_b = EpisodeWorkload("validation_b", (image, image), (reference, reference))
    workload = (episode_a, episode_b)

    independent = audit_policy(workload, "independent_frame_encoding", patch_size=16)
    exact = audit_policy(workload, "exact_patch_reuse", patch_size=16)
    local = audit_policy(workload, "approximate_reuse_0.05", patch_size=16)
    global_policy = audit_policy(
        workload, "approximate_reuse_0.05_global_fallback", patch_size=16
    )

    assert independent.total_patch_count == 4
    assert independent.recomputed_patches == 4
    assert independent.reused_patches == 0
    for result in (exact, local, global_policy):
        assert result.total_patch_count == 4
        assert result.recomputed_patches == 2
        assert result.reused_patches == 2
        assert result.feature_mean_absolute_error == 0.0
    assert independent.peak_cache_memory_estimate_bytes == 0
    assert exact.peak_cache_memory_estimate_bytes > 0
    assert (
        global_policy.peak_cache_memory_estimate_bytes
        > exact.peak_cache_memory_estimate_bytes
    )


def test_timing_summary_validates_statistics_and_paired_speedup() -> None:
    summaries = summarize_timings(
        {
            "independent_frame_encoding": (0.010, 0.020, 0.030),
            "exact_patch_reuse": (0.005, 0.010, 0.060),
        }
    )

    baseline = summaries["independent_frame_encoding"]
    exact = summaries["exact_patch_reuse"]
    assert baseline["median_latency_seconds"] == 0.020
    assert exact["median_latency_seconds"] == 0.010
    assert exact["mean_latency_seconds"] == pytest.approx(0.025)
    assert exact["standard_deviation_seconds"] > 0
    assert exact["p95_latency_seconds"] > exact["median_latency_seconds"]
    assert exact["median_speedup_relative_to_independent"] == 2.0
    assert exact["ratio_of_median_latencies_relative_to_independent"] == 2.0

    with pytest.raises(ValueError, match="same repetition count"):
        summarize_timings(
            {
                "independent_frame_encoding": (0.01, 0.02),
                "exact_patch_reuse": (0.01,),
            }
        )


def test_validation_benchmark_serializes_raw_timings_and_figure(
    tmp_path: Path,
) -> None:
    manifest = generate_sequences(
        tmp_path / "validation",
        width=32,
        height=32,
        sequence_length=2,
        num_episodes=2,
        seed=814,
        transformations=("unchanged",),
        episode_id_prefix="validation",
    )
    output_json = tmp_path / "results" / "latency.json"
    figure_path = tmp_path / "figures" / "latency.png"

    result = run_cpu_latency_benchmark(
        manifest,
        output_json,
        figure_path,
        patch_size=16,
        warmup_repetitions=5,
        timed_repetitions=30,
        policy_order_seed=948,
        expected_episode_count=2,
        expected_frames_per_episode=2,
    )
    serialized = json.loads(output_json.read_text(encoding="utf-8"))
    feature_reference = result["feature_vectors"]
    feature_path = output_json.parent / feature_reference["relative_path"]
    feature_records = [
        json.loads(line)
        for line in feature_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result == serialized
    assert result["partition"] == "validation"
    assert result["test_partition_accessed"] is False
    assert result["frame_count"] == 4
    assert result["feature_dimensions"] == 6
    assert result["processing_cost_proxy"]["unit"] == "patch_operations"
    assert "not a neural FLOP count" in result["processing_cost_proxy"]["definition"]
    assert feature_reference["format"] == "JSONL"
    assert not Path(feature_reference["relative_path"]).is_absolute()
    assert len(feature_records) == 4
    assert feature_records[0]["episode_id"] == "validation_episode_0000"
    assert feature_records[0]["frame_index"] == 0
    assert feature_records[0]["patch_grid_dimensions"] == {"rows": 2, "columns": 2}
    assert set(feature_records[0]["features_by_policy"]) == set(POLICY_NAMES)
    for feature_batch in feature_records[0]["features_by_policy"].values():
        assert len(feature_batch) == 4
        assert all(len(vector) == 6 for vector in feature_batch)
    assert (
        feature_records[0]["features_by_policy"]["exact_patch_reuse"]
        == feature_records[0]["features_by_policy"]["independent_frame_encoding"]
    )
    assert len(result["warmup_policy_orders"]) == 5
    assert len(result["raw_timing_repetitions"]) == 30
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for policy_name in POLICY_NAMES:
        policy = result["policies"][policy_name]
        assert policy["median_latency_seconds"] > 0
        assert policy["mean_latency_seconds"] > 0
        assert policy["standard_deviation_seconds"] >= 0
        assert policy["p95_latency_seconds"] > 0
        assert policy["median_speedup_relative_to_independent"] > 0
        assert (
            len(
                [
                    row["latency_seconds_by_policy"][policy_name]
                    for row in result["raw_timing_repetitions"]
                ]
            )
            == 30
        )


def test_invalid_benchmark_repetition_counts_and_policy_names_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = generate_sequences(
        tmp_path / "validation",
        width=16,
        height=16,
        sequence_length=2,
        num_episodes=1,
        seed=20,
        transformations=("unchanged",),
        episode_id_prefix="validation",
    )
    with pytest.raises(ValueError, match="five warm-up"):
        run_cpu_latency_benchmark(manifest, warmup_repetitions=4)
    with pytest.raises(ValueError, match="thirty timed"):
        run_cpu_latency_benchmark(manifest, timed_repetitions=29)
    with pytest.raises(ValueError, match="unknown policy"):
        audit_policy((), "unknown_policy")

    nonvalidation_manifest = generate_sequences(
        tmp_path / "not_validation",
        width=16,
        height=16,
        sequence_length=1,
        num_episodes=1,
        seed=21,
        transformations=("unchanged",),
    )
    with pytest.raises(ValueError, match="validation-prefixed"):
        load_validation_workload(nonvalidation_manifest)
