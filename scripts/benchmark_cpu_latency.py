#!/usr/bin/env python3
"""Benchmark complete in-memory encoding-policy latency on validation data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.latency import (  # noqa: E402
    POLICY_NAMES,
    run_cpu_latency_benchmark,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure in-memory CPU latency for independent, exact, approximate, "
            "and global-fallback encoding policies on validation data only."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cpu_latency_benchmark.json"),
        help="CPU latency benchmark configuration",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="override the configured validation manifest",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/latency_benchmark.json"),
        help="machine-readable JSON output including raw repetition timings",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/latency_comparison.png"),
        help="PNG latency comparison figure",
    )
    parser.add_argument(
        "--features-jsonl",
        type=Path,
        help=(
            "optional destination for per-frame feature vectors; defaults to a "
            "sibling of the JSON summary"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark configured for at least five warm-ups and 30 samples."""
    args = _parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read config {args.config}: {error}") from error
    if not isinstance(config, dict):
        raise SystemExit("latency configuration must be a JSON object")
    if tuple(config.get("policies", ())) != POLICY_NAMES:
        raise SystemExit(
            "config policy list does not match the fixed benchmark methods"
        )

    manifest = args.manifest or Path(config["validation_manifest"])
    result = run_cpu_latency_benchmark(
        manifest,
        args.output_json,
        args.figure,
        feature_vectors_path=args.features_jsonl,
        patch_size=config["patch_size"],
        warmup_repetitions=config["warmup_repetitions"],
        timed_repetitions=config["timed_repetitions"],
        policy_order_seed=config["policy_order_seed"],
        global_change_threshold=config["global_change_threshold"],
        expected_episode_count=config["expected_episode_count"],
        expected_frames_per_episode=config["expected_frames_per_episode"],
    )
    print(result["experiment_label"])
    print(
        f"{result['frame_count']} preloaded frames; "
        f"{result['warmup_repetitions_per_policy']} warm-ups and "
        f"{result['timed_repetitions_per_policy']} timed runs per policy"
    )
    system = result["system"]
    print(
        f"Python {system['python_version'].split()[0]}; "
        f"{system['operating_system']}; CPU {system['cpu_identifier']} "
        f"({system['cpu_architecture']}, {system['logical_cpu_count']} logical CPUs)"
    )
    print(
        "policy median_ms mean_ms sd_ms p95_ms paired_median_speedup "
        "recompute_fraction feature_MAE cache_estimate_bytes"
    )
    for policy_name, policy in result["policies"].items():
        print(
            f"{policy_name:>42} "
            f"{1000 * policy['median_latency_seconds']:.3f} "
            f"{1000 * policy['mean_latency_seconds']:.3f} "
            f"{1000 * policy['standard_deviation_seconds']:.3f} "
            f"{1000 * policy['p95_latency_seconds']:.3f} "
            f"{policy['median_speedup_relative_to_independent']:.4f} "
            f"{policy['all_frame_patch_recomputation_fraction']:.4f} "
            f"{policy['feature_mean_absolute_error']:.6f} "
            f"{policy['peak_cache_memory_estimate_bytes']}"
        )
    print(f"JSON: {args.output_json}")
    print(f"Figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
