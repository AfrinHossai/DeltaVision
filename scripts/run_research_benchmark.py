#!/usr/bin/env python3
"""Evaluate fixed reuse policies on each benchmark split independently."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.research_benchmark import (  # noqa: E402
    SPLIT_NAMES,
    evaluate_research_benchmark,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact and fixed-threshold patch reuse on the train, "
            "validation, and held-out test partitions."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research_benchmark.json"),
        help="benchmark JSON configuration",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/generated/research_benchmark"),
        help="benchmark root produced by build_research_benchmark.py",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/research_benchmark.json"),
        help="machine-readable JSON output",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/research_benchmark.csv"),
        help="flat aggregate and per-transformation CSV output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run benchmark evaluations and print partition-separated metrics."""
    args = _parse_args(argv)
    result = evaluate_research_benchmark(
        args.config, args.data, args.output_json, args.output_csv
    )
    print(result["experiment_label"])
    print("No threshold is selected; the fixed grid is evaluated as configured.")
    for split_name in SPLIT_NAMES:
        partition = result["partitions"][split_name]
        print(f"\n{split_name.upper()} — {partition['episode_count']} episodes")
        print(
            "method TP FP TN FN precision recall F1 missed false_refresh "
            "refresh feature_MAE"
        )
        for method_name, method in partition["methods"].items():
            aggregate = method["aggregate"]
            detection = aggregate["change_detection"]
            features = aggregate["feature_approximation"]
            print(
                f"{method_name:>16} "
                f"{detection['true_positives']:>5} {detection['false_positives']:>4} "
                f"{detection['true_negatives']:>5} {detection['false_negatives']:>4} "
                f"{detection['precision']:.4f} {detection['recall']:.4f} "
                f"{detection['f1']:.4f} {detection['missed_change_rate']:.4f} "
                f"{detection['false_refresh_rate']:.4f} "
                f"{detection['transition_refresh_fraction']:.4f} "
                f"{features['mean_absolute_feature_error']:.6f}"
            )
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
