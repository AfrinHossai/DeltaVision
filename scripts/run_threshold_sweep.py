#!/usr/bin/env python3
"""Run the synthetic threshold-reuse sweep and write JSON plus PNG artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.approximate import (  # noqa: E402
    SUPPORTED_THRESHOLDS,
    run_threshold_sweep,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mean-absolute-RGB threshold reuse on synthetic sequence "
            "manifest frames."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="manifest.jsonl written by scripts/build_sequences.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/threshold_sweep.json"),
        help="JSON result path (default: results/threshold_sweep.json)",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/threshold_tradeoff.png"),
        help="PNG chart path (default: docs/figures/threshold_tradeoff.png)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="square patch size in pixels (default: 16)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        choices=SUPPORTED_THRESHOLDS,
        default=SUPPORTED_THRESHOLDS,
        metavar="THRESHOLD",
        help="thresholds to evaluate (default: 0 0.01 0.03 0.05 0.10)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep, print a compact metrics table, and return success."""
    args = _parse_args(argv)
    summary = run_threshold_sweep(
        args.manifest,
        args.output,
        args.figure,
        patch_size=args.patch_size,
        thresholds=args.thresholds,
    )
    print(summary["experiment_label"])
    print("threshold  refresh  mean_abs_error  cosine  wall_clock_s")
    for row in summary["results"]:
        print(
            f"{row['threshold']:>9.2f}  {row['refresh_fraction']:>7.3f}  "
            f"{row['mean_absolute_feature_error']:>14.6f}  "
            f"{row['mean_feature_vector_cosine_similarity']:>6.4f}  "
            f"{row['wall_clock_latency_seconds']:>12.6f}"
        )
    print(f"JSON: {args.output}")
    print(f"Figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
