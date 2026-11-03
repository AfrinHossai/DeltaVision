#!/usr/bin/env python3
"""Evaluate threshold refresh decisions against synthetic pixel-change masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.approximate import SUPPORTED_THRESHOLDS  # noqa: E402
from deltavision.evaluation import evaluate_change_detection  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the actual threshold patch gate with synthetic "
            "pixel-difference masks."
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
        default=Path("results/change_detection.json"),
        help="JSON results path (default: results/change_detection.json)",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/change_detection_tradeoff.png"),
        help="PNG figure path (default: docs/figures/change_detection_tradeoff.png)",
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
        help="supported threshold values (default: 0 0.01 0.03 0.05 0.10)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and print a compact detection table."""
    args = _parse_args(argv)
    summary = evaluate_change_detection(
        args.manifest,
        args.output,
        args.figure,
        patch_size=args.patch_size,
        thresholds=args.thresholds,
    )
    print(summary["experiment_label"])
    print(
        "threshold TP FP TN FN precision recall F1 missed false_refresh "
        "transition_refresh feature_MAE"
    )
    for row in summary["results"]:
        detection = row["change_detection"]
        approximation = row["feature_approximation"]
        print(
            f"{row['threshold']:>9.2f} "
            f"{detection['true_positives']:>4} {detection['false_positives']:>2} "
            f"{detection['true_negatives']:>2} {detection['false_negatives']:>2} "
            f"{detection['precision']:.4f} {detection['recall']:.4f} "
            f"{detection['f1']:.4f} {detection['missed_change_rate']:.4f} "
            f"{detection['false_refresh_rate']:.4f} "
            f"{detection['refresh_fraction']:.4f} "
            f"{approximation['mean_absolute_feature_error']:.6f}"
        )
    print(f"JSON: {args.output}")
    print(f"Figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
