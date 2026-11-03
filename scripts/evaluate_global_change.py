#!/usr/bin/env python3
"""Evaluate the optional global refresh fallback on validation data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.global_evaluation import (  # noqa: E402
    LOCAL_THRESHOLDS,
    evaluate_global_change_validation,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the global scene-change full-refresh fallback using only "
            "a validation-split manifest."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/global_change_validation.json"),
        help="JSON configuration with validation manifest and thresholds",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="override the configured validation manifest path",
    )
    parser.add_argument(
        "--global-threshold",
        type=float,
        help="override the changed-pixel-fraction threshold (default: config)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/global_change_validation.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/global_change_comparison.png"),
        help="PNG comparison figure path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation-only comparison and print aggregate metrics."""
    args = _parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read config {args.config}: {error}") from error
    if not isinstance(config, dict):
        raise SystemExit("global-change config must be a JSON object")
    if tuple(config.get("local_thresholds", ())) != LOCAL_THRESHOLDS:
        raise SystemExit("config local_thresholds must be 0.01, 0.03, and 0.05")
    manifest = args.manifest or Path(config["validation_manifest"])
    patch_size = config["patch_size"]
    global_threshold = (
        args.global_threshold
        if args.global_threshold is not None
        else config["global_change_threshold"]
    )
    summary = evaluate_global_change_validation(
        manifest,
        args.output_json,
        args.figure,
        patch_size=patch_size,
        global_change_threshold=global_threshold,
    )
    print(summary["experiment_label"])
    print(f"Global changed-pixel threshold: {global_threshold}")
    print(
        "method precision recall F1 missed refresh full_refresh_freq "
        "feature_MAE detector_ms/frame"
    )
    for row in summary["methods"]:
        aggregate = row["aggregate"]
        detection = aggregate["change_detection"]
        feature = aggregate["feature_approximation"]
        runtime = aggregate["runtime"]
        detector_ms = 1000 * runtime["mean_global_detector_runtime_seconds"]
        print(
            f"{row['method']:>43} "
            f"{detection['precision']:.4f} {detection['recall']:.4f} "
            f"{detection['f1']:.4f} {detection['missed_change_rate']:.4f} "
            f"{detection['refresh_fraction']:.4f} "
            f"{detection['full_frame_refresh_frequency']:.4f} "
            f"{feature['mean_absolute_feature_error']:.6f} {detector_ms:.4f}"
        )
    print(f"JSON: {args.output_json}")
    print(f"Figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
