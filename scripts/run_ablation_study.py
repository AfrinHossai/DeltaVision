#!/usr/bin/env python3
"""Run the controlled patch-size and temporal-change ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.ablation import run_ablation_study  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare independent RGB-statistics encoding, exact reuse, and "
            "threshold reuse over controlled synthetic validation sequences."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablation_study.json"),
        help="fixed synthetic ablation configuration",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/generated/ablation_study"),
        help="generated condition episodes (excluded from Git)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/ablation_study.json"),
        help="machine-readable experiment output including raw timings",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/ablation_study.csv"),
        help="one-row-per-method/configuration comparison table",
    )
    parser.add_argument(
        "--patch-size-figure",
        type=Path,
        default=Path("docs/figures/ablation_patch_size.png"),
        help="median latency versus patch size figure",
    )
    parser.add_argument(
        "--change-frequency-figure",
        type=Path,
        default=Path("docs/figures/ablation_change_frequency.png"),
        help="relative latency versus change-condition figure",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ablation and print cache-versus-independent summaries."""
    args = _parse_args(argv)
    try:
        result = run_ablation_study(
            args.config,
            data_dir=args.data_dir,
            output_json=args.output_json,
            output_csv=args.output_csv,
            patch_size_figure=args.patch_size_figure,
            change_frequency_figure=args.change_frequency_figure,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"ablation study failed: {error}") from error

    print(result["experiment_label"])
    print(
        f"{len(result['scenario_profiles'])} conditions; "
        f"{len(result['results'])} method/configuration rows; "
        f"{result['configuration']['warmup_repetitions']} warm-ups and "
        f"{result['configuration']['timed_repetitions']} timed runs per method"
    )
    print("patch_size condition method speedup category median_latency_ms refresh")
    for row in result["results"]:
        if row["method"] == "independent_frame_encoding":
            continue
        print(
            f"{row['patch_size']:>10} {row['condition']:<34} "
            f"{row['method']:<28} {row['relative_speedup']:.3f}x "
            f"{row['speed_category']:<7} "
            f"{row['median_processing_latency_seconds'] * 1000:.3f} "
            f"{row['refresh_fraction']:.4f}"
        )
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")
    print(f"Figures: {args.patch_size_figure}, {args.change_frequency_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
