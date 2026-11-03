#!/usr/bin/env python3
"""Build and validate the configured train, validation, and test data splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.research_benchmark import build_research_benchmark  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic multi-seed synthetic benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research_benchmark.json"),
        help="benchmark JSON configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/research_benchmark"),
        help="output root for the split datasets",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the dataset and print validated partition sizes."""
    args = _parse_args(argv)
    result = build_research_benchmark(args.config, args.output)
    validation = result["split_validation"]["splits"]
    print(f"Built {result['benchmark_name']}")
    for split_name, split_result in validation.items():
        print(
            f"{split_name}: {split_result['episode_count']} episodes, "
            f"seeds {split_result['base_seed']}..{split_result['last_seed']}"
        )
    print(f"Manifest: {args.output / 'benchmark_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
