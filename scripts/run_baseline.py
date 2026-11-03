#!/usr/bin/env python3
"""Run the frame-independent RGB-statistics baseline on a JSONL manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.baseline import run_baseline  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently encode each frame in a synthetic sequence manifest "
            "using RGB patch statistics."
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
        default=Path("data/generated/baseline.jsonl"),
        help="output JSONL path (default: data/generated/baseline.jsonl)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="square patch size in pixels (default: 16)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the baseline CLI and print the output path."""
    args = _parse_args(argv)
    output_path = run_baseline(args.manifest, args.output, patch_size=args.patch_size)
    print(f"Wrote frame-independent features to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
