#!/usr/bin/env python3
"""Build a local synthetic visual-sequence dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.synthetic import (  # noqa: E402
    TRANSFORMATION_TYPES,
    generate_sequences,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic RGB sequences and change masks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/synthetic_sequences"),
        help="output directory (default: data/generated/synthetic_sequences)",
    )
    parser.add_argument("--width", type=int, default=128, help="frame width in pixels")
    parser.add_argument(
        "--height", type=int, default=128, help="frame height in pixels"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=8, help="frames per episode"
    )
    parser.add_argument(
        "--episodes", type=int, default=4, help="number of episodes to generate"
    )
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=TRANSFORMATION_TYPES,
        type=str,
        default=None,
        metavar="TRANSFORM",
        help=(
            "transformation types to sample; defaults to all: "
            + ", ".join(TRANSFORMATION_TYPES)
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sequence generator from the command line."""
    args = _parse_args(argv)
    transformations = tuple(args.transforms) if args.transforms is not None else None
    manifest_path = generate_sequences(
        args.output,
        width=args.width,
        height=args.height,
        sequence_length=args.sequence_length,
        num_episodes=args.episodes,
        seed=args.seed,
        transformations=transformations,
    )
    print(f"Generated {args.episodes} episodes; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
