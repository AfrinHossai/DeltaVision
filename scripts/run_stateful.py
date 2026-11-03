#!/usr/bin/env python3
"""Run the exact stateful patch-reuse baseline on a JSONL manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.state import run_stateful  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse RGB patch features for byte-identical patches between "
            "consecutive frames."
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
        default=Path("data/generated/stateful.jsonl"),
        help="output JSONL path (default: data/generated/stateful.jsonl)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="square patch size in pixels (default: 16)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stateful baseline and print the output path."""
    args = _parse_args(argv)
    output_path = run_stateful(args.manifest, args.output, patch_size=args.patch_size)
    print(f"Wrote stateful patch-reuse features to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
