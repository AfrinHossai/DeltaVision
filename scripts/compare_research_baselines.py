#!/usr/bin/env python3
"""Build a provenance-checked comparison of existing validation experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.research_comparison import (  # noqa: E402
    build_final_comparison,
    load_comparison_sources,
    write_comparison_outputs,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize validation-only handcrafted and pretrained experiments "
            "with their source provenance."
        )
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPOSITORY_ROOT / "results/final_baseline_comparison.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=REPOSITORY_ROOT / "docs/figures/research_baseline_comparison.png",
    )
    return parser.parse_args(argv)


def _repo_path(value: Path, label: str) -> Path:
    path = value if value.is_absolute() else REPOSITORY_ROOT / value
    resolved = path.resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError(f"{label} must resolve inside the repository")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output_json = _repo_path(args.output_json, "JSON output")
        figure = _repo_path(args.figure, "figure output")
        sources = load_comparison_sources(REPOSITORY_ROOT)
        report = build_final_comparison(sources, repository_root=REPOSITORY_ROOT)
        json_path, figure_path = write_comparison_outputs(report, output_json, figure)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise SystemExit(f"Research baseline comparison failed: {error}") from error

    print(f"Saved comparison: {json_path}")
    print(f"Saved figure: {figure_path}")
    print("Held-out test reports were excluded and were not opened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
