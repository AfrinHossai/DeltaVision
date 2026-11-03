#!/usr/bin/env python3
"""Run a frozen pretrained visual encoder on every manifest frame independently."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.pretrained import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    FrozenVisionEncoderAdapter,
    ImagePreprocessingConfig,
    ModelLoadingError,
    RepresentationKind,
)
from deltavision.pretrained_baseline import run_pretrained_baseline  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode every frame independently with a frozen pretrained vision "
            "model. Checkpoint downloads require --allow-download."
        )
    )
    parser.add_argument("manifest", type=Path, help="existing synthetic manifest.jsonl")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_baseline.json"),
        help="model and preprocessing JSON config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pretrained_baseline"),
        help="output directory for metadata and feature arrays",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face to download missing model files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Load the configured checkpoint, run the manifest, and print its output."""
    args = _parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read config {args.config}: {error}") from error
    preprocessing_values = config.get("preprocessing", {})
    for key in ("image_mean", "image_std"):
        if key in preprocessing_values:
            preprocessing_values[key] = tuple(preprocessing_values[key])
    preprocessing = ImagePreprocessingConfig(**preprocessing_values)
    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    revision = config.get("revision", DEFAULT_MODEL_REVISION)
    representation: RepresentationKind = config.get("representation", "spatial_tokens")
    try:
        adapter = FrozenVisionEncoderAdapter.from_pretrained(
            model_id=model_id,
            revision=revision,
            device=config.get("device", "cpu"),
            deterministic=config.get("deterministic", True),
            preprocessing=preprocessing,
            cache_dir=config.get("cache_dir"),
            allow_download=args.allow_download,
        )
    except (ModelLoadingError, ValueError) as error:
        raise SystemExit(f"Could not initialize pretrained encoder: {error}") from error
    try:
        output_dir = run_pretrained_baseline(
            args.manifest,
            args.output_dir,
            adapter,
            representation=representation,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Pretrained baseline failed: {error}") from error
    print(f"Wrote pretrained baseline representations to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
