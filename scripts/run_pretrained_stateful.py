#!/usr/bin/env python3
"""Run exact whole-frame caching with the pinned frozen DINOv2 adapter."""

from __future__ import annotations

import argparse
import json
import os
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
from deltavision.pretrained_stateful import run_pretrained_stateful  # noqa: E402

DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/generated/demo/manifest.jsonl"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/pretrained_baseline.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/pretrained_stateful"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse complete DINOv2 representations only when preprocessed frame "
            "inputs match exactly. Model downloads are disabled."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-capacity",
        type=int,
        default=8,
        help="maximum cached frame representations per episode; 0 disables storage",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = args.manifest.resolve()
    if "test" in manifest.parts:
        raise SystemExit("held-out test split paths are excluded from this runner")
    if args.cache_capacity < 0:
        raise SystemExit("--cache-capacity must be non-negative")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read config {args.config}: {error}") from error

    preprocessing_values = dict(config.get("preprocessing", {}))
    for key in ("image_mean", "image_std"):
        if key in preprocessing_values:
            preprocessing_values[key] = tuple(preprocessing_values[key])
    preprocessing = ImagePreprocessingConfig(**preprocessing_values)
    cache_dir = config.get("cache_dir")
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = REPOSITORY_ROOT / cache_path
        cache_dir = str(cache_path)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
            cache_dir=cache_dir,
            allow_download=False,
        )
        output = run_pretrained_stateful(
            manifest,
            args.output_dir,
            adapter,
            representation=representation,
            cache_capacity=args.cache_capacity,
        )
    except (ModelLoadingError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Stateful pretrained baseline failed: {error}") from error

    metrics = json.loads((output / "stateful_metrics.json").read_text("utf-8"))
    cache_metrics = metrics["cache_metrics"]
    difference_metrics = metrics["numerical_differences_against_independent_inference"]
    print(f"Wrote stateful pretrained results to {output}")
    print(f"Frames: {cache_metrics['total_frames']}")
    print(f"DINOv2 forward calls: {cache_metrics['actual_dinov2_forward_calls']}")
    print(f"Cached frame hits: {cache_metrics['cached_frame_hits']}")
    print(f"Cache misses: {cache_metrics['cache_misses']}")
    print(f"Cache hit rate: {cache_metrics['cache_hit_rate']:.3%}")
    print(
        f"Model inference time: {cache_metrics['model_inference_time_seconds']:.6f} s"
    )
    print(
        f"Cache lookup overhead: {cache_metrics['cache_lookup_overhead_seconds']:.6f} s"
    )
    print(
        "Cache retrieval overhead: "
        f"{cache_metrics['cache_retrieval_overhead_seconds']:.6f} s"
    )
    print(
        "Cache storage overhead: "
        f"{cache_metrics['cache_storage_overhead_seconds']:.6f} s"
    )
    print(
        "End-to-end stateful processing time: "
        f"{cache_metrics['total_processing_time_seconds']:.6f} s"
    )
    print(
        "Peak cached tensor payload: "
        f"{cache_metrics['peak_cache_memory_estimate_bytes']} bytes"
    )
    print(
        "Independent-output maximum absolute difference: "
        f"{difference_metrics['aggregate']['maximum_absolute_difference']:.9g}"
    )
    print(
        "Independent-output mean absolute difference: "
        f"{difference_metrics['aggregate']['mean_absolute_difference']:.9g}"
    )
    print(
        "Independent-output cosine similarity: "
        f"{difference_metrics['aggregate']['cosine_similarity']:.12f}"
    )
    comparison_calls = difference_metrics["comparison_forward_calls"]
    comparison_seconds = difference_metrics[
        "comparison_time_seconds_excluded_from_stateful_total"
    ]
    print(
        "Independent comparison pass: "
        f"{comparison_calls} forwards in {comparison_seconds:.6f} s "
        "(excluded from stateful totals)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
