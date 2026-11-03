#!/usr/bin/env python3
"""Measure exact DINOv2 cache-capacity effects on synthetic validation traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.pretrained import ModelLoadingError  # noqa: E402
from deltavision.pretrained_cache_ablation import (  # noqa: E402
    build_ablation_report,
    evaluate_capacity_ablation,
    generate_capacity_scenarios,
    load_ablation_config,
    write_ablation_outputs,
)
from deltavision.pretrained_cache_evaluation import (  # noqa: E402
    load_baseline_settings,
    make_adapter,
    preload_validation_episodes,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CPU DINOv2 independent encoding with exact whole-frame "
            "cache capacities 1, 2, 4, 8, 16, and 32."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/pretrained_cache_ablation.json",
    )
    return parser.parse_args(argv)


def _repo_path(value: str, label: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError(f"{label} must resolve inside the repository")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_ablation_config(args.config)
        baseline_path = _repo_path(config["baseline_config"], "baseline config")
        validation_root = _repo_path(config["validation_root"], "validation root")
        output_json = _repo_path(config["output_json"], "JSON output")
        output_csv = _repo_path(config["output_csv"], "CSV output")
        figure_path = _repo_path(config["figure"], "figure output")
        baseline_config = load_baseline_settings(baseline_path)
        manifests = generate_capacity_scenarios(validation_root, config)
        scenarios = {
            name: (manifest, preload_validation_episodes(manifest))
            for name, manifest in manifests.items()
        }
        adapter = make_adapter(baseline_config, repository_root=REPOSITORY_ROOT)
        representation = baseline_config.get("representation", "spatial_tokens")
        evaluation = evaluate_capacity_ablation(
            adapter,
            scenarios,
            config,
            representation=representation,
        )
        report = build_ablation_report(
            evaluation,
            config=config,
            baseline_config=baseline_config,
            adapter=adapter,
            config_path=args.config,
            baseline_config_path=baseline_path,
        )
        paths = write_ablation_outputs(report, output_json, output_csv, figure_path)
    except (ModelLoadingError, OSError, RuntimeError, ValueError, KeyError) as error:
        raise SystemExit(
            f"Pretrained cache capacity ablation failed: {error}"
        ) from error

    print(f"Saved report: {paths[0]}")
    print(f"Saved raw timing CSV: {paths[1]}")
    print(f"Saved figure: {paths[2]}")
    print("Condition | capacity | frames | calls | hits | hit rate | median speedup")
    for condition, result in report["condition_results"].items():
        for capacity in result["capacities"]:
            print(
                f"{condition} | {capacity['capacity']} | "
                f"{capacity['total_frames_per_pass']} | "
                f"{capacity['actual_model_forward_calls_per_pass']} | "
                f"{capacity['cached_frame_hits_per_pass']} | "
                f"{capacity['cache_hit_rate']:.1%} | "
                f"{capacity['independent_over_cache_median_speedup_ratio']:.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
