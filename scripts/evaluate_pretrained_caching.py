#!/usr/bin/env python3
"""Evaluate independent DINOv2 inference against exact whole-frame reuse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.pretrained import ModelLoadingError  # noqa: E402
from deltavision.pretrained_cache_evaluation import (  # noqa: E402
    DEFAULT_EVALUATION_CONFIG,
    build_evaluation_report,
    evaluate_conditions,
    generate_validation_scenarios,
    load_baseline_settings,
    load_evaluation_config,
    make_adapter,
    preload_validation_episodes,
    write_evaluation_outputs,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen independent DINOv2 inference with exact whole-frame "
            "reuse on deterministic synthetic validation episodes."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EVALUATION_CONFIG,
        help="experiment JSON; defaults to the committed CPU evaluation config",
    )
    return parser.parse_args(argv)


def _repository_path(value: str, label: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError(f"{label} must resolve inside the repository")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_evaluation_config(args.config)
        baseline_path = _repository_path(config["baseline_config"], "baseline config")
        validation_root = _repository_path(config["validation_root"], "validation root")
        output_json = _repository_path(config["output_json"], "JSON output")
        output_csv = _repository_path(config["output_csv"], "CSV output")
        figure_dir = _repository_path(config["figure_dir"], "figure directory")
        baseline_config = load_baseline_settings(baseline_path)

        manifests = generate_validation_scenarios(validation_root, config)
        scenarios = {
            name: (manifest, preload_validation_episodes(manifest))
            for name, manifest in manifests.items()
        }
        adapter = make_adapter(baseline_config, repository_root=REPOSITORY_ROOT)
        representation = baseline_config.get("representation", "spatial_tokens")
        evaluation = evaluate_conditions(
            adapter,
            scenarios,
            config,
            representation=representation,
        )
        report = build_evaluation_report(
            evaluation,
            config=config,
            baseline_config=baseline_config,
            adapter=adapter,
            config_path=args.config,
            baseline_config_path=baseline_path,
        )
        paths = write_evaluation_outputs(report, output_json, output_csv, figure_dir)
    except (ModelLoadingError, OSError, RuntimeError, ValueError, KeyError) as error:
        raise SystemExit(f"Pretrained cache evaluation failed: {error}") from error

    print(f"Saved report: {paths[0]}")
    print(f"Saved raw timing CSV: {paths[1]}")
    print(f"Saved figures: {paths[2][0]}, {paths[2][1]}")
    print("Condition | independent calls | cached calls | hit rate | latency ratio")
    for name, result in report["condition_results"].items():
        latency = result["latency_comparison"]
        ratio = latency["independent_over_exact_cache_speedup_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}x"
        print(
            f"{name} | "
            f"{result['independent']['model_forward_calls_per_pass']} | "
            f"{result['exact_cache']['model_forward_calls_per_pass']} | "
            f"{result['exact_cache']['cache_hit_rate']:.1%} | {ratio_text}"
        )
    print("A forward-call reduction is not itself evidence of an end-to-end speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
