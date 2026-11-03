"""Analyze Task 17 outputs with an episode-cluster bootstrap."""

from __future__ import annotations

import argparse

from deltavision.final_evaluation import FINAL_CONFIG_PATH, FINAL_OUTPUT_PATH, _resolve
from deltavision.final_statistics import analyze_final_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(FINAL_OUTPUT_PATH))
    parser.add_argument("--config", default=str(FINAL_CONFIG_PATH))
    parser.add_argument("--output")
    parser.add_argument("--figure-dir")
    arguments = parser.parse_args()
    output = arguments.output
    figure_dir = arguments.figure_dir
    if output is None:
        from deltavision.final_evaluation import load_final_config

        output = str(
            _resolve(load_final_config(arguments.config)["outputs"]["statistics_json"])
        )
    result = analyze_final_results(
        arguments.input,
        config_path=arguments.config,
        output_path=output,
        figure_dir=figure_dir,
    )
    print(
        "Final analysis complete: "
        f"{result['handcrafted']['episode_count']} held-out episodes; "
        f"{len(result['pretrained_dinov2']['conditions'])} validation conditions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
