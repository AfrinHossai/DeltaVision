"""Run the frozen Task 17 final evaluation."""

from __future__ import annotations

import argparse

from deltavision.final_evaluation import FINAL_CONFIG_PATH, run_final_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(FINAL_CONFIG_PATH))
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    arguments = parser.parse_args()
    result = run_final_evaluation(
        arguments.config,
        output_json=arguments.output_json,
        output_csv=arguments.output_csv,
    )
    print(
        "Final evaluation complete: "
        f"{result['handcrafted']['episodes']} handcrafted episodes, "
        f"{len(result['pretrained_dinov2']['conditions'])} pretrained conditions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
