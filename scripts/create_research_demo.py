"""Create Task 18 figures, validation gallery, and visual demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

from deltavision.final_evaluation import REPOSITORY_ROOT
from deltavision.research_demo import (
    load_demo_sequence,
    render_demo_frame_slides,
    write_demo_gif,
    write_demo_metadata,
    write_mp4_if_available,
)
from deltavision.research_visuals import (
    generate_task18_assets,
    load_visual_sources,
)


def _default(path: str) -> Path:
    return REPOSITORY_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-evaluation",
        default=str(_default("results/final_evaluation.json")),
    )
    parser.add_argument(
        "--final-statistics",
        default=str(_default("results/final_statistics.json")),
    )
    parser.add_argument(
        "--capacity-ablation",
        default=str(_default("results/pretrained_cache_ablation.json")),
    )
    parser.add_argument(
        "--global-validation",
        default=str(_default("results/global_change_validation.json")),
    )
    parser.add_argument(
        "--research-config",
        default=str(_default("configs/research_benchmark.json")),
    )
    parser.add_argument(
        "--validation-manifest",
        default=str(
            _default("data/generated/research_benchmark/validation/manifest.jsonl")
        ),
    )
    parser.add_argument(
        "--pretrained-validation-root",
        default=str(_default("data/generated/pretrained_cache_evaluation/validation")),
    )
    parser.add_argument("--docs-root", default=str(_default("docs")))
    parser.add_argument("--episode-id", default=None)
    args = parser.parse_args()

    outputs = generate_task18_assets(
        final_evaluation_path=args.final_evaluation,
        final_statistics_path=args.final_statistics,
        capacity_ablation_path=args.capacity_ablation,
        global_validation_path=args.global_validation,
        research_config_path=args.research_config,
        validation_manifest=args.validation_manifest,
        pretrained_validation_root=args.pretrained_validation_root,
        docs_root=Path(args.docs_root),
    )
    sources = load_visual_sources(
        final_evaluation_path=args.final_evaluation,
        final_statistics_path=args.final_statistics,
        capacity_ablation_path=args.capacity_ablation,
        global_validation_path=args.global_validation,
    )
    sequence = load_demo_sequence(
        args.validation_manifest,
        episode_id=args.episode_id,
        patch_size=int(
            sources["final_evaluation"]["config"]["handcrafted"]["patch_size"]
        ),
    )
    demo_dir = Path(args.docs_root) / "demo"
    gif_path = demo_dir / "deltavision_demo.gif"
    write_demo_gif(
        sequence,
        sources["final_evaluation"]["pretrained_dinov2"],
        gif_path,
    )
    slides = render_demo_frame_slides(
        sequence,
        sources["final_evaluation"]["pretrained_dinov2"],
    )
    mp4_path = write_mp4_if_available(slides, demo_dir / "deltavision_demo.mp4")
    metadata = write_demo_metadata(
        sequence,
        gif_path,
        mp4_path,
        demo_dir / "deltavision_demo.json",
    )
    print(f"Generated {len(outputs['figures'])} research figures.")
    print(f"Generated {len(outputs['gallery']['cases'])} failure-gallery cards.")
    print(f"Demo GIF: {gif_path}")
    print(f"Demo metadata: {metadata}")
    if mp4_path is None:
        print("MP4 not generated: no local ffmpeg encoder is available.")
    else:
        print(f"Demo MP4: {mp4_path}")
    print("No pretrained model was loaded and no benchmark was rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
