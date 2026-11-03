"""Tests for figure, validation-gallery, and demo rendering utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from deltavision.encoder import RGBImage
from deltavision.research_demo import DemoSequence, write_demo_gif
from deltavision.research_visuals import (
    load_json_artifact,
    select_gallery_candidate,
    validate_visual_provenance,
    write_accuracy_efficiency_figure,
    write_architecture_figure,
    write_cache_capacity_figure,
    write_final_summary_figure,
    write_pretrained_speedup_figure,
)


def _figure_sources() -> dict[str, object]:
    """Small drawing fixture; values exercise layout only, not research claims."""
    method_names = (
        "independent_encoding",
        "exact_patch_reuse",
        "approximate_patch_reuse_0.01",
        "global_change_fallback",
    )
    f1_values = (0.35, 1.0, 0.99, 0.97)
    refresh_values = (1.0, 0.31, 0.32, 0.30)
    methods = {
        name: {
            "aggregate": {
                "f1": f1,
                "all_frame_refresh_fraction": refresh,
                "feature_mean_absolute_error": 0.0,
                "missed_change_rate": 0.0,
            }
        }
        for name, f1, refresh in zip(
            method_names, f1_values, refresh_values, strict=True
        )
    }
    conditions = {
        name: {
            "label": name.replace("_", " "),
            "independent_forward_calls_per_pass": 32,
            "stateful_forward_calls_per_pass": calls,
            "cached_frame_hits_per_pass": 32 - calls,
            "forward_call_reduction": (32 - calls) / 32,
            "median_speedup": speedup,
            "cache_memory_peak_bytes": 1024 * 1024,
        }
        for name, calls, speedup in (
            ("identical_consecutive_frames", 4, 5.6),
            ("sparse_visual_changes", 8, 3.3),
            ("moderate_visual_changes", 16, 1.8),
            ("frequent_widespread_changes", 32, 0.95),
        )
    }
    capacities = (1, 2, 4, 8, 16, 32)
    condition_results = {
        name: {
            "capacities": [
                {
                    "capacity": capacity,
                    "cache_hit_rate": min(0.95, capacity / 32),
                    "independent_over_cache_median_speedup_ratio": 0.8 + capacity / 25,
                    "total_frames_per_pass": 64,
                }
                for capacity in capacities
            ]
        }
        for name in (
            "repeated_identical_frames",
            "recurring_nonconsecutive_frames",
            "sparse_changes",
            "frequent_widespread_changes",
        )
    }
    return {
        "final_evaluation": {
            "handcrafted": {"methods": methods},
            "pretrained_dinov2": {"conditions": conditions},
        },
        "final_statistics": {
            "handcrafted": {
                "methods": {
                    name: {
                        "overall_confidence_intervals": {
                            "f1": {"lower": max(0.0, f1 - 0.02), "upper": f1}
                        }
                    }
                    for name, f1 in zip(method_names, f1_values, strict=True)
                }
            }
        },
        "capacity_ablation": {"condition_results": condition_results},
    }


def test_all_publication_figures_write_valid_pngs(tmp_path: Path) -> None:
    sources = _figure_sources()
    paths = (
        write_architecture_figure(tmp_path / "architecture.png"),
        write_final_summary_figure(sources, tmp_path / "summary.png"),
        write_accuracy_efficiency_figure(sources, tmp_path / "tradeoff.png"),
        write_pretrained_speedup_figure(sources, tmp_path / "speedup.png"),
        write_cache_capacity_figure(sources, tmp_path / "capacity.png"),
    )

    assert len(paths) == 5
    for path in paths:
        assert path.is_file()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 1000
            image.verify()


def test_visual_provenance_requires_validation_and_prior_test_disclosure() -> None:
    final = {
        "config_sha256": "same",
        "data_provenance": {
            "handcrafted_final_evaluation": {
                "partition": "previously_inspected_held_out_test"
            },
            "pretrained_final_evaluation": {"partition": "synthetic_validation"},
        },
        "research_integrity": {
            "held_out_is_blind": False,
            "known_prior_held_out_inspection": True,
            "pretrained_used_validation_only": True,
        },
        "pretrained_dinov2": {"status": "complete", "partition": "validation"},
    }
    stats = {"source_config_sha256": "same"}
    ablation = {"partition": "validation", "held_out_test_partition_accessed": False}
    global_validation = {"partition": "validation"}

    validate_visual_provenance(final, stats, ablation, global_validation)

    ablation["partition"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        validate_visual_provenance(final, stats, ablation, global_validation)


def test_gallery_case_selection_is_deterministic_and_prefers_small_miss() -> None:
    candidates = [
        {
            "episode_id": "validation_b",
            "transition_index": 2,
            "transformation_type": "recoloring",
            "approximate_missed_patch_indices": (1,),
            "changed_pixel_counts": (0, 6),
        },
        {
            "episode_id": "validation_a",
            "transition_index": 1,
            "transformation_type": "recoloring",
            "approximate_missed_patch_indices": (3,),
            "changed_pixel_counts": (0, 2, 0, 1),
        },
    ]

    selected = select_gallery_candidate(
        candidates,
        transformation_type="recoloring",
        require_policy_miss="approximate",
        smallest_miss=True,
    )

    assert selected["episode_id"] == "validation_a"
    with pytest.raises(ValueError, match="no validation transition"):
        select_gallery_candidate(
            candidates,
            transformation_type="global_brightness",
            require_policy_miss="approximate",
        )


def test_demo_gif_is_deterministic_and_valid(tmp_path: Path) -> None:
    frames = (
        RGBImage(32, 32, bytes([20, 50, 80] * (32 * 32))),
        RGBImage(32, 32, bytes([22, 50, 80] * (32 * 32))),
        RGBImage(32, 32, bytes([22, 90, 80] * (32 * 32))),
    )
    masks = (bytes([255] * (32 * 32)), bytes([255] * (32 * 32)))
    sequence = DemoSequence(
        episode_id="fixture_episode",
        random_seed=7,
        partition="synthetic_validation",
        frames=frames,
        masks=masks,
        transformations=("movement", "recoloring"),
        patch_size=16,
    )
    conditions = {
        name: {
            "label": name,
            "independent_forward_calls_per_pass": 32,
            "stateful_forward_calls_per_pass": calls,
            "cached_frame_hits_per_pass": 32 - calls,
            "forward_call_reduction": (32 - calls) / 32,
            "median_speedup": 1.5,
        }
        for name, calls in (
            ("identical_consecutive_frames", 4),
            ("sparse_visual_changes", 8),
            ("moderate_visual_changes", 16),
            ("frequent_widespread_changes", 32),
        )
    }
    pretrained = {"conditions": conditions}
    first = write_demo_gif(sequence, pretrained, tmp_path / "first.gif")
    second = write_demo_gif(sequence, pretrained, tmp_path / "second.gif")

    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.format == "GIF"
        assert image.n_frames == 4
        image.seek(0)
        assert image.size == (1200, 690)


def test_missing_saved_result_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="required saved experiment output"):
        load_json_artifact(tmp_path / "not-generated.json")


def test_demo_rejects_heldout_manifest_path(tmp_path: Path) -> None:
    from deltavision.research_demo import load_demo_sequence

    path = tmp_path / "test" / "manifest.jsonl"
    with pytest.raises(ValueError, match="validation data only"):
        load_demo_sequence(path)
