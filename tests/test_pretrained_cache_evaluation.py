"""Deterministic tests for the pretrained exact-cache evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deltavision.encoder import RGBImage
from deltavision.pretrained import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    FrozenVisionEncoderAdapter,
    ImagePreprocessingConfig,
)
from deltavision.pretrained_cache_evaluation import (
    DEFAULT_EVALUATION_CONFIG,
    compare_policy_features,
    evaluate_conditions,
    generate_validation_scenarios,
    load_evaluation_config,
    preload_validation_episodes,
    run_policy_pass,
    summarize_visual_changes,
    timing_summary,
    validate_evaluation_config,
    write_evaluation_outputs,
)


class _Processor:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def __call__(self, *, images: RGBImage, return_tensors: str) -> dict[str, Any]:
        assert return_tensors == "pt"
        level = images.pixels[0] / 255.0
        pixels = self.torch.full((1, 3, 4, 4), level, dtype=self.torch.float32)
        return {"pixel_values": pixels}


def _adapter() -> tuple[FrozenVisionEncoderAdapter, Any]:
    torch = pytest.importorskip("torch")

    class _TinyDino(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(
                model_type="mock-dinov2",
                patch_size=2,
                hidden_size=2,
                num_register_tokens=0,
            )
            self.calls = 0

        def forward(self, pixel_values: Any) -> SimpleNamespace:
            self.calls += 1
            scalar = pixel_values.mean() * self.weight
            offsets = torch.arange(
                5, dtype=pixel_values.dtype, device=pixel_values.device
            )
            sequence = torch.stack((scalar + offsets, scalar - offsets), dim=-1)
            return SimpleNamespace(last_hidden_state=sequence.unsqueeze(0))

    model = _TinyDino()
    adapter = FrozenVisionEncoderAdapter(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        model=model,
        processor=_Processor(torch),
        device="cpu",
        deterministic=True,
        preprocessing=ImagePreprocessingConfig(
            resize_shortest_edge=4,
            crop_height=4,
            crop_width=4,
        ),
        torch_module=torch,
        image_converter=lambda image: image,
    )
    return adapter, model


def _image(value: int) -> RGBImage:
    return RGBImage(4, 4, bytes((value, 32, 64)) * 16)


def _small_config() -> dict[str, Any]:
    config = load_evaluation_config(DEFAULT_EVALUATION_CONFIG)
    config["width"] = 32
    config["height"] = 32
    config["num_episodes"] = 2
    return config


def test_scenario_generation_is_deterministic(tmp_path: Path) -> None:
    config = _small_config()
    first = generate_validation_scenarios(tmp_path / "one" / "validation", config)
    second = generate_validation_scenarios(tmp_path / "two" / "validation", config)

    assert set(first) == set(second)
    for name in first:
        first_rows = [
            json.loads(line)
            for line in first[name].read_text(encoding="utf-8").splitlines()
        ]
        second_rows = [
            json.loads(line)
            for line in second[name].read_text(encoding="utf-8").splitlines()
        ]
        assert first_rows == second_rows
        for left, right in zip(first_rows, second_rows, strict=True):
            for left_path, right_path in zip(
                left["frame_paths"], right["frame_paths"], strict=True
            ):
                left_hash = hashlib.sha256(
                    (first[name].parent / left_path).read_bytes()
                ).digest()
                right_hash = hashlib.sha256(
                    (second[name].parent / right_path).read_bytes()
                ).digest()
                assert left_hash == right_hash


def test_scenario_conditions_have_predeclared_change_and_call_counts(
    tmp_path: Path,
) -> None:
    config = _small_config()
    manifests = generate_validation_scenarios(tmp_path / "validation", config)
    expected_forward_calls = {
        "identical_consecutive_frames": 2,
        "sparse_visual_changes": 4,
        "moderate_visual_changes": 8,
        "frequent_widespread_changes": 16,
    }
    for name, manifest in manifests.items():
        episodes = preload_validation_episodes(manifest)
        visual = summarize_visual_changes(episodes)
        if name == "identical_consecutive_frames":
            assert visual["changed_transition_count"] == 0
        elif name == "frequent_widespread_changes":
            assert visual["changed_transition_fraction"] == 1.0
            assert visual["mean_changed_pixel_fraction_per_transition"] == 1.0
        adapter, _model = _adapter()
        result = run_policy_pass(
            adapter,
            episodes,
            policy="exact_cache",
            representation="spatial_tokens",
            cache_capacity=config["cache_capacity"],
        )
        assert result["model_forward_calls"] == expected_forward_calls[name]
        if name == "frequent_widespread_changes":
            assert result["cached_frame_hits"] == 0


def test_policy_accounting_uses_one_call_per_independent_frame_and_miss() -> None:
    adapter, model = _adapter()
    episodes = (("episode_a", (_image(10), _image(10), _image(80), _image(10))),)
    independent = run_policy_pass(
        adapter,
        episodes,
        policy="independent",
        representation="spatial_tokens",
        cache_capacity=4,
    )
    assert independent["model_forward_calls"] == 4
    assert independent["cached_frame_hits"] == 0

    cached = run_policy_pass(
        adapter,
        episodes,
        policy="exact_cache",
        representation="spatial_tokens",
        cache_capacity=4,
    )
    assert cached["model_forward_calls"] == model.calls - 4 == 2
    assert cached["cache_misses"] == 2
    assert cached["cached_frame_hits"] == 2
    assert cached["cache_hit_rate"] == pytest.approx(0.5)
    assert cached["avoided_forward_percentage"] == pytest.approx(50.0)


def test_feature_equivalence_compares_complete_cached_representations() -> None:
    adapter, model = _adapter()
    episodes = (("episode_a", (_image(12), _image(12), _image(90))),)

    comparison = compare_policy_features(
        adapter,
        episodes,
        representation="spatial_tokens",
        cache_capacity=4,
        maximum_absolute_difference_tolerance=0.0,
        minimum_cosine_similarity_tolerance=1.0,
    )

    assert comparison["independent_comparison_forward_calls"] == 3
    assert comparison["stateful_comparison_forward_calls"] == 2
    assert comparison["exact_match"] is True
    assert comparison["maximum_absolute_difference"] == 0.0
    assert comparison["within_documented_tolerance"] is True
    assert model.calls == 5


def test_timing_aggregation_reports_mean_median_and_sample_deviation() -> None:
    summary = timing_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["mean_seconds"] == pytest.approx(3.0)
    assert summary["median_seconds"] == pytest.approx(3.0)
    assert summary["sample_standard_deviation_seconds"] == pytest.approx(1.58113883)
    assert summary["minimum_seconds"] == 1.0
    assert summary["maximum_seconds"] == 5.0
    assert summary["repetition_count"] == 5


def test_evaluation_configuration_is_reproducible_and_requires_repetitions() -> None:
    first = load_evaluation_config(DEFAULT_EVALUATION_CONFIG)
    second = load_evaluation_config(DEFAULT_EVALUATION_CONFIG)
    assert first == second
    assert first["policy_order_seed"] == 515215
    assert first["warmup_repetitions"] >= 3
    assert first["timed_repetitions"] >= 5

    first["timed_repetitions"] = 4
    with pytest.raises(ValueError, match="five timed"):
        validate_evaluation_config(first)


def test_evaluation_output_serialization_writes_json_csv_and_figures(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    summary = {
        "mean_seconds": 1.0,
        "median_seconds": 1.0,
        "sample_standard_deviation_seconds": 0.1,
        "minimum_seconds": 0.9,
        "maximum_seconds": 1.1,
        "repetition_count": 5,
    }
    report = {
        "condition_results": {
            "mock_condition": {
                "label": "Mock condition",
                "independent": {"mean_processing_latency": summary},
                "exact_cache": {
                    "mean_processing_latency": summary,
                    "cache_hit_rate": 0.5,
                },
            }
        },
        "raw_timed_trials": [
            {
                "condition": "mock_condition",
                "condition_label": "Mock condition",
                "repetition": 1,
                "execution_order": 1,
                "policy": "independent",
                "total_frames": 2,
                "model_forward_calls": 2,
                "cached_frame_hits": 0,
                "cache_misses": 2,
                "cache_hit_rate": 0.0,
                "avoided_forward_percentage": 0.0,
                "processing_latency_seconds": 1.0,
            }
        ],
    }

    json_path, csv_path, figures = write_evaluation_outputs(
        report,
        tmp_path / "evaluation.json",
        tmp_path / "evaluation.csv",
        tmp_path / "figures",
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["raw_timed_trials"]
    assert csv_path.read_text(encoding="utf-8").startswith("condition,")
    assert all(path.is_file() and path.stat().st_size > 100 for path in figures)


def test_evaluation_keeps_raw_trial_order_and_uses_configured_repetitions() -> None:
    adapter, _model = _adapter()
    config = load_evaluation_config(DEFAULT_EVALUATION_CONFIG)
    config["warmup_repetitions"] = 3
    config["timed_repetitions"] = 5
    episodes = (("episode_a", (_image(7), _image(7))),)
    scenarios = {
        condition["name"]: (
            Path("validation") / condition["name"] / "manifest.jsonl",
            episodes,
        )
        for condition in config["conditions"]
    }

    report = evaluate_conditions(
        adapter,
        scenarios,
        config,
        representation="spatial_tokens",
    )

    assert len(report["raw_timed_trials"]) == 4 * 5 * 2
    assert len(report["execution_schedule"]) == 4 * (3 + 5)
    assert all(
        result["feature_equivalence"]["within_documented_tolerance"]
        for result in report["conditions"].values()
    )
