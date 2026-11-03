"""Deterministic tests for the whole-frame cache-capacity ablation tooling."""

from __future__ import annotations

import csv
import hashlib
import json
from array import array
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
from deltavision.pretrained_cache_ablation import (
    CAPACITY_VALUES,
    POLICY_NAMES,
    _array_feature_metrics,
    _audit_cached_features,
    _timing_summary,
    build_ablation_report,
    evaluate_capacity_ablation,
    generate_capacity_scenarios,
    load_ablation_config,
    validate_ablation_config,
    write_ablation_outputs,
)
from deltavision.pretrained_cache_evaluation import (
    preload_validation_episodes,
    run_policy_pass,
)
from deltavision.research_comparison import (
    HELD_OUT_REPORTS_NOT_OPENED,
    SAFE_JSON_SOURCES,
    validate_validation_provenance,
    write_comparison_outputs,
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
            tokens = torch.stack((scalar + offsets, scalar - offsets), dim=-1)
            return SimpleNamespace(last_hidden_state=tokens.unsqueeze(0))

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
    config = load_ablation_config(
        Path(__file__).resolve().parents[1] / "configs/pretrained_cache_ablation.json"
    )
    config["width"] = 32
    config["height"] = 32
    return config


def test_capacity_enforcement_lru_eviction_and_call_accounting() -> None:
    adapter, model = _adapter()
    episodes = (
        ("episode_a", (_image(10), _image(20), _image(10), _image(30), _image(20))),
    )

    result = run_policy_pass(
        adapter,
        episodes,
        policy="exact_cache",
        representation="spatial_tokens",
        cache_capacity=2,
    )

    assert result["model_forward_calls"] == model.calls == 4
    assert result["cache_misses"] == 4
    assert result["cached_frame_hits"] == 1
    assert result["cache_evictions"] == 2
    assert result["peak_cache_memory_estimate_bytes"] > 0


def test_recurring_nonconsecutive_frame_is_reused_when_resident() -> None:
    adapter, model = _adapter()
    episodes = (("episode_a", (_image(10), _image(20), _image(10))),)

    result = run_policy_pass(
        adapter,
        episodes,
        policy="exact_cache",
        representation="spatial_tokens",
        cache_capacity=2,
    )

    assert result["model_forward_calls"] == model.calls == 2
    assert result["cached_frame_hits"] == 1
    assert result["cache_hit_rate"] == pytest.approx(1 / 3)


def test_feature_audit_confirms_exact_representation_equivalence() -> None:
    adapter, model = _adapter()
    episodes = (("episode_a", (_image(12), _image(34), _image(12))),)

    result = _audit_cached_features(
        adapter,
        episodes,
        representation="spatial_tokens",
        cache_capacity=2,
        maximum_absolute_difference_tolerance=0.0,
        minimum_cosine_similarity_tolerance=1.0,
    )

    assert result["independent_comparison_forward_calls"] == 3
    assert result["stateful_comparison_forward_calls"] == 2
    assert result["exact_match"] is True
    assert result["maximum_absolute_difference"] == 0.0
    assert result["mean_absolute_difference"] == 0.0
    assert result["cosine_similarity"] == pytest.approx(1.0)
    assert model.calls == 5


def test_cosine_similarity_metrics_stay_in_valid_range() -> None:
    torch = pytest.importorskip("torch")
    values = array("f", [0.2, 0.5, 0.8] * 1000)

    result = _array_feature_metrics(torch, values, values)

    assert result["exact_match"] is True
    assert -1.0 <= result["cosine_similarity"] <= 1.0


def test_timing_summary_aggregates_raw_repetitions() -> None:
    result = _timing_summary([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["median_seconds"] == 3.0
    assert result["mean_seconds"] == 3.0
    assert result["repetition_count"] == 5
    assert result["sample_standard_deviation_seconds"] == pytest.approx(1.58113883)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _timing_summary([1.0, float("nan")])


def test_capacity_config_and_trace_shape_are_reproducible() -> None:
    root = Path(__file__).resolve().parents[1]
    first = load_ablation_config(root / "configs/pretrained_cache_ablation.json")
    second = load_ablation_config(root / "configs/pretrained_cache_ablation.json")

    assert first == second
    assert first["cache_capacities"] == list(CAPACITY_VALUES)
    assert first["policy_order_seed"] == 516017
    assert first["warmup_repetitions"] >= 3
    assert first["timed_repetitions"] >= 5
    validate_ablation_config(first)

    first["cache_capacities"] = [1, 2]
    with pytest.raises(ValueError, match="cache_capacities"):
        validate_ablation_config(first)


def test_synthetic_capacity_manifests_and_frames_are_deterministic(
    tmp_path: Path,
) -> None:
    config = _small_config()
    left = generate_capacity_scenarios(tmp_path / "left" / "validation", config)
    right = generate_capacity_scenarios(tmp_path / "right" / "validation", config)

    assert set(left) == set(right)
    for name in left:
        left_row = json.loads(left[name].read_text(encoding="utf-8"))
        right_row = json.loads(right[name].read_text(encoding="utf-8"))
        assert left_row == right_row
        assert (
            left_row["random_seed"]
            == config["base_seed"] + list(left).index(name) * 100
        )
        for left_frame, right_frame in zip(
            left_row["frame_paths"], right_row["frame_paths"], strict=True
        ):
            left_hash = hashlib.sha256(
                (left[name].parent / left_frame).read_bytes()
            ).digest()
            right_hash = hashlib.sha256(
                (right[name].parent / right_frame).read_bytes()
            ).digest()
            assert left_hash == right_hash

    assert len(left_row["frame_paths"]) == 64


def test_mock_capacity_evaluation_accounts_for_every_timed_policy(
    tmp_path: Path,
) -> None:
    config = _small_config()
    manifests = generate_capacity_scenarios(tmp_path / "validation", config)
    scenarios = {
        name: (manifest, preload_validation_episodes(manifest))
        for name, manifest in manifests.items()
    }
    adapter, _model = _adapter()

    result = evaluate_capacity_ablation(
        adapter,
        scenarios,
        config,
        representation="spatial_tokens",
    )

    assert len(result["raw_timed_trials"]) == 4 * 7 * 5
    assert len(result["execution_schedule"]) == 4 * (3 + 5)
    conditions = result["condition_results"]
    assert (
        conditions["repeated_identical_frames"]["unique_preprocessed_input_count"] == 1
    )
    assert (
        conditions["recurring_nonconsecutive_frames"]["unique_preprocessed_input_count"]
        == 32
    )
    assert (
        conditions["frequent_widespread_changes"]["unique_preprocessed_input_count"]
        == 64
    )
    for condition in conditions.values():
        assert len(condition["capacities"]) == 6
        assert all(
            item["feature_equivalence"]["exact_match"]
            for item in condition["capacities"]
        )


def test_provenance_guard_accepts_validation_and_rejects_test(tmp_path: Path) -> None:
    validation_report = {
        "partition": "validation",
        "test_partition_accessed": False,
        "manifest_path": "data/generated/example/validation/manifest.jsonl",
    }
    validate_validation_provenance(
        validation_report, "results/example.json", source_id="example"
    )

    with pytest.raises(ValueError, match="does not establish validation-only"):
        validate_validation_provenance(
            {"partition": "test"}, "results/example.json", source_id="example"
        )
    with pytest.raises(ValueError, match="not validation-prefixed"):
        validate_validation_provenance(
            {"partition": "validation", "manifest_path": "data/test/manifest.jsonl"},
            "results/example.json",
            source_id="example",
        )

    assert not set(SAFE_JSON_SOURCES.values()).intersection(HELD_OUT_REPORTS_NOT_OPENED)
    config = _small_config()
    with pytest.raises(ValueError, match="held-out test paths"):
        generate_capacity_scenarios(tmp_path / "test", config)


def test_report_records_model_and_validation_provenance() -> None:
    adapter, _model = _adapter()
    audit = {
        "independent_comparison_forward_calls": 2,
        "stateful_comparison_forward_calls": 1,
    }
    evaluation = {
        "raw_timed_trials": [
            {"policy": "independent", "model_forward_calls": 2},
            {
                "policy": "exact_cache",
                "model_forward_calls": 1,
                "total_frames": 2,
            },
        ],
        "condition_results": {
            "mock_condition": {
                "capacities": [{"feature_equivalence": audit}],
                "warmup_model_forward_calls_by_policy": dict.fromkeys(POLICY_NAMES, 1),
            }
        },
        "execution_schedule": [],
        "timing_scope": "Preloaded fixture mock.",
    }
    report = build_ablation_report(
        evaluation,
        config={
            "cache_capacities": list(CAPACITY_VALUES),
            "base_seed": 516016,
            "warmup_repetitions": 3,
            "timed_repetitions": 5,
            "policy_order_seed": 516017,
        },
        baseline_config={"representation": "spatial_tokens"},
        adapter=adapter,
        config_path="configs/pretrained_cache_ablation.json",
        baseline_config_path="configs/pretrained_baseline.json",
    )

    assert report["partition"] == "validation"
    assert report["held_out_test_partition_accessed"] is False
    assert report["model"]["model_revision"] == DEFAULT_MODEL_REVISION
    assert report["model"]["evaluation_mode"] is True
    assert report["model"]["frozen_inference"] is True
    assert (
        report["forward_call_accounting"][
            "timed_full_model_calls_avoided_across_capacity_pairs"
        ]
        == 1
    )
    assert (
        report["forward_call_accounting"]["separate_feature_equivalence_audits"][
            "included_in_timed_latency"
        ]
        is False
    )


def test_ablation_serialization_writes_raw_csv_and_pngs(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    summary = {
        "median_seconds": 0.1,
        "mean_seconds": 0.11,
        "sample_standard_deviation_seconds": 0.01,
        "minimum_seconds": 0.09,
        "maximum_seconds": 0.12,
        "repetition_count": 5,
    }
    conditions = {
        name: {
            "label": name,
            "independent": {"median_processing_latency": summary},
            "capacities": [
                {
                    "capacity": capacity,
                    "median_processing_latency": summary,
                }
                for capacity in CAPACITY_VALUES
            ],
        }
        for name in (
            "repeated_identical_frames",
            "recurring_nonconsecutive_frames",
            "sparse_changes",
            "frequent_widespread_changes",
        )
    }
    report = {
        "condition_results": conditions,
        "raw_timed_trials": [
            {
                "condition": "repeated_identical_frames",
                "condition_label": "Repeated identical",
                "repetition": 1,
                "execution_order": 1,
                "policy": "exact_cache",
                "cache_capacity": 1,
                "total_frames": 64,
                "model_forward_calls": 1,
                "cached_frame_hits": 63,
                "cache_misses": 1,
                "cache_hit_rate": 63 / 64,
                "cache_evictions": 0,
                "avoided_forward_percentage": 63 / 64 * 100,
                "processing_latency_seconds": 0.1,
                "model_inference_seconds": 0.02,
                "preprocessing_seconds": 0.01,
                "cache_lookup_overhead_seconds": 0.001,
                "cache_retrieval_overhead_seconds": 0.001,
                "cache_storage_overhead_seconds": 0.002,
                "peak_cache_memory_estimate_bytes": 1024,
            }
        ],
    }

    json_path, csv_path, figure_path = write_ablation_outputs(
        report,
        tmp_path / "ablation.json",
        tmp_path / "ablation.csv",
        tmp_path / "ablation.png",
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["raw_timed_trials"]
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["cache_capacity"] == "1"
    from PIL import Image

    with Image.open(figure_path) as image:
        assert image.format == "PNG"
        assert image.size == (1500, 900)


def test_comparison_serialization_is_valid_and_scoped(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    report = {
        "comparison_table_A_handcrafted_rgb_statistics": {
            "rows": [
                {
                    "method": "exact_patch_reuse",
                    "median_speedup_relative_to_independent": 2.0,
                }
            ]
        },
        "comparison_table_B_pretrained_dinov2": {
            "capacity_8_independent_vs_cache_rows": [
                {
                    "label": "Identical",
                    "independent_over_cached_speedup_ratio": 3.0,
                }
            ]
        },
    }

    json_path, figure_path = write_comparison_outputs(
        report, tmp_path / "comparison.json", tmp_path / "comparison.png"
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    from PIL import Image

    with Image.open(figure_path) as image:
        assert image.format == "PNG"
        assert image.size == (1500, 850)
