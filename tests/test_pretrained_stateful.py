"""Offline tests for exact whole-frame pretrained representation reuse."""

from __future__ import annotations

import array
import copy
import json
from contextlib import AbstractContextManager
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
from deltavision.pretrained_stateful import (
    ExactWholeFrameCache,
    run_pretrained_stateful,
)


def _shape(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list):
        return ()
    if not values:
        return (0,)
    return (len(values), *_shape(values[0]))


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        return [number for child in values for number in _flatten(child)]
    return [float(values)]


def _product(shape: tuple[int, ...]) -> int:
    product = 1
    for dimension in shape:
        product *= dimension
    return product


def _nest(values: list[float], shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return values
    stride = _product(shape[1:])
    return [
        _nest(values[index * stride : (index + 1) * stride], shape[1:])
        for index in range(shape[0])
    ]


class _MockNumpyArray:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tobytes(self, order: str = "C") -> bytes:
        assert order == "C"
        return array.array("f", self.values).tobytes()


class MockTensor:
    def __init__(self, values: Any) -> None:
        self.values = values
        self.shape = _shape(values)
        self.dtype = "torch.float32"

    def to(self, *_args: Any, **_kwargs: Any) -> MockTensor:
        return self

    def detach(self) -> MockTensor:
        return self

    def cpu(self) -> MockTensor:
        return self

    def contiguous(self) -> MockTensor:
        return self

    def clone(self) -> MockTensor:
        return MockTensor(copy.deepcopy(self.values))

    def numel(self) -> int:
        return _product(self.shape)

    def element_size(self) -> int:
        return 4

    def numpy(self) -> _MockNumpyArray:
        return _MockNumpyArray(_flatten(self.values))

    def tolist(self) -> Any:
        return self.values

    def __getitem__(self, index: Any) -> MockTensor:
        indexes = index if isinstance(index, tuple) else (index,)

        def select(values: Any, depth: int) -> Any:
            if depth == len(indexes):
                return values
            current = indexes[depth]
            if isinstance(current, slice):
                return [select(item, depth + 1) for item in values[current]]
            return select(values[current], depth + 1)

        return MockTensor(select(self.values, 0))

    def reshape(self, *shape: int) -> MockTensor:
        flattened = _flatten(self.values)
        if len(flattened) != _product(shape):
            raise ValueError("invalid mock tensor reshape")
        return MockTensor(_nest(flattened, shape))


class _MockInferenceContext(AbstractContextManager[None]):
    def __init__(self, torch: MockTorch) -> None:
        self.torch = torch

    def __enter__(self) -> None:
        self.torch.inference_active = True
        return None

    def __exit__(self, *_args: Any) -> None:
        self.torch.inference_active = False


class MockTorch:
    float32 = "torch.float32"
    uint8 = "torch.uint8"

    def __init__(self) -> None:
        self.deterministic = False
        self.inference_active = False
        self.cuda = SimpleNamespace(
            is_available=lambda: False,
            synchronize=lambda: None,
        )
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True)
        )

    def equal(self, left: MockTensor, right: MockTensor) -> bool:
        return left.shape == right.shape and left.tolist() == right.tolist()

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic

    def use_deterministic_algorithms(self, value: bool) -> None:
        self.deterministic = value

    def inference_mode(self) -> _MockInferenceContext:
        return _MockInferenceContext(self)


class MockParameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> MockParameter:
        self.requires_grad = value
        return self


class MockProcessor:
    def __call__(self, *, images: Any, return_tensors: str) -> dict[str, MockTensor]:
        assert return_tensors == "pt"
        pixels = images.pixels
        channels = []
        for channel in range(3):
            scalar = pixels[channel] / 255.0
            channels.append([[scalar] * 4 for _ in range(4)])
        return {"pixel_values": MockTensor([channels])}


class MockModel:
    def __init__(self, torch: MockTorch) -> None:
        self.torch = torch
        self.config = SimpleNamespace(
            model_type="mock-vit",
            patch_size=2,
            hidden_size=2,
            num_register_tokens=0,
        )
        self.training = True
        self.parameter = MockParameter()
        self.calls = 0

    def eval(self) -> MockModel:
        self.training = False
        return self

    def parameters(self) -> list[MockParameter]:
        return [self.parameter]

    def to(self, _device: str) -> MockModel:
        return self

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        assert self.torch.inference_active
        assert self.torch.deterministic
        self.calls += 1
        pixel_values = inputs["pixel_values"].tolist()
        scalar = sum(_flatten(pixel_values))
        sequence = [[scalar + token, scalar - token] for token in range(5)]
        return SimpleNamespace(last_hidden_state=MockTensor([sequence]))


def _make_adapter() -> tuple[FrozenVisionEncoderAdapter, MockModel]:
    torch = MockTorch()
    model = MockModel(torch)
    adapter = FrozenVisionEncoderAdapter(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        model=model,
        processor=MockProcessor(),
        preprocessing=ImagePreprocessingConfig(
            resize_shortest_edge=4,
            crop_height=4,
            crop_width=4,
        ),
        torch_module=torch,
        image_converter=lambda image: SimpleNamespace(
            mode="RGB",
            size=(image.width, image.height),
            pixels=image.pixels,
        ),
    )
    return adapter, model


def _image(first_pixel: int, *, width: int = 4, height: int = 4) -> RGBImage:
    pixels = bytes((first_pixel, 20, 30)) * (width * height)
    return RGBImage(width, height, pixels)


def test_first_frame_is_a_miss_and_calls_the_model_once() -> None:
    adapter, model = _make_adapter()
    cache = ExactWholeFrameCache(adapter, capacity=4)

    result = cache.encode_frame(_image(1))

    assert result.cache_hit is False
    assert cache.cache_misses == 1
    assert cache.actual_model_forward_calls == model.calls == 1


@pytest.mark.parametrize(
    ("representation", "expected_shape"),
    [("spatial_tokens", (2, 2, 2)), ("global_embedding", (2,))],
)
def test_identical_preprocessed_frame_reuses_the_complete_representation(
    representation: str, expected_shape: tuple[int, ...]
) -> None:
    adapter, model = _make_adapter()
    cache = ExactWholeFrameCache(adapter, capacity=4)

    first = cache.encode_frame(_image(12), representation=representation)  # type: ignore[arg-type]
    second = cache.encode_frame(_image(12), representation=representation)  # type: ignore[arg-type]

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.encoding.feature_shape == expected_shape
    assert first.encoding.features.tolist() == second.encoding.features.tolist()
    assert cache.cache_hits == 1
    assert cache.actual_model_forward_calls == model.calls == 1


def test_changed_preprocessed_frame_runs_a_new_full_forward() -> None:
    adapter, model = _make_adapter()
    cache = ExactWholeFrameCache(adapter, capacity=4)

    first = cache.encode_frame(_image(10))
    changed = cache.encode_frame(_image(11))

    assert first.cache_hit is False
    assert changed.cache_hit is False
    assert first.content_hash != changed.content_hash
    assert first.encoding.features.tolist() != changed.encoding.features.tolist()
    assert cache.cache_misses == cache.actual_model_forward_calls == model.calls == 2


def test_episode_reset_clears_entries_but_preserves_run_counters() -> None:
    adapter, model = _make_adapter()
    cache = ExactWholeFrameCache(adapter, capacity=4)
    frame = _image(22)

    cache.encode_frame(frame)
    assert cache.encode_frame(frame).cache_hit is True
    cache.reset_episode()
    after_reset = cache.encode_frame(frame)

    assert after_reset.cache_hit is False
    assert cache.episode_resets == 1
    assert cache.cache_hits == 1
    assert cache.cache_misses == cache.actual_model_forward_calls == model.calls == 2


def test_hash_collision_requires_exact_tensor_equality() -> None:
    adapter, model = _make_adapter()
    cache = ExactWholeFrameCache(
        adapter,
        capacity=4,
        hash_function=lambda _inputs: "forced-collision",
    )

    first = cache.encode_frame(_image(30))
    second = cache.encode_frame(_image(31))
    first_again = cache.encode_frame(_image(30))

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert first_again.cache_hit is True
    assert first.encoding.features.tolist() == first_again.encoding.features.tolist()
    assert first.encoding.features.tolist() != second.encoding.features.tolist()
    assert cache.actual_model_forward_calls == model.calls == 2


def test_lru_eviction_is_deterministic_and_respects_capacity() -> None:
    adapter, model = _make_adapter()
    cache = ExactWholeFrameCache(adapter, capacity=2)

    cache.encode_frame(_image(1))
    cache.encode_frame(_image(2))
    assert cache.encode_frame(_image(1)).cache_hit is True
    cache.encode_frame(_image(3))
    assert cache.current_cache_entries == 2
    assert cache.encode_frame(_image(1)).cache_hit is True
    assert cache.encode_frame(_image(2)).cache_hit is False

    assert cache.cache_evictions == 2
    assert cache.actual_model_forward_calls == model.calls == 4


def test_cache_output_matches_an_independent_full_inference() -> None:
    adapter, model = _make_adapter()
    cache = ExactWholeFrameCache(adapter, capacity=2)
    image = _image(55)

    cache.encode_frame(image)
    repeated = cache.encode_frame(image)
    independent = adapter.encode_frame(image)

    assert repeated.cache_hit
    assert repeated.encoding.features.tolist() == independent.features.tolist()
    assert cache.actual_model_forward_calls == 1
    assert model.calls == 2  # one cache miss plus the explicit independent call


def test_cache_hit_preserves_current_input_dimensions() -> None:
    adapter, _ = _make_adapter()
    cache = ExactWholeFrameCache(adapter, capacity=2)
    small = _image(70, width=2, height=2)
    larger = _image(70, width=4, height=4)

    cache.encode_frame(small)
    hit = cache.encode_frame(larger)

    assert hit.cache_hit
    assert (hit.encoding.input_width, hit.encoding.input_height) == (4, 4)
    assert (hit.encoding.processed_width, hit.encoding.processed_height) == (4, 4)


def test_stateful_runner_serializes_metadata_metrics_and_feature_arrays(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    episode_dir = dataset / "episode_0000"
    episode_dir.mkdir(parents=True)
    frame_path = episode_dir / "frame_0000.ppm"
    frame_path.write_bytes(b"P6\n4 4\n255\n" + bytes((8, 18, 28)) * 16)
    manifest = dataset / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "episode_id": "episode_0000",
                "frame_paths": [
                    "episode_0000/frame_0000.ppm",
                    "episode_0000/frame_0000.ppm",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter, model = _make_adapter()

    output = run_pretrained_stateful(
        manifest,
        tmp_path / "results" / "stateful",
        adapter,
        cache_capacity=2,
    )

    metadata = json.loads((output / "run_metadata.json").read_text("utf-8"))
    metrics = json.loads((output / "stateful_metrics.json").read_text("utf-8"))
    records = [
        json.loads(line)
        for line in (output / "frames.jsonl").read_text("utf-8").splitlines()
    ]
    assert metadata["model_identifier"] == DEFAULT_MODEL_ID
    assert metadata["model_revision"] == DEFAULT_MODEL_REVISION
    assert metadata["temporal_state_or_cache"] is True
    assert metadata["cache_design"]["partial_token_computation"] is False
    assert len(records) == 2
    assert [record["frame_index"] for record in records] == [0, 1]
    assert [record["cache_hit"] for record in records] == [False, True]
    assert records[0]["source_frame_path"] == records[1]["source_frame_path"]
    assert records[0]["input_dimensions"] == records[1]["input_dimensions"]
    assert records[0]["feature_shape"] == records[1]["feature_shape"] == [2, 2, 2]
    assert all((output / record["feature_array_path"]).is_file() for record in records)
    assert metrics["cache_metrics"]["total_frames"] == 2
    assert metrics["cache_metrics"]["actual_dinov2_forward_calls"] == 1
    assert metrics["cache_metrics"]["cached_frame_hits"] == 1
    assert metrics["cache_metrics"]["full_model_forward_calls_avoided"] == 1
    assert (
        metrics["numerical_differences_against_independent_inference"]["aggregate"][
            "exact_match"
        ]
        is True
    )
    assert (
        metrics["numerical_differences_against_independent_inference"][
            "comparison_forward_calls"
        ]
        == 2
    )
    assert model.calls == 3  # one stateful miss plus two independent comparison calls


def test_zero_capacity_disables_storage_but_not_correctness() -> None:
    adapter, model = _make_adapter()
    cache = ExactWholeFrameCache(adapter, capacity=0)
    frame = _image(88)

    first = cache.encode_frame(frame)
    second = cache.encode_frame(frame)

    assert not first.cache_hit and not second.cache_hit
    assert cache.current_cache_entries == 0
    assert cache.actual_model_forward_calls == model.calls == 2


def test_negative_cache_capacity_is_rejected() -> None:
    adapter, _ = _make_adapter()
    with pytest.raises(ValueError, match="non-negative"):
        ExactWholeFrameCache(adapter, capacity=-1)


def test_runner_blocks_held_out_test_manifest_before_reading_it(
    tmp_path: Path,
) -> None:
    adapter, _ = _make_adapter()
    held_out_manifest = tmp_path / "research_benchmark" / "test" / "manifest.jsonl"

    with pytest.raises(ValueError, match="held-out test split"):
        run_pretrained_stateful(
            held_out_manifest,
            tmp_path / "results" / "stateful",
            adapter,
        )
