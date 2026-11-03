"""Offline tests for frozen pretrained encoder integration."""

from __future__ import annotations

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
    ModelLoadingError,
)
from deltavision.pretrained_baseline import run_pretrained_baseline
from deltavision.synthetic import generate_sequences


class FakeTensor:
    """Small tensor stand-in supporting the adapter's required tensor methods."""

    def __init__(self, values: Any) -> None:
        self.values = values
        self.shape = _shape(values)

    def to(self, *_args: Any, **_kwargs: Any) -> FakeTensor:
        return self

    def __getitem__(self, index: Any) -> FakeTensor:
        indexes = index if isinstance(index, tuple) else (index,)

        def select(values: Any, depth: int) -> Any:
            if depth == len(indexes):
                return values
            current = indexes[depth]
            if isinstance(current, slice):
                return [select(item, depth + 1) for item in values[current]]
            return select(values[current], depth + 1)

        return FakeTensor(select(self.values, 0))

    def reshape(self, *shape: int) -> FakeTensor:
        flat = _flatten(self.values)
        if len(flat) != _product(shape):
            raise ValueError("bad reshape")
        return FakeTensor(_nest(flat, shape))

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> Any:
        return self.values


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


def _product(values: tuple[int, ...]) -> int:
    product = 1
    for value in values:
        product *= value
    return product


def _nest(values: list[float], shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return values
    stride = _product(shape[1:])
    return [
        _nest(values[index * stride : (index + 1) * stride], shape[1:])
        for index in range(shape[0])
    ]


class FakeInferenceContext(AbstractContextManager[None]):
    def __init__(self, torch: FakeTorch) -> None:
        self.torch = torch

    def __enter__(self) -> None:
        self.torch.inference_active = True
        return None

    def __exit__(self, *_args: Any) -> None:
        self.torch.inference_active = False


class FakeTorch:
    def __init__(self) -> None:
        self._deterministic = False
        self.inference_active = False
        self.cuda = SimpleNamespace(
            is_available=lambda: False, synchronize=lambda: None
        )
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True)
        )

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._deterministic

    def use_deterministic_algorithms(self, value: bool) -> None:
        self._deterministic = value

    def inference_mode(self) -> FakeInferenceContext:
        return FakeInferenceContext(self)


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> FakeParameter:
        self.requires_grad = value
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, *, images: Any, return_tensors: str) -> dict[str, FakeTensor]:
        self.calls.append((images, return_tensors))
        zeros = [[[[0.0] * 4 for _ in range(4)] for _ in range(3)]]
        return {"pixel_values": FakeTensor(zeros)}


class FakeModel:
    def __init__(self, torch: FakeTorch, tokens: int = 5) -> None:
        self.config = SimpleNamespace(
            patch_size=2,
            hidden_size=2,
            num_register_tokens=0,
            model_type="mock-vit",
        )
        self.training = True
        self.parameter = FakeParameter()
        self.torch = torch
        self.tokens = tokens
        self.calls = 0

    def eval(self) -> FakeModel:
        self.training = False
        return self

    def parameters(self) -> list[FakeParameter]:
        return [self.parameter]

    def to(self, _device: str) -> FakeModel:
        return self

    def __call__(self, **_inputs: Any) -> SimpleNamespace:
        assert self.torch.inference_active
        assert self.torch._deterministic
        self.calls += 1
        values = [[[float(token), float(token + 1)] for token in range(self.tokens)]]
        return SimpleNamespace(last_hidden_state=FakeTensor(values))


def _make_adapter(
    *, tokens: int = 5
) -> tuple[FrozenVisionEncoderAdapter, FakeModel, FakeProcessor, FakeTorch]:
    torch = FakeTorch()
    model = FakeModel(torch, tokens=tokens)
    processor = FakeProcessor()
    adapter = FrozenVisionEncoderAdapter(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        model=model,
        processor=processor,
        preprocessing=ImagePreprocessingConfig(
            resize_shortest_edge=8,
            crop_height=4,
            crop_width=4,
        ),
        torch_module=torch,
        image_converter=lambda image: SimpleNamespace(
            mode="RGB", size=(image.width, image.height)
        ),
    )
    return adapter, model, processor, torch


def test_adapter_returns_spatial_tokens_and_applies_preprocessing() -> None:
    adapter, model, processor, torch = _make_adapter()
    image = RGBImage(3, 2, bytes((10, 20, 30)) * 6)

    result = adapter.encode_frame(image)

    assert result.representation == "spatial_tokens"
    assert result.feature_shape == (2, 2, 2)
    assert result.spatial_grid == (2, 2)
    assert (result.input_width, result.input_height) == (3, 2)
    assert (result.processed_width, result.processed_height) == (4, 4)
    assert result.inference_latency_seconds >= 0
    pil_image, tensor_format = processor.calls[0]
    assert pil_image.mode == "RGB"
    assert pil_image.size == (3, 2)
    assert tensor_format == "pt"
    assert processor.size == {"shortest_edge": 8}
    assert processor.crop_size == {"height": 4, "width": 4}
    assert processor.image_mean == [0.485, 0.456, 0.406]
    assert model.calls == 1
    assert not model.training
    assert model.parameter.requires_grad is False
    assert torch._deterministic is False
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


@pytest.mark.parametrize(
    ("kind", "shape", "grid"),
    [
        ("spatial_tokens", (2, 2, 2), (2, 2)),
        ("global_embedding", (2,), None),
        ("token_sequence", (5, 2), (2, 2)),
    ],
)
def test_adapter_supports_representation_shapes(
    kind: str, shape: tuple[int, ...], grid: tuple[int, int] | None
) -> None:
    adapter, _, _, _ = _make_adapter()

    result = adapter.encode_frame(
        RGBImage(4, 4, bytes((1, 2, 3)) * 16), representation=kind
    )  # type: ignore[arg-type]

    assert result.feature_shape == shape
    assert result.spatial_grid == grid


def test_adapter_rejects_invalid_model_token_output_shape() -> None:
    adapter, _, _, _ = _make_adapter(tokens=4)

    with pytest.raises(ValueError, match="token count"):
        adapter.encode_frame(RGBImage(4, 4, bytes((1, 2, 3)) * 16))


def test_adapter_rejects_invalid_preprocessing_and_device() -> None:
    with pytest.raises(ValueError, match="image_std"):
        ImagePreprocessingConfig(image_std=(1.0, 0.0, 1.0))
    adapter, *_ = _make_adapter()
    with pytest.raises(ValueError, match="unsupported representation"):
        adapter.encode_frame(
            RGBImage(4, 4, bytes((1, 2, 3)) * 16),
            representation="unknown",  # type: ignore[arg-type]
        )


def test_loader_reports_missing_cached_checkpoint_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingLoader:
        @staticmethod
        def from_pretrained(*_args: Any, **kwargs: Any) -> Any:
            assert kwargs["local_files_only"] is True
            raise OSError("not cached")

    transformers = SimpleNamespace(
        AutoImageProcessor=MissingLoader,
        AutoModel=MissingLoader,
    )
    monkeypatch.setattr(
        "deltavision.pretrained.importlib.import_module",
        lambda name: transformers if name == "transformers" else None,
    )

    with pytest.raises(ModelLoadingError, match="Pass --allow-download"):
        FrozenVisionEncoderAdapter.from_pretrained()


def test_manifest_runner_writes_referenced_features_and_metadata(
    tmp_path: Path,
) -> None:
    manifest = generate_sequences(
        tmp_path / "dataset",
        width=24,
        height=20,
        sequence_length=2,
        num_episodes=1,
        seed=500,
    )
    adapter, model, _, _ = _make_adapter()
    output_dir = tmp_path / "results" / "pretrained"

    result_dir = run_pretrained_baseline(manifest, output_dir, adapter)

    metadata = json.loads((result_dir / "run_metadata.json").read_text("utf-8"))
    records = [
        json.loads(line)
        for line in (result_dir / "frames.jsonl").read_text("utf-8").splitlines()
    ]
    assert metadata["model_identifier"] == DEFAULT_MODEL_ID
    assert metadata["model_revision"] == DEFAULT_MODEL_REVISION
    assert metadata["license"] == "Apache-2.0"
    assert metadata["preprocessing"]["crop_height"] == 4
    assert metadata["representation"] == "spatial_tokens"
    assert metadata["temporal_state_or_cache"] is False
    assert metadata["input_frames_encoded"] == 2
    assert len(records) == 2
    assert [(record["episode_id"], record["frame_index"]) for record in records] == [
        ("episode_0000", 0),
        ("episode_0000", 1),
    ]
    assert model.calls == 2
    for record in records:
        feature_path = Path(record["feature_array_path"])
        assert not feature_path.is_absolute()
        feature_bytes = (result_dir / feature_path).read_bytes()
        assert len(feature_bytes) == 2 * 2 * 2 * 4
        assert record["feature_shape"] == [2, 2, 2]
        assert record["spatial_grid"] == {"rows": 2, "columns": 2}
        assert record["inference_latency_seconds"] >= 0


def test_runner_rejects_output_that_overlaps_source_dataset(tmp_path: Path) -> None:
    manifest = generate_sequences(
        tmp_path / "dataset", width=16, height=16, sequence_length=1, seed=4
    )
    adapter, *_ = _make_adapter()

    with pytest.raises(ValueError, match="overlap"):
        run_pretrained_baseline(manifest, manifest.parent / "outputs", adapter)
