"""Frozen adapters for independent-frame pretrained vision representations.

The default model is Meta's DINOv2-small checkpoint. Its spatial outputs are
contextual transformer tokens, not independently encoded raw image patches.
This module deliberately provides no temporal state or token cache.
"""

from __future__ import annotations

import importlib
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterator, Literal, Mapping

from .encoder import RGBImage

DEFAULT_MODEL_ID = "facebook/dinov2-small"
DEFAULT_MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_SOURCE_URL = "https://huggingface.co/facebook/dinov2-small"
MODEL_REVISION_URL = (
    f"https://huggingface.co/facebook/dinov2-small/tree/{DEFAULT_MODEL_REVISION}"
)
MODEL_LICENSE = "Apache-2.0"
RepresentationKind = Literal["spatial_tokens", "global_embedding", "token_sequence"]


class ModelLoadingError(RuntimeError):
    """Raised when optional libraries or the requested checkpoint are unavailable."""


@dataclass(frozen=True, slots=True)
class ImagePreprocessingConfig:
    """Image preprocessing options compatible with HF image processors.

    Defaults match the pinned ``facebook/dinov2-small`` processor revision:
    convert to RGB, resize the shortest edge to 256 with bicubic interpolation,
    center-crop to 224 square, rescale bytes by 1/255, and apply ImageNet
    channel normalization.
    """

    do_convert_rgb: bool = True
    do_resize: bool = True
    resize_shortest_edge: int = 256
    resample: int = 3
    do_center_crop: bool = True
    crop_height: int = 224
    crop_width: int = 224
    do_rescale: bool = True
    rescale_factor: float = 1.0 / 255.0
    do_normalize: bool = True
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.resize_shortest_edge < 1:
            raise ValueError("resize_shortest_edge must be positive")
        if self.crop_height < 1 or self.crop_width < 1:
            raise ValueError("crop_height and crop_width must be positive")
        if self.resample not in (0, 1, 2, 3, 4):
            raise ValueError("resample must be a supported PIL interpolation value")
        if self.rescale_factor <= 0:
            raise ValueError("rescale_factor must be positive")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must each contain 3 values")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("image_std values must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this configuration."""
        return asdict(self)

    def apply_to_processor(self, processor: Any) -> None:
        """Set the configured options on a Hugging Face image processor."""
        options = {
            "do_convert_rgb": self.do_convert_rgb,
            "do_resize": self.do_resize,
            "size": {"shortest_edge": self.resize_shortest_edge},
            "resample": self.resample,
            "do_center_crop": self.do_center_crop,
            "crop_size": {"height": self.crop_height, "width": self.crop_width},
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
        }
        for name, value in options.items():
            setattr(processor, name, value)


@dataclass(frozen=True, slots=True)
class PreparedVisionInput:
    """One image after the configured processor, before device transfer."""

    model_inputs: Mapping[str, Any]
    input_width: int
    input_height: int
    processed_width: int
    processed_height: int


@dataclass(frozen=True, slots=True)
class VisionEncoding:
    """One image's selected representation and shape metadata."""

    features: Any
    representation: RepresentationKind
    input_width: int
    input_height: int
    processed_width: int
    processed_height: int
    feature_shape: tuple[int, ...]
    spatial_grid: tuple[int, int] | None
    inference_latency_seconds: float
    model_forward_latency_seconds: float = 0.0


class FrozenVisionEncoderAdapter:
    """Run a frozen model independently on each RGB image.

    ``spatial_tokens`` returns the learned patch-token grid, excluding class
    and register tokens. DINOv2's self-attention makes these tokens contextual:
    each token can depend on every input patch, so these are not raw image
    patches and cannot be recomputed independently. ``global_embedding`` is
    the CLS token. ``token_sequence`` retains all output tokens, including
    special tokens. No representation is cached between calls.
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        model: Any,
        processor: Any,
        device: str = "cpu",
        deterministic: bool = True,
        preprocessing: ImagePreprocessingConfig | None = None,
        torch_module: Any | None = None,
        image_converter: Callable[[RGBImage], Any] | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("revision must be a full 40-character commit SHA")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        self.model_id = model_id
        self.revision = revision.lower()
        self.model = model
        self.processor = processor
        self.device = device
        self.deterministic = deterministic
        self.preprocessing = preprocessing or ImagePreprocessingConfig()
        self.preprocessing.apply_to_processor(self.processor)
        self._torch = torch_module
        self._image_converter = image_converter
        if self._torch is None:
            try:
                self._torch = importlib.import_module("torch")
            except ImportError as error:
                raise ModelLoadingError(
                    "PyTorch is required for pretrained inference. Install the "
                    "optional dependencies with `pip install -e '.[pretrained]'`."
                ) from error
        if device == "cuda" and not self._torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")

        self.model.eval()
        parameters = getattr(self.model, "parameters", None)
        if parameters is not None:
            for parameter in parameters():
                parameter.requires_grad_(False)
        self.model = self.model.to(device)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
        deterministic: bool = True,
        preprocessing: ImagePreprocessingConfig | None = None,
        cache_dir: str | None = None,
        allow_download: bool = False,
    ) -> FrozenVisionEncoderAdapter:
        """Load pinned HF model files, never downloading unless explicitly allowed."""
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("revision must be a full 40-character commit SHA")
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise ModelLoadingError(
                "Transformers is required for checkpoint loading. Install the "
                "optional dependencies with `pip install -e '.[pretrained]'`."
            ) from error

        load_options: dict[str, Any] = {
            "revision": revision,
            "local_files_only": not allow_download,
        }
        if cache_dir is not None:
            load_options["cache_dir"] = cache_dir
        try:
            processor = transformers.AutoImageProcessor.from_pretrained(
                model_id, **load_options
            )
            model = transformers.AutoModel.from_pretrained(
                model_id,
                use_safetensors=True,
                trust_remote_code=False,
                **load_options,
            )
        except (OSError, ValueError, RuntimeError) as error:
            download_hint = (
                " Pass --allow-download to fetch it from Hugging Face."
                if not allow_download
                else " Check network access and the model revision."
            )
            raise ModelLoadingError(
                f"Could not load pretrained weights for {model_id}@{revision}. "
                "The checkpoint may be missing from the local Hugging Face cache."
                f"{download_hint}"
            ) from error
        return cls(
            model_id=model_id,
            revision=revision,
            model=model,
            processor=processor,
            device=device,
            deterministic=deterministic,
            preprocessing=preprocessing,
        )

    def encode_frame(
        self, image: RGBImage, *, representation: RepresentationKind = "spatial_tokens"
    ) -> VisionEncoding:
        """Preprocess and encode one frame with frozen inference settings."""
        self._validate_representation(representation)
        self._synchronize()
        preprocessing_started = time.perf_counter()
        prepared = self.prepare_frame(image)
        preprocessing_latency = time.perf_counter() - preprocessing_started
        encoding = self.encode_preprocessed(prepared, representation=representation)
        return replace(
            encoding,
            inference_latency_seconds=(
                preprocessing_latency + encoding.inference_latency_seconds
            ),
        )

    def prepare_frame(self, image: RGBImage) -> PreparedVisionInput:
        """Apply the existing image processor and retain its complete CPU inputs."""
        try:
            pil_image = (
                self._image_converter(image)
                if self._image_converter is not None
                else self._to_pil(image)
            )
        except ImportError as error:
            raise ModelLoadingError(
                "Pillow is required for image preprocessing. Install the optional "
                "dependencies with `pip install -e '.[pretrained]'`."
            ) from error

        inputs = self.processor(images=pil_image, return_tensors="pt")
        if not isinstance(inputs, Mapping):
            raise ValueError("image processor must return a mapping of model inputs")
        model_inputs = dict(inputs)
        pixel_values = model_inputs.get("pixel_values")
        if pixel_values is None or len(pixel_values.shape) != 4:
            raise ValueError(
                "processor must return pixel_values with shape [1, C, H, W]"
            )
        if pixel_values.shape[0] != 1:
            raise ValueError("processor must return exactly one preprocessed image")
        processed_height, processed_width = (
            int(pixel_values.shape[-2]),
            int(pixel_values.shape[-1]),
        )
        return PreparedVisionInput(
            model_inputs=model_inputs,
            input_width=image.width,
            input_height=image.height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

    def encode_preprocessed(
        self,
        prepared: PreparedVisionInput,
        *,
        representation: RepresentationKind = "spatial_tokens",
    ) -> VisionEncoding:
        """Run the full frozen model on an already processed single image."""
        self._validate_representation(representation)
        self._synchronize()
        started = time.perf_counter()
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in prepared.model_inputs.items()
        }
        pixel_values = model_inputs.get("pixel_values")
        if pixel_values is None or len(pixel_values.shape) != 4:
            raise ValueError(
                "prepared model inputs must include pixel_values with shape "
                "[1, C, H, W]"
            )
        if pixel_values.shape[0] != 1:
            raise ValueError("prepared inputs must contain exactly one image")
        self._synchronize()
        forward_started = time.perf_counter()
        with self._deterministic_inference_context():
            output = self.model(**model_inputs)
        self._synchronize()
        forward_latency = time.perf_counter() - forward_started
        inference_latency = time.perf_counter() - started
        sequence = _last_hidden_state(output)
        if len(sequence.shape) != 3 or sequence.shape[0] != 1:
            raise ValueError(
                "model last_hidden_state must have shape [1, tokens, hidden]"
            )
        hidden_size = int(sequence.shape[-1])
        config = getattr(self.model, "config", None)
        configured_hidden = getattr(config, "hidden_size", hidden_size)
        if int(configured_hidden) != hidden_size:
            raise ValueError("model output hidden size does not match model config")
        grid, num_special_tokens = _spatial_grid(
            config, prepared.processed_height, prepared.processed_width
        )

        if representation == "token_sequence":
            features = sequence[0]
            feature_shape = tuple(int(value) for value in features.shape)
        elif representation == "global_embedding":
            features = sequence[0, 0]
            feature_shape = tuple(int(value) for value in features.shape)
        else:
            if grid is None:
                raise ValueError(
                    "model config must define patch_size to return spatial_tokens"
                )
            grid_rows, grid_columns = grid
            expected_tokens = grid_rows * grid_columns + num_special_tokens
            if int(sequence.shape[1]) != expected_tokens:
                raise ValueError(
                    "model token count does not match its spatial grid and special "
                    f"token count: got {sequence.shape[1]}, expected {expected_tokens}"
                )
            patch_tokens = sequence[0, num_special_tokens:, :]
            features = patch_tokens.reshape(grid_rows, grid_columns, hidden_size)
            feature_shape = tuple(int(value) for value in features.shape)

        return VisionEncoding(
            features=features,
            representation=representation,
            input_width=prepared.input_width,
            input_height=prepared.input_height,
            processed_width=prepared.processed_width,
            processed_height=prepared.processed_height,
            feature_shape=feature_shape,
            spatial_grid=grid if representation != "global_embedding" else None,
            inference_latency_seconds=inference_latency,
            model_forward_latency_seconds=forward_latency,
        )

    @staticmethod
    def _validate_representation(representation: RepresentationKind) -> None:
        if representation not in (
            "spatial_tokens",
            "global_embedding",
            "token_sequence",
        ):
            raise ValueError(f"unsupported representation: {representation}")

    def _to_pil(self, image: RGBImage) -> Any:
        pil_image_module = importlib.import_module("PIL.Image")
        return pil_image_module.frombytes(
            "RGB", (image.width, image.height), image.pixels
        )

    def _synchronize(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.synchronize()

    @contextmanager
    def _deterministic_inference_context(self) -> Iterator[None]:
        torch = self._torch
        prior_algorithms = torch.are_deterministic_algorithms_enabled()
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        prior_cudnn_deterministic = getattr(cudnn, "deterministic", None)
        prior_cudnn_benchmark = getattr(cudnn, "benchmark", None)
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            if cudnn is not None:
                cudnn.deterministic = True
                cudnn.benchmark = False
        try:
            with torch.inference_mode():
                yield
        finally:
            if self.deterministic:
                torch.use_deterministic_algorithms(prior_algorithms)
                if cudnn is not None:
                    cudnn.deterministic = prior_cudnn_deterministic
                    cudnn.benchmark = prior_cudnn_benchmark


def _last_hidden_state(output: Any) -> Any:
    if hasattr(output, "last_hidden_state"):
        sequence = output.last_hidden_state
    elif isinstance(output, Mapping) and "last_hidden_state" in output:
        sequence = output["last_hidden_state"]
    elif isinstance(output, tuple) and output:
        sequence = output[0]
    else:
        raise ValueError("model output must provide last_hidden_state")
    if not hasattr(sequence, "shape"):
        raise ValueError("model last_hidden_state must be a tensor")
    return sequence


def _spatial_grid(
    config: Any, processed_height: int, processed_width: int
) -> tuple[tuple[int, int] | None, int]:
    patch_size = getattr(config, "patch_size", None)
    if patch_size is None:
        return None, 1 + int(getattr(config, "num_register_tokens", 0))
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        patch_height, patch_width = (int(value) for value in patch_size)
    else:
        raise ValueError("model config patch_size must be an int or height-width pair")
    if patch_height < 1 or patch_width < 1:
        raise ValueError("model config patch_size values must be positive")
    if processed_height % patch_height or processed_width % patch_width:
        raise ValueError(
            "processed image dimensions must be divisible by model patch size"
        )
    rows, columns = processed_height // patch_height, processed_width // patch_width
    special = 1 + int(getattr(config, "num_register_tokens", 0))
    return (rows, columns), special
