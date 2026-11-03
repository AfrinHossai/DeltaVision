"""Runner and portable output format for frozen independent-frame features."""

from __future__ import annotations

import array
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, TextIO

from .baseline import load_manifest
from .encoder import read_ppm
from .pretrained import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    MODEL_LICENSE,
    MODEL_REVISION_URL,
    MODEL_SOURCE_URL,
    FrozenVisionEncoderAdapter,
    RepresentationKind,
    VisionEncoding,
)


def run_pretrained_baseline(
    manifest_path: str | Path,
    output_dir: str | Path,
    adapter: FrozenVisionEncoderAdapter,
    *,
    representation: RepresentationKind = "spatial_tokens",
) -> Path:
    """Encode each frame independently and store JSONL metadata and float data.

    Frame loading and output serialization are outside the adapter's reported
    inference latency. Each feature file contains row-major little-endian
    float32 values; its shape and relative path are stored in ``frames.jsonl``.
    The model is called once per frame, with no state or token reuse.
    """
    episodes = load_manifest(manifest_path)
    manifest = Path(manifest_path).resolve()
    dataset_root = manifest.parent
    destination = Path(output_dir)
    resolved_destination = destination.resolve()
    if (
        resolved_destination == dataset_root
        or resolved_destination.is_relative_to(dataset_root)
        or dataset_root.is_relative_to(resolved_destination)
    ):
        raise ValueError("output_dir must not overlap the source dataset directory")
    if resolved_destination == manifest:
        raise ValueError("output_dir must not overwrite the input manifest")
    destination.mkdir(parents=True, exist_ok=True)
    feature_root = destination / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    record_path = destination / "frames.jsonl"
    with record_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for episode in episodes:
            for episode_frame_index, frame in enumerate(episode.frames):
                image = read_ppm(frame.path)
                result = adapter.encode_frame(image, representation=representation)
                feature_relative_path = f"features/frame_{frame_index:08d}.f32le"
                feature_path = destination / feature_relative_path
                _write_float32_array(feature_path, result)
                record = _frame_record(
                    episode.episode_id,
                    episode_frame_index,
                    frame.relative_path,
                    result,
                    feature_relative_path,
                )
                _write_jsonl_record(output_file, record)
                frame_index += 1

    metadata = _run_metadata(adapter, representation, frame_index)
    metadata["frame_records_path"] = "frames.jsonl"
    metadata["feature_storage"] = {
        "format": "raw-float-array",
        "dtype": "little-endian float32",
        "order": "row-major",
        "one_file_per_frame": True,
    }
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _frame_record(
    episode_id: str,
    frame_index: int,
    source_path: str,
    result: VisionEncoding,
    feature_path: str,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "frame_index": frame_index,
        "source_frame_path": source_path,
        "input_dimensions": {
            "width": result.input_width,
            "height": result.input_height,
        },
        "processed_dimensions": {
            "width": result.processed_width,
            "height": result.processed_height,
        },
        "representation": result.representation,
        "feature_shape": list(result.feature_shape),
        "spatial_grid": (
            {"rows": result.spatial_grid[0], "columns": result.spatial_grid[1]}
            if result.spatial_grid is not None
            else None
        ),
        "inference_latency_seconds": result.inference_latency_seconds,
        "feature_array_path": feature_path,
    }


def _write_float32_array(path: Path, result: VisionEncoding) -> None:
    values = _flatten_values(result.features)
    expected_values = 1
    for dimension in result.feature_shape:
        if dimension < 1:
            raise ValueError("feature dimensions must be positive")
        expected_values *= dimension
    if len(values) != expected_values:
        raise ValueError(
            f"feature value count {len(values)} does not match declared shape "
            f"{result.feature_shape}"
        )
    packed = array.array("f", values)
    if packed.itemsize != 4:
        raise RuntimeError("this platform does not provide 32-bit float array storage")
    if sys.byteorder != "little":
        packed.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed.tobytes())


def _flatten_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return flattened
    try:
        return [float(value)]
    except (TypeError, ValueError) as error:
        raise ValueError("feature output must contain numeric values") from error


def _run_metadata(
    adapter: FrozenVisionEncoderAdapter,
    representation: RepresentationKind,
    frame_count: int,
) -> dict[str, Any]:
    is_verified_default = (
        adapter.model_id == DEFAULT_MODEL_ID
        and adapter.revision == DEFAULT_MODEL_REVISION
    )
    model_config = getattr(adapter.model, "config", None)
    return {
        "model_identifier": adapter.model_id,
        "model_revision": adapter.revision,
        "model_source": MODEL_SOURCE_URL if is_verified_default else None,
        "revision_source": MODEL_REVISION_URL if is_verified_default else None,
        "license": MODEL_LICENSE if is_verified_default else "not verified by runner",
        "preprocessing": adapter.preprocessing.to_dict(),
        "representation": representation,
        "representation_semantics": _representation_semantics(representation),
        "frozen_inference": True,
        "deterministic_evaluation": adapter.deterministic,
        "device": adapter.device,
        "input_frames_encoded": frame_count,
        "model_config": {
            "model_type": getattr(model_config, "model_type", None),
            "patch_size": getattr(model_config, "patch_size", None),
            "hidden_size": getattr(model_config, "hidden_size", None),
            "num_register_tokens": getattr(model_config, "num_register_tokens", 0),
        },
        "hardware": _hardware_metadata(adapter),
        "timing_scope": (
            "per-frame preprocessing and model inference; excludes PPM loading, "
            "output serialization, plotting, and interpreter startup"
        ),
        "temporal_state_or_cache": False,
    }


def _representation_semantics(kind: RepresentationKind) -> str:
    if kind == "spatial_tokens":
        return (
            "Learned contextual patch-token grid; not raw image patches, and tokens "
            "may depend on pixels throughout the entire input image."
        )
    if kind == "global_embedding":
        return "Global CLS image embedding from the final transformer layer."
    return "Full final-layer learned token sequence, including special tokens."


def _hardware_metadata(adapter: FrozenVisionEncoderAdapter) -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
    }
    for package, key in (
        ("torch", "torch_version"),
        ("transformers", "transformers_version"),
    ):
        try:
            hardware[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            hardware[key] = None
    hardware["cuda_available"] = bool(adapter._torch.cuda.is_available())
    if adapter.device == "cuda":
        hardware["cuda_device"] = adapter._torch.cuda.get_device_name()
    return hardware


def _write_jsonl_record(output_file: TextIO, record: dict[str, Any]) -> None:
    output_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
    output_file.write("\n")
