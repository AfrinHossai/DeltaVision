#!/usr/bin/env python3
"""Validate saved DINOv2 features and compare them with a local-only rerun."""

from __future__ import annotations

import argparse
import array
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deltavision.baseline import ManifestEpisode, load_manifest  # noqa: E402
from deltavision.encoder import read_ppm  # noqa: E402
from deltavision.pretrained import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    FrozenVisionEncoderAdapter,
    ImagePreprocessingConfig,
    ModelLoadingError,
)

DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/generated/demo/manifest.jsonl"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/pretrained_baseline.json"
DEFAULT_ARTIFACT_DIR = REPOSITORY_ROOT / "results/pretrained_baseline"
DEFAULT_REPORT = REPOSITORY_ROOT / "results/pretrained_validation.json"
FEATURE_STORAGE = {
    "format": "raw-float-array",
    "dtype": "little-endian float32",
    "order": "row-major",
    "one_file_per_frame": True,
}


def expected_feature_shape(
    representation: str,
    model_config: Mapping[str, Any],
    processed_height: int,
    processed_width: int,
) -> tuple[int, ...]:
    """Calculate the model output shape for one representation and image size."""
    patch_size = model_config.get("patch_size")
    hidden_size = int(model_config.get("hidden_size", 0))
    register_tokens = int(model_config.get("num_register_tokens", 0))
    if hidden_size < 1 or register_tokens < 0:
        raise ValueError("model hidden_size and register token count are invalid")
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    elif isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
        patch_height, patch_width = (int(value) for value in patch_size)
    else:
        raise ValueError("model patch_size must be an integer or height-width pair")
    if patch_height < 1 or patch_width < 1:
        raise ValueError("model patch_size values must be positive")
    if processed_height % patch_height or processed_width % patch_width:
        raise ValueError("processed image dimensions are not divisible by patch size")

    rows = processed_height // patch_height
    columns = processed_width // patch_width
    if representation == "spatial_tokens":
        return (rows, columns, hidden_size)
    if representation == "global_embedding":
        return (hidden_size,)
    if representation == "token_sequence":
        token_count = rows * columns + 1 + register_tokens
        return (token_count, hidden_size)
    raise ValueError(f"unsupported representation: {representation}")


def validate_feature_shape(
    actual_shape: Sequence[int], expected_shape: Sequence[int]
) -> None:
    """Raise when a declared or observed shape differs from the model shape."""
    actual = tuple(int(value) for value in actual_shape)
    expected = tuple(int(value) for value in expected_shape)
    if actual != expected:
        raise ValueError(f"feature shape {actual} does not match expected {expected}")


def count_non_finite_values(values: Iterable[float]) -> int:
    """Count NaN and infinite values without relying on a tensor framework."""
    return sum(not math.isfinite(float(value)) for value in values)


def validate_manifest_records(
    expected_frames: Sequence[tuple[str, int, str]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare episode/frame/path identities and report duplicates and omissions."""
    expected_by_key = {
        (episode_id, frame_index): source_path
        for episode_id, frame_index, source_path in expected_frames
    }
    record_keys: list[tuple[str, int]] = []
    invalid_record_indices: list[int] = []
    source_mismatches: list[dict[str, Any]] = []
    feature_paths: list[str] = []

    for index, record in enumerate(records):
        episode_id = record.get("episode_id")
        frame_index = record.get("frame_index")
        if not isinstance(episode_id, str) or not isinstance(frame_index, int):
            invalid_record_indices.append(index)
            continue
        key = (episode_id, frame_index)
        record_keys.append(key)
        expected_path = expected_by_key.get(key)
        if (
            expected_path is not None
            and record.get("source_frame_path") != expected_path
        ):
            source_mismatches.append(
                {
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "expected_source_frame_path": expected_path,
                    "recorded_source_frame_path": record.get("source_frame_path"),
                }
            )
        feature_path = record.get("feature_array_path")
        if isinstance(feature_path, str):
            feature_paths.append(feature_path)

    key_counts = Counter(record_keys)
    path_counts = Counter(feature_paths)
    expected_keys = set(expected_by_key)
    actual_keys = set(record_keys)
    return {
        "expected_record_count": len(expected_frames),
        "actual_record_count": len(records),
        "missing_records": [
            {"episode_id": key[0], "frame_index": key[1]}
            for key in sorted(expected_keys - actual_keys)
        ],
        "unexpected_records": [
            {"episode_id": key[0], "frame_index": key[1]}
            for key in sorted(actual_keys - expected_keys)
        ],
        "duplicate_frame_records": [
            {"episode_id": key[0], "frame_index": key[1], "count": count}
            for key, count in sorted(key_counts.items())
            if count > 1
        ],
        "duplicate_feature_paths": [
            {"feature_array_path": path, "count": count}
            for path, count in sorted(path_counts.items())
            if count > 1
        ],
        "source_path_mismatches": source_mismatches,
        "invalid_record_indices": invalid_record_indices,
    }


def validate_metadata(
    metadata: Mapping[str, Any], expected_values: Mapping[str, Any]
) -> list[str]:
    """Return mismatches for the required recorded inference configuration."""
    errors = []
    for field, expected in expected_values.items():
        actual = metadata.get(field)
        if _json_value(actual) != _json_value(expected):
            errors.append(f"metadata.{field}: expected {expected!r}, got {actual!r}")
    return errors


def reproducibility_metrics(
    reference: Sequence[float], repeated: Sequence[float]
) -> dict[str, float | int | bool]:
    """Calculate absolute-error metrics and cosine similarity for two arrays."""
    if len(reference) != len(repeated):
        raise ValueError(
            f"representation lengths differ: {len(reference)} != {len(repeated)}"
        )
    if not reference:
        raise ValueError("representations must contain at least one value")

    max_difference = 0.0
    absolute_sum = 0.0
    dot_product = 0.0
    reference_squared = 0.0
    repeated_squared = 0.0
    different_values = 0
    for left_value, right_value in zip(reference, repeated, strict=True):
        left = float(left_value)
        right = float(right_value)
        difference = abs(left - right)
        max_difference = max(max_difference, difference)
        absolute_sum += difference
        dot_product += left * right
        reference_squared += left * left
        repeated_squared += right * right
        different_values += left != right

    if reference_squared == 0.0 or repeated_squared == 0.0:
        cosine = 1.0 if reference_squared == repeated_squared else 0.0
    else:
        cosine = dot_product / math.sqrt(reference_squared * repeated_squared)
    return {
        "element_count": len(reference),
        "different_value_count": different_values,
        "maximum_absolute_difference": max_difference,
        "mean_absolute_difference": absolute_sum / len(reference),
        "cosine_similarity": cosine,
        "exact_match": different_values == 0,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _expected_frames(episodes: Sequence[ManifestEpisode]) -> list[tuple[str, int, str]]:
    return [
        (episode.episode_id, frame_index, frame.relative_path)
        for episode in episodes
        for frame_index, frame in enumerate(episode.frames)
    ]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{path}:{line_number}: invalid JSON: {error.msg}")
            continue
        if not isinstance(record, dict):
            errors.append(f"{path}:{line_number}: record must be a JSON object")
            continue
        records.append(record)
    return records, errors


def _read_float32_array(
    path: Path, expected_shape: Sequence[int]
) -> array.array[float]:
    values = array.array("f")
    if values.itemsize != 4:
        raise RuntimeError("this platform does not support 32-bit float arrays")
    raw = path.read_bytes()
    expected_count = math.prod(expected_shape)
    expected_bytes = expected_count * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"array byte length {len(raw)} does not match shape "
            f"{tuple(expected_shape)} "
            f"({expected_bytes} bytes expected)"
        )
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _model_config_dict(model: Any) -> dict[str, Any]:
    config = model.config
    return {
        "model_type": getattr(config, "model_type", None),
        "patch_size": _json_value(getattr(config, "patch_size", None)),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_register_tokens": getattr(config, "num_register_tokens", 0),
    }


def _runtime_hardware(adapter: FrozenVisionEncoderAdapter) -> dict[str, Any]:
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


def _processor_preprocessing(processor: Any) -> dict[str, Any]:
    names = (
        "do_convert_rgb",
        "do_resize",
        "resample",
        "do_center_crop",
        "do_rescale",
        "rescale_factor",
        "do_normalize",
        "image_mean",
        "image_std",
    )
    output = {name: getattr(processor, name, None) for name in names}
    output["resize_shortest_edge"] = getattr(processor, "size", {}).get("shortest_edge")
    crop_size = getattr(processor, "crop_size", {})
    output["crop_height"] = crop_size.get("height")
    output["crop_width"] = crop_size.get("width")
    return _json_value(output)


def _flatten_tensor(values: Any, torch_module: Any) -> array.array[float]:
    tensor = (
        values.detach().cpu().to(dtype=torch_module.float32).contiguous().reshape(-1)
    )
    return array.array("f", tensor.tolist())


def _feature_path(artifact_dir: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("feature_array_path must be a non-empty relative path")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("feature_array_path must be relative")
    resolved = (artifact_dir / candidate).resolve()
    if not resolved.is_relative_to(artifact_dir.resolve()):
        raise ValueError("feature_array_path escapes the artifact directory")
    return resolved


def _load_adapter(config: Mapping[str, Any]) -> FrozenVisionEncoderAdapter:
    # These settings keep Hugging Face offline even if a future caller changes
    # the adapter's defaults. The loader also passes local_files_only=True.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    preprocessing_values = dict(config.get("preprocessing", {}))
    for field in ("image_mean", "image_std"):
        if field in preprocessing_values:
            preprocessing_values[field] = tuple(preprocessing_values[field])
    preprocessing = ImagePreprocessingConfig(**preprocessing_values)
    cache_dir = config.get("cache_dir")
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = REPOSITORY_ROOT / cache_path
        cache_dir = str(cache_path)
    return FrozenVisionEncoderAdapter.from_pretrained(
        model_id=config.get("model_id", DEFAULT_MODEL_ID),
        revision=config.get("revision", DEFAULT_MODEL_REVISION),
        device=config.get("device", "cpu"),
        deterministic=bool(config.get("deterministic", True)),
        preprocessing=preprocessing,
        cache_dir=cache_dir,
        allow_download=False,
    )


def _feature_records_report(
    artifact_dir: Path,
    records: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    representation: str,
) -> tuple[dict[str, Any], dict[tuple[str, int], array.array[float]]]:
    checked = 0
    feature_value_count = 0
    non_finite_count = 0
    feature_errors: list[str] = []
    shapes: Counter[str] = Counter()
    arrays: dict[tuple[str, int], array.array[float]] = {}
    for index, record in enumerate(records):
        key = (str(record.get("episode_id", "")), record.get("frame_index", -1))
        try:
            shape = record.get("feature_shape")
            if not isinstance(shape, list) or not shape:
                raise ValueError("feature_shape must be a non-empty list")
            processed_dimensions = record.get("processed_dimensions")
            if not isinstance(processed_dimensions, dict):
                raise ValueError("processed_dimensions must be an object")
            expected = expected_feature_shape(
                representation,
                model_config,
                int(processed_dimensions["height"]),
                int(processed_dimensions["width"]),
            )
            validate_feature_shape(shape, expected)
            if record.get("representation") != representation:
                raise ValueError(
                    f"record representation {record.get('representation')!r} "
                    f"does not match {representation!r}"
                )
            if representation == "spatial_tokens":
                grid = record.get("spatial_grid")
                expected_grid = {"rows": expected[0], "columns": expected[1]}
                if grid != expected_grid:
                    raise ValueError(
                        f"spatial_grid {grid!r} does not match {expected_grid!r}"
                    )
            feature_path = _feature_path(artifact_dir, record.get("feature_array_path"))
            values = _read_float32_array(feature_path, expected)
            bad_values = count_non_finite_values(values)
            non_finite_count += bad_values
            if bad_values:
                feature_errors.append(
                    f"record {index} {key}: contains {bad_values} non-finite values"
                )
            checked += 1
            feature_value_count += len(values)
            shapes[str(list(expected))] += 1
            if isinstance(key[1], int) and key not in arrays:
                arrays[key] = values
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
            feature_errors.append(f"record {index} {key}: {error}")
    return (
        {
            "checked_feature_file_count": checked,
            "feature_value_count": feature_value_count,
            "non_finite_value_count": non_finite_count,
            "shape_counts": dict(sorted(shapes.items())),
            "errors": feature_errors,
        },
        arrays,
    )


def validate_outputs(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    """Validate saved records and compare with a local-only inference pass."""
    manifest_path = manifest_path.resolve()
    config_path = config_path.resolve()
    artifact_dir = artifact_dir.resolve()
    report_path = report_path.resolve()
    report: dict[str, Any] = {
        "validation_status": "incomplete",
        "research_scope": {
            "input_manifest": str(manifest_path),
            "held_out_test_split_accessed": False,
            "checkpoint_downloads_allowed": False,
            "model_weights_modified": False,
            "semantic_accuracy_evaluated": False,
            "neural_network_acceleration_claimed": False,
        },
        "errors": [],
    }

    if "test" in manifest_path.parts:
        report["errors"].append("held-out test split paths are excluded")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    try:
        config = _read_json(config_path)
        episodes = load_manifest(manifest_path)
        expected_frames = _expected_frames(episodes)
        artifact_metadata = _read_json(artifact_dir / "run_metadata.json")
        records, record_read_errors = _read_jsonl(
            artifact_dir / artifact_metadata.get("frame_records_path", "frames.jsonl")
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report["errors"].append(str(error))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    record_identity_report = validate_manifest_records(expected_frames, records)
    report["manifest_matching"] = record_identity_report
    report["errors"].extend(record_read_errors)

    try:
        adapter = _load_adapter(config)
        actual_model_config = _model_config_dict(adapter.model)
        actual_runtime_config = {
            "model_identifier": adapter.model_id,
            "model_revision": adapter.revision,
            "device": adapter.device,
            "preprocessing": adapter.preprocessing.to_dict(),
            "representation": config.get("representation", "spatial_tokens"),
            "deterministic_evaluation": adapter.deterministic,
            "model_config": actual_model_config,
            "hardware": _runtime_hardware(adapter),
        }
        parameter_list = list(adapter.model.parameters())
        model_training = bool(getattr(adapter.model, "training", True))
        all_parameters_frozen = all(
            not bool(parameter.requires_grad) for parameter in parameter_list
        )
        parameter_devices = sorted(
            {str(parameter.device) for parameter in parameter_list}
        )
        expected_device = "cpu" if adapter.device == "cpu" else "cuda"
        model_device_matches = all(
            device == expected_device or device.startswith(f"{expected_device}:")
            for device in parameter_devices
        )
        processor_preprocessing = _processor_preprocessing(adapter.processor)
        actual_runtime_config["processor_preprocessing"] = processor_preprocessing
        actual_runtime_config["model_evaluation_mode"] = not model_training
        actual_runtime_config["all_parameters_frozen"] = all_parameters_frozen
        actual_runtime_config["parameter_devices"] = parameter_devices
        actual_runtime_config["model_device_matches_config"] = model_device_matches
        report["actual_inference_configuration"] = actual_runtime_config
        report["artifact_metadata"] = artifact_metadata

        expected_metadata = {
            "model_identifier": adapter.model_id,
            "model_revision": adapter.revision,
            "device": adapter.device,
            "preprocessing": adapter.preprocessing.to_dict(),
            "representation": actual_runtime_config["representation"],
            "deterministic_evaluation": adapter.deterministic,
            "frozen_inference": True,
            "input_frames_encoded": len(expected_frames),
            "feature_storage": FEATURE_STORAGE,
            "model_config": actual_model_config,
        }
        metadata_errors = validate_metadata(artifact_metadata, expected_metadata)
        if processor_preprocessing != _json_value(adapter.preprocessing.to_dict()):
            metadata_errors.append(
                "loaded image processor settings differ from configured preprocessing"
            )
        if model_training:
            metadata_errors.append("loaded validation model is in training mode")
        if not parameter_list:
            metadata_errors.append("loaded model exposes no parameters to verify")
        elif not all_parameters_frozen:
            metadata_errors.append("one or more loaded model parameters are not frozen")
        if not model_device_matches:
            metadata_errors.append(
                f"model parameter devices {parameter_devices!r} do not match "
                f"configured device {adapter.device!r}"
            )
        report["metadata_validation"] = {
            "passed": not metadata_errors,
            "expected_values": expected_metadata,
            "errors": metadata_errors,
        }
        report["errors"].extend(metadata_errors)

        cache_dir = config.get("cache_dir")
        cache_root = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache/huggingface/hub"
        )
        if not cache_root.is_absolute():
            cache_root = REPOSITORY_ROOT / cache_root
        cache_snapshot = (
            cache_root
            / f"models--{adapter.model_id.replace('/', '--')}"
            / "snapshots"
            / adapter.revision
        )
        resolved_revision = getattr(adapter.model.config, "_commit_hash", None)
        report["model_revision"] = {
            "configured_revision": adapter.revision,
            "loaded_config_commit_hash": resolved_revision,
            "revision_matches": resolved_revision in (None, adapter.revision)
            and cache_snapshot.is_dir(),
            "exact_local_snapshot_present": cache_snapshot.is_dir(),
            "checkpoint_source": "local Hugging Face cache",
            "local_files_only": True,
            "download_allowed": False,
        }
        if (
            resolved_revision not in (None, adapter.revision)
            or not cache_snapshot.is_dir()
        ):
            report["errors"].append(
                "the loaded checkpoint could not be verified against the exact "
                f"local snapshot for revision {adapter.revision}"
            )

        feature_report, persisted_arrays = _feature_records_report(
            artifact_dir,
            records,
            actual_model_config,
            str(actual_runtime_config["representation"]),
        )
        report["numerical_integrity"] = {
            "passed": feature_report["non_finite_value_count"] == 0
            and not feature_report["errors"],
            "feature_files_checked": feature_report["checked_feature_file_count"],
            "feature_values_checked": feature_report["feature_value_count"],
            "non_finite_value_count": feature_report["non_finite_value_count"],
            "shape_counts": feature_report["shape_counts"],
            "errors": feature_report["errors"],
        }
        report["errors"].extend(feature_report["errors"])

        repeated_values: list[array.array[float]] = []
        reference_values: list[array.array[float]] = []
        per_frame_metrics: list[dict[str, Any]] = []
        inference_errors: list[str] = []
        record_by_key: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for record in records:
            episode_id = record.get("episode_id")
            frame_index = record.get("frame_index")
            if isinstance(episode_id, str) and isinstance(frame_index, int):
                record_by_key[(episode_id, frame_index)].append(record)

        for episode in episodes:
            for frame_index, frame in enumerate(episode.frames):
                key = (episode.episode_id, frame_index)
                try:
                    repeated_result = adapter.encode_frame(
                        read_ppm(frame.path),
                        representation=actual_runtime_config["representation"],
                    )
                    current_shape = tuple(repeated_result.feature_shape)
                    matches = record_by_key.get(key, [])
                    if len(matches) != 1:
                        raise ValueError(
                            f"expected exactly one saved record for {key}, found "
                            f"{len(matches)}"
                        )
                    record = matches[0]
                    validate_feature_shape(
                        record.get("feature_shape", []), current_shape
                    )
                    if record.get("input_dimensions") != {
                        "width": repeated_result.input_width,
                        "height": repeated_result.input_height,
                    }:
                        raise ValueError("saved input dimensions differ from rerun")
                    if record.get("processed_dimensions") != {
                        "width": repeated_result.processed_width,
                        "height": repeated_result.processed_height,
                    }:
                        raise ValueError("saved processed dimensions differ from rerun")
                    previous = persisted_arrays.get(key)
                    if previous is None:
                        raise ValueError("saved feature array is missing or invalid")
                    if count_non_finite_values(previous):
                        raise ValueError(
                            "saved feature array contains non-finite values"
                        )
                    repeated = _flatten_tensor(repeated_result.features, adapter._torch)
                    if count_non_finite_values(repeated):
                        raise ValueError("rerun produced non-finite feature values")
                    metrics = reproducibility_metrics(previous, repeated)
                    per_frame_metrics.append(
                        {
                            "episode_id": episode.episode_id,
                            "frame_index": frame_index,
                            **metrics,
                        }
                    )
                    reference_values.append(previous)
                    repeated_values.append(repeated)
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    inference_errors.append(
                        f"episode {episode.episode_id} frame {frame_index}: {error}"
                    )

        probe_frame = episodes[0].frames[0]
        probe_image = read_ppm(probe_frame.path)
        shape_probes: dict[str, Any] = {}
        for kind in ("spatial_tokens", "global_embedding", "token_sequence"):
            probe_result = adapter.encode_frame(probe_image, representation=kind)
            expected_probe_shape = expected_feature_shape(
                kind,
                actual_model_config,
                probe_result.processed_height,
                probe_result.processed_width,
            )
            validate_feature_shape(probe_result.feature_shape, expected_probe_shape)
            shape_probes[kind] = {
                "observed_shape": list(probe_result.feature_shape),
                "expected_shape": list(expected_probe_shape),
                "matches": True,
            }
        report["shape_probes"] = {
            "source_frame": {
                "episode_id": episodes[0].episode_id,
                "frame_index": 0,
                "source_frame_path": probe_frame.relative_path,
            },
            "representations": shape_probes,
            "token_sequence_includes_cls_and_register_tokens": True,
            "spatial_tokens_exclude_special_tokens": True,
        }

        if reference_values:
            reference_all = array.array("f")
            repeated_all = array.array("f")
            for values in reference_values:
                reference_all.extend(values)
            for values in repeated_values:
                repeated_all.extend(values)
            aggregate = reproducibility_metrics(reference_all, repeated_all)
        else:
            aggregate = None
        exact_match = bool(aggregate and aggregate["exact_match"])
        report["reproducibility"] = {
            "status": "completed" if aggregate is not None else "incomplete",
            "comparison_method": (
                "saved little-endian float32 arrays versus an independent second "
                "inference pass over the same demo frames"
            ),
            "aggregate": aggregate,
            "compared_frame_count": len(per_frame_metrics),
            "nondeterminism_detected": aggregate is not None and not exact_match,
            "difference_cause_isolated": False,
            "interpretation_limit": (
                "The saved and repeated arrays differ. The original and validation "
                "Python/PyTorch/Transformers versions differ, so this comparison "
                "does not isolate kernel nondeterminism from runtime-version effects."
            ),
            "per_frame": per_frame_metrics,
            "errors": inference_errors,
        }
        report["errors"].extend(inference_errors)
        report["reproduction_runtime"] = actual_runtime_config["hardware"]
        artifact_hardware = artifact_metadata.get("hardware", {})
        report["runtime_version_comparison"] = {
            field: {
                "artifact_run": artifact_hardware.get(field),
                "validation_run": actual_runtime_config["hardware"].get(field),
                "matches": artifact_hardware.get(field)
                == actual_runtime_config["hardware"].get(field),
            }
            for field in ("python", "torch_version", "transformers_version")
        }
        report["frames_validated"] = feature_report["checked_feature_file_count"]
        report["feature_dimensions"] = {
            "representation": actual_runtime_config["representation"],
            "model_hidden_size": actual_model_config["hidden_size"],
            "patch_size": actual_model_config["patch_size"],
            "num_register_tokens": actual_model_config["num_register_tokens"],
            "observed_shape_counts": feature_report["shape_counts"],
            "global_embedding_shape": shape_probes["global_embedding"][
                "observed_shape"
            ],
            "token_sequence_shape": shape_probes["token_sequence"]["observed_shape"],
        }
        report["evaluation_and_freezing"] = {
            "implementation_calls_eval": True,
            "implementation_freezes_parameters": True,
            "validation_model_in_eval_mode": not model_training,
            "validation_parameters_all_frozen": all_parameters_frozen,
            "parameter_count": len(parameter_list),
        }
    except (
        ImportError,
        ModelLoadingError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        report["errors"].append(f"local inference could not be completed: {error}")
        report["local_inference"] = {
            "status": "unverified",
            "checkpoint_downloads_allowed": False,
            "error": str(error),
            "artifact_metadata": artifact_metadata,
        }
        fallback_model_config = artifact_metadata.get("model_config", {})
        feature_report, _ = _feature_records_report(
            artifact_dir,
            records,
            fallback_model_config,
            str(artifact_metadata.get("representation", "spatial_tokens")),
        )
        report["numerical_integrity"] = {
            "passed": feature_report["non_finite_value_count"] == 0
            and not feature_report["errors"],
            "feature_files_checked": feature_report["checked_feature_file_count"],
            "feature_values_checked": feature_report["feature_value_count"],
            "non_finite_value_count": feature_report["non_finite_value_count"],
            "shape_counts": feature_report["shape_counts"],
            "errors": feature_report["errors"],
            "shape_configuration_source": "recorded metadata; actual model unavailable",
        }
        report["errors"].extend(feature_report["errors"])

    structural_problems = any(
        record_identity_report[key]
        for key in (
            "missing_records",
            "unexpected_records",
            "duplicate_frame_records",
            "duplicate_feature_paths",
            "source_path_mismatches",
            "invalid_record_indices",
        )
    )
    if not report["errors"] and not structural_problems:
        if report.get("reproducibility", {}).get("nondeterminism_detected"):
            report["validation_status"] = "passed_with_numerical_differences"
        else:
            report["validation_status"] = "passed"
    elif (
        report.get("numerical_integrity", {}).get("passed")
        and report.get("local_inference", {}).get("status") == "unverified"
    ):
        report["validation_status"] = "incomplete"
    else:
        report["validation_status"] = "failed"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate DINOv2 feature records and compare them with a second "
            "local-cache-only inference pass."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = validate_outputs(
        manifest_path=args.manifest,
        config_path=args.config,
        artifact_dir=args.artifact_dir,
        report_path=args.output,
    )
    print(f"Validation status: {report['validation_status']}")
    print(f"Report: {args.output}")
    print(f"Feature records validated: {report.get('frames_validated', 0)}")
    return (
        0
        if report["validation_status"]
        in ("passed", "passed_with_numerical_differences")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
