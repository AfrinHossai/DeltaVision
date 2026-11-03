"""Exact whole-frame reuse for frozen pretrained vision representations."""

from __future__ import annotations

import array
import copy
import hashlib
import json
import math
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .baseline import load_manifest
from .encoder import read_ppm
from .pretrained import (
    FrozenVisionEncoderAdapter,
    PreparedVisionInput,
    RepresentationKind,
    VisionEncoding,
)
from .pretrained_baseline import (
    _flatten_values,
    _frame_record,
    _run_metadata,
    _write_float32_array,
    _write_jsonl_record,
)

HashFunction = Callable[[Mapping[str, Any]], str]
CacheKey = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class StatefulFrameEncoding:
    """An encoded frame plus the cache decision and measured overheads."""

    encoding: VisionEncoding
    cache_hit: bool
    content_hash: str
    preprocessing_seconds: float
    cache_lookup_seconds: float
    cache_retrieval_seconds: float
    cache_storage_seconds: float


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    representation: RepresentationKind
    content_hash: str
    model_inputs: Mapping[str, Any]
    encoding: VisionEncoding
    tensor_payload_bytes: int


def _tensor_bytes(value: Any, torch_module: Any) -> bytes:
    tensor = value.detach().cpu().contiguous()
    try:
        return tensor.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        # NumPy cannot expose every PyTorch dtype directly (for example bfloat16).
        return tensor.view(torch_module.uint8).numpy().tobytes(order="C")


def _canonical_value_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "preprocessed model inputs must contain tensors or canonical "
            "JSON-compatible values"
        ) from error
    return encoded.encode("utf-8")


def stable_model_input_hash(model_inputs: Mapping[str, Any], torch_module: Any) -> str:
    """Hash all preprocessed model inputs, including names, shapes, and dtypes."""
    digest = hashlib.sha256()
    for key in sorted(model_inputs):
        value = model_inputs[key]
        key_bytes = key.encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "little"))
        digest.update(key_bytes)
        if hasattr(value, "shape") and hasattr(value, "detach"):
            header = {
                "kind": "tensor",
                "dtype": str(value.dtype),
                "shape": [int(dimension) for dimension in value.shape],
            }
            payload = _tensor_bytes(value, torch_module)
        else:
            header = {"kind": type(value).__qualname__}
            payload = _canonical_value_bytes(value)
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        digest.update(len(header_bytes).to_bytes(8, "little"))
        digest.update(header_bytes)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _clone_input_value(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().cpu().contiguous().clone()
    return copy.deepcopy(value)


def _clone_model_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _clone_input_value(value) for key, value in inputs.items()}


def _inputs_equal(
    cached: Mapping[str, Any], current: Mapping[str, Any], torch_module: Any
) -> bool:
    if cached.keys() != current.keys():
        return False
    for key in cached:
        left = cached[key]
        right = current[key]
        left_tensor = hasattr(left, "shape") and hasattr(left, "detach")
        right_tensor = hasattr(right, "shape") and hasattr(right, "detach")
        if left_tensor != right_tensor:
            return False
        if left_tensor:
            if str(left.dtype) != str(right.dtype) or tuple(left.shape) != tuple(
                right.shape
            ):
                return False
            if not bool(torch_module.equal(left, right)) or _tensor_bytes(
                left, torch_module
            ) != _tensor_bytes(right, torch_module):
                return False
        elif type(left) is not type(right) or _canonical_value_bytes(
            left
        ) != _canonical_value_bytes(right):
            return False
    return True


def _tensor_payload_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel()) * int(value.element_size())
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        values = array.array("f", _flatten_values(value))
        return len(values) * values.itemsize
    return 0


class ExactWholeFrameCache:
    """Per-episode LRU cache keyed by a hash and verified tensor equality.

    The cache stores complete preprocessed model inputs and complete selected
    model representations. It never slices, recomputes, or approximates tokens.
    """

    def __init__(
        self,
        adapter: FrozenVisionEncoderAdapter,
        *,
        capacity: int = 8,
        hash_function: HashFunction | None = None,
    ) -> None:
        if capacity < 0:
            raise ValueError("cache capacity must be non-negative")
        self.adapter = adapter
        self.capacity = capacity
        self._hash_function = hash_function or (
            lambda inputs: stable_model_input_hash(inputs, adapter._torch)
        )
        self._entries: OrderedDict[CacheKey, _CacheEntry] = OrderedDict()
        self._cache_payload_bytes = 0
        self.peak_cache_payload_bytes = 0
        self.total_frames = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.actual_model_forward_calls = 0
        self.cache_evictions = 0
        self.preprocessing_seconds = 0.0
        self.cache_lookup_overhead_seconds = 0.0
        self.cache_retrieval_overhead_seconds = 0.0
        self.cache_storage_overhead_seconds = 0.0
        self.model_inference_time_seconds = 0.0
        self.episode_resets = 0

    @property
    def current_cache_entries(self) -> int:
        return len(self._entries)

    @property
    def current_cache_payload_bytes(self) -> int:
        return self._cache_payload_bytes

    def reset_episode(self) -> None:
        """Clear all cache entries at an episode boundary."""
        self._entries.clear()
        self._cache_payload_bytes = 0
        self.episode_resets += 1

    def encode_frame(
        self,
        image: Any,
        *,
        representation: RepresentationKind = "spatial_tokens",
    ) -> StatefulFrameEncoding:
        """Preprocess, exact-match, then reuse or run one complete model forward."""
        preprocessing_started = time.perf_counter()
        prepared = self.adapter.prepare_frame(image)
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        self.preprocessing_seconds += preprocessing_seconds

        lookup_started = time.perf_counter()
        content_hash = self._hash_function(prepared.model_inputs)
        cache_key = self._find_exact_match(
            representation, content_hash, prepared.model_inputs
        )
        cache_lookup_seconds = time.perf_counter() - lookup_started
        self.cache_lookup_overhead_seconds += cache_lookup_seconds
        self.total_frames += 1

        if cache_key is not None:
            self._entries.move_to_end(cache_key)
            entry = self._entries[cache_key]
            self.cache_hits += 1
            retrieval_started = time.perf_counter()
            encoding = self._encoding_for_current_frame(prepared, entry)
            cache_retrieval_seconds = time.perf_counter() - retrieval_started
            self.cache_retrieval_overhead_seconds += cache_retrieval_seconds
            return StatefulFrameEncoding(
                encoding=encoding,
                cache_hit=True,
                content_hash=content_hash,
                preprocessing_seconds=preprocessing_seconds,
                cache_lookup_seconds=cache_lookup_seconds,
                cache_retrieval_seconds=cache_retrieval_seconds,
                cache_storage_seconds=0.0,
            )

        self.cache_misses += 1
        encoding = self.adapter.encode_preprocessed(
            prepared, representation=representation
        )
        self.actual_model_forward_calls += 1
        self.model_inference_time_seconds += encoding.model_forward_latency_seconds
        cache_storage_seconds = self._store(
            representation, content_hash, prepared, encoding
        )
        self.cache_storage_overhead_seconds += cache_storage_seconds
        return StatefulFrameEncoding(
            encoding=encoding,
            cache_hit=False,
            content_hash=content_hash,
            preprocessing_seconds=preprocessing_seconds,
            cache_lookup_seconds=cache_lookup_seconds,
            cache_retrieval_seconds=0.0,
            cache_storage_seconds=cache_storage_seconds,
        )

    def _find_exact_match(
        self,
        representation: RepresentationKind,
        content_hash: str,
        model_inputs: Mapping[str, Any],
    ) -> CacheKey | None:
        for key, entry in reversed(self._entries.items()):
            if (
                entry.representation == representation
                and entry.content_hash == content_hash
                and _inputs_equal(entry.model_inputs, model_inputs, self.adapter._torch)
            ):
                return key
        return None

    def _store(
        self,
        representation: RepresentationKind,
        content_hash: str,
        prepared: PreparedVisionInput,
        encoding: VisionEncoding,
    ) -> float:
        if self.capacity == 0:
            return 0.0
        started = time.perf_counter()
        collision_index = 1 + max(
            (
                key[2]
                for key, entry in self._entries.items()
                if entry.representation == representation
                and entry.content_hash == content_hash
            ),
            default=-1,
        )
        cache_key = (representation, content_hash, collision_index)
        stored_inputs = _clone_model_inputs(prepared.model_inputs)
        stored_features = _clone_input_value(encoding.features)
        stored_encoding = VisionEncoding(
            features=stored_features,
            representation=encoding.representation,
            input_width=encoding.input_width,
            input_height=encoding.input_height,
            processed_width=encoding.processed_width,
            processed_height=encoding.processed_height,
            feature_shape=encoding.feature_shape,
            spatial_grid=encoding.spatial_grid,
            inference_latency_seconds=encoding.inference_latency_seconds,
            model_forward_latency_seconds=encoding.model_forward_latency_seconds,
        )
        payload_bytes = sum(
            _tensor_payload_bytes(value) for value in stored_inputs.values()
        ) + _tensor_payload_bytes(stored_features)
        if len(self._entries) >= self.capacity:
            _, evicted = self._entries.popitem(last=False)
            self._cache_payload_bytes -= evicted.tensor_payload_bytes
            self.cache_evictions += 1
        self._entries[cache_key] = _CacheEntry(
            representation=representation,
            content_hash=content_hash,
            model_inputs=stored_inputs,
            encoding=stored_encoding,
            tensor_payload_bytes=payload_bytes,
        )
        self._cache_payload_bytes += payload_bytes
        self.peak_cache_payload_bytes = max(
            self.peak_cache_payload_bytes, self._cache_payload_bytes
        )
        return time.perf_counter() - started

    @staticmethod
    def _encoding_for_current_frame(
        prepared: PreparedVisionInput, entry: _CacheEntry
    ) -> VisionEncoding:
        return VisionEncoding(
            features=_clone_input_value(entry.encoding.features),
            representation=entry.representation,
            input_width=prepared.input_width,
            input_height=prepared.input_height,
            processed_width=prepared.processed_width,
            processed_height=prepared.processed_height,
            feature_shape=entry.encoding.feature_shape,
            spatial_grid=entry.encoding.spatial_grid,
            inference_latency_seconds=0.0,
            model_forward_latency_seconds=0.0,
        )

    def metrics(self, *, total_processing_time_seconds: float) -> dict[str, Any]:
        hit_rate = self.cache_hits / self.total_frames if self.total_frames else 0.0
        return {
            "total_frames": self.total_frames,
            "actual_dinov2_forward_calls": self.actual_model_forward_calls,
            "cached_frame_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "model_inference_time_seconds": self.model_inference_time_seconds,
            "preprocessing_time_seconds": self.preprocessing_seconds,
            "cache_lookup_overhead_seconds": self.cache_lookup_overhead_seconds,
            "cache_retrieval_overhead_seconds": (self.cache_retrieval_overhead_seconds),
            "cache_storage_overhead_seconds": self.cache_storage_overhead_seconds,
            "total_processing_time_seconds": total_processing_time_seconds,
            "cache_capacity": self.capacity,
            "cache_evictions": self.cache_evictions,
            "episode_cache_resets": self.episode_resets,
            "peak_cache_memory_estimate_bytes": self.peak_cache_payload_bytes,
            "peak_cache_memory_estimate_mib": self.peak_cache_payload_bytes / (1024**2),
            "memory_estimate_scope": (
                "Tensor payload bytes for complete preprocessed model inputs and "
                "cached selected representations resident in the cache; excludes "
                "in-flight tensors and Python, hash, and map object overhead."
            ),
            "uncached_independent_frame_forward_calls": self.total_frames,
            "full_model_forward_calls_avoided": self.cache_hits,
            "full_model_forward_call_reduction_fraction": hit_rate,
            "forward_calls_equal_cache_misses": (
                self.actual_model_forward_calls == self.cache_misses
            ),
            "end_to_end_latency_is_speedup_claim": False,
        }


def _reproducibility_metrics(
    reference: Sequence[float], repeated: Sequence[float]
) -> dict[str, Any]:
    if len(reference) != len(repeated):
        raise ValueError("reference and repeated arrays have different lengths")
    if not reference:
        raise ValueError("representations must be non-empty")
    absolute_sum = 0.0
    maximum = 0.0
    dot = 0.0
    reference_squared = 0.0
    repeated_squared = 0.0
    different = 0
    for left_value, right_value in zip(reference, repeated, strict=True):
        left = float(left_value)
        right = float(right_value)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("representation comparison contains non-finite values")
        difference = abs(left - right)
        maximum = max(maximum, difference)
        absolute_sum += difference
        dot += left * right
        reference_squared += left * left
        repeated_squared += right * right
        different += left != right
    cosine = (
        1.0
        if reference_squared == repeated_squared == 0.0
        else (
            0.0
            if reference_squared == 0.0 or repeated_squared == 0.0
            else dot / math.sqrt(reference_squared * repeated_squared)
        )
    )
    return {
        "element_count": len(reference),
        "different_value_count": different,
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": absolute_sum / len(reference),
        "cosine_similarity": cosine,
        "exact_match": different == 0,
    }


def _read_little_endian_f32(path: Path) -> array.array[float]:
    values = array.array("f")
    if values.itemsize != 4:
        raise RuntimeError("this platform does not support 32-bit float arrays")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _compare_with_independent_inference(
    manifest_path: Path,
    artifact_dir: Path,
    adapter: FrozenVisionEncoderAdapter,
    representation: RepresentationKind,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    episodes = load_manifest(manifest_path)
    by_identity = {
        (record["episode_id"], record["frame_index"]): record for record in records
    }
    all_saved = array.array("f")
    all_independent = array.array("f")
    per_frame = []
    started = time.perf_counter()
    independent_calls = 0
    for episode in episodes:
        for frame_index, frame in enumerate(episode.frames):
            record = by_identity[(episode.episode_id, frame_index)]
            saved = _read_little_endian_f32(artifact_dir / record["feature_array_path"])
            independent = adapter.encode_frame(
                read_ppm(frame.path), representation=representation
            )
            independent_values = array.array("f", _flatten_values(independent.features))
            metrics = _reproducibility_metrics(saved, independent_values)
            per_frame.append(
                {
                    "episode_id": episode.episode_id,
                    "frame_index": frame_index,
                    **metrics,
                }
            )
            all_saved.extend(saved)
            all_independent.extend(independent_values)
            independent_calls += 1
    elapsed = time.perf_counter() - started
    return {
        "status": "completed",
        "comparison": (
            "whole-frame cached outputs versus independent full-frame inference"
        ),
        "comparison_forward_calls": independent_calls,
        "comparison_time_seconds_excluded_from_stateful_total": elapsed,
        "aggregate": _reproducibility_metrics(all_saved, all_independent),
        "per_frame": per_frame,
    }


def run_pretrained_stateful(
    manifest_path: str | Path,
    output_dir: str | Path,
    adapter: FrozenVisionEncoderAdapter,
    *,
    representation: RepresentationKind = "spatial_tokens",
    cache_capacity: int = 8,
    hash_function: HashFunction | None = None,
) -> Path:
    """Run exact whole-frame caching and compare outputs to independent inference."""
    manifest = Path(manifest_path).resolve()
    if "test" in manifest.parts:
        raise ValueError("held-out test split paths are excluded from this runner")
    episodes = load_manifest(manifest)
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
    record_path = destination / "frames.jsonl"
    cache = ExactWholeFrameCache(
        adapter,
        capacity=cache_capacity,
        hash_function=hash_function,
    )
    total_frame_count = sum(len(episode.frames) for episode in episodes)
    frame_index_global = 0
    records: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    with record_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for episode in episodes:
            cache.reset_episode()
            for frame_index, frame in enumerate(episode.frames):
                image = read_ppm(frame.path)
                stateful_result = cache.encode_frame(
                    image,
                    representation=representation,
                )
                feature_relative_path = f"features/frame_{frame_index_global:08d}.f32le"
                _write_float32_array(
                    destination / feature_relative_path,
                    stateful_result.encoding,
                )
                record = _frame_record(
                    episode.episode_id,
                    frame_index,
                    frame.relative_path,
                    stateful_result.encoding,
                    feature_relative_path,
                )
                record.update(
                    {
                        "cache_hit": stateful_result.cache_hit,
                        "preprocessed_input_hash": stateful_result.content_hash,
                        "cache_lookup_overhead_seconds": (
                            stateful_result.cache_lookup_seconds
                        ),
                        "cache_retrieval_overhead_seconds": (
                            stateful_result.cache_retrieval_seconds
                        ),
                        "cache_storage_overhead_seconds": (
                            stateful_result.cache_storage_seconds
                        ),
                        "preprocessing_seconds": (
                            stateful_result.preprocessing_seconds
                        ),
                        "model_forward_latency_seconds": (
                            stateful_result.encoding.model_forward_latency_seconds
                        ),
                    }
                )
                _write_jsonl_record(output_file, record)
                records.append(record)
                frame_index_global += 1

    total_processing_seconds = time.perf_counter() - run_started
    cache_metrics = cache.metrics(
        total_processing_time_seconds=total_processing_seconds
    )
    if cache.total_frames != total_frame_count:
        raise RuntimeError(
            f"stateful runner processed {cache.total_frames} frames, "
            f"expected {total_frame_count}"
        )

    metadata = _run_metadata(adapter, representation, total_frame_count)
    metadata["temporal_state_or_cache"] = True
    metadata["frame_records_path"] = "frames.jsonl"
    metadata["feature_storage"] = {
        "format": "raw-float-array",
        "dtype": "little-endian float32",
        "order": "row-major",
        "one_file_per_frame": True,
    }
    metadata["cache_design"] = {
        "type": "exact whole-frame reuse",
        "key": "SHA-256 of all preprocessed model inputs plus exact tensor equality",
        "capacity_scope": "per episode",
        "eviction_policy": "least-recently-used with deterministic tie handling",
        "cached_object": "complete selected model representation",
        "partial_token_computation": False,
        "approximate_reuse": False,
    }
    metadata["cache_capacity_per_episode"] = cache_capacity
    metadata["timing_scope"] = (
        "Stateful total includes PPM reads, preprocessing, hash and exact comparison, "
        "model forwards, cache storage, feature writes, and frame-record writes; "
        "excludes model loading, interpreter startup, summary JSON writing, and the "
        "independent-inference equivalence pass. Model inference time measures only "
        "the synchronized full-model forward calls."
    )

    comparison = _compare_with_independent_inference(
        manifest,
        destination,
        adapter,
        representation,
        records,
    )
    report = {
        "model_identifier": adapter.model_id,
        "model_revision": adapter.revision,
        "device": adapter.device,
        "representation": representation,
        "cache_metrics": cache_metrics,
        "numerical_differences_against_independent_inference": comparison,
        "independent_comparison_forward_calls_separately_reported": True,
        "frame_count_matches_manifest": cache.total_frames == total_frame_count,
        "model_calls_match_cache_misses": cache_metrics[
            "forward_calls_equal_cache_misses"
        ],
        "hardware": metadata["hardware"],
        "model_config": metadata["model_config"],
        "preprocessing": metadata["preprocessing"],
        "research_limits": [
            "A cache hit is not an end-to-end speedup claim.",
            "No individual DINOv2 tokens are independently recomputed.",
            "The equivalence pass measures numerical agreement, not semantic accuracy.",
            "Results are limited to the supplied synthetic episode manifest.",
        ],
    }
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "stateful_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
