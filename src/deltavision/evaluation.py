"""Patch-level change-detection evaluation against synthetic pixel masks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

from .approximate import (
    SUPPORTED_THRESHOLDS,
    ThresholdPatchCache,
    compare_feature_vectors,
)
from .baseline import ManifestEpisode, load_manifest
from .encoder import FEATURE_DEFINITION, RGBImage, encode_image, read_ppm
from .plotting import write_change_detection_png

EXPERIMENT_LABEL = "Synthetic-benchmark findings"
GROUND_TRUTH_DEFINITION = (
    "A patch is ground-truth changed when its PGM mask region contains at least "
    "one non-zero pixel-change value. Masks label rendered RGB pixel changes, "
    "not semantic changes."
)
PREDICTION_DEFINITION = (
    "A patch is predicted refreshed when the existing ThresholdPatchCache gate "
    "recomputes it: normalized mean absolute RGB difference > threshold."
)
METRIC_DEFINITIONS = {
    "precision": "TP / (TP + FP), or 0 when the denominator is 0.",
    "recall": "TP / (TP + FN), or 0 when there are no positive labels.",
    "f1": "2 * precision * recall / (precision + recall), or 0 when undefined.",
    "missed_change_rate": "FN / (TP + FN), or 0 when there are no positive labels.",
    "false_refresh_rate": "FP / (FP + TN), or 0 when there are no negative labels.",
}

FeatureVectors = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Patch-level binary classification counts."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total(self) -> int:
        """Total classified transition patches."""
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )


@dataclass(frozen=True, slots=True)
class _PreparedEpisode:
    episode_id: str
    frames: tuple[RGBImage, ...]
    baseline_features: tuple[FeatureVectors, ...]
    ground_truth_transitions: tuple[tuple[bool, ...], ...]


def ground_truth_mask_to_patch_labels(
    mask_pixels: bytes, width: int, height: int, patch_size: int
) -> tuple[bool, ...]:
    """Convert a binary pixel mask to row-major changed-patch labels.

    A patch label is true when at least one non-zero mask pixel lies inside its
    in-bounds region. Partial edge patches include only real image pixels.
    """
    if width < 1 or height < 1:
        raise ValueError("mask width and height must be positive")
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if len(mask_pixels) != width * height:
        raise ValueError("mask byte count does not match its dimensions")

    grid_columns = (width + patch_size - 1) // patch_size
    grid_rows = (height + patch_size - 1) // patch_size
    labels: list[bool] = []
    for patch_row in range(grid_rows):
        y_start = patch_row * patch_size
        y_stop = min(y_start + patch_size, height)
        for patch_column in range(grid_columns):
            x_start = patch_column * patch_size
            x_stop = min(x_start + patch_size, width)
            labels.append(
                any(
                    mask_pixels[y * width + x] > 0
                    for y in range(y_start, y_stop)
                    for x in range(x_start, x_stop)
                )
            )
    return tuple(labels)


def count_patch_decisions(
    ground_truth_changed: Sequence[bool], predicted_refresh: Sequence[bool]
) -> ConfusionCounts:
    """Count patch-level TP, FP, TN, and FN for one or more transitions."""
    if len(ground_truth_changed) != len(predicted_refresh):
        raise ValueError("ground-truth and predicted patch counts must match")
    true_positives = false_positives = true_negatives = false_negatives = 0
    for actual, predicted in zip(ground_truth_changed, predicted_refresh, strict=True):
        if actual and predicted:
            true_positives += 1
        elif not actual and predicted:
            false_positives += 1
        elif not actual and not predicted:
            true_negatives += 1
        else:
            false_negatives += 1
    return ConfusionCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )


def detection_metrics(counts: ConfusionCounts) -> dict[str, float | int]:
    """Calculate detection metrics with zero returned for undefined rates.

    Precision is 0 when there are no predicted positives; recall and missed
    change rate are 0 when there are no ground-truth positives; false-refresh
    rate is 0 when there are no ground-truth negatives; and F1 is 0 when its
    denominator is 0.
    """
    predicted_positive_count = counts.true_positives + counts.false_positives
    positive_count = counts.true_positives + counts.false_negatives
    negative_count = counts.false_positives + counts.true_negatives
    precision = (
        counts.true_positives / predicted_positive_count
        if predicted_positive_count
        else 0.0
    )
    recall = counts.true_positives / positive_count if positive_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    missed_change_rate = (
        counts.false_negatives / positive_count if positive_count else 0.0
    )
    false_refresh_rate = (
        counts.false_positives / negative_count if negative_count else 0.0
    )
    return {
        "true_positives": counts.true_positives,
        "false_positives": counts.false_positives,
        "true_negatives": counts.true_negatives,
        "false_negatives": counts.false_negatives,
        "classified_transition_patches": counts.total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed_change_rate": missed_change_rate,
        "false_refresh_rate": false_refresh_rate,
    }


def evaluate_change_detection(
    manifest_path: str | Path,
    output_path: str | Path = "results/change_detection.json",
    figure_path: str | Path = "docs/figures/change_detection_tradeoff.png",
    *,
    patch_size: int = 16,
    thresholds: Sequence[float] = SUPPORTED_THRESHOLDS,
) -> dict[str, object]:
    """Evaluate the actual threshold gate against manifest pixel masks.

    The episode's first frame is encoded to initialize each threshold cache,
    but is excluded from confusion-matrix counts because it has no preceding
    transition. Feature approximation metrics include all frames and compare
    cached features with independently computed baseline features.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    selected_thresholds = tuple(float(value) for value in thresholds)
    if not selected_thresholds or len(set(selected_thresholds)) != len(
        selected_thresholds
    ):
        raise ValueError("thresholds must be a non-empty sequence without duplicates")
    for threshold in selected_thresholds:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite value between 0 and 1")

    episodes = load_manifest(manifest_path)
    manifest_rows = _read_manifest_rows(manifest_path)
    if len(episodes) != len(manifest_rows):
        raise ValueError("manifest episode records changed while loading")

    output = Path(output_path)
    figure = Path(figure_path)
    input_path = Path(manifest_path).resolve()
    resolved_output = output.resolve()
    resolved_figure = figure.resolve()
    if resolved_output == input_path or resolved_figure == input_path:
        raise ValueError("output paths must not overwrite the input manifest")
    source_frames = {frame.path for episode in episodes for frame in episode.frames}
    if resolved_output in source_frames or resolved_figure in source_frames:
        raise ValueError("output paths must not overwrite input frame images")
    if resolved_output == resolved_figure:
        raise ValueError("JSON output and figure paths must be different")

    dataset_root = Path(manifest_path).parent.resolve()
    prepared_episodes = tuple(
        _prepare_episode(dataset_root, episode, row, patch_size)
        for episode, row in zip(episodes, manifest_rows, strict=True)
    )
    result_rows: list[dict[str, object]] = []

    for threshold in selected_thresholds:
        total_patch_count = 0
        recomputed_patches = 0
        reused_patches = 0
        transition_patch_count = 0
        transition_recomputed_patches = 0
        counts = ConfusionCounts(0, 0, 0, 0)
        error_sum = 0.0
        feature_value_count = 0
        maximum_feature_error = 0.0
        cosine_sum = 0.0
        feature_vector_count = 0

        for episode in prepared_episodes:
            cache = ThresholdPatchCache(threshold, patch_size=patch_size)
            for frame_index, (image, baseline_features) in enumerate(
                zip(episode.frames, episode.baseline_features, strict=True)
            ):
                encoded = cache.process_frame(image)
                total_patch_count += encoded.patch_count
                recomputed_patches += encoded.recomputed_patches
                reused_patches += encoded.reused_patches

                feature_metrics = compare_feature_vectors(
                    encoded.features, baseline_features
                )
                error_sum += (
                    feature_metrics.mean_absolute_error
                    * feature_metrics.feature_value_count
                )
                feature_value_count += feature_metrics.feature_value_count
                maximum_feature_error = max(
                    maximum_feature_error, feature_metrics.maximum_absolute_error
                )
                cosine_sum += (
                    feature_metrics.mean_cosine_similarity
                    * feature_metrics.feature_vector_count
                )
                feature_vector_count += feature_metrics.feature_vector_count

                if frame_index == 0:
                    continue
                ground_truth = episode.ground_truth_transitions[frame_index - 1]
                if len(ground_truth) != encoded.patch_count:
                    raise ValueError(
                        f"patch grid mismatch in episode {episode.episode_id!r}"
                    )
                transition_counts = count_patch_decisions(
                    ground_truth, encoded.refresh_decisions
                )
                counts = _sum_confusion_counts(counts, transition_counts)
                transition_patch_count += encoded.patch_count
                transition_recomputed_patches += encoded.recomputed_patches

        detection = detection_metrics(counts)
        result_rows.append(
            {
                "threshold": threshold,
                "change_detection": {
                    **detection,
                    "transition_patch_count": transition_patch_count,
                    "transition_recomputed_patches": transition_recomputed_patches,
                    "refresh_fraction": (
                        transition_recomputed_patches / transition_patch_count
                        if transition_patch_count
                        else 0.0
                    ),
                },
                "feature_approximation": {
                    "mean_absolute_feature_error": error_sum / feature_value_count,
                    "maximum_absolute_feature_error": maximum_feature_error,
                    "mean_feature_vector_cosine_similarity": (
                        cosine_sum / feature_vector_count
                    ),
                    "scope": "All frames, including each episode's first frame.",
                },
                "computation": {
                    "total_patch_count": total_patch_count,
                    "recomputed_patches": recomputed_patches,
                    "reused_patches": reused_patches,
                    "refresh_fraction": recomputed_patches / total_patch_count,
                    "patch_computation_proxy": {
                        "name": "recomputed_patch_feature_operations",
                        "value": recomputed_patches,
                        "unit": "patch_operations",
                    },
                },
            }
        )

    summary: dict[str, object] = {
        "experiment_label": EXPERIMENT_LABEL,
        "method": "Threshold gate evaluated against pixel-change masks",
        "manifest_path": Path(manifest_path).as_posix(),
        "episode_count": len(prepared_episodes),
        "frame_count": sum(len(episode.frames) for episode in prepared_episodes),
        "patch_size": patch_size,
        "thresholds": list(selected_thresholds),
        "ground_truth_definition": GROUND_TRUTH_DEFINITION,
        "prediction_definition": PREDICTION_DEFINITION,
        "metric_definitions": METRIC_DEFINITIONS,
        "first_frame_policy": (
            "First frames fully initialize each episode cache and contribute "
            "to feature and computation metrics, but not detection confusion "
            "counts."
        ),
        "feature_approximation_definition": FEATURE_DEFINITION,
        "results": result_rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    write_change_detection_png(result_rows, figure)
    return summary


def _prepare_episode(
    dataset_root: Path,
    episode: ManifestEpisode,
    manifest_row: dict[str, object],
    patch_size: int,
) -> _PreparedEpisode:
    episode_id = episode.episode_id
    if manifest_row.get("episode_id") != episode_id:
        raise ValueError("manifest episode order is inconsistent")
    frame_paths = tuple(frame.path for frame in episode.frames)
    frames = tuple(read_ppm(path) for path in frame_paths)
    baseline_features = tuple(
        encode_image(image, patch_size=patch_size).features for image in frames
    )

    raw_mask_paths = manifest_row.get("ground_truth_mask_paths")
    if not isinstance(raw_mask_paths, list) or len(raw_mask_paths) != len(frames) - 1:
        raise ValueError(
            f"episode {episode_id!r} must have one ground-truth mask per transition"
        )
    transition_labels: list[tuple[bool, ...]] = []
    for transition_index, raw_path in enumerate(raw_mask_paths):
        mask_path = _resolve_mask_path(
            dataset_root,
            raw_path,
            episode_id=episode_id,
            transition_index=transition_index,
        )
        mask_width, mask_height, pixels = _read_pgm_mask(mask_path)
        previous = frames[transition_index]
        current = frames[transition_index + 1]
        if (
            mask_width != previous.width
            or mask_height != previous.height
            or (current.width, current.height) != (previous.width, previous.height)
        ):
            raise ValueError(
                f"frame and mask dimensions differ in episode {episode_id!r} "
                f"transition {transition_index}"
            )
        transition_labels.append(
            ground_truth_mask_to_patch_labels(
                pixels, mask_width, mask_height, patch_size
            )
        )
    return _PreparedEpisode(
        episode_id=episode_id,
        frames=frames,
        baseline_features=baseline_features,
        ground_truth_transitions=tuple(transition_labels),
    )


def _read_manifest_rows(manifest_path: str | Path) -> tuple[dict[str, object], ...]:
    rows = []
    for line_number, line in enumerate(
        Path(manifest_path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        rows.append(row)
    return tuple(rows)


def _resolve_mask_path(
    dataset_root: Path,
    raw_path: object,
    *,
    episode_id: str,
    transition_index: int,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(
            f"episode {episode_id!r} transition {transition_index} has an "
            "invalid mask path"
        )
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or PureWindowsPath(raw_path).drive
        or ".." in relative.parts
        or not relative.parts
    ):
        raise ValueError("ground-truth mask paths must stay within the dataset")
    resolved = (dataset_root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(dataset_root) or not resolved.is_file():
        raise ValueError(
            f"ground-truth mask path is missing or escapes the dataset: {raw_path}"
        )
    return resolved


def _read_pgm_mask(path: Path) -> tuple[int, int, bytes]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read PGM mask {path}: {error}") from error
    header = data.split(b"\n", 3)
    if len(header) != 4 or header[0].strip() != b"P5":
        raise ValueError(f"invalid binary PGM mask header: {path}")
    try:
        width, height = (int(value) for value in header[1].split())
        maximum = int(header[2].strip())
    except ValueError as error:
        raise ValueError(f"invalid PGM mask dimensions or maximum: {path}") from error
    pixels = header[3]
    if width < 1 or height < 1 or maximum != 255:
        raise ValueError(f"unsupported PGM mask dimensions or depth: {path}")
    if len(pixels) != width * height:
        raise ValueError(f"PGM mask pixel count does not match dimensions: {path}")
    return width, height, pixels


def _sum_confusion_counts(
    first: ConfusionCounts, second: ConfusionCounts
) -> ConfusionCounts:
    return ConfusionCounts(
        true_positives=first.true_positives + second.true_positives,
        false_positives=first.false_positives + second.false_positives,
        true_negatives=first.true_negatives + second.true_negatives,
        false_negatives=first.false_negatives + second.false_negatives,
    )
