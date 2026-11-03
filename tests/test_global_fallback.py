"""Tests for the optional global scene-change refresh policy."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from deltavision.approximate import ThresholdPatchCache
from deltavision.encoder import RGBImage, encode_image
from deltavision.global_evaluation import evaluate_global_change_validation
from deltavision.global_fallback import (
    GlobalFallbackPatchCache,
    changed_pixel_fraction,
)
from deltavision.synthetic import generate_sequences


def _solid_image(width: int, height: int, rgb: tuple[int, int, int]) -> RGBImage:
    return RGBImage(width, height, bytes(rgb) * width * height)


def _add_square(image: RGBImage, x: int, y: int, size: int) -> RGBImage:
    pixels = bytearray(image.pixels)
    for row in range(y, y + size):
        for column in range(x, x + size):
            offset = (row * image.width + column) * 3
            pixels[offset : offset + 3] = bytes((220, 30, 50))
    return RGBImage(image.width, image.height, bytes(pixels))


def _without_runtime(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_runtime(child)
            for key, child in value.items()
            if "runtime" not in key
        }
    if isinstance(value, list):
        return [_without_runtime(child) for child in value]
    return value


def _make_validation_manifest(
    root: Path, *, transformations: tuple[str, ...], sequence_length: int = 3
) -> Path:
    return generate_sequences(
        root,
        width=32,
        height=32,
        sequence_length=sequence_length,
        num_episodes=2,
        seed=814,
        transformations=transformations,  # type: ignore[arg-type]
        episode_id_prefix="validation",
    )


def test_unchanged_frames_do_not_trigger_global_refresh() -> None:
    image = _solid_image(32, 32, (30, 60, 90))
    cache = GlobalFallbackPatchCache(
        local_threshold=0.05, global_change_threshold=0.50, patch_size=16
    )
    cache.process_frame(image)

    result = cache.process_frame(image)

    assert result.global_change_score == 0.0
    assert not result.global_change_detected
    assert result.full_refresh_reason is None
    assert result.encoding.recomputed_patches == 0
    assert result.encoding.reused_patches == 4


def test_global_brightness_change_forces_full_refresh_and_updates_cache() -> None:
    before = _solid_image(32, 32, (80, 100, 120))
    after = _solid_image(32, 32, (140, 160, 180))
    cache = GlobalFallbackPatchCache(
        local_threshold=0.05, global_change_threshold=0.50, patch_size=16
    )
    cache.process_frame(before)

    refreshed = cache.process_frame(after)
    repeated = cache.process_frame(after)

    assert refreshed.global_change_score == 1.0
    assert refreshed.global_change_detected
    assert refreshed.full_refresh_reason == "global_change"
    assert refreshed.encoding.recomputed_patches == refreshed.encoding.patch_count == 4
    assert refreshed.encoding.reused_patches == 0
    assert refreshed.encoding.features == encode_image(after, patch_size=16).features
    assert not repeated.global_change_detected
    assert repeated.encoding.recomputed_patches == 0
    assert repeated.encoding.reused_patches == 4
    assert repeated.encoding.features == refreshed.encoding.features


def test_global_threshold_boundary_uses_strict_greater_than_rule() -> None:
    before = _solid_image(16, 16, (30, 60, 90))
    after = _solid_image(16, 16, (130, 160, 190))
    cache = GlobalFallbackPatchCache(
        local_threshold=0.05, global_change_threshold=1.0, patch_size=16
    )
    cache.process_frame(before)

    result = cache.process_frame(after)

    assert result.global_change_score == 1.0
    assert not result.global_change_detected
    assert result.full_refresh_reason == "local_threshold_all_patches"


def test_localized_object_movement_does_not_trigger_global_fallback() -> None:
    background = _solid_image(32, 32, (10, 20, 30))
    before = _add_square(background, 3, 5, 8)
    after = _add_square(background, 7, 5, 8)

    score = changed_pixel_fraction(after, before)
    cache = GlobalFallbackPatchCache(
        local_threshold=0.05, global_change_threshold=0.50, patch_size=16
    )
    cache.process_frame(before)
    result = cache.process_frame(after)

    assert score < 0.50
    assert result.global_change_score == score
    assert not result.global_change_detected
    assert result.full_refresh_reason != "global_change"


def test_episode_reset_clears_global_and_local_state() -> None:
    first_episode = _solid_image(32, 32, (20, 30, 40))
    second_episode = _solid_image(32, 32, (200, 210, 220))
    cache = GlobalFallbackPatchCache(global_change_threshold=0.50, patch_size=16)
    cache.process_frame(first_episode)
    cache.process_frame(first_episode)

    cache.reset()
    result = cache.process_frame(second_episode)

    assert result.global_change_score is None
    assert not result.global_change_detected
    assert result.full_refresh_reason == "episode_initialization"
    assert result.encoding.recomputed_patches == 4


def test_disabled_fallback_reproduces_local_threshold_decisions() -> None:
    frames = (
        _solid_image(32, 32, (80, 100, 120)),
        _solid_image(32, 32, (84, 104, 124)),
        _solid_image(32, 32, (180, 20, 50)),
    )
    local = ThresholdPatchCache(threshold=0.05, patch_size=16)
    disabled = GlobalFallbackPatchCache(
        local_threshold=0.05,
        global_change_threshold=None,
        patch_size=16,
    )

    for frame in frames:
        expected = local.process_frame(frame)
        actual = disabled.process_frame(frame)
        assert actual.encoding.features == expected.features
        assert actual.encoding.refresh_decisions == expected.refresh_decisions
        assert actual.encoding.recomputed_patches == expected.recomputed_patches
        assert actual.encoding.reused_patches == expected.reused_patches
        assert actual.global_detector_runtime_seconds == 0.0
        assert actual.global_change_score is None


def test_validation_evaluation_metrics_and_serialization(tmp_path: Path) -> None:
    manifest = _make_validation_manifest(
        tmp_path / "validation", transformations=("unchanged",), sequence_length=2
    )
    output = tmp_path / "results" / "global.json"
    figure = tmp_path / "figures" / "global.png"

    result = evaluate_global_change_validation(manifest, output, figure, patch_size=16)
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert result == serialized
    assert result["partition"] == "validation"
    assert result["threshold_selection"]["held_out_test_accessed"] is False
    assert result["threshold_selection"]["preferred_threshold_selected"] is False
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">2I", figure.read_bytes()[16:24]) == (1100, 930)

    for method in result["methods"]:
        aggregate = method["aggregate"]
        detection = aggregate["change_detection"]
        assert detection["true_positives"] == 0
        assert detection["false_positives"] == 0
        assert detection["true_negatives"] == 8
        assert detection["false_negatives"] == 0
        assert detection["precision"] == detection["recall"] == detection["f1"] == 0.0
        assert detection["missed_change_rate"] == 0.0
        assert detection["false_refresh_rate"] == 0.0
        assert detection["transition_patch_count"] == 8
        assert detection["full_frame_refresh_frequency"] == 0.0
        assert aggregate["feature_approximation"]["mean_absolute_feature_error"] == 0.0


def test_global_validation_evaluation_is_reproducible_except_runtime(
    tmp_path: Path,
) -> None:
    manifest = _make_validation_manifest(
        tmp_path / "validation", transformations=("global_brightness",)
    )

    first = evaluate_global_change_validation(
        manifest, tmp_path / "first.json", tmp_path / "first.png"
    )
    second = evaluate_global_change_validation(
        manifest, tmp_path / "second.json", tmp_path / "second.png"
    )

    assert _without_runtime(first) == _without_runtime(second)
    global_method = first["methods"][-1]
    assert global_method["aggregate"]["runtime"]["global_detector_runtime_seconds"] > 0


def test_global_brightness_metric_counts_are_refreshed(tmp_path: Path) -> None:
    manifest = _make_validation_manifest(
        tmp_path / "validation",
        transformations=("global_brightness",),
        sequence_length=2,
    )

    summary = evaluate_global_change_validation(
        manifest, tmp_path / "brightness.json", tmp_path / "brightness.png"
    )
    global_method = summary["methods"][-1]
    detection = global_method["aggregate"]["change_detection"]
    brightness = global_method["per_transformation"]["global_brightness"]

    assert detection["recall"] == 1.0
    assert detection["false_negatives"] == 0
    assert detection["global_full_refresh_count"] == 2
    assert brightness["change_detection"]["recall"] == 1.0
    assert brightness["change_detection"]["refresh_fraction"] == 1.0


def test_non_validation_manifest_and_invalid_threshold_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = generate_sequences(
        tmp_path / "other",
        width=32,
        height=32,
        sequence_length=2,
        num_episodes=1,
        seed=21,
        transformations=("unchanged",),
    )
    with pytest.raises(ValueError, match="validation-prefixed"):
        evaluate_global_change_validation(manifest)
    with pytest.raises(ValueError, match="global_change_threshold"):
        GlobalFallbackPatchCache(global_change_threshold=1.1)
    with pytest.raises(ValueError, match="matching dimensions"):
        changed_pixel_fraction(
            _solid_image(2, 2, (0, 0, 0)), _solid_image(1, 1, (0, 0, 0))
        )
