"""Optional global-change fallback for approximate patch reuse."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .approximate import ThresholdEncodingResult, ThresholdPatchCache
from .encoder import RGBImage


@dataclass(frozen=True, slots=True)
class GlobalFallbackResult:
    """Local-cache encoding plus global detector and full-refresh metadata."""

    encoding: ThresholdEncodingResult
    global_change_score: float | None
    global_change_detected: bool
    full_refresh_reason: str | None
    global_detector_runtime_seconds: float
    processing_latency_seconds: float


def changed_pixel_fraction(current: RGBImage, previous: RGBImage) -> float:
    """Return the fraction of pixels with any changed RGB channel.

    The statistic lies in ``[0, 1]`` and compares complete RGB triplets. It
    measures how much of the rendered frame changed, not change magnitude or
    semantic scene change.
    """
    if (current.width, current.height) != (previous.width, previous.height):
        raise ValueError("global change images must have matching dimensions")
    changed_pixels = 0
    for offset in range(0, len(current.pixels), 3):
        if (
            current.pixels[offset] != previous.pixels[offset]
            or current.pixels[offset + 1] != previous.pixels[offset + 1]
            or current.pixels[offset + 2] != previous.pixels[offset + 2]
        ):
            changed_pixels += 1
    return changed_pixels / (current.width * current.height)


class GlobalFallbackPatchCache:
    """Wrap local threshold reuse with an opt-in global full-refresh check.

    When enabled, the detector computes the fraction of pixel locations whose
    RGB triplet changed from the preceding frame. If this fraction is strictly
    greater than ``global_change_threshold``, the local cache is reset and the
    current frame is fully encoded. Otherwise the unchanged local threshold
    policy decides which patches to refresh. The frame reference and local raw
    patch observations advance after every frame.

    Pass ``global_change_threshold=None`` to disable the detector; in this mode
    features and refresh decisions match :class:`ThresholdPatchCache` exactly.
    Create a new instance or call :meth:`reset` for each episode.
    """

    def __init__(
        self,
        local_threshold: float = 0.05,
        global_change_threshold: float | None = 0.50,
        patch_size: int = 16,
    ) -> None:
        if global_change_threshold is not None and (
            isinstance(global_change_threshold, bool)
            or not isinstance(global_change_threshold, (int, float))
            or not math.isfinite(global_change_threshold)
            or not 0.0 <= global_change_threshold <= 1.0
        ):
            raise ValueError("global_change_threshold must be between 0 and 1")
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self._cache = ThresholdPatchCache(local_threshold, patch_size=patch_size)
        self._global_change_threshold = global_change_threshold
        self._previous_frame: RGBImage | None = None
        self._previous_dimensions: tuple[int, int] | None = None
        self._has_seen_frame = False

    @property
    def local_threshold(self) -> float:
        """Threshold used by the existing local patch policy."""
        return self._cache.threshold

    @property
    def global_change_threshold(self) -> float | None:
        """Configured global changed-pixel-fraction threshold, if enabled."""
        return self._global_change_threshold

    @property
    def detector_enabled(self) -> bool:
        """Whether global image comparisons are enabled."""
        return self._global_change_threshold is not None

    def reset(self) -> None:
        """Clear both local patch state and global frame observations."""
        self._cache.reset()
        self._previous_frame = None
        self._previous_dimensions = None
        self._has_seen_frame = False

    def process_frame(self, image: RGBImage) -> GlobalFallbackResult:
        """Process one frame and optionally force a global full refresh."""
        processing_started = time.perf_counter()
        dimensions = (image.width, image.height)
        detector_runtime = 0.0
        score: float | None = None
        global_change_detected = False

        if not self._has_seen_frame:
            refresh_reason = "episode_initialization"
        elif dimensions != self._previous_dimensions:
            refresh_reason = "image_dimensions_changed"
        elif self._global_change_threshold is not None:
            detector_started = time.perf_counter()
            if self._previous_frame is None:
                raise RuntimeError("global detector frame state is inconsistent")
            score = changed_pixel_fraction(image, self._previous_frame)
            detector_runtime = time.perf_counter() - detector_started
            global_change_detected = score > self._global_change_threshold
            refresh_reason = "global_change" if global_change_detected else None
        else:
            refresh_reason = None

        if global_change_detected:
            self._cache.reset()
        encoding = self._cache.process_frame(image)
        if (
            refresh_reason is None
            and encoding.recomputed_patches == encoding.patch_count
        ):
            refresh_reason = "local_threshold_all_patches"

        self._has_seen_frame = True
        self._previous_dimensions = dimensions
        if self._global_change_threshold is not None:
            self._previous_frame = image
        processing_latency = time.perf_counter() - processing_started
        return GlobalFallbackResult(
            encoding=encoding,
            global_change_score=score,
            global_change_detected=global_change_detected,
            full_refresh_reason=refresh_reason,
            global_detector_runtime_seconds=detector_runtime,
            processing_latency_seconds=processing_latency,
        )
