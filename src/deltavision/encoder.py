"""Frame-independent RGB-statistics features for P6 PPM images.

This module contains a handcrafted baseline feature extractor. It is not a
learned or pretrained vision encoder, and it keeps no state between calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

FEATURE_DIMENSIONS = 6
FEATURE_DEFINITION = (
    "For each row-major patch, [R mean, G mean, B mean, R population standard "
    "deviation, G population standard deviation, B population standard "
    "deviation]. Values are normalized to [0, 1] by dividing by 255."
)


@dataclass(frozen=True, slots=True)
class RGBImage:
    """An 8-bit, interleaved RGB image held in row-major byte order."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("image width and height must be positive")
        if len(self.pixels) != self.width * self.height * 3:
            raise ValueError("RGB byte count does not match image dimensions")


@dataclass(frozen=True, slots=True)
class EncodingResult:
    """Patch features and the grid layout for one independently encoded image."""

    grid_rows: int
    grid_columns: int
    feature_dimensions: int
    features: tuple[tuple[float, ...], ...]

    @property
    def processed_patches(self) -> int:
        """Number of patch feature vectors computed for this image."""
        return len(self.features)


def read_ppm(path: str | Path) -> RGBImage:
    """Read a binary P6 PPM file with 8-bit RGB channels.

    The reader accepts P6 files with a maximum channel value of 255, including
    the format emitted by :mod:`deltavision.synthetic`. Other image formats
    and higher bit depths are rejected instead of being implicitly converted.

    Args:
        path: PPM file to load.

    Returns:
        A validated :class:`RGBImage`.

    Raises:
        ValueError: If the file is not a supported P6 image or has invalid data.
    """
    image_path = Path(path)
    try:
        data = image_path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read PPM image {image_path}: {error}") from error

    tokens: list[bytes] = []
    offset = 0
    for _ in range(4):
        token, offset = _read_ppm_token(data, offset)
        tokens.append(token)
    if tokens[0] != b"P6":
        raise ValueError(f"unsupported image format in {image_path}; expected P6 PPM")
    try:
        width, height, maximum = (int(token) for token in tokens[1:])
    except ValueError as error:
        raise ValueError(f"invalid PPM header in {image_path}") from error
    if width < 1 or height < 1:
        raise ValueError(f"PPM dimensions must be positive in {image_path}")
    if maximum != 255:
        raise ValueError(f"PPM maximum value must be 255 in {image_path}")

    expected_bytes = width * height * 3
    pixels = data[offset:]
    if len(pixels) != expected_bytes:
        raise ValueError(
            f"PPM pixel data has {len(pixels)} bytes; expected {expected_bytes} "
            f"for {width}x{height} RGB image"
        )
    return RGBImage(width, height, pixels)


def encode_image(image: RGBImage, patch_size: int = 16) -> EncodingResult:
    """Compute six normalized RGB statistics for every image patch.

    Patches are traversed in row-major order. If the image dimensions are not
    divisible by ``patch_size``, the rightmost and bottom patches are smaller;
    their statistics use only in-bounds pixels, with no padding.

    Every call computes features from the supplied image pixels. This function
    has no temporal state, cache, or cross-call reuse.

    Args:
        image: Validated RGB image to encode.
        patch_size: Positive square patch width and height in pixels.

    Returns:
        Patch feature vectors and their grid dimensions.

    Raises:
        ValueError: If ``patch_size`` is not positive.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")

    grid_columns = (image.width + patch_size - 1) // patch_size
    grid_rows = (image.height + patch_size - 1) // patch_size
    features: list[tuple[float, ...]] = []

    for patch_row in range(grid_rows):
        y_start = patch_row * patch_size
        y_stop = min(y_start + patch_size, image.height)
        for patch_column in range(grid_columns):
            x_start = patch_column * patch_size
            x_stop = min(x_start + patch_size, image.width)
            features.append(_patch_statistics(image, x_start, x_stop, y_start, y_stop))

    return EncodingResult(
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        feature_dimensions=FEATURE_DIMENSIONS,
        features=tuple(features),
    )


def encode_patch(
    image: RGBImage, patch_row: int, patch_column: int, patch_size: int = 16
) -> tuple[float, ...]:
    """Compute the existing six RGB statistics for one row-major patch.

    This exposes the same per-patch calculation used by :func:`encode_image`
    so selective recomputation can use the exact same numerical definition.

    Args:
        image: Validated RGB image containing the patch.
        patch_row: Zero-based patch row in the image grid.
        patch_column: Zero-based patch column in the image grid.
        patch_size: Positive square patch width and height in pixels.

    Returns:
        Normalized RGB means and population standard deviations.

    Raises:
        ValueError: If the patch size or grid coordinates are invalid.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    grid_columns = (image.width + patch_size - 1) // patch_size
    grid_rows = (image.height + patch_size - 1) // patch_size
    if not 0 <= patch_row < grid_rows or not 0 <= patch_column < grid_columns:
        raise ValueError("patch coordinates are outside the image grid")
    x_start = patch_column * patch_size
    y_start = patch_row * patch_size
    x_stop = min(x_start + patch_size, image.width)
    y_stop = min(y_start + patch_size, image.height)
    return _patch_statistics(image, x_start, x_stop, y_start, y_stop)


def _patch_statistics(
    image: RGBImage, x_start: int, x_stop: int, y_start: int, y_stop: int
) -> tuple[float, ...]:
    """Compute population means and standard deviations using Welford updates."""
    count = 0
    means = [0.0, 0.0, 0.0]
    sums_of_squares = [0.0, 0.0, 0.0]

    for y in range(y_start, y_stop):
        for x in range(x_start, x_stop):
            offset = (y * image.width + x) * 3
            count += 1
            for channel in range(3):
                value = image.pixels[offset + channel]
                delta = value - means[channel]
                means[channel] += delta / count
                delta_after_update = value - means[channel]
                sums_of_squares[channel] += delta * delta_after_update

    normalized_means = [mean / 255.0 for mean in means]
    normalized_standard_deviations = [
        math.sqrt(max(0.0, sum_of_squares / count)) / 255.0
        for sum_of_squares in sums_of_squares
    ]
    return tuple(normalized_means + normalized_standard_deviations)


def _read_ppm_token(data: bytes, offset: int) -> tuple[bytes, int]:
    """Read one ASCII PPM header token, skipping whitespace and comments."""
    whitespace = b" \t\r\n\v\f"
    while offset < len(data):
        if data[offset] in whitespace:
            offset += 1
        elif data[offset] == ord("#"):
            newline = data.find(b"\n", offset)
            if newline == -1:
                return b"", len(data)
            offset = newline + 1
        else:
            break

    token_start = offset
    while offset < len(data) and data[offset] not in whitespace:
        offset += 1
    token = data[token_start:offset]
    if offset < len(data):
        if data[offset] == ord("\r") and offset + 1 < len(data):
            if data[offset + 1] == ord("\n"):
                offset += 2
            else:
                offset += 1
        else:
            offset += 1
    return token, offset
