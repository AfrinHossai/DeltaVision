"""Dependency-free generation of deterministic synthetic image sequences.

Frames are stored as binary PPM (P6) RGB images. Ground-truth masks are binary
PGM (P5) images whose pixels are 255 when any RGB channel differs from the
previous rendered frame, and 0 otherwise. Masks therefore describe rendered
pixel differences, not semantic object edits.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

ShapeType = Literal["rectangle", "circle", "triangle"]
TransformationType = Literal[
    "movement",
    "recoloring",
    "appearance",
    "disappearance",
    "partial_occlusion",
    "global_brightness",
    "unchanged",
]
RGBColor = tuple[int, int, int]

TRANSFORMATION_TYPES: tuple[TransformationType, ...] = (
    "movement",
    "recoloring",
    "appearance",
    "disappearance",
    "partial_occlusion",
    "global_brightness",
    "unchanged",
)

MASK_DEFINITION = (
    "A mask pixel is 255 if any rendered RGB channel differs from the "
    "previous frame at that location, and 0 otherwise. Masks represent "
    "pixel differences rather than semantic object changes."
)


@dataclass(slots=True)
class SceneObject:
    """One filled geometric object in a synthetic scene."""

    object_id: str
    shape: ShapeType
    color: RGBColor
    x: int
    y: int
    size: int
    visible: bool = True


@dataclass(slots=True)
class SceneState:
    """Mutable scene attributes used to render consecutive frames."""

    width: int
    height: int
    background: RGBColor
    objects: list[SceneObject]
    brightness: float = 1.0
    occlusion: tuple[int, int, int, int] | None = None


def generate_sequences(
    output_dir: str | Path,
    *,
    width: int = 128,
    height: int = 128,
    sequence_length: int = 8,
    num_episodes: int = 4,
    seed: int = 0,
    transformations: Sequence[TransformationType] | None = None,
    episode_id_prefix: str = "",
) -> Path:
    """Generate RGB episodes, pixel-difference masks, and a JSONL manifest.

    Each episode uses ``seed + episode_index`` as its seed. One requested
    transformation is selected for every pair of adjacent frames. If a
    requested appearance or disappearance is currently impossible, that
    transition is an unchanged frame. Paths in the returned manifest are
    relative to ``output_dir`` so the dataset can be moved as a directory.
    ``episode_id_prefix`` can namespace IDs when building several splits.

    Args:
        output_dir: Directory where episodes and ``manifest.jsonl`` are saved.
        width: Frame width in pixels. Must be at least 16.
        height: Frame height in pixels. Must be at least 16.
        sequence_length: Number of frames in every episode; must be positive.
        num_episodes: Number of independent episodes; must be positive.
        seed: Base seed from which each episode seed is derived.
        transformations: Transformation types to sample. Defaults to all
            supported types. Supplying one type makes that transformation
            deterministic whenever its scene preconditions are met.
        episode_id_prefix: Optional portable prefix added to episode IDs and
            episode directory names.

    Returns:
        The path to the written ``manifest.jsonl`` file.

    Raises:
        ValueError: If dimensions, counts, or transformation types are invalid.
    """
    _validate_generation_options(width, height, sequence_length, num_episodes)
    if not isinstance(episode_id_prefix, str) or any(
        not (character.isascii() and (character.isalnum() or character in "_-"))
        for character in episode_id_prefix
    ):
        raise ValueError(
            "episode_id_prefix must contain only ASCII letters, digits, _ or -"
        )
    selected_transformations = _normalize_transformations(transformations)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"
    rows: list[dict[str, object]] = []

    for episode_index in range(num_episodes):
        episode_stem = f"episode_{episode_index:04d}"
        episode_id = (
            f"{episode_id_prefix}_{episode_stem}" if episode_id_prefix else episode_stem
        )
        episode_seed = seed + episode_index
        rng = random.Random(episode_seed)
        scene = _make_scene(width, height, rng)
        episode_dir = root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        frame_paths: list[str] = []
        mask_paths: list[str] = []
        transitions: list[dict[str, object]] = []

        previous_frame = _render(scene)
        first_frame_relative = Path(episode_id) / "frame_0000.ppm"
        _write_ppm(root / first_frame_relative, width, height, previous_frame)
        frame_paths.append(first_frame_relative.as_posix())

        for frame_index in range(1, sequence_length):
            transformation_type = _choose_transformation(
                scene, selected_transformations, rng
            )
            parameters = _apply_transformation(scene, transformation_type, rng)
            current_frame = _render(scene)
            changed_mask = _pixel_difference_mask(
                previous_frame, current_frame, width, height
            )

            frame_relative = Path(episode_id) / f"frame_{frame_index:04d}.ppm"
            mask_relative = Path(episode_id) / f"mask_{frame_index - 1:04d}.pgm"
            _write_ppm(root / frame_relative, width, height, current_frame)
            _write_pgm(root / mask_relative, width, height, changed_mask)

            frame_paths.append(frame_relative.as_posix())
            mask_paths.append(mask_relative.as_posix())
            transitions.append(
                {
                    "from_frame": frame_index - 1,
                    "to_frame": frame_index,
                    "transformation_type": transformation_type,
                    "transformation_parameters": parameters,
                    "ground_truth_mask_path": mask_relative.as_posix(),
                }
            )
            previous_frame = current_frame

        rows.append(
            {
                "episode_id": episode_id,
                "random_seed": episode_seed,
                "ground_truth_mask_definition": MASK_DEFINITION,
                "frame_paths": frame_paths,
                "ground_truth_mask_paths": mask_paths,
                "transitions": transitions,
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for row in rows:
            manifest.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            manifest.write("\n")
    return manifest_path


def _validate_generation_options(
    width: int, height: int, sequence_length: int, num_episodes: int
) -> None:
    if width < 16 or height < 16:
        raise ValueError("width and height must both be at least 16 pixels")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")


def _normalize_transformations(
    transformations: Sequence[TransformationType] | None,
) -> tuple[TransformationType, ...]:
    if transformations is None:
        return TRANSFORMATION_TYPES
    selected = tuple(dict.fromkeys(transformations))
    invalid = set(selected).difference(TRANSFORMATION_TYPES)
    if invalid:
        raise ValueError(f"unsupported transformation types: {sorted(invalid)}")
    if not selected:
        raise ValueError("at least one transformation type must be selected")
    return selected


def _make_scene(width: int, height: int, rng: random.Random) -> SceneState:
    """Create four spatially separated objects, with one initially hidden."""
    background: RGBColor = tuple(rng.randint(18, 72) for _ in range(3))  # type: ignore[assignment]
    size = max(2, min(width, height) // 10)
    positions = (
        (width // 4, height // 4),
        (3 * width // 4, height // 4),
        (width // 4, 3 * height // 4),
        (3 * width // 4, 3 * height // 4),
    )
    shapes: tuple[ShapeType, ...] = ("rectangle", "circle", "triangle", "circle")
    objects = [
        SceneObject(
            object_id=f"object_{index}",
            shape=shapes[index],
            color=tuple(rng.randint(100, 245) for _ in range(3)),  # type: ignore[arg-type]
            x=min(max(x, size), width - size - 1),
            y=min(max(y, size), height - size - 1),
            size=size,
            visible=index < len(positions) - 1,
        )
        for index, (x, y) in enumerate(positions)
    ]
    return SceneState(width, height, background, objects)


def _available_transformations(
    scene: SceneState, requested: tuple[TransformationType, ...]
) -> tuple[TransformationType, ...]:
    visible_count = sum(obj.visible for obj in scene.objects)
    has_hidden = any(not obj.visible for obj in scene.objects)
    available = []
    for transformation in requested:
        if transformation == "appearance" and not has_hidden:
            continue
        if transformation == "disappearance" and visible_count <= 1:
            continue
        available.append(transformation)
    return tuple(available)


def _choose_transformation(
    scene: SceneState,
    requested: tuple[TransformationType, ...],
    rng: random.Random,
) -> TransformationType:
    available = _available_transformations(scene, requested)
    if not available:
        return "unchanged"
    return rng.choice(available)


def _apply_transformation(
    scene: SceneState, transformation: TransformationType, rng: random.Random
) -> dict[str, object]:
    if transformation == "unchanged":
        return {"reason": "no_scene_change"}

    visible_objects = [obj for obj in scene.objects if obj.visible]
    if transformation == "movement":
        obj = rng.choice(visible_objects)
        moves = ((-obj.size, 0), (obj.size, 0), (0, -obj.size), (0, obj.size))
        valid_moves = [
            (dx, dy)
            for dx, dy in moves
            if obj.size <= obj.x + dx <= scene.width - obj.size - 1
            and obj.size <= obj.y + dy <= scene.height - obj.size - 1
        ]
        dx, dy = rng.choice(valid_moves)
        obj.x += dx
        obj.y += dy
        return {"object_id": obj.object_id, "dx": dx, "dy": dy}

    if transformation == "recoloring":
        obj = rng.choice(visible_objects)
        previous_color = obj.color
        choices = [tuple(rng.randint(80, 255) for _ in range(3)) for _ in range(8)]
        obj.color = next(color for color in choices if color != previous_color)
        return {
            "object_id": obj.object_id,
            "previous_rgb": list(previous_color),
            "rgb": list(obj.color),
        }

    if transformation == "appearance":
        obj = rng.choice(
            [candidate for candidate in scene.objects if not candidate.visible]
        )
        obj.visible = True
        return {"object_id": obj.object_id, "shape": obj.shape}

    if transformation == "disappearance":
        obj = rng.choice(visible_objects)
        obj.visible = False
        return {"object_id": obj.object_id}

    if transformation == "partial_occlusion":
        obj = rng.choice(visible_objects)
        half = max(1, obj.size // 2)
        previous_occlusion = scene.occlusion
        offsets = [(0, 0), (half, 0), (-half, 0), (0, half), (0, -half)]
        candidates = []
        for offset_x, offset_y in offsets:
            bounds = (
                max(0, obj.x + offset_x - half),
                max(0, obj.y + offset_y - half),
                min(scene.width - 1, obj.x + offset_x + half),
                min(scene.height - 1, obj.y + offset_y + half),
            )
            if bounds != previous_occlusion:
                candidates.append((bounds, offset_x, offset_y))
        bounds, offset_x, offset_y = rng.choice(candidates)
        scene.occlusion = bounds
        return {
            "object_id": obj.object_id,
            "occlusion_bounds_xyxy": list(bounds),
            "occlusion_offset_xy": [offset_x, offset_y],
            "previous_occlusion_bounds_xyxy": (
                list(previous_occlusion) if previous_occlusion is not None else None
            ),
        }

    if transformation == "global_brightness":
        brightness_levels = (0.65, 0.8, 1.0, 1.2, 1.35)
        previous_brightness = scene.brightness
        scene.brightness = rng.choice(
            [level for level in brightness_levels if level != previous_brightness]
        )
        return {
            "previous_factor": previous_brightness,
            "factor": scene.brightness,
        }

    raise ValueError(f"unsupported transformation type: {transformation}")


def _render(scene: SceneState) -> bytes:
    pixels = bytearray(scene.width * scene.height * 3)
    _fill_rectangle(
        pixels,
        scene.width,
        scene.height,
        0,
        0,
        scene.width - 1,
        scene.height - 1,
        _adjust_brightness(scene.background, scene.brightness),
    )
    for obj in scene.objects:
        if obj.visible:
            _draw_object(pixels, scene.width, scene.height, obj, scene.brightness)
    if scene.occlusion is not None:
        left, top, right, bottom = scene.occlusion
        _fill_rectangle(
            pixels,
            scene.width,
            scene.height,
            left,
            top,
            right,
            bottom,
            _adjust_brightness(scene.background, scene.brightness),
        )
    return bytes(pixels)


def _adjust_brightness(color: RGBColor, brightness: float) -> RGBColor:
    return tuple(min(255, max(0, round(channel * brightness))) for channel in color)  # type: ignore[return-value]


def _draw_object(
    pixels: bytearray, width: int, height: int, obj: SceneObject, brightness: float
) -> None:
    left = max(0, obj.x - obj.size)
    right = min(width - 1, obj.x + obj.size)
    top = max(0, obj.y - obj.size)
    bottom = min(height - 1, obj.y + obj.size)
    color = _adjust_brightness(obj.color, brightness)

    if obj.shape == "rectangle":
        _fill_rectangle(pixels, width, height, left, top, right, bottom, color)
        return

    if obj.shape == "circle":
        radius_squared = obj.size * obj.size
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                if (x - obj.x) ** 2 + (y - obj.y) ** 2 <= radius_squared:
                    _set_pixel(pixels, width, x, y, color)
        return

    vertices = (
        (obj.x, obj.y - obj.size),
        (obj.x - obj.size, obj.y + obj.size),
        (obj.x + obj.size, obj.y + obj.size),
    )
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if _inside_triangle(x, y, vertices):
                _set_pixel(pixels, width, x, y, color)


def _inside_triangle(x: int, y: int, vertices: tuple[tuple[int, int], ...]) -> bool:
    (ax, ay), (bx, by), (cx, cy) = vertices
    first = (x - bx) * (ay - by) - (ax - bx) * (y - by)
    second = (x - cx) * (by - cy) - (bx - cx) * (y - cy)
    third = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)
    has_negative = first < 0 or second < 0 or third < 0
    has_positive = first > 0 or second > 0 or third > 0
    return not (has_negative and has_positive)


def _fill_rectangle(
    pixels: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: RGBColor,
) -> None:
    for y in range(max(0, top), min(height - 1, bottom) + 1):
        for x in range(max(0, left), min(width - 1, right) + 1):
            _set_pixel(pixels, width, x, y, color)


def _set_pixel(pixels: bytearray, width: int, x: int, y: int, color: RGBColor) -> None:
    index = (y * width + x) * 3
    pixels[index : index + 3] = bytes(color)


def _pixel_difference_mask(
    previous: bytes, current: bytes, width: int, height: int
) -> bytes:
    if len(previous) != width * height * 3 or len(current) != len(previous):
        raise ValueError("frame byte lengths do not match the configured dimensions")
    mask = bytearray(width * height)
    for pixel_index in range(width * height):
        byte_index = pixel_index * 3
        if (
            previous[byte_index : byte_index + 3]
            != current[byte_index : byte_index + 3]
        ):
            mask[pixel_index] = 255
    return bytes(mask)


def _write_ppm(path: Path, width: int, height: int, pixels: bytes) -> None:
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)


def _write_pgm(path: Path, width: int, height: int, pixels: bytes) -> None:
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + pixels)
