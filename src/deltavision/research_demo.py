"""CPU-only visual demonstration of saved RGB and DINOv2 findings."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .approximate import compare_feature_vectors
from .baseline import load_manifest
from .encoder import RGBImage, encode_image, read_ppm
from .final_evaluation import REPOSITORY_ROOT
from .research_benchmark import (
    _read_jsonl_rows,
    _read_pgm_mask,
    _resolve_dataset_path,
)
from .state import StatefulPatchCache

INK = "#17212b"
MUTED = "#536273"
GRID = "#d9e1e8"
BLUE = "#2878a8"
TEAL = "#16866b"
ORANGE = "#d9822b"
RED = "#bd3b36"
GREEN = "#58a67b"
PURPLE = "#7951a8"
WHITE = "#ffffff"


def _portable_path(path: str | Path) -> str:
    """Represent repository outputs without machine-specific absolute paths."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.name


@dataclass(frozen=True, slots=True)
class DemoSequence:
    """A fully preloaded synthetic validation episode and its pixel masks."""

    episode_id: str
    random_seed: int
    partition: str
    frames: tuple[RGBImage, ...]
    masks: tuple[bytes, ...]
    transformations: tuple[str, ...]
    patch_size: int = 16


def load_demo_sequence(
    manifest_path: str | Path,
    *,
    episode_id: str | None = None,
    patch_size: int = 16,
) -> DemoSequence:
    """Load one existing validation episode; held-out paths are rejected."""
    manifest = Path(manifest_path).resolve()
    parts = {part.lower() for part in manifest.parts}
    if "validation" not in parts or "test" in parts:
        raise ValueError("research demo accepts validation data only")
    rows = _read_jsonl_rows(manifest)
    episodes = load_manifest(manifest)
    if len(rows) != len(episodes):
        raise ValueError("demo manifest rows and frame records do not match")
    selected_index = 0
    if episode_id is not None:
        matches = [
            index
            for index, episode in enumerate(episodes)
            if episode.episode_id == episode_id
        ]
        if not matches:
            raise ValueError(
                f"episode_id not found in validation manifest: {episode_id}"
            )
        selected_index = matches[0]
    episode = episodes[selected_index]
    row = rows[selected_index]
    frames = tuple(read_ppm(frame.path) for frame in episode.frames)
    mask_paths = row.get("ground_truth_mask_paths")
    transitions = row.get("transitions")
    if not isinstance(mask_paths, list) or not isinstance(transitions, list):
        raise ValueError("demo episode lacks generated masks or transitions")
    if len(mask_paths) != len(frames) - 1 or len(transitions) != len(frames) - 1:
        raise ValueError("demo episode masks and transitions do not match its frames")
    masks: list[bytes] = []
    transform_types: list[str] = []
    for path, transition in zip(mask_paths, transitions, strict=True):
        mask_path = _resolve_dataset_path(
            manifest.parent, path, f"demo episode {episode.episode_id} mask"
        )
        width, height, pixels = _read_pgm_mask(mask_path)
        if (width, height) != (frames[0].width, frames[0].height):
            raise ValueError("demo mask dimensions do not match its frames")
        masks.append(pixels)
        transform_types.append(transition["transformation_type"])
    return DemoSequence(
        episode_id=episode.episode_id,
        random_seed=int(row["random_seed"]),
        partition="synthetic_validation",
        frames=frames,
        masks=tuple(masks),
        transformations=tuple(transform_types),
        patch_size=patch_size,
    )


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _paste_frame(
    canvas: Image.Image, frame: RGBImage, box: tuple[int, int, int, int]
) -> None:
    image = Image.frombytes("RGB", (frame.width, frame.height), frame.pixels).resize(
        (box[2] - box[0], box[3] - box[1]), Image.Resampling.NEAREST
    )
    canvas.paste(image, (box[0], box[1]))


def _draw_patch_work(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    decisions: Sequence[bool],
    patch_size: int,
) -> None:
    rows = math.ceil(math.sqrt(len(decisions)))
    columns = math.ceil(len(decisions) / rows)
    cell_width = (box[2] - box[0]) / columns
    cell_height = (box[3] - box[1]) / rows
    for index, refreshed in enumerate(decisions):
        column, row = index % columns, index // columns
        x0 = round(box[0] + column * cell_width)
        y0 = round(box[1] + row * cell_height)
        x1 = round(box[0] + (column + 1) * cell_width)
        y1 = round(box[1] + (row + 1) * cell_height)
        draw.rectangle(
            (x0, y0, x1, y1),
            fill=RED if refreshed else GREEN,
            outline=WHITE,
            width=2,
        )
    draw.rectangle(box, outline=INK, width=2)
    draw.text(
        (box[0], box[3] + 8),
        f"{columns}x{rows} patches at {patch_size}px",
        fill=MUTED,
        font=_font(14),
    )


def render_demo_frame_slides(
    sequence: DemoSequence,
    pretrained_results: Mapping[str, Any],
) -> list[Image.Image]:
    """Create sequence slides plus a separate saved-DINOv2 findings slide."""
    if len(sequence.frames) < 2:
        raise ValueError("demo needs at least two sequential frames")
    if len(sequence.masks) != len(sequence.frames) - 1:
        raise ValueError("demo mask count must equal transition count")
    stateful = StatefulPatchCache(patch_size=sequence.patch_size)
    slides: list[Image.Image] = []
    for frame_index, current in enumerate(sequence.frames):
        independent = encode_image(current, patch_size=sequence.patch_size)
        cached = stateful.process_frame(current)
        feature_error = compare_feature_vectors(cached.features, independent.features)
        if feature_error.maximum_absolute_error != 0.0:
            raise ValueError(
                "exact handcrafted cache differed from independent encoding"
            )
        canvas = Image.new("RGB", (1200, 690), WHITE)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (35, 22),
            "DeltaVision | exact handcrafted RGB-patch reuse",
            fill=INK,
            font=_font(28, bold=True),
        )
        draw.text(
            (38, 64),
            f"Validation episode {sequence.episode_id} | seed "
            f"{sequence.random_seed} | frame "
            f"{frame_index + 1}/{len(sequence.frames)}",
            fill=MUTED,
            font=_font(16),
        )
        panels = (
            (32, 134, 277, 379),
            (319, 134, 564, 379),
            (606, 134, 851, 379),
            (893, 134, 1168, 379),
        )
        titles = (
            "Previous RGB frame",
            "Current RGB frame",
            "Rendered pixel-change mask",
            "Patch computations",
        )
        for box, title in zip(panels, titles, strict=True):
            draw.rounded_rectangle(
                box, radius=10, fill="#f7f9fb", outline=GRID, width=2
            )
            draw.text(
                (box[0] + 10, box[1] + 10), title, fill=INK, font=_font(15, bold=True)
            )
        previous = sequence.frames[max(0, frame_index - 1)]
        _paste_frame(
            canvas,
            previous,
            (
                panels[0][0] + 13,
                panels[0][1] + 42,
                panels[0][2] - 13,
                panels[0][3] - 13,
            ),
        )
        _paste_frame(
            canvas,
            current,
            (
                panels[1][0] + 13,
                panels[1][1] + 42,
                panels[1][2] - 13,
                panels[1][3] - 13,
            ),
        )
        if frame_index == 0:
            mask_image = Image.new("RGB", (current.width, current.height), "black")
            transition_name = "episode initialization"
        else:
            mask = sequence.masks[frame_index - 1]
            mask_image = Image.frombytes(
                "L", (current.width, current.height), mask
            ).convert("RGB")
            transition_name = sequence.transformations[frame_index - 1]
        mask_image = mask_image.resize((190, 190), Image.Resampling.NEAREST)
        canvas.paste(mask_image, (panels[2][0] + 27, panels[2][1] + 46))
        patch_box = (
            panels[3][0] + 18,
            panels[3][1] + 47,
            panels[3][2] - 18,
            panels[3][3] - 34,
        )
        _draw_patch_work(draw, patch_box, cached.refresh_decisions, sequence.patch_size)

        draw.rounded_rectangle(
            (32, 425, 1168, 545), radius=12, fill="#f3f8fb", outline=GRID, width=2
        )
        draw.text(
            (54, 445),
            f"Transition: {transition_name}",
            fill=INK,
            font=_font(18, bold=True),
        )
        draw.text(
            (54, 480),
            f"Independent encoding: {independent.processed_patches} / "
            f"{independent.processed_patches} patch computations",
            fill=BLUE,
            font=_font(16),
        )
        draw.text(
            (54, 510),
            f"Exact reuse: recomputed {cached.recomputed_patches}, reused "
            f"{cached.reused_patches}; max feature difference 0",
            fill=TEAL,
            font=_font(16),
        )
        draw.text(
            (52, 585),
            "Green = exact cached patch reused; red = patch statistics "
            "recomputed. First frame initializes every patch.",
            fill=MUTED,
            font=_font(15),
        )
        draw.text(
            (52, 618),
            "This is the handcrafted RGB-statistics method. DINOv2 results "
            "below cache complete frame representations.",
            fill=PURPLE,
            font=_font(15),
        )
        slides.append(canvas)
    slides.append(_render_pretrained_findings_slide(pretrained_results))
    return slides


def _render_pretrained_findings_slide(
    pretrained_results: Mapping[str, Any],
) -> Image.Image:
    conditions = pretrained_results["conditions"]
    order = (
        "identical_consecutive_frames",
        "sparse_visual_changes",
        "moderate_visual_changes",
        "frequent_widespread_changes",
    )
    canvas = Image.new("RGB", (1200, 690), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (38, 24),
        "Saved DINOv2 validation measurements",
        fill=INK,
        font=_font(28, bold=True),
    )
    draw.text(
        (42, 66),
        "No pretrained inference is rerun for this slide. Each cache hit "
        "reuses a complete exact-matched frame representation.",
        fill=MUTED,
        font=_font(15),
    )
    draw.text(
        (42, 94),
        "No selective DINOv2 token recomputation.",
        fill=RED,
        font=_font(16, bold=True),
    )
    left, right, top, bottom = 350, 1120, 158, 542
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + int((right - left) * tick)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.text((x - 18, bottom + 7), f"{tick:.0%}", fill=MUTED, font=_font(14))
    colors = (TEAL, BLUE, ORANGE, RED)
    band = (bottom - top) / 4
    for index, name in enumerate(order):
        item = conditions[name]
        y = top + int(band * (index + 0.5))
        fraction = float(item["forward_call_reduction"])
        draw.text((42, y - 24), item["label"], fill=INK, font=_font(17, bold=True))
        draw.rectangle(
            (left, y - 13, left + int((right - left) * fraction), y + 13),
            fill=colors[index],
        )
        draw.text(
            (left, y + 18),
            f"calls {item['independent_forward_calls_per_pass']} -> "
            f"{item['stateful_forward_calls_per_pass']} | "
            f"hits {item['cached_frame_hits_per_pass']} | "
            f"speedup {item['median_speedup']:.3f}x",
            fill=MUTED,
            font=_font(14),
        )
    draw.text((350, 584), "Avoided complete forwards", fill=INK, font=_font(15))
    draw.text(
        (42, 634),
        "Synthetic validation data; CPU timing is workload- and "
        "machine-specific. Feature equality does not show semantic accuracy.",
        fill=MUTED,
        font=_font(14),
    )
    return canvas


def write_demo_gif(
    sequence: DemoSequence,
    pretrained_results: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a deterministic looping GIF using only CPU and saved reports."""
    slides = render_demo_frame_slides(sequence, pretrained_results)
    palette_frames = [
        slide.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        for slide in slides
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    durations = [850] * (len(palette_frames) - 1) + [3000]
    palette_frames[0].save(
        destination,
        format="GIF",
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return destination


def write_mp4_if_available(
    slides: Sequence[Image.Image], output_path: str | Path
) -> Path | None:
    """Encode MP4 with a local ffmpeg executable when one is available."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="deltavision-demo-", dir=destination.parent
    ) as temporary:
        frame_dir = Path(temporary)
        for index, slide in enumerate(slides):
            slide.save(
                frame_dir / f"frame_{index:04d}.png", format="PNG", optimize=True
            )
        command = [
            executable,
            "-y",
            "-framerate",
            "1",
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True)
    return destination


def write_demo_metadata(
    sequence: DemoSequence,
    gif_path: str | Path,
    mp4_path: str | Path | None,
    output_path: str | Path,
) -> Path:
    """Record partition, frame count, and rendering limitations next to GIF."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "partition": sequence.partition,
        "episode_id": sequence.episode_id,
        "random_seed": sequence.random_seed,
        "frame_count": len(sequence.frames),
        "transition_types": list(sequence.transformations),
        "feature_family": "handcrafted_rgb_statistics",
        "gif": _portable_path(gif_path),
        "mp4": _portable_path(mp4_path) if mp4_path is not None else None,
        "limitation": (
            "Visual demonstration only; CPU-generated handcrafted patch "
            "features, not a timing benchmark. The separate DINOv2 panel "
            "reuses saved validation results and does not perform inference."
        ),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
