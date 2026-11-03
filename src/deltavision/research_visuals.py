"""Research figures and validation-only failure gallery from saved outputs."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .approximate import ThresholdPatchCache
from .baseline import load_manifest
from .encoder import RGBImage, read_ppm
from .final_evaluation import (
    REPOSITORY_ROOT,
    validate_final_manifest_provenance,
    validate_result_provenance,
)
from .global_fallback import GlobalFallbackPatchCache
from .pretrained_cache_evaluation import preload_validation_episodes
from .research_benchmark import (
    _read_jsonl_rows,
    _read_pgm_mask,
    _resolve_dataset_path,
    load_research_config,
)

PAGE = "#ffffff"
INK = "#17212b"
MUTED = "#536273"
GRID = "#d9e1e8"
BLUE = "#2878a8"
TEAL = "#16866b"
ORANGE = "#d9822b"
RED = "#bd3b36"
GREEN = "#58a67b"
PURPLE = "#7951a8"
GOLD = "#bb8a13"


def _repository_relative_path(path: str | Path) -> str:
    """Return portable source provenance for paths contained in this repo."""
    return Path(path).resolve().relative_to(REPOSITORY_ROOT).as_posix()


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    """Load an existing result JSON and give an actionable missing-file error."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"required saved experiment output is missing: {source}"
        )
    try:
        result = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid experiment JSON {source}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"experiment JSON must contain an object: {source}")
    return result


def validate_visual_provenance(
    final_evaluation: Mapping[str, Any],
    final_statistics: Mapping[str, Any],
    capacity_ablation: Mapping[str, Any],
    global_validation: Mapping[str, Any],
) -> None:
    """Check that each plotted source has explicit compatible provenance."""
    validate_result_provenance(final_evaluation)
    if final_evaluation.get("pretrained_dinov2", {}).get("status") != "complete":
        raise ValueError("saved DINOv2 final evaluation is incomplete")
    if final_evaluation.get("pretrained_dinov2", {}).get("partition") != "validation":
        raise ValueError("DINOv2 summary figures require validation-only outputs")
    if final_statistics.get("source_config_sha256") != final_evaluation.get(
        "config_sha256"
    ):
        raise ValueError("statistics and evaluation config hashes do not match")
    if capacity_ablation.get("partition") != "validation":
        raise ValueError("cache-capacity figure requires validation-only ablation")
    if capacity_ablation.get("held_out_test_partition_accessed") is not False:
        raise ValueError(
            "cache-capacity source must declare test data was not accessed"
        )
    if global_validation.get("partition") != "validation":
        raise ValueError("global-fallback gallery source must be validation-only")
    handcrafted = final_evaluation["data_provenance"]["handcrafted_final_evaluation"]
    if handcrafted.get("partition") != "previously_inspected_held_out_test":
        raise ValueError(
            "handcrafted final summary must disclose prior test inspection"
        )


def load_visual_sources(
    *,
    final_evaluation_path: str | Path,
    final_statistics_path: str | Path,
    capacity_ablation_path: str | Path,
    global_validation_path: str | Path,
) -> dict[str, Any]:
    """Read and cross-check the saved Task 15–17 reports used by all figures."""
    sources = {
        "final_evaluation": load_json_artifact(final_evaluation_path),
        "final_statistics": load_json_artifact(final_statistics_path),
        "capacity_ablation": load_json_artifact(capacity_ablation_path),
        "global_validation": load_json_artifact(global_validation_path),
        "source_paths": {
            "final_evaluation": _repository_relative_path(final_evaluation_path),
            "final_statistics": _repository_relative_path(final_statistics_path),
            "capacity_ablation": _repository_relative_path(capacity_ablation_path),
            "global_validation": _repository_relative_path(global_validation_path),
        },
    }
    validate_visual_provenance(
        sources["final_evaluation"],
        sources["final_statistics"],
        sources["capacity_ablation"],
        sources["global_validation"],
    )
    return sources


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


def _new_canvas(
    size: tuple[int, int], title: str, subtitle: str = ""
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, PAGE)
    draw = ImageDraw.Draw(image)
    draw.text((42, 26), title, fill=INK, font=_font(30, bold=True))
    if subtitle:
        draw.text((44, 68), subtitle, fill=MUTED, font=_font(16))
    return image, draw


def _wrap(text: str, width: int = 112) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    width: int = 112,
    font: ImageFont.ImageFont | None = None,
    fill: str = MUTED,
    line_height: int = 22,
) -> int:
    y = xy[1]
    selected_font = font or _font(16)
    for line in _wrap(text, width):
        draw.text((xy[0], y), line, fill=fill, font=selected_font)
        y += line_height
    return y


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    *,
    fill: str = "#f6f8fa",
    outline: str = GRID,
) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=outline, width=2)
    draw.text((box[0] + 16, box[1] + 12), title, fill=INK, font=_font(18, bold=True))


def _draw_line_chart_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    x_label: str,
    y_label: str,
    x_max: float,
    y_max: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    plot = (left + 82, top + 28, right - 22, bottom - 68)
    x0, y0, x1, y1 = plot
    draw.line((x0, y1, x1, y1), fill=INK, width=2)
    draw.line((x0, y0, x0, y1), fill=INK, width=2)
    for tick in x_ticks:
        x = x0 + int((x1 - x0) * tick / x_max)
        draw.line((x, y0, x, y1), fill=GRID, width=1)
        draw.text((x - 16, y1 + 8), f"{tick:g}", fill=MUTED, font=_font(13))
    for tick in y_ticks:
        y = y1 - int((y1 - y0) * tick / y_max)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        draw.text((x0 - 48, y - 8), f"{tick:g}", fill=MUTED, font=_font(13))
    draw.text((x0 + (x1 - x0) // 2 - 80, y1 + 36), x_label, fill=INK, font=_font(15))
    draw.text((left + 8, top + 14), y_label, fill=INK, font=_font(15, bold=True))
    return plot


def write_architecture_figure(path: str | Path) -> Path:
    """Draw a schematic that keeps handcrafted and learned paths distinct."""
    image, draw = _new_canvas(
        (1200, 760),
        "DeltaVision: stateful visual encoding under change",
        "Architecture schematic; arrows describe methods, not measured speed "
        "or semantic accuracy.",
    )
    _panel(draw, (42, 118, 1158, 202), "Sequential RGB observations")
    for i, x in enumerate((78, 185, 292, 399)):
        draw.rounded_rectangle((x, 158, x + 58, 184), radius=5, fill="#d7e9f5")
        draw.text((x + 15, 163), f"t{i}", fill=INK, font=_font(14, bold=True))
        if i < 3:
            draw.line((x + 66, 171, x + 95, 171), fill=MUTED, width=3)
            draw.polygon(((x + 95, 171), (x + 85, 165), (x + 85, 177)), fill=MUTED)
    draw.text(
        (535, 158),
        "same preloaded sequence for each policy",
        fill=MUTED,
        font=_font(16),
    )

    _panel(draw, (42, 242, 568, 585), "Handcrafted RGB-statistics path", fill="#f3f8fb")
    draw.rounded_rectangle(
        (70, 300, 245, 380), radius=10, fill="#dcecf7", outline=BLUE, width=2
    )
    draw.text((88, 315), "Split into raw RGB", fill=INK, font=_font(16, bold=True))
    draw.text((100, 341), "patches", fill=INK, font=_font(16, bold=True))
    draw.line((245, 340, 282, 340), fill=BLUE, width=3)
    draw.polygon(((282, 340), (271, 334), (271, 346)), fill=BLUE)
    draw.rounded_rectangle(
        (290, 300, 525, 380), radius=10, fill="#dcecf7", outline=BLUE, width=2
    )
    draw.text((310, 314), "6 RGB statistics", fill=INK, font=_font(16, bold=True))
    draw.text((312, 341), "per 16 px patch", fill=MUTED, font=_font(15))
    draw.text((74, 411), "Independent: recompute all patches", fill=INK, font=_font(15))
    draw.text(
        (74, 444), "Exact: reuse only byte-identical patches", fill=INK, font=_font(15)
    )
    draw.text(
        (74, 477), "Approximate: local RGB-difference gate", fill=INK, font=_font(15)
    )
    draw.text(
        (74, 510),
        "Fallback: local gate plus image-area check",
        fill=INK,
        font=_font(15),
    )
    draw.text(
        (74, 550),
        "Output: handcrafted vectors; pixel-change evaluation",
        fill=BLUE,
        font=_font(14, bold=True),
    )

    _panel(draw, (590, 242, 1158, 585), "Frozen pretrained DINOv2 path", fill="#f8f5fb")
    draw.rounded_rectangle(
        (616, 300, 770, 380), radius=10, fill="#eee5f7", outline=PURPLE, width=2
    )
    draw.text((642, 315), "Pinned", fill=INK, font=_font(16, bold=True))
    draw.text((632, 341), "preprocessing", fill=MUTED, font=_font(15))
    draw.line((770, 340, 800, 340), fill=PURPLE, width=3)
    draw.polygon(((800, 340), (789, 334), (789, 346)), fill=PURPLE)
    draw.rounded_rectangle(
        (808, 300, 1002, 380), radius=10, fill="#eee5f7", outline=PURPLE, width=2
    )
    draw.text((828, 315), "Frozen DINOv2", fill=INK, font=_font(16, bold=True))
    draw.text((831, 341), "full forward", fill=MUTED, font=_font(15))
    draw.line((1002, 340, 1020, 340), fill=PURPLE, width=3)
    draw.polygon(((1020, 340), (1009, 334), (1009, 346)), fill=PURPLE)
    draw.rounded_rectangle(
        (1024, 292, 1135, 390), radius=10, fill="#e7d9f2", outline=PURPLE, width=2
    )
    draw.text((1034, 308), "Spatial", fill=INK, font=_font(15, bold=True))
    draw.text((1033, 331), "tokens", fill=INK, font=_font(15, bold=True))
    draw.text((1032, 355), "16x16x384", fill=MUTED, font=_font(13))
    draw.text(
        (616, 420),
        "Independent: one full model forward per frame",
        fill=INK,
        font=_font(15),
    )
    draw.text(
        (616, 453),
        "Exact cache: hash + tensor equality, capacity 8/episode",
        fill=INK,
        font=_font(15),
    )
    draw.text(
        (616, 486),
        "A hit reuses the complete frame representation",
        fill=INK,
        font=_font(15),
    )
    draw.text(
        (616, 519),
        "No selective DINOv2 token computation",
        fill=RED,
        font=_font(15, bold=True),
    )
    draw.text(
        (616, 550),
        "Output: numerical equivalence and full-call counts",
        fill=PURPLE,
        font=_font(14, bold=True),
    )

    _panel(
        draw,
        (42, 615, 1158, 715),
        "Evidence boundary",
        fill="#fffaf0",
        outline="#ead9a7",
    )
    _draw_wrapped(
        draw,
        "Synthetic pixel-change masks are not semantic ground truth. RGB "
        "patch-operation proxies and DINOv2 full-model forward calls are "
        "different quantities. Similar or identical features do not establish "
        "semantic accuracy.",
        (64, 655),
        width=135,
        font=_font(15),
        line_height=20,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return destination


def write_final_summary_figure(sources: Mapping[str, Any], path: str | Path) -> Path:
    """Summarize strongest findings without combining unlike cost measures."""
    evaluation = sources["final_evaluation"]
    handcrafted = evaluation["handcrafted"]["methods"]
    dinov2 = evaluation["pretrained_dinov2"]["conditions"]
    ablation = sources["capacity_ablation"]["condition_results"]
    exact = handcrafted["exact_patch_reuse"]["aggregate"]
    approximate = handcrafted["approximate_patch_reuse_0.01"]["aggregate"]
    fallback = handcrafted["global_change_fallback"]["aggregate"]
    identical = dinov2["identical_consecutive_frames"]
    widespread = dinov2["frequent_widespread_changes"]
    image, draw = _new_canvas(
        (1200, 825),
        "Validated findings from synthetic studies",
        "Task 17 handcrafted results use the previously inspected test split; "
        "DINOv2 and Task 16 capacity results use validation workloads.",
    )
    cards = [
        (
            "Exact RGB patch reuse",
            f"F1 {exact['f1']:.3f}; "
            "all-frame patch operations "
            f"{exact['all_frame_refresh_fraction']:.1%}; "
            "feature MAE 0.",
            BLUE,
        ),
        (
            "Approximate RGB reuse (0.01)",
            f"F1 {approximate['f1']:.3f}; "
            "missed-change "
            f"{approximate['missed_change_rate']:.2%}; "
            "feature MAE "
            f"{approximate['feature_mean_absolute_error']:.2g}.",
            ORANGE,
        ),
        (
            "Global fallback (0.05 / 0.50)",
            f"F1 {fallback['f1']:.3f}; "
            "missed-change "
            f"{fallback['missed_change_rate']:.1%}. "
            "Small-area changes can remain under its area trigger.",
            PURPLE,
        ),
        (
            "DINOv2: identical frames",
            f"{identical['independent_forward_calls_per_pass']} "
            "to "
            f"{identical['stateful_forward_calls_per_pass']} "
            "full forwards; "
            f"{identical['median_speedup']:.2f}x "
            "measured CPU ratio.",
            TEAL,
        ),
        (
            "DINOv2: widespread changes",
            f"{widespread['cached_frame_hits_per_pass']} "
            "hits; "
            f"{widespread['median_speedup']:.3f}x "
            "measured CPU ratio. "
            "Every frame still runs the full model.",
            RED,
        ),
        (
            "Capacity on recurring frames",
            _capacity_summary_sentence(ablation),
            GOLD,
        ),
    ]
    positions = [(42, 124), (616, 124), (42, 345), (616, 345), (42, 566), (616, 566)]
    for (title, body, color), (x, y) in zip(cards, positions, strict=True):
        draw.rounded_rectangle(
            (x, y, x + 542, y + 188), radius=16, fill="#f8fafc", outline=GRID, width=2
        )
        draw.rounded_rectangle((x, y, x + 12, y + 188), radius=6, fill=color)
        draw.text((x + 30, y + 22), title, fill=INK, font=_font(20, bold=True))
        _draw_wrapped(
            draw, body, (x + 30, y + 65), width=50, font=_font(17), line_height=25
        )
    draw.text(
        (46, 786),
        "Patch operations are handcrafted work proxies; DINOv2 calls are "
        "complete neural forwards. No semantic or natural-video claim.",
        fill=MUTED,
        font=_font(15),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return destination


def _capacity_summary_sentence(ablation: Mapping[str, Any]) -> str:
    recurring = ablation["recurring_nonconsecutive_frames"]["capacities"]
    widest = max(recurring, key=lambda item: item["capacity"])
    smallest = min(recurring, key=lambda item: item["capacity"])
    return (
        f"On {widest['total_frames_per_pass']} recurring frames, capacity "
        f"{widest['capacity']} reached {widest['cache_hit_rate']:.1%} hits "
        f"and {widest['independent_over_cache_median_speedup_ratio']:.2f}x; "
        f"capacity {smallest['capacity']} had {smallest['cache_hit_rate']:.1%} hits."
    )


def write_accuracy_efficiency_figure(
    sources: Mapping[str, Any], path: str | Path
) -> Path:
    """Plot handcrafted transition F1 against all-frame patch-operation share."""
    evaluation = sources["final_evaluation"]
    statistics_result = sources["final_statistics"]["handcrafted"]["methods"]
    method_order = (
        "independent_encoding",
        "exact_patch_reuse",
        "approximate_patch_reuse_0.01",
        "global_change_fallback",
    )
    colors = ("#697987", BLUE, ORANGE, PURPLE)
    image, draw = _new_canvas(
        (1200, 800),
        "RGB-statistics accuracy and computation trade-off",
        "Previously inspected synthetic test partition; transition F1 versus "
        "all-frame handcrafted patch-feature operations.",
    )
    plot = _draw_line_chart_axes(
        draw,
        (65, 130, 1150, 610),
        x_ticks=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        y_ticks=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        x_label="Patch-feature operations / independent baseline",
        y_label="Transition F1",
        x_max=1.0,
        y_max=1.0,
    )
    x0, y0, x1, y1 = plot
    for index, name in enumerate(method_order):
        method = evaluation["handcrafted"]["methods"][name]
        aggregate = method["aggregate"]
        summary = statistics_result[name]
        x_value = float(aggregate["all_frame_refresh_fraction"])
        y_value = float(aggregate["f1"])
        interval = summary["overall_confidence_intervals"]["f1"]
        px = x0 + int((x1 - x0) * x_value)
        py = y1 - int((y1 - y0) * y_value)
        y_low = y1 - int((y1 - y0) * float(interval["lower"]))
        y_high = y1 - int((y1 - y0) * float(interval["upper"]))
        draw.line((px, y_high, px, y_low), fill=colors[index], width=4)
        draw.line((px - 8, y_high, px + 8, y_high), fill=colors[index], width=3)
        draw.line((px - 8, y_low, px + 8, y_low), fill=colors[index], width=3)
        radius = 10 if index == 0 else 7 + (index - 1) * 3
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=None,
            outline=colors[index],
            width=3,
        )
    draw.text(
        (75, 670),
        "Whiskers: 95% episode-cluster bootstrap interval. Reuse markers are "
        "nested to show overlapping points; marker size has no data meaning.",
        fill=MUTED,
        font=_font(14),
    )
    legend = (
        (90, "Independent", colors[0]),
        (340, "Exact reuse", colors[1]),
        (570, "Approximate 0.01", colors[2]),
        (850, "Global fallback", colors[3]),
    )
    for x, label, color in legend:
        draw.ellipse((x, 715, x + 14, 729), fill=color)
        draw.text((x + 21, 713), label, fill=INK, font=_font(14))
    draw.text(
        (75, 760),
        "Previously inspected test partition. RGB pixel-change masks are not "
        "semantic labels.",
        fill=MUTED,
        font=_font(14),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return destination


def write_pretrained_speedup_figure(
    sources: Mapping[str, Any], path: str | Path
) -> Path:
    """Plot saved validation DINOv2 call reduction separately from latency."""
    conditions = sources["final_evaluation"]["pretrained_dinov2"]["conditions"]
    labels = (
        "identical_consecutive_frames",
        "sparse_visual_changes",
        "moderate_visual_changes",
        "frequent_widespread_changes",
    )
    image, draw = _new_canvas(
        (1200, 700),
        "DINOv2 exact whole-frame cache on validation sequences",
        "Each condition: four 8-frame episodes. Bars show complete model-forward "
        "calls avoided; latency ratio is measured CPU end-to-end.",
    )
    left, right, top, bottom = 360, 1110, 155, 570
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = left + int((right - left) * tick)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.text((x - 14, bottom + 8), f"{tick:.0%}", fill=MUTED, font=_font(14))
    colors = (TEAL, BLUE, GOLD, RED)
    band = (bottom - top) / 4
    for index, name in enumerate(labels):
        item = conditions[name]
        y = top + int(band * (index + 0.5))
        reduction = float(item["forward_call_reduction"])
        draw.text((45, y - 22), item["label"], fill=INK, font=_font(17, bold=True))
        draw.rectangle(
            (left, y - 15, left + int((right - left) * reduction), y + 15),
            fill=colors[index],
        )
        draw.text(
            (left, y + 22),
            f"{item['independent_forward_calls_per_pass']} to "
            f"{item['stateful_forward_calls_per_pass']} forwards; "
            f"{item['cached_frame_hits_per_pass']} hits; "
            f"{item['median_speedup']:.3f}x measured latency ratio; "
            f"{item['cache_memory_peak_bytes'] / (1024 * 1024):.2f} MiB payload",
            fill=MUTED,
            font=_font(14),
        )
    draw.text(
        (left, bottom + 39),
        "Avoided full-model forwards / independent calls",
        fill=INK,
        font=_font(15),
    )
    draw.text(
        (45, 647),
        "Capacity 8 per episode. Exact input equality is checked on cache hits; "
        "no selective spatial-token recomputation.",
        fill=MUTED,
        font=_font(14),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return destination


def write_cache_capacity_figure(sources: Mapping[str, Any], path: str | Path) -> Path:
    """Plot validation hit rates and measured ratios by Task 16 capacity."""
    condition_results = sources["capacity_ablation"]["condition_results"]
    selected = (
        ("repeated_identical_frames", "Repeated identical", TEAL),
        ("recurring_nonconsecutive_frames", "Recurring nonconsecutive", BLUE),
        ("sparse_changes", "Sparse changes", ORANGE),
        ("frequent_widespread_changes", "Frequent widespread", RED),
    )
    image, draw = _new_canvas(
        (1200, 850),
        "Validation cache-capacity ablation",
        "Task 16 synthetic inputs; capacity is per episode. Left: cache-hit rate. "
        "Right: independent/cache median latency ratio.",
    )
    capacities = [
        int(item["capacity"])
        for item in condition_results["repeated_identical_frames"]["capacities"]
    ]
    panels = ((55, 132, 580, 565), (620, 132, 1145, 565))
    for panel, y_max, y_ticks, y_label in (
        (panels[0], 1.0, (0.0, 0.25, 0.5, 0.75, 1.0), "Hit rate"),
        (panels[1], 15.0, (0.0, 3.0, 6.0, 9.0, 12.0, 15.0), "Latency ratio"),
    ):
        plot = _draw_line_chart_axes(
            draw,
            panel,
            x_ticks=tuple(float(value) for value in capacities),
            y_ticks=y_ticks,
            x_label="Cache capacity",
            y_label=y_label,
            x_max=float(max(capacities)),
            y_max=y_max,
        )
        x0, y0, x1, y1 = plot
        for _name, _label, color in selected:
            values = condition_results[_name]["capacities"]
            metric = (
                "cache_hit_rate"
                if y_max == 1.0
                else "independent_over_cache_median_speedup_ratio"
            )
            points = [
                (
                    x0 + int((x1 - x0) * float(item["capacity"]) / max(capacities)),
                    y1 - int((y1 - y0) * min(y_max, float(item[metric])) / y_max),
                )
                for item in values
            ]
            draw.line(points, fill=color, width=3, joint="curve")
            for px, py in points:
                draw.ellipse(
                    (px - 5, py - 5, px + 5, py + 5), fill=color, outline=PAGE, width=1
                )
        if y_max > 1.0:
            y_one = y1 - int((y1 - y0) * 1.0 / y_max)
            draw.line((x0, y_one, x1, y_one), fill=INK, width=1)
            draw.text(
                (x0 + 8, y_one - 21), "1.0 = equal latency", fill=INK, font=_font(12)
            )
    legend_x = 155
    for index, (_name, label, color) in enumerate(selected):
        x = legend_x + index * 245
        draw.line((x, 630, x + 28, 630), fill=color, width=4)
        draw.text((x + 36, 620), label, fill=INK, font=_font(13))
    recurring = condition_results["recurring_nonconsecutive_frames"]["capacities"]
    widest = max(recurring, key=lambda item: item["capacity"])
    unique = condition_results["frequent_widespread_changes"]["capacities"]
    smallest_speed = min(
        item["independent_over_cache_median_speedup_ratio"] for item in unique
    )
    draw.text(
        (58, 680),
        f"Recurring trace: capacity {widest['capacity']} reached "
        f"{widest['cache_hit_rate']:.1%} hits and "
        f"{widest['independent_over_cache_median_speedup_ratio']:.2f}x. "
        "All-unique trace: no hits; ratios stayed below 1 "
        f"({smallest_speed:.3f}x at the lowest point).",
        fill=INK,
        font=_font(14),
    )
    draw.text(
        (58, 730),
        "Payload includes resident tensor data only. Call reduction and "
        "wall-clock latency are distinct measures.",
        fill=MUTED,
        font=_font(14),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return destination


def create_research_figures(
    sources: Mapping[str, Any], figure_dir: str | Path
) -> dict[str, Path]:
    """Generate the five requested research figures from saved experiment data."""
    output = Path(figure_dir)
    return {
        "architecture": write_architecture_figure(output / "architecture.png"),
        "final_research_summary": write_final_summary_figure(
            sources, output / "final_research_summary.png"
        ),
        "accuracy_efficiency_tradeoff": write_accuracy_efficiency_figure(
            sources, output / "accuracy_efficiency_tradeoff.png"
        ),
        "pretrained_speedup": write_pretrained_speedup_figure(
            sources, output / "pretrained_speedup.png"
        ),
        "cache_capacity_summary": write_cache_capacity_figure(
            sources, output / "cache_capacity_summary.png"
        ),
    }


def _patch_changed_pixel_counts(
    mask: bytes, width: int, height: int, patch_size: int
) -> tuple[int, ...]:
    columns = (width + patch_size - 1) // patch_size
    rows = (height + patch_size - 1) // patch_size
    counts: list[int] = []
    for patch_y in range(rows):
        for patch_x in range(columns):
            counts.append(
                sum(
                    mask[y * width + x] > 0
                    for y in range(
                        patch_y * patch_size, min(height, (patch_y + 1) * patch_size)
                    )
                    for x in range(
                        patch_x * patch_size, min(width, (patch_x + 1) * patch_size)
                    )
                )
            )
    return tuple(counts)


def select_gallery_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    transformation_type: str | None = None,
    require_policy_miss: str | None = None,
    smallest_miss: bool = False,
) -> Mapping[str, Any]:
    """Select an existing transition deterministically for gallery rendering."""
    filtered = [
        item
        for item in candidates
        if (
            transformation_type is None
            or item["transformation_type"] == transformation_type
        )
        and (
            require_policy_miss is None
            or bool(item[f"{require_policy_miss}_missed_patch_indices"])
        )
    ]
    if not filtered:
        raise ValueError(
            "no validation transition satisfies the requested failure-gallery case"
        )
    if smallest_miss and require_policy_miss is not None:
        return min(
            filtered,
            key=lambda item: (
                min(
                    item["changed_pixel_counts"][patch]
                    for patch in item[f"{require_policy_miss}_missed_patch_indices"]
                ),
                item["episode_id"],
                item["transition_index"],
            ),
        )
    return min(
        filtered, key=lambda item: (item["episode_id"], item["transition_index"])
    )


def _scan_validation_transitions(
    manifest_path: str | Path, research_config_path: str | Path
) -> list[dict[str, Any]]:
    manifest = Path(manifest_path).resolve()
    if "validation" not in {part.lower() for part in manifest.parts}:
        raise ValueError("failure-gallery source manifest must be validation-only")
    if "test" in {part.lower() for part in manifest.parts}:
        raise ValueError("failure gallery must not read the held-out test partition")
    config = load_research_config(research_config_path)
    rows = _read_jsonl_rows(manifest)
    expected = config.splits["validation"]
    validate_final_manifest_provenance(
        manifest,
        rows,
        expected_partition="validation",
        expected_count=expected.episode_count,
        expected_seed_start=expected.base_seed,
    )
    episodes = load_manifest(manifest)
    if len(rows) != len(episodes):
        raise ValueError("validation manifest metadata and frame records differ")
    split_root = manifest.parent
    candidates: list[dict[str, Any]] = []
    for episode, row in zip(episodes, rows, strict=True):
        frames = tuple(read_ppm(frame.path) for frame in episode.frames)
        transitions = row.get("transitions")
        mask_paths = row.get("ground_truth_mask_paths")
        if not isinstance(transitions, list) or not isinstance(mask_paths, list):
            raise ValueError("validation manifest is missing transition masks")
        if len(transitions) != len(frames) - 1 or len(mask_paths) != len(frames) - 1:
            raise ValueError(
                "validation transition and mask counts do not match frames"
            )
        approximate = ThresholdPatchCache(0.01, patch_size=config.patch_size)
        fallback = GlobalFallbackPatchCache(
            local_threshold=0.05,
            global_change_threshold=0.50,
            patch_size=config.patch_size,
        )
        for frame_index, frame in enumerate(frames):
            approx_result = approximate.process_frame(frame)
            fallback_result = fallback.process_frame(frame)
            if frame_index == 0:
                continue
            transition_index = frame_index - 1
            raw_mask = mask_paths[transition_index]
            mask_path = _resolve_dataset_path(
                split_root, raw_mask, f"episode {episode.episode_id} mask"
            )
            mask_width, mask_height, mask_pixels = _read_pgm_mask(mask_path)
            if (mask_width, mask_height) != (frame.width, frame.height):
                raise ValueError("validation mask dimensions do not match RGB frames")
            if frame_index - 1 >= len(transitions):
                raise ValueError("manifest transition index is invalid")
            transformation = transitions[transition_index].get("transformation_type")
            changed_pixel_counts = _patch_changed_pixel_counts(
                mask_pixels, frame.width, frame.height, config.patch_size
            )
            labels = tuple(value > 0 for value in changed_pixel_counts)
            fallback_encoding = fallback_result.encoding
            candidates.append(
                {
                    "episode_id": episode.episode_id,
                    "random_seed": row["random_seed"],
                    "transition_index": transition_index,
                    "transformation_type": transformation,
                    "before": frames[frame_index - 1],
                    "after": frame,
                    "mask_pixels": mask_pixels,
                    "width": frame.width,
                    "height": frame.height,
                    "patch_size": config.patch_size,
                    "ground_truth_patch_labels": labels,
                    "changed_pixel_counts": changed_pixel_counts,
                    "approximate_refresh_decisions": approx_result.refresh_decisions,
                    "approximate_difference_scores": (
                        approx_result.mean_difference_scores
                    ),
                    "approximate_missed_patch_indices": tuple(
                        index
                        for index, (actual, predicted) in enumerate(
                            zip(labels, approx_result.refresh_decisions, strict=True)
                        )
                        if actual and not predicted
                    ),
                    "fallback_refresh_decisions": fallback_encoding.refresh_decisions,
                    "fallback_missed_patch_indices": tuple(
                        index
                        for index, (actual, predicted) in enumerate(
                            zip(
                                labels, fallback_encoding.refresh_decisions, strict=True
                            )
                        )
                        if actual and not predicted
                    ),
                    "fallback_global_change_score": fallback_result.global_change_score,
                    "fallback_global_change_detected": (
                        fallback_result.global_change_detected
                    ),
                    "global_changed_pixel_fraction": (
                        sum(mask_pixels) / (255 * len(mask_pixels))
                        if mask_pixels
                        else 0.0
                    ),
                }
            )
    return candidates


def _rgb_image(image: RGBImage, size: tuple[int, int]) -> Image.Image:
    return Image.frombytes("RGB", (image.width, image.height), image.pixels).resize(
        size, Image.Resampling.NEAREST
    )


def _mask_image(
    mask: bytes, width: int, height: int, size: tuple[int, int]
) -> Image.Image:
    return (
        Image.frombytes("L", (width, height), mask)
        .convert("RGB")
        .resize(size, Image.Resampling.NEAREST)
    )


def _decision_grid(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    decisions: Sequence[bool],
    labels: Sequence[bool],
    *,
    patch_size: int,
) -> None:
    left, top, right, bottom = box
    width = int(math.sqrt(len(decisions)))
    if width * width != len(decisions):
        width = max(1, round(math.sqrt(len(decisions))))
    rows = math.ceil(len(decisions) / width)
    cell_width = (right - left) / width
    cell_height = (bottom - top) / rows
    for index, decision in enumerate(decisions):
        x_index = index % width
        y_index = index // width
        x0 = round(left + x_index * cell_width)
        y0 = round(top + y_index * cell_height)
        x1 = round(left + (x_index + 1) * cell_width)
        y1 = round(top + (y_index + 1) * cell_height)
        color = ORANGE if labels[index] and not decision else RED if decision else GREEN
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=PAGE, width=2)
    draw.rectangle(box, outline=INK, width=2)
    draw.text(
        (left, bottom + 6),
        f"{width}x{rows} patches, {patch_size}px each",
        fill=MUTED,
        font=_font(12),
    )


def _render_failure_card(
    candidate: Mapping[str, Any],
    *,
    title: str,
    explanation: str,
    policy: str,
    path: str | Path,
    dino_summary: Mapping[str, Any] | None = None,
) -> Path:
    image, draw = _new_canvas(
        (1200, 690),
        title,
        "Validation synthetic sequence; masks encode rendered RGB changes, "
        "not semantic object labels.",
    )
    labels = candidate.get("ground_truth_patch_labels", ())
    width, height = int(candidate["width"]), int(candidate["height"])
    patch_size = int(candidate.get("patch_size", 16))
    boxes = (
        (30, 116, 310, 396),
        (327, 116, 607, 396),
        (624, 116, 904, 396),
        (921, 116, 1190, 396),
    )
    panel_titles = ("Previous frame", "Current frame", "RGB-difference mask", policy)
    for box, panel_title in zip(boxes, panel_titles, strict=True):
        _panel(draw, box, panel_title)
    image_size = (210, 210)
    for box, frame in zip(
        boxes[:2], (candidate["before"], candidate["after"]), strict=True
    ):
        rendered = _rgb_image(frame, image_size)
        image.paste(rendered, (box[0] + 35, box[1] + 60))
    mask_rendered = _mask_image(candidate["mask_pixels"], width, height, image_size)
    image.paste(mask_rendered, (boxes[2][0] + 35, boxes[2][1] + 60))
    draw.rectangle(
        (
            boxes[2][0] + 35,
            boxes[2][1] + 60,
            boxes[2][0] + 35 + image_size[0],
            boxes[2][1] + 60 + image_size[1],
        ),
        outline=INK,
        width=2,
    )
    if dino_summary is None:
        decisions = candidate[
            "approximate_refresh_decisions"
            if policy.startswith("Threshold")
            else "fallback_refresh_decisions"
        ]
        grid_box = (
            boxes[3][0] + 17,
            boxes[3][1] + 52,
            boxes[3][2] - 17,
            boxes[3][3] - 13,
        )
        _decision_grid(draw, grid_box, decisions, labels, patch_size=patch_size)
        missed = candidate[
            "approximate_missed_patch_indices"
            if policy.startswith("Threshold")
            else "fallback_missed_patch_indices"
        ]
        draw.text(
            (38, 434),
            f"Episode {candidate['episode_id']} | seed "
            f"{candidate['random_seed']} | transition "
            f"{candidate['transition_index']}",
            fill=INK,
            font=_font(14, bold=True),
        )
        draw.text(
            (38, 462),
            f"Transformation: {candidate['transformation_type']} | "
            f"missed changed patches: {len(missed)}",
            fill=INK,
            font=_font(14),
        )
        if candidate.get("fallback_global_change_score") is not None:
            draw.text(
                (38, 490),
                "Global changed-pixel fraction: "
                f"{candidate['fallback_global_change_score']:.3f}; fallback "
                f"triggered: {candidate['fallback_global_change_detected']}",
                fill=INK,
                font=_font(14),
            )
    else:
        draw.rounded_rectangle(
            (boxes[3][0] + 15, boxes[3][1] + 49, boxes[3][2] - 15, boxes[3][3] - 12),
            radius=10,
            fill="#f8f5fb",
            outline=PURPLE,
            width=2,
        )
        _draw_wrapped(
            draw,
            "Full DINOv2 forwards: "
            f"{dino_summary['independent_forward_calls_per_pass']} "
            "independent / "
            f"{dino_summary['stateful_forward_calls_per_pass']} cached. Hits: "
            f"{dino_summary['cached_frame_hits_per_pass']}. Cache hit rate: "
            f"{dino_summary['cache_hit_rate']:.1%}. "
            "Rendered RGB difference in shown pair: "
            f"{candidate['global_changed_pixel_fraction']:.1%}.",
            (boxes[3][0] + 29, boxes[3][1] + 76),
            width=27,
            font=_font(15),
            line_height=21,
        )
        draw.text(
            (boxes[3][0] + 29, boxes[3][1] + 222),
            f"Independent/cache ratio: {dino_summary['median_speedup']:.3f}x",
            fill=RED,
            font=_font(15, bold=True),
        )
        draw.text(
            (38, 434),
            f"Episode {candidate['episode_id']} | seed "
            f"{candidate.get('random_seed', 'validation')}",
            fill=INK,
            font=_font(14, bold=True),
        )
        draw.text(
            (38, 462),
            "Saved Task 17 DINOv2 validation output",
            fill=INK,
            font=_font(14),
        )
        draw.text(
            (38, 490),
            "The cache reuses the complete representation only when the "
            "preprocessed input matches exactly.",
            fill=INK,
            font=_font(14),
        )
        draw.text(
            (38, 518),
            "Peak resident tensor payload: "
            f"{dino_summary['cache_memory_peak_bytes'] / (1024 * 1024):.2f} MiB",
            fill=INK,
            font=_font(14),
        )
    _draw_wrapped(
        draw,
        explanation,
        (38, 560 if dino_summary is not None else 535),
        width=140,
        font=_font(15),
        fill=MUTED,
        line_height=22,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return destination


def _dino_frequency_candidate(
    validation_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(validation_root).resolve()
    if "validation" not in {part.lower() for part in root.parts} or "test" in {
        part.lower() for part in root.parts
    }:
        raise ValueError("DINO gallery input must be a validation-only root")
    manifest = root / "frequent_widespread_changes" / "manifest.jsonl"
    episodes = preload_validation_episodes(manifest)
    rows = _read_jsonl_rows(manifest)
    if not episodes or len(episodes[0][1]) < 2:
        raise ValueError("frequent-change validation input needs two frames")
    episode_id, frames = episodes[0]
    row = next(item for item in rows if item["episode_id"] == episode_id)
    before, after = frames[0], frames[1]
    mask = bytes(
        255
        if before.pixels[offset : offset + 3] != after.pixels[offset : offset + 3]
        else 0
        for offset in range(0, len(before.pixels), 3)
    )
    candidate = {
        "episode_id": episode_id,
        "random_seed": row["random_seed"],
        "transition_index": 0,
        "transformation_type": "global_brightness_sweep",
        "before": before,
        "after": after,
        "mask_pixels": mask,
        "width": after.width,
        "height": after.height,
        "patch_size": 14,
        "global_changed_pixel_fraction": sum(mask) / (255 * len(mask)),
    }
    return candidate, {"manifest": str(manifest)}


def create_failure_gallery(
    sources: Mapping[str, Any],
    *,
    validation_manifest: str | Path,
    research_config_path: str | Path,
    pretrained_validation_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build deterministic qualitative cards using generated validation data."""
    candidates = _scan_validation_transitions(validation_manifest, research_config_path)
    approx_miss = select_gallery_candidate(
        candidates, require_policy_miss="approximate", smallest_miss=True
    )
    brightness = select_gallery_candidate(
        candidates, transformation_type="global_brightness"
    )
    partial = select_gallery_candidate(
        candidates,
        transformation_type="partial_occlusion",
        require_policy_miss="fallback",
    )
    recoloring = select_gallery_candidate(
        candidates,
        transformation_type="recoloring",
        require_policy_miss="fallback",
    )
    frequent_candidate, _source = _dino_frequency_candidate(pretrained_validation_root)
    dino_summary = sources["final_evaluation"]["pretrained_dinov2"]["conditions"][
        "frequent_widespread_changes"
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    case_definitions = (
        (
            "approximate_small_change_miss",
            approx_miss,
            "Threshold reuse at 0.01",
            "The orange cells are pixel-change patches that the fixed 0.01 "
            "RGB-difference gate reused. This is a handcrafted feature "
            "staleness example; the mask records rendered pixel differences only.",
            "Patch decisions (0.01)",
            None,
        ),
        (
            "global_brightness_shift",
            brightness,
            "Global brightness change",
            "A broad intensity change can make nearly every RGB pixel differ. "
            "The pixel mask therefore marks a large area and the configured "
            "global-area detector may force a full handcrafted refresh; this "
            "is visual change, not a semantic judgment.",
            "Fallback decisions",
            None,
        ),
        (
            "partial_occlusion_global_fallback_miss",
            partial,
            "Partial occlusion under global fallback",
            "The saved validation analysis records missed changed patches for "
            "this policy category. The local 0.05 gate and 0.50 global "
            "changed-area trigger can both leave small-area changes "
            "under-refreshed.",
            "Global-fallback patch decisions",
            None,
        ),
        (
            "recoloring_global_fallback_miss",
            recoloring,
            "Recoloring under global fallback",
            "Recoloring changes RGB statistics without adding semantic ground "
            "truth. In validation, the local 0.05/global 0.50 policy missed "
            "some pixel-changed patches in this category.",
            "Global-fallback patch decisions",
            None,
        ),
        (
            "frequent_widespread_changes",
            frequent_candidate,
            "Frequent widespread DINOv2 changes",
            "All preprocessed frames were unique in the saved validation "
            "condition. The exact whole-frame cache had no hit and retained "
            "preprocessing, lookup, equality, and storage overhead.",
            "DINOv2 cache result",
            dino_summary,
        ),
        (
            "cache_no_speed_advantage",
            frequent_candidate,
            "When caching provides no speed advantage",
            "On this frequent-change validation condition, the cache made every "
            "complete model forward and measured slower than independent "
            "inference. Cache-hit rate is not a speedup; capacity cannot reuse "
            "different preprocessed tensors.",
            "DINOv2 cache result",
            dino_summary,
        ),
    )
    cards: list[dict[str, Any]] = []
    for name, candidate, title, explanation, policy, pretrained in case_definitions:
        image_path = output / f"{name}.png"
        _render_failure_card(
            candidate,
            title=title,
            explanation=explanation,
            policy=policy,
            path=image_path,
            dino_summary=pretrained,
        )
        cards.append(
            {
                "case": name,
                "image": image_path.name,
                "partition": "synthetic_validation",
                "episode_id": candidate["episode_id"],
                "random_seed": candidate["random_seed"],
                "transition_index": candidate["transition_index"],
                "transformation_type": candidate["transformation_type"],
                "source_manifest": (
                    _repository_relative_path(
                        Path(pretrained_validation_root)
                        / "frequent_widespread_changes"
                        / "manifest.jsonl"
                    )
                    if pretrained is not None
                    else _repository_relative_path(validation_manifest)
                ),
                "approximate_missed_patch_indices": list(
                    candidate.get("approximate_missed_patch_indices", ())
                ),
                "fallback_missed_patch_indices": list(
                    candidate.get("fallback_missed_patch_indices", ())
                ),
                "description": explanation,
            }
        )
    index_path = output / "index.md"
    index_lines = [
        "# DeltaVision synthetic validation failure gallery",
        "",
        "These deterministic examples come from the synthetic validation "
        "manifests. Masks show rendered RGB changes, not semantic annotations. "
        "The aggregate held-out RGB results are reported separately and the "
        "held-out partition is known to have been inspected in earlier work.",
        "",
    ]
    for card in cards:
        index_lines.extend(
            [
                f"## {card['case'].replace('_', ' ').title()}",
                "",
                f"![{card['case']}](./{card['image']})",
                "",
                card["description"],
                "",
                f"Source: validation episode `{card['episode_id']}`, seed "
                f"`{card['random_seed']}`, transition "
                f"`{card['transition_index']}` "
                f"({card['transformation_type']}).",
                "",
            ]
        )
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    manifest_path = output / "gallery_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "partition": "synthetic_validation",
                "mask_interpretation": (
                    "rendered RGB pixel differences; not semantic labels"
                ),
                "source_global_validation_report": sources["source_paths"][
                    "global_validation"
                ],
                "source_pretrained_evaluation": sources["source_paths"][
                    "final_evaluation"
                ],
                "cases": cards,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"cases": cards, "index": index_path, "manifest": manifest_path}


def generate_task18_assets(
    *,
    final_evaluation_path: str | Path,
    final_statistics_path: str | Path,
    capacity_ablation_path: str | Path,
    global_validation_path: str | Path,
    research_config_path: str | Path,
    validation_manifest: str | Path,
    pretrained_validation_root: str | Path,
    docs_root: str | Path,
) -> dict[str, Any]:
    """Create figures and validation gallery without loading model weights."""
    sources = load_visual_sources(
        final_evaluation_path=final_evaluation_path,
        final_statistics_path=final_statistics_path,
        capacity_ablation_path=capacity_ablation_path,
        global_validation_path=global_validation_path,
    )
    docs = Path(docs_root)
    figures = create_research_figures(sources, docs / "figures")
    gallery = create_failure_gallery(
        sources,
        validation_manifest=validation_manifest,
        research_config_path=research_config_path,
        pretrained_validation_root=pretrained_validation_root,
        output_dir=docs / "failure_gallery",
    )
    return {"sources": sources["source_paths"], "figures": figures, "gallery": gallery}
