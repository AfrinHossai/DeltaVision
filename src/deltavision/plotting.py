"""Tiny standard-library PNG chart renderer for reproducible sweep artifacts."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Sequence

_FONT: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
}

_INK = (38, 48, 60)
_GRID = (222, 227, 232)
_REFRESH = (43, 112, 176)
_ERROR = (218, 114, 40)


class _Canvas:
    """Minimal RGB raster canvas with lines and a compact bitmap font."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray((255, 255, 255) * width * height)

    def pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(color)

    def line(
        self,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        dx = abs(x_end - x_start)
        sx = 1 if x_start < x_end else -1
        dy = -abs(y_end - y_start)
        sy = 1 if y_start < y_end else -1
        error = dx + dy
        while True:
            radius = max(0, thickness // 2)
            for y_offset in range(-radius, radius + 1):
                for x_offset in range(-radius, radius + 1):
                    self.pixel(x_start + x_offset, y_start + y_offset, color)
            if x_start == x_end and y_start == y_end:
                break
            double_error = 2 * error
            if double_error >= dy:
                error += dy
                x_start += sx
            if double_error <= dx:
                error += dx
                y_start += sy

    def marker(
        self, x: int, y: int, color: tuple[int, int, int], radius: int = 5
    ) -> None:
        for y_offset in range(-radius, radius + 1):
            for x_offset in range(-radius, radius + 1):
                if x_offset * x_offset + y_offset * y_offset <= radius * radius:
                    self.pixel(x + x_offset, y + y_offset, color)

    def text(
        self,
        value: str,
        x: int,
        y: int,
        color: tuple[int, int, int] = _INK,
        scale: int = 1,
    ) -> None:
        cursor = x
        for character in value.upper():
            glyph = _FONT.get(character)
            if glyph is not None:
                for row, bits in enumerate(glyph):
                    for column, bit in enumerate(bits):
                        if bit == "1":
                            for y_offset in range(scale):
                                for x_offset in range(scale):
                                    self.pixel(
                                        cursor + column * scale + x_offset,
                                        y + row * scale + y_offset,
                                        color,
                                    )
            cursor += 6 * scale


def write_threshold_tradeoff_png(
    results: Sequence[dict[str, object]], path: str | Path
) -> Path:
    """Render refresh fraction and mean feature error against threshold.

    The chart uses only Python's standard library, so generating sweep figures
    does not add a plotting dependency to the package.
    """
    if not results:
        raise ValueError("at least one threshold result is required to plot")
    points = sorted(results, key=lambda row: float(row["threshold"]))
    thresholds = [float(row["threshold"]) for row in points]
    refresh_values = [float(row["refresh_fraction"]) for row in points]
    error_values = [float(row["mean_absolute_feature_error"]) for row in points]
    threshold_maximum = max(thresholds)
    if threshold_maximum <= 0:
        threshold_maximum = 1.0
    error_maximum = max(error_values)
    error_limit = error_maximum * 1.15 if error_maximum > 0 else 0.01

    canvas = _Canvas(1100, 760)
    canvas.text("SYNTHETIC-BENCHMARK FINDINGS", 50, 28, scale=2)
    canvas.text("THRESHOLD REUSE TRADE-OFF", 50, 58, color=(96, 107, 119))
    canvas.text("REFRESH FRACTION", 52, 91)
    canvas.text("MEAN ABSOLUTE FEATURE ERROR", 52, 378)

    left, right = 135, 1055
    top_panel = (124, 302)
    bottom_panel = (416, 614)
    _draw_panel(canvas, left, right, *top_panel, maximum=1.0, tick_format=".2f")
    _draw_panel(
        canvas,
        left,
        right,
        *bottom_panel,
        maximum=error_limit,
        tick_format=".4f",
    )

    def x_position(threshold: float) -> int:
        return left + round((right - left) * threshold / threshold_maximum)

    def y_position(value: float, panel: tuple[int, int], maximum: float) -> int:
        panel_top, panel_bottom = panel
        return panel_bottom - round((panel_bottom - panel_top) * value / maximum)

    for values, color, panel, maximum in (
        (refresh_values, _REFRESH, top_panel, 1.0),
        (error_values, _ERROR, bottom_panel, error_limit),
    ):
        coordinates = [
            (x_position(threshold), y_position(value, panel, maximum))
            for threshold, value in zip(thresholds, values, strict=True)
        ]
        for first, second in zip(coordinates, coordinates[1:]):
            canvas.line(*first, *second, color, thickness=3)
        for x, y in coordinates:
            canvas.marker(x, y, color)

    for threshold in thresholds:
        x = x_position(threshold)
        canvas.line(x, top_panel[1], x, top_panel[1] + 6, _INK)
        canvas.line(x, bottom_panel[1], x, bottom_panel[1] + 6, _INK)
        label = f"{threshold:.2f}"
        canvas.text(label, x - len(label) * 3, bottom_panel[1] + 13)
    canvas.text("MEAN ABSOLUTE RGB DIFFERENCE THRESHOLD", 390, 684)

    output = Path(path)
    output.write_bytes(_encode_png(canvas))
    return output


def write_change_detection_png(
    results: Sequence[dict[str, object]], path: str | Path
) -> Path:
    """Plot transition refresh fraction and missed-change rate by threshold."""
    if not results:
        raise ValueError("at least one change-detection result is required to plot")
    points = sorted(results, key=lambda row: float(row["threshold"]))
    thresholds = [float(row["threshold"]) for row in points]
    detection_rows = [row["change_detection"] for row in points]
    refresh_values = [float(row["refresh_fraction"]) for row in detection_rows]
    missed_values = [float(row["missed_change_rate"]) for row in detection_rows]
    threshold_maximum = max(thresholds)
    if threshold_maximum <= 0:
        threshold_maximum = 1.0

    canvas = _Canvas(1100, 760)
    canvas.text("SYNTHETIC-BENCHMARK FINDINGS", 50, 28, scale=2)
    canvas.text("PATCH REFRESH VS MISSED CHANGE", 50, 58, color=(96, 107, 119))
    canvas.text("TRANSITION REFRESH FRACTION", 52, 91)
    canvas.text("MISSED-CHANGE RATE", 52, 378)

    left, right = 135, 1055
    top_panel = (124, 302)
    bottom_panel = (416, 614)
    _draw_panel(canvas, left, right, *top_panel, maximum=1.0, tick_format=".2f")
    _draw_panel(canvas, left, right, *bottom_panel, maximum=1.0, tick_format=".2f")

    def x_position(threshold: float) -> int:
        return left + round((right - left) * threshold / threshold_maximum)

    def y_position(value: float, panel: tuple[int, int]) -> int:
        panel_top, panel_bottom = panel
        return panel_bottom - round((panel_bottom - panel_top) * value)

    for values, color, panel in (
        (refresh_values, _REFRESH, top_panel),
        (missed_values, _ERROR, bottom_panel),
    ):
        coordinates = [
            (x_position(threshold), y_position(value, panel))
            for threshold, value in zip(thresholds, values, strict=True)
        ]
        for first, second in zip(coordinates, coordinates[1:]):
            canvas.line(*first, *second, color, thickness=3)
        for x, y in coordinates:
            canvas.marker(x, y, color)

    for threshold in thresholds:
        x = x_position(threshold)
        canvas.line(x, top_panel[1], x, top_panel[1] + 6, _INK)
        canvas.line(x, bottom_panel[1], x, bottom_panel[1] + 6, _INK)
        label = f"{threshold:.2f}"
        canvas.text(label, x - len(label) * 3, bottom_panel[1] + 13)
    canvas.text("RGB DIFFERENCE THRESHOLD", 450, 684)

    output = Path(path)
    output.write_bytes(_encode_png(canvas))
    return output


def write_global_change_comparison_png(
    methods: Sequence[dict[str, object]], path: str | Path
) -> Path:
    """Plot recall, transition refresh, and full-refresh frequency by policy."""
    if not methods:
        raise ValueError("at least one method result is required to plot")

    rows = sorted(methods, key=lambda row: str(row["method"]))
    method_names = [str(row["method"]) for row in rows]
    labels = [
        "EXACT"
        if name == "exact_reuse"
        else (
            "T 0.05 GLOBAL"
            if name.endswith("global_fallback")
            else f"T {name.rsplit('_', 1)[-1]}"
        )
        for name in method_names
    ]
    aggregates = [row["aggregate"] for row in rows]
    detection_rows = [row["change_detection"] for row in aggregates]
    recall_values = [float(row["recall"]) for row in detection_rows]
    refresh_values = [float(row["refresh_fraction"]) for row in detection_rows]
    full_refresh_values = [
        float(row["full_frame_refresh_frequency"]) for row in detection_rows
    ]

    canvas = _Canvas(1100, 930)
    canvas.text("VALIDATION ONLY", 48, 25, scale=2)
    canvas.text("GLOBAL SCENE-CHANGE FALLBACK COMPARISON", 48, 57, color=(96, 107, 119))
    canvas.text(
        "BLUE RECALL   ORANGE TRANSITION REFRESH   GREEN FULL-FRAME REFRESH", 48, 91
    )

    left, right = 160, 1045
    panels = ((142, 310), (382, 550), (622, 790))
    panel_titles = (
        "PATCH-LEVEL RECALL",
        "TRANSITION REFRESH FRACTION",
        "FULL-FRAME REFRESH FREQUENCY",
    )
    for title, (top, bottom) in zip(panel_titles, panels, strict=True):
        canvas.text(title, 52, top - 24)
        _draw_panel(canvas, left, right, top, bottom, maximum=1.0, tick_format=".2f")

    if len(rows) == 1:
        x_positions = [(left + right) // 2]
    else:
        x_positions = [
            left + round((right - left) * index / (len(rows) - 1))
            for index in range(len(rows))
        ]
    series = (
        (recall_values, _REFRESH, panels[0]),
        (refresh_values, _ERROR, panels[1]),
        (full_refresh_values, (57, 145, 92), panels[2]),
    )
    for values, color, (top, bottom) in series:
        points = [
            (x, bottom - round((bottom - top) * value))
            for x, value in zip(x_positions, values, strict=True)
        ]
        for first, second in zip(points, points[1:]):
            canvas.line(*first, *second, color, thickness=3)
        for point in points:
            canvas.marker(*point, color)

    for x, label in zip(x_positions, labels, strict=True):
        canvas.text(label, x - len(label) * 3, 824)
    canvas.text("METHOD", 535, 866)
    output = Path(path)
    output.write_bytes(_encode_png(canvas))
    return output


def write_latency_comparison_png(
    policies: dict[str, dict[str, object]], path: str | Path
) -> Path:
    """Render validation median CPU latency and paired median speedup."""
    if not policies:
        raise ValueError("at least one policy result is required to plot")
    labels_by_policy = {
        "independent_frame_encoding": "INDEPENDENT",
        "exact_patch_reuse": "EXACT REUSE",
        "approximate_reuse_0.01": "THRESHOLD 0.01",
        "approximate_reuse_0.05": "THRESHOLD 0.05",
        "approximate_reuse_0.05_global_fallback": "T 0.05 GLOBAL",
    }
    preferred_order = (
        "independent_frame_encoding",
        "exact_patch_reuse",
        "approximate_reuse_0.01",
        "approximate_reuse_0.05",
        "approximate_reuse_0.05_global_fallback",
    )
    policy_names = [name for name in preferred_order if name in policies]
    policy_names.extend(name for name in policies if name not in preferred_order)
    labels = [labels_by_policy.get(name, name.upper()) for name in policy_names]
    latency_ms = [
        float(policies[name]["median_latency_seconds"]) * 1000 for name in policy_names
    ]
    speedups = [
        float(policies[name]["median_speedup_relative_to_independent"])
        for name in policy_names
    ]
    latency_maximum = max(latency_ms) * 1.15 if max(latency_ms) > 0 else 1.0
    speedup_maximum = max(speedups) * 1.15 if max(speedups) > 0 else 1.0
    colors = (
        (82, 94, 108),
        (66, 133, 186),
        (75, 155, 120),
        (218, 145, 57),
        (178, 83, 89),
    )

    canvas = _Canvas(1200, 790)
    canvas.text("SYNTHETIC VALIDATION CPU LATENCY", 48, 28, scale=2)
    canvas.text(
        "SAME PRELOADED WORKLOAD, RANDOMIZED POLICY ORDER", 48, 61, color=(96, 107, 119)
    )
    canvas.text("MEDIAN END-TO-END IN-MEMORY PROCESSING (MS)", 52, 102)
    canvas.text("PAIRED MEDIAN SPEEDUP VS INDEPENDENT ENCODING", 52, 410)

    left, right = 155, 1150
    top_panel = (138, 340)
    bottom_panel = (446, 650)
    _draw_panel(
        canvas,
        left,
        right,
        *top_panel,
        maximum=latency_maximum,
        tick_format=".1f",
    )
    _draw_panel(
        canvas,
        left,
        right,
        *bottom_panel,
        maximum=speedup_maximum,
        tick_format=".2f",
    )
    if len(policy_names) == 1:
        x_positions = [(left + right) // 2]
    else:
        x_positions = [
            left + round((right - left) * index / (len(policy_names) - 1))
            for index in range(len(policy_names))
        ]
    for x, latency, speedup, label, color in zip(
        x_positions, latency_ms, speedups, labels, colors, strict=True
    ):
        latency_y = top_panel[1] - round(
            (top_panel[1] - top_panel[0]) * latency / latency_maximum
        )
        speedup_y = bottom_panel[1] - round(
            (bottom_panel[1] - bottom_panel[0]) * speedup / speedup_maximum
        )
        canvas.line(x, top_panel[1], x, latency_y, color, thickness=36)
        canvas.marker(x, latency_y, color, radius=4)
        canvas.line(x, bottom_panel[1], x, speedup_y, color, thickness=36)
        canvas.marker(x, speedup_y, color, radius=4)
        canvas.text(label, x - len(label) * 3, 682)
    canvas.text("POLICY", 585, 738)
    output = Path(path)
    output.write_bytes(_encode_png(canvas))
    return output


def write_ablation_patch_size_png(
    results: Sequence[dict[str, object]], path: str | Path
) -> Path:
    """Plot median full-pass latency over patch size, faceted by change rate."""
    if not results:
        raise ValueError("at least one ablation result is required to plot")
    condition_order = (
        "no_visual_changes",
        "sparse_local_changes",
        "moderate_local_changes",
        "frequent_widespread_changes",
    )
    condition_labels = {
        "no_visual_changes": "NO VISUAL CHANGES",
        "sparse_local_changes": "SPARSE LOCAL CHANGES",
        "moderate_local_changes": "MODERATE LOCAL CHANGES",
        "frequent_widespread_changes": "FREQUENT WIDESPREAD CHANGES",
    }
    method_order = (
        "independent_frame_encoding",
        "exact_patch_reuse",
        "approximate_reuse_0.01",
    )
    method_labels = {
        "independent_frame_encoding": "INDEPENDENT",
        "exact_patch_reuse": "EXACT REUSE",
        "approximate_reuse_0.01": "APPROXIMATE 0.01",
    }
    colors = {
        "independent_frame_encoding": (82, 94, 108),
        "exact_patch_reuse": (66, 133, 186),
        "approximate_reuse_0.01": (218, 145, 57),
    }
    indexed = {
        (str(row["condition"]), str(row["method"]), int(row["patch_size"])): (
            float(row["median_processing_latency_seconds"]) * 1000
        )
        for row in results
    }
    patch_sizes = sorted({int(row["patch_size"]) for row in results})
    if set(condition_order) - {condition for condition, _, _ in indexed}:
        raise ValueError("ablation results are missing one or more conditions")
    maximum_latency = max(indexed.values())
    latency_limit = maximum_latency * 1.15 if maximum_latency > 0 else 1.0

    canvas = _Canvas(1400, 950)
    canvas.text("CPU SYNTHETIC ABLATION FINDINGS", 42, 24, scale=2)
    canvas.text("MEDIAN FULL WORKLOAD LATENCY BY PATCH GRANULARITY", 46, 64)
    canvas.text("PATCH SIZE IN PIXELS", 46, 105)
    legend_x = 360
    for method in method_order:
        color = colors[method]
        canvas.line(legend_x, 111, legend_x + 22, 111, color, thickness=3)
        canvas.marker(legend_x + 11, 111, color, radius=3)
        canvas.text(method_labels[method], legend_x + 30, 107)
        legend_x += 265

    panels = (
        (116, 660, 171, 397),
        (752, 1296, 171, 397),
        (116, 660, 566, 792),
        (752, 1296, 566, 792),
    )
    for condition, (axis_left, axis_right, top, bottom) in zip(
        condition_order, panels, strict=True
    ):
        canvas.text(condition_labels[condition], axis_left, top - 31)
        _draw_panel(
            canvas,
            axis_left + 48,
            axis_right,
            top,
            bottom,
            maximum=latency_limit,
            tick_format=".0f",
        )
        if len(patch_sizes) == 1:
            x_positions = [(axis_left + 48 + axis_right) // 2]
        else:
            x_positions = [
                axis_left
                + 48
                + round(
                    (axis_right - (axis_left + 48)) * index / (len(patch_sizes) - 1)
                )
                for index in range(len(patch_sizes))
            ]
        for method in method_order:
            values = [indexed[(condition, method, size)] for size in patch_sizes]
            points = [
                (
                    x,
                    bottom - round((bottom - top) * value / latency_limit),
                )
                for x, value in zip(x_positions, values, strict=True)
            ]
            for first, second in zip(points, points[1:]):
                canvas.line(*first, *second, colors[method], thickness=3)
            for point in points:
                canvas.marker(*point, colors[method], radius=4)
        for x, patch_size in zip(x_positions, patch_sizes, strict=True):
            canvas.text(str(patch_size), x - 3, bottom + 12)

    canvas.text("PATCH SIZE", 633, 904)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_encode_png(canvas))
    return output


def write_ablation_change_frequency_png(
    results: Sequence[dict[str, object]], path: str | Path
) -> Path:
    """Plot cache-to-independent median latency ratios by change condition."""
    if not results:
        raise ValueError("at least one ablation result is required to plot")
    patch_sizes = sorted({int(row["patch_size"]) for row in results})
    condition_order = (
        "no_visual_changes",
        "sparse_local_changes",
        "moderate_local_changes",
        "frequent_widespread_changes",
    )
    condition_labels = ("NO CHANGE", "SPARSE", "MODERATE", "WIDESPREAD")
    method_order = ("exact_patch_reuse", "approximate_reuse_0.01")
    method_labels = {
        "exact_patch_reuse": "EXACT REUSE",
        "approximate_reuse_0.01": "APPROXIMATE 0.01",
    }
    colors = {
        "exact_patch_reuse": (66, 133, 186),
        "approximate_reuse_0.01": (218, 145, 57),
    }
    indexed = {
        (str(row["condition"]), str(row["method"]), int(row["patch_size"])): float(
            row["median_latency_ratio_to_independent"]
        )
        for row in results
        if row["method"] in method_order
    }
    required_conditions = set(condition_order)
    if required_conditions - {condition for condition, _, _ in indexed}:
        raise ValueError("ablation results are missing one or more conditions")
    maximum_ratio = max(indexed.values())
    ratio_limit = max(1.2, maximum_ratio * 1.15)

    canvas = _Canvas(1400, 760)
    canvas.text("CPU SYNTHETIC ABLATION FINDINGS", 42, 22, scale=2)
    canvas.text("MEDIAN LATENCY RATIO: BELOW 1 IS FASTER THAN INDEPENDENT", 46, 61)
    legend_x = 400
    for method in method_order:
        color = colors[method]
        canvas.line(legend_x, 101, legend_x + 22, 101, color, thickness=3)
        canvas.marker(legend_x + 11, 101, color, radius=3)
        canvas.text(method_labels[method], legend_x + 30, 97)
        legend_x += 310

    plot_left = 95
    panel_width = 370
    gap = 65
    top, bottom = 172, 505
    for patch_index, patch_size in enumerate(patch_sizes):
        axis_left = plot_left + patch_index * (panel_width + gap)
        axis_right = axis_left + panel_width
        canvas.text(f"PATCH SIZE {patch_size}", axis_left, top - 31)
        _draw_panel(
            canvas,
            axis_left + 42,
            axis_right,
            top,
            bottom,
            maximum=ratio_limit,
            tick_format=".2f",
        )
        reference_y = bottom - round((bottom - top) / ratio_limit)
        canvas.line(axis_left + 42, reference_y, axis_right, reference_y, _INK)
        x_positions = [
            axis_left
            + 42
            + round((axis_right - axis_left - 42) * index / (len(condition_order) - 1))
            for index in range(len(condition_order))
        ]
        for method in method_order:
            values = [
                indexed[(condition, method, patch_size)]
                for condition in condition_order
            ]
            points = [
                (
                    x,
                    bottom - round((bottom - top) * value / ratio_limit),
                )
                for x, value in zip(x_positions, values, strict=True)
            ]
            for first, second in zip(points, points[1:]):
                canvas.line(*first, *second, colors[method], thickness=3)
            for point in points:
                canvas.marker(*point, colors[method], radius=4)
        for x, label in zip(x_positions, condition_labels, strict=True):
            canvas.text(label, x - len(label) * 3, bottom + 12)
    canvas.text("TEMPORAL CHANGE CONDITION", 545, 692)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_encode_png(canvas))
    return output


def _draw_panel(
    canvas: _Canvas,
    left: int,
    right: int,
    top: int,
    bottom: int,
    *,
    maximum: float,
    tick_format: str,
) -> None:
    for tick_index in range(5):
        fraction = tick_index / 4
        y = bottom - round((bottom - top) * fraction)
        canvas.line(left, y, right, y, _GRID)
        label = format(maximum * fraction, tick_format)
        canvas.text(label, left - len(label) * 6 - 12, y - 3)
    canvas.line(left, top, left, bottom, _INK, thickness=2)
    canvas.line(left, bottom, right, bottom, _INK, thickness=2)


def _encode_png(canvas: _Canvas) -> bytes:
    scanlines = bytearray()
    stride = canvas.width * 3
    for row in range(canvas.height):
        scanlines.append(0)
        start = row * stride
        scanlines.extend(canvas.pixels[start : start + stride])
    image_header = struct.pack(">2I5B", canvas.width, canvas.height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", image_header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", checksum)
