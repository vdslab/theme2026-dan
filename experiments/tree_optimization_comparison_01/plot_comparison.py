"""Render the selected layouts in a common orthographic projection."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


AZIMUTH = -60
ELEVATION = 34
Z_PER_LAYER = 6 / 40
BACKGROUND = "#131923"


def projection():
    azimuth, elevation = np.radians([AZIMUTH, ELEVATION])
    sa, ca = np.sin(azimuth), np.cos(azimuth)
    se, ce = np.sin(elevation), np.cos(elevation)
    return np.array(
        [[-sa, -ca * se, ca * ce], [ca, -sa * se, sa * ce], [0, ce, se]]
    )


def project(xy, layers, matrix):
    return np.column_stack((xy, np.broadcast_to(layers, len(xy)) * Z_PER_LAYER)) @ matrix


def read_layout(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def common_limits(layouts, matrix, focus=None):
    extent = []
    for _, layout in layouts:
        xy = {node["id"]: (node["x"], node["y"]) for node in layout["nodes"]}
        for edge in layout["hyperedges"]:
            if focus is not None and focus not in edge["nodes"]:
                continue
            if edge["nodes"]:
                points = [xy[node] for node in edge["nodes"]]
                extent.append(project(points, edge["layer"], matrix)[:, :2])
    bounds = np.vstack(extent)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    padding = 0.05 * max(high - low)
    return low - padding, high + padding


def make_figures(representatives, output_dir):
    """Use the same framing for each condition, including the initial layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layouts = [(title, read_layout(path)) for title, path in representatives]
    members = [set(edge["nodes"]) for edge in layouts[0][1]["hyperedges"]]
    focus_counts = defaultdict(int)
    for edge in members:
        if len(edge) >= 3:
            for node in edge:
                focus_counts[node] += 1
    focus_id = max(focus_counts, key=lambda node: (focus_counts[node], -node))
    matrix = projection()
    focus_label = next(
        node["label"] for node in layouts[0][1]["nodes"] if node["id"] == focus_id
    )
    columns = min(len(layouts), 3 if len(layouts) <= 6 else 4)
    rows = (len(layouts) + columns - 1) // columns
    results = []
    for mode, filename in (
        ("overview", "layout_comparison.png"),
        ("shared vertex", "shared_vertex_comparison.png"),
    ):
        figure, axes = plt.subplots(
            rows, columns, figsize=(4.7 * columns, 3.9 * rows), squeeze=False,
            facecolor=BACKGROUND, layout="constrained",
        )
        focus = focus_id if mode == "shared vertex" else None
        low, high = common_limits(layouts, matrix, focus)
        if mode == "overview":
            figure.suptitle("Full layout | Polygon outlines and shared-vertex pillars", color="white")
        else:
            figure.suptitle(
                f"Shared vertex | {focus_label} (ID {focus_id}), "
                f"{focus_counts[focus_id]} incident polygons",
                color="white",
            )
        for ax, (title, layout) in zip(axes.flat, layouts):
            focus_span = draw_layout(ax, layout, matrix, focus)
            subtitle = f" | vertex span {focus_span:.0f}" if focus is not None else ""
            ax.set_title(title + subtitle, fontsize=9, color="#e4edf8", pad=2)
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])
            ax.set_aspect("equal")
            ax.set_axis_off()
            if focus is not None:
                results.append({"panel": title, "vertex_span": focus_span})
        for ax in list(axes.flat)[len(layouts):]:
            ax.set_axis_off()
        figure.savefig(output_dir / filename, dpi=170, facecolor=BACKGROUND)
        plt.close(figure)
    return {
        "focus_vertex_id": focus_id,
        "focus_vertex_label": focus_label,
        "incident_polygons": focus_counts[focus_id],
        "panels": results,
    }


def draw_layout(ax, layout, matrix, focus):
    from matplotlib.collections import LineCollection, PolyCollection

    xy = {node["id"]: (node["x"], node["y"]) for node in layout["nodes"]}
    levels = defaultdict(set)
    faces = []
    segments = []
    for edge in layout["hyperedges"]:
        z = edge["layer"]
        for node in edge["nodes"]:
            levels[node].add(z)
        if len(edge["nodes"]) < 2:
            continue
        points = project([xy[node] for node in edge["nodes"]], z, matrix)
        red, green, blue = (edge["color"][channel] / 255 for channel in ("r", "g", "b"))
        highlighted = focus is not None and focus in edge["nodes"]
        faded = focus is not None and not highlighted
        rgba = (red, green, blue, 0.22 if faded else 0.92)
        record = (float(points[:, 2].mean()), points[:, :2], rgba, 0.035 if faded else 0.15)
        (faces if len(points) >= 3 else segments).append(record)

    pillars = [
        project([xy[node], xy[node]], [min(zs), max(zs)], matrix)[:, :2]
        for node, zs in levels.items()
        if len(zs) >= 2
    ]
    ax.add_collection(
        LineCollection(
            pillars, colors="#a4bfda", linewidths=0.38,
            alpha=0.13 if focus is not None else 0.3,
        )
    )
    faces.sort(key=lambda record: record[0])
    ax.add_collection(
        PolyCollection(
            [points for _, points, _, _ in faces],
            facecolors=[(*color[:3], fill) for _, _, color, fill in faces],
            edgecolors=[color for _, _, color, _ in faces],
            linewidths=0.55,
        )
    )
    ax.add_collection(
        LineCollection(
            [points for _, points, _, _ in segments],
            colors=[color for _, _, color, _ in segments], linewidths=0.5,
        )
    )
    if focus is None:
        return None
    heights = sorted(levels[focus])
    markers = project([xy[focus]] * len(heights), heights, matrix)[:, :2]
    ax.plot(markers[:, 0], markers[:, 1], color="#ffe176", lw=1.2, zorder=4)
    ax.scatter(markers[:, 0], markers[:, 1], s=9, c="#ffe176", edgecolors=BACKGROUND,
               linewidths=0.4, zorder=5)
    return heights[-1] - heights[0]
