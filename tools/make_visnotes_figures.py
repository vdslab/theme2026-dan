"""Build local VisNotes figures from checked layouts and experiment summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
import json
from math import cos, radians, sin
from pathlib import Path
from statistics import median
import sys

import matplotlib
import numpy as np
from shapely.geometry import Polygon

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from experiments.tree_optimization_comparison_01.review_results import file_hash, make_geometry, read_json
from experiments.visnotes_metrics import read_layout_metrics

AZIMUTH = 45.0
ELEVATION = 35.0
REGULARIZATION = 1e-5
FOCUS_COLOR = "#bf5326"
INK = "#18323f"
MUTED = "#506875"
VARIANTS = ("original", "paper", "scalable", "arranged")
METHOD_NAMES = {
    "greedy-input": "First-fit",
    "greedy-size": "First-fit · size",
    "greedy-degree": "First-fit · degree",
    "dsatur": "DSatur",
    "dsatur-free": "DSatur + free",
    "qp-l2": "QP",
    "lp-l1": "LP",
    "tree-sa": "Tree SA",
    "swap": "Swap",
    "swap-move": "+ Move",
    "swap-move-kempe": "+ Kempe",
    "full": "+ Shift",
}
OBJECTIVE_LABELS = {
    "pillar": "$B$ only", "squared": "$F_{\\mathrm{scaled}}$ only",
    "degree-span": "Degree-weighted spans",
    "hybrid-0.1": "$\\alpha=0.1$", "hybrid-0.25": "$\\alpha=0.25$",
    "hybrid-0.5": "$\\alpha=0.5$", "hybrid-1": "$\\alpha=1$",
}


@dataclass(frozen=True)
class LocalView:
    edge_ids: tuple
    rings: tuple
    kinds: tuple
    layer_step: float
    colors: tuple


def first_fit_levels(geometry):
    """Check the unmodified input-order first-fit rule independently."""
    earlier_neighbors = [[] for _ in range(geometry.size)]
    for left, right in geometry.overlaps:
        earlier_neighbors[right].append(left)
    levels = []
    for neighbors in earlier_neighbors:
        used = {levels[index] for index in neighbors}
        level = 0
        while level in used:
            level += 1
        levels.append(level)
    return tuple(levels)


def local_view(geometry, metrics):
    """Freeze the subset using constrained incidence, before considering heights."""
    if not metrics["constrained_hubs"]:
        raise ValueError("The input has no shared vertex in at least two constrained edges")
    hub = metrics["constrained_hubs"][0]
    node_id = hub["node_id"]
    nodes = {node["id"]: node for node in geometry.data["nodes"]}
    xy = {key: (node["x"], node["y"]) for key, node in nodes.items()}
    focus = np.array(xy[node_id])
    domain = np.ptp(np.array(list(xy.values())), axis=0).max()
    if not np.isfinite(domain) or domain <= 0:
        raise ValueError("The input has no positive finite XY span")
    selected = tuple(
        index for index, edge in enumerate(geometry.data["hyperedges"])
        if node_id in edge["nodes"]
    )
    ids = []
    rings = []
    kinds = []
    arities = []
    for index in selected:
        edge = geometry.data["hyperedges"][index]
        vertices = [point for point, _ in groupby(xy[key] for key in edge["nodes"])]
        if len(vertices) > 1 and vertices[0] == vertices[-1]:
            vertices.pop()
        ring = np.array(vertices, dtype=float) - focus
        arity = len(set(edge["nodes"]))
        if arity < 3:
            kind = "point" if len(ring) == 1 else "line"
        else:
            polygon = Polygon(ring) if len(ring) >= 3 else None
            kind = "face" if polygon is not None and polygon.is_valid and polygon.area > 0 else "outline"
        ids.append(edge["id"])
        rings.append(ring)
        kinds.append(kind)
        arities.append(arity)
    max_arity = max(arities)
    palette = matplotlib.colormaps["viridis"]
    colors = tuple(palette(0.13 + 0.67 * np.log(value) / np.log(max_arity)) for value in arities)
    return nodes[node_id], selected, LocalView(
        tuple(ids), tuple(rings), tuple(kinds), float(domain) / 100, colors,
    )


def project(points, level, layer_step, overhead=False):
    """Project an XY ring at a supplied global level, without scaling the panels."""
    if overhead:
        return points
    yaw, pitch = radians(AZIMUTH), radians(ELEVATION)
    x, y = points.T
    return np.column_stack((
        -sin(yaw) * x + cos(yaw) * y,
        -sin(pitch) * (cos(yaw) * x + sin(yaw) * y) + cos(pitch) * level * layer_step,
    ))


def panel_extent(view, profiles):
    points = np.concatenate([
        project(ring, level, view.layer_step, overhead)
        for overhead, levels in profiles
        for ring, level in zip(view.rings, levels)
    ])
    low = points.min(axis=0)
    high = points.max(axis=0)
    margin = max(float((high - low).max()) * 0.07, view.layer_step)
    return [float(low[0] - margin), float(high[0] + margin),
            float(low[1] - margin), float(high[1] + margin)]


def draw_panel(axis, view, levels, extent, overhead=False):
    for index in sorted(range(len(levels)), key=levels.__getitem__):
        screen = project(view.rings[index], levels[index], view.layer_step, overhead)
        color, kind = view.colors[index], view.kinds[index]
        zorder = levels[index] * 2 + index / (2 * len(levels))
        if kind == "face":
            axis.fill(*screen.T, color=color, alpha=0.34, linewidth=0, zorder=zorder)
        if kind == "point":
            axis.scatter(*screen[0], s=11, facecolors="none", edgecolors=[color], zorder=zorder + 0.2)
        else:
            path = np.vstack((screen, screen[0])) if kind in ("face", "outline") else screen
            axis.plot(*path.T, color=color, linewidth=0.75, alpha=0.88,
                      linestyle="--" if kind == "outline" else "-", zorder=zorder + 0.2)

    origin = np.zeros((1, 2))
    stops = sorted(set(levels))
    joints = np.concatenate([project(origin, level, view.layer_step, overhead) for level in stops])
    foreground = max(levels) * 2 + 2
    if not overhead:
        axis.plot(*joints.T, color="white", linewidth=2.7, zorder=foreground, solid_capstyle="round")
        axis.plot(*joints.T, color=FOCUS_COLOR, linewidth=1.45, zorder=foreground + 0.1, solid_capstyle="round")
    axis.scatter(*joints.T, color=FOCUS_COLOR, edgecolor="white",
                 s=23 if overhead else 10, linewidth=0.65, zorder=foreground + 0.3)
    axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
    axis.set_axis_off()


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b2c3ca")
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e3ebee", linewidth=0.75)


def save_figure(figure, out, name):
    figure.savefig(out / f"{name}.png", dpi=240, facecolor="white")
    figure.savefig(out / f"{name}.svg", facecolor="white", metadata={"Date": None})
    plt.close(figure)


def teaser(out, view, focus, baseline, proposal, layer_counts):
    panels = (
        ("Planar 2D", "All edges on the same plane", True, [0] * len(baseline)),
        ("Qu et al. first-fit", f"{layer_counts[0]} layers overall · focus span {max(baseline) - min(baseline)}", False, baseline),
        ("Proposed", f"{layer_counts[1]} layers overall · focus span {max(proposal) - min(proposal)}", False, proposal),
    )
    extent = panel_extent(view, [(overhead, levels) for _, _, overhead, levels in panels])
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 4.45))
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.17, top=0.80, wspace=0.07)
    figure.text(0.025, 0.95, "Following a shared vertex across layers", fontsize=17,
                color=INK, weight="bold", va="top")
    figure.text(0.025, 0.88, f"{focus['label']}  ·  {len(view.rings)} incident hyperedges from the full layout",
                fontsize=10, color=MUTED, va="top")
    for index, (axis, (title, detail, overhead, levels)) in enumerate(zip(axes, panels)):
        draw_panel(axis, view, levels, extent, overhead)
        axis.set_title(f"{chr(97 + index)}) {title}", fontsize=11, color=INK, weight="bold", pad=7)
        axis.text(0.5, -0.035, detail, ha="center", va="top", transform=axis.transAxes,
                  color=MUTED, fontsize=8)
    key = Line2D([0], [0], marker="o", color=FOCUS_COLOR, markersize=4, linewidth=1.45)
    figure.legend([key], ["Selected vertex and pillar"], frameon=False, loc="lower left",
                  bbox_to_anchor=(0.015, 0.061), fontsize=8, labelcolor=INK)
    figure.text(0.025, 0.04, "Identical incident edges and scale across panels. Stacked views use the same oblique camera.",
                fontsize=8, color=MUTED)
    save_figure(figure, out, "fig1-teaser")
    return extent


def checked_summary(path):
    summary = read_json(path)
    if summary.get("error") or not summary.get("cases"):
        raise ValueError(f"Unusable or empty experiment summary: {path}")
    return summary


def verified_records(summary, input_hash=None):
    records = [
        row for row in summary["cases"]
        if row.get("status") == "verified" and (input_hash is None or row["input_sha256"] == input_hash)
    ]
    keys = [(row["input"], row["condition"], row["seed"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate verified cases in the experiment summary")
    return records


def verify_figure_case(records, condition, seed, path, measured):
    matches = [row for row in records if row["condition"] == condition and row["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one verified {condition!r} case for seed {seed}")
    row = matches[0]
    expected_path = (ROOT / row["case_directory"] / "layout.json").resolve()
    if path.resolve() != expected_path:
        raise ValueError(f"The supplied layout is outside the verified {condition!r} case")
    for key in ("used_layer_count", "pillar_length", "overlap_violation_count", "constrained_hubs"):
        if row["metrics"][key] != measured[key]:
            raise ValueError(f"Measured {key} does not match the verified {condition!r} report")
    if not np.isclose(row["metrics"]["l2_scaled_qp"], measured["l2_scaled_qp"], rtol=1e-9, atol=1e-7):
        raise ValueError(f"Measured F_scaled disagrees with the verified {condition!r} report")


def verify_proposal_report(path, geometry, levels, seed):
    report_path = path.with_name("report.json")
    report = read_json(report_path)
    settings = report["settings"]
    if (
        report["input_sha256"] != geometry.hashes["input_sha256"]
        or report["problem_signature"] != geometry.hashes["problem_signature"]
        or report["quadratic_metric"] != "l2_scaled_qp"
        or settings["method"] != "anneal"
        or settings["objective"] != "hybrid"
        or settings["alpha"] != 0.25
        or settings["seed"] != seed
        or settings["reg"] != REGULARIZATION
        or tuple(settings["moves"]) != ("move", "kempe", "swap", "shift")
        or settings["optimize_free_edges"] is not True
        or tuple(report["stages"][-1]["layers"]) != levels
    ):
        raise ValueError("The proposed layout does not match the recorded input, seed, method, and final layers")
    return source_record(report_path)


def xy_figure(out, summary):
    profiles = summary["input_profiles"]
    variants = [name for name in VARIANTS if name in profiles]
    records = verified_records(summary)
    if not variants:
        raise ValueError("The XY summary has no recognized saved variants")
    inputs = [profiles[name] for name in variants]
    classes = ["0", "1", "2", ">=3"]
    decompositions = [profile["overlap_profile"] for profile in inputs]
    totals = [part["cumulative_shared_vertices"][">=0"] for part in decompositions]
    for part, total in zip(decompositions, totals):
        if sum(part["exact_shared_vertices"][key]["overlap_count"] for key in classes) != total["overlap_count"]:
            raise ValueError("The overlap groups do not sum to their reported total")
        if not 0 <= total["clique_lower_bound"] <= total["dsatur_upper_bound"]:
            raise ValueError("The layer bounds are inconsistent")
    positions = np.arange(len(variants))
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    figure.subplots_adjust(left=0.085, right=0.97, bottom=0.22, top=0.79, wspace=0.29)
    figure.text(0.085, 0.95, "XY variants change the required layers", fontsize=17, weight="bold", color=INK, va="top")
    base = np.zeros(len(variants))
    for key, label, color in zip(classes, ("0", "1", "2", "≥3"), ("#568da5", "#8fb296", "#cdad70", "#a2799f")):
        heights = np.array([part["exact_shared_vertices"][key]["overlap_count"] for part in decompositions])
        left.bar(positions, heights, bottom=base, color=color, width=0.63, label=label)
        base += heights
    left.legend(title="Shared vertices", loc="upper left", bbox_to_anchor=(-0.035, -0.13),
                ncols=4, frameon=False, fontsize=8, title_fontsize=8, columnspacing=1.0)
    left.set_ylabel("Overlap pairs")
    left.set_title("Overlaps grouped by shared vertices", fontsize=10, color=INK, loc="left")
    lower = [total["clique_lower_bound"] for total in totals]
    upper = [total["dsatur_upper_bound"] for total in totals]
    right.vlines(positions - 0.08, lower, upper, color="#64828e", linewidth=1.1)
    right.scatter(positions - 0.08, lower, marker="v", color="#406377", s=33, label="Clique witness")
    right.scatter(positions - 0.08, upper, marker="^", facecolors="white", edgecolors="#406377",
                  s=33, label="DSatur upper bound")
    proposal_label = "Proposed · median and range"
    for pos, name, source in zip(positions, variants, inputs):
        values = [
            row["metrics"]["used_layer_count"] for row in records
            if row["input"] == name and row["input_sha256"] == source["input_sha256"] and row["condition"] == "full"
        ]
        if values:
            middle = median(values)
            right.errorbar(pos + 0.10, middle, yerr=[[middle - min(values)], [max(values) - middle]],
                           color=FOCUS_COLOR, marker="o", markersize=4, capsize=3,
                           label=proposal_label)
            proposal_label = None
            right.annotate(f"n={len(values)}", (pos + 0.10, max(values)), xytext=(3, 5),
                           textcoords="offset points", color=MUTED, fontsize=7)
    right.set_ylabel("Layers")
    right.set_title("Witness bounds and observed layers", fontsize=10, color=INK, loc="left")
    right.legend(loc="upper left", bbox_to_anchor=(-0.035, -0.13), frameon=False, fontsize=7)
    for axis in (left, right):
        axis.set_xticks(positions, [name.capitalize() for name in variants])
        style_axis(axis)
    save_figure(figure, out, "fig2-xy-z")


def benchmark_figure(out, records, seed, omitted):
    by_method = defaultdict(list)
    for row in records:
        if row["condition"] in METHOD_NAMES:
            by_method[row["condition"]].append(row)
    methods = [name for name in METHOD_NAMES if name in by_method]
    if not methods:
        return
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 5.1), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.24, top=0.81, wspace=0.35)
    figure.text(0.028, 0.95, "Pillar length, projected coverage, and search progress",
                fontsize=16, color=INK, weight="bold", va="top")
    subtitle = f"Original input · {len(records)} verified cases"
    if omitted:
        subtitle += f" · {omitted} other planned cases omitted"
    figure.text(0.028, 0.89, subtitle, fontsize=9, color=MUTED, va="top")
    for axis, key, title in zip(axes, ("pillar_length", "overdraw_pixel_count"),
                                ("Total pillar length", "Pixels covered by ≥2 faces")):
        for position, method in enumerate(methods):
            records_for_method = by_method[method]
            if key == "pillar_length":
                values = [row["metrics"][key] for row in records_for_method]
            else:
                values = [row["projection"][key] for row in records_for_method if key in row.get("projection", {})]
            if values:
                middle = median(values)
                color = FOCUS_COLOR if method == "full" else "#3f738b"
                axis.errorbar(middle, position, xerr=[[middle - min(values)], [max(values) - middle]],
                              color=color, marker="o", markersize=4, capsize=3)
        axis.set_title(title, fontsize=9, color=INK, loc="left")
        axis.set_yticks(range(len(methods)))
        axis.invert_yaxis()
        style_axis(axis)
    axes[0].set_yticklabels([f"{METHOD_NAMES[method]}  ({len(by_method[method])})" for method in methods])
    axes[1].set_yticklabels([])
    progress = axes[2]
    for method, color in zip(("swap", "swap-move", "swap-move-kempe", "full"),
                             ("#7692ab", "#378986", "#7964a3", FOCUS_COLOR)):
        candidates = [row for row in by_method[method] if row["seed"] == seed and row.get("stages")]
        if candidates:
            trajectory = candidates[0]["stages"][0]["trajectory"]
            progress.step([point["elapsed_seconds"] for point in trajectory],
                          [point["optimized_value"] for point in trajectory],
                          where="post", label=METHOD_NAMES[method], color=color, linewidth=1.4)
    progress.set(xlabel="Elapsed solver time (s)", ylabel=r"$B + 0.25 F_{\mathrm{scaled}}$")
    progress.set_title(f"Best objective · seed {seed}", fontsize=9, color=INK, loc="left")
    if progress.lines:
        progress.legend(frameon=False, fontsize=8)
    style_axis(progress)
    figure.text(0.028, 0.05, "Dots show the median, bars the observed range, and parentheses the number of verified cases.",
                fontsize=8, color=MUTED)
    save_figure(figure, out, "fig3-benchmark")


def objective_figure(out, records):
    colors = ("#57578a", "#457159", "#347d99", "#6e9e98", FOCUS_COLOR, "#a170aa", "#895a5c")
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.6))
    figure.subplots_adjust(left=0.105, right=0.972, bottom=0.31, top=0.80, wspace=0.39)
    figure.text(0.045, 0.95, "Objective weights and layer-cap trade-offs", color=INK,
                fontsize=16, weight="bold", va="top")
    for (name, label), color in zip(OBJECTIVE_LABELS.items(), colors):
        cases = [row for row in records if row["condition"] == name]
        if not cases:
            continue
        squared = [row["metrics"]["l2_scaled_qp"] for row in cases]
        pillars = [row["metrics"]["pillar_length"] for row in cases]
        left.scatter(squared, pillars, color=color, alpha=0.27, s=17)
        left.scatter(median(squared), median(pillars), color=color, s=32, marker="D",
                     label=f"{label} ({len(cases)})")
    left.set(xlabel="Normalized squared distance", ylabel="Total pillar length")
    left.set_title("Fixed cap · verified seed results", fontsize=10, color=INK, loc="left")
    if left.get_legend_handles_labels()[0]:
        left.legend(loc="upper center", bbox_to_anchor=(0.47, -0.27), fontsize=7, ncols=3, frameon=False)

    for configuration, label, marker, color in (
        ("hybrid-0.25", "Fixed cap", "s", "#3f738b"),
        ("hybrid-0.25-plus-1", "Through cap +1", "o", "#7964a3"),
        ("hybrid-0.25-plus-2", "Through cap +2", "^", FOCUS_COLOR),
    ):
        candidates = [
            stage["metrics"] for row in records if row["condition"] == configuration for stage in row["stages"]
        ]
        if candidates:
            right.scatter([point["used_layer_count"] for point in candidates],
                          [point["pillar_length"] for point in candidates],
                          color=color, label=label, marker=marker, alpha=0.56, s=25)
    right.set(xlabel="Layers used", ylabel="Total pillar length")
    right.xaxis.get_major_locator().set_params(integer=True)
    right.set_title(r"Observed stage solutions · $\alpha=0.25$", fontsize=10, color=INK, loc="left")
    if right.get_legend_handles_labels()[0]:
        right.legend(loc="upper center", bbox_to_anchor=(0.50, -0.27), fontsize=7, ncols=3, frameon=False)
    else:
        right.text(0.5, 0.5, "No verified stage solutions", transform=right.transAxes,
                   ha="center", color=MUTED, fontsize=9)
    for axis in (left, right):
        style_axis(axis)
    figure.text(0.045, 0.039, "Diamonds mark coordinate-wise medians. Warm-start configurations share a total trial budget.",
                fontsize=8, color=MUTED)
    save_figure(figure, out, "fig4-objectives")


def write_case(out, focus, view, metadata):
    baseline = metadata["layouts"]["first_fit"]
    proposal = metadata["layouts"]["proposal"]
    count = len(view.edge_ids)
    text = f"""# Local case study for VisNotes

## Scope

Internal draft using the existing original input. Public use of the data and derived figures is pending a decision. The example follows **{focus['label']}** (vertex ID `{focus['id']}`), selected by constrained incidence, then by total incidence and original input order. All {count} incident hyperedges are displayed in each panel, using their layers from the full layout. The selection depends on input incidence and constraints.

## Reading Fig. 1

The planar panel places the selected relationships at Z=0. The two stacked panels show input-order first-fit from Qu et al. (2017, Section 6) and the proposed arrangement for seed {metadata['proposal_seed']}. Both stacked views use a {AZIMUTH:g}° azimuth, a {ELEVATION:g}° elevation, and one layer step equal to 1/100 of the full input XY span. All panels use the same display scale. The marker identifies the selected vertex, with a vertical segment connecting its layers in the two stacked panels. Degenerate boundaries are marked as dashed outlines when present.

| Quantity | First-fit | Proposed |
| --- | ---: | ---: |
| Full-layout layers | {baseline['used_layer_count']} | {proposal['used_layer_count']} |
| Full-layout overlap violations | {baseline['overlap_violation_count']} | {proposal['overlap_violation_count']} |
| Full-layout total pillar length | {baseline['pillar_length']:,} | {proposal['pillar_length']:,} |
| Selected vertex span | {baseline['focus_span']} | {proposal['focus_span']} |

The local subset contains {metadata['local_overlap_pairs']} pairs of hyperedges that overlap in XY. This view supports inspection of the selected vertex and its incident layers at a fixed scale. Reading accuracy and task time have yet to be assessed.

[Teaser](fig1-teaser.svg) · [Figure provenance](figure-manifest.json)
"""
    (out / "case-study.md").write_text(text, encoding="utf-8")


def source_record(path):
    path = path.resolve()
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return {"path": name, "sha256": file_hash(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--first-fit", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--benchmark-summary", type=Path)
    parser.add_argument("--xy-summary", type=Path)
    parser.add_argument("--objective-summary", type=Path)
    parser.add_argument("--proposal-seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args(argv)

    geometry = make_geometry(args.input, REGULARIZATION)
    first_fit, baseline_metrics = read_layout_metrics(read_json(args.first_fit), geometry, "First-fit")
    if first_fit != first_fit_levels(geometry):
        raise ValueError("The first-fit layout does not match the input-order greedy rule")
    proposal, proposal_metrics = read_layout_metrics(read_json(args.proposal), geometry, "Proposed")
    focus, selected, view = local_view(geometry, baseline_metrics)
    local_baseline = [first_fit[index] for index in selected]
    local_proposal = [proposal[index] for index in selected]
    sources = {
        "input": source_record(args.input),
        "proposal_report": verify_proposal_report(args.proposal, geometry, proposal, args.proposal_seed),
        "generator": source_record(Path(__file__)),
    }
    records = None
    benchmark_counts = None
    omitted = 0
    if args.benchmark_summary:
        benchmark = checked_summary(args.benchmark_summary)
        records = verified_records(benchmark, geometry.hashes["input_sha256"])
        input_names = {row["input"] for row in records}
        omitted = sum(row["input"] in input_names and row.get("status") != "verified"
                      for row in benchmark["cases"])
        benchmark_counts = benchmark["case_counts"]
        verify_figure_case(records, "greedy-input", None, args.first_fit, baseline_metrics)
        verify_figure_case(records, "full", args.proposal_seed, args.proposal, proposal_metrics)
        sources["benchmark_summary"] = source_record(args.benchmark_summary)
    xy_summary = checked_summary(args.xy_summary) if args.xy_summary else None
    if xy_summary is not None:
        sources["xy_summary"] = source_record(args.xy_summary)
    objectives = None
    if args.objective_summary:
        objectives = verified_records(checked_summary(args.objective_summary), geometry.hashes["input_sha256"])
        if not any(row["condition"] in OBJECTIVE_LABELS for row in objectives):
            raise ValueError("The objective summary has no verified fixed-cap objective cases for the supplied input")
        sources["objective_summary"] = source_record(args.objective_summary)

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "visnotes"})
    extent = teaser(out, view, focus, local_baseline, local_proposal,
                    (baseline_metrics["used_layer_count"], proposal_metrics["used_layer_count"]))
    if records is not None:
        benchmark_figure(out, records, args.proposal_seed, omitted)
    if xy_summary is not None:
        xy_figure(out, xy_summary)
    if objectives is not None:
        objective_figure(out, objectives)
    selected_set = set(selected)
    metadata = {
        "audience": "internal-draft",
        "source_article": "https://vdslab.esa.io/posts/9193",
        "sources": sources,
        "matplotlib_version": matplotlib.__version__,
        "proposal_seed": args.proposal_seed,
        "matched_verified_benchmark_cases": records is not None,
        "benchmark_case_counts": benchmark_counts,
        "hyperedge_count": geometry.size,
        "focus_vertex": focus["id"],
        "incident_edge_ids": view.edge_ids,
        "local_overlap_pairs": sum(a in selected_set and b in selected_set for a, b in geometry.overlaps),
        "view": {"azimuth": AZIMUTH, "elevation": ELEVATION, "z_step_world": view.layer_step,
                 "shared_extent": extent, "outline_only_edge_count": view.kinds.count("outline")},
        "figures": {path.name: file_hash(path) for path in sorted(out.glob("fig*.*"))
                    if path.suffix in (".png", ".svg")},
        "layouts": {
            key: {
                **source_record(path),
                **metrics,
                "focus_span": max(levels) - min(levels),
                "incident_levels": levels,
            }
            for key, path, metrics, levels in (
                ("first_fit", args.first_fit, baseline_metrics, local_baseline),
                ("proposal", args.proposal, proposal_metrics, local_proposal),
            )
        },
    }
    (out / "figure-manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    write_case(out, focus, view, metadata)
    print(f"Created VisNotes figures and case study in {out}")


if __name__ == "__main__":
    main()
