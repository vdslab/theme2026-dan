"""Replot geometry and layer counts from the four archived exp05 inputs."""

import argparse
import json
from collections import Counter
from itertools import combinations
from math import pi, tan
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/experiments/visnotes_remaining_01/exp05/summary.json"
VARIANTS = ("original", "paper", "scalable", "arranged")


def measured_shapes(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    xy = {node["id"]: (node["x"], node["y"]) for node in data["nodes"]}
    polygons, members, regularities = {}, {}, []
    for index, edge in enumerate(data["hyperedges"]):
        ids = list(dict.fromkeys(edge["nodes"]))
        members[edge["id"]] = set(ids)
        if len(ids) < 3:
            continue
        face = Polygon([xy[node] for node in ids])
        if not face.is_valid or face.area <= 0:
            raise ValueError(f"Invalid polygon in {path.name}, edge {index}")
        polygons[index] = face
        regularities.append(face.length**2 / (4 * len(ids) * tan(pi / len(ids)) * face.area))

    overlaps = set()
    shared = Counter()
    for i, j in combinations(polygons, 2):
        if polygons[i].intersection(polygons[j]).area > 1e-9:
            overlaps.add((i, j))
            a, b = data["hyperedges"][i], data["hyperedges"][j]
            shared[len(members[a["id"]] & members[b["id"]])] += 1
    return members, overlaps, shared, regularities


def read_rows():
    archive = json.loads(SUMMARY.read_text(encoding="utf-8"))
    baseline = None
    rows = []
    for name in VARIANTS:
        profile = archive["input_profiles"][name]
        members, overlaps, shared, ratios = measured_shapes(ROOT / profile["file"])
        if baseline is None:
            baseline = members
        if members != baseline or len(ratios) != 168:
            raise ValueError(f"Input membership or face count mismatch for {name}")
        profile_counts = profile["overlap_profile"]
        all_pairs = profile_counts["cumulative_shared_vertices"][">=0"]
        if (shared[0] != profile_counts["exact_shared_vertices"]["0"]["overlap_count"]
                or len(overlaps) != all_pairs["overlap_count"]):
            raise ValueError(f"Geometric overlap counts disagree with exp05 for {name}")
        witness = all_pairs["clique_witness_indices"]
        if (len(set(witness)) != all_pairs["clique_lower_bound"]
                or any(tuple(sorted(pair)) not in overlaps for pair in combinations(witness, 2))):
            raise ValueError(f"Inconsistent clique lower bound for {name}")
        rows.append((mean(ratios), shared[0], len(witness), all_pairs["dsatur_upper_bound"]))
    return rows


def plot(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.75), sharey=True)
    fig.subplots_adjust(left=.105, right=.965, bottom=.23, top=.71, wspace=.3)
    fig.text(.03, .93, "Shape and overlap across four saved layouts", fontsize=19, weight="bold")
    fig.text(.03, .84, "Same hypergraph  ·  168 polygonal faces  ·  Original membership sets retained",
             fontsize=11, color="#475569")
    specs = [
        ("Mean regularity ratio", 5.9, (0, 1, 2, 3, 4, 5), "1 = regular polygon"),
        ("Overlapping pairs", 7200, (0, 2000, 4000, 6000), "Pairs with no common vertex"),
        ("Layers", 92, (0, 20, 40, 60, 80), "DSatur coloring"),
    ]
    for col, (ax, (title, limit, ticks, label)) in enumerate(zip(axes, specs)):
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=8)
        ax.set_xlim(0, limit)
        ax.set_xticks(ticks)
        ax.set_xlabel(label, fontsize=10, labelpad=7)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dfe7ef", lw=.75)
        ax.tick_params(axis="both", length=0, labelsize=10)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b9c4d0")
        for y, (ratio, count, lower, upper) in enumerate(rows):
            value = (ratio, count, upper)[col]
            ax.barh(y, value, height=.47, color="#49788b" if y else "#314052")
            number = f"{ratio:.2f}" if col == 0 else f"{value:,}"
            ax.annotate(number, (value, y), xytext=(4, 0), textcoords="offset points",
                        fontsize=10, ha="left", va="center")
            if col == 2:
                ax.scatter(lower, y, marker="|", s=135, linewidths=2, color="#edaa48", zorder=3)
    axes[0].set_yticks(range(len(VARIANTS)), [name.capitalize() for name in VARIANTS])
    axes[0].invert_yaxis()
    fig.text(.03, .045,
             "Orange ticks show verified clique lower bounds (39, 76, 71, 69). "
             "The relayouts also change boundary order.", fontsize=10, color="#475569")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        fig.savefig(f, format="png", dpi=180)
    plt.close(fig)


def plot_example(output):
    """Show the same preselected two faces, keeping orientation and XY scale."""
    profiles = json.loads(SUMMARY.read_text(encoding="utf-8"))["input_profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharex=True, sharey=True)
    panels = []
    for variant in ("original", "arranged"):
        source = json.loads((ROOT / profiles[variant]["file"]).read_text(encoding="utf-8"))
        xy = {v["id"]: (v["x"], v["y"]) for v in source["nodes"]}
        edges = {e["id"]: e["nodes"] for e in source["hyperedges"]}
        if set(edges[49]) & set(edges[169]) != {198}:
            raise ValueError("The example must share exactly vertex 198.")
        ox, oy = xy[198]
        panels.append({i: Polygon([(xy[n][0] - ox, xy[n][1] - oy) for n in edges[i]])
                       for i in (49, 169)})
    for ax, panel, title in zip(axes, panels, ("Original XY", "After arrangement")):
        for i, color, style, marker in ((49, "#006a83", "-", "o"), (169, "#a94914", "--", "s")):
            x, y = panel[i].exterior.xy
            ax.fill(x, y, color=color, alpha=.075)
            ax.plot(x, y, color=color, linestyle=style, marker=marker, ms=3, lw=1.6,
                    markerfacecolor="white", label=f"E{i}")
        ax.plot(0, 0, marker="o", markerfacecolor="white", markeredgewidth=2,
                color="#a4134d", ms=7, label="V198")
        ax.set_title(title, fontsize=13, weight="bold", color="#163440")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    axes[0].legend(loc="upper left", frameon=False, ncol=1, fontsize=9)
    fig.suptitle("Two faces, one shared vertex", weight="bold", fontsize=17, x=.055, ha="left")
    fig.text(.055, .03, "Both panels center V198 by translation and use the same XY units. Only E49 and E169 are shown.", fontsize=9)
    fig.subplots_adjust(top=.82, bottom=.12, wspace=.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        fig.savefig(f, format="png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new PNG file")
    parser.add_argument("--example-output", type=Path, help="New PNG comparing the two XY outlines")
    args = parser.parse_args()
    results = read_rows()
    for variant, (ratio, count, lower, upper) in zip(VARIANTS, results):
        print(f"{variant:9} regularity {ratio:.3f}, nonmember pairs {count}, layers [{lower}, {upper}]")
    plot(results, args.output)
    if args.example_output:
        plot_example(args.example_output)
