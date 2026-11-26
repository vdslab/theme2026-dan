"""Render the paired-seed comparison from independently verified cases."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    if not data["complete"]:
        raise ValueError("Expected a complete comparison")
    methods = ("tree-squared", "compact-squared", "compact-hybrid")
    rows = {(case["condition"], case["seed"]): case for case in data["cases"]}
    labels = ("Tree SA\nSquared", "Direct SA\nSquared", "Direct SA\nHybrid")
    colors = ("#596c80", "#267d81", "#a54416")
    plots = [
        ("Used layers", "metrics", "used_layer_count"),
        ("Total pillar length", "metrics", "pillar_length"),
        ("Normalized squared distance", "metrics", "l2_scaled_qp"),
        ("Pixels covered by ≥2 faces", "projection", "overdraw_pixel_count"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), layout="constrained")
    figure.get_layout_engine().set(hspace=0.1, wspace=0.12)
    figure.suptitle("Rooted-tree and direct searches", fontsize=19, weight="bold", color="#173040")
    for axis, (label, group, metric) in zip(axes.flat, plots):
        series = [[rows[method, seed][group][metric] for method in methods] for seed in range(5)]
        axis.set_title(label, loc="left", fontsize=11)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e2e9ec")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#adbec6")
        for values in series:
            axis.plot(range(3), values, color="#97a8b2", alpha=0.48, linewidth=1, marker="o", markersize=3)
        for index, values in enumerate(zip(*series)):
            center = median(values)
            axis.scatter(index, center, marker="D", s=50, color=colors[index], edgecolor="white", zorder=3)
            axis.annotate(f"{center:,.0f}", (index, center), xytext=(7, 5), textcoords="offset points",
                          color=colors[index], fontsize=9, weight="bold")
        axis.set_xlim(-0.25, 2.31)
        axis.margins(y=0.14)
        axis.set_xticks(range(3), labels)
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    figure.supxlabel("Same input · seeds 0–4 paired by line · diamonds show medians", fontsize=10)
    with args.output.open("xb") as image:
        figure.savefig(image, format="png", dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
