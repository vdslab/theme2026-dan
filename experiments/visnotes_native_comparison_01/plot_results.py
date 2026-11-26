"""Plot paired Python and C++ timings and independently verified layout quality."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    if not report["complete"]:
        raise ValueError("Expected a complete, verified comparison")
    cases = {(row["seed"], row["backend"]): row for row in report["cases"] if row["profile"] == "fixed"}
    seeds = sorted({seed for seed, _ in cases})
    backends = ("python", "native")
    metrics = (
        ("End-to-end wall time (s)", lambda row: row["solver_wall_seconds"], "{x:,.2f}"),
        ("Search-stage time (s)", lambda row: row["stage_seconds"], "{x:,.2f}"),
        ("Used layers", lambda row: row["metrics"]["used_layer_count"], "{x:,.0f}"),
        ("Hybrid objective", lambda row: row["optimized_value"], "{x:,.0f}"),
    )
    colors = ("#596c80", "#0f766e")
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 6.8), layout="constrained")
    figure.suptitle("Python and C++ layer search", fontsize=19, weight="bold", color="#173040")
    for axis, (title, value, tick_format) in zip(axes.flat, metrics, strict=True):
        series = [[value(cases[seed, backend]) for backend in backends] for seed in seeds]
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e2e9ec")
        axis.spines[["top", "right"]].set_visible(False)
        for row in series:
            axis.plot((0, 1), row, color="#8398a5", alpha=0.48, marker="o", markersize=4)
        for index, values in enumerate(zip(*series)):
            center = median(values)
            axis.scatter(index, center, marker="D", s=52, color=colors[index], edgecolor="white", zorder=3)
            axis.annotate(tick_format.format(x=center), (index, center), xytext=(9, 6),
                          textcoords="offset points", color=colors[index], weight="bold")
        axis.set_xlim(-0.25, 1.35)
        axis.margins(y=0.16)
        if title == "Used layers":
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            if all(row == series[0] for row in series):
                axis.set_ylim(min(series[0]) - 1.5, max(series[0]) + 1.5)
        axis.set_xticks((0, 1), ("Python", "C++"))
        axis.yaxis.set_major_formatter(StrMethodFormatter(tick_format))
    figure.supxlabel(f"Fixed cap. {len(seeds)} paired seeds. Diamonds show medians.", fontsize=10)
    with args.image.open("xb") as image:
        figure.savefig(image, format="png", dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
