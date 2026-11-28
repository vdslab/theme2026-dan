"""Draw counts derived from the full-layout and pillar audits, for seed zero."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def build(run, destination):
    analysis = json.loads((run / "analysis/final/summary.json").read_bytes())
    if not analysis["complete"]:
        raise ValueError("Main audit incomplete")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    figure.subplots_adjust(left=.12, right=.97, bottom=.22, top=.70, wspace=.34)
    figure.suptitle("Dense layouts retain ambiguous contacts", x=.025, ha="left",
                    fontsize=17, weight="bold", color="#18323f")
    figure.text(.025, .82, "Full inputs, seed 0. Labels show the numerator and denominator.", fontsize=10)
    for index, (dataset, folder) in enumerate((("original", "original"), ("dblp", "DBLP-TVCG"))):
        case = analysis["datasets"][dataset]
        contacts = json.loads((run / "contacts/final" / folder / "contacts.json").read_bytes())
        for offset, (method, color, label) in enumerate((
            ("first-fit", "#687981", "Input-order first-fit"), ("proposed", "#106c75", "Proposed")
        )):
            values = contacts["methods"][method]["all"]
            row, = [r for r in case["projections"] if r["method"] == method and r["scope"] == "valid"
                    and r["elevation"] == 35 and r["seed"] in (0, None)]
            ratios = [(values["affected_pillars"], values["nonzero_pillars"]),
                      (row["fp"], row["tp"] + row["fp"])]
            for ax, (num, denom) in zip(axes, ratios, strict=True):
                y = index + (offset - .5) * .36
                percent = 100 * num / denom
                ax.barh(y, percent, height=.30, color=color, label=label if index == 0 else None)
                ax.text(percent + 1.4, y, f"{num}/{denom}", va="center", fontsize=9)
    for ax, title in zip(axes, ("Pillars pierced by an unrelated face",
                                "Projected overlaps with no shared ID"), strict=True):
        ax.set_title(title, loc="left", fontsize=11.5, weight="bold")
        ax.set(yticks=[0, 1], yticklabels=["original", "DBLP"], xlim=(0, 107), ylim=(1.48, -.48),
               xticks=[0, 25, 50, 75, 100], xlabel="Percent")
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    if destination.exists():
        raise FileExistsError(destination)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    build(options.run, options.output)
