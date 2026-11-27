"""Build the public-data teaser and paired comparison figures from verified runs."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import REG, file_hash, make_geometry, read_json, save_json
from experiments.visnotes_metrics import read_layout_metrics
from tools.make_visnotes_figures import first_fit_levels, local_view, plt, save_figure, style_axis, teaser


def build_teaser(source, baseline_path, proposed_path, output):
    geometry = make_geometry(source, REG)
    layouts = []
    cases = []
    for file, condition in ((baseline_path, "greedy-input"), (proposed_path, "full")):
        case = read_json(file.with_name("case.json"))
        if case["status"] != "verified" or case["condition"] != condition:
            raise ValueError(f"{file}: expected a verified {condition} case.")
        if case["input_sha256"] != geometry.hashes["input_sha256"]:
            raise ValueError(f"{file}: input differs from the recorded case.")
        levels, metrics = read_layout_metrics(read_json(file), geometry, condition)
        if case["metrics"] != metrics:
            raise ValueError(f"{file}: saved measurements differ from the case record.")
        layouts.append((levels, metrics))
        cases.append(case)
    if cases[1]["seed"] != 0:
        raise ValueError("The preselected illustration uses seed 0.")
    baseline, proposal = (item[0] for item in layouts)
    if baseline != first_fit_levels(geometry):
        raise ValueError("The baseline must implement input-order first-fit.")
    focus, indices, view = local_view(geometry, layouts[1][1])
    baseline_local = [baseline[i] for i in indices]
    proposal_local = [proposal[i] for i in indices]
    output.mkdir(parents=True, exist_ok=False)
    layer_counts = [item[1]["used_layer_count"] for item in layouts]
    extent = teaser(output, view, focus, baseline_local, proposal_local, layer_counts)
    save_json(output / "figure-manifest.json", {
        "inputs": {
            "source": file_hash(source), "first_fit": file_hash(baseline_path),
            "proposed": file_hash(proposed_path),
        },
        "focus_vertex": focus["id"], "incident_edge_ids": list(view.edge_ids),
        "selection_rule": "maximum constrained incidence, then incidence, then input order",
        "layer_counts": layer_counts, "window": extent,
        "focus_spans": [max(local) - min(local) for local in (baseline_local, proposal_local)],
        "notice": "Identical public input and incident edges; top-down 2D and matched oblique 2.5D.",
    })


def check_summary(path, phase):
    summary = read_json(path)
    if summary["phase"] != phase or not summary["sources_unchanged"]:
        raise ValueError("The summary has the wrong phase or its inputs changed during measurement.")
    return summary


def build_benchmark(summary, output):
    conditions = ["greedy-input", "dsatur-free", "swap-move-kempe", "full"]
    panels = (
        ("Used layers", "used_layer_count"),
        ("Total pillar length", "pillar_length"),
        ("Squared distance", "l2_scaled_qp"),
        ("Pixels covered by two or more faces", "overdraw_pixel_count"),
    )
    cases = {(row["input"], row["condition"]): row
             for row in summary["comparisons"] if row["condition"] in conditions}
    inputs = list(dict.fromkeys(key[0] for key in cases))
    if not inputs or any((name, label) not in cases for name in inputs for label in conditions):
        raise ValueError("The figure needs every comparison condition for each input.")
    names = {"dblp": "DBLP TVCG", "dblp-medium": "DBLP TPAMI", "marseille": "Marseille"}
    palette = ("#5f6d7b", "#7a96a1", "#287b83", "#09594e")
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.5), layout="constrained")
    figure.suptitle("Layer assignment across public datasets", weight="bold", fontsize=17)
    for axis, (title, key) in zip(axes.flat, panels, strict=True):
        for idx, (label, color) in enumerate(zip(conditions, palette, strict=True)):
            medians = [cases[name, label]["metrics"][key]["median"] for name in inputs]
            x = [i + (idx - 1.5) * 0.18 for i in range(len(inputs))]
            axis.bar(x, medians, width=0.16, label=label, color=color)
        axis.set_xticks(list(range(len(inputs))), [names.get(name, name) for name in inputs])
        axis.set_title(title, loc="left")
        style_axis(axis)
    handles, _ = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, ["Qu first-fit", "DSatur + free edges", "Proposed without shift", "Proposed"],
                  ncols=4, loc="outside lower center")
    output.mkdir(parents=True, exist_ok=False)
    save_figure(figure, output, "public-benchmark")


def build_xy(summary, output):
    conditions = ("base-xy-base-order", "alternate-xy-base-order",
                  "base-xy-alternate-order", "alternate-xy-alternate-order")
    labels = ("Base", "XY only", "Order only", "Both")
    figure, axes = plt.subplots(1, 2, figsize=(9.1, 4.3), layout="constrained")
    figure.suptitle("Separating XY coordinates from polygon boundary order", fontsize=15, weight="bold")
    for axis, (scope, title) in zip(
        axes, (("all", "All edges, repair permitted"), ("valid", "Common valid subset")), strict=True,
    ):
        profiles = [summary["inputs"][f"{scope}-{condition}"]["overlap_profile"] for condition in conditions]
        shared_zero = [row["exact_shared_vertices"]["0"]["overlap_count"] for row in profiles]
        total = [row["cumulative_shared_vertices"][">=0"]["overlap_count"] for row in profiles]
        axis.bar(labels, shared_zero, color="#25777d", label="0 shared vertices")
        axis.bar(labels, [a - b for a, b in zip(total, shared_zero, strict=True)], bottom=shared_zero,
                 color="#acbdc6", label="1 or more shared vertices")
        for i, value in enumerate(total):
            axis.annotate(f"{value:,}", (i, value), xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=8)
        axis.set_title(title, loc="left")
        axis.set_ylabel("Overlapping polygon pairs")
        axis.set_ylim(0, max(total) * 1.14)
        style_axis(axis)
    handles, legend = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend, ncols=2, loc="outside lower center")
    output.mkdir(parents=True, exist_ok=False)
    save_figure(figure, output, "xy-controls")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    figure = sub.add_parser("teaser")
    for parameter in ("source", "first-fit", "proposed", "output"):
        figure.add_argument(f"--{parameter}", required=True, type=Path)
    for kind in ("benchmark", "xy"):
        comparison = sub.add_parser(kind)
        comparison.add_argument("summary", type=Path)
        comparison.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.kind == "teaser":
        build_teaser(args.source, args.first_fit, args.proposed, args.output)
    elif args.kind == "benchmark":
        build_benchmark(check_summary(args.summary, "benchmark"), args.output)
    else:
        build_xy(check_summary(args.summary, "xy"), args.output)


if __name__ == "__main__":
    main()
