"""Render matched style prototypes from the two archived full seed-0 layouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as Patch
import numpy as np
import shapely
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.make_visnotes_figures import AZIMUTH, ELEVATION, project

RESULTS = ROOT / "results/experiments"
RECORD = RESULTS / "visnotes_polygon_plan_01/figure-data.json"
ORIGINAL = RESULTS / "visnotes_remaining_01/exp04/original/full/seed-0/layout.json"
HASHES = {
    "original": "b271a0e540ef14d6a5ab528659131cc170e08505820e2929dafcd2acc3087970",
    "dblp": "6ef3dd5aa31c8fd9a715579288d555ab1f167377d07e12fcd69e8d468ac0fac4",
}
CASES = {
    "original": (198, ((49, 169), (46, 49))),
    "dblp": (48, ((27, 338), (112, 235))),
}
INK, MUTED, BLUE, ORANGE, JOINT = "#163440", "#4c6470", "#006a83", "#a94914", "#a4134d"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def numeric_layout(path, name):
    raw = path.read_bytes()
    if digest(raw) != HASHES[name]:
        raise ValueError(f"{name} must use the archived seed-0 layout from esa #9278.")
    layout = json.loads(raw)
    return {
        "vertices": [[v["id"], v["x"], v["y"]] for v in layout["nodes"]],
        "edges": [[e["id"], e["layer"], e["nodes"]] for e in layout["hyperedges"]],
    }


def geometry_digest(layout):
    return digest(json.dumps(layout, separators=(",", ":")).encode("utf-8"))


def row_geometry(pair, hub, expected, memberships, levels, xy, flat, projected, step):
    a, b = pair
    common = sorted(set(memberships[a]) & set(memberships[b]))
    if common != expected:
        raise ValueError(f"Unexpected membership for {pair}: {common}")
    overlap = flat[a].intersection(flat[b])
    if overlap.area < 0.1 * min(flat[a].area, flat[b].area):
        raise ValueError(f"No substantial XY overlap for {pair}")
    probe = overlap.representative_point()
    covering = sorted(i for i, p in flat.items() if i not in pair and p.contains(probe))
    extra = set()
    # Include a distractor for each boundary when available, even when every
    # XY-probe face belongs to the hub. Selection is independent of style.
    selected_by_boundary = {}
    for target in pair:
        ranked = []
        for i, p in projected.items():
            if i in pair or i in covering or hub in memberships[i]:
                continue
            length = projected[target].boundary.intersection(p).length
            # When there is no positive-length overlap, include the nearest
            # nonmember. A point contact is distinct from covering a segment.
            distance = projected[target].boundary.distance(p)
            ranked.append((-length if length > step * 1e-7 else 0, distance, i))
        if ranked:
            negative_length, distance, chosen = min(ranked)
            extra.add(chosen)
            selected_by_boundary[target] = {
                "edge_id": chosen, "covered_length": -negative_length, "distance": distance,
            }
    context = sorted(set(covering) | extra)
    if not any(hub not in memberships[i] for i in context):
        raise ValueError(f"No nonmember context for {pair}")
    joints = [project(np.array([xy[hub]]), levels[i], step)[0].tolist() for i in pair] if common else []
    cloud = np.concatenate([np.asarray(projected[i].exterior.coords) for i in pair])
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
    width, height = np.ptp(cloud, axis=0)
    # Match the common display window exactly to the physical axes aspect.
    ratio = .448 * 10.8 / (.302 * 8.1)
    half_height = max(height, width / ratio) * .67
    pad = np.array([half_height * ratio, half_height])
    bounds = [*(center-pad), *(center+pad)]
    window = box(*bounds)
    return {
        "edge_ids": list(pair), "shared_vertex_ids": common, "layers": [levels[i] for i in pair],
        "xy_overlap_probe": [probe.x, probe.y], "xy_probe_context_ids": covering,
        "added_projected_nonmember_ids": sorted(extra), "boundary_neighbors": selected_by_boundary,
        "context_ids": context,
        "visible_context_ids": [i for i in context if window.intersection(projected[i]).area > 0],
        "context_without_hub": [i for i in context if hub not in memberships[i]],
        "contacts_on_target_faces": joints,
        "projected_bounds": bounds,
    }


def draw_panel(ax, row, projected, levels, outline):
    a, b = row["edge_ids"]
    x0, y0, x1, y1 = row["projected_bounds"]
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal", xticks=[], yticks=[], facecolor="#fbfcfc")
    for spine in ax.spines.values():
        spine.set_color("#cfdbdf")
    context = sorted(row["context_ids"], key=lambda i: (levels[i], i))
    patches = [Patch(projected[i].exterior.coords) for i in context]
    ax.add_collection(PatchCollection(
        patches, facecolors="none" if outline else "#648491",
        edgecolors="#79909a" if outline else "#4f7381",
        linewidths=.58 if outline else .38, alpha=.48 if outline else .21))
    for eid, color, line, marker in ((a, BLUE, "-", "o"), (b, ORANGE, "--", "s")):
        ring = np.asarray(projected[eid].exterior.coords)
        ax.fill(*ring.T, facecolor=color, alpha=.055 if outline else .48)
        ax.plot(*ring.T, color=color, lw=1.85 if outline else .85, ls=line if outline else "-",
                solid_joinstyle="round", dash_capstyle="round")
        if outline:
            ax.plot(*ring[:-1].T, ls="none", marker=marker, markersize=2.6,
                    markerfacecolor="white", markeredgewidth=.7, color=color)
        left = eid == a
        index = np.argmin(ring[:, 0]) if left else np.argmax(ring[:, 0])
        ax.annotate(f"E{eid}", xy=ring[index], xytext=(-4 if left else 4, 2),
                    textcoords="offset points", ha="right" if left else "left", va="bottom",
                    color=color, fontsize=9, weight="bold",
                    bbox=dict(fc="white", ec="none", alpha=.9, pad=1), zorder=12)
    if row["contacts_on_target_faces"]:
        contacts = np.asarray(row["contacts_on_target_faces"])
        ax.plot(*contacts.T, color="white", lw=4, zorder=7)
        ax.plot(*contacts.T, color=JOINT, lw=1.65, marker="o", markersize=4.8,
                markerfacecolor="white", markeredgewidth=1.4, zorder=8)


def render(name, layout, output):
    hub, pairs = CASES[name]
    xy = {i: [x,y] for i,x,y in layout["vertices"]}
    memberships = {i: nodes for i, _, nodes in layout["edges"]}
    levels = {i: z for i, z, _ in layout["edges"]}
    step = float(np.ptp(list(xy.values()), axis=0).max()) / 100
    flat, projected = {}, {}
    for i, ids in memberships.items():
        if len(set(ids)) < 3:
            continue
        points = np.asarray([xy[v] for v in ids])
        poly = Polygon(points)
        if poly.is_valid and poly.area > 0:
            flat[i] = poly
            projected[i] = Polygon(project(points, levels[i], step))
    fig = plt.figure(figsize=(10.8, 8.1), facecolor="white")
    title = "Original" if name == "original" else "DBLP TVCG"
    fig.text(.037, .95, f"{title}  |  Tracing polygon boundaries", size=22, color=INK, weight="bold")
    fig.text(.038, .909, "Two rendering styles applied to the same seed-0 geometry and surrounding faces.",
             color=MUTED, size=11.5)
    fig.text(.038, .852, "Filled faces", color=INK, size=15, weight="bold")
    fig.text(.515, .852, "Emphasized outlines", color=INK, size=15, weight="bold")
    rows = []
    for index, pair in enumerate(pairs):
        expected = [hub] if index == 0 else []
        row = row_geometry(pair, hub, expected, memberships, levels, xy, flat, projected, step)
        rows.append(row)
        bottom = .493 if index == 0 else .108
        truth = f"share V{hub}" if expected else "share no vertex"
        a, b = pair
        z_a, z_b = row["layers"]
        fig.text(.038, bottom+.316, f"E{a} and E{b} {truth}    ·    Layers {z_a} and {z_b}    ·    "
                 f"{len(row['visible_context_ids'])} of {len(row['context_ids'])} context faces in view",
                 color=INK, size=10.5)
        for left, outline in ((.038, False), (.515, True)):
            ax = fig.add_axes([left, bottom, .448, .302])
            draw_panel(ax, row, projected, levels, outline)
    fig.text(.038, .060, "Magenta marks the two selected faces’ shared vertex and the pillar segment between them.",
             size=10, color=INK)
    fig.text(.038, .032, "Gray shows nearby faces. Each row uses one camera, crop and scale. The panels are explanatory static renderings.",
             size=9.5, color=MUTED)
    fig.savefig(output, dpi=155)
    plt.close(fig)
    return {"full_vertices": len(xy), "full_edges": len(levels), "valid_faces": len(flat),
            "layer_step": step, "full_layout_sha256": HASHES[name],
            "numeric_geometry_sha256": geometry_digest(layout), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--dblp-layout", type=Path,
                        help="use the archived full seed-0 DBLP layout instead of the bundled ID-only copy")
    args = parser.parse_args()
    start = time.perf_counter()
    original = numeric_layout(ORIGINAL, "original")
    if args.dblp_layout:
        dblp = numeric_layout(args.dblp_layout, "dblp")
    else:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        dblp = record["public_dblp_geometry"]
        if geometry_digest(dblp) != record["datasets"]["dblp"]["numeric_geometry_sha256"]:
            raise ValueError("Bundled public DBLP geometry has changed.")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    data = {
        "experiment_id": "visnotes_polygon_plan_01", "source_report": "https://vdslab.esa.io/posts/9278",
        "projection": {"azimuth_degrees": AZIMUTH, "elevation_degrees": ELEVATION,
                       "z_step_as_fraction_of_full_xy_extent": .01},
        "seed": 0, "optimization_trials": 0, "budget_seconds": {"per_input": 120, "total": 300},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "python": platform.python_version(), "matplotlib": matplotlib.__version__, "shapely": shapely.__version__,
        "datasets": {},
    }
    for name, layout in (("original", original), ("dblp", dblp)):
        begin = time.perf_counter()
        result = render(name, layout, output / f"{name}-context.png")
        result["elapsed_seconds"] = round(time.perf_counter() - begin, 3)
        data["datasets"][name] = result
        if result["elapsed_seconds"] > 120 or time.perf_counter() - start > 300:
            raise TimeoutError("Figure generation exceeded the stated budget.")
    data["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    data["stop_reason"] = "Rendered two styles for four preselected pairs."
    # Keep the metadata readable and the numeric-only public source compact.
    metadata = json.dumps(data, ensure_ascii=False, indent=2)
    public = json.dumps(dblp, separators=(",", ":"))
    (output / "figure-data.json").write_text(metadata[:-1].rstrip() + ',\n  "public_dblp_geometry": '
                                             + public + "\n}\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
