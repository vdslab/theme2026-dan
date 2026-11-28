"""Measure unrelated faces along actual pillars and illustrate crowded edge pairs."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.visnotes_common import file_hash, read_json
from tools.make_visnotes_figures import project


def source_geometry(data):
    xy = {node["id"]: (node["x"], node["y"]) for node in data["nodes"]}
    rings, polygons, members = {}, {}, {}
    incident = defaultdict(set)
    for edge in data["hyperedges"]:
        eid = edge["id"]
        ids = set(edge["nodes"])
        members[eid] = ids
        for vid in ids:
            incident[vid].add(eid)
        if len(ids) >= 3:
            ring = np.array([xy[vid] for vid in edge["nodes"]])
            shape = Polygon(ring)
            if shape.is_valid and shape.area > 0:
                rings[eid] = ring
                polygons[eid] = shape
    return xy, members, incident, rings, polygons


def contacts(xy, incident, polygons, heights, hub):
    ids = list(polygons)
    spatial = STRtree(list(polygons.values()))
    aggregate = dict(shared_vertices=0, nonzero_pillars=0, affected_pillars=0,
                     nonmember_faces_on_pillars=0, contacts_at_incident_heights=0)
    focus = None
    for vid, edges in incident.items():
        if len(edges) < 2:
            continue
        aggregate["shared_vertices"] += 1
        stops = sorted({heights[eid] for eid in edges})
        crossings = []
        if len(stops) > 1:
            aggregate["nonzero_pillars"] += 1
            for i in spatial.query(Point(xy[vid]), predicate="within"):
                eid = ids[i]
                z = heights[eid]
                if eid not in edges and stops[0] <= z <= stops[-1]:
                    crossings.append({"edge": eid, "height": z, "at_incident_height": z in stops})
        aggregate["affected_pillars"] += bool(crossings)
        aggregate["nonmember_faces_on_pillars"] += len(crossings)
        aggregate["contacts_at_incident_heights"] += sum(c["at_incident_height"] for c in crossings)
        if vid == hub:
            focus = {"id": vid, "degree": len(edges), "incident_heights": stops,
                     "nonmember_contacts": sorted(crossings, key=lambda c: (c["height"], c["edge"]))}
    if focus is None:
        raise ValueError("The specified shared vertex is missing")
    return {"all": aggregate, "focus": focus}


def choose_pairs(polygons, members, hub):
    ids = list(polygons)
    spatial = STRtree(list(polygons.values()))
    selected = {}
    counts = defaultdict(int)
    for a, b in combinations(ids, 2):
        if hub not in members[a] and hub not in members[b]:
            continue
        common = members[a] & members[b]
        if common and common != {hub}:
            continue
        left, right = polygons[a], polygons[b]
        overlap = left.intersection(right)
        if overlap.area < 0.1 * min(left.area, right.area):
            continue
        point = overlap.representative_point()
        others = sum(ids[i] not in (a, b) for i in spatial.query(point, predicate="within"))
        kind = "shared" if common else "disjoint"
        counts[kind] += 1
        key = (-others, min(a, b), max(a, b))
        if kind not in selected or key < selected[kind][0]:
            selected[kind] = (key, {"edges": [a, b], "shared_ids": sorted(common),
                                     "other_faces_at_xy_probe": others})
    if set(selected) != {"shared", "disjoint"}:
        raise ValueError("An eligible positive or negative pair is missing")
    return {k: {**v[1], "eligible_pairs": counts[k]} for k, v in selected.items()}


def projected_hub_contacts(xy, rings, incident, heights, hub):
    """Count screen intersections with nonincident faces at two fixed elevations."""
    span = np.ptp(list(xy.values()), axis=0).max()
    focus = np.asarray(xy[hub])
    yaw = np.deg2rad(45)
    stops = [heights[eid] for eid in incident[hub]]
    counts = {}
    for degrees in (35, 75):
        elevation = np.deg2rad(degrees)
        z_unit = np.cos(elevation) / 100
        if min(stops) == max(stops):
            counts[str(degrees)] = 0
            continue
        segment = LineString([(0, z_unit * min(stops)), (0, z_unit * max(stops))])
        count = 0
        for eid, ring in rings.items():
            if eid in incident[hub]:
                continue
            x, y = ((ring - focus) / span).T
            u = -np.sin(yaw) * x + np.cos(yaw) * y
            v = -np.sin(elevation) * (np.cos(yaw) * x + np.sin(yaw) * y) + heights[eid] * z_unit
            count += Polygon(np.column_stack((u, v))).intersection(segment).length > 1e-9
        counts[str(degrees)] = count
    return counts


def draw_cases(output, title, xy, rings, pairs, methods, hub):
    step = np.ptp(np.asarray(list(xy.values())), axis=0).max() / 100
    figure, axes = plt.subplots(2, 3, figsize=(10.0, 5.8))
    figure.subplots_adjust(left=.18, right=.985, bottom=.12, top=.78, wspace=.08, hspace=.36)
    figure.suptitle(title + " | Shared vertex or projected overlap?", x=.025, ha="left",
                    fontsize=16, weight="bold", color="#18323f")
    figure.text(.026, .883, "Crowded pairs from the full layout. Each panel shows two edges. "
                "Only the selected shared vertex is marked.", fontsize=9.4)
    for ax, label in zip(axes[0], ("Planar XY", "First-fit", "Proposed"), strict=True):
        ax.set_title(label, size=12, weight="bold")
    colors = ("#006c7b", "#975326")
    for row, key in enumerate(("shared", "disjoint")):
        case = pairs[key]
        eids = case["edges"]
        center = np.asarray(xy[hub])
        panels = []
        for method in (None, *methods.values()):
            z = [0, 0] if method is None else [method[eid] for eid in eids]
            flat = method is None
            faces = [project(rings[eid] - center, level, step, flat)
                     for eid, level in zip(eids, z, strict=True)]
            joints = [project(np.zeros((1, 2)), level, step, flat)[0] for level in z]
            panels.append((faces, joints, z))
        cloud = [np.concatenate(p[0]) for p in panels]
        oblique = np.concatenate(cloud[1:])
        size = np.maximum(np.ptp(cloud[0], axis=0), np.ptp(oblique, axis=0))
        scale = max(size[0], size[1] * 1.45) * .56
        for col, (faces, joints, heights) in enumerate(panels):
            ax = axes[row, col]
            all_points = cloud[0] if col == 0 else oblique
            mid = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
            for i in sorted((0, 1), key=lambda idx: heights[idx]):
                ring = faces[i]
                closed = np.vstack((ring, ring[0]))
                ax.fill(*ring.T, color=colors[i], alpha=.12)
                ax.plot(*closed.T, color=colors[i], lw=1.2, ls=("-", "--")[i])
                ax.scatter(*ring.T, s=9, facecolors="white", edgecolors=colors[i],
                           marker=("o", "s")[i], linewidths=.6)
            if key == "shared":
                ax.plot(*np.asarray(joints).T, color="#b72441", lw=2.1, marker="o",
                        markersize=4.0, markerfacecolor="white", zorder=9)
            ax.set(xlim=(mid[0]-scale, mid[0]+scale), ylim=(mid[1]-scale/1.45, mid[1]+scale/1.45),
                   aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col:
                ax.set_xlabel(f"Layers {heights[0]} and {heights[1]}", fontsize=9)
        y = (.636, .313)[row]
        figure.text(.025, y, "Shared" if key == "shared" else "No shared ID", size=12, weight="bold")
        figure.text(.025, y-.05, f"Edges {eids[0]} and {eids[1]}", size=9)
        figure.text(.025, y-.093, f"{case['other_faces_at_xy_probe']} other faces\nat XY probe", size=8)
    legends = [Line2D([], [], color=colors[0], marker="o", label="First edge"),
               Line2D([], [], color=colors[1], ls="--", marker="s", label="Second edge"),
               Line2D([], [], color="#b72441", marker="o", label=f"Shared vertex {hub} and pillar")]
    figure.legend(handles=legends, loc="lower center", frameon=False, ncol=3)
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def run(source, baseline, proposal, hub, name, output):
    data = read_json(source)
    xy, members, incident, rings, polygons = source_geometry(data)
    methods = {}
    for label, path in (("first-fit", baseline), ("proposed", proposal)):
        layout = read_json(path)
        if layout["nodes"] != data["nodes"]:
            raise ValueError(f"{label} changes vertex records")
        ordered = [(e["id"], e["nodes"]) for e in layout["hyperedges"]]
        if ordered != [(e["id"], e["nodes"]) for e in data["hyperedges"]]:
            raise ValueError(f"{label} changes ordered memberships")
        methods[label] = {e["id"]: e["layer"] for e in layout["hyperedges"]}
    pairs = choose_pairs(polygons, members, hub)
    result = {"dataset": name, "source_hash": file_hash(source), "layout_hashes": {
        "first-fit": file_hash(baseline), "proposed": file_hash(proposal)},
        "valid_faces": len(polygons), "all_edges": len(members), "pairs": pairs,
        "methods": {k: contacts(xy, incident, polygons, heights, hub) for k, heights in methods.items()}}
    for name_key, heights in methods.items():
        result["methods"][name_key]["focus"]["nonmember_screen_intersections"] = projected_hub_contacts(
            xy, rings, incident, heights, hub)
    output.mkdir(exist_ok=False, parents=True)
    draw_cases(output / "crowded-pairs.png", name, xy, rings, pairs, methods, hub)
    (output / "contacts.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "first-fit", "proposed", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--focus", type=int, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.source, args.first_fit, args.proposed, args.focus, args.name, args.output)
