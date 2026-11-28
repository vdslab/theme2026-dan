"""Audit complete saved layouts and the use of projected face overlap as a membership cue."""

import argparse
from collections import Counter
from itertools import groupby
from math import comb, cos, radians, sin
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import REG, THREAD_LIMITS, file_hash, make_geometry, read_json, root_relative, save_json
from experiments.visnotes_metrics import read_layout_metrics
from tools.make_visnotes_figures import first_fit_levels
import numpy as np
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Polygon


RUNS = ROOT / "results/experiments"
PUBLIC = RUNS / "visnotes_public_validation_01"
EXPERIMENT = "visnotes_dense_validation_01"
AREA_EPS = 1e-12
VIEWS = (35, 75)
INPUTS = {
    "original": (ROOT / "original.json", RUNS / "visnotes_remaining_01/exp04/original"),
    "dblp": (PUBLIC / "inputs/dblp.json", PUBLIC / "public-main/dblp"),
}


def check_time(deadline):
    if monotonic() >= deadline:
        raise TimeoutError("Verification time budget exhausted.")


def prepare_faces(geometry):
    """Normalize all source XY once. Record every excluded edge and optional repair."""
    data = geometry.data
    xy = {n["id"]: (n["x"], n["y"]) for n in data["nodes"]}
    points = np.asarray(list(xy.values()))
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    span = max(np.ptp(points, axis=0)) or 1
    normalized = {key: tuple((point - center) / span) for key, point in xy.items()}
    faces = []
    counts = Counter()
    for i, edge in enumerate(data["hyperedges"]):
        degree = len(set(edge["nodes"]))
        if degree < 3:
            counts[f"degree_{degree}"] += 1
            continue
        ring = [p for p, _ in groupby(normalized[n] for n in edge["nodes"])]
        if ring and ring[-1] == ring[0]:
            ring.pop()
        if len(set(ring)) < 3:
            counts["degenerate"] += 1
            continue
        polygon = Polygon(ring)
        valid = polygon.is_valid and polygon.area > 0
        if not valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            counts["empty_after_repair"] += 1
            continue
        counts["valid" if valid else "repaired_nonempty"] += 1
        faces.append((i, polygon, valid))
    return faces, {k: counts[k] for k in (
        "degree_0", "degree_1", "degree_2", "degenerate", "empty_after_repair", "valid", "repaired_nonempty",
    )}


def projection_counts(geometry, faces, levels, method, seed, elevation):
    pitch, yaw = radians(elevation), radians(45)
    projected = []
    for index, polygon, _ in faces:
        if method == "planar":
            projected.append(polygon)
        else:
            matrix = [-sin(yaw), cos(yaw), -cos(yaw) * sin(pitch), -sin(yaw) * sin(pitch),
                      0, levels[index] * cos(pitch) / 100]
            projected.append(affine_transform(polygon, matrix))

    shapes = np.asarray(projected, dtype=object)
    indices = np.asarray([i for i, _, _ in faces], dtype=int)
    strict = np.asarray([valid for _, _, valid in faces], dtype=bool)
    shared = np.zeros((geometry.size, geometry.size), dtype=bool)
    for i, j, count in geometry.pair_counts:
        shared[i, j] = bool(count)
    overlap = np.zeros((len(faces), len(faces)), dtype=bool)
    touches = np.zeros_like(overlap)
    left, right = shapely.STRtree(shapes).query(shapes, predicate="intersects")
    keep = left < right
    left, right = left[keep], right[keep]
    intersection = shapely.intersection(shapes[left], shapes[right])
    overlap[left, right] = shapely.area(intersection) > AREA_EPS
    touches[left, right] = shapely.touches(shapes[left], shapes[right])

    rows = []
    for scope, allowed in (("valid", strict), ("repaired", np.ones_like(strict))):
        a, b = np.triu_indices(len(faces), k=1)
        selected = allowed[a] & allowed[b]
        a, b = a[selected], b[selected]
        hit, truth = overlap[a, b], shared[indices[a], indices[b]]
        classification = {"tp": hit & truth, "fp": hit & ~truth, "fn": ~hit & truth, "tn": ~hit & ~truth}
        examples = {}
        for key, mask in classification.items():
            matches = np.flatnonzero(mask)
            if matches.size:
                first = matches[0]
                examples[key] = [geometry.data["hyperedges"][indices[j]]["id"] for j in (a[first], b[first])]
        rows.append({
            "scope": scope, "method": method, "seed": seed, "elevation": elevation,
            "azimuth": None if method == "planar" else 45, "faces": int(allowed.sum()), "pairs": len(a),
            **{k: int(v.sum()) for k, v in classification.items()},
            "boundary_touch_with_shared_id": int(np.count_nonzero(touches[a, b] & truth)),
            "first_pair_ids_in_input_order": examples,
        })
    return rows


def vertex_spans(data, levels):
    incident = {n["id"]: [] for n in data["nodes"]}
    for edge, level in zip(data["hyperedges"], levels, strict=True):
        for node in set(edge["nodes"]):
            incident[node].append(level)
    return {node: (len(z), max(z) - min(z) if z else 0) for node, z in incident.items()}


def span_changes(before, after, seed):
    totals = {"improved": 0, "equal": 0, "worse": 0}
    by_degree = {}
    for node, (degree, previous_span) in before.items():
        if degree < 2:
            continue
        difference = after[node][1] - previous_span
        trend = "equal"
        if difference < 0:
            trend = "improved"
        elif difference > 0:
            trend = "worse"
        totals[trend] += 1
        group = by_degree.setdefault(str(degree), {"improved": 0, "equal": 0, "worse": 0})
        group[trend] += 1
    return {"seed": seed, "shared_vertices": sum(totals.values()), **totals, "by_degree": by_degree}


def analyze(name, source, case_root, seeds, deadline):
    started = monotonic()
    geometry = make_geometry(source, REG)
    data = geometry.data
    if any(len({row["id"] for row in data[k]}) != len(data[k]) for k in ("nodes", "hyperedges")):
        raise ValueError(f"{name}: source IDs are not unique.")
    faces, coverage = prepare_faces(geometry)
    profile = {
        "vertices": len(data["nodes"]), "hyperedges": geometry.size, "edge_pairs": comb(geometry.size, 2),
        "shared_edge_pairs": sum(w > 0 for _, _, w in geometry.pair_counts), "xy_constraint_pairs": len(geometry.overlaps),
        "face_population": coverage, "edges_with_duplicate_members": sum(
            len(e["nodes"]) != len(set(e["nodes"])) for e in data["hyperedges"]
        ),
    }
    rows = projection_counts(geometry, faces, None, "planar", None, 90)
    layouts, changes = [], []
    reference = {}
    for seed in (None, *seeds):
        check_time(deadline)
        condition, subdir = ("greedy-input", "deterministic") if seed is None else ("full", f"seed-{seed}")
        method = "first-fit" if seed is None else "proposed"
        location = case_root / condition / subdir
        case = read_json(location / "case.json")
        if (case["input"], case["condition"], case["seed"], case["status"], case["input_sha256"]) != (
            name, condition, seed, "verified", geometry.hashes["input_sha256"]
        ):
            raise ValueError(f"Unexpected saved case or input hash at {location}")
        layout_path = location / "layout.json"
        levels, metrics = read_layout_metrics(read_json(layout_path), geometry, method)
        if metrics != case["metrics"]:
            raise ValueError(f"Recomputed metrics differ from {location / 'case.json'}")
        spans = vertex_spans(data, levels)
        if sum(span for _, span in spans.values()) != metrics["pillar_length"]:
            raise ValueError(f"{location}: incident-level and geometry totals differ.")
        if seed is None:
            if levels != first_fit_levels(geometry):
                raise ValueError("Baseline does not follow input-order first-fit.")
            reference = spans
        else:
            changes.append(span_changes(reference, spans, seed))
        layouts.append({
            "method": method, "seed": seed, "layers": metrics["used_layer_count"],
            "pillar_length": metrics["pillar_length"], "height_range": metrics["height_range"],
            "l2_scaled_qp": metrics["l2_scaled_qp"], "constraint_violations": metrics["overlap_violation_count"],
            "layout_path": root_relative(layout_path), "layout_sha256": file_hash(layout_path),
            "case_sha256": file_hash(location / "case.json"), "verified_nodes_membership_and_xy": True,
            "solver_wall_seconds": case["solver_wall_seconds"], "iterations": case["iterations"],
            "solver_stop": case["termination_reason"],
            "focus_vertex": metrics["constrained_hubs"][0],
        })
        for elevation in VIEWS:
            check_time(deadline)
            rows.extend(projection_counts(geometry, faces, levels, method, seed, elevation))
    check_time(deadline)
    proposed = [row for row in layouts if row["method"] == "proposed"]
    ranges = {}
    for metric in ("layers", "pillar_length", "l2_scaled_qp", "solver_wall_seconds"):
        values = [row[metric] for row in proposed]
        ranges[metric] = {"min": min(values), "median": median(values), "max": max(values)}
    return {
        "profile": profile, "input_sha256": geometry.hashes["input_sha256"], "input_path": root_relative(source),
        "layouts": layouts, "proposed_seed_statistics": ranges, "span_changes": changes,
        "projections": rows, "elapsed_seconds": monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new directory")
    parser.add_argument("--inputs", nargs="+", choices=tuple(INPUTS), default=list(INPUTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    if len(set(args.inputs)) != len(args.inputs) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Repeated input or seed.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets = [Path(__file__).resolve(), ROOT / "experiments/visnotes_common.py",
               ROOT / "experiments/visnotes_metrics.py", ROOT / "tools/make_visnotes_figures.py",
               ROOT / "experiments/tree_optimization_comparison_01/review_results.py"]
    for name in args.inputs:
        source, case_root = INPUTS[name]
        targets.append(source)
        for folder in [case_root / "greedy-input/deterministic", *(case_root / "full" / f"seed-{s}" for s in args.seeds)]:
            targets.extend([folder / "layout.json", folder / "case.json"])
    hashes = {root_relative(p): file_hash(p) for p in targets}
    manifest = {
        "experiment": EXPERIMENT, "inputs": args.inputs, "seeds": args.seeds,
        "views": {"planar": "source XY", "stacked": [[45, e] for e in VIEWS], "z_step_per_xy_span": 0.01},
        "normalized_intersection_area_strictly_greater_than": AREA_EPS,
        "truth": "share one or more distinct input vertex IDs", "classification": "positive-area pair predicts sharing",
        "repaired_scope": "all valid polygons plus nonempty positive-area buffer(0) repairs",
        "input_budget_seconds": 180, "total_budget_seconds": 600, "solver_reruns": 0,
        "sources_sha256": hashes, "python": sys.executable, "python_version": platform.python_version(),
        "shapely": shapely.__version__, "geos": shapely.geos_version_string, "numpy": np.__version__,
        "thread_limits": THREAD_LIMITS, "command": [sys.executable, "-B", *sys.argv],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain", "--", *hashes], cwd=ROOT, text=True).splitlines(),
    }
    save_json(output / "manifest.json", manifest)
    start = monotonic()
    summary = {"complete": False, "datasets": {}}
    try:
        for name in args.inputs:
            summary["datasets"][name] = analyze(name, *INPUTS[name], args.seeds, min(start + 600, monotonic() + 180))
        if any(file_hash(ROOT / path) != expected for path, expected in hashes.items()):
            raise ValueError("Inputs or sources changed during verification.")
        summary["complete"] = True
        summary["stop_reason"] = "all_planned_cases_verified"
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        summary["stop_reason"] = str(exc)
        raise
    finally:
        summary["elapsed_seconds"] = monotonic() - start
        save_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
