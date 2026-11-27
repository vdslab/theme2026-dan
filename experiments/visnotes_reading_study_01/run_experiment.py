"""Prepare local, counterbalanced 2D and 2.5D reading-task stimuli."""

import argparse
from hashlib import sha256
from itertools import permutations
import json
from math import cos, radians, sin
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
from shapely import __version__ as shapely_version
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import REG, make_geometry, read_json, save_json
from experiments.visnotes_metrics import read_layout_metrics
from tools.make_visnotes_figures import first_fit_levels

EXPERIMENT = "visnotes_reading_study_01"
STIMULUS_VERSION = 1
CONDITIONS = ("planar", "first_fit", "proposed")
SLOTS = 18
WIDTH, HEIGHT = 900, 520
MARGIN = 45
GUIDE_CLEARANCE = 8
YAW, PITCH = radians(45), radians(35)


def canonical_hash(content):
    return sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def screen_points(points, layer, step, planar):
    if planar:
        return points
    x, y = points.T
    return np.column_stack((
        -sin(YAW) * x + cos(YAW) * y,
        -sin(PITCH) * (cos(YAW) * x + sin(YAW) * y) + cos(PITCH) * layer * step,
    ))


def item_panels(target, options, polygons, xy, layouts, step):
    panels = {}
    origin = np.asarray(xy[target])
    target_xy = np.zeros((1, 2))
    visible = [levels[edge] for levels in layouts.values() for edge in options]
    min_layer, max_layer = min(visible), max(visible)
    for condition in CONDITIONS:
        flat = condition == "planar"
        layers = (0,) * len(options) if flat else tuple(layouts[condition][edge] for edge in options)
        faces = []
        for code, edge, layer in zip("ABCD", options, layers, strict=True):
            shape = polygons[edge]
            coordinates = np.asarray(shape.exterior.coords[:-1])
            interior = np.asarray(shape.representative_point().coords[:])
            points = screen_points(np.vstack((coordinates, interior)) - origin,
                                   layer - min_layer, step, flat)
            faces.append((layer, code, points))
        guide_levels = (0,) if flat else (0, max_layer - min_layer)
        guide = np.concatenate([screen_points(target_xy, k, step, flat) for k in guide_levels])
        panels[condition] = (sorted(faces, key=lambda face: (face[0], face[1])), guide)

    # Keep one bounding domain and one units-to-pixels ratio across all three conditions.
    extent = np.concatenate([points for faces, guide in panels.values()
                             for points in (*[face[2] for face in faces], guide)])
    span = np.ptp(extent, axis=0)
    center = (extent.min(axis=0) + extent.max(axis=0)) / 2
    scale = min((WIDTH - 2 * MARGIN) / max(span[0], step),
                (HEIGHT - 2 * MARGIN) / max(span[1], step))

    def normalized(points):
        result = (points - center) * scale
        result *= (1, -1)
        result += (WIDTH / 2, HEIGHT / 2)
        return np.round(result, 3).tolist()

    return {
        condition: {
            "faces": [{"code": code, "ring": normalized(points[:-1]),
                       "anchor": normalized(points[-1:])[0]}
                      for _, code, points in faces],
            "guide": normalized(guide),
        }
        for condition, (faces, guide) in panels.items()
    }


def corners_clear_guide(panels, correct_code):
    for panel in panels.values():
        start, end = map(np.asarray, (panel["guide"][0], panel["guide"][-1]))
        vector = end - start
        squared_length = vector @ vector
        for face in panel["faces"]:
            if face["code"] == correct_code:
                continue
            offsets = np.asarray(face["ring"]) - start
            fractions = np.clip(offsets @ vector / squared_length, 0, 1) if squared_length else 0
            nearest = np.outer(fractions, vector) if squared_length else 0
            if (np.linalg.norm(offsets - nearest, axis=1) < GUIDE_CLEARANCE).any():
                return False
    return True


def generate_items(data, layouts, count, seed):
    xy = {entry["id"]: (entry["x"], entry["y"]) for entry in data["nodes"]}
    edges = data["hyperedges"]
    polygons = {}
    membership = {}
    for index, edge in enumerate(edges):
        members = set(edge["nodes"])
        if len(members) < 3:
            continue
        polygon = Polygon([xy[node] for node in edge["nodes"]])
        if polygon.is_valid and polygon.area > 0:
            polygons[index] = polygon
            membership[index] = members

    rng = random.Random(seed)
    targets = list(xy)
    rng.shuffle(targets)
    letters = rng.sample(list("ABCD"), 4)
    correct_codes = [letters[index % len(letters)] for index in range(count)]
    rng.shuffle(correct_codes)
    used_faces, items, key = set(), [], []
    step = float(np.ptp(np.asarray(list(xy.values())), axis=0).max()) / 100
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Input must have a positive finite XY span.")

    for target in targets:
        remaining = [edge for edge in polygons if edge not in used_faces]
        incident = [edge for edge in remaining if target in membership[edge]]
        other = [edge for edge in remaining if target not in membership[edge]
                 and all(xy[member] != xy[target] for member in membership[edge])]
        all_incident = sum(target in members for members in membership.values())
        if all_incident < 2 or not incident or len(other) < 3:
            continue
        point = Point(xy[target])
        # Prefer unrelated polygons lying close to the selected shared vertex.
        distractors = sorted(other, key=lambda edge: (
            polygons[edge].distance(point), polygons[edge].centroid.distance(point), edge,
        ))[:3]
        correct = rng.choice(incident)
        correct_code = correct_codes[len(items)]
        rng.shuffle(distractors)
        choices = distractors
        choices.insert("ABCD".index(correct_code), correct)
        panels = item_panels(target, choices, polygons, xy, layouts, step)
        if not corners_clear_guide(panels, correct_code):
            continue
        item = f"T{len(items) + 1:02d}"
        items.append({"id": item, "panels": panels})
        key.append({"id": item, "target_node_id": target,
                    "choices": {code: edges[edge]["id"] for code, edge in zip("ABCD", choices, strict=True)},
                    "correct": correct_code})
        used_faces.update(choices)
        if len(items) == count:
            return items, key
    raise ValueError(f"Could only generate {len(items)} of {count} tasks with nonrepeating targets and faces.")


def assignments(items, seed):
    orders = list(permutations(CONDITIONS))
    result = []
    for slot in range(SLOTS):
        rotation, order = slot // len(orders), orders[slot % len(orders)]
        parts = {condition: [] for condition in CONDITIONS}
        for index, item in enumerate(items):
            condition = CONDITIONS[(index + rotation) % len(CONDITIONS)]
            parts[condition].append({"item": item["id"], "condition": condition})
        rng = random.Random(seed + slot + 1)
        for group in parts.values():
            rng.shuffle(group)
        result.append([trial for condition in order for trial in parts[condition]])
    return result


def plan(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--first-fit", type=Path, required=True)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--trials-per-condition", type=int, choices=range(1, 11), default=2)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be a nonnegative 32-bit integer")
    return args


def main(argv=None):
    started = perf_counter()
    args = plan(argv)
    geometry = make_geometry(args.source, REG)
    first_fit, _ = read_layout_metrics(read_json(args.first_fit), geometry, "first-fit")
    proposed, _ = read_layout_metrics(read_json(args.proposed), geometry, "proposed")
    if tuple(first_fit) != first_fit_levels(geometry):
        raise ValueError("Baseline layers do not match input-order first-fit.")
    inputs = {name: sha256(path.read_bytes()).hexdigest() for name, path in (
        ("source", args.source), ("first_fit", args.first_fit), ("proposed", args.proposed),
    )}
    items, key = generate_items(
        geometry.data, {"first_fit": first_fit, "proposed": proposed},
        args.trials_per_condition * len(CONDITIONS), args.seed,
    )
    content = {"schema": STIMULUS_VERSION, "experiment": EXPERIMENT, "seed": args.seed,
               "trials_per_condition": args.trials_per_condition, "allocation_slots": SLOTS,
               "width": WIDTH, "height": HEIGHT, "items": items, "input_sha256": inputs,
               "guide_clearance_px": GUIDE_CLEARANCE,
               "assignments": assignments(items, args.seed), "key_sha256": canonical_hash(key)}
    study_id = canonical_hash(content)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "study.json", dict(content, study_id=study_id))
    save_json(args.output / "key.json", {"schema": STIMULUS_VERSION, "study_id": study_id, "items": key})
    tracked = [Path(__file__).parent / name
               for name in ("run_experiment.py", "score.py", "index.html", "study.js", "study.css")]
    sources = {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
               for path in tracked}
    save_json(args.output / "manifest.json", {
        "experiment": EXPERIMENT, "study_id": study_id, "input_sha256": inputs,
        "source_sha256": sources, "argv": sys.argv if argv is None else argv,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *sources],
            cwd=ROOT, text=True,
        ).splitlines(),
        "python": platform.python_version(), "numpy": np.__version__, "shapely": shapely_version,
        "elapsed_seconds": perf_counter() - started, "termination_reason": "stimulus_generated",
    })
    print(f"{len(items)} tasks saved to {args.output} ({study_id})")


if __name__ == "__main__":
    main()
