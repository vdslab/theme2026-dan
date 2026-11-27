"""Check whether projected overlap changes with camera angle or raster resolution."""

import argparse
from itertools import groupby
from math import cos, radians, sin
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import REG, file_hash, make_geometry, read_json, save_json
from experiments.visnotes_metrics import projection_metrics, read_layout_metrics
from shapely import STRtree
from shapely.affinity import affine_transform
from shapely.geometry import Polygon


VIEWS = ((45, 35), (0, 35), (90, 35), (45, 15), (45, 75), (45, 90))
AREA_TOLERANCE = 1e-12


def projectable_faces(data):
    xy = {node["id"]: (node["x"], node["y"]) for node in data["nodes"]}
    columns = list(zip(*xy.values(), strict=True))
    spans = [max(values) - min(values) for values in columns]
    span = max(spans) or 1
    centers = [(max(values) + min(values)) / 2 for values in columns]
    normalized = {key: tuple((coord - center) / span for coord, center in zip(point, centers, strict=True))
                  for key, point in xy.items()}
    faces = []
    for index, edge in enumerate(data["hyperedges"]):
        ring = [point for point, _ in groupby(normalized[key] for key in edge["nodes"])]
        if ring and ring[-1] == ring[0]:
            ring.pop()
        if len(ring) < 3:
            continue
        polygon = Polygon(ring)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.area > 0:
            faces.append((index, polygon))
    return faces


def positive_area_pairs(faces, levels, azimuth, elevation):
    yaw, pitch = map(radians, (azimuth, elevation))
    x = (-sin(yaw), cos(yaw))
    y = (-cos(yaw) * sin(pitch), -sin(yaw) * sin(pitch))
    scale = cos(pitch) / 100
    projected = [
        affine_transform(polygon, [*x, *y, 0, levels[index] * scale])
        for index, polygon in faces
    ]
    pairs = STRtree(projected).query(projected, predicate="intersects").T
    return int(sum(a < b and projected[a].intersection(projected[b]).area > AREA_TOLERANCE
                   for a, b in pairs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve(strict=True)
    manifest = read_json(run / "manifest.json")
    summary = read_json(run / "summary.json")
    if not summary["sources_unchanged"]:
        raise ValueError("The benchmark sources changed during measurement.")
    rows = []
    for name, source in manifest["inputs"].items():
        geometry = make_geometry(ROOT / source["path"], REG)
        if geometry.hashes["input_sha256"] != source["sha256"]:
            raise ValueError(f"{name}: input does not match its manifest.")
        faces = projectable_faces(geometry.data)
        for condition, seed in (("greedy-input", None), ("full", 0)):
            case, = (row for row in summary["cases"] if row["input"] == name
                     and row["condition"] == condition and row["seed"] == seed and row["status"] == "verified")
            path = ROOT / case["case_directory"] / "layout.json"
            levels, _ = read_layout_metrics(read_json(path), geometry, condition)
            for azimuth, elevation in VIEWS:
                raster = projection_metrics(geometry, levels, azimuth, elevation)
                rows.append({
                    "input": name, "condition": condition, "seed": seed,
                    "view": [azimuth, elevation], "layout_sha256": file_hash(path),
                    "positive_area_face_pairs": positive_area_pairs(faces, levels, azimuth, elevation),
                    "covered_pixels": raster["covered_pixel_count"],
                    "multicovered_pixels": raster["overdraw_pixel_count"],
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {
        "benchmark_summary_sha256": file_hash(run / "summary.json"),
        "code_sha256": file_hash(Path(__file__)),
        "normalized_area_tolerance": AREA_TOLERANCE, "pixels_per_input_span": 384,
        "cases": rows,
    })


if __name__ == "__main__":
    main()
