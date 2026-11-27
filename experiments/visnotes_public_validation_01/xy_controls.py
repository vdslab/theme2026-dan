"""Cross the coordinates and ordered boundaries of two layouts of one input."""

import argparse
from collections import Counter
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import file_hash, read_json, root_relative, save_json
from shapely.geometry import Polygon


def index_rows(rows):
    values = {row["id"]: row for row in rows}
    if len(values) != len(rows):
        raise ValueError("Input contains duplicate IDs.")
    return values


def crossed_layout(base, alternate, use_alternate_xy, use_alternate_order):
    base_nodes, alt_nodes = (index_rows(data["nodes"]) for data in (base, alternate))
    base_edges, alt_edges = (index_rows(data["hyperedges"]) for data in (base, alternate))
    if base_nodes.keys() != alt_nodes.keys() or base_edges.keys() != alt_edges.keys():
        raise ValueError("The variants must contain the same vertex and hyperedge IDs.")
    if any(Counter(edge["nodes"]) != Counter(alt_edges[key]["nodes"])
           for key, edge in base_edges.items()):
        raise ValueError("The variants must contain the same memberships, including multiplicities.")
    positions = alt_nodes if use_alternate_xy else base_nodes
    orders = alt_edges if use_alternate_order else base_edges
    return {
        "nodes": [dict(node, x=positions[key]["x"], y=positions[key]["y"])
                  for key, node in base_nodes.items()],
        "hyperedges": [
            {**{key: value for key, value in edge.items() if key != "layer"},
             "nodes": orders[edge_id]["nodes"]}
            for edge_id, edge in base_edges.items()
        ],
    }


def geometry_audit(data):
    coordinates = {node["id"]: (node["x"], node["y"]) for node in data["nodes"]}
    invalid, collapsed, empty_after_repair = [], [], []
    for edge in data["hyperedges"]:
        if len(set(edge["nodes"])) < 3:
            continue
        ring = [coordinates[node_id] for node_id in edge["nodes"]]
        if len(set(ring)) < 3:
            collapsed.append(edge["id"])
            continue
        polygon = Polygon(ring)
        if not polygon.is_valid:
            invalid.append(edge["id"])
            if polygon.buffer(0).is_empty:
                empty_after_repair.append(edge["id"])
    return {
        "invalid_polygon_ids": invalid, "collapsed_polygon_ids": collapsed,
        "empty_after_buffer_zero_ids": empty_after_repair,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("alternate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--common-valid", action="store_true",
                        help="Exclude the same invalid or collapsed faces from all four cells.")
    args = parser.parse_args(argv)
    base, alternate = (read_json(path) for path in (args.base, args.alternate))
    cells = []
    for alt_xy, alt_order in product((False, True), repeat=2):
        name = f"{'alternate' if alt_xy else 'base'}-xy-{'alternate' if alt_order else 'base'}-order"
        cells.append((name, alt_xy, alt_order, crossed_layout(base, alternate, alt_xy, alt_order)))
    unfiltered_audits = {name: geometry_audit(data) for name, _, _, data in cells}
    excluded = set()
    if args.common_valid:
        excluded = {
            edge_id for entry in unfiltered_audits.values()
            for kind in ("invalid_polygon_ids", "collapsed_polygon_ids")
            for edge_id in entry[kind]
        }
        for _, _, _, data in cells:
            data["hyperedges"] = [edge for edge in data["hyperedges"] if edge["id"] not in excluded]
            if not data["hyperedges"]:
                raise ValueError("No hyperedges remain in the common valid subset.")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = {}
    for name, alt_xy, alt_order, data in cells:
        path = args.output / f"{name}.json"
        save_json(path, data)
        rows[name] = {
            "file": path.name, "sha256": file_hash(path),
            "coordinate_source": "alternate" if alt_xy else "base",
            "boundary_order_source": "alternate" if alt_order else "base",
            **geometry_audit(data),
        }
    save_json(args.output / "manifest.json", {
        "base": {"file": root_relative(args.base.resolve()), "sha256": file_hash(args.base)},
        "alternate": {"file": root_relative(args.alternate.resolve()), "sha256": file_hash(args.alternate)},
        "preparation_code_sha256": file_hash(Path(__file__)),
        "common_valid_subset": args.common_valid,
        "excluded_edge_ids": [edge["id"] for edge in base["hyperedges"] if edge["id"] in excluded],
        "unfiltered_geometry_audits": unfiltered_audits,
        "vertices_with_changed_coordinates": sum(
            (a["x"], a["y"]) != (b["x"], b["y"])
            for a, b in zip(cells[0][3]["nodes"], cells[-1][3]["nodes"], strict=True)
        ),
        "hyperedges_with_changed_member_order": sum(
            a["nodes"] != b["nodes"]
            for a, b in zip(cells[0][3]["hyperedges"], cells[-1][3]["hyperedges"], strict=True)
        ),
        "cells": rows,
    })


if __name__ == "__main__":
    main()
