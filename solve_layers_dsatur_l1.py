#!/usr/bin/env python3
"""Compare L1 and L2 objectives while keeping solve_layers.py's DSatur order.

The LP minimizes sum_{i<j} (W_ij + reg) * abs(x_i - x_j), with x_0 = 0
and the same directed overlap constraints as solve_layers.py. Heights remain
continuous; hyperedges of the same color are not forced to share a height.

Save a QP baseline, then solve and compare the L1 layout:
    python solve_layers.py original.json layered_hypergraph_l2.json
    python solve_layers_dsatur_l1.py --compare layered_hypergraph_l2.json

The default output, layered_hypergraph.json, can be opened in index.html.
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from shapely.geometry import Polygon

from solve_layers import dsatur_coloring


CONSTRAINT_TOLERANCE = 1e-5


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", nargs="?", default="original.json")
    parser.add_argument("output_file", nargs="?", default="layered_hypergraph.json")
    parser.add_argument(
        "--reg", type=float, default=1e-5,
        help="Weight added to EVERY off-diagonal pair, as in solve_layers.py (default: 1e-5)",
    )
    parser.add_argument(
        "--compare", metavar="QP_JSON",
        help="Saved solve_layers.py output to evaluate with the same weights and constraints",
    )
    return parser.parse_args()


def extract_overlaps(nodes, hyperedges):
    """Use the same polygon construction, repair and threshold as solve_layers.py."""
    node_by_id = {node["id"]: node for node in nodes}
    polygons = []
    for he in hyperedges:
        pts = []
        for nid in he.get("nodes", []):
            if nid in node_by_id:
                pt = (node_by_id[nid]["x"], node_by_id[nid]["y"])
                if not pts or pts[-1] != pt:
                    pts.append(pt)
            else:
                print(
                    f"Warning: Node ID {nid} in hyperedge {he['id']} not found in nodes list.",
                    file=sys.stderr,
                )
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts.pop()
        if len(pts) < 3:
            polygons.append(None)
            continue
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            polygons.append(poly)
        except Exception as exc:
            print(
                f"Warning: Could not create valid polygon for hyperedge {he['id']}: {exc}",
                file=sys.stderr,
            )
            polygons.append(None)

    bounds = [
        poly.bounds if poly is not None and not poly.is_empty else None
        for poly in polygons
    ]
    overlaps = []
    for i, poly_i in enumerate(polygons):
        if bounds[i] is None:
            continue
        minx_i, miny_i, maxx_i, maxy_i = bounds[i]
        for j in range(i + 1, len(polygons)):
            if bounds[j] is None:
                continue
            minx_j, miny_j, maxx_j, maxy_j = bounds[j]
            if maxx_i < minx_j or minx_i > maxx_j or maxy_i < miny_j or miny_i > maxy_j:
                continue
            poly_j = polygons[j]
            if poly_i.intersects(poly_j) and poly_i.intersection(poly_j).area > 1e-9:
                overlaps.append((i, j))
    return overlaps


def compute_weights(hyperedges, reg_val):
    n = len(hyperedges)
    memberships = [set(he.get("nodes", [])) for he in hyperedges]
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            weights[i, j] = weights[j, i] = len(memberships[i] & memberships[j])
    regularized = weights + reg_val
    np.fill_diagonal(regularized, 0.0)
    laplacian = np.diag(np.sum(regularized, axis=1)) - regularized
    largest = float(np.max(np.abs(laplacian)))
    # A single hyperedge (or all-zero weights with --reg 0) has zero cost.
    scale = 1.0 / largest if largest > 0 else 1.0
    return weights, regularized, scale


def orient_overlaps(n, overlaps, colors):
    sorted_indices = sorted(range(n), key=lambda idx: (colors[idx], idx))
    pos = {idx: rank for rank, idx in enumerate(sorted_indices)}
    directed = [
        (i, j) if pos[i] < pos[j] else (j, i)
        for i, j in overlaps
    ]
    return np.asarray(directed, dtype=int).reshape(-1, 2)


def build_lp(regularized, directed):
    """Build a sparse LP with x_i heights and t_ij >= abs(x_i - x_j)."""
    n = len(regularized)
    pair_i, pair_j = np.triu_indices(n, k=1)
    active = regularized[pair_i, pair_j] > 0
    pair_i, pair_j = pair_i[active], pair_j[active]
    m = len(pair_i)
    k = len(directed)
    variables = n + m
    c = np.zeros(variables, dtype=float)
    c[n:] = regularized[pair_i, pair_j]

    overlap_rows = np.repeat(np.arange(k), 2)
    overlap_cols = directed.ravel()
    overlap_values = np.tile([1.0, -1.0], k)

    # Each pair contributes two three-coefficient inequalities.
    abs_rows = k + 2 * np.arange(m)
    abs_cols = np.column_stack((pair_i, pair_j, n + np.arange(m))).ravel()
    rows = np.concatenate((overlap_rows, np.repeat(abs_rows, 3), np.repeat(abs_rows + 1, 3)))
    cols = np.concatenate((overlap_cols, abs_cols, abs_cols))
    values = np.concatenate((
        overlap_values,
        np.tile([1.0, -1.0, -1.0], m),
        np.tile([-1.0, 1.0, -1.0], m),
    ))
    a_ub = coo_matrix((values, (rows, cols)), shape=(k + 2 * m, variables)).tocsc()
    b_ub = np.concatenate((-np.ones(k), np.zeros(2 * m)))
    bounds = [(None, None)] * n + [(0.0, None)] * m
    bounds[0] = (0.0, 0.0)
    return c, a_ub, b_ub, bounds


def evaluate_layout(x, nodes, hyperedges, weights, regularized, scale, directed):
    pair_i, pair_j = np.triu_indices(len(x), k=1)
    gaps = np.abs(x[pair_i] - x[pair_j])
    w = weights[pair_i, pair_j]
    w_reg = regularized[pair_i, pair_j]
    l2 = float(np.dot(w_reg, gaps ** 2))

    # index.html draws one pillar per node, from its lowest to highest incident edge.
    pillars = {node["id"]: [] for node in nodes}
    for he, height in zip(hyperedges, x):
        for nid in set(he.get("nodes", [])):
            if nid in pillars:
                pillars[nid].append(float(height))
    pillar_length = sum(max(levels) - min(levels) for levels in pillars.values() if levels)

    order_gaps = x[directed[:, 1]] - x[directed[:, 0]]
    return {
        "l1_regularized": float(np.dot(w_reg, gaps)),
        "l1_shared_vertices": float(np.dot(w, gaps)),
        "l2_regularized": l2,
        "l2_shared_vertices": float(np.dot(w, gaps ** 2)),
        "l2_scaled_qp": 2.0 * scale * l2,
        "height_min": float(np.min(x)),
        "height_max": float(np.max(x)),
        "height_range": float(np.max(x) - np.min(x)),
        "pillar_length": float(pillar_length),
        "overlap_violation_count": int(np.count_nonzero(np.abs(order_gaps) < 1.0 - CONSTRAINT_TOLERANCE)),
        "order_violation_count": int(np.count_nonzero(order_gaps < 1.0 - CONSTRAINT_TOLERANCE)),
        "max_order_violation": float(max(0.0, np.max(1.0 - order_gaps))) if len(order_gaps) else 0.0,
    }


def load_comparison(path, nodes, hyperedges, overlaps):
    with open(path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    baseline_edges = baseline.get("hyperedges", [])
    expected_edges = [(he.get("id"), he.get("nodes", [])) for he in hyperedges]
    actual_edges = [(he.get("id"), he.get("nodes", [])) for he in baseline_edges]
    expected_nodes = {node["id"]: (node["x"], node["y"]) for node in nodes}
    actual_nodes = {node["id"]: (node["x"], node["y"]) for node in baseline.get("nodes", [])}
    baseline_overlaps = {tuple(pair) for pair in baseline.get("overlaps", [])}
    if actual_edges != expected_edges or actual_nodes != expected_nodes or baseline_overlaps != set(overlaps):
        raise ValueError("Comparison JSON must contain the same input geometry, hyperedge order and overlaps.")
    x = np.asarray([he.get("layer", np.nan) for he in baseline_edges], dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Comparison JSON contains missing or non-finite layer coordinates.")
    return x


def print_metrics(metrics, baseline_metrics=None):
    rows = [
        ("L1 (W + reg)", "l1_regularized"),
        ("L1 (shared vertices only)", "l1_shared_vertices"),
        ("L2 squared (W + reg)", "l2_regularized"),
        ("L2 squared (shared vertices only)", "l2_shared_vertices"),
        ("L2 scaled as solve_layers.py", "l2_scaled_qp"),
        ("Height range", "height_range"),
        ("Pillar length (before viewer Z scale)", "pillar_length"),
        ("Overlap constraint violations", "overlap_violation_count"),
        ("DSatur order violations", "order_violation_count"),
    ]
    header = f"{'Metric':<39} {'DSatur L1':>18}"
    if baseline_metrics is not None:
        header += f" {'QP baseline':>18}"
    print(header)
    for label, key in rows:
        line = f"{label:<39} {metrics[key]:>18.6f}"
        if baseline_metrics is not None:
            line += f" {baseline_metrics[key]:>18.6f}"
        print(line)


def solve_layer_assignment(input_path, output_path, reg_val=1e-5, comparison_path=None):
    if not np.isfinite(reg_val) or reg_val < 0:
        raise ValueError("--reg must be finite and nonnegative.")
    print(f"Reading hypergraph data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes, hyperedges = data.get("nodes", []), data.get("hyperedges", [])
    n = len(hyperedges)
    if n == 0:
        raise ValueError("No hyperedges found in input data.")
    print(f"Loaded {len(nodes)} nodes and {n} hyperedges.")
    print("Computing polygons and overlap set C (same as solve_layers.py)...")
    overlaps = extract_overlaps(nodes, hyperedges)
    print(f"Found {len(overlaps)} overlapping pairs (C set).")
    weights, regularized, scale = compute_weights(hyperedges, reg_val)
    print(f"Regularization on every off-diagonal pair: {reg_val:g}")
    print("Phase 1: Computing Graph Coloring using DSatur on overlap graph C...")
    colors = dsatur_coloring(n, overlaps)
    num_colors = max(colors.values()) + 1
    print(f"DSatur completed. Initial colors: {num_colors} (not the final number of heights)")
    directed = orient_overlaps(n, overlaps, colors)
    print(f"Objective scale factor for QP comparison: {scale:.6e}")

    # Load the baseline before writing output, including when both paths are identical.
    baseline_metrics = None
    if comparison_path is not None:
        baseline_x = load_comparison(comparison_path, nodes, hyperedges, overlaps)
        baseline_metrics = evaluate_layout(
            baseline_x, nodes, hyperedges, weights, regularized, scale, directed,
        )
        if baseline_metrics["order_violation_count"]:
            raise ValueError(
                "Comparison layout does not satisfy the DSatur overlap order. "
                "Use solve_layers.py output for a controlled L1/L2 comparison."
            )
        print(f"QP baseline loaded from {comparison_path} (same DSatur overlap order).")

    print("Phase 2: Building sparse LP for the weighted absolute-value objective...")
    c, a_ub, b_ub, bounds = build_lp(regularized, directed)
    print(f"Solving LP with {len(c)} continuous variables and {len(b_ub)} constraints...")
    start_time = time.perf_counter()
    res = linprog(
        c, A_ub=a_ub if len(b_ub) else None, b_ub=b_ub if len(b_ub) else None,
        bounds=bounds, method="highs",
    )
    duration = time.perf_counter() - start_time
    if not res.success or res.x is None:
        raise RuntimeError(f"LP optimization failed (status {res.status}): {res.message}")
    x_opt = res.x[:n]
    if not np.all(np.isfinite(x_opt)):
        raise RuntimeError("LP returned non-finite layer coordinates.")
    metrics = evaluate_layout(x_opt, nodes, hyperedges, weights, regularized, scale, directed)
    if metrics["order_violation_count"]:
        raise RuntimeError(
            f"LP result violates {metrics['order_violation_count']} directed overlap constraints."
        )
    print(f"Optimization converged successfully ({duration:.3f} s).")
    print(f"Optimal objective value (regularized L1): {res.fun:.6f}")
    print("All overlap constraints and DSatur directions satisfied.")
    print(f"Anchor coordinate x_0: {x_opt[0]:.6f}")
    print(f"Coordinate range: min={metrics['height_min']:.4f}, max={metrics['height_max']:.4f}")
    print_metrics(metrics, baseline_metrics)

    for he, height in zip(hyperedges, x_opt):
        he["layer"] = float(height)
    output_data = {
        "nodes": nodes,
        "hyperedges": hyperedges,
        "overlaps": overlaps,
        "method": "dsatur_l1",
        "regularization": reg_val,
        "dsatur_num_colors": num_colors,
        "objective_value": float(res.fun),
        "objective_value_l2": metrics["l2_regularized"],
        "objective_value_l2_scaled": metrics["l2_scaled_qp"],
        "metrics": metrics,
        "solver": {
            "method": "scipy.optimize.linprog (HiGHS)",
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "iterations": int(res.nit),
            "seconds": duration,
        },
    }
    if baseline_metrics is not None:
        output_data["comparison"] = {"file": str(comparison_path), "metrics": baseline_metrics}
    print(f"Writing output to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("Layer assignment complete!")
    return output_data


if __name__ == "__main__":
    args = parse_arguments()
    try:
        solve_layer_assignment(args.input_file, args.output_file, args.reg, args.compare)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
