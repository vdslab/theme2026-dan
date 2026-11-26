"""Independent layout checks and geometric measurements for the VisNotes runs."""

from itertools import combinations, groupby
import math

import numpy as np
from shapely import contains_xy
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from experiments.tree_optimization_comparison_01.review_results import (
    compare_metrics,
    evaluate,
    placement_levels,
)
from solve_layers import dsatur_coloring
from solve_layers_dsatur_l1 import CONSTRAINT_TOLERANCE, orient_overlaps


def clique_lower_bound(size, overlaps):
    """Return a checked clique witness, even when this bound is not tight."""
    neighbors = [set() for _ in range(size)]
    for a, b in overlaps:
        neighbors[a].add(b)
        neighbors[b].add(a)

    best = []
    for seed in sorted(range(size), key=lambda i: (-len(neighbors[i]), i)):
        candidates = neighbors[seed].copy()
        if len(candidates) + 1 <= len(best):
            continue
        clique = [seed]
        while candidates:
            vertex = max(
                candidates,
                key=lambda i: (len(candidates & neighbors[i]), len(neighbors[i]), -i),
            )
            clique.append(vertex)
            candidates.intersection_update(neighbors[vertex])
        if len(clique) > len(best):
            best = clique
    witness = sorted(best)
    if any(b not in neighbors[a] for a, b in combinations(witness, 2)):
        raise ValueError("invalid clique witness")
    return witness


def overlap_bounds(size, overlaps):
    witness = clique_lower_bound(size, overlaps)
    colors = dsatur_coloring(size, overlaps)
    if any(colors[a] == colors[b] for a, b in overlaps):
        raise ValueError("DSatur returned an invalid coloring")
    lower, upper = len(witness), len(set(colors.values()))
    return {
        "overlap_count": len(overlaps),
        "clique_lower_bound": lower,
        "clique_witness_indices": witness,
        "dsatur_upper_bound": upper,
        "chromatic_number": lower if lower == upper else None,
    }


def overlap_profile(geometry):
    shared_count = {(a, b): shared for a, b, shared in geometry.pair_counts}
    exact = {
        "0": tuple(pair for pair in geometry.overlaps if shared_count[pair] == 0),
        "1": tuple(pair for pair in geometry.overlaps if shared_count[pair] == 1),
        "2": tuple(pair for pair in geometry.overlaps if shared_count[pair] == 2),
        ">=3": tuple(pair for pair in geometry.overlaps if shared_count[pair] >= 3),
    }
    cumulative = {
        ">=0": geometry.overlaps,
        ">=1": tuple(pair for pair in geometry.overlaps if shared_count[pair] >= 1),
        ">=2": tuple(pair for pair in geometry.overlaps if shared_count[pair] >= 2),
        ">=3": exact[">=3"],
    }
    return {
        "exact_shared_vertices": {
            label: overlap_bounds(geometry.size, pairs) for label, pairs in exact.items()
        },
        "cumulative_shared_vertices": {
            label: overlap_bounds(geometry.size, pairs) for label, pairs in cumulative.items()
        },
    }


def incidence_metrics(geometry, levels):
    edges_by_node = {node["id"]: [] for node in geometry.data["nodes"]}
    for index, edge in enumerate(geometry.data["hyperedges"]):
        for node_id in set(edge["nodes"]):
            edges_by_node[node_id].append(index)
    constrained = {index for pair in geometry.overlaps for index in pair}
    vertices = []
    for node_id, incident in edges_by_node.items():
        incident_levels = [levels[i] for i in incident]
        span = max(incident_levels) - min(incident_levels) if incident_levels else 0
        vertices.append({
            "node_id": node_id,
            "degree": len(incident),
            "constrained_degree": sum(index in constrained for index in incident),
            "span": span,
        })
    # Prefer high constrained degree, then total degree, then input order.
    hubs = sorted(
        (row for row in vertices if row["constrained_degree"] >= 2),
        key=lambda row: (-row["constrained_degree"], -row["degree"]),
    )[:5]
    return {
        "pillar_square_sum": sum(row["span"] ** 2 for row in vertices),
        "degree_span_square_sum": sum(row["degree"] * row["span"] ** 2 for row in vertices),
        "top5_constrained_hub_mean_span": sum(row["span"] for row in hubs) / len(hubs) if hubs else None,
        "constrained_hubs": hubs,
    }


def objective_value(metrics, objective, alpha, beta):
    pillar = metrics["pillar_length"]
    squared = metrics["l2_scaled_qp"]
    if objective == "pillar":
        return pillar
    if objective == "squared":
        return squared
    if objective == "hybrid":
        return pillar + alpha * squared
    if objective == "degree-span":
        return pillar + beta * metrics["degree_span_square_sum"]
    raise ValueError(f"unknown objective {objective!r}")


def read_layout_metrics(layout, geometry, label, continuous=False):
    """Verify schema and overlap constraints before measuring the saved layout."""
    errors = []
    if continuous:
        # The legacy QP and LP orient an overlap by the DSatur color order.
        if layout["nodes"] != geometry.data["nodes"]:
            errors.append(f"{label}: vertex fields, order, or XY coordinates differ")
        actual = [{k: v for k, v in edge.items() if k != "layer"} for edge in layout["hyperedges"]]
        expected = [{k: v for k, v in edge.items() if k != "layer"} for edge in geometry.data["hyperedges"]]
        if actual != expected:
            errors.append(f"{label}: hyperedge fields or ordered memberships differ")
        if sorted(map(tuple, layout["overlaps"])) != list(geometry.overlaps):
            errors.append(f"{label}: saved overlaps differ from the independent STRtree result")
        levels = tuple(edge.get("layer") for edge in layout["hyperedges"])
        if any(type(level) not in (int, float) or not math.isfinite(level) for level in levels):
            raise ValueError(f"{label}: heights must be finite numbers")
    else:
        levels = placement_levels(layout, geometry, label, errors)
    if errors:
        raise ValueError("; ".join(errors))
    metrics = evaluate(geometry, levels)
    if continuous:
        colors = dsatur_coloring(geometry.size, geometry.overlaps)
        oriented = orient_overlaps(geometry.size, geometry.overlaps, colors)
        gaps = np.asarray(levels)[oriented[:, 1]] - np.asarray(levels)[oriented[:, 0]]
        minimum = 1.0 - CONSTRAINT_TOLERANCE
        metrics["overlap_violation_count"] = int(np.count_nonzero(np.abs(gaps) < minimum))
        metrics["order_violation_count"] = int(np.count_nonzero(gaps < minimum))
        metrics["max_order_violation"] = max(0.0, 1.0 - float(np.min(gaps))) if len(gaps) else 0.0
        metrics["distinct_height_values"] = metrics.pop("used_layer_count")
    if metrics["overlap_violation_count"] or metrics["order_violation_count"]:
        raise ValueError(f"{label}: overlap height constraints are violated")
    metrics.update(incidence_metrics(geometry, levels))
    return levels, metrics


def check_report_metrics(reported, measured, label, check_spans=False):
    keys = (
        "l1_regularized", "l1_shared_vertices", "l2_regularized", "l2_shared_vertices",
        "l2_scaled_qp", "height_min", "height_max", "height_range", "pillar_length",
        "overlap_violation_count", "order_violation_count", "max_order_violation",
        "used_layer_count",
    )
    expected = {key: measured[key] for key in keys if key in measured}
    errors = []
    if check_spans:
        expected["layer_count"] = measured["used_layer_count"]
        for key in ("degree_span_square_sum", "pillar_square_sum", "top5_constrained_hub_mean_span"):
            if measured[key] is None:
                if key not in reported or reported[key] is not None:
                    errors.append(f"{label}.{key}: expected null for an empty hub set")
            else:
                expected[key] = measured[key]
        if reported.get("constrained_hubs") != measured["constrained_hubs"]:
            errors.append(f"{label}.constrained_hubs: reported and measured hubs differ")
    compare_metrics(reported, expected, label, errors)
    if errors:
        raise ValueError("; ".join(errors))


def projection_metrics(geometry, levels, azimuth=45.0, elevation=35.0):
    """Count pixel-center face coverage on a fixed orthographic lattice.

    Projection is in pixels per original XY span. The lattice is anchored at
    the input XY center with the lowest height at Z=0 for every case. Polygon
    walls, outlines, line glyphs and markers are outside the face metric.
    """
    xy = {node["id"]: (node["x"], node["y"]) for node in geometry.data["nodes"]}
    minimum = min(levels)
    x, y = zip(*xy.values())
    span = max(max(x) - min(x), max(y) - min(y))
    xy_span = span if span > 0 else 1.0
    pixels_per_span = 384
    factor = pixels_per_span / xy_span
    x_mid, y_mid = (min(x) + max(x)) / 2, (min(y) + max(y)) / 2
    z_step = xy_span / 100
    yaw, pitch = map(math.radians, (azimuth, elevation))
    a, b = -math.sin(yaw) * factor, math.cos(yaw) * factor
    d, e = -math.cos(yaw) * math.sin(pitch) * factor, -math.sin(yaw) * math.sin(pitch) * factor
    height_in_pixels = math.cos(pitch) * z_step * factor
    projected = []
    for edge, level in zip(geometry.data["hyperedges"], levels):
        ring = [point for point, _ in groupby(xy[nid] for nid in edge["nodes"])]
        if ring and ring[-1] == ring[0]:
            ring.pop()
        if len(ring) < 3:
            continue
        polygon = Polygon(ring)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 0:
            projected.append(affine_transform(
                polygon,
                [a, b, d, e, -a * x_mid - b * y_mid,
                 -d * x_mid - e * y_mid + (level - minimum) * height_in_pixels],
            ))

    # A sparse row map keeps the amount of memory independent of height gaps.
    scanlines = {}
    for polygon in projected:
        left, top, right, bottom = polygon.bounds
        u0, u1 = math.ceil(left - 0.5), math.floor(right - 0.5)
        v0, v1 = math.ceil(top - 0.5), math.floor(bottom - 0.5)
        if u1 < u0 or v1 < v0:
            continue
        screen_x = np.arange(u0, u1 + 1) + 0.5
        covered = contains_xy(polygon, screen_x[None, :], np.arange(v0, v1 + 1)[:, None] + 0.5)
        for v, row in zip(range(v0, v1 + 1), covered):
            if not row.any():
                continue
            values = scanlines.setdefault(v, {})
            for u in np.flatnonzero(row) + u0:
                values[u] = values.get(u, 0) + 1
    counts = [count for row in scanlines.values() for count in row.values()]
    return {
        "projection": "orthographic",
        "azimuth_degrees": azimuth,
        "elevation_degrees": elevation,
        "pixels_per_xy_span": pixels_per_span,
        "z_step_world": z_step,
        "polygons_measured": len(projected),
        "covered_pixel_count": len(counts),
        "overdraw_pixel_count": sum(count > 1 for count in counts),
        "overdraw_sample_count": sum(count - 1 for count in counts),
        "max_face_depth": max(counts, default=0),
        "projected_pillar_length_px": evaluate(geometry, levels)["pillar_length"] * height_in_pixels,
    }
