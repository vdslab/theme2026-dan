#!/usr/bin/env python3
"""Optimize one shared XY coordinate per node using polygon layout energies.

Paper mode reimplements Qu et al. (2021/2022) equations and alternating search.
Scalable mode adds regular-polygon consensus initialization and a fitting term.
Arrange mode reduces XY overlaps in an existing layout while guarding its shape.
See polygon_layout_reproduction.md for sources, scope and limitations.
"""

import argparse
from collections import Counter
import copy
import html
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon


def hull_indices(points):
    """CCW convex hull, including a deterministic fallback for collinear data."""
    # A monotone chain avoids Qhull's per-call temporary-file overhead on
    # Windows. Duplicate coordinates are omitted only from the auxiliary hull.
    coords = points.tolist()
    unique = {}
    for i, point in enumerate(coords):
        unique.setdefault(tuple(point), i)
    order = sorted(unique.values(), key=lambda i: coords[i])
    if len(order) < 3:
        return np.asarray(order, dtype=int)

    def turn(a, b, c):
        ax, ay = coords[a]; bx, by = coords[b]; cx, cy = coords[c]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    lower, upper = [], []
    for sequence, chain in ((order, lower), (reversed(order), upper)):
        for i in sequence:
            while len(chain) > 1 and turn(chain[-2], chain[-1], i) <= 0:
                chain.pop()
            chain.append(i)
    return np.asarray(lower[:-1] + upper[:-1], dtype=int)


def signed_area(points):
    return .5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                        - points[:, 1] * np.roll(points[:, 0], -1))


def starrize(indices, xy):
    """Keep every member, sorting around the area centroid of its convex hull."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 3:
        return indices.copy()
    points = xy[indices]
    hull = hull_indices(points)
    hp = points[hull]
    cross = hp[:, 0] * np.roll(hp[:, 1], -1) - np.roll(hp[:, 0], -1) * hp[:, 1]
    twice_area = np.sum(cross)
    if abs(twice_area) > 1e-12:
        center = np.sum((hp + np.roll(hp, -1, axis=0)) * cross[:, None], axis=0) / (3 * twice_area)
    else:
        center = np.mean(points, axis=0)
    delta = points - center
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    # Stable radial/ID tie breaks matter for imported collinear coordinates.
    order = np.lexsort((indices, np.sum(delta * delta, axis=1), angles))
    return indices[order]


def clique_adjacency(problem):
    """Temporary ordinary graph: every hyperedge induces a complete subgraph.

    Shared IDs use the SAME graph vertex. Original hyperedges are retained
    separately because the union graph alone cannot recover incidence.
    """
    adjacency = np.zeros((problem.n, problem.n), dtype=float)
    for members in problem.members:
        adjacency[np.ix_(members, members)] = 1.
    np.fill_diagonal(adjacency, 0.)
    return adjacency


class PolygonProblem:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.ids = [node["id"] for node in data["nodes"]]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Node IDs must be nonempty and unique")
        self.index = {nid: i for i, nid in enumerate(self.ids)}
        self.initial = np.asarray([[node["x"], node["y"]] for node in data["nodes"]], dtype=float)
        if not np.isfinite(self.initial).all():
            raise ValueError("All input coordinates must be finite")
        if len({e["id"] for e in data["hyperedges"]}) != len(data["hyperedges"]):
            raise ValueError("Hyperedge IDs must be unique")
        self.members = []
        for edge in data["hyperedges"]:
            missing = set(edge["nodes"]) - set(self.ids)
            if missing:
                raise ValueError(f"Edge {edge['id']} references missing nodes: {missing}")
            self.members.append(np.asarray([self.index[n] for n in dict.fromkeys(edge["nodes"])], dtype=int))
        self.faces = [i for i, members in enumerate(self.members) if len(members) >= 3]
        self.active = [i for i, members in enumerate(self.members) if len(members) >= 2]
        self.n = len(self.ids)
        lengths = []
        for edge in self.active:
            p = self.initial[starrize(self.members[edge], self.initial)]
            lengths.extend(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1))
        nonzero = np.asarray(lengths)[np.asarray(lengths) > 1e-10]
        self.unit = float(np.median(nonzero)) if len(nonzero) else 1.0
        self.origin = np.mean(self.initial, axis=0)


def measure(problem, xy, reorder=True):
    ratios, deficits = [], []
    invalid, concave, degenerate = 0, 0, 0
    for edge in problem.faces:
        ids = starrize(problem.members[edge], xy) if reorder else problem.members[edge]
        p = xy[ids]
        poly = Polygon(p)
        invalid += not poly.is_valid
        a = abs(signed_area(p))
        ah = abs(signed_area(p[hull_indices(p)]))
        deficit = max(0.0, 1 - a / max(ah, 1e-15))
        concave += deficit > 1e-7
        degenerate += a < 1e-12
        perimeter = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum()
        ratios.append(float(perimeter ** 2 / max(4 * len(ids) * np.tan(np.pi / len(ids)) * a, 1e-15)))
        deficits.append(deficit)
    nearest = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.array([math.inf])
    return {"faces": len(problem.faces), "invalid_faces": int(invalid),
            "degenerate_faces": int(degenerate), "concave_faces": int(concave),
            "mean_regularity_ratio": float(np.mean(ratios)) if ratios else 1.,
            "median_regularity_ratio": float(np.median(ratios)) if ratios else 1.,
            "mean_hull_deficit": float(np.mean(deficits)) if deficits else 0.,
            "minimum_node_distance": float(np.min(nearest)) if len(xy) > 1 else None}


def face_metrics(problem, xy, reorder=True):
    result = []
    for edge in sorted(problem.faces, key=lambda i: len(problem.members[i]), reverse=True):
        ids = starrize(problem.members[edge], xy) if reorder else problem.members[edge]
        points = xy[ids]
        area = abs(signed_area(points))
        hull_area = abs(signed_area(points[hull_indices(points)]))
        lengths = np.linalg.norm(points-np.roll(points, -1, axis=0), axis=1)
        polygon = Polygon(points)
        n = len(ids)
        result.append({'id': problem.data['hyperedges'][edge]['id'],
                       'label': problem.data['hyperedges'][edge].get('label', ''),
                       'vertices': n,
                       'area': float(area), 'perimeter': float(lengths.sum()),
                       'is_valid': bool(polygon.is_valid), 'is_degenerate': bool(area < 1e-12),
                       'regularity_ratio': float(lengths.sum()**2/max(4*n*np.tan(np.pi/n)*area, 1e-15)),
                       'hull_deficit': float(max(0., 1-area/max(hull_area, 1e-15))),
                       'side_length_cv': float(np.std(lengths)/max(np.mean(lengths), 1e-15))})
    return result


def assert_preserved(before, after):
    """IDs, labels, metadata and membership multiplicities must remain exact."""
    if set(before) != set(after):
        raise ValueError("Top-level schema changed")
    if len(before["nodes"]) != len(after["nodes"]) or len(before["hyperedges"]) != len(after["hyperedges"]):
        raise ValueError("Node or hyperedge count changed")
    for old, new in zip(before["nodes"], after["nodes"]):
        if {k: v for k, v in old.items() if k not in ("x", "y")} != {k: v for k, v in new.items() if k not in ("x", "y")}:
            raise ValueError("Node identity or metadata changed")
        if not np.isfinite([new["x"], new["y"]]).all():
            raise ValueError("Non-finite output coordinate")
    for old, new in zip(before["hyperedges"], after["hyperedges"]):
        if Counter(old["nodes"]) != Counter(new["nodes"]):
            raise ValueError("Incidence or membership multiplicity changed")
        if {k: v for k, v in old.items() if k != "nodes"} != {k: v for k, v in new.items() if k != "nodes"}:
            raise ValueError("Hyperedge identity or metadata changed")
    for key in set(before) - {"nodes", "hyperedges"}:
        if before[key] != after[key]:
            raise ValueError("Additional input metadata changed")


def overlap_statistics(problem, xy, reorder=True):
    """Measure actual face intersections separately from circle-based PS energy."""
    polygons = []
    for edge in problem.faces:
        ids = starrize(problem.members[edge], xy) if reorder else problem.members[edge]
        polygon = Polygon(xy[ids])
        polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
    total, no_common, at_most_two = 0, 0, 0
    memberships = [set(problem.members[edge]) for edge in problem.faces]
    for i, left in enumerate(polygons):
        for j in range(i + 1, len(polygons)):
            right = polygons[j]
            if left.intersects(right) and left.intersection(right).area > 1e-9:
                count = len(memberships[i] & memberships[j])
                total += 1
                no_common += count == 0
                at_most_two += count <= 2
    return {"area_threshold": 1e-9, "overlapping_pairs": total,
            "pairs_without_shared_nodes": no_common,
            "pairs_with_at_most_two_shared_nodes": at_most_two}


def make_output(problem, xy, reorder=True):
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (problem.n, 2) or not np.isfinite(xy).all():
        raise ValueError('Output must have one finite XY pair per node')
    output = copy.deepcopy(problem.data)
    for node, point in zip(output["nodes"], xy):
        node["x"], node["y"] = map(float, point)
    for edge, members in zip(output["hyperedges"], problem.members):
        if not reorder:
            continue
        counts = Counter(edge["nodes"])
        edge["nodes"] = [problem.ids[i] for i in starrize(members, xy) for _ in range(counts[problem.ids[i]])]
    assert_preserved(problem.data, output)
    return output


def shape_replacement_check(problem, output, relative_tolerance=0., limits=None):
    """Guard same-file shape runs against losing a better existing layout.

    The optimization energy mixes shape, separation and scale; reducing that
    energy alone does not imply better-looking polygons. Compare the actual
    saved boundary order and scale-invariant shape metrics independently.
    """
    candidate = PolygonProblem(output)
    from polygon_layout.polygon_quality import all_face_judgment
    judgment = all_face_judgment(problem, candidate, relative_tolerance, limits)
    before = measure(problem, problem.initial, reorder=False)
    after = measure(candidate, candidate.initial, reorder=False)
    details = {'before': before, 'candidate': after, 'reasons': [],
               'per_face_relative_tolerance': relative_tolerance,
               'large_face_min_vertices': 30, 'all_face_judgment': judgment}
    if after['invalid_faces'] or after['degenerate_faces']:
        details['reasons'].append('invalid_candidate')
        return False, details
    old_faces = face_metrics(problem, problem.initial, reorder=False)
    new_faces = {f['id']: f for f in face_metrics(candidate, candidate.initial, reorder=False)}
    comparable = [f for f in old_faces if f['is_valid'] and not f['is_degenerate']]
    details['comparable_valid_faces'] = len(comparable)
    if comparable and (np.mean([new_faces[f['id']]['regularity_ratio'] for f in comparable])
                       > np.mean([f['regularity_ratio'] for f in comparable])*(1+1e-8)):
        details['reasons'].append('mean_regularity_worsened')
    degraded = judgment['regressed_face_ids']
    details['degraded_face_ids'] = degraded
    if degraded:
        details['reasons'].append('individual_face_regularity_worsened')
    large = [f for f in comparable if f['vertices'] >= 30]
    if large:
        old_mean = float(np.mean([f['regularity_ratio'] for f in large]))
        new_mean = float(np.mean([new_faces[f['id']]['regularity_ratio'] for f in large]))
        details.update(large_face_count=len(large), large_mean_before=old_mean, large_mean_candidate=new_mean)
        if new_mean > old_mean*(1+1e-8):
            details['reasons'].append('large_face_regularity_worsened')
    return not details['reasons'], details


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_preview(path, problem, before, after, reorder=True):
    """Standalone SVG: individual high-cardinality and representative small faces."""
    selected = sorted(problem.faces, key=lambda i: len(problem.members[i]), reverse=True)[:8]
    selected += sorted([i for i in problem.faces if 4 <= len(problem.members[i]) <= 8],
                       key=lambda i: len(problem.members[i]), reverse=True)[:4]
    width, panel_width, panel_height = 1120, 560, 250
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{70 + panel_height * len(selected)}" viewBox="0 0 {width} {70 + panel_height * len(selected)}">',
             '<rect width="100%" height="100%" fill="#fafafa"/>',
             '<g font-family="Arial,sans-serif" fill="#222">',
             '<text x="25" y="30" font-size="20">Before (input coordinates)</text>',
             '<text x="585" y="30" font-size="20">After (XY optimization)</text>',
             '<text x="25" y="53" font-size="12">Each panel is fitted independently. Dots are all member vertices; no member is omitted.</text>']
    for row, edge_index in enumerate(selected):
        edge = problem.data["hyperedges"][edge_index]
        color = edge.get("color", {"r": 80, "g": 130, "b": 200})
        rgb = f'rgb({color["r"]},{color["g"]},{color["b"]})'
        for column, xy in enumerate((before, after)):
            ids = problem.members[edge_index] if column == 0 or not reorder else starrize(problem.members[edge_index], xy)
            points = xy[ids]
            low, high = np.min(points, axis=0), np.max(points, axis=0)
            scale = min((panel_width - 60) / max(high[0] - low[0], 1e-9),
                        (panel_height - 65) / max(high[1] - low[1], 1e-9))
            display = (points - (low + high) / 2) * [scale, -scale]
            display += [column * panel_width + panel_width / 2, 70 + row * panel_height + panel_height / 2 + 12]
            vertices = " ".join(f"{x:.3f},{y:.3f}" for x, y in display)
            parts.append(f'<rect x="{column * panel_width + 8}" y="{70 + row * panel_height}" width="544" height="242" fill="white" stroke="#ddd"/>')
            parts.append(f'<text x="{column * panel_width + 22}" y="{92 + row * panel_height}" font-size="14">{html.escape(str(edge.get("label", edge["id"])))} ({len(ids)} vertices)</text>')
            parts.append(f'<polygon points="{vertices}" fill="{rgb}" fill-opacity=".32" stroke="#34495e" stroke-width="1.2"/>')
            parts.extend(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.2" fill="#17202a"/>' for x, y in display)
    parts.append('</g></svg>')
    Path(path).write_text("\n".join(parts), encoding="utf-8", newline="\n")


def optimize(problem, **kwargs):
    tolerance = kwargs.pop('shape_tolerance', 0.)
    if kwargs.get('mode') == 'arrange':
        from polygon_layout.polygon_arrange import optimize_placement
        xy, report = optimize_placement(problem, iterations=kwargs.get('iterations', 250),
                                        rounds=kwargs.get('max_cycles', 3), shape_tolerance=tolerance,
                                        progress=kwargs.get('progress', True))
    else:
        from polygon_layout.polygon_optimize import optimize_reproduction
        xy, report = optimize_reproduction(problem, **kwargs)
    from polygon_layout.polygon_quality import all_face_judgment
    candidate = PolygonProblem(make_output(problem, xy, reorder=kwargs.get('mode') != 'arrange'))
    report['all_face_judgment'] = all_face_judgment(problem, candidate, tolerance)
    report['all_faces_quality_passed'] = report['all_face_judgment']['all_faces_pass_quality']
    return xy, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="original.json")
    parser.add_argument("output", nargs="?", default="original.json")
    parser.add_argument("--iterations", type=int, default=600, help="L-BFGS iteration budget per cycle")
    parser.add_argument("--max-cycles", type=int, default=6, help="Full continuous/swap cycles per phase")
    parser.add_argument("--mode", choices=("paper", "scalable", "arrange"), default="scalable")
    parser.add_argument("--shape-tolerance", type=float, default=0.,
                        help="Allowed per-face regularity regression (default 0: no worsening for any face)")
    parser.add_argument('--audit-only', action='store_true', help='Judge every polygon without optimizing or changing layout JSON')
    parser.add_argument('--audit-reference', help='Audit-only: compare with a previous JSON with identical IDs and memberships')
    parser.add_argument('--quality-preview', help='Complete all-face HTML review (default: mode prefix + _all_faces.html)')
    parser.add_argument('--quality-regularity-max', type=float, default=1.20, help='Absolute review limit; dimensionless ratio, 1 is ideal')
    parser.add_argument('--quality-hull-deficit-max', type=float, default=.02, help='Absolute review limit for missing hull area fraction')
    parser.add_argument('--quality-side-cv-max', type=float, default=.25, help='Absolute review limit for side-length variation')
    parser.add_argument("--dual", action="store_true", help="Jointly optimize the dual view (DD energy)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fit-weight", type=float, default=1.)
    parser.add_argument("--collision-weight", type=float, default=.05)
    parser.add_argument("--convexity-weight", type=float, default=10.)
    parser.add_argument("--initial-layout", help="Optional JSON with the same IDs/memberships for warm starting")
    parser.add_argument("--initialization", choices=("input", "circular", "random", "hu-clique", "hu-bipartite"),
                        help="Default: Hu clique for paper mode; Hu bipartite for scalable mode")
    parser.add_argument("--sfdp", help="Path to Graphviz sfdp executable")
    parser.add_argument("--report", help="Default: mode prefix + _report.json (audit: polygon_quality_report.json)")
    parser.add_argument("--preview", help="Selected SVG, or full HTML for --audit-only (default: mode-dependent name)")
    args = parser.parse_args()
    prefix = 'polygon_quality' if args.audit_only else 'polygon_arrangement' if args.mode == 'arrange' else 'polygon_layout'
    args.report = args.report or prefix+'_report.json'
    args.preview = args.preview or prefix+('_all_faces.html' if args.audit_only else '_comparison.svg')
    args.quality_preview = args.quality_preview or (args.preview if args.audit_only else str(Path(args.report).with_name(prefix+'_all_faces.html')))
    if args.audit_reference and not args.audit_only:
        parser.error('--audit-reference requires --audit-only')
    if (args.iterations < 1 or args.max_cycles < 1
            or not np.isfinite([args.fit_weight, args.collision_weight, args.convexity_weight]).all()
            or min(args.fit_weight, args.collision_weight, args.convexity_weight) < 0):
        parser.error("Invalid optimization budget or weight")
    if not np.isfinite(args.shape_tolerance) or not 0 <= args.shape_tolerance <= 1:
        parser.error('--shape-tolerance must be between 0 and 1')
    quality_limits = {'regularity_ratio': args.quality_regularity_max,
                      'hull_deficit': args.quality_hull_deficit_max, 'side_length_cv': args.quality_side_cv_max}
    if (not np.isfinite(list(quality_limits.values())).all() or args.quality_regularity_max < 1
            or min(args.quality_hull_deficit_max, args.quality_side_cv_max) < 0):
        parser.error('Invalid absolute quality limits')
    if args.mode == 'arrange' and (args.dual or args.initial_layout or args.initialization not in (None, 'input')):
        parser.error('Arrange mode starts from the input XY; dual and alternative initializations are not supported')
    source, destination = Path(args.input), Path(args.output)
    products = (args.report, args.quality_preview) if args.audit_only else (args.output, args.report, args.preview, args.quality_preview)
    if len({Path(p).resolve() for p in products}) != len(products):
        parser.error("Output, reports and previews must use different paths")
    if source.resolve() in {Path(args.report).resolve(), Path(args.preview).resolve(), Path(args.quality_preview).resolve()}:
        parser.error("Report and preview cannot overwrite the input")
    if args.audit_reference and Path(args.audit_reference).resolve() in {Path(p).resolve() for p in products}:
        parser.error('Reports and previews cannot overwrite the audit reference')
    source_bytes = source.read_bytes()
    data = json.loads(source_bytes.decode("utf-8"))
    problem = PolygonProblem(data)
    from polygon_layout.polygon_quality import all_face_judgment, write_quality_review
    if args.audit_only:
        reference_bytes = Path(args.audit_reference).read_bytes() if args.audit_reference else source_bytes
        reference = PolygonProblem(json.loads(reference_bytes.decode('utf-8')))
        judgment = all_face_judgment(reference, problem, args.shape_tolerance, quality_limits)
        report = {'mode': 'audit', 'input': str(source), 'reference': args.audit_reference,
                  'input_sha256': hashlib.sha256(source_bytes).hexdigest(),
                  'reference_sha256': hashlib.sha256(reference_bytes).hexdigest(),
                  'wrote_layout': False, 'all_face_judgment': judgment}
        write_quality_review(args.quality_preview, reference, problem, judgment)
        atomic_json(args.report, report)
        print(json.dumps({k: judgment[k] for k in ('checked_faces', 'quality_pass_count', 'quality_fail_count',
                                                 'quality_status', 'regression_count', 'changes', 'limits')}, indent=2), flush=True)
        print(f'All-face audit: {args.report}; complete review: {args.quality_preview}', flush=True)
        return
    print(f"Optimizing {problem.n} nodes, {len(problem.members)} hyperedges, {len(problem.faces)} polygon faces.", flush=True)
    start_xy = None
    initialization = args.initialization or ("input" if args.initial_layout else None)
    if args.initial_layout:
        if initialization != "input":
            parser.error("An initial layout file requires --initialization input")
        initial_data = json.loads(Path(args.initial_layout).read_text(encoding="utf-8"))
        assert_preserved(data, initial_data)
        start_xy = np.asarray([[n["x"], n["y"]] for n in initial_data["nodes"]])
    xy, report = optimize(problem, iterations=args.iterations, max_cycles=args.max_cycles,
                          seed=args.seed, mode=args.mode, dual=args.dual, start_xy=start_xy,
                          fit_weight=args.fit_weight, collision_weight=args.collision_weight,
                          convexity_weight=args.convexity_weight,
                          shape_tolerance=args.shape_tolerance,
                          initialization=initialization, sfdp=args.sfdp)
    report["initial_layout"] = args.initial_layout
    output = make_output(problem, xy, reorder=args.mode != 'arrange')
    candidate_problem = PolygonProblem(output)
    report['all_face_judgment'] = all_face_judgment(problem, candidate_problem, args.shape_tolerance, quality_limits)
    report['all_faces_quality_passed'] = report['all_face_judgment']['all_faces_pass_quality']
    report['quality_review'] = str(args.quality_preview)
    def save_previews():
        write_preview(args.preview, problem, problem.initial, xy, reorder=args.mode != 'arrange')
        write_quality_review(args.quality_preview, problem, candidate_problem, report['all_face_judgment'])
    if report["after"]["invalid_faces"] or report["after"]["degenerate_faces"]:
        raise RuntimeError("Output contains invalid or degenerate faces; input was not overwritten")
    report["input"] = str(source); report["output"] = str(destination)
    report["input_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    report['wrote_layout'] = True
    if args.mode != 'arrange':
        allowed, guard = shape_replacement_check(problem, output, args.shape_tolerance, quality_limits)
        report['shape_replacement_check'] = guard
        report['inplace_replacement_allowed'] = allowed
        if not allowed and source.resolve() == destination.resolve():
            report['wrote_layout'] = False
            report['output_preserved_reason'] = 'Candidate failed shape replacement check'
            # Preview/report describe the rejected candidate; the input remains
            # byte-for-byte intact and no misleading new backup is created.
            save_previews()
            atomic_json(args.report, report)
            print('Candidate worsened polygon shapes; input kept unchanged. '
                  'Report and preview describe the rejected candidate. '
                  'Use a separate output filename to inspect experimental tradeoffs.', flush=True)
            print(json.dumps({k: v for k, v in guard.items() if k != 'all_face_judgment'}, indent=2), flush=True)
            return
    if args.mode == 'arrange' and not report['all_face_judgment']['all_faces_non_regressing']:
        raise RuntimeError('Placement output failed exhaustive face regression check; layout was not written')
    if args.mode == 'arrange' and not report['improved'] and source.resolve() == destination.resolve():
        report['wrote_layout'] = False
        save_previews()
        atomic_json(args.report, report)
        print('No candidate improved actual overlaps within the shape limits; input kept unchanged.', flush=True)
        return
    if source.resolve() == destination.resolve():
        if source.read_bytes() != source_bytes:
            raise RuntimeError("Input changed during optimization; refusing to overwrite it")
        suffix = '_before_polygon_arrangement' if args.mode == 'arrange' else '_before_polygon_layout'
        backup = source.with_name(source.stem + suffix + source.suffix)
        if backup.exists():
            backup = source.with_name(source.stem + suffix + f"_{time.time_ns()}" + source.suffix)
        # Preserve the original bytes, including formatting, before replacement.
        with backup.open("xb") as stream:
            stream.write(source_bytes)
        report["backup"] = str(backup)
    save_previews()
    atomic_json(args.report, report)
    atomic_json(destination, output)
    summary = {"before": report["before"], "after": report["after"]}
    summary['all_face_judgment'] = {k: report['all_face_judgment'][k]
                                  for k in ('checked_faces', 'quality_pass_count', 'quality_fail_count', 'regression_count', 'quality_status')}
    if args.mode == 'arrange':
        summary.update(placement_before=report['placement_before'], placement_after=report['placement_after'])
    else:
        summary.update(paper_initial_energy=report['initial_energy'],
                       paper_phase_final_energy=report['paper_baseline_energy'],
                       final_objective_energy=report['final_energy'])
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {destination}; comparison: {args.preview}; metrics: {args.report}", flush=True)


if __name__ == "__main__":
    main()
