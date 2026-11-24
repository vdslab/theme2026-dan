"""Read-only, independent verification of saved experiment artifacts."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from itertools import combinations, groupby, permutations
import json
import math
from pathlib import Path
import sys

from shapely.geometry import Polygon
from shapely.strtree import STRtree


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "results" / "experiments" / "tree_optimization_comparison_01"
OBJECTIVES = ("squared", "weighted-l1", "pillar", "squared-plus-layers")
EXPECTED_SEEDS = tuple(range(5))
EXPECTED_REG = 1e-5
EXPECTED_PENALTY = 100.0
EXPECTED_CASE_BUDGET = 180.0
EXPECTED_TOTAL_BUDGET = 7200.0


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def close(got, expected):
    return (
        isinstance(got, (int, float))
        and not isinstance(got, bool)
        and math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-7)
    )


def compare_metrics(actual, expected, label, errors):
    for name, value in expected.items():
        reported = actual.get(name)
        if not close(reported, value):
            errors.append(f"{label}.{name}: reported {reported!r}, recomputed {value!r}")


def overlap_pairs(nodes, memberships):
    """Recreate C from XY, using a spatial index rather than solver pair traversal."""
    xy = {node["id"]: (node["x"], node["y"]) for node in nodes}
    polygons = []
    source_indices = []
    for i, members in enumerate(memberships):
        ring = [point for point, _ in groupby(xy[nid] for nid in members)]
        if ring and ring[-1] == ring[0]:
            ring.pop()
        if len(ring) < 3:
            continue
        geometry = Polygon(ring)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if not geometry.is_empty:
            source_indices.append(i)
            polygons.append(geometry)

    tree = STRtree(polygons)
    overlaps = set()
    for left, right in tree.query(polygons, predicate="intersects").T:
        if left < right and polygons[left].intersection(polygons[right]).area > 1e-9:
            overlaps.add((source_indices[left], source_indices[right]))
    return tuple(sorted(overlaps))


@dataclass(frozen=True)
class InputGeometry:
    data: dict
    hashes: dict
    pair_counts: tuple
    incident: tuple
    overlaps: tuple
    reg: float
    scale: float

    @property
    def size(self):
        return len(self.data["hyperedges"])


def make_geometry(path, reg):
    data = read_json(path)
    raw_members = [edge["nodes"] for edge in data["hyperedges"]]
    memberships = list(map(set, raw_members))
    overlaps = overlap_pairs(data["nodes"], raw_members)
    incident = defaultdict(list)
    for i, members in enumerate(memberships):
        for node in members:
            incident[node].append(i)
    pair_counts = []
    row_counts = [0] * len(memberships)
    for i, j in combinations(range(len(memberships)), 2):
        weight = len(memberships[i].intersection(memberships[j]))
        pair_counts.append((i, j, weight))
        row_counts[i] += weight
        row_counts[j] += weight
    # The Laplacian's largest absolute entry is the maximum row-weight sum.
    max_degree = max(row_counts) + reg * (len(memberships) - 1)
    canonical = json.dumps(
        {"data": data, "overlaps": overlaps, "reg": reg},
        sort_keys=True, ensure_ascii=False, allow_nan=False,
    )
    hashes = {
        "input_sha256": file_hash(path),
        "problem_signature": sha256(canonical.encode("utf-8")).hexdigest(),
        "overlaps_sha256": sha256(json.dumps(overlaps).encode("utf-8")).hexdigest(),
    }
    return InputGeometry(
        data, hashes, tuple(pair_counts), tuple(incident.values()), overlaps, reg,
        1.0 / max_degree if max_degree else 1.0,
    )


def evaluate(geometry, levels):
    l1 = l2 = gap_sum = gap_square_sum = 0
    for i, j, shared in geometry.pair_counts:
        gap = abs(levels[i] - levels[j])
        square = gap * gap
        l1 += shared * gap
        l2 += shared * square
        gap_sum += gap
        gap_square_sum += square
    l2_reg = l2 + geometry.reg * gap_square_sum
    pillar_length = 0
    for edges in geometry.incident:
        heights = [levels[i] for i in edges]
        pillar_length += max(heights) - min(heights)
    low, high = min(levels), max(levels)
    violations = sum(levels[i] == levels[j] for i, j in geometry.overlaps)
    return {
        "l1_regularized": l1 + geometry.reg * gap_sum,
        "l1_shared_vertices": l1,
        "l2_regularized": l2_reg,
        "l2_shared_vertices": l2,
        "l2_scaled_qp": 2.0 * geometry.scale * l2_reg,
        "height_min": low,
        "height_max": high,
        "height_range": high - low,
        "pillar_length": pillar_length,
        "overlap_violation_count": violations,
        "order_violation_count": violations,
        "max_order_violation": int(bool(violations)),
        "used_layer_count": len(set(levels)),
    }


def optimized_values(metrics, penalty):
    return {
        "squared": metrics["l2_scaled_qp"],
        "weighted-l1": metrics["l1_regularized"],
        "pillar": metrics["pillar_length"],
        "squared-plus-layers": metrics["l2_scaled_qp"] + penalty * metrics["used_layer_count"],
    }


def compact(levels):
    return set(levels) == set(range(max(levels) + 1))


def placement_levels(layout, geometry, label, errors):
    if layout["nodes"] != geometry.data["nodes"]:
        errors.append(f"{label}: vertex fields, order, or XY coordinates differ from the input")
    expected = [{k: v for k, v in edge.items() if k != "layer"} for edge in geometry.data["hyperedges"]]
    actual = [{k: v for k, v in edge.items() if k != "layer"} for edge in layout["hyperedges"]]
    if actual != expected:
        raise ValueError(f"{label}: hyperedge fields or ordered memberships differ from the input")
    if sorted(map(tuple, layout["overlaps"])) != list(geometry.overlaps):
        errors.append(f"{label}: saved overlaps differ from independently reconstructed XY overlaps")
    levels = [edge.get("layer") for edge in layout["hyperedges"]]
    if any(
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not 0 <= level < geometry.size
        or int(level) != level
        for level in levels
    ):
        raise ValueError(f"{label}: layers must be integers in [0, {geometry.size - 1}]")
    levels = tuple(map(int, levels))
    if not compact(levels):
        errors.append(f"{label}: layers do not start at 0 or have gaps")
    failed = [(i, j) for i, j in geometry.overlaps if levels[i] == levels[j]]
    if failed:
        errors.append(f"{label}: {len(failed)} overlap violations, first pairs {failed[:5]!r}")
    return levels


def check_tree(record, levels, label, errors):
    size = len(levels)
    if record.get("layers") != list(levels):
        errors.append(f"{label}: tree state's layers differ from the placed layers")
    depth = [level + 1 for level in levels] + [0]
    if record.get("depth") != depth:
        errors.append(f"{label}: stored depths differ from placed layers plus one")
    parents, children = record["parent"], record["children"]
    if len(parents) != size or len(children) != size + 1:
        errors.append(f"{label}: invalid parent or children arrays or virtual root")
        return
    expected = [[] for _ in children]
    for child, parent in enumerate(parents):
        if type(parent) is not int or not 0 <= parent <= size or depth[parent] + 1 != depth[child]:
            errors.append(f"{label}: invalid tree parent or depth at hyperedge index {child}")
            return
        expected[parent].append(child)
    if [sorted(group) for group in children] != expected:
        errors.append(f"{label}: stored children do not agree with parent indices")


def check_embedded(layout, levels, geometry, metrics, label, errors):
    if "tree_layer_search" not in layout:
        errors.append(f"{label}: missing saved tree state")
        return
    metadata = layout["tree_layer_search"]
    for name in ("problem_signature", "overlaps_sha256"):
        if metadata.get(name) != geometry.hashes[name]:
            errors.append(f"{label}: embedded {name} differs from the recomputed input hash")
    if not close(metadata.get("scale"), geometry.scale):
        errors.append(f"{label}: embedded scale differs from the recomputed normalization")
    if metadata.get("index_to_hyperedge_id") != [edge["id"] for edge in geometry.data["hyperedges"]]:
        errors.append(f"{label}: embedded hyperedge index order differs from the input")
    best = metadata["best"]
    check_tree(best, levels, label, errors)
    compare_metrics(best["metrics"], metrics, f"{label}.tree_leaf.metrics", errors)


def partitions(size, labels=(0,), used=1):
    """Name each block by order of first appearance to enumerate partitions once."""
    if len(labels) == size:
        yield labels, used
        return
    for label in range(used + 1):
        yield from partitions(size, labels + (label,), max(used, label + 1))


def exact_optima(geometry, penalty):
    if not 4 <= geometry.size <= 6:
        raise ValueError("an exhaustive fixture must have four to six hyperedges")
    count = 0
    minima = {name: {"value": math.inf} for name in OBJECTIVES}
    for blocks, used in partitions(geometry.size):
        if any(blocks[i] == blocks[j] for i, j in geometry.overlaps):
            continue
        for order in permutations(range(used)):
            levels = tuple(order[block] for block in blocks)
            count += 1
            values = optimized_values(evaluate(geometry, levels), penalty)
            for objective, value in values.items():
                if value < minima[objective]["value"]:
                    minima[objective] = {"value": value, "layers": list(levels)}
    return count, minima


def artifact_path(run, name):
    path = (run / name).resolve()
    if not path.is_relative_to(run):
        raise ValueError(f"artifact path outside the run directory: {name}")
    return path


def verify_oracle(run):
    oracle = run / "small" / "oracle.json"
    if not oracle.is_file():
        return {}, {}, []
    fixtures, optima, errors = {}, {}, []
    saved = read_json(oracle)
    compare_metrics(saved, {"reg": EXPECTED_REG, "penalty": EXPECTED_PENALTY}, "small/oracle.json", errors)
    for fixture in saved["fixtures"]:
        name = fixture["name"]
        issue = []
        try:
            geometry = make_geometry(artifact_path(run, fixture["input_path"]), EXPECTED_REG)
            count, minima = exact_optima(geometry, EXPECTED_PENALTY)
            if fixture.get("input_sha256") != geometry.hashes["input_sha256"]:
                issue.append(f"{name}: fixture hash differs from its saved oracle hash")
            if fixture.get("hyperedge_count") != geometry.size:
                issue.append(f"{name}: hyperedge count disagrees with input")
            if fixture["overlap_count"] != len(geometry.overlaps):
                issue.append(f"{name}: overlap count disagrees with XY geometry")
            if tuple(map(tuple, fixture["overlaps"])) != geometry.overlaps:
                issue.append(f"{name}: saved oracle overlap pairs disagree with XY geometry")
            if fixture["feasible_count"] != count:
                issue.append(f"{name}: feasible count {fixture['feasible_count']} != {count}")
            for objective, minimum in minima.items():
                reported = fixture["optima"][objective]
                if not close(reported["value"], minimum["value"]):
                    issue.append(f"{name}.{objective}: oracle minimum disagrees with the full enumeration")
                levels = reported["layers"]
                if (
                    len(levels) != geometry.size
                    or any(type(level) is not int or not 0 <= level < geometry.size for level in levels)
                    or not compact(levels)
                    or any(levels[i] == levels[j] for i, j in geometry.overlaps)
                ):
                    issue.append(f"{name}.{objective}: saved optimal placement is infeasible")
                    continue
                value = optimized_values(evaluate(geometry, levels), EXPECTED_PENALTY)[objective]
                if not close(value, minimum["value"]):
                    issue.append(f"{name}.{objective}: saved optimal placement has score {value!r}")
            if name in fixtures:
                issue.append(f"{name}: duplicate fixture in oracle")
            fixtures[name] = geometry
            optima[name] = {
                "hyperedge_count": geometry.size,
                "overlap_count": len(geometry.overlaps),
                "feasible_count": count,
                "optima": minima,
                "status": "failed" if issue else "verified",
                "errors": issue,
            }
        except (OSError, ValueError, KeyError, TypeError) as ex:
            issue.append(f"{oracle}: {name}: {ex}")
            optima[name] = {"status": "failed", "errors": issue}
        errors.extend(issue)
    if len({geometry.overlaps for geometry in fixtures.values()}) != len(fixtures):
        errors.append(f"{oracle}: repeated overlap patterns")
    return fixtures, optima, errors


def checked_layout(path, geometry, errors):
    layout = read_json(path)
    levels = placement_levels(layout, geometry, path.name, errors)
    metrics = evaluate(geometry, levels)
    check_embedded(layout, levels, geometry, metrics, path.name, errors)
    return layout, levels, metrics


def original_checks(path, geometry, errors):
    counts = (len(geometry.data["nodes"]), geometry.size, len(geometry.overlaps))
    if counts != (627, 383, 4431):
        errors.append(f"{path}: original input or overlap count changed: {counts!r}")


def verify_case(case, directory, geometry, optima, initial_by_input, plan, source, errors):
    for filename in ("initial.json", "checkpoint.json", "layout.json", "report.json", "run.log"):
        path = directory / filename
        if not path.is_file() or not path.stat().st_size:
            errors.append(f"{path}: missing or empty case artifact")
    if errors:
        return None

    report = read_json(directory / "report.json")
    metadata = report["case"]
    for name in ("key", "phase", "stage", "fixture", "method", "objective", "seed", "path"):
        if metadata.get(name) != case.get(name):
            errors.append(f"report.case.{name}: {metadata.get(name)!r} disagrees with progress {case.get(name)!r}")

    provenance = report["provenance"]
    if provenance.get("input_sha256") != geometry.hashes["input_sha256"]:
        errors.append("report.provenance.input_sha256: differs from the exact bytes of the expected input")
    if provenance.get("source_sha256") != source.get("source_sha256"):
        errors.append("report.provenance.source_sha256: differs from manifest source hashes")
    if provenance.get("environment") != source.get("environment"):
        errors.append("report.provenance.environment: differs from manifest execution environment")
    if len(provenance.get("git_commit", "")) != 40 or not provenance.get("command"):
        errors.append("report.provenance: missing Git commit or execution command")
    if not isinstance(provenance.get("git_status"), list) or not provenance.get("tracked_diff_sha256"):
        errors.append("report.provenance: missing uncommitted source provenance")
    if report.get("problem_signature") != geometry.hashes["problem_signature"]:
        errors.append("report.problem_signature: differs from independently calculated signature")
    if report.get("overlap_count") != len(geometry.overlaps):
        errors.append("report.overlap_count: differs from independent geometry")

    small = case["phase"] == "small"
    expected_input_path = f"small/{case['fixture']}/input.json" if small else plan["input_path"]
    if provenance.get("input_path") != expected_input_path or provenance.get("input_base") != (
        "output" if small else "repository"
    ):
        errors.append("report.provenance: input path or base differs from the planned artifact")
    settings = report["settings"]
    budget = plan.get(f"{case['phase']}_case_budget_seconds")
    expected_settings = {
        **plan["solver_settings"],
        "reg": EXPECTED_REG,
        "penalty": EXPECTED_PENALTY,
        "seed": case["seed"],
        "method": "anneal" if case["method"] == "anneal" else "greedy",
        "optimized_objective": case["objective"],
        "restart": case["method"] == "restart-greedy",
        "max_iterations": plan["small_case_max_attempts"] if small else plan["time_case_iteration_guard"],
        "budget_seconds": budget,
        "time_budget": budget or 0.0,
        "finalization_reserve_seconds": plan["finalization_reserve_seconds"] if budget else 0.0,
    }
    for name, expected in expected_settings.items():
        if settings.get(name) != expected or name not in settings:
            errors.append(f"report.settings.{name}: {settings.get(name)!r} disagrees with manifest {expected!r}")

    initial_layout, initial, initial_metrics = checked_layout(directory / "initial.json", geometry, errors)
    _, checkpoint, _ = checked_layout(directory / "checkpoint.json", geometry, errors)
    layout, levels, metrics = checked_layout(directory / "layout.json", geometry, errors)
    if checkpoint != levels:
        errors.append("checkpoint.json: stored best layers differ from layout.json")
    initial_tree = initial_layout["tree_layer_search"]["best"]
    start = (initial, initial_tree["parent"], initial_tree["children"])
    if start != initial_by_input.setdefault(geometry.hashes["input_sha256"], start):
        errors.append("initial.json: BFS + repair placement or tree differs from other cases with the same input")
    if "tree_layer_search" in layout:
        tree_initial = layout["tree_layer_search"]["initial"]
        check_tree(tree_initial, initial, "layout.json.initial", errors)
        if tree_initial != initial_tree:
            errors.append("layout.json.initial: differs from the state in initial.json")
        compare_metrics(tree_initial["metrics"], initial_metrics, "layout.json.initial.metrics", errors)
    compare_metrics(report["initial_metrics"], initial_metrics, "report.initial_metrics", errors)
    compare_metrics(report["best_metrics"], metrics, "report.best_metrics", errors)

    objective = case["objective"]
    scores = optimized_values(metrics, EXPECTED_PENALTY)
    best_score = scores[objective]
    initial_score = optimized_values(initial_metrics, EXPECTED_PENALTY)[objective]
    compare_metrics(report["all_objectives"], scores, "report.all_objectives", errors)
    compare_metrics(report, {"initial_optimized_value": initial_score}, "report", errors)
    if not close(report.get("optimized_value"), best_score):
        errors.append(f"report.optimized_value: {report.get('optimized_value')!r}, recomputed {best_score!r}")
    if initial_score < best_score and not close(initial_score, best_score):
        errors.append("layout.json: optimized value is worse than its starting layout")
    metrics["optimized_value"] = best_score
    metrics["optimized_values"] = scores
    metrics["initial_metrics"] = initial_metrics
    search = report["search"]
    attempts = search["attempts"]
    timing = report["timing_seconds"]
    wall = case["wall_seconds"]
    if attempts != case["search_attempts"] or not 0 < attempts <= expected_settings["max_iterations"]:
        errors.append("report.search.attempts: invalid count or differs from progress.json")
    stop = "max_iterations_reached" if small else "time_budget_exhausted"
    if search["stop_reason"] != stop or case["reason"] != stop:
        errors.append(f"report.search.stop_reason: expected {stop}, got {search['stop_reason']!r}")
    if not 0 <= timing["search"] <= timing["total"] <= wall or budget is not None and wall > budget:
        errors.append(f"report.timing_seconds: invalid search {timing['search']}, total {timing['total']}, wall {wall}")
    if sum(row["attempts"] for row in search["by_move"].values()) != attempts:
        errors.append("report.search.by_move: attempt counts do not sum to the total")
    if sum(row["accepted"] for row in search["by_move"].values()) != search["accepted"]:
        errors.append("report.search.by_move: acceptance counts do not sum to the total")
    if expected_settings["method"] == "greedy" and search["uphill_accepted"] != 0:
        errors.append("report.search.uphill_accepted: greedy accepted an uphill move")
    if expected_settings["restart"]:
        restart_count = len(plan["restart"]["fractions"])
        seeds = [case["seed"] + i * plan["restart"]["seed_step"] for i in range(1, restart_count + 1)]
        if [restart["seed"] for restart in search["restarts"]] != seeds:
            errors.append("report.search.restarts: restart count or seed schedule differs from the manifest")
    elif search["restarts"]:
        errors.append("report.search.restarts: unexpected restarts")
    history = report["history"]
    if (
        not history
        or len(history) != search["best_improvements"] + 1
        or history[0]["attempt"] != 0
        or not close(history[0]["objective"], initial_score)
        or not close(history[-1]["objective"], best_score)
        or history[-1]["attempt"] > attempts
        or history[-1]["elapsed_seconds"] > wall
        or any(
            next_event["attempt"] <= event["attempt"]
            or next_event["elapsed_seconds"] < event["elapsed_seconds"]
            or next_event["objective"] >= event["objective"]
            for event, next_event in zip(history, history[1:])
        )
    ):
        errors.append("report.history: nonmonotonic or inconsistent best-value history")
    metrics.update(total_seconds=wall, search_seconds=timing["search"], attempts=attempts)
    if small and optima:
        minimum = optima["optima"][objective]["value"]
        gap = best_score - minimum
        if gap < 0 and not close(gap, 0):
            errors.append(f"layout.json: optimized value {best_score!r} is below the exhaustive optimum {minimum!r}")
        if not close(report.get("small_exact_gap"), gap):
            errors.append(f"report.small_exact_gap: {report.get('small_exact_gap')!r}, recomputed {gap!r}")
        if not close(report.get("small_exact_value"), minimum):
            errors.append(f"report.small_exact_value: {report.get('small_exact_value')!r}, exhaustive {minimum!r}")
        metrics["exact_optimum"] = minimum
        metrics["exact_gap"] = 0.0 if close(gap, 0) else gap
    return metrics


def case_timestamps(path, key, reason, attempts):
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix, start, logged_key = lines[0].split(maxsplit=2)
    if prefix != "Started" or logged_key != key:
        raise ValueError(f"{path}: case start or identifier mismatch")
    expected = f". {reason}, {attempts} attempts."
    finishes = [line for line in lines if line.startswith("Finished ")]
    if len(finishes) != 1 or not finishes[0].endswith(expected):
        raise ValueError(f"{path}: case completion or attempt count mismatch")
    begin = datetime.fromisoformat(start)
    end = datetime.fromisoformat(finishes[0].removeprefix("Finished ").removesuffix(expected))
    if end < begin:
        raise ValueError(f"{path}: finish timestamp precedes start")
    return begin, end


PHASES = {
    "smoke": {"smoke"},
    "small": {"small"},
    "measure": {"measure"},
    "all": {"smoke", "small", "measure"},
}
MAIN_CONDITIONS = {
    ("stage1", "greedy", "squared"),
    ("stage1", "anneal", "squared"),
    ("stage1", "restart-greedy", "squared"),
    ("stage2", "anneal", "weighted-l1"),
    ("stage2", "anneal", "pillar"),
    ("stage2", "anneal", "squared-plus-layers"),
}


def check_plan(plan, progress, original, errors):
    compare_metrics(
        plan,
        {
            "reg": EXPECTED_REG,
            "penalty": EXPECTED_PENALTY,
            "measure_case_budget_seconds": EXPECTED_CASE_BUDGET,
            "measure_total_budget_seconds": EXPECTED_TOTAL_BUDGET,
        },
        "manifest.plan",
        errors,
    )
    if plan.get("experiment_id") != DEFAULT_RUN.name or progress.get("experiment_id") != DEFAULT_RUN.name:
        errors.append("manifest.json or progress.json: unexpected experiment ID")
    if plan.get("input_path") != original.name or plan.get("input_base") != "repository":
        errors.append("manifest.plan: must use original.json from the repository")
    if plan.get("input_sha256") != file_hash(original):
        errors.append("manifest.plan.input_sha256: original.json digest differs")
    if plan.get("seeds") != list(EXPECTED_SEEDS):
        errors.append("manifest.plan.seeds: expected seeds 0–4")
    if not 0 <= progress["measurement_seconds"] <= EXPECTED_TOTAL_BUDGET:
        errors.append("progress.measurement_seconds: outside the declared total budget")

    expected = {(stage, method, objective, seed) for stage, method, objective in MAIN_CONDITIONS for seed in EXPECTED_SEEDS}
    measured = [row for row in plan["cases"] if row["phase"] == "measure"]
    if len(measured) != len(expected) or {
        (row["stage"], row["method"], row["objective"], row["seed"]) for row in measured
    } != expected:
        errors.append("manifest.plan.cases: original-input measurement matrix differs from requested comparisons")
    if any(row["fixture"] != "original" for row in measured):
        errors.append("manifest.plan.cases: all measurement cases must use original.json")

    expected_baselines = {f"stage1/anneal-squared/seed-{seed}" for seed in EXPECTED_SEEDS}
    reused = plan["reused_baselines"]
    if len(reused) != len(EXPECTED_SEEDS) or any(
        row["source"] != row["source_case_key"]
        or row["source"] not in expected_baselines
        or (row["stage"], row["method"], row["objective"]) != ("stage2", "anneal", "squared")
        or row["source"] != f"stage1/anneal-squared/seed-{row['seed']}"
        for row in reused
    ) or {row["source"] for row in reused} != expected_baselines:
        errors.append("manifest.plan.reused_baselines: expected identical stage 1 anneal-squared cases")

    fields = ("key", "phase", "stage", "fixture", "method", "objective", "seed")
    plan_fields = [tuple(row[field] for field in fields) for row in plan["cases"]]
    actual_fields = [tuple(row[field] for field in fields) for row in progress["cases"]]
    if plan_fields != actual_fields:
        errors.append("progress.cases: row order or conditions differ from the saved plan")


def verify_run(run, phase):
    report = {
        "schema_version": 1,
        "run": str(run),
        "status": "verified",
        "errors": [],
        "cases": [],
        "small": {},
    }
    errors = report["errors"]
    selected = PHASES[phase]
    try:
        manifest = read_json(run / "manifest.json")
        plan, provenance = manifest["plan"], manifest["provenance"]
        progress = read_json(run / "progress.json")
        source = REPO / "original.json"
        check_plan(plan, progress, source, errors)
        all_cases = progress["cases"]
        cases = [case for case in all_cases if case["phase"] in selected]
        small, report["small"], small_errors = verify_oracle(run) if "small" in selected else ({}, {}, [])
        errors.extend(small_errors)
        if any(case["phase"] == "small" for case in cases) and not small:
            errors.append("small/oracle.json: missing exhaustive fixture results")
        if not cases:
            errors.append(f"progress.json: no planned cases for {phase}")
        original = make_geometry(source, EXPECTED_REG) if any(case["phase"] != "small" for case in cases) else None
        if original is not None:
            original_checks(source, original, errors)
    except (OSError, ValueError, KeyError, TypeError) as ex:
        errors.append(f"{run}: {ex}")
        report["status"] = "failed"
        return report

    initial_by_input = {}
    seen = set()
    intervals = []
    for case in cases:
        result = {name: case[name] for name in ("key", "phase", "stage", "fixture", "method", "objective", "seed", "path")}
        result["status"] = "pending"
        result["execution_status"] = case["status"]
        result["report"] = f"{case['path']}/report.json"
        result["layout"] = f"{case['path']}/layout.json"
        issue = result["errors"] = []
        report["cases"].append(result)
        if result["key"] in seen:
            issue.append("progress.json: duplicate case key")
        seen.add(result["key"])
        if case["status"] != "completed":
            if case["status"] == "failed":
                issue.append(f"progress.json: run reported failure for {case['key']}: {case.get('reason')}")
            elif case["status"] not in {"pending", "running", "not_run", "interrupted"}:
                issue.append(f"progress.json: unknown case status {case['status']!r}")
            if issue:
                result["status"] = "failed"
            continue
        try:
            directory = artifact_path(run, case["path"])
            if case["phase"] == "small":
                geometry = small[case["fixture"]]
                oracle = report["small"][case["fixture"]]
                if oracle["status"] != "verified":
                    issue.append("small/oracle.json: the case fixture failed independent enumeration")
            else:
                geometry, oracle = original, None
            metrics = verify_case(case, directory, geometry, oracle, initial_by_input, plan, provenance, issue)
            if not issue:
                started, finished = case_timestamps(
                    directory / "run.log", case["key"], case["reason"], case["search_attempts"],
                )
                if case["phase"] == "measure":
                    intervals.append((started, finished, case["key"]))
                result["metrics"] = metrics
                result["report_sha256"] = file_hash(directory / "report.json")
                result["layout_sha256"] = file_hash(directory / "layout.json")
        except (OSError, ValueError, KeyError, TypeError) as ex:
            issue.append(str(ex))
        result["status"] = "failed" if issue else "verified"

    ordered = sorted(intervals)
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] > current[0]:
            errors.append(f"Measurement overlap between {previous[2]} and {current[2]}")

    report["counts"] = {
        status: sum(case["status"] == status for case in report["cases"])
        for status in ("verified", "failed", "pending")
    }
    if errors or report["counts"]["failed"]:
        report["status"] = "failed"
    elif report["counts"]["pending"]:
        report["status"] = "partial"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--phase", choices=tuple(PHASES), default="all")
    parser.add_argument("--output", type=Path, help="Write only the review JSON to this file")
    args = parser.parse_args()
    summary = verify_run(args.run.resolve(), args.phase)
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.write_text(serialized, encoding="utf-8")
        print(f"{summary['status']}: {summary.get('counts', {})}")
    return int(summary["status"] == "failed")


if __name__ == "__main__":
    sys.exit(main())
