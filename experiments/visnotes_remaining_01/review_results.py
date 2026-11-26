"""Audit saved VisNotes artifacts without running any optimizer."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations, groupby, product
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
THREAD_LIMITS = dict.fromkeys(
    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1",
)
os.environ.update(THREAD_LIMITS)

import numpy as np
from shapely import contains_xy
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from experiments.tree_optimization_comparison_01.review_results import (
    check_embedded, check_tree, compare_metrics, evaluate, file_hash, make_geometry, placement_levels, read_json,
)


DEFAULT_RUN = ROOT / "results/experiments/visnotes_remaining_01"
ARTICLE = "https://vdslab.esa.io/posts/9193"
REGULARIZATION = 1e-5
TRIALS = 20_000
SEEDS = tuple(range(5))
XY_VARIANTS = ("original", "paper", "scalable", "arranged")
FIXTURES = ("four-path", "five-cycle", "six-mixed")
SOURCES = {
    "original": ROOT / "original.json",
    **{name: ROOT / f"results/layouts/original_polygon_{name}.json" for name in XY_VARIANTS[1:]},
    **{
        name: ROOT / f"results/experiments/tree_optimization_comparison_01/small/{name}/input.json"
        for name in FIXTURES
    },
}
COMPACT = "solve_layers_compact.py"
TREE = "solve_layers_tree.py"
QP = "solve_layers.py"
LP = "solve_layers_dsatur_l1.py"
BASELINES = ("qp-l2", "lp-l1")


@dataclass(frozen=True)
class Phase:
    runner: str
    inputs: tuple
    conditions: tuple
    total_budget: int
    case_budget: int | None
    seeds: tuple = SEEDS


OBJECTIVES = (
    "pillar", "squared", "degree-span", "hybrid-0.1", "hybrid-0.25", "hybrid-0.5", "hybrid-1",
    "hybrid-0.25-plus-1", "hybrid-0.25-plus-2",
)
BENCHMARKS = (
    "greedy-input", "greedy-size", "greedy-degree", "dsatur", "dsatur-free", *BASELINES,
    "tree-sa", "swap", "swap-move", "swap-move-kempe", "full",
)
PHASES = {
    "exp04": Phase("exp04_solver_benchmark", ("original",), BENCHMARKS, 540, 35),
    "exp05": Phase("exp05_xy_z_paradox_study", XY_VARIANTS, ("full",), 480, 35),
    "exp02": Phase("exp02_clean_benchmarks_prep", XY_VARIANTS, (), 40, None),
    "exp03": Phase("exp03_objective_hub_pareto", ("original",), OBJECTIVES, 720, 35),
    "exp03-small": Phase("exp03_objective_hub_pareto", FIXTURES, OBJECTIVES, 315, 35, (0,)),
    "exp04-legacy-reference": Phase("exp04_solver_benchmark", ("original",), BASELINES, 390, 180),
}


def condition_settings(name):
    solver, method, objective = COMPACT, "anneal", "hybrid"
    alpha, warm = 0.25, 0
    moves = ["move", "kempe", "swap", "shift"]
    if name.startswith("greedy-") or name.startswith("dsatur"):
        method = name if name != "dsatur-free" else "dsatur"
    elif name in BASELINES or name == "tree-sa":
        solver, objective = {
            "qp-l2": (QP, "squared"), "lp-l1": (LP, "weighted-l1"), "tree-sa": (TREE, "squared"),
        }[name]
    elif name.startswith("swap"):
        moves = name.split("-")
    elif name in {"pillar", "squared", "degree-span"}:
        objective = name
    elif name.startswith("hybrid-"):
        parts = name.split("-")
        alpha = float(parts[1])
        warm = int(parts[-1]) if len(parts) == 4 else 0
    elif name != "full":
        raise ValueError(f"Unexpected condition {name}")
    return {
        "solver": solver, "method": method, "objective": objective, "alpha": alpha,
        "beta": 0.005, "warm_start": warm, "moves": moves,
        "optimize_free_edges": True if name == "dsatur-free" else None,
    }


def planned_cases(phase):
    result = []
    for source in phase.inputs:
        for condition in phase.conditions:
            settings = condition_settings(condition)
            deterministic = settings["solver"] in (QP, LP) or settings["method"] != "anneal"
            for seed in (None,) if deterministic else phase.seeds:
                result.append({"input": source, "condition": condition, "seed": seed})
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, label):
    errors = []
    expected_numbers = {}
    for metric, value in expected.items():
        if value is None or isinstance(value, list):
            if metric not in actual or actual[metric] != value:
                errors.append(f"{label}.{metric} differs")
        else:
            expected_numbers[metric] = value
    compare_metrics(actual, expected_numbers, label, errors)
    require(not errors, "; ".join(errors))


def first_fit(geometry, method="anneal", overlaps=None):
    neighbors = [set() for _ in range(geometry.size)]
    for a, b in geometry.overlaps if overlaps is None else overlaps:
        neighbors[a].add(b)
        neighbors[b].add(a)
    occupied = [set() for _ in neighbors]
    colors = [None] * geometry.size
    remaining = set(range(geometry.size))
    sizes = [len(set(edge["nodes"])) for edge in geometry.data["hyperedges"]]
    while remaining:
        if method == "greedy-size":
            node = min(remaining, key=lambda i: (-sizes[i], i))
        elif method == "greedy-degree":
            node = min(remaining, key=lambda i: (-len(neighbors[i]), i))
        elif method == "greedy-input":
            node = min(remaining)
        else:
            node = min(remaining, key=lambda i: (-len(occupied[i]), -len(neighbors[i]), i))
        color = next(c for c in range(geometry.size) if c not in occupied[node])
        colors[node] = color
        remaining.remove(node)
        for neighbor in neighbors[node] & remaining:
            occupied[neighbor].add(color)
    return colors


def measured(geometry, levels, continuous=False):
    require(len(levels) == geometry.size, "layer vector length differs from input edge count")
    metrics = evaluate(geometry, levels)
    if continuous:
        require(all(math.isfinite(level) for level in levels), "continuous height must be finite")
        require(abs(levels[0]) <= 1e-6, "continuous layout lost its zero anchor")
        colors = first_fit(geometry)
        gaps = [
            levels[b] - levels[a] if colors[a] < colors[b] else levels[a] - levels[b]
            for a, b in geometry.overlaps
        ]
        metrics["overlap_violation_count"] = sum(abs(gap) < 1.0 - 1e-5 for gap in gaps)
        metrics["order_violation_count"] = sum(gap < 1.0 - 1e-5 for gap in gaps)
        metrics["max_order_violation"] = max(0.0, 1.0 - min(gaps)) if gaps else 0.0
        metrics["distinct_height_values"] = metrics.pop("used_layer_count")
    else:
        require(
            all(type(level) is int for level in levels)
            and set(levels) == set(range(len(set(levels)))),
            "integer levels must be compact, nonnegative, and integral",
        )
    require(
        metrics["overlap_violation_count"] == metrics["order_violation_count"] == 0,
        "saved levels violate independently reconstructed overlap constraints",
    )
    by_node = defaultdict(list)
    constrained = {edge for pair in geometry.overlaps for edge in pair}
    for index, edge in enumerate(geometry.data["hyperedges"]):
        for node in set(edge["nodes"]):
            by_node[node].append(index)
    rows = []
    for node in geometry.data["nodes"]:
        members = by_node[node["id"]]
        heights = [levels[index] for index in members]
        rows.append({
            "node_id": node["id"],
            "degree": len(members),
            "constrained_degree": sum(index in constrained for index in members),
            "span": max(heights) - min(heights) if heights else 0,
        })
    hubs = sorted(
        (row for row in rows if row["constrained_degree"] >= 2),
        key=lambda row: (-row["constrained_degree"], -row["degree"]),
    )[:5]
    metrics.update(
        pillar_square_sum=sum(row["span"] ** 2 for row in rows),
        degree_span_square_sum=sum(row["degree"] * row["span"] ** 2 for row in rows),
        top5_constrained_hub_mean_span=sum(hub["span"] for hub in hubs) / len(hubs) if hubs else None,
        constrained_hubs=hubs,
    )
    return metrics


def measured_layout(geometry, layout, continuous):
    require(layout["nodes"] == geometry.data["nodes"], "nodes or fixed XY changed")
    if continuous:
        actual = [{k: v for k, v in edge.items() if k != "layer"} for edge in layout["hyperedges"]]
        expected = [{k: v for k, v in edge.items() if k != "layer"} for edge in geometry.data["hyperedges"]]
        require(actual == expected, "edge metadata, order or memberships changed")
        require(sorted(map(tuple, layout["overlaps"])) == list(geometry.overlaps), "overlaps differ from STRtree")
        levels = tuple(edge["layer"] for edge in layout["hyperedges"])
    else:
        errors = []
        levels = placement_levels(layout, geometry, "layout", errors)
        require(not errors, "; ".join(errors))
    return levels, measured(geometry, levels, continuous)


def check_clique(witness, overlaps, size):
    require(
        bool(witness) and len(set(witness)) == len(witness)
        and all(type(node) is int and 0 <= node < size for node in witness)
        and all(pair in overlaps for pair in combinations(sorted(witness), 2)),
        "clique certificate is invalid",
    )


def score(metrics, settings):
    pillar = metrics["pillar_length"]
    squared = metrics["l2_scaled_qp"]
    return {
        "hybrid": pillar + settings["alpha"] * squared,
        "squared": squared,
        "pillar": pillar,
        "degree-span": pillar + settings["beta"] * metrics["degree_span_square_sum"],
    }[settings["objective"]]


def audit_small(geometry, stage, settings, oracle_cache):
    cap = stage["cap"]
    cache_key = geometry.hashes["problem_signature"], cap
    if cache_key not in oracle_cache:
        oracle_cache[cache_key] = [
            measured(geometry, levels)
            for levels in product(range(cap), repeat=geometry.size)
            if set(levels) == set(range(len(set(levels))))
            and all(levels[a] != levels[b] for a, b in geometry.overlaps)
        ]
    feasible = oracle_cache[cache_key]
    optimum = min(score(metrics, settings) for metrics in feasible)
    oracle = stage["oracle"]
    enumeration = oracle["enumeration"]
    require(enumeration["feasible_count"] == len(feasible), "enumeration has the wrong feasible count")
    require(max(enumeration["layers"]) < cap, "enumeration witness exceeds its cap")
    value = score(measured(geometry, enumeration["layers"]), settings)
    compare(enumeration, {"value": optimum}, "oracle.enumeration")
    compare(stage, {"optimality_gap": max(0.0, stage["optimized_value"] - optimum)}, "oracle.gap")
    require(math.isclose(value, optimum, rel_tol=1e-9, abs_tol=1e-7), "enumeration witness is not optimal")
    milp = oracle["milp"]
    if milp["status"] == 0:
        require(max(milp["layers"]) < cap, "MILP witness exceeds its cap")
        value = score(measured(geometry, milp["layers"]), settings)
        compare(milp, {"value": optimum, "reported_value": optimum, "lower_bound": optimum}, "oracle.milp")
        require(math.isclose(value, optimum, rel_tol=1e-9, abs_tol=1e-7), "MILP witness is not optimal")
    else:
        require(milp["status"] == 1, "MILP status is incompatible with feasible enumeration")
    return milp["status"] == 0


def audit_compact(geometry, levels, report, case, method, cache):
    internal = case["solver_internal_budget"]
    settings = {
        "method": method["method"], "objective": method["objective"], "alpha": method["alpha"],
        "beta": method["beta"], "seed": case["seed"] or 0, "reg": REGULARIZATION,
        "max_iterations": TRIALS, "time_budget": internal, "layer_cap": None,
        "warm_start_extra_layers": method["warm_start"], "moves": method["moves"],
        "optimize_free_edges": method["optimize_free_edges"] or method["method"] == "anneal",
    }
    require(report["settings"] == settings, "compact solver settings differ from fixed conditions")
    require(
        report["input_sha256"] == geometry.hashes["input_sha256"]
        and report["problem_signature"] == geometry.hashes["problem_signature"]
        and report["quadratic_metric"] == "l2_scaled_qp",
        "compact report has incompatible source, geometry, or objective normalization",
    )
    initial = first_fit(geometry, method["method"])
    require(report["initial_layers"] == initial, "initial heights do not match independent coloring")
    prior_metrics = measured(geometry, initial)
    compare(report["initial_metrics"], prior_metrics, "report.initial")
    initial_colors = prior_metrics["used_layer_count"]
    require(
        report["initial_color_count"] == initial_colors
        and report["dsatur_upper_bound"] == (
            initial_colors if method["method"] in ("anneal", "dsatur") else None
        ),
        "reported initialization color bounds differ",
    )
    caps = list(range(initial_colors, min(geometry.size, initial_colors + method["warm_start"]) + 1))
    stages = report["stages"]
    require(
        stages and report["planned_caps"] == caps
        and [stage["cap"] for stage in stages] == caps[:len(stages)]
        and list(levels) == stages[-1]["layers"] and len(stages) == len(case["stages"]),
        "cap sequence or final layout differs",
    )
    final_reason = report["termination_reason"]
    normal = "iteration_budget" if method["method"] == "anneal" else "initialization_only"
    elapsed = report["elapsed_seconds"]
    require(
        0 <= report["preprocessing_seconds"] <= elapsed <= case["solver_wall_seconds"]
        and final_reason == case["termination_reason"]
        and final_reason in (normal, "time_budget")
        and (final_reason != "time_budget" or elapsed >= internal - 1e-6)
        and (len(stages) == len(caps) or final_reason == "time_budget"),
        "reported compact timing and termination are inconsistent",
    )
    attempts = 0
    prior_elapsed = report["preprocessing_seconds"]
    prior_levels = initial
    upper = initial_colors
    normal_iterations, remainder = divmod(TRIALS if method["method"] == "anneal" else 0, len(caps))
    small_complete = True
    for index, (stage, recorded) in enumerate(zip(stages, case["stages"])):
        require(stage["initial_layers"] == prior_levels, "a stage does not start from the previous solution")
        stage_metrics = measured(geometry, stage["layers"])
        compare(stage["metrics"], stage_metrics, "report.stage")
        compare(recorded["metrics"], stage_metrics, "case.stage")
        require(max(stage["layers"]) < stage["cap"], "completed stage exceeds layer cap")
        for field in ("cap", "iterations", "elapsed_seconds", "termination_reason", "trajectory"):
            require(recorded[field] == stage[field], f"case stage differs from report on {field}")
        value = score(stage_metrics, method)
        compare(stage, {"optimized_value": value}, "report.stage")
        compare(recorded, {"optimized_value": value}, "case.stage")
        budget = normal_iterations + (index < remainder)
        count = stage["iterations"]
        stage_reason = stage["termination_reason"]
        require(
            stage["iteration_budget"] == budget and type(count) is int and 0 <= count <= budget
            and 0 <= stage["elapsed_seconds"] <= elapsed
            and (
                (stage_reason == normal and count == budget)
                or (stage_reason == final_reason == "time_budget" and index + 1 == len(stages))
            ),
            "stage iteration count or timing does not support the recorded termination",
        )
        trajectory = stage["trajectory"]
        require(trajectory and trajectory[0]["iteration"] == attempts, "stage trajectory lacks its start")
        prior_value = score(prior_metrics, method)
        compare(
            trajectory[0],
            {"optimized_value": prior_value, "used_layer_count": prior_metrics["used_layer_count"]},
            "trajectory.start",
        )
        compare(
            trajectory[-1],
            {"optimized_value": value, "used_layer_count": stage_metrics["used_layer_count"]},
            "trajectory.end",
        )
        prior_iteration = attempts
        for point in trajectory:
            require(
                prior_elapsed <= point["elapsed_seconds"] <= elapsed
                and prior_iteration <= point["iteration"] <= attempts + count
                and 0 <= point["optimized_value"] <= prior_value + 1e-6
                and 1 <= point["used_layer_count"] <= stage["cap"],
                "stage trajectory ordering, cap or values are invalid",
            )
            prior_elapsed, prior_iteration, prior_value = (
                point["elapsed_seconds"], point["iteration"], point["optimized_value"],
            )
        prior_levels, prior_metrics = stage["layers"], stage_metrics
        upper = min(upper, stage_metrics["used_layer_count"])
        attempts += count
        if case["input"] in FIXTURES:
            small_complete = audit_small(geometry, recorded, method, cache) and small_complete
    compare(report["metrics"], prior_metrics, "report.final")
    compare(report, {"optimized_value": score(prior_metrics, method)}, "report.final")
    check_clique(report["clique_witness"], set(geometry.overlaps), geometry.size)
    lower = len(report["clique_witness"])
    require(
        report["clique_lower_bound"] == lower <= upper == report["coloring_upper_bound"]
        and report["proven_minimum_layers"] == (upper if lower == upper else None)
        and case["iterations"] == report["iterations"] == attempts <= TRIALS,
        "reported clique bound, chromatic certificate or iteration total is unsupported",
    )
    require(
        (case["status"] == "verified") == (small_complete and len(stages) == len(caps)),
        "case status disagrees with whether all planned stages and oracles completed",
    )
    return len(stages)


def audit_tree(geometry, levels, metrics, report, case, directory):
    expected = {
        "method": "anneal", "seed": case["seed"], "reg": REGULARIZATION,
        "max_iterations": TRIALS, "time_budget": case["solver_internal_budget"],
    }
    require(
        all(report["settings"].get(name) == value for name, value in expected.items())
        and report["provenance"]["input_sha256"] == geometry.hashes["input_sha256"]
        and report["problem_signature"] == geometry.hashes["problem_signature"],
        "Tree report does not describe its fixed settings and input",
    )
    summary = report["summary"]
    require(
        summary["stop_reason"] == case["termination_reason"] and summary["attempts"] == case["iterations"]
        and 0 < summary["attempts"] <= TRIALS and 0 <= summary["total_seconds"] <= case["solver_wall_seconds"]
        and (
            (summary["stop_reason"] == "max_iterations_reached" and summary["attempts"] == TRIALS)
            or (
                summary["stop_reason"] == "time_budget_exhausted"
                and summary["total_seconds"] >= expected["time_budget"] - 1e-6
            )
        ),
        "Tree report timing or termination reason is inconsistent",
    )
    errors = []
    check_tree(report["best"], levels, "report.best", errors)
    for filename, label in (("initial.json", "initial"), ("checkpoint.json", "best")):
        layout = read_json(directory / filename)
        saved_levels, saved_metrics = measured_layout(geometry, layout, continuous=False)
        check_tree(report[label], saved_levels, f"report.{label}", errors)
        base_metrics = evaluate(geometry, saved_levels)
        check_embedded(layout, saved_levels, geometry, base_metrics, filename, errors)
        compare(report[label]["metrics"], base_metrics, f"report.{label}")
        if label == "best":
            require(saved_levels == levels, "Tree final checkpoint differs from layout")
    require(not errors, "; ".join(errors))
    compare(report["best_metrics"], evaluate(geometry, levels), "report.best_metrics")
    compare(report, {"objective_value_solve_layers": metrics["l2_scaled_qp"]}, "report")
    require(case["trajectory"] == report["history"], "Tree summary trajectory differs from report")


def audit_legacy(layout, metrics, case, log, condition):
    if condition == "lp-l1":
        require(
            layout["solver"]["success"] and layout["regularization"] == REGULARIZATION
            and layout["solver"]["iterations"] == case["iterations"],
            "LP convergence or configuration is inconsistent",
        )
        compare(layout, {"objective_value": metrics["l1_regularized"]}, "layout")
        compare(layout["metrics"], {key: metrics[key] for key in layout["metrics"]}, "layout.metrics")
    else:
        match = re.search(r"Optimization converged successfully in (\d+) iterations", log)
        objective = re.search(r"Optimal objective value: ([-+0-9.eE]+)", log)
        require(match and int(match[1]) == case["iterations"] and objective, "QP convergence log is incomplete")
        require(
            math.isclose(float(objective[1]), metrics["l2_scaled_qp"], rel_tol=1e-9, abs_tol=1e-6),
            "QP objective in convergence log differs from independently recomputed value",
        )
    require(case["termination_reason"] == "converged", "legacy solver success lacks convergence")


def phase_command(phase, directory, python):
    result = [
        python, "-B", str(ROOT / "experiments" / phase.runner / "run_experiment.py"),
        "--output", str(directory), "--total-budget", str(phase.total_budget), "--inputs", *phase.inputs,
    ]
    if phase.case_budget is not None:
        result.extend((
            "--case-budget", str(phase.case_budget), "--max-iterations", str(TRIALS),
            "--seeds", *map(str, phase.seeds), "--conditions", *phase.conditions,
        ))
    return result


def phase_arguments(argv):
    require(len(argv) >= 4 and argv[1] == "-B", "phase command is missing its interpreter or runner")
    options = {}
    for token in argv[3:]:
        if token.startswith("--"):
            require(token not in options, f"duplicate phase option {token}")
            options[token] = []
        else:
            require(options, "phase command has an unlabeled value")
            options[next(reversed(options))].append(token)
    require(len(options.get("--output", [])) == 1, "phase command has an ambiguous output path")
    options["--output"] = [(ROOT / options["--output"][0]).resolve()]
    return (ROOT / argv[0]).resolve(), (ROOT / argv[2]).resolve(), options


def expected_solver_command(case, directory, method, python, internal):
    command = [
        python, "-B", str(ROOT / method["solver"]), str(SOURCES[case["input"]]),
        str(directory / "layout.json"), "--reg", str(REGULARIZATION),
    ]
    if method["solver"] == COMPACT:
        command.extend((
            "--report", str(directory / "report.json"), "--method", method["method"],
            "--objective", method["objective"], "--alpha", str(method["alpha"]), "--beta", str(method["beta"]),
            "--seed", str(case["seed"] or 0), "--max-iterations", str(TRIALS), "--time-budget", str(internal),
            "--warm-start-extra-layers", str(method["warm_start"]), "--moves", *method["moves"],
        ))
        if method["optimize_free_edges"]:
            command.append("--optimize-free-edges")
    if method["solver"] == TREE:
        command.extend((
            "--report", str(directory / "report.json"), "--initial-output", str(directory / "initial.json"),
            "--checkpoint", str(directory / "checkpoint.json"), "--seed", str(case["seed"]),
            "--max-iterations", str(TRIALS), "--time-budget", str(internal),
        ))
    return command


def audit_manifest(directory, phase, provenance):
    manifest = read_json(directory / "manifest.json")
    python = provenance if provenance is not None else manifest["provenance"]["environment"]["python"]
    local_python = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    require(Path(python).resolve() == local_python.resolve(), "manifest Python is not the planned virtual environment")
    require(
        manifest["experiment_id"] == phase.runner and manifest["source_article"] == ARTICLE
        and manifest["regularization"] == REGULARIZATION
        and manifest["trial_limit"] == (TRIALS if phase.case_budget is not None else None)
        and manifest["case_budget_seconds"] == phase.case_budget
        and manifest["total_budget_seconds"] == phase.total_budget
        and manifest["cases"] == planned_cases(phase),
        "manifest differs from fixed case matrix or budget",
    )
    require(
        manifest["provenance"]["environment"]["python"] == python
        and manifest["provenance"]["environment"]["thread_limits"] == THREAD_LIMITS
        and phase_arguments(manifest["provenance"]["command"]) == phase_arguments(phase_command(phase, directory, python))
        and Path(manifest["provenance"]["working_directory"]) == ROOT,
        "manifest command, interpreter, threads or working directory differ from plan",
    )
    commit = manifest["provenance"]["git_commit"]
    require(
        re.fullmatch(r"[a-f0-9]{40}", commit) and subprocess.run(
            ["git", "cat-file", "-t", commit], cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip() == "commit",
        "run manifest Git commit is unavailable",
    )
    require(manifest["conditions"] == {
        condition: condition_settings(condition) for condition in phase.conditions
    }, "manifest method settings differ")
    hashes = {
        name: {"file": SOURCES[name].relative_to(ROOT).as_posix(), "sha256": file_hash(SOURCES[name])}
        for name in phase.inputs
    }
    require(manifest["input_files"] == hashes, "manifest input paths or hashes differ")
    expected_source = {
        f"experiments/{phase.runner}/run_experiment.py",
        "experiments/visnotes_common.py", "experiments/visnotes_metrics.py", "experiments/visnotes_exact.py",
        "experiments/tree_optimization_comparison_01/review_results.py", QP, LP,
    }
    solvers = {settings["solver"] for settings in manifest["conditions"].values()}
    expected_source.update(solvers)
    if solvers & {COMPACT, TREE}:
        expected_source.update((TREE, "solve_layers_dummy.py"))
    source_hashes = manifest["provenance"]["source_sha256"]
    require(set(source_hashes) == expected_source, "manifest does not describe all executable source paths")
    for path, digest in source_hashes.items():
        require(file_hash(ROOT / path) == digest, f"source hash differs from run manifest for {path}")
    return manifest, python


def audit_timeout_checkpoint(directory, geometry, case):
    path = directory / "checkpoint.json"
    if not path.is_file():
        return None
    layout = read_json(path)
    levels, metrics = measured_layout(geometry, layout, continuous=False)
    errors = []
    check_embedded(layout, levels, geometry, evaluate(geometry, levels), "timeout checkpoint", errors)
    require(not errors, "; ".join(errors))
    meta = layout["tree_layer_search"]
    require(
        meta["settings"]["seed"] == case["seed"] and meta["settings"]["reg"] == REGULARIZATION
        and meta["settings"]["method"] == "anneal"
        and meta["settings"]["max_iterations"] == TRIALS
        and meta["settings"]["time_budget"] == case["solver_internal_budget"]
        and meta["provenance"]["input_sha256"] == geometry.hashes["input_sha256"],
        "timeout checkpoint came from a different input or case setting",
    )
    summary = meta["summary"]
    history = meta["history"]
    last = history[-1]
    require(0 <= last["attempt"] <= summary["attempts"] <= TRIALS, "checkpoint attempt counters disagree")
    compare(
        last, {
            "objective": metrics["l2_scaled_qp"], "used_layer_count": metrics["used_layer_count"],
            "height_range": metrics["height_range"],
        },
        "timeout checkpoint history",
    )
    recorded_phase = summary.get("phase")
    require(
        recorded_phase in ("initial", "improvement") or (
            recorded_phase is None and summary.get("stop_reason") in ("max_iterations_reached", "time_budget_exhausted")
        ),
        "timeout checkpoint has no identifiable save phase",
    )
    return {
        "seed": case["seed"],
        "file": str(path),
        "sha256": file_hash(path),
        "input_sha256": geometry.hashes["input_sha256"],
        "last_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "save_phase": recorded_phase or "final_checkpoint",
        "attempts_at_save": summary["attempts"],
        "best_attempt": last["attempt"],
        "reported_total_seconds": summary.get("total_seconds"),
        "used_layer_count": metrics["used_layer_count"],
        "l2_scaled_qp": metrics["l2_scaled_qp"],
        "pillar_length": metrics["pillar_length"],
        "top5_constrained_hub_mean_span": metrics["top5_constrained_hub_mean_span"],
    }


def projected_counts(geometry, levels, factor, anchor_x, anchor_y):
    xy = {node["id"]: (node["x"], node["y"]) for node in geometry.data["nodes"]}
    angle, pitch = math.radians(45), math.radians(35)
    a, b = -math.sin(angle) * factor, math.cos(angle) * factor
    d = -math.cos(angle) * math.sin(pitch) * factor
    e = -math.sin(angle) * math.sin(pitch) * factor
    z_pixel = math.cos(pitch) * 3.84
    bottom = min(levels)
    projected = []
    for edge, height in zip(geometry.data["hyperedges"], levels):
        ring = [point for point, _ in groupby(xy[node] for node in edge["nodes"])]
        if ring and ring[-1] == ring[0]:
            ring.pop()
        if len(ring) < 3:
            continue
        polygon = Polygon(ring)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.area > 0:
            projected.append(affine_transform(
                polygon, [a, b, d, e, -a * anchor_x - b * anchor_y,
                          -d * anchor_x - e * anchor_y + (height - bottom) * z_pixel],
            ))
    boxes = [
        (
            math.ceil(low_x - 0.5), math.ceil(low_y - 0.5),
            math.floor(high_x - 0.5) + 1, math.floor(high_y - 0.5) + 1,
        )
        for low_x, low_y, high_x, high_y in (polygon.bounds for polygon in projected)
    ]
    left = min((box[0] for box in boxes), default=0)
    top = min((box[1] for box in boxes), default=0)
    width = max((box[2] for box in boxes), default=0) - left
    height = max((box[3] for box in boxes), default=0) - top
    require(0 <= width * height <= 10_000_000, "fixed-view projection exceeds the expected raster extent")
    counts = np.zeros((height, width), dtype=np.int32)
    for polygon, (x0, y0, x1, y1) in zip(projected, boxes):
        if x1 > x0 and y1 > y0:
            inside = contains_xy(polygon, np.arange(x0, x1)[None, :] + 0.5, np.arange(y0, y1)[:, None] + 0.5)
            counts[y0 - top:y1 - top, x0 - left:x1 - left] += inside
    return {
        "polygons_measured": len(projected),
        "covered_pixel_count": int(np.count_nonzero(counts)),
        "overdraw_pixel_count": int(np.count_nonzero(counts > 1)),
        "overdraw_sample_count": int(np.maximum(counts - 1, 0).sum()),
        "max_face_depth": int(counts.max(initial=0)),
    }


def audit_case(directory, phase, case, geometry, python, oracle_cache):
    status = case["status"]
    require(status in ("pending", "timeout", "failed", "partial", "verified"), "unknown case status")
    tag = "deterministic" if case["seed"] is None else f"seed-{case['seed']}"
    case_directory = directory / case["input"] / case["condition"] / tag
    if status == "pending":
        require(
            not (case_directory / "case.json").exists()
            and case["termination_reason"] in ("total_budget", "run_error", "interrupted"),
            "pending case has completed result or unknown reason",
        )
        return None, 0, None
    require(
        case_directory.resolve().is_relative_to(directory) and case_directory == (ROOT / case["case_directory"]),
        "case directory lies outside the phase or is mislabeled",
    )
    require(case == read_json(case_directory / "case.json"), "phase summary differs from case artifact")
    method = condition_settings(case["condition"])
    solver = method["solver"]
    wall = phase.case_budget
    internal = wall - max(min(2.0, wall * 0.25), wall * 0.08) if solver in (COMPACT, TREE) else None
    require(
        case["subprocess_wall_budget"] == wall and case["solver_internal_budget"] == internal
        and case["input_sha256"] == geometry.hashes["input_sha256"]
        and case["command"] == expected_solver_command(case, case_directory, method, python, internal),
        "case command, limits or input hash differ from the fixed matrix",
    )
    require(0 <= case.get("solver_wall_seconds", 0) <= case["case_elapsed_seconds"], "invalid case wall measurement")
    log = case_directory / "run.log"
    require(log.is_file(), "case has no saved solver log")
    if status in ("timeout", "failed"):
        if status == "timeout":
            require(
                case["termination_reason"] == "case_wall_budget" and case["solver_wall_seconds"] + 0.02 >= wall,
                "timeout does not reflect the recorded subprocess budget",
            )
            checkpoint = audit_timeout_checkpoint(case_directory, geometry, case) if solver == TREE else None
            return None, 0, checkpoint
        require(case["termination_reason"] == "case_error" and case["error"], "failed case has no error")
        return None, 0, None
    require(
        case["exit_code"] == 0 and case["solver_wall_seconds"] <= wall + 0.5,
        "success has a nonzero exit or exceeded the wall budget",
    )
    layout = read_json(case_directory / "layout.json")
    continuous = solver in (LP, QP)
    levels, metrics = measured_layout(geometry, layout, continuous)
    compare(case["metrics"], metrics, "case.metrics")
    require(case["height_type"] == ("continuous" if continuous else "integer"), "case mislabeled height space")
    stages = 0
    if continuous:
        audit_legacy(layout, metrics, case, log.read_text(encoding="utf-8"), case["condition"])
    else:
        report = read_json(case_directory / "report.json")
        if solver == TREE:
            audit_tree(geometry, levels, metrics, report, case, case_directory)
        else:
            stages = audit_compact(geometry, levels, report, case, method, oracle_cache)
    if phase.runner == "exp04_solver_benchmark":
        projection = case["projection"]
        require(projection["projection"] == "orthographic", "comparison projection changed")
        x = [node["x"] for node in geometry.data["nodes"]]
        y = [node["y"] for node in geometry.data["nodes"]]
        span = max(max(x) - min(x), max(y) - min(y)) or 1.0
        compare(
            projection, {
                "azimuth_degrees": 45.0, "elevation_degrees": 35.0, "pixels_per_xy_span": 384,
                "z_step_world": span / 100,
                "projected_pillar_length_px": metrics["pillar_length"] * math.cos(math.radians(35)) * 3.84,
                **projected_counts(geometry, levels, 384 / span, (min(x) + max(x)) / 2, (min(y) + max(y)) / 2),
            },
            "projection settings and pillar lengths",
        )
    return metrics, stages, None


def profile_groups(geometry):
    shared = {(a, b): count for a, b, count in geometry.pair_counts}
    exact = {
        str(number): {pair for pair in geometry.overlaps if shared[pair] == number}
        for number in range(3)
    }
    exact[">=3"] = {pair for pair in geometry.overlaps if shared[pair] >= 3}
    cumulative = {
        f">={number}": {pair for pair in geometry.overlaps if shared[pair] >= number}
        for number in range(4)
    }
    return {"exact_shared_vertices": exact, "cumulative_shared_vertices": cumulative}


def audit_input_profile(geometry, name, profile, reference=None):
    require(all(profile[key] == value for key, value in geometry.hashes.items()), "profile geometry hashes differ")
    members = [edge["nodes"] for edge in geometry.data["hyperedges"]]
    require(
        profile["file"] == SOURCES[name].relative_to(ROOT).as_posix()
        and profile["nodes"] == len(geometry.data["nodes"]) and profile["hyperedges"] == geometry.size
        and profile["hyperedges_by_distinct_member_count"] == dict(Counter(str(len(set(row))) for row in members))
        and profile["hyperedges_with_duplicate_memberships"] == sum(len(row) != len(set(row)) for row in members),
        "profile input shape counts differ",
    )
    if reference is not None:
        require(
            [row["id"] for row in geometry.data["nodes"]] == [row["id"] for row in reference["nodes"]],
            "XY variant does not preserve vertex IDs and order",
        )
        old_members = [edge["nodes"] for edge in reference["hyperedges"]]
        require(
            [row["id"] for row in geometry.data["hyperedges"]] == [row["id"] for row in reference["hyperedges"]]
            and list(map(Counter, members)) == list(map(Counter, old_members))
            and profile["changed_member_orders_vs_original"] == sum(a != b for a, b in zip(members, old_members)),
            "XY variant changes topology or misreports membership reordering",
        )
    groups = profile_groups(geometry)
    require(set(profile["overlap_profile"]) == set(groups), "profile overlap categories differ")
    for category, expected in groups.items():
        observed = profile["overlap_profile"][category]
        require(set(observed) == set(expected), "profile overlap strata differ")
        for key, pairs in expected.items():
            row = observed[key]
            witness = row["clique_witness_indices"]
            check_clique(witness, pairs, geometry.size)
            lower = len(witness)
            upper = len(set(first_fit(geometry, overlaps=pairs)))
            require(
                row["overlap_count"] == len(pairs) and row["clique_lower_bound"] == lower <= upper
                and row["dsatur_upper_bound"] == upper
                and row["chromatic_number"] == (upper if lower == upper else None),
                f"profile {category} {key} counts or certificate differ",
            )


def audit_comparisons(rows, verified):
    expected = defaultdict(list)
    for case, metrics in verified:
        expected[(case["input"], case["condition"])].append((case, metrics))
    indexed = {(row["input"], row["condition"]): row for row in rows}
    require(len(indexed) == len(rows) and indexed.keys() == expected.keys(), "comparison groups are missing or duplicated")
    for key, group in expected.items():
        values = defaultdict(list)
        for case, metrics in group:
            for name in (
                "used_layer_count", "distinct_height_values", "height_range", "l2_scaled_qp", "pillar_length",
                "top5_constrained_hub_mean_span", "overlap_violation_count",
            ):
                if metrics.get(name) is not None:
                    values[name].append(metrics[name])
            values["solver_wall_seconds"].append(case["solver_wall_seconds"])
            if projection := case.get("projection"):
                for name in ("overdraw_pixel_count", "projected_pillar_length_px"):
                    values[name].append(projection[name])
        row = indexed[key]
        require(
            row["verified_cases"] == len(group) and set(row["metrics"]) == set(values),
            "comparison count or metrics differ from independently verified cases",
        )
        for name, sample in values.items():
            compare(
                row["metrics"][name],
                {"minimum": min(sample), "median": median(sample), "maximum": max(sample)},
                f"comparison {key} {name}",
            )


def audit_frontiers(rows, verified):
    samples = defaultdict(list)
    for case, _ in verified:
        method = condition_settings(case["condition"])
        key = case["input"], case["seed"], method["objective"], method["alpha"], method["beta"]
        for stage in case["stages"]:
            samples[key].append({
                "condition": case["condition"], "cap": stage["cap"],
                **{
                    name: stage["metrics"][name]
                    for name in ("used_layer_count", "pillar_length", "l2_scaled_qp", "top5_constrained_hub_mean_span")
                },
            })
    keys = ("input", "seed", "objective", "alpha", "beta")
    indexed = {tuple(row[name] for name in keys): row["observed_frontier"] for row in rows}
    require(len(indexed) == len(rows) and indexed.keys() == samples.keys(), "observed frontier groups differ")
    for key, points in samples.items():
        pairs = [(row["used_layer_count"], row["pillar_length"]) for row in points]
        expected = [
            row for row, (layers, pillars) in zip(points, pairs)
            if not any(k <= layers and b <= pillars and (k < layers or b < pillars) for k, b in pairs)
        ]
        record_key = lambda row: (row["condition"], row["cap"])
        require(
            sorted(indexed[key], key=record_key) == sorted(expected, key=record_key),
            f"frontier {key} contains a missing or dominated point",
        )


def audit_phase(directory, phase, manifest, python, geometries, oracle_cache):
    summary = read_json(directory / "summary.json")
    cases = summary["cases"]
    labels = ("input", "condition", "seed")
    require(
        [{field: case[field] for field in labels} for case in cases] == manifest["cases"],
        "summary case list differs from the fixed manifest",
    )
    require(summary["case_counts"] == dict(Counter(case["status"] for case in cases)), "case counts do not add up")
    require(
        0 <= sum(case.get("case_elapsed_seconds", 0) for case in cases) <= summary["elapsed_seconds"] + 0.05,
        "serial case wall times exceed total phase elapsed time",
    )
    errors = []
    for name in phase.inputs:
        if name not in geometries:
            geometries[name] = make_geometry(SOURCES[name], REGULARIZATION)
    profiles = summary["input_profiles"]
    if directory.name == "exp05":
        ref = SOURCES["original"]
        require(manifest["structure_reference"] == {"file": "original.json", "sha256": file_hash(ref)}, "XY reference hash differs")
        reference = geometries["original"].data
    else:
        reference = None
    if directory.name in ("exp02", "exp05"):
        if summary["complete"]:
            require(set(profiles) == set(phase.inputs), "complete phase lacks input profiles")
        for name, profile in profiles.items():
            audit_input_profile(geometries[name], name, profile, reference)
    else:
        require(not profiles, "unexpected input profiles")
    completed = (
        not summary["interrupted"] and summary["error"] is None
        and all(case["status"] == "verified" for case in cases)
        and (directory.name not in ("exp02", "exp05") or set(profiles) == set(phase.inputs))
        and summary["elapsed_seconds"] <= phase.total_budget + 0.05
    )
    require(summary["complete"] == completed, "phase completion label differs from elapsed time and coverage")
    verified = []
    measurements = 0
    checked_stages = 0
    checkpoints = []
    for case in cases:
        try:
            measurements_data, stages, snapshot = audit_case(
                directory, phase, case, geometries[case["input"]], python, oracle_cache,
            )
        except (KeyError, TypeError, OSError, ValueError) as error:
            errors.append(
                f"{case['input']}/{case['condition']}/{case['seed']}: {error}",
            )
            continue
        if measurements_data is not None:
            measurements += 1
            checked_stages += stages
            if case["status"] == "verified":
                verified.append((case, measurements_data))
        if snapshot:
            checkpoints.append(snapshot)
    if not errors:
        comparable = verified if summary["error"] is None else []
        audit_comparisons(summary["comparisons"], comparable)
        if directory.name.startswith("exp03"):
            audit_frontiers(summary["observed_layer_pillar_frontiers"], comparable)
    result = {
        "planned_cases": len(cases),
        "recorded_counts": summary["case_counts"],
        "completed_within_budget": completed,
        "elapsed_seconds": summary["elapsed_seconds"],
        "recomputed_layouts": measurements,
        "recomputed_stages": checked_stages,
        "profiles_verified": len(profiles),
        "errors": errors,
    }
    overruns = [
        {key: case[key] for key in (
            "input", "condition", "seed", "subprocess_wall_budget", "solver_wall_seconds",
        )}
        for case in cases
        if case["status"] == "timeout" and case["solver_wall_seconds"] > case["subprocess_wall_budget"] + 5
    ]
    if overruns:
        result["recorded_timeout_overruns"] = {
            "cases": overruns,
            "note": "These observations are abnormal wall-clock durations with unknown cause. "
                    "They are excluded from the solver comparison metrics.",
        }
    if checkpoints:
        snapshot_stats = {}
        for field in ("used_layer_count", "l2_scaled_qp", "pillar_length", "top5_constrained_hub_mean_span"):
            values = [item[field] for item in checkpoints if item[field] is not None]
            if values:
                snapshot_stats[field] = {"minimum": min(values), "median": median(values), "maximum": max(values)}
        result["tree_timeout_checkpoints"] = {
            "count": len(checkpoints),
            "records": checkpoints,
            "observed_snapshot_statistics": snapshot_stats,
            "note": "Filesystem modification times and saved attempt counters describe snapshots only. "
                    "Checkpoint records do not imply a final solution or a measured 35-second objective.",
        }
    return result


def required_legacy_conditions(run):
    return tuple(
        case["condition"] for case in read_json(run / "exp04/summary.json")["cases"]
        if case["condition"] in BASELINES and case["status"] != "verified"
    )


def audit_journals(run, metadata_only):
    journals = sorted(run.glob("run-*.jsonl"))
    require(journals, "run has no launcher journal")
    launcher = Path(__file__).with_name("run_experiment.py")
    windows = {}
    outcomes = {}
    python = None
    for path in journals:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        require(events and events[0]["event"] == "plan", "launcher journal does not start with its plan")
        times = [datetime.fromisoformat(event["time_utc"]) for event in events]
        require(all(time.tzinfo is not None for time in times) and times == sorted(times), "journal timestamps are unordered")
        plan = events[0]
        require(
            plan["article"] == ARTICLE and plan["thread_limits"] == THREAD_LIMITS
            and plan["launcher_sha256"] == file_hash(launcher),
            "launcher digest, source article or thread settings differ",
        )
        order = plan["phase_order"]
        require(
            order == [phase for phase in PHASES if phase in order] and set(order) == set(plan["commands"]) and order,
            "launcher phase order or command assignments differ",
        )
        current = plan["commands"][order[0]][0]
        require(python is None or python == current, "the launcher used multiple Python executables")
        python = current
        for name in order:
            require(
                plan["commands"][name] == phase_command(PHASES[name], run / name, python),
                f"launcher command differs for {name}",
            )
        starts = {}
        observed = []
        for event in events[1:]:
            name = event.get("phase")
            kind = event["event"]
            if name is None:
                require(kind in ("done", "stopped") and event is events[-1], "unexpected launcher terminal event")
                continue
            if kind in ("started", "existing", "skipped"):
                observed.append(name)
            if kind == "started":
                require(not starts, "launcher started phases concurrently")
                phase = PHASES[name]
                if name == "exp04-legacy-reference":
                    phase = replace(phase, conditions=required_legacy_conditions(run))
                require(
                    event["command"] == phase_command(phase, run / name, python),
                    f"started phase differs from declared command for {name}",
                )
                starts[name] = event["time_utc"]
            elif kind == "finished" or kind == "interrupted":
                require(name in starts, f"launcher end event has no start for {name}")
                if kind == "finished":
                    require(
                        (event["exit_code"] == 0) == (event["status"] == "complete"),
                        f"launcher exit code and disposition disagree for {name}",
                    )
                windows[name] = starts.pop(name), event["time_utc"]
                outcomes[name] = event
            elif kind == "skipped":
                require(
                    name == "exp04-legacy-reference" and event["reason"] == "both_baselines_verified"
                    and not required_legacy_conditions(run),
                    "launcher skipped a required phase",
                )
                outcomes[name] = event
            elif kind == "existing":
                outcomes.setdefault(name, event)
            else:
                raise ValueError(f"unexpected launcher event {kind}")
        require(observed == order[:len(observed)], "launcher did not start phases in declared order")
        require(len(starts) <= int(metadata_only), "launcher phase has no end event")
        if not metadata_only:
            require(events[-1]["event"] in ("done", "stopped"), "launcher journal is still running")
            if events[-1]["event"] == "done":
                require(observed == order, "completed launcher journal did not process all selected phases")
                completed = all(
                    outcomes[name].get("status") == "complete" or outcomes[name]["event"] == "skipped"
                    for name in order
                )
                require(events[-1]["complete"] == completed, "launcher completion disagrees with phase outcomes")
    return python, outcomes, windows


def repeatability(run):
    repeats = (("exp03", "hybrid-0.25"), ("exp04", "full"), ("exp05", "full"))
    records = []
    for seed in SEEDS:
        cases = [run / phase / "original" / name / f"seed-{seed}" for phase, name in repeats]
        paths = [directory / "case.json" for directory in cases]
        if not all(
            path.is_file() and (case := read_json(path))["status"] == "verified" and case["iterations"] == TRIALS
            for path in paths
        ):
            records.append({"seed": seed, "matched": None, "reason": "fixed-iteration repeat did not complete"})
            continue
        hashes = {phase: file_hash(directory / "layout.json") for (phase, _), directory in zip(repeats, cases)}
        require(len(set(hashes.values())) == 1, f"fixed-iteration layouts differ across runs for seed {seed}")
        records.append({"seed": seed, "matched": True, "sha256": hashes})
    return records


def audit_small_recovery(run, python, geometries, oracle_cache, report):
    originals = read_json(run / "exp03-small/summary.json")["cases"]
    keys = ("input", "condition", "seed")

    def name(case):
        return tuple(case[key] for key in keys)

    completed = {name(case) for case in originals if case["status"] == "verified"}
    backlog = [name(case) for case in originals if case["status"] != "verified"]
    coverage = {"planned": len(originals), "initial_verified": len(completed), "recovery_verified": 0}
    recovery_path = run / "exp03-small-recovery-01"
    if recovery_path.is_dir():
        conditions = (
            "squared", "degree-span", "hybrid-0.1", "hybrid-0.25", "hybrid-0.5", "hybrid-1",
            "hybrid-0.25-plus-1", "hybrid-0.25-plus-2",
        )
        phase = Phase("exp03_objective_hub_pareto", ("six-mixed",), conditions, 160, 35, (0,))
        require([name(case) for case in planned_cases(phase)] == backlog, "small recovery differs from the exact backlog")
        manifest, _ = audit_manifest(recovery_path, phase, python)
        result = audit_phase(recovery_path, phase, manifest, python, geometries, oracle_cache)
        report["phases"][recovery_path.name] = result
        require(not result["errors"], "small recovery has independent case verification errors")
        recovery_cases = read_json(recovery_path / "summary.json")["cases"]
        recovered = {name(case) for case in recovery_cases if case["status"] == "verified"}
        require(not completed & recovered, "small recovery repeats an originally verified case")
        completed.update(recovered)
        coverage["recovery_verified"] = len(recovered)
    coverage["verified_unique"] = len(completed)
    coverage["remaining"] = [dict(zip(keys, case)) for case in backlog if case not in completed]
    return coverage


def audit(run, phases, metadata_only=False):
    report = {"run": str(run), "metadata_only": metadata_only, "phases": {}, "errors": []}
    try:
        python, outcomes, windows = audit_journals(run, metadata_only)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        report["errors"].append(f"journal: {error}")
        return report
    geometries = {}
    oracle_cache = {}
    for name in phases:
        try:
            phase = PHASES[name]
            if name == "exp04-legacy-reference":
                conditions = required_legacy_conditions(run)
                if not conditions:
                    require(outcomes[name]["event"] == "skipped", "unneeded legacy reference was not skipped")
                    report["phases"][name] = {"skipped": True}
                    continue
                phase = replace(phase, conditions=conditions)
            directory = run / name
            manifest, _ = audit_manifest(directory, phase, python)
            if metadata_only:
                report["phases"][name] = {"planned_cases": len(manifest["cases"])}
                continue
            result = audit_phase(directory, phase, manifest, python, geometries, oracle_cache)
            report["phases"][name] = result
            event = outcomes[name]
            require(
                result["recorded_counts"] == event["counts"]
                and result["completed_within_budget"] == (event["status"] == "complete"),
                "phase summary differs from launcher disposition",
            )
            if snapshots := result.get("tree_timeout_checkpoints"):
                start, end = map(datetime.fromisoformat, windows[name])
                for checkpoint in snapshots["records"]:
                    require(
                        start <= datetime.fromisoformat(checkpoint["last_modified_utc"]) <= end,
                        "checkpoint modification time falls outside the logged phase",
                    )
                snapshots["logged_phase_window_utc"] = windows[name]
            report["errors"].extend(f"{name}/{error}" for error in result["errors"])
        except (OSError, ValueError, KeyError, IndexError, OverflowError, TypeError) as error:
            report["errors"].append(f"{name}: {error}")
    if not metadata_only and "exp03-small" in report["phases"] and not report["phases"]["exp03-small"]["errors"]:
        try:
            report["small_oracle_coverage"] = audit_small_recovery(run, python, geometries, oracle_cache, report)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            report["errors"].append(f"small oracle coverage: {error}")
    if not metadata_only and not report["errors"] and {"exp03", "exp04", "exp05"} <= set(phases):
        try:
            report["fixed_iteration_repeatability"] = repeatability(run)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["errors"].append(f"fixed-iteration repeatability: {error}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=tuple(PHASES))
    parser.add_argument("--metadata-only", action="store_true", help="skip geometry and result checks")
    parser.add_argument("--report", type=Path, help="write a new JSON report")
    args = parser.parse_args(argv)
    if len(set(args.phases)) != len(args.phases):
        parser.error("phases must be unique")
    report = audit(args.run.resolve(), args.phases, args.metadata_only)
    report["audit_passed"] = not report["errors"]
    content = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    else:
        print(content, end="")
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
