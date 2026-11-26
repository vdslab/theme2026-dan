"""Run and verify the existing-data VisNotes experiment cases."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import re
from statistics import median
import subprocess
import sys
from time import monotonic


THREAD_LIMITS = dict.fromkeys(
    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1"
)
os.environ.update(THREAD_LIMITS)

import numpy
import scipy
import shapely

from experiments.tree_optimization_comparison_01.review_results import (
    file_hash,
    make_geometry,
    read_json,
)
from experiments.visnotes_exact import verify_optimum
from experiments.visnotes_metrics import (
    check_report_metrics,
    objective_value,
    overlap_profile,
    projection_metrics,
    read_layout_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "original": ROOT / "original.json",
    **{
        name: ROOT / "results" / "layouts" / f"original_polygon_{name}.json"
        for name in ("paper", "scalable", "arranged")
    },
}
FIXTURES = {
    name: ROOT / "results" / "experiments" / "tree_optimization_comparison_01" / "small" / name / "input.json"
    for name in ("four-path", "five-cycle", "six-mixed")
}
EXPERIMENT_IDS = {
    "prep": "exp02_clean_benchmarks_prep",
    "objectives": "exp03_objective_hub_pareto",
    "benchmark": "exp04_solver_benchmark",
    "xy": "exp05_xy_z_paradox_study",
}
REG = 1e-5


@dataclass(frozen=True)
class Method:
    solver: str
    method: str = "anneal"
    objective: str = "hybrid"
    alpha: float = 0.25
    beta: float = 0.005
    warm_start: int = 0
    moves: tuple = ("move", "kempe", "swap", "shift")
    optimize_free_edges: bool | None = None

    @property
    def deterministic(self):
        return self.solver in ("solve_layers.py", "solve_layers_dsatur_l1.py") or self.method != "anneal"


COMPACT = Method("solve_layers_compact.py")
OBJECTIVES = {
    "pillar": replace(COMPACT, objective="pillar"),
    "squared": replace(COMPACT, objective="squared"),
    "degree-span": replace(COMPACT, objective="degree-span"),
    **{f"hybrid-{alpha:g}": replace(COMPACT, alpha=alpha) for alpha in (0.1, 0.25, 0.5, 1.0)},
}
CONDITIONS = {
    "prep": {},
    "objectives": {
        name if not offset else f"{name}-plus-{offset}": replace(method, warm_start=offset)
        for name, method in OBJECTIVES.items()
        for offset in (0, 1, 2)
    },
    "benchmark": {
        **{
            name: replace(COMPACT, method=name)
            for name in ("greedy-input", "greedy-size", "greedy-degree", "dsatur")
        },
        "dsatur-free": replace(COMPACT, method="dsatur", optimize_free_edges=True),
        "qp-l2": Method("solve_layers.py", objective="squared"),
        "lp-l1": Method("solve_layers_dsatur_l1.py", objective="weighted-l1"),
        "tree-sa": Method("solve_layers_tree.py", objective="squared"),
        "swap": replace(COMPACT, moves=("swap",)),
        "swap-move": replace(COMPACT, moves=("swap", "move")),
        "swap-move-kempe": replace(COMPACT, moves=("swap", "move", "kempe")),
        "full": COMPACT,
    },
    "xy": {"full": COMPACT},
}


def save_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def root_relative(path):
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def manifest_sources(mode, methods):
    files = [
        Path(__file__).resolve(),
        ROOT / "experiments" / "visnotes_metrics.py",
        ROOT / "experiments" / "visnotes_exact.py",
        ROOT / "experiments" / EXPERIMENT_IDS[mode] / "run_experiment.py",
        ROOT / "experiments" / "tree_optimization_comparison_01" / "review_results.py",
        ROOT / "solve_layers.py",
        ROOT / "solve_layers_dsatur_l1.py",
        *(ROOT / method.solver for method in methods if method.solver not in (
            "solve_layers.py", "solve_layers_dsatur_l1.py",
        )),
    ]
    if any(method.solver in (COMPACT.solver, "solve_layers_tree.py") for method in methods):
        files.extend((ROOT / "solve_layers_tree.py", ROOT / "solve_layers_dummy.py"))
    hashes = {root_relative(path): file_hash(path) for path in files}
    paths = list(hashes)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths], cwd=ROOT, text=True,
    ).splitlines()
    return {
        "git_commit": git_commit,
        "relevant_git_status": git_status,
        "source_sha256": hashes,
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "shapely": shapely.__version__,
            "geos": shapely.geos_version_string,
            "thread_limits": THREAD_LIMITS,
        },
        "command": [sys.executable, "-B", *sys.argv],
        "working_directory": str(Path.cwd()),
    }


def parse_args(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"Run {EXPERIMENT_IDS[mode]} on existing inputs")
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    default_inputs = tuple(INPUTS) if mode in ("prep", "xy") else ("original",)
    allowed_inputs = tuple(INPUTS) if mode in ("prep", "xy") else ("original", *FIXTURES)
    parser.add_argument("--inputs", nargs="+", choices=allowed_inputs, default=default_inputs)
    parser.add_argument("--total-budget", type=float, required=True, help="wall seconds for the run")
    if mode != "prep":
        parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS[mode]), default=tuple(CONDITIONS[mode]))
        parser.add_argument("--seeds", nargs="+", type=int, default=tuple(range(5)))
        parser.add_argument("--max-iterations", type=int, default=20000)
        parser.add_argument("--case-budget", type=float, required=True, help="wall seconds per solver subprocess")
    args = parser.parse_args(argv)
    for name in ("total_budget", "case_budget"):
        if hasattr(args, name) and (not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("inputs", "conditions", "seeds"):
        if hasattr(args, name) and len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"--{name} entries must be unique")
    if mode != "prep" and (min(args.seeds) < 0 or args.max_iterations < 0):
        parser.error("seeds and iteration limit must be nonnegative")
    return args


def solver_command(method, source, directory, seed, iterations, internal_budget):
    output = directory / "layout.json"
    command = [
        sys.executable, "-B", str(ROOT / method.solver), str(source), str(output),
        "--reg", str(REG),
    ]
    if method.solver == "solve_layers_tree.py":
        command += [
            "--report", str(directory / "report.json"),
            "--initial-output", str(directory / "initial.json"),
            "--checkpoint", str(directory / "checkpoint.json"),
            "--seed", str(seed), "--max-iterations", str(iterations), "--time-budget", str(internal_budget),
        ]
    elif method.solver == COMPACT.solver:
        command += [
            "--report", str(directory / "report.json"),
            "--method", method.method, "--objective", method.objective,
            "--alpha", str(method.alpha), "--beta", str(method.beta), "--seed", str(seed or 0),
            "--max-iterations", str(iterations), "--time-budget", str(internal_budget),
            "--warm-start-extra-layers", str(method.warm_start), "--moves", *method.moves,
        ]
        if method.optimize_free_edges:
            command.append("--optimize-free-edges")
    return command


def stage_layout(geometry, levels):
    if len(levels) != geometry.size:
        raise ValueError("saved stage layout has the wrong number of layers")
    return {
        "nodes": geometry.data["nodes"],
        "hyperedges": [dict(edge, layer=level) for edge, level in zip(geometry.data["hyperedges"], levels)],
        "overlaps": geometry.overlaps,
    }


def checked_compact_report(report, geometry, case, method, levels, metrics, deadline, oracle_cache,
                           iterations, internal_budget, subprocess_seconds):
    name = case["condition"]
    check_report_metrics(report["metrics"], metrics, name, check_spans=True)
    settings = report["settings"]
    expected_settings = {
        "method": method.method, "reg": REG, "seed": case["seed"] or 0,
        "objective": method.objective, "alpha": method.alpha, "beta": method.beta,
        "warm_start_extra_layers": method.warm_start, "moves": list(method.moves),
        "max_iterations": iterations, "time_budget": internal_budget, "layer_cap": None,
        "optimize_free_edges": method.optimize_free_edges if method.optimize_free_edges is not None else method.method == "anneal",
    }
    if any(field not in settings or settings[field] != value for field, value in expected_settings.items()):
        raise ValueError(f"{name}: saved solver settings differ from the case manifest")
    if (
        report["input_sha256"] != geometry.hashes["input_sha256"]
        or report["problem_signature"] != geometry.hashes["problem_signature"]
        or report["quadratic_metric"] != "l2_scaled_qp"
        or not 0 <= report["preprocessing_seconds"] <= report["elapsed_seconds"] <= subprocess_seconds
    ):
        raise ValueError(f"{name}: saved input, normalization or elapsed time differs")
    _, initial_metrics = read_layout_metrics(stage_layout(geometry, report["initial_layers"]), geometry, name)
    check_report_metrics(report["initial_metrics"], initial_metrics, name, check_spans=True)
    initial_colors = initial_metrics["used_layer_count"]
    if (
        report["initial_color_count"] != initial_colors
        or report["dsatur_upper_bound"] != (initial_colors if method.method in ("anneal", "dsatur") else None)
    ):
        raise ValueError(f"{name}: initial color count does not match the saved layout")
    witness = report["clique_witness"]
    pairs = set(geometry.overlaps)
    if len(set(witness)) != len(witness) or any(
        type(index) is not int or not 0 <= index < geometry.size for index in witness
    ) or any(pair not in pairs for pair in combinations(sorted(witness), 2)):
        raise ValueError(f"{name}: invalid saved clique witness")
    if report["clique_lower_bound"] != len(witness):
        raise ValueError(f"{name}: saved clique bound disagrees with its witness")

    saved_stages = report["stages"]
    if not saved_stages or saved_stages[-1]["layers"] != list(levels):
        raise ValueError(f"{name}: final saved stage is not the output layout")
    expected_caps = list(range(initial_colors, min(initial_colors + method.warm_start, geometry.size) + 1))
    caps = [stage["cap"] for stage in saved_stages]
    normal_stop = "iteration_budget" if method.method == "anneal" else "initialization_only"
    overall_stop = report["termination_reason"]
    if overall_stop not in (normal_stop, "time_budget") or (
        overall_stop == "time_budget" and (
            internal_budget <= 0 or report["elapsed_seconds"] + 1e-6 < internal_budget
        )
    ):
        raise ValueError(f"{name}: incorrect run termination reason")
    if report["planned_caps"] != expected_caps or caps != expected_caps[:len(caps)] or (
        len(caps) != len(expected_caps) and overall_stop != "time_budget"
    ):
        raise ValueError(f"{name}: incorrect warm-start stage caps")
    upper = min(initial_colors, *(len(set(stage["layers"])) for stage in saved_stages))
    if (
        report["coloring_upper_bound"] != upper
        or report["proven_minimum_layers"] != (upper if upper == len(witness) else None)
    ):
        raise ValueError(f"{name}: coloring bound or minimum-layer claim is not backed by the saved layouts")
    is_fixture = case["input"] in FIXTURES
    stages = []
    iteration_total = 0
    all_oracles_complete = True
    each, remainder = divmod(iterations if method.method == "anneal" else 0, len(expected_caps))
    expected_initial = report["initial_layers"]
    previous_value = objective_value(initial_metrics, method.objective, method.alpha, method.beta)
    previous_elapsed = report["preprocessing_seconds"]
    for index, stage in enumerate(saved_stages):
        cap = stage["cap"]
        if stage["initial_layers"] != expected_initial:
            raise ValueError(f"{name}: cap {cap} was not initialized from the previous layout")
        layout = stage_layout(geometry, stage["layers"])
        _, measured = read_layout_metrics(layout, geometry, name)
        check_report_metrics(stage["metrics"], measured, name, check_spans=True)
        if measured["height_max"] >= cap:
            raise ValueError(f"{name}: saved layout exceeds cap {cap}")
        value = objective_value(measured, method.objective, method.alpha, method.beta)
        if value > previous_value + 1e-6 or not math.isclose(
            stage["optimized_value"], value, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError(f"{name}: stage objective differs from saved layout")
        budget = each + int(index < remainder)
        attempts = stage["iterations"]
        trajectory = stage["trajectory"]
        if stage["iteration_budget"] != budget or type(attempts) is not int or not 0 <= attempts <= budget:
            raise ValueError(f"{name}: stage iterations exceed the allocated budget")
        stopped = stage["termination_reason"]
        if stopped not in (normal_stop, "time_budget") or (
            stopped == normal_stop and attempts != budget
        ) or (
            stopped == "time_budget" and (overall_stop != "time_budget" or index != len(saved_stages) - 1)
        ):
            raise ValueError(f"{name}: incorrect stage termination reason")
        if not trajectory or not 0 <= stage["elapsed_seconds"] <= report["elapsed_seconds"] or any(
            not iteration_total <= point["iteration"] <= iteration_total + attempts
            or not previous_elapsed <= point["elapsed_seconds"] <= report["elapsed_seconds"]
            or not math.isfinite(point["optimized_value"])
            or not 1 <= point["used_layer_count"] <= cap
            for point in trajectory
        ) or any(
            previous["iteration"] > current["iteration"]
            or previous["elapsed_seconds"] > current["elapsed_seconds"]
            or previous["optimized_value"] + 1e-6 < current["optimized_value"]
            for previous, current in zip(trajectory, trajectory[1:])
        ) or trajectory[0]["iteration"] != iteration_total or (
            trajectory[0]["used_layer_count"] != len(set(expected_initial))
        ) or not math.isclose(
            trajectory[0]["optimized_value"], previous_value, rel_tol=1e-9, abs_tol=1e-6
        ) or not math.isclose(
            trajectory[-1]["optimized_value"], value, rel_tol=1e-9, abs_tol=1e-6
        ) or trajectory[-1]["used_layer_count"] != measured["used_layer_count"]:
            raise ValueError(f"{name}: stage trajectory disagrees with the iteration budget or objective")
        iteration_total += attempts
        expected_initial, previous_value = stage["layers"], value
        previous_elapsed = trajectory[-1]["elapsed_seconds"]
        record = {
            "cap": cap, "metrics": measured, "optimized_value": value,
            "trajectory": trajectory, "iterations": attempts,
            "elapsed_seconds": stage["elapsed_seconds"], "termination_reason": stage["termination_reason"],
        }
        if is_fixture and method.method == "anneal":
            key = geometry.hashes["input_sha256"], cap, method.objective, method.alpha, method.beta
            if key not in oracle_cache:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("run budget exhausted before exact verification")
                oracle_cache[key] = verify_optimum(
                    geometry, cap, method.objective, method.alpha, method.beta, remaining,
                )
            exact = oracle_cache[key]
            gap = value - exact["enumeration"]["value"]
            if gap < -1e-6:
                raise ValueError(f"{name}: stage objective is below the exhaustive optimum")
            record["oracle"] = exact
            record["optimality_gap"] = max(0.0, gap)
            all_oracles_complete = all_oracles_complete and exact["milp"]["status"] == 0
        stages.append(record)
    if iteration_total != report["iterations"] or iteration_total > iterations:
        raise ValueError(f"{name}: incorrect total annealing iterations")
    if not math.isclose(report["optimized_value"], previous_value, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{name}: final objective disagrees with the saved layout")
    return stages, len(caps) == len(expected_caps), all_oracles_complete


def run_case(case, args, mode, method, source, geometry, output, deadline, oracle_cache):
    name, seed = case["condition"], case["seed"]
    tag = "deterministic" if seed is None else f"seed-{seed}"
    directory = output / case["input"] / name / tag
    remaining = deadline - monotonic()
    if remaining < args.case_budget:
        return dict(case, status="pending", termination_reason="total_budget")
    directory.mkdir(parents=True, exist_ok=False)
    case_wall = args.case_budget
    # The earlier in-process limit allows the heuristic CLI to flush a valid
    # layout and checkpoint under the common external subprocess limit.
    internal_budget = case_wall - max(min(2.0, case_wall * 0.25), case_wall * 0.08)
    command = solver_command(method, source, directory, seed, args.max_iterations, internal_budget)
    record = dict(
        case, command=command, case_directory=root_relative(directory),
        input_sha256=geometry.hashes["input_sha256"],
        subprocess_wall_budget=case_wall,
        solver_internal_budget=internal_budget if method.solver in (COMPACT.solver, "solve_layers_tree.py") else None,
    )
    start = monotonic()
    all_stages_verified = all_oracles_complete = True
    try:
        log_path = directory / "run.log"
        with log_path.open("wb") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=case_wall, check=False,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        record["solver_wall_seconds"] = monotonic() - start
        record["exit_code"] = result.returncode
        if result.returncode != 0:
            raise RuntimeError(f"solver exited with code {result.returncode}")
        if file_hash(source) != geometry.hashes["input_sha256"]:
            raise ValueError("input changed during case")
        layout = read_json(directory / "layout.json")
        continuous = method.solver in ("solve_layers.py", "solve_layers_dsatur_l1.py")
        levels, metrics = read_layout_metrics(layout, geometry, name, continuous)
        record.update(metrics=metrics, height_type="continuous" if continuous else "integer")

        if method.solver == COMPACT.solver:
            report = read_json(directory / "report.json")
            record["stages"], all_stages_verified, all_oracles_complete = checked_compact_report(
                report, geometry, case, method, levels, metrics, deadline, oracle_cache,
                args.max_iterations, internal_budget, record["solver_wall_seconds"],
            )
            record.update(iterations=report["iterations"], termination_reason=report["termination_reason"])
        elif method.solver == "solve_layers_tree.py":
            report = read_json(directory / "report.json")
            check_report_metrics(report["best_metrics"], metrics, name)
            settings = report["settings"]
            attempts = report["summary"]["attempts"]
            if (
                settings["method"] != "anneal" or settings["reg"] != REG or settings["seed"] != seed
                or settings["max_iterations"] != args.max_iterations or settings["time_budget"] != internal_budget
                or not 0 <= attempts <= args.max_iterations or report["best"]["layers"] != list(levels)
                or report["provenance"]["input_sha256"] != geometry.hashes["input_sha256"]
                or not 0 <= report["summary"]["total_seconds"] <= record["solver_wall_seconds"]
            ):
                raise ValueError("tree solver report differs from the saved layout or case settings")
            for filename, expected in (("initial.json", report["initial"]), ("checkpoint.json", report["best"])):
                saved_levels, saved_metrics = read_layout_metrics(read_json(directory / filename), geometry, filename)
                if list(saved_levels) != expected["layers"]:
                    raise ValueError(f"{filename}: saved layout differs from the tree report")
                check_report_metrics(expected["metrics"], saved_metrics, filename)
            record.update(
                iterations=attempts, termination_reason=report["summary"]["stop_reason"],
                trajectory=report["history"],
            )
        elif method.solver == "solve_layers_dsatur_l1.py":
            check_report_metrics(layout["metrics"], metrics, name)
            if not layout["solver"]["success"] or layout["regularization"] != REG or not math.isclose(
                layout["objective_value"], metrics["l1_regularized"], rel_tol=1e-9, abs_tol=1e-6
            ):
                raise ValueError("LP report did not verify against regularized weighted L1")
            record.update(iterations=layout["solver"]["iterations"], termination_reason="converged")
        else:
            log_text = log_path.read_text(encoding="utf-8")
            convergence = re.search(r"Optimization converged successfully in (\d+) iterations", log_text)
            objective = re.search(r"Optimal objective value: ([-+0-9.eE]+)", log_text)
            if convergence is None or objective is None or not math.isclose(
                float(objective[1]), metrics["l2_scaled_qp"], rel_tol=1e-9, abs_tol=1e-6
            ):
                raise ValueError("QP report did not verify against the normalized quadratic metric")
            record.update(iterations=int(convergence[1]), termination_reason="converged")
        if mode == "benchmark":
            record["projection"] = projection_metrics(geometry, levels)
        if all_stages_verified and all_oracles_complete:
            record["status"] = "verified"
        else:
            record["status"] = "partial"
            record["incomplete"] = "solver_stages" if not all_stages_verified else "milp_time_limit"
    except subprocess.TimeoutExpired:
        record.update(status="timeout", termination_reason="case_wall_budget", solver_wall_seconds=monotonic() - start)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        record.update(status="failed", termination_reason="case_error", error=str(exc))
    record["case_elapsed_seconds"] = monotonic() - start
    save_json(directory / "case.json", record)
    print(f"{case['input']}/{name}/{tag}: {record['status']}", flush=True)
    return record


def input_record(geometry, source):
    arities = Counter(len(set(edge["nodes"])) for edge in geometry.data["hyperedges"])
    return {
        "file": root_relative(source),
        **geometry.hashes,
        "nodes": len(geometry.data["nodes"]),
        "hyperedges": geometry.size,
        "hyperedges_by_distinct_member_count": dict(sorted(arities.items())),
        "hyperedges_with_duplicate_memberships": sum(
            len(edge["nodes"]) != len(set(edge["nodes"])) for edge in geometry.data["hyperedges"]
        ),
        "overlap_profile": overlap_profile(geometry),
    }


def xy_variant_structure(reference, data):
    node_ids = [node["id"] for node in data["nodes"]]
    if node_ids != [node["id"] for node in reference["nodes"]]:
        raise ValueError("XY variant differs from the reference vertex IDs or order")
    edges = data["hyperedges"]
    original_edges = reference["hyperedges"]
    if len(edges) != len(original_edges):
        raise ValueError("XY variant differs from the reference edge count")
    changed_order = 0
    for old, new in zip(original_edges, edges):
        if (
            {k: v for k, v in old.items() if k != "nodes"} != {k: v for k, v in new.items() if k != "nodes"}
            or Counter(old["nodes"]) != Counter(new["nodes"])
        ):
            raise ValueError(f"XY variant differs in metadata or membership of edge {old['id']}")
        changed_order += int(old["nodes"] != new["nodes"])
    return changed_order


def summarize(records):
    comparisons = defaultdict(list)
    for record in records:
        if record["status"] == "verified":
            comparisons[(record["input"], record["condition"])].append(record)
    rows = []
    for (input_name, condition), group in comparisons.items():
        keys = (
            "used_layer_count", "distinct_height_values", "height_range", "l2_scaled_qp", "pillar_length",
            "top5_constrained_hub_mean_span", "overlap_violation_count",
        )
        samples = {
            key: [row["metrics"][key] for row in group if row["metrics"].get(key) is not None] for key in keys
        }
        samples["solver_wall_seconds"] = [row["solver_wall_seconds"] for row in group]
        if "projection" in group[0]:
            for key in ("overdraw_pixel_count", "projected_pillar_length_px"):
                samples[key] = [row["projection"][key] for row in group]
        rows.append({
            "input": input_name, "condition": condition, "verified_cases": len(group),
            "metrics": {
                key: {"median": median(sample), "minimum": min(sample), "maximum": max(sample)}
                for key, sample in samples.items() if sample
            },
        })
    return rows


def cap_frontiers(records, methods):
    """Report observed, nondominated layer/pillar trade-offs per objective and seed."""
    groups = defaultdict(list)
    for case in records:
        if case["status"] != "verified":
            continue
        method = methods[case["condition"]]
        key = case["input"], case["seed"], method.objective, method.alpha, method.beta
        for stage in case["stages"]:
            metrics = stage["metrics"]
            groups[key].append({
                "condition": case["condition"], "cap": stage["cap"],
                "used_layer_count": metrics["used_layer_count"], "pillar_length": metrics["pillar_length"],
                "l2_scaled_qp": metrics["l2_scaled_qp"],
                "top5_constrained_hub_mean_span": metrics["top5_constrained_hub_mean_span"],
            })
    frontiers = []
    for (input_name, seed, objective, alpha, beta), group in groups.items():
        def point(row):
            return row["used_layer_count"], row["pillar_length"]

        unique = set(map(point, group))
        nondominated = {
            (k, b) for k, b in unique
            if not any(other_k <= k and other_b <= b and (other_k, other_b) != (k, b) for other_k, other_b in unique)
        }
        frontiers.append({
            "input": input_name, "seed": seed, "objective": objective, "alpha": alpha, "beta": beta,
            "observed_frontier": sorted((row for row in group if point(row) in nondominated), key=point),
        })
    return frontiers


def main(mode, argv=None):
    args = parse_args(mode, argv)
    started = monotonic()
    deadline = started + args.total_budget
    output = args.output.resolve()
    inputs = {name: (INPUTS | FIXTURES)[name] for name in args.inputs}
    methods = {name: CONDITIONS[mode][name] for name in getattr(args, "conditions", ())}
    cases = [
        {"input": input_name, "condition": condition, "seed": seed}
        for input_name in inputs
        for condition, method in methods.items()
        for seed in ((None,) if method.deterministic else args.seeds)
    ]
    manifest = {
        "experiment_id": EXPERIMENT_IDS[mode],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_article": "https://vdslab.esa.io/posts/9193",
        "regularization": REG,
        "input_files": {
            name: {"file": root_relative(path), "sha256": file_hash(path)} for name, path in inputs.items()
        },
        "conditions": {name: asdict(method) for name, method in methods.items()},
        "case_budget_seconds": getattr(args, "case_budget", None),
        "total_budget_seconds": args.total_budget,
        "trial_limit": getattr(args, "max_iterations", None),
        "cases": cases,
        "provenance": manifest_sources(mode, methods.values()),
    }
    if mode == "xy":
        manifest["structure_reference"] = {
            "file": root_relative(INPUTS["original"]), "sha256": file_hash(INPUTS["original"]),
        }
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", manifest)
    records, profiles, oracle_cache = [], {}, {}
    geometries = {}
    interrupted = False
    error = None
    try:
        reference = read_json(INPUTS["original"]) if mode == "xy" else None
        for input_name, source in inputs.items():
            if monotonic() >= deadline:
                break
            geometry = make_geometry(source, REG)
            if geometry.hashes["input_sha256"] != manifest["input_files"][input_name]["sha256"]:
                raise ValueError(f"{input_name}: input changed during preflight")
            if mode in ("prep", "xy"):
                profiles[input_name] = input_record(geometry, source)
            if reference is not None:
                profiles[input_name]["changed_member_orders_vs_original"] = xy_variant_structure(reference, geometry.data)
            geometries[input_name] = geometry
        for case in cases:
            input_name = case["input"]
            if input_name not in geometries:
                records.append(dict(case, status="pending", termination_reason="total_budget"))
            else:
                records.append(run_case(
                    case, args, mode, methods[case["condition"]], inputs[input_name],
                    geometries[input_name], output, deadline, oracle_cache,
                ))
    except KeyboardInterrupt:
        for case in cases[len(records):]:
            records.append(dict(case, status="pending", termination_reason="interrupted"))
        interrupted = True
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        error = str(exc)
        for case in cases[len(records):]:
            records.append(dict(case, status="pending", termination_reason="run_error"))
    if any(
        file_hash(ROOT / path) != digest for path, digest in manifest["provenance"]["source_sha256"].items()
    ) or (mode == "xy" and file_hash(INPUTS["original"]) != manifest["structure_reference"]["sha256"]):
        error = "source code or structure reference changed during run"
    counts = dict(Counter(row["status"] for row in records))
    completed = (
        not interrupted and error is None and len(geometries) == len(inputs)
        and counts.get("verified", 0) == len(cases) and monotonic() <= deadline
    )
    summary = {
        "complete": completed,
        "interrupted": interrupted,
        "error": error,
        "total_budget_exhausted": monotonic() > deadline or any(
            row["termination_reason"] == "total_budget" for row in records
        ),
        "elapsed_seconds": monotonic() - started,
        "input_profiles": profiles,
        "case_counts": counts,
        "cases": records,
        "comparisons": summarize(records) if error is None else [],
    }
    if mode == "objectives":
        summary["observed_layer_pillar_frontiers"] = cap_frontiers(records, methods) if error is None else []
    save_json(output / "summary.json", summary)
    return 0 if completed else 130 if interrupted else 1
