#!/usr/bin/env python3
"""Search overlap orders using a symbolic, capacity-one dummy layer.

DSatur + QP supplies the baseline. Legal moves/swaps change discrete layer
membership; each completed assignment supplies NEW directed overlap constraints
to a continuous solver. The dummy is a combinatorial buffer, not a physical
collision-free animation or an additional optimization variable.

Examples:
    python solve_layers_dummy.py
    python solve_layers_dummy.py original.json layered_hypergraph_dummy.json
    python solve_layers_dummy.py --max-blockers 3 --max-solver-calls-total 30
    python solve_layers_dummy.py --max-batches 0  # baseline only

The default objective is exactly solve_layers.py's scaled squared objective.
Optional --objective l1 uses the same full regularized weights for baseline and
search. Equal-valued assignments are not adopted. Search budgets do not include
the initial solve; an active solve may exceed the soft time budget.
"""

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize

from solve_layers import dsatur_coloring
from solve_layers_dsatur_l1 import (
    build_lp, compute_weights, evaluate_layout, extract_overlaps, orient_overlaps,
)


@dataclass(frozen=True)
class Config:
    reg: float = 1e-5
    objective: str = "l2"
    max_candidates: int = 100
    max_attempts: int = 10000
    max_solver_calls_per_batch: int = 3
    max_solver_calls_total: int = 15
    max_batches: int = 5
    time_budget: float = 1800.0
    max_blockers: int = 0
    constraint_tol: float = 1e-5
    objective_rtol: float = 1e-7
    objective_atol: float = 1e-6
    improvement_abs_tol: float = 1e-3
    improvement_rel_tol: float = 1e-6
    maxiter: int = 1000
    ftol: float = 1e-6
    strict_validation: bool = False
    solver_output: bool = False

    def validate(self):
        if self.objective not in ("l2", "l1"):
            raise ValueError("objective must be l2 or l1")
        for name in ("reg", "time_budget", "constraint_tol", "objective_rtol",
                     "objective_atol", "improvement_abs_tol", "improvement_rel_tol", "ftol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("max_candidates", "max_attempts", "max_solver_calls_total", "max_batches"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_solver_calls_per_batch < 1 or self.maxiter < 1:
            raise ValueError("per-batch solver calls and maxiter must be positive")
        if self.max_blockers != 0 and self.max_blockers < 2:
            raise ValueError("max_blockers must be 0 (disabled) or at least 2")


@dataclass(frozen=True)
class Problem:
    data: dict
    n: int
    overlaps: tuple
    adjacency: tuple
    weights: np.ndarray
    regularized: np.ndarray
    laplacian: np.ndarray
    scale: float
    reg: float
    signature: str


@dataclass(frozen=True)
class Solution:
    z: np.ndarray
    metrics: dict
    solver_info: dict


@dataclass(frozen=True)
class LayoutState:
    state_id: str
    colors: tuple
    layer_members: tuple
    directed: np.ndarray
    z_seed: np.ndarray
    z: np.ndarray
    metrics: dict
    objective_config: dict
    solver_info: dict
    problem_signature: str


@dataclass(frozen=True)
class Candidate:
    base_state_id: str
    generation_index: int
    move_type: str
    changes: tuple
    operations: tuple
    changed_polygons: tuple
    colors_signature: tuple
    direction_signature: bytes
    changes_overlap_order: bool
    status: str = "generated"
    z: object = None
    metrics: object = None


@dataclass(frozen=True)
class ValidatedCandidate:
    original: Candidate
    colors: tuple
    layer_members: tuple
    directed: np.ndarray
    direction_signature: bytes
    changes_overlap_order: bool
    status: str = "validated"

    @property
    def changes(self):
        return self.original.changes

    @property
    def operations(self):
        return self.original.operations


@dataclass(frozen=True)
class ValidationResult:
    status: str
    candidate: object = None
    reason: str = ""


@dataclass(frozen=True)
class SearchResult:
    initial: LayoutState
    best: LayoutState
    history: list
    summary: dict
    candidate_logs: list
    generation_logs: list


class OptimizationError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def readonly(values, dtype=float):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def prepare_problem(data, reg=1e-5, overlaps=None):
    """Cache geometry and weights; an explicit C is useful for small experiments."""
    if not math.isfinite(reg) or reg < 0:
        raise ValueError("reg must be finite and nonnegative")
    data = copy.deepcopy(data)
    nodes, hyperedges = data["nodes"], data["hyperedges"]
    n = len(hyperedges)
    if n < 1:
        raise ValueError("At least one hyperedge is required")
    for node in nodes:
        if not all(math.isfinite(float(node[axis])) for axis in ("x", "y")):
            raise ValueError("Node coordinates must be finite")
    raw_pairs = extract_overlaps(nodes, hyperedges) if overlaps is None else overlaps
    pairs = set()
    for left, right in raw_pairs:
        if not all(isinstance(i, (int, np.integer)) and 0 <= i < n for i in (left, right)):
            raise ValueError("Invalid overlap endpoint")
        if left == right:
            raise ValueError("Self-overlap is not allowed")
        pairs.add(tuple(sorted((int(left), int(right)))))
    pairs = tuple(sorted(pairs))
    adjacency = [set() for _ in range(n)]
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    weights, regularized, scale = compute_weights(hyperedges, reg)
    laplacian = np.diag(regularized.sum(axis=1)) - regularized
    signature = hashlib.sha256(json.dumps(
        {"input": data, "overlaps": pairs, "reg": reg},
        sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return Problem(data, n, pairs, tuple(frozenset(a) for a in adjacency),
                   readonly(weights), readonly(regularized), readonly(laplacian),
                   scale, reg, signature)


def make_feasible_seed(colors):
    colors = np.asarray(colors, dtype=float)
    return colors - colors[0]


def layer_members(colors, count):
    return tuple(tuple(i for i, color in enumerate(colors) if color == layer)
                 for layer in range(count))


def check_assignment(problem, colors, members):
    k = len(members)
    if len(colors) != problem.n or k < 1:
        raise ValueError("Invalid assignment size")
    if any(type(color) is not int or not 0 <= color < k for color in colors):
        raise ValueError("An ordinary-layer index is invalid (or dummy remains)")
    if sorted(i for group in members for i in group) != list(range(problem.n)):
        raise ValueError("Missing or duplicate polygon membership")
    if members != layer_members(colors, k):
        raise ValueError("colors and layer_members disagree")
    if any(colors[i] == colors[j] for i, j in problem.overlaps):
        raise ValueError("Overlapping polygons occupy the same layer")


def build_directed_constraints(problem, colors):
    directed = orient_overlaps(problem.n, problem.overlaps, colors)
    a_y = np.zeros((len(directed), problem.n - 1))
    for row, (lower, upper) in enumerate(directed):
        if lower > 0:
            a_y[row, lower - 1] = -1.0
        if upper > 0:
            a_y[row, upper - 1] = 1.0
    return directed, a_y


def direction_signature(problem, colors):
    return bytes(int(colors[i] < colors[j]) for i, j in problem.overlaps)


def numerical_config(problem, config):
    return {"objective": config.objective, "reg": config.reg, "gap": 1.0,
            "anchor_index": 0, "anchor_height": 0.0, "scale": problem.scale,
            "maxiter": config.maxiter, "ftol": config.ftol,
            "constraint_tol": config.constraint_tol,
            "objective_rtol": config.objective_rtol,
            "objective_atol": config.objective_atol,
            "solver": "SLSQP" if config.objective == "l2" else "HiGHS"}


def get_primary_objective(metrics, config):
    return float(metrics["l2_scaled_qp" if config.objective == "l2" else "l1_regularized"])


def checked_metrics(problem, z, directed, config, reported_objective=None):
    z = np.asarray(z, dtype=float)
    if z.shape != (problem.n,) or not np.all(np.isfinite(z)):
        raise OptimizationError("invalid_numeric", "Nonfinite heights or incorrect shape")
    if abs(z[0]) > config.constraint_tol:
        raise OptimizationError("constraint_violation", "Anchor is not zero")
    gaps = z[directed[:, 1]] - z[directed[:, 0]]
    if np.any(gaps < 1.0 - config.constraint_tol):
        raise OptimizationError("constraint_violation", "Signed overlap constraint failed")
    metrics = evaluate_layout(z, problem.data["nodes"], problem.data["hyperedges"],
                              problem.weights, problem.regularized, problem.scale, directed)
    metrics["order_violation_count"] = int(np.count_nonzero(gaps < 1 - config.constraint_tol))
    metrics["overlap_violation_count"] = int(np.count_nonzero(np.abs(gaps) < 1 - config.constraint_tol))
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise OptimizationError("invalid_numeric", "Nonfinite metric")
    objective = get_primary_objective(metrics, config)
    if reported_objective is not None and (not math.isfinite(float(reported_objective)) or
            not math.isclose(objective, float(reported_objective),
                             rel_tol=config.objective_rtol, abs_tol=config.objective_atol)):
        raise OptimizationError("objective_mismatch", "Recomputed objective differs from solver result")
    return metrics


def solve_heights_qp(problem, colors, config):
    """Solve with a fresh signed C. Same-colored polygons remain independent."""
    config.validate()
    if config.reg != problem.reg:
        raise ValueError("Config reg differs from prepared problem")
    check_assignment(problem, tuple(colors), layer_members(colors, max(colors) + 1))
    directed, a_y = build_directed_constraints(problem, colors)
    seed = make_feasible_seed(colors)
    start = time.perf_counter()
    if problem.n == 1:
        z, objective = np.zeros(1), 0.0
        info = {"success": True, "message": "Single anchored polygon", "iterations": 0}
    elif config.objective == "l1":
        c, a_ub, b_ub, bounds = build_lp(problem.regularized, directed)
        try:
            result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs",
                             options={"maxiter": config.maxiter, "disp": config.solver_output})
        except Exception as exc:
            raise OptimizationError("solver_failed", f"HiGHS exception: {exc}") from exc
        if not result.success:
            raise OptimizationError("solver_failed", str(result.message))
        z, objective = result.x[:problem.n], result.fun
        info = {"success": True, "message": str(result.message), "iterations": int(result.nit)}
    else:
        l_yy = problem.laplacian[1:, 1:]
        scale = problem.scale
        def objective_function(y):
            return float(2.0 * scale * (y @ l_yy @ y))
        def gradient(y):
            return 4.0 * scale * (l_yy @ y)
        constraints = ([LinearConstraint(a_y, np.ones(len(directed)), np.full(len(directed), np.inf))]
                       if len(directed) else [])
        try:
            result = minimize(objective_function, seed[1:].copy(), jac=gradient,
                              constraints=constraints, method="SLSQP",
                              options={"maxiter": config.maxiter, "ftol": config.ftol,
                                       "disp": config.solver_output})
        except Exception as exc:
            raise OptimizationError("solver_failed", f"SLSQP exception: {exc}") from exc
        if not result.success:
            raise OptimizationError("solver_failed", str(result.message))
        z, objective = np.concatenate(([0.0], result.x)), result.fun
        info = {"success": True, "message": str(result.message), "iterations": int(result.nit)}
    metrics = checked_metrics(problem, z, directed, config, objective)
    info.update({"elapsed_seconds": time.perf_counter() - start,
                 "reported_objective": float(objective), "status": "optimized"})
    return Solution(readonly(z), metrics, info)


def assemble_state(problem, colors, k, solution, config, state_id):
    colors = tuple(int(c) for c in colors)
    members = layer_members(colors, k)
    check_assignment(problem, colors, members)
    directed, _ = build_directed_constraints(problem, colors)
    metrics = checked_metrics(problem, solution.z, directed, config,
                              get_primary_objective(solution.metrics, config))
    return LayoutState(state_id, colors, members, readonly(directed, int),
                       readonly(make_feasible_seed(colors)), readonly(solution.z),
                       metrics, numerical_config(problem, config), copy.deepcopy(solution.solver_info),
                       problem.signature)


def create_initial_state(problem, config):
    config.validate()
    coloring = dsatur_coloring(problem.n, problem.overlaps)
    colors = tuple(int(coloring[i]) for i in range(problem.n))
    solution = solve_heights_qp(problem, colors, config)
    return assemble_state(problem, colors, max(colors) + 1, solution, config, "state-000000")


def clone_state(state):
    return LayoutState(state.state_id, state.colors, state.layer_members,
                       readonly(state.directed, int), readonly(state.z_seed), readonly(state.z),
                       copy.deepcopy(state.metrics), copy.deepcopy(state.objective_config),
                       copy.deepcopy(state.solver_info), state.problem_signature)


def candidate_specs(problem, state, kind, config):
    """Yield raw finite neighbourhood specifications, including illegal ones."""
    colors, groups = state.colors, state.layer_members
    k = len(groups)
    if kind == "swap":
        for a in range(problem.n):
            for b in range(a + 1, problem.n):
                if colors[a] != colors[b]:
                    yield a, colors[b], (b,)
    else:
        for a in range(problem.n):
            for target in range(k):
                if target == colors[a]:
                    continue
                blockers = tuple(sorted(problem.adjacency[a].intersection(groups[target])))
                if kind == "move":
                    yield a, target, ()
                elif 2 <= len(blockers) <= config.max_blockers:
                    yield a, target, blockers
                else:
                    # Counts the inspected target against the generation budget.
                    yield a, target, None


def candidate_from_spec(problem, state, kind, spec, generation_index):
    a, target, others = spec
    if others is None:
        return None
    source = state.colors[a]
    dummy = len(state.layer_members)
    groups = [set(group) for group in state.layer_members] + [set()]
    operations = ((a, source, dummy),) + tuple((b, target, source) for b in others) + ((a, dummy, target),)
    for polygon, before, after in operations:
        if polygon not in groups[before] or (after == dummy and groups[dummy]):
            return None
        if problem.adjacency[polygon].intersection(groups[after]):
            return None
        groups[before].remove(polygon)
        groups[after].add(polygon)
    changes = tuple(sorted(((a, source, target),) + tuple((b, target, source) for b in others)))
    colors = list(state.colors)
    for polygon, _, after in changes:
        colors[polygon] = after
    signature = direction_signature(problem, colors)
    return Candidate(state.state_id, generation_index, kind, changes, operations,
                     tuple(p for p, _, _ in changes), tuple(colors), signature,
                     signature != direction_signature(problem, state.colors))


def distributed_limits(total, kinds):
    return {kind: total // len(kinds) + int(index < total % len(kinds))
            for index, kind in enumerate(kinds)}


def generate_candidates(problem, state, config):
    """Bound attempts and unique candidates per kind; retain empty ordinary layers."""
    kinds = ["move", "swap"] + (["blocker_exchange"] if config.max_blockers else [])
    limits = distributed_limits(config.max_candidates, kinds)
    attempts = distributed_limits(config.max_attempts, kinds)
    iterators = {kind: iter(candidate_specs(problem, state, kind, config)) for kind in kinds}
    stats = {kind: {"attempts": 0, "generated": 0, "illegal": 0, "duplicate": 0,
                    "exhausted": False} for kind in kinds}
    candidates, seen = [], {state.colors}
    active = set(kinds)
    generation_index = 0
    while active:
        for kind in kinds:
            if kind not in active:
                continue
            counts = stats[kind]
            if counts["attempts"] >= attempts[kind] or counts["generated"] >= limits[kind]:
                active.remove(kind)
                continue
            try:
                spec = next(iterators[kind])
            except StopIteration:
                counts["exhausted"] = True
                active.remove(kind)
                continue
            counts["attempts"] += 1
            generation_index += 1
            candidate = candidate_from_spec(problem, state, kind, spec, generation_index)
            if candidate is None:
                counts["illegal"] += 1
            elif candidate.colors_signature in seen:
                counts["duplicate"] += 1
            else:
                seen.add(candidate.colors_signature)
                candidates.append(candidate)
                counts["generated"] += 1
    return candidates, {"base_state_id": state.state_id, "by_type": stats,
                        "generated": len(candidates),
                        "attempts": sum(s["attempts"] for s in stats.values()),
                        "truncated": any(not s["exhausted"] for s in stats.values())}


def validate_candidate(problem, state, candidate, config):
    """Independently reconstruct AND replay the buffer operations before solving."""
    if candidate.base_state_id != state.state_id:
        return ValidationResult("stale", reason="Candidate belongs to a different state")
    try:
        if candidate.status != "generated" or candidate.z is not None or candidate.metrics is not None:
            raise ValueError("Expected an unevaluated generated candidate")
        if candidate.move_type not in ("move", "swap", "blocker_exchange"):
            raise ValueError("Unknown move type")
        k = len(state.layer_members)
        colors = list(state.colors)
        changed = []
        for polygon, before, after in candidate.changes:
            if any(type(value) is not int for value in (polygon, before, after)):
                raise ValueError("Change fields must be integers")
            if not 0 <= polygon < problem.n or polygon in changed:
                raise ValueError("Invalid or duplicate changed polygon")
            if before != state.colors[polygon] or not 0 <= after < k or before == after:
                raise ValueError("Change source/target is invalid")
            changed.append(polygon)
            colors[polygon] = after
        if tuple(sorted(changed)) != tuple(candidate.changed_polygons):
            raise ValueError("changed_polygons disagrees with changes")
        if not candidate.operations:
            raise ValueError("Missing operation trace")
        a, source, dummy = candidate.operations[0]
        if type(a) is not int or not 0 <= a < problem.n or dummy != k or source != state.colors[a]:
            raise ValueError("First operation must move a polygon into the dummy")
        change_by_id = {p: (before, after) for p, before, after in candidate.changes}
        if a not in change_by_id:
            raise ValueError("Dummy polygon missing from changes")
        target = change_by_id[a][1]
        others = tuple(sorted(p for p in changed if p != a))
        expected_count = {"move": 0, "swap": 1}.get(candidate.move_type)
        if expected_count is not None and len(others) != expected_count:
            raise ValueError("Change count does not match move type")
        if candidate.move_type == "blocker_exchange":
            blockers = tuple(sorted(problem.adjacency[a].intersection(state.layer_members[target])))
            if not 2 <= len(others) <= config.max_blockers or others != blockers:
                raise ValueError("Blocker exchange must move every blocker within the configured limit")
        if any(change_by_id[b] != (target, source) for b in others):
            raise ValueError("Exchange changes must reverse source and target")
        expected = ((a, source, k),) + tuple((b, target, source) for b in others) + ((a, k, target),)
        if tuple(candidate.operations) != expected:
            raise ValueError("Operation trace does not match the declared move")
        groups = [set(group) for group in state.layer_members] + [set()]
        actual = list(state.colors)
        for polygon, before, after in candidate.operations:
            if any(type(value) is not int for value in (polygon, before, after)):
                raise ValueError("Operation fields must be integers")
            if actual[polygon] != before or polygon not in groups[before]:
                raise ValueError("Operation source is incorrect")
            if after == k and groups[k]:
                raise ValueError("Dummy capacity exceeded")
            if problem.adjacency[polygon].intersection(groups[after]):
                return ValidationResult("invalid_assignment", reason="Intermediate endpoint has a same-layer overlap")
            groups[before].remove(polygon)
            groups[after].add(polygon)
            actual[polygon] = after
        if groups[k] or actual != colors:
            raise ValueError("Dummy remains occupied or operation result differs from changes")
        colors = tuple(colors)
        members = tuple(tuple(sorted(group)) for group in groups[:k])
        try:
            check_assignment(problem, colors, members)
        except ValueError as exc:
            return ValidationResult("invalid_assignment", reason=str(exc))
        signature = direction_signature(problem, colors)
        changed_order = signature != direction_signature(problem, state.colors)
        if (candidate.colors_signature != colors or candidate.direction_signature != signature or
                candidate.changes_overlap_order != changed_order):
            raise ValueError("Declared signatures or order-change flag disagree with reconstruction")
        if colors == state.colors:
            return ValidationResult("skipped_unchanged", reason="Assignment is unchanged")
        directed, _ = build_directed_constraints(problem, colors)
        return ValidationResult("validated", ValidatedCandidate(
            candidate, colors, members, readonly(directed, int), signature, changed_order))
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        return ValidationResult("malformed_candidate", reason=str(exc))


def is_significant_improvement(current_value, candidate_value, config):
    if not math.isfinite(current_value) or not math.isfinite(candidate_value):
        return False
    threshold = max(config.improvement_abs_tol,
                    config.improvement_rel_tol * max(1.0, abs(current_value), abs(candidate_value)))
    return current_value - candidate_value > threshold


def prioritize_validated_candidates(problem, candidates, config):
    """Seed scores are feasible upper bounds, used ONLY for ordering, not pruning."""
    groups = {}
    scores = {}
    for candidate in candidates:
        seed = make_feasible_seed(candidate.colors)
        if config.objective == "l2":
            score = float(2 * problem.scale * (seed @ problem.laplacian @ seed))
        else:
            i, j = np.triu_indices(problem.n, 1)
            score = float(np.dot(problem.regularized[i, j], np.abs(seed[i] - seed[j])))
        scores[candidate.original.generation_index] = score
        groups.setdefault(candidate.original.move_type, []).append(candidate)
    for group in groups.values():
        group.sort(key=lambda c: (scores[c.original.generation_index], c.original.generation_index))
    ordered = []
    position = 0
    while any(position < len(group) for group in groups.values()):
        for group in groups.values():
            if position < len(group):
                ordered.append(group[position])
        position += 1
    return ordered, scores


def cache_key(problem, config, signature):
    settings = json.dumps(numerical_config(problem, config), sort_keys=True, allow_nan=False)
    return problem.signature, settings, signature


def candidate_log(candidate):
    return {"base_state_id": candidate.base_state_id, "generation_index": candidate.generation_index,
            "move_type": candidate.move_type, "changes": candidate.changes,
            "operations": candidate.operations, "changed_polygons": candidate.changed_polygons,
            "direction_signature": candidate.direction_signature.hex(),
            "changes_overlap_order": candidate.changes_overlap_order,
            "status": "generated", "accepted": False}


def run_dummy_layer_search(problem, config, initial_state=None, checkpoint_callback=None,
                           progress_callback=None):
    config.validate()
    initial_start = time.perf_counter()
    initial = clone_state(initial_state) if initial_state is not None else create_initial_state(problem, config)
    if initial.problem_signature != problem.signature or initial.objective_config != numerical_config(problem, config):
        raise ValueError("Initial state belongs to a different problem or solver configuration")
    check_assignment(problem, initial.colors, initial.layer_members)
    expected_directed, _ = build_directed_constraints(problem, initial.colors)
    if not np.array_equal(initial.directed, expected_directed):
        raise ValueError("Initial directions disagree with assignment")
    checked_metrics(problem, initial.z, expected_directed, config, get_primary_objective(initial.metrics, config))
    initial_seconds = time.perf_counter() - initial_start
    current, best = clone_state(initial), clone_state(initial)
    cache = {cache_key(problem, config, direction_signature(problem, initial.colors)):
             Solution(readonly(initial.z), copy.deepcopy(initial.metrics), copy.deepcopy(initial.solver_info))}
    failed_keys = set()
    history, logs, generation_logs = [], [], []
    summary = {"solver_calls": 0, "cache_hits": 0, "failure_cache_hits": 0,
               "solver_failures": 0, "validation_failures": 0, "evaluated": 0,
               "batches": 0, "adoptions": 0, "initial_seconds": initial_seconds,
               "timing_seconds": {"generation": 0.0, "validation": 0.0, "priority": 0.0, "solving": 0.0}}
    if checkpoint_callback:
        checkpoint_callback(initial, best, history, {**summary, "stop_reason": "initial_checkpoint"})
    start = time.perf_counter()
    queue, cursor, queue_state_id = [], 0, None
    log_by_index = {}
    stop_reason = "max_batches_reached"
    error_detail = None
    active_log = None
    try:
        while True:
            if time.perf_counter() - start >= config.time_budget:
                stop_reason = "time_budget_exhausted"
                break
            if summary["batches"] >= config.max_batches:
                stop_reason = "max_batches_reached"
                break
            if summary["solver_calls"] >= config.max_solver_calls_total:
                stop_reason = "solver_budget_exhausted"
                break
            if queue_state_id != current.state_id:
                before = time.perf_counter()
                candidates, generation = generate_candidates(problem, current, config)
                summary["timing_seconds"]["generation"] += time.perf_counter() - before
                generation_logs.append(generation)
                before = time.perf_counter()
                validated = []
                log_by_index = {}
                seen_colors = set()
                for candidate in candidates:
                    log = candidate_log(candidate)
                    logs.append(log)
                    log_by_index[candidate.generation_index] = log
                    validation = validate_candidate(problem, current, candidate, config)
                    log.update(status=validation.status, reason=validation.reason)
                    if validation.status == "validated":
                        if validation.candidate.colors in seen_colors:
                            log["status"] = "skipped_duplicate"
                        else:
                            seen_colors.add(validation.candidate.colors)
                            validated.append(validation.candidate)
                    else:
                        summary["validation_failures"] += 1
                        if config.strict_validation:
                            raise ValueError(f"Generated candidate failed validation: {validation.reason}")
                summary["timing_seconds"]["validation"] += time.perf_counter() - before
                before = time.perf_counter()
                queue, scores = prioritize_validated_candidates(problem, validated, config)
                summary["timing_seconds"]["priority"] += time.perf_counter() - before
                for index, score in scores.items():
                    log_by_index[index]["seed_upper_bound"] = score
                cursor, queue_state_id = 0, current.state_id
                if progress_callback:
                    progress_callback("pool", {"state_id": current.state_id, "candidates": len(queue),
                                               "truncated": generation["truncated"]})
                if not queue:
                    stop_reason = ("evaluation_failed" if candidates else
                                   "generation_budget_exhausted" if generation["truncated"] else "no_legal_candidates")
                    break
            summary["batches"] += 1
            batch_calls, optimized, budget_stop = 0, [], None
            while cursor < len(queue):
                if time.perf_counter() - start >= config.time_budget:
                    budget_stop = "time_budget_exhausted"
                    break
                candidate = queue[cursor]
                key = cache_key(problem, config, candidate.direction_signature)
                needs_solve = key not in cache and key not in failed_keys
                if needs_solve and summary["solver_calls"] >= config.max_solver_calls_total:
                    budget_stop = "solver_budget_exhausted"
                    break
                if needs_solve and batch_calls >= config.max_solver_calls_per_batch:
                    break
                cursor += 1
                log = log_by_index[candidate.original.generation_index]
                active_log = log
                log["batch"] = summary["batches"]
                if candidate.original.base_state_id != current.state_id:
                    log.update(status="stale", reason="Base state changed before evaluation")
                    active_log = None
                    continue
                if key in failed_keys:
                    log.update(status="solver_failed", reason="Previously failed identical numerical problem",
                               reused_failure=True)
                    summary["failure_cache_hits"] += 1
                    active_log = None
                    continue
                if key in cache:
                    saved = cache[key]
                    solution = Solution(readonly(saved.z), copy.deepcopy(saved.metrics), copy.deepcopy(saved.solver_info))
                    summary["cache_hits"] += 1
                    log["cache_hit"] = True
                else:
                    summary["solver_calls"] += 1
                    batch_calls += 1
                    if progress_callback:
                        progress_callback("solve", {"batch": summary["batches"],
                            "move_type": candidate.original.move_type,
                            "generation_index": candidate.original.generation_index,
                            "solver_calls": summary["solver_calls"]})
                    before = time.perf_counter()
                    try:
                        solution = solve_heights_qp(problem, candidate.colors, config)
                        checked_metrics(problem, solution.z, candidate.directed, config,
                                        get_primary_objective(solution.metrics, config))
                    except OptimizationError as exc:
                        failed_keys.add(key)
                        summary["solver_failures"] += 1
                        log.update(status=exc.status, reason=str(exc))
                        active_log = None
                        continue
                    finally:
                        summary["timing_seconds"]["solving"] += time.perf_counter() - before
                    cache[key] = Solution(readonly(solution.z), copy.deepcopy(solution.metrics),
                                          copy.deepcopy(solution.solver_info))
                    log["cache_hit"] = False
                # Independently verify cached results in this candidate's directions.
                checked_metrics(problem, solution.z, candidate.directed, config,
                                get_primary_objective(solution.metrics, config))
                objective = get_primary_objective(solution.metrics, config)
                summary["evaluated"] += 1
                log.update(status="optimized", primary_objective=objective,
                           solver_info=copy.deepcopy(solution.solver_info),
                           reason="Not a significant improvement")
                optimized.append((objective, candidate.original.generation_index, candidate, solution, log))
                active_log = None
            eligible = [row for row in optimized if is_significant_improvement(
                get_primary_objective(current.metrics, config), row[0], config)]
            for row in eligible:
                row[4]["reason"] = "Improving candidate not selected in this batch"
            if eligible:
                _, _, chosen, solution, log = min(eligible, key=lambda row: (row[0], row[1]))
                # Full fresh validation at commit, then atomically replace the state.
                validation = validate_candidate(problem, current, chosen.original, config)
                if validation.status != "validated" or validation.candidate.colors != chosen.colors:
                    raise ValueError("Selected candidate failed commit validation")
                previous = current
                new_id = f"state-{len(history) + 1:06d}"
                replacement = assemble_state(problem, chosen.colors, len(current.layer_members), solution, config, new_id)
                if not is_significant_improvement(get_primary_objective(previous.metrics, config),
                                                  get_primary_objective(replacement.metrics, config), config):
                    raise ValueError("Selected candidate no longer meets the improvement threshold")
                current, best = replacement, clone_state(replacement)
                log.update(accepted=True, reason="Significant primary-objective improvement")
                history.append({"from_state_id": previous.state_id, "to_state_id": current.state_id,
                                "move_type": chosen.original.move_type, "changes": chosen.original.changes,
                                "operations": chosen.original.operations,
                                "before_metrics": copy.deepcopy(previous.metrics),
                                "after_metrics": copy.deepcopy(current.metrics),
                                "flipped_overlap_pairs": sum(a != b for a, b in zip(
                                    direction_signature(problem, previous.colors), chosen.direction_signature))})
                summary["adoptions"] = len(history)
                for pending in queue[cursor:]:
                    log_by_index[pending.original.generation_index].update(
                        status="stale", reason="Unevaluated pool discarded after adoption")
                if checkpoint_callback:
                    checkpoint_callback(initial, best, history, {**summary, "stop_reason": "adoption_checkpoint"})
            if budget_stop:
                stop_reason = budget_stop
                break
            if not eligible and cursor == len(queue):
                successful = any(log.get("status") == "optimized" for log in log_by_index.values())
                stop_reason = "no_improvement_in_evaluated_candidates" if successful else "evaluation_failed"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        if active_log is not None:
            active_log.update(status="interrupted", reason="Active evaluation was interrupted")
    except Exception as exc:
        stop_reason, error_detail = "search_error", f"{type(exc).__name__}: {exc}"
    for log in logs:
        if log["status"] == "validated":
            log.update(status="budget_deferred", reason=f"Search stopped: {stop_reason}")
    # A final independent check precedes any final serialization.
    check_assignment(problem, best.colors, best.layer_members)
    directed, _ = build_directed_constraints(problem, best.colors)
    checked_metrics(problem, best.z, directed, config, get_primary_objective(best.metrics, config))
    initial_value, final_value = get_primary_objective(initial.metrics, config), get_primary_objective(best.metrics, config)
    summary.update(stop_reason=stop_reason, error_detail=error_detail,
                   search_seconds=time.perf_counter() - start,
                   known_unevaluated=sum(log["status"] in ("budget_deferred", "stale", "interrupted") for log in logs),
                   generation_truncated=any(log["truncated"] for log in generation_logs),
                   initial_objective=initial_value, best_objective=final_value,
                   improvement=initial_value - final_value,
                   improvement_percent=(100 * (initial_value - final_value) / initial_value if initial_value else None))
    summary["total_seconds"] = summary["initial_seconds"] + summary["search_seconds"]
    return SearchResult(initial, best, history, summary, logs, generation_logs)


def state_record(state):
    return {"state_id": state.state_id, "colors": state.colors,
            "layer_members": state.layer_members, "heights": state.z.tolist(),
            "metrics": state.metrics, "objective_config": state.objective_config,
            "solver_info": state.solver_info, "problem_signature": state.problem_signature}


def output_payload(problem, config, initial, best, history, summary,
                   candidate_logs=None, generation_logs=None):
    payload = copy.deepcopy(problem.data)
    for index, he in enumerate(payload["hyperedges"]):
        he["layer"] = float(best.z[index])
        he["logical_layer"] = best.colors[index]
    payload["overlaps"] = [list(pair) for pair in problem.overlaps]
    payload["dummy_layer_search"] = {
        "format_version": 1, "problem_signature": problem.signature, "settings": asdict(config),
        "ordinary_layer_count": len(initial.layer_members), "dummy_occupied": False,
        "initial": state_record(initial), "best": state_record(best), "summary": summary,
        "history": history, "candidates": candidate_logs or [], "generation": generation_logs or [],
        "limitations": ["Finite deterministic neighbourhood and budget; no optimality guarantee",
                        "Equal-valued completed assignments are not adopted",
                        "Dummy operations describe discrete endpoints, not continuous collision-free motion",
                        "Time budget is checked between solves; an active solve may overrun"],
    }
    return payload


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", nargs="?", default="original.json")
    parser.add_argument("output_file", nargs="?", default="layered_hypergraph.json")
    parser.add_argument("--initial-output", help="Initial viewer JSON (default: output directory/layered_hypergraph_initial.json)")
    parser.add_argument("--checkpoint", help="Checkpoint JSON (default: output stem + _checkpoint.json)")
    parser.add_argument("--objective", choices=("l2", "l1"), default="l2")
    defaults = Config()
    for field in ("reg", "time_budget", "constraint_tol", "objective_rtol", "objective_atol",
                  "improvement_abs_tol", "improvement_rel_tol", "ftol"):
        parser.add_argument("--" + field.replace("_", "-"), type=float, default=getattr(defaults, field))
    for field in ("max_candidates", "max_attempts", "max_solver_calls_per_batch", "max_solver_calls_total",
                  "max_batches", "max_blockers", "maxiter"):
        parser.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(defaults, field))
    parser.add_argument("--strict-validation", action="store_true")
    parser.add_argument("--solver-output", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    config.validate()
    output = Path(args.output_file).resolve()
    initial_path = Path(args.initial_output).resolve() if args.initial_output else output.with_name("layered_hypergraph_initial.json")
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else output.with_name(output.stem + "_checkpoint.json")
    paths = [Path(args.input_file).resolve(), output, initial_path, checkpoint_path]
    if len(set(paths)) != 4:
        raise ValueError("Input, output, initial output and checkpoint paths must be distinct")
    total_start = time.perf_counter()
    print(f"Reading hypergraph data from {args.input_file}...", flush=True)
    with open(args.input_file, encoding="utf-8") as handle:
        data = json.load(handle)
    problem = prepare_problem(data, config.reg)
    print(f"Loaded {len(data['nodes'])} nodes, {problem.n} hyperedges, {len(problem.overlaps)} overlapping pairs.", flush=True)
    print(f"Initial DSatur + {config.objective.upper()} solve (reg={config.reg}, scale={problem.scale:.9g})...", flush=True)
    def checkpoint(initial, best, history, summary):
        payload = output_payload(problem, config, initial, best, history, summary)
        if not history:
            atomic_write_json(initial_path, payload)
            print(f"Initial objective: {get_primary_objective(initial.metrics, config):.6f}; "
                  f"ordinary layers: {len(initial.layer_members)}; height range: {initial.metrics['height_range']:.6f}", flush=True)
            print(f"Initial layout: {initial_path}\nSearching completed dummy-layer candidates...", flush=True)
        else:
            print(f"Accepted {best.state_id}: objective={get_primary_objective(best.metrics, config):.6f}, "
                  f"operation={history[-1]['move_type']}", flush=True)
        atomic_write_json(checkpoint_path, payload)
    def progress(event, details):
        if event == "pool":
            print(f"{details['state_id']}: {details['candidates']} validated candidates; "
                  f"generation truncated={details['truncated']}", flush=True)
        elif event == "solve":
            print(f"Batch {details['batch']}: {details['move_type']} candidate "
                  f"{details['generation_index']}; new solve {details['solver_calls']}/"
                  f"{config.max_solver_calls_total}...", flush=True)
    result = run_dummy_layer_search(problem, config, checkpoint_callback=checkpoint,
                                   progress_callback=progress)
    result.summary["preprocessing_and_total_seconds"] = time.perf_counter() - total_start
    payload = output_payload(problem, config, result.initial, result.best, result.history,
                             result.summary, result.candidate_logs, result.generation_logs)
    atomic_write_json(output, payload)
    atomic_write_json(checkpoint_path, payload)
    print("Metric                         Initial             Best", flush=True)
    for metric in ("l2_scaled_qp", "l1_regularized", "height_range", "pillar_length", "order_violation_count"):
        print(f"{metric:28s} {result.initial.metrics[metric]:14.6f} {result.best.metrics[metric]:16.6f}", flush=True)
    print(f"Adoptions: {result.summary['adoptions']}; new solves: {result.summary['solver_calls']}; "
          f"cache hits: {result.summary['cache_hits']}; stop: {result.summary['stop_reason']}", flush=True)
    print(f"Improvement: {result.summary['improvement']:.6f}; output: {output}", flush=True)
    if result.summary["error_detail"]:
        print(f"Search error (last valid result saved): {result.summary['error_detail']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, OptimizationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
