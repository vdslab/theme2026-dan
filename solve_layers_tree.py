#!/usr/bin/env python3
"""Assign integer layers using a virtual-root tree, repair and annealing.

Run from the repository root:
    python solve_layers_tree.py --time-budget 300 --seed 0
    python solve_layers_tree.py --initial-only
    python solve_layers_tree.py --time-budget 0 --max-iterations 10000 --seed 0

Every evaluated search state is feasible. Repairs may move either side of a
collision, selecting the smallest one-step squared-cost change. Parents need
not overlap their children. Compatible parent links and sibling order survive
repair. There are no QP solves or fixed coloring directions in this solver.
"""

import argparse
from collections import deque
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

from solve_layers_dsatur_l1 import (
    compute_weights, evaluate_layout, extract_overlaps, load_comparison,
)
from solve_layers_dummy import atomic_write_json


class DeadlineExceeded(RuntimeError):
    pass


class RepairFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    reg: float = 1e-5
    seed: int = 0
    max_iterations: int = 10000
    time_budget: float = 300.0  # Includes preprocessing and initialization in CLI.
    method: str = "anneal"
    initial_temperature: object = None
    temperature_samples: int = 16
    initial_acceptance: float = 0.8
    final_temperature_ratio: float = 0.001
    move_weights: tuple = (0.4, 0.3, 0.25, 0.05)  # move, swap, subtree, rewire
    max_repair_steps: object = None
    log_limit: int = 0
    checkpoint_interval: float = 5.0

    def validate(self):
        for name in ("reg", "time_budget", "checkpoint_interval"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("seed", "max_iterations", "temperature_samples", "log_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_repair_steps is not None and (
                type(self.max_repair_steps) is not int or self.max_repair_steps < 0):
            raise ValueError("max_repair_steps must be a nonnegative integer")
        if self.method not in ("anneal", "greedy"):
            raise ValueError("method must be anneal or greedy")
        if self.initial_temperature is not None and (
                not math.isfinite(self.initial_temperature) or self.initial_temperature <= 0):
            raise ValueError("initial_temperature must be finite and positive")
        if not 0 < self.initial_acceptance < 1:
            raise ValueError("initial_acceptance must be between zero and one")
        if not 0 < self.final_temperature_ratio <= 1:
            raise ValueError("final_temperature_ratio must be in (0, 1]")
        if (len(self.move_weights) != 4 or
                any(not math.isfinite(w) or w < 0 for w in self.move_weights) or
                not 0 < sum(self.move_weights) < math.inf):
            raise ValueError("four nonnegative move weights with a positive sum are required")


@dataclass(frozen=True)
class Problem:
    data: dict
    n: int
    overlaps: tuple
    edges: np.ndarray
    adjacency: tuple
    weights: np.ndarray
    regularized: np.ndarray
    row_sums: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_weights: np.ndarray
    scale: float
    reg: float
    signature: str


@dataclass(frozen=True)
class Tree:
    parent: tuple  # Real nodes only. Virtual root has index N.
    children: tuple  # Includes the virtual root's children.


@dataclass(frozen=True)
class State:
    state_id: str
    tree: Tree
    layers: tuple
    metrics: dict

    @property
    def objective(self):
        return self.metrics["l2_scaled_qp"]


@dataclass(frozen=True)
class Proposal:
    move_type: str
    layers: tuple
    tree: Tree
    details: dict


@dataclass(frozen=True)
class RepairResult:
    layers: tuple
    steps: tuple
    initial_violations: int


@dataclass(frozen=True)
class SearchResult:
    initial: State
    best: State
    current: State
    summary: dict
    history: list
    logs: list


MOVE_TYPES = ("move", "swap", "subtree", "rewire")


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise DeadlineExceeded("Time budget exhausted")


def readonly(values, dtype=float):
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def integer_layers(values, n):
    array = np.asarray(values)
    if array.shape != (n,) or array.dtype.kind not in "iuf":
        raise ValueError("Expected one integer layer per hyperedge")
    if not np.isfinite(array).all() or np.any(array != np.floor(array)):
        raise ValueError("Layers must be finite integers")
    # Leave room for a temporary layer above/below the current range.
    if np.any(array <= -(2**52)) or np.any(array >= 2**52):
        raise ValueError("Layer coordinates are too large for exact arithmetic")
    return np.asarray(array, dtype=np.int64).copy()


def prepare_problem(data, reg=1e-5, overlaps=None, progress=None):
    if not math.isfinite(reg) or reg < 0:
        raise ValueError("reg must be finite and nonnegative")
    data = copy.deepcopy(data)
    nodes, hyperedges = data["nodes"], data["hyperedges"]
    if not hyperedges:
        raise ValueError("At least one hyperedge is required")
    node_ids = [node["id"] for node in nodes]
    edge_ids = [edge["id"] for edge in hyperedges]
    if len(set(node_ids)) != len(nodes) or len(set(edge_ids)) != len(hyperedges):
        raise ValueError("Node IDs and hyperedge IDs must each be unique")
    if any(not math.isfinite(float(node[axis])) for node in nodes for axis in ("x", "y")):
        raise ValueError("Node coordinates must be finite")
    known = set(node_ids)
    if any(set(edge["nodes"]) - known for edge in hyperedges):
        raise ValueError("A hyperedge references an unknown node")
    n = len(hyperedges)
    pairs = set()
    if progress is not None:
        progress("Computing polygons from node order using Shapely...")
        progress("Computing overlap set C...")
    raw = extract_overlaps(nodes, hyperedges) if overlaps is None else overlaps
    for i, j in raw:
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or not 0 <= v < n for v in (i, j)) or i == j:
            raise ValueError("Invalid overlap endpoints")
        pairs.add(tuple(sorted((int(i), int(j)))))
    overlaps = tuple(sorted(pairs))
    if progress is not None:
        progress(f"Found {len(overlaps)} overlapping pairs (C set).")
        progress("Computing weight matrix W...")
        progress(f"Applying regularization (reg = {reg}) to ensure connectivity...")
    adjacency = [set() for _ in range(n)]
    for i, j in overlaps:
        adjacency[i].add(j)
        adjacency[j].add(i)
    weights, regularized, scale = compute_weights(hyperedges, reg)
    pair_i, pair_j = np.triu_indices(n, 1)
    signature = hashlib.sha256(json.dumps(
        {"data": data, "overlaps": overlaps, "reg": reg},
        sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return Problem(
        data, n, overlaps, readonly(overlaps, int).reshape(-1, 2),
        tuple(frozenset(a) for a in adjacency), readonly(weights),
        readonly(regularized), readonly(regularized.sum(axis=1)),
        readonly(pair_i, int), readonly(pair_j, int),
        readonly(regularized[pair_i, pair_j]), scale, reg, signature,
    )


def objective(problem, layers):
    z = np.asarray(layers, dtype=float)
    if z.shape != (problem.n,) or not np.isfinite(z).all():
        raise ValueError("Invalid objective coordinates")
    differences = z[problem.pair_i] - z[problem.pair_j]
    value = float(np.dot(problem.pair_weights, differences * differences))
    if not math.isfinite(value):
        raise ValueError("Nonfinite objective")
    return value


def solve_layers_objective(problem, layers):
    """Evaluate solve_layers.py's anchored QP expression without solving a QP.

    Tree heights need not have z[0] == 0. Translation by z[0] supplies the old
    solver's anchor while preserving all pairwise differences and its cost.
    """
    z = np.asarray(layers, dtype=float)
    if z.shape != (problem.n,) or not np.isfinite(z).all():
        raise ValueError("Invalid objective coordinates")
    y = z[1:] - z[0]
    l_yy = np.diag(problem.row_sums[1:]) - problem.regularized[1:, 1:]
    value = float(2.0 * problem.scale * (y @ l_yy @ y))
    if not math.isfinite(value):
        raise ValueError("Nonfinite solve_layers.py objective")
    return value


def moments(problem, layers):
    # Avoid BLAS thread-start overhead for these small dense matrix-vector products.
    return np.einsum("ij,j->i", problem.regularized, np.asarray(layers, float))


def move_delta(problem, layers, node, target, weighted_moments=None):
    z = np.asarray(layers)
    m = moments(problem, z) if weighted_moments is None else weighted_moments
    old = int(z[node])
    return float((target - old) * ((target + old) * problem.row_sums[node] - 2 * m[node]))


def changed_layers_delta(problem, before, after):
    """Sequential moment updates count interactions between moved nodes once."""
    z = integer_layers(before, problem.n)
    target = integer_layers(after, problem.n)
    m = moments(problem, z)
    delta = 0.0
    for node in np.flatnonzero(z != target):
        new = int(target[node])
        delta += move_delta(problem, z, node, new, m)
        difference = new - int(z[node])
        m += problem.regularized[:, node] * difference
        z[node] = new
    return delta


def violating_edges(problem, layers):
    z = np.asarray(layers)
    return problem.edges[z[problem.edges[:, 0]] == z[problem.edges[:, 1]]]


def validate_tree(tree, n):
    if len(tree.parent) != n or len(tree.children) != n + 1:
        raise ValueError("Incorrect tree dimensions")
    seen = set()
    for parent, children in enumerate(tree.children):
        for child in children:
            if type(child) is not int or not 0 <= child < n or child in seen:
                raise ValueError("Missing, duplicate or invalid child")
            if type(tree.parent[child]) is not int or tree.parent[child] != parent:
                raise ValueError("Parent and children disagree")
            seen.add(child)
    if seen != set(range(n)):
        raise ValueError("Every real node must have exactly one parent")
    depths = [-1] * (n + 1)
    depths[n] = 0
    queue = deque([n])
    while queue:
        parent = queue.popleft()
        for child in tree.children[parent]:
            if depths[child] != -1:
                raise ValueError("Tree contains a cycle")
            depths[child] = depths[parent] + 1
            queue.append(child)
    if any(depth < 0 for depth in depths):
        raise ValueError("Tree is disconnected or cyclic")
    return tuple(depths)


def bfs_order(tree):
    queue = deque(tree.children[-1])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        queue.extend(tree.children[node])
    return tuple(order)


def compact_layers(layers):
    return tuple(int(z) for z in np.unique(layers, return_inverse=True)[1])


def tree_for_layers(layers, previous):
    """Keep every compatible edge and old sibling order; append orphans only."""
    n = len(previous.parent)
    validate_tree(previous, n)
    z = compact_layers(integer_layers(layers, n))
    parent = [-1] * n
    children = [[] for _ in range(n + 1)]
    for old_parent, old_children in enumerate(previous.children):
        for node in old_children:
            compatible = (z[node] == 0 if old_parent == n
                          else z[node] == z[old_parent] + 1)
            if compatible:
                parent[node] = old_parent
                children[old_parent].append(node)
    preceding = [n]
    for level in range(max(z) + 1):
        for node in range(n):
            if z[node] == level and parent[node] < 0:
                parent[node] = preceding[0]
                children[preceding[0]].append(node)
        preceding = [child for p in preceding for child in children[p]]
    tree = Tree(tuple(parent), tuple(tuple(row) for row in children))
    depth = validate_tree(tree, n)
    if tuple(d - 1 for d in depth[:-1]) != z:
        raise ValueError("Rebuilt tree does not match requested layers")
    return tree, z


def bfs_tree(problem):
    remaining = set(range(problem.n))
    parent = [-1] * problem.n
    children = [[] for _ in range(problem.n + 1)]
    affinity = problem.weights.sum(axis=1)
    while remaining:
        root = min(remaining, key=lambda i: (-len(problem.adjacency[i]), -affinity[i], i))
        parent[root] = problem.n
        children[problem.n].append(root)
        remaining.remove(root)
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for child in sorted(problem.adjacency[node]):
                if child in remaining:
                    remaining.remove(child)
                    parent[child] = node
                    children[node].append(child)
                    queue.append(child)
    return Tree(tuple(parent), tuple(tuple(row) for row in children))


def repair_layers(problem, layers, max_steps=None, deadline=None):
    """Greedy legal single-node moves strictly decrease the collision count."""
    z = integer_layers(layers, problem.n)
    bad = violating_edges(problem, z)
    initial_count = len(bad)
    if not initial_count:
        return RepairResult(tuple(map(int, z)), (), 0)
    expected = objective(problem, z)
    m = moments(problem, z)
    steps = []
    while len(bad):
        check_deadline(deadline)
        if max_steps is not None and len(steps) >= max_steps:
            raise RepairFailed("Repair step budget exhausted")
        # Gaps in a temporary coordinate array can serve as free layers too.
        targets = np.arange(z.min() - 1, z.max() + 2, dtype=np.int64)
        best = None
        for count, raw_node in enumerate(np.unique(bad)):
            if count % 16 == 0:
                check_deadline(deadline)
            node = int(raw_node)
            forbidden = z[list(problem.adjacency[node])]
            allowed = targets[~np.isin(targets, forbidden)]
            deltas = ((allowed - z[node]) *
                      ((allowed + z[node]) * problem.row_sums[node] - 2 * m[node]))
            choice = int(np.lexsort((allowed, np.abs(allowed - z[node]), deltas))[0])
            target = int(allowed[choice])
            key = (float(deltas[choice]), abs(target - int(z[node])), node, target)
            if best is None or key < best:
                best = key
        delta, _, node, target = best
        old, count_before = int(z[node]), len(bad)
        m += problem.regularized[:, node] * (target - old)
        z[node] = target
        expected += delta
        bad = violating_edges(problem, z)
        if len(bad) >= count_before:
            raise RuntimeError("Repair did not reduce the collision count")
        steps.append({"node": node, "from": old, "to": target, "delta": delta,
                      "violations_before": count_before, "violations_after": len(bad)})
    if not math.isclose(expected, objective(problem, z), rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError("Repair's incremental objective differs from full calculation")
    return RepairResult(tuple(map(int, z)), tuple(steps), initial_count)


def layout_metrics(problem, layers):
    z = np.asarray(layers, float)
    directed = np.asarray([
        (i, j) if z[i] < z[j] else (j, i) for i, j in problem.overlaps
    ], dtype=int).reshape(-1, 2)
    metrics = evaluate_layout(z, problem.data["nodes"], problem.data["hyperedges"],
                              problem.weights, problem.regularized, problem.scale, directed)
    metrics["used_layer_count"] = len(set(layers))
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Nonfinite metric")
    return metrics


def make_state(problem, tree, state_id="state-000000"):
    depth = validate_tree(tree, problem.n)
    layers = tuple(d - 1 for d in depth[:-1])
    if len(violating_edges(problem, layers)):
        raise ValueError("Tree violates an original overlap constraint")
    metrics = layout_metrics(problem, layers)
    return State(state_id, tree, layers, metrics)


def validate_state(problem, state):
    fresh = make_state(problem, state.tree, state.state_id)
    if fresh.layers != state.layers or fresh.metrics.keys() != state.metrics.keys():
        raise ValueError("State's stored layers or metrics disagree with the tree")
    for key, value in fresh.metrics.items():
        if not math.isclose(value, state.metrics[key], rel_tol=1e-10, abs_tol=1e-7):
            raise ValueError(f"Stored metric disagrees: {key}")


def initial_state(problem, config, deadline=None):
    tree = bfs_tree(problem)
    depth = validate_tree(tree, problem.n)
    repair = repair_layers(problem, [d - 1 for d in depth[:-1]],
                           config.max_repair_steps, deadline)
    tree, _ = tree_for_layers(repair.layers, tree)
    return make_state(problem, tree), repair


def subtree_nodes(tree, node):
    queue = deque([node])
    result = []
    while queue:
        current = queue.popleft()
        result.append(current)
        queue.extend(tree.children[current])
    return tuple(result)


def reparent(tree, node, parent):
    n = len(tree.parent)
    if not 0 <= node < n or not 0 <= parent <= n:
        raise ValueError("Invalid reparent indices")
    if parent in subtree_nodes(tree, node):
        raise ValueError("Cannot attach a subtree to itself or its descendants")
    parents = list(tree.parent)
    children = [list(row) for row in tree.children]
    if parent != parents[node]:
        children[parents[node]].remove(node)
        children[parent].append(node)
        parents[node] = parent
    result = Tree(tuple(parents), tuple(tuple(row) for row in children))
    validate_tree(result, n)
    return result


def propose(problem, state, rng, config):
    kind = rng.choices(MOVE_TYPES, weights=config.move_weights, k=1)[0]
    n, z, tree = problem.n, list(state.layers), state.tree
    node = rng.randrange(n)
    details = {"node": node}
    if kind == "move":
        targets = [level for level in range(-1, max(z) + 2) if level != z[node]]
        target = rng.choice(targets)
        details.update(target=target)
        z[node] = target
    elif kind == "swap":
        others = [i for i in range(n) if z[i] != z[node]]
        if not others:
            return None, kind
        other = rng.choice(others)
        details.update(other=other)
        z[node], z[other] = z[other], z[node]
    else:
        forbidden = set(subtree_nodes(tree, node))
        parents = [p for p in range(n + 1) if p not in forbidden and p != tree.parent[node]]
        if kind == "rewire":
            depth = validate_tree(tree, n)
            parents = [p for p in parents if depth[p] == depth[tree.parent[node]]]
        if not parents:
            return None, kind
        parent = rng.choice(parents)
        details.update(parent=parent)
        tree = reparent(tree, node, parent)
        depth = validate_tree(tree, n)
        z = [d - 1 for d in depth[:-1]]
    return Proposal(kind, tuple(z), tree, details), kind


def evaluate_proposal(problem, proposal, config, state_id, deadline=None):
    before = time.monotonic()
    repair = repair_layers(problem, proposal.layers, config.max_repair_steps, deadline)
    repair_seconds = time.monotonic() - before
    before = time.monotonic()
    tree, layers = tree_for_layers(repair.layers, proposal.tree)
    rebuilding_seconds = time.monotonic() - before
    before = time.monotonic()
    state = make_state(problem, tree, state_id)
    # Independently check the multi-node delta against the full objective.
    delta = changed_layers_delta(problem, proposal.layers, layers)
    if not math.isclose(objective(problem, proposal.layers) + delta,
                        state.metrics["l2_regularized"], rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError("Candidate's incremental objective differs from full calculation")
    return state, repair, {
        "repair": repair_seconds, "rebuilding": rebuilding_seconds,
        "evaluation": time.monotonic() - before,
    }


def accept_move(delta, temperature, rng, method="anneal"):
    if delta <= 0:
        return True
    return (method == "anneal" and temperature > 0 and
            rng.random() < math.exp(-delta / temperature))


def calibrate_temperature(problem, state, config, deadline=None):
    if config.method == "greedy":
        return 0.0, {"samples": 0, "positive_samples": 0, "source": "greedy"}
    if config.initial_temperature is not None:
        return config.initial_temperature, {"samples": 0, "positive_samples": 0, "source": "explicit"}
    rng = random.Random(config.seed ^ 0x75A94E)
    positive = []
    evaluated = 0
    for index in range(config.temperature_samples):
        check_deadline(deadline)
        proposal, _ = propose(problem, state, rng, config)
        if proposal is None:
            continue
        try:
            candidate, _, _ = evaluate_proposal(
                problem, proposal, config, "temperature-sample", deadline)
        except RepairFailed:
            continue
        evaluated += 1
        if candidate.objective > state.objective:
            positive.append(candidate.objective - state.objective)
    if positive:
        value = -float(np.median(positive)) / math.log(config.initial_acceptance)
        source = "sampled"
    else:
        value = max(1.0, abs(state.objective) * 0.01)
        source = "fallback_no_positive_deltas"
    return value, {"samples": evaluated, "positive_samples": len(positive), "source": source}


def run_search(problem, config, checkpoint=None, starting_state=None, started_at=None):
    config.validate()
    if config.reg != problem.reg:
        raise ValueError("Config regularization differs from prepared problem")
    start = time.monotonic() if started_at is None else started_at
    deadline = start + config.time_budget if config.time_budget else None
    summary = {
        "attempts": 0, "accepted": 0, "best_improvements": 0,
        "repair_steps": 0, "repair_failures": 0, "returned_to_current": 0,
        "equal_cost_tree_changes": 0,
        "temperature": {"samples": 0, "positive_samples": 0,
                        "initial": None, "source": "not_started",
                        "final_ratio": config.final_temperature_ratio},
        "timing_seconds": {k: 0.0 for k in (
            "initialization", "calibration", "generation", "repair", "rebuilding",
            "evaluation", "saving")},
        "by_move": {kind: {"attempts": 0, "evaluated": 0, "accepted": 0,
                          "improvements": 0, "repair_steps": 0,
                          "repair_failures": 0, "seconds": 0.0} for kind in MOVE_TYPES},
    }
    before = time.monotonic()
    check_deadline(deadline)
    if starting_state is None:
        initial, repair = initial_state(problem, config, deadline)
    else:
        validate_state(problem, starting_state)
        initial, repair = starting_state, RepairResult(starting_state.layers, (), 0)
    summary["timing_seconds"]["initialization"] = time.monotonic() - before
    summary["initial_repair"] = {"violations": repair.initial_violations, "steps": list(repair.steps)}
    current = best = initial
    history, logs = [], []

    def save(phase):
        if checkpoint is not None:
            validate_state(problem, best)
            before = time.monotonic()
            checkpoint(initial, best, {**summary, "phase": phase}, history)
            summary["timing_seconds"]["saving"] += time.monotonic() - before

    stop_reason = "max_iterations_reached" if config.max_iterations else "initial_only"
    temperature = 0.0
    try:
        save("initial")
        before = time.monotonic()
        summary["temperature"]["source"] = "calibrating"
        try:
            if config.max_iterations:
                temperature, info = calibrate_temperature(problem, initial, config, deadline)
            else:
                info = {"samples": 0, "positive_samples": 0, "source": "initial_only"}
        finally:
            summary["timing_seconds"]["calibration"] = time.monotonic() - before
        summary["temperature"] = {**info, "initial": temperature,
                                  "final_ratio": config.final_temperature_ratio}
        rng = random.Random(config.seed)
        acceptance_rng = random.Random(config.seed ^ 0x128ABD)
        search_start = time.monotonic()
        for attempt in range(1, config.max_iterations + 1):
            check_deadline(deadline)
            summary["attempts"] = attempt
            before = time.monotonic()
            proposal, kind = propose(problem, current, rng, config)
            summary["timing_seconds"]["generation"] += time.monotonic() - before
            counts = summary["by_move"][kind]
            counts["attempts"] += 1
            if proposal is None:
                continue
            before = time.monotonic()
            try:
                candidate, repair, timing = evaluate_proposal(
                    problem, proposal, config, f"state-{attempt:06d}", deadline)
            except RepairFailed:
                summary["repair_failures"] += 1
                counts["repair_failures"] += 1
                counts["seconds"] += time.monotonic() - before
                summary["timing_seconds"]["repair"] += time.monotonic() - before
                continue
            counts["seconds"] += time.monotonic() - before
            for key, value in timing.items():
                summary["timing_seconds"][key] += value
            counts["evaluated"] += 1
            counts["repair_steps"] += len(repair.steps)
            summary["repair_steps"] += len(repair.steps)
            unchanged = candidate.layers == current.layers and candidate.tree == current.tree
            if unchanged:
                summary["returned_to_current"] += 1
            progress = (attempt - 1) / max(1, config.max_iterations - 1)
            if deadline is not None:
                progress = max(progress, min(1.0, (time.monotonic() - search_start) /
                                             max(deadline - search_start, 1e-12)))
            t = temperature * config.final_temperature_ratio ** progress
            delta = candidate.objective - current.objective
            accepted = not unchanged and accept_move(delta, t, acceptance_rng, config.method)
            log = {"attempt": attempt, "move_type": kind, "details": proposal.details,
                   "repair_steps": len(repair.steps), "delta_scaled": delta,
                   "temperature": t, "accepted": accepted, "returned_to_current": unchanged}
            if len(logs) < config.log_limit:
                logs.append(log)
            if accepted:
                if candidate.layers == current.layers and candidate.tree != current.tree:
                    summary["equal_cost_tree_changes"] += 1
                current = candidate
                summary["accepted"] += 1
                counts["accepted"] += 1
                if delta < 0:
                    counts["improvements"] += 1
                if current.objective < best.objective - max(1e-9, abs(best.objective) * 1e-12):
                    validate_state(problem, current)
                    best = current
                    summary["best_improvements"] += 1
                    history.append({"attempt": attempt, "move_type": kind,
                                    "objective": best.objective,
                                    "height_range": best.metrics["height_range"],
                                    "used_layer_count": best.metrics["used_layer_count"]})
                    save("improvement")
    except DeadlineExceeded:
        stop_reason = "time_budget_exhausted"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    validate_state(problem, best)
    validate_state(problem, current)
    summary.update(
        stop_reason=stop_reason, initial_objective=initial.objective,
        best_objective=best.objective, current_objective=current.objective,
        improvement=initial.objective - best.objective,
        improvement_percent=(100 * (initial.objective - best.objective) / initial.objective
                             if initial.objective else 0.0),
        total_seconds=time.monotonic() - start,
        returned_to_current_ratio=(summary["returned_to_current"] / summary["attempts"]
                                   if summary["attempts"] else 0.0),
    )
    return SearchResult(initial, best, current, summary, history, logs)


def state_record(state):
    return {"state_id": state.state_id, "parent": state.tree.parent,
            "children": state.tree.children,
            "depth": tuple(z + 1 for z in state.layers) + (0,),
            "layers": state.layers, "metrics": copy.deepcopy(state.metrics)}


def output_payload(problem, config, initial, best, summary, history, logs=None, provenance=None):
    validate_state(problem, initial)
    validate_state(problem, best)
    payload = copy.deepcopy(problem.data)
    for edge, layer in zip(payload["hyperedges"], best.layers):
        edge["layer"] = float(layer)
    payload["overlaps"] = [list(pair) for pair in problem.overlaps]
    payload["tree_layer_search"] = {
        "format_version": 1, "problem_signature": problem.signature,
        "provenance": provenance or {}, "settings": asdict(config),
        "virtual_root_index": problem.n, "scale": problem.scale,
        "index_to_hyperedge_id": [edge["id"] for edge in problem.data["hyperedges"]],
        "overlaps_sha256": hashlib.sha256(json.dumps(problem.overlaps).encode()).hexdigest(),
        "initial": state_record(initial), "best": state_record(best),
        "summary": copy.deepcopy(summary), "history": copy.deepcopy(history),
        "candidate_logs": copy.deepcopy(logs or []),
        "validation": {"tree_valid": True, "overlap_violation_count": 0,
                       "metrics_independently_recomputed": True},
        "limitations": [
            "Integer heights; not the same search space as continuous QP",
            "Greedy repair and asymmetric proposals; no global optimality guarantee",
            "Repair may undo a proposed move; both sides are eligible",
        ],
    }
    return payload


def comparison_metrics(problem, path):
    z = load_comparison(path, problem.data["nodes"], problem.data["hyperedges"], problem.overlaps)
    metrics = layout_metrics(problem, z)
    if metrics["overlap_violation_count"]:
        raise ValueError("Comparison layout violates overlap separation")
    metrics["distinct_height_count"] = metrics.pop("used_layer_count")
    return {"path": str(path), "metrics": metrics,
            "note": "Recomputed with this input's weights and scale; heights may be continuous"}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", nargs="?", default="original.json")
    parser.add_argument("output_file", nargs="?", default="layers_hypergraph.json",
                        help="Completed viewer JSON, matching solve_layers.py (default: layers_hypergraph.json)")
    parser.add_argument("--initial-output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--report")
    parser.add_argument("--compare")
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--method", choices=("anneal", "greedy"), default="anneal")
    defaults = Config()
    for name in ("reg", "time_budget", "initial_acceptance", "final_temperature_ratio",
                 "checkpoint_interval"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--initial-temperature", type=float)
    for name in ("seed", "max_iterations", "temperature_samples", "log_limit"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--max-repair-steps", type=int)
    parser.add_argument("--move-weights", type=float, nargs=4, default=defaults.move_weights,
                        metavar=("MOVE", "SWAP", "SUBTREE", "REWIRE"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    fields = {name: getattr(args, name) for name in Config.__dataclass_fields__}
    fields["move_weights"] = tuple(fields["move_weights"])
    if args.initial_only:
        fields["max_iterations"] = 0
    config = Config(**fields)
    config.validate()
    source, output = Path(args.input_file).resolve(), Path(args.output_file).resolve()
    auxiliary_directory = (Path("results/layers").resolve()
                           if output.parent == Path.cwd().resolve() else output.parent)
    initial_path = (Path(args.initial_output).resolve() if args.initial_output
                    else auxiliary_directory / (output.stem + "_initial.json"))
    checkpoint_path = (Path(args.checkpoint).resolve() if args.checkpoint
                       else auxiliary_directory / (output.stem + "_checkpoint.json"))
    report_path = (Path(args.report).resolve() if args.report else
                   Path("results/reports", output.stem + "_report.json").resolve())
    products = [output, initial_path, checkpoint_path, report_path]
    protected = {source, Path(__file__).resolve().with_name("layered_hypergraph.json")}
    if args.compare:
        protected.add(Path(args.compare).resolve())
    if len(set(products)) != len(products) or any(p in protected for p in products):
        raise ValueError("Outputs must be distinct and must not replace the input, comparison or active viewer data")

    def log(message):
        if not args.quiet:
            print(message, flush=True)

    started = time.monotonic()
    log(f"Reading hypergraph data from {args.input_file}...")
    source_bytes = source.read_bytes()
    data = json.loads(source_bytes.decode("utf-8"))
    log(f"Loaded {len(data['nodes'])} nodes and {len(data['hyperedges'])} hyperedges.")
    problem = prepare_problem(data, config.reg, progress=log)
    comparison = comparison_metrics(problem, args.compare) if args.compare else None
    provenance = {"input": str(source), "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "preprocessing_seconds": time.monotonic() - started}

    last_checkpoint = -math.inf

    def checkpoint(initial, best, summary, history):
        nonlocal last_checkpoint
        now = time.monotonic()
        if summary["phase"] != "initial" and now - last_checkpoint < config.checkpoint_interval:
            return
        payload = output_payload(problem, config, initial, best, summary, history,
                                 provenance=provenance)
        if summary["phase"] == "initial":
            atomic_write_json(initial_path, payload)
        atomic_write_json(checkpoint_path, payload)
        last_checkpoint = time.monotonic()
        if summary["phase"] == "initial":
            log(f"Tree initialization and repair completed. Total layers used: "
                f"{best.metrics['used_layer_count']}")
            log(f"Initial objective value (solve_layers.py): {best.objective:.6f}")
            if config.max_iterations:
                log(f"Phase 2: Optimizing tree layers using {config.method}...")
        else:
            log(f"Improved objective value (solve_layers.py): {best.objective:.6f}; "
                f"layers={best.metrics['used_layer_count']}; "
                f"height={best.metrics['height_range']:.0f}")

    log("Phase 1: Computing a virtual-root BFS tree on overlap graph C and repairing conflicts...")
    log(f"Objective scale factor: {problem.scale:.6e}")
    result = run_search(problem, config, checkpoint, started_at=started)
    legacy_objective = solve_layers_objective(problem, result.best.layers)
    if not math.isclose(legacy_objective, result.best.objective, rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError("solve_layers.py objective disagrees with tree-search objective")
    result.summary["objective_value_solve_layers"] = legacy_objective
    log(f"Optimization finished after {result.summary['attempts']} attempts "
        f"({result.summary['total_seconds']:.3f} s). Stop reason: {result.summary['stop_reason']}")
    log(f"Best objective value: {result.best.objective:.6f}")
    log(f"Objective value (solve_layers.py): {legacy_objective:.6f}")
    log(f"Unscaled squared objective value: {result.best.metrics['l2_regularized']:.6f}")
    log("All overlap constraints successfully satisfied (|x_i - x_j| >= 1).")
    log(f"Coordinate x_0 (first hyperedge): {result.best.layers[0]:.6f}")
    log("Anchor coordinate x_0 for solve_layers.py objective evaluation: 0.000000")
    log(f"Coordinate range: min={result.best.metrics['height_min']:.4f}, "
        f"max={result.best.metrics['height_max']:.4f}")
    if comparison is not None:
        baseline = comparison["metrics"]["l2_scaled_qp"]
        log(f"Comparison objective value (solve_layers.py; {args.compare}): {baseline:.6f}")
        log(f"Objective improvement over comparison: {baseline - legacy_objective:.6f}")
    log("Preparing output JSON...")
    payload = output_payload(problem, config, result.initial, result.best, result.summary,
                             result.history, result.logs, provenance)
    # Keep the completed output identical to the legacy viewer schema.
    # Full input metadata and tree details remain in the sidecar files.
    viewer_payload = {key: payload[key] for key in ("nodes", "hyperedges", "overlaps")}
    log(f"Writing output to {args.output_file}...")
    atomic_write_json(output, viewer_payload)
    atomic_write_json(checkpoint_path, payload)
    report = {"method": "rooted_tree_repair_" + config.method, "settings": asdict(config),
              "provenance": provenance, "problem_signature": problem.signature,
              "nodes": len(problem.data["nodes"]), "hyperedges": problem.n,
              "overlaps": len(problem.overlaps), "scale": problem.scale,
              "initial_metrics": result.initial.metrics, "best_metrics": result.best.metrics,
              "current_metrics": result.current.metrics, "summary": result.summary,
              "history": result.history, "comparison": comparison,
              "objective_value_solve_layers": legacy_objective,
              "objective_formula_solve_layers": "2 * scale * y.T @ L_yy @ y; y = z[1:] - z[0]",
              "initial": state_record(result.initial), "best": state_record(result.best),
              "outputs": {"initial": str(initial_path), "best": str(output),
                          "checkpoint": str(checkpoint_path)},
              "validation": payload["tree_layer_search"]["validation"],
              "limitations": payload["tree_layer_search"]["limitations"]}
    atomic_write_json(report_path, report)
    log(f"Initial -> best objective: {result.initial.objective:.6f} -> "
        f"{result.best.objective:.6f} ({result.summary['improvement_percent']:.2f}% improvement)")
    log(f"Report: {report_path}")
    log("Layer assignment complete!")
    return 130 if result.summary["stop_reason"] == "interrupted" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted before a feasible initial state was saved.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, KeyError, OSError, RepairFailed, DeadlineExceeded) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
