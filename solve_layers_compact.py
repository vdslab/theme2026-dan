#!/usr/bin/env python3
"""Find compact, feasible integer layers without changing the input XY layout.

The quadratic term in the mixed objective is l2_scaled_qp from the existing
solvers. An overlap forbids equal layers; it does not fix their relative order.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

from solve_layers import dsatur_coloring
from solve_layers_tree import Problem, layout_metrics, prepare_problem
from native.compact_search import DEFAULT_LIBRARY, INT32_MAX, NativeSearch


METHODS = ("greedy-input", "greedy-size", "greedy-degree", "dsatur", "anneal")
OBJECTIVES = ("pillar", "squared", "hybrid", "degree-span")
MOVES = ("move", "kempe", "swap", "shift")
BACKENDS = ("auto", "native", "python")
IMPROVEMENT_TOLERANCE = 1e-9
SHIFT_CACHE_AFTER = 6


@dataclass(frozen=True)
class Config:
    method: str = "anneal"
    objective: str = "hybrid"
    alpha: float = 0.25
    beta: float = 0.005
    seed: int = 0
    reg: float = 1e-5
    max_iterations: int = 20_000
    time_budget: float = 0.0
    layer_cap: int | None = None
    warm_start_extra_layers: int = 0
    moves: tuple[str, ...] = MOVES
    optimize_free_edges: bool | None = None

    def __post_init__(self):
        if self.method not in METHODS or self.objective not in OBJECTIVES:
            raise ValueError("Unknown method or objective.")
        if any(not math.isfinite(value) or value < 0
               for value in (self.alpha, self.beta, self.reg, self.time_budget)):
            raise ValueError("Coefficients and the time budget must be finite and nonnegative.")
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ValueError("The iteration budget must be a nonnegative integer.")
        if self.layer_cap is not None and (type(self.layer_cap) is not int or self.layer_cap < 1):
            raise ValueError("The layer cap must be a positive integer.")
        if self.warm_start_extra_layers not in (0, 1, 2):
            raise ValueError("Warm starts can add at most two layers.")
        if self.warm_start_extra_layers and self.method != "anneal":
            raise ValueError("Warm starts require annealing.")
        if not self.moves or len(set(self.moves)) != len(self.moves) or set(self.moves) - set(MOVES):
            raise ValueError("Specify one or more distinct supported moves.")

    @property
    def refine_free_edges(self):
        if self.optimize_free_edges is None:
            return self.method == "anneal"
        return self.optimize_free_edges


def compact(layers):
    """Remove unused levels while preserving order."""
    occupied = np.bincount(layers) > 0
    return (np.cumsum(occupied) - 1)[layers]


def initial_coloring(problem, method):
    if method in ("dsatur", "anneal"):
        colors = dsatur_coloring(problem.n, problem.overlaps)
        return np.array([colors[index] for index in range(problem.n)], dtype=np.int64)

    order = range(problem.n)
    if method == "greedy-size":
        edges = problem.data["hyperedges"]
        order = sorted(order, key=lambda index: -len(set(edges[index]["nodes"])))
    if method == "greedy-degree":
        order = sorted(order, key=lambda index: -len(problem.adjacency[index]))
    colors = np.full(problem.n, -1, dtype=np.int64)
    for index in order:
        forbidden = {colors[neighbor] for neighbor in problem.adjacency[index]}
        color = 0
        while color in forbidden:
            color += 1
        colors[index] = color
    return colors


def clique_witness(adjacency, upper_bound, deadline=None):
    """Return a verified lower-bound witness, with no claim of maximal size."""
    order = sorted(range(len(adjacency)), key=lambda index: (-len(adjacency[index]), index))
    best = [order[0]]
    for start in order:
        if len(best) == upper_bound or past_deadline(deadline):
            break
        clique = [start]
        candidates = set(adjacency[start])
        while len(clique) + len(candidates) > len(best) and candidates:
            index = max(
                candidates,
                key=lambda candidate: (
                    len(candidates & adjacency[candidate]),
                    len(adjacency[candidate]),
                    -candidate,
                ),
            )
            clique.append(index)
            candidates.intersection_update(adjacency[index])
        if len(clique) > len(best):
            best = clique
    return sorted(best)


class Objective:
    """Sparse pair and incidence arrays for fast evaluation of trial moves."""

    def __init__(self, problem: Problem, config: Config):
        self.problem = problem
        self.config = config
        node_indices = {node["id"]: i for i, node in enumerate(problem.data["nodes"])}
        incident = [[] for _ in node_indices]
        for edge_index, edge in enumerate(problem.data["hyperedges"]):
            for node_id in set(edge["nodes"]):
                incident[node_indices[node_id]].append(edge_index)
        self.incident = tuple(tuple(edges) for edges in incident)
        self.degrees = np.array([len(edges) for edges in self.incident], dtype=np.int64)
        self.shared_vertices = np.flatnonzero(self.degrees >= 2)
        groups = [self.incident[index] for index in self.shared_vertices]
        self.incidence_edges = np.fromiter((edge for group in groups for edge in group), dtype=int)
        self.group_offsets = np.cumsum(self.degrees[self.shared_vertices]) - self.degrees[self.shared_vertices]
        self.shared_degrees = self.degrees[self.shared_vertices]
        self.incidence_groups = np.repeat(np.arange(len(groups)), self.shared_degrees)

        pair_weights = problem.weights[problem.pair_i, problem.pair_j]
        active = pair_weights > 0
        self.pair_weights = pair_weights[active]
        self.shared_weight_sum = float(self.pair_weights.sum())
        self.left = problem.pair_i[active]
        self.right = problem.pair_j[active]
        self.quadratic_factor = 2.0 * problem.scale * {
            "pillar": 0.0,
            "squared": 1.0,
            "hybrid": config.alpha,
            "degree-span": 0.0,
        }[config.objective]
        self.use_pillars = config.objective != "squared"
        self.degree_factor = config.beta if config.objective == "degree-span" else 0.0

        self.constrained = tuple(i for i, neighbors in enumerate(problem.adjacency) if neighbors)
        self.constrained_degrees = [
            sum(bool(problem.adjacency[edge]) for edge in edges) for edges in self.incident
        ]
        self.free_components = self._free_components()
        self.coordinate_groups = [[] for _ in range(problem.n)]
        for group in groups:
            for edge in group:
                if not problem.adjacency[edge]:
                    others = np.array([member for member in group if member != edge], dtype=int)
                    self.coordinate_groups[edge].append(others)

    def _free_components(self):
        remaining = set(range(self.problem.n)).difference(self.constrained)
        components = []
        while remaining:
            first = min(remaining)
            remaining.remove(first)
            component = [first]
            for edge in component:
                neighbors = remaining.intersection(np.flatnonzero(self.problem.weights[edge]))
                component.extend(sorted(neighbors))
                remaining.difference_update(neighbors)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def shared_spans(self, layers):
        if not len(self.group_offsets):
            return np.empty(0, dtype=np.int64)
        incident_layers = layers[self.incidence_edges]
        return (np.maximum.reduceat(incident_layers, self.group_offsets)
                - np.minimum.reduceat(incident_layers, self.group_offsets))

    def value(self, layers):
        value = 0.0
        if self.use_pillars:
            spans = self.shared_spans(layers)
            value = float(spans.sum())
            if self.degree_factor:
                value += self.degree_factor * float(np.dot(self.shared_degrees, spans * spans))
        if self.quadratic_factor:
            differences = layers[self.left] - layers[self.right]
            energy = float(np.dot(self.pair_weights, differences * differences))
            if self.config.reg:
                # Sum of all unordered squared gaps, including zero shared weights.
                total = int(layers.sum())
                squared_gaps = self.problem.n * int(np.dot(layers, layers)) - total * total
                energy += self.config.reg * squared_gaps
            value += self.quadratic_factor * energy
        return value

    def coordinate_deltas(self, layers, edge, targets):
        """Evaluate every permissible target relative to the current integer layer."""
        old = int(layers[edge])
        values = np.zeros(len(targets), dtype=float)
        if self.use_pillars:
            for others in self.coordinate_groups[edge]:
                levels = layers[others]
                low, high = levels.min(), levels.max()
                spans = np.maximum(high, targets) - np.minimum(low, targets)
                old_span = max(high, old) - min(low, old)
                values += spans - old_span
                if self.degree_factor:
                    values += self.degree_factor * (len(others) + 1) * (spans ** 2 - old_span ** 2)
        if self.quadratic_factor:
            row_weights = self.problem.regularized[edge]
            values += self.quadratic_factor * (targets - old) * (
                (targets + old) * self.problem.row_sums[edge] - 2 * np.dot(row_weights, layers)
            )
        return values

    def metrics(self, layers):
        result = layout_metrics(self.problem, layers)
        spans = np.zeros(len(self.incident), dtype=np.int64)
        spans[self.shared_vertices] = self.shared_spans(layers)
        nodes = self.problem.data["nodes"]
        ranked = sorted(
            (index for index, degree in enumerate(self.constrained_degrees) if degree >= 2),
            key=lambda index: (-self.constrained_degrees[index], -self.degrees[index], index),
        )
        hubs = [
            {"node_id": nodes[index]["id"], "degree": int(self.degrees[index]),
             "constrained_degree": self.constrained_degrees[index], "span": int(spans[index])}
            for index in ranked[:5]
        ]
        result.update(
            layer_count=result["used_layer_count"],
            pillar_square_sum=int(np.dot(spans, spans)),
            degree_span_square_sum=int(np.dot(self.degrees, spans * spans)),
            top5_constrained_hub_mean_span=(sum(hub["span"] for hub in hubs) / len(hubs) if hubs else None),
            constrained_hubs=hubs,
        )
        if result["pillar_length"] != int(spans.sum()):
            raise RuntimeError("Pillar lengths differ between proposal and report evaluation.")
        return result

    def reported_value(self, metrics):
        pillar = metrics["pillar_length"]
        quadratic = metrics["l2_scaled_qp"]
        values = {
            "pillar": pillar,
            "squared": quadratic,
            "hybrid": pillar + self.config.alpha * quadratic,
            "degree-span": pillar + self.config.beta * metrics["degree_span_square_sum"],
        }
        return values[self.config.objective]


def insertion_permutation(layer_count, source, target):
    permutation = np.arange(layer_count)
    if source < target:
        permutation[source + 1:target + 1] -= 1
    else:
        permutation[target:source] += 1
    permutation[source] = target
    return permutation


def rectangle_sum(prefix, row_start, row_stop, col_start, col_stop):
    return ((prefix[row_stop, col_stop] - prefix[row_start, col_stop])
            - (prefix[row_stop, col_start] - prefix[row_start, col_start]))


class ShiftCosts:
    """Layer aggregates for insertions from one fixed, compact search state."""

    @staticmethod
    def supported(objective, layer_count):
        # The prefix sums are exact integers in float64. Bound their largest
        # intermediate and avoid a large dense layer table on sparse problems.
        return (
            layer_count ** 2 <= 250_000
            and layer_count * len(objective.shared_vertices) <= 1_000_000
            and 2 * objective.shared_weight_sum * max((layer_count - 1) ** 2, 2 * layer_count - 1) <= 2 ** 53
        )

    def __init__(self, objective, layers, layer_count):
        self.objective = objective
        self.layer_count = layer_count
        if objective.use_pillars and len(objective.shared_vertices):
            labels = layers[objective.incidence_edges]
            groups = objective.incidence_groups
            starts = objective.group_offsets
            self.low = np.minimum.reduceat(labels, starts)
            self.high = np.maximum.reduceat(labels, starts)
            self.second_low = np.minimum.reduceat(
                np.where(labels != self.low[groups], labels, layer_count), starts,
            )
            self.second_high = np.maximum.reduceat(
                np.where(labels != self.high[groups], labels, -1), starts,
            )
            self.member = np.zeros((layer_count, len(self.low)), dtype=bool)
            self.member[labels, groups] = True
            self.member[:, self.low == self.high] = False

        if objective.quadratic_factor:
            self.levels = np.arange(layer_count)
            self.population = np.bincount(layers, minlength=layer_count)
            bins = layers[objective.left] * layer_count + layers[objective.right]
            weight = np.bincount(
                bins, objective.pair_weights, minlength=layer_count ** 2,
            ).reshape(layer_count, layer_count)
            self.weight = weight + weight.T
            differences = self.levels[:, None] - self.levels
            self.shared_energy = float(np.sum(weight * differences ** 2))
            # For translating layer i by +1, (i + 1 - j)^2 - (i - j)^2
            # is 2(i - j) + 1. The transpose handles a translation by -1.
            self.crossing = np.zeros((layer_count + 1, layer_count + 1))
            weights = self.weight * (2 * differences + 1)
            self.crossing[1:, 1:] = weights.cumsum(0).cumsum(1)

    def value(self, permutation, source, target):
        objective = self.objective
        value = 0.0
        if objective.use_pillars and len(objective.shared_vertices):
            # The layer mapping is monotone except at source. For groups that
            # occupy source, first exclude it, map the extrema, then reinsert it.
            spans = permutation[self.high] - permutation[self.low]
            affected = self.member[source]
            low, high = self.low[affected], self.high[affected]
            other_low = np.where(low == source, self.second_low[affected], low)
            other_high = np.where(high == source, self.second_high[affected], high)
            spans[affected] = (
                np.maximum(target, permutation[other_high]) - np.minimum(target, permutation[other_low])
            )
            value = float(spans.sum())
            if objective.degree_factor:
                value += objective.degree_factor * float(np.dot(objective.shared_degrees, spans * spans))

        if objective.quadratic_factor:
            # Only pairs involving source and pairs crossing the boundary
            # of the uniformly translated interval can change their gap.
            before = source - self.levels
            after = target - permutation
            energy = self.shared_energy + float(np.dot(self.weight[source], after ** 2 - before ** 2))
            if source < target:
                translated_start, translated_stop = source + 1, target + 1
                outside_left, outside_right = source, target + 1
                crossing = self.crossing.T
            else:
                translated_start, translated_stop = target, source
                outside_left, outside_right = target, source + 1
                crossing = self.crossing
            energy += rectangle_sum(crossing, translated_start, translated_stop, 0, outside_left)
            energy += rectangle_sum(
                crossing, translated_start, translated_stop, outside_right, self.layer_count,
            )
            if objective.config.reg:
                total = int(np.dot(self.population, permutation))
                gaps = objective.problem.n * int(np.dot(self.population, permutation ** 2)) - total ** 2
                energy += objective.config.reg * gaps
            value += objective.quadratic_factor * energy
        return value


def past_deadline(deadline):
    return deadline is not None and time.perf_counter() >= deadline


def place_free_edges(objective, layers, cap, deadline=None):
    """Sweep shared-vertex components using exact single-coordinate minimizers."""
    layers = layers.copy()
    targets = np.arange(cap, dtype=np.int64)
    updates = 0
    while not past_deadline(deadline):
        moved = False
        for component in objective.free_components:
            # Uniform regularization couples even components with no shared vertices.
            # Reading all current coordinates makes each sequential update exact.
            for edge in component:
                if past_deadline(deadline):
                    return compact(layers), updates
                deltas = objective.coordinate_deltas(layers, edge, targets)
                target = int(np.argmin(deltas))
                if deltas[target] < -IMPROVEMENT_TOLERANCE:
                    layers[edge] = target
                    updates += 1
                    moved = True
        compacted = compact(layers)
        if not moved and np.array_equal(layers, compacted):
            break
        layers = compacted
    return compact(layers), updates


def propose(objective, layers, cap, move, rng):
    if cap < 2:
        return None
    if move in ("swap", "shift"):
        source, target = rng.sample(range(cap), 2)
        if move == "swap":
            permutation = np.arange(cap)
            permutation[[source, target]] = [target, source]
        else:
            permutation = insertion_permutation(cap, source, target)
        candidate = permutation[layers]
    else:
        if not objective.constrained:
            return None
        adjacency = objective.problem.adjacency
        edge = rng.choice(objective.constrained)
        source = layers[edge]
        target = rng.randrange(cap - 1)
        if target >= source:
            target += 1
        candidate = layers.copy()
        if move == "move":
            if any(layers[neighbor] == target for neighbor in adjacency[edge]):
                return None
            candidate[edge] = target
        else:
            chain = {edge}
            queue = deque([edge])
            while queue:
                index = queue.popleft()
                for neighbor in adjacency[index]:
                    if neighbor not in chain and layers[neighbor] in (source, target):
                        chain.add(neighbor)
                        queue.append(neighbor)
            indices = list(chain)
            candidate[indices] = source + target - layers[indices]
    return compact(candidate)


def check_layout(problem, layers, cap):
    if (layers.shape != (problem.n,) or layers.min() != 0 or layers.max() >= cap
            or not np.array_equal(layers, compact(layers))):
        raise RuntimeError("Invalid layout dimensions or noncompact layer coordinates.")
    if np.any(layers[problem.edges[:, 0]] == layers[problem.edges[:, 1]]):
        raise RuntimeError("Proposed layout violates overlap separation.")


def record_improvement(trajectory, layers, value, step, started_at):
    trajectory.append({
        "iteration": step,
        "elapsed_seconds": time.perf_counter() - started_at,
        "optimized_value": value,
        "used_layer_count": int(layers.max()) + 1,
    })


def better(layers, value, incumbent, incumbent_value):
    return value < incumbent_value - IMPROVEMENT_TOLERANCE or (
        value <= incumbent_value and layers.max() < incumbent.max()
    )


def search_stage(objective, initial, cap, budget, first_iteration, rng, started_at, deadline):
    config = objective.config
    stage_start = time.perf_counter()
    best = initial
    best_value = objective.value(best)
    iterations = updates = 0
    trajectory = []
    record_improvement(trajectory, best, best_value, first_iteration, started_at)

    if config.refine_free_edges:
        best, updates = place_free_edges(objective, best, cap, deadline)
        refined_value = objective.value(best)
        if better(best, refined_value, initial, best_value):
            record_improvement(trajectory, best, refined_value, first_iteration, started_at)
        best_value = refined_value

    current, current_value = best, best_value
    shift_costs = None
    shift_candidates = 0
    start_temperature = max(1e-3, best_value / math.sqrt(objective.problem.n))
    move_stats = {move: {"attempted": 0, "evaluated": 0, "accepted": 0} for move in config.moves}
    for index in range(budget):
        if past_deadline(deadline):
            break
        iterations += 1
        move = rng.choice(config.moves)
        stats = move_stats[move]
        stats["attempted"] += 1
        if move == "shift" and cap >= 2:
            source, target = rng.sample(range(cap), 2)
            layer_count = int(current.max()) + 1
            # Compaction removes holes above the occupied suffix. An empty
            # source preserves order; an empty target inserts at the end.
            target = min(target, layer_count - 1)
            if source >= layer_count or source == target:
                continue
            permutation = insertion_permutation(layer_count, source, target)
            candidate = permutation[current]
            shift_candidates += 1
            # Short-lived states use the direct evaluator. Building the
            # layer aggregates pays off only when a state is reused.
            if shift_candidates == SHIFT_CACHE_AFTER and ShiftCosts.supported(objective, layer_count):
                shift_costs = ShiftCosts(objective, current, layer_count)
            value = (objective.value(candidate) if shift_costs is None
                     else shift_costs.value(permutation, source, target))
        else:
            candidate = propose(objective, current, cap, move, rng)
            if candidate is None or np.array_equal(candidate, current):
                continue
            value = objective.value(candidate)
        stats["evaluated"] += 1
        delta = value - current_value
        temperature = start_temperature * 0.001 ** (index / max(1, budget - 1))
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            stats["accepted"] += 1
            current, current_value = candidate, value
            shift_costs = None
            shift_candidates = 0
            if better(current, current_value, best, best_value):
                best, best_value = current, current_value
                record_improvement(trajectory, best, best_value, first_iteration + iterations, started_at)

    if config.refine_free_edges and iterations:
        refined, last_updates = place_free_edges(objective, best, cap, deadline)
        updates += last_updates
        refined_value = objective.value(refined)
        if better(refined, refined_value, best, best_value):
            best, best_value = refined, refined_value
            record_improvement(trajectory, best, best_value, first_iteration + iterations, started_at)

    return finish_stage(objective, initial, best, best_value, cap, budget, deadline, {
        "iterations": iterations,
        "elapsed_seconds": time.perf_counter() - stage_start,
        "trajectory": trajectory,
        "moves": move_stats,
        "free_coordinate_updates": updates,
    })


def finish_stage(objective, initial, best, best_value, cap, budget, deadline, stats, timed_out=False):
    check_layout(objective.problem, best, cap)
    metrics = objective.metrics(best)
    reported_value = objective.reported_value(metrics)
    if not math.isclose(best_value, reported_value, rel_tol=1e-10, abs_tol=1e-8):
        raise RuntimeError("Proposal objective differs from the reported metric.")
    if timed_out or past_deadline(deadline):
        reason = "time_budget"
    else:
        reason = "iteration_budget" if objective.config.method == "anneal" else "initialization_only"
    return best, {
        "cap": cap,
        "initial_layers": initial.tolist(),
        "layers": best.tolist(),
        "iteration_budget": budget,
        "termination_reason": reason,
        "optimized_value": reported_value,
        "metrics": metrics,
        **stats,
    }


def solve(data, config=Config(), *, overlaps=None, started_at=None, backend="auto", native_library=None):
    """Return viewer JSON and metrics for every cap that reached initialization."""
    started_at = time.perf_counter() if started_at is None else started_at
    deadline = started_at + config.time_budget if config.time_budget else None
    problem = prepare_problem(data, reg=config.reg, overlaps=overlaps)
    objective = Objective(problem, config)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown search backend: {backend}")
    library = Path(native_library) if native_library is not None else DEFAULT_LIBRARY
    budget = config.max_iterations if config.method == "anneal" else 0
    if backend == "native" and budget > INT32_MAX:
        raise ValueError("Native search supports at most 2,147,483,647 proposals.")
    use_native = backend == "native" or (
        backend == "auto" and budget <= INT32_MAX
        and (native_library is not None or library.is_file())
    )
    if use_native and not library.is_file():
        raise FileNotFoundError(f"Native search library not found: {library}. Build it first.")
    native = NativeSearch(objective, MOVES, library) if use_native else None
    initial = initial_coloring(problem, config.method)
    colors = int(initial.max()) + 1
    base_cap = min(config.layer_cap or colors, problem.n)
    if base_cap < colors:
        raise ValueError(f"Cannot initialize at cap {base_cap}; {config.method} found {colors} colors.")
    last_cap = min(base_cap + config.warm_start_extra_layers, problem.n)
    caps = tuple(range(base_cap, last_cap + 1))
    iterations, stages = 0, []
    rng = random.Random(config.seed)
    each, remainder = divmod(budget, len(caps))
    initial_metrics = objective.metrics(initial)
    preprocessing_seconds = time.perf_counter() - started_at
    layers = initial
    for stage_index, cap in enumerate(caps):
        if stage_index and past_deadline(deadline):
            break
        stage_budget = each + int(stage_index < remainder)
        if native is None:
            layers, stage = search_stage(
                objective, layers, cap, stage_budget, iterations, rng, started_at, deadline,
            )
        else:
            best, stats, value, timed_out = native.run(
                layers, cap, stage_budget, iterations, rng, started_at, deadline,
            )
            layers, stage = finish_stage(
                objective, layers, best, value, cap, stage_budget, deadline, stats, timed_out,
            )
        stages.append(stage)
        iterations += stage["iterations"]
    final = stages[-1]
    output = {
        "nodes": problem.data["nodes"],
        "hyperedges": [
            {**edge, "layer": layer} for edge, layer in zip(problem.data["hyperedges"], final["layers"])
        ],
        "overlaps": problem.overlaps,
    }
    upper_bound = min(colors, *(stage["metrics"]["used_layer_count"] for stage in stages))
    witness = clique_witness(problem.adjacency, upper_bound, deadline)
    report = {
        "backend": "native" if use_native else "python",
        "settings": {**asdict(config), "optimize_free_edges": config.refine_free_edges},
        "quadratic_metric": "l2_scaled_qp",
        "problem_signature": problem.signature,
        "initial_layers": initial.tolist(),
        "initial_metrics": initial_metrics,
        "initial_color_count": colors,
        "dsatur_upper_bound": colors if config.method in ("dsatur", "anneal") else None,
        "clique_witness": witness,
        "clique_lower_bound": len(witness),
        "coloring_upper_bound": upper_bound,
        "proven_minimum_layers": upper_bound if len(witness) == upper_bound else None,
        "planned_caps": caps,
        "stages": stages,
        "iterations": iterations,
        "preprocessing_seconds": preprocessing_seconds,
        "elapsed_seconds": time.perf_counter() - started_at,
        "termination_reason": "time_budget" if past_deadline(deadline) else final["termination_reason"],
        "optimized_value": final["optimized_value"],
        "metrics": final["metrics"],
    }
    return output, report


def check_output_paths(input_path, output_path, report_path):
    paths = [path.resolve() for path in (input_path, output_path, report_path)]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, output and report must use distinct paths.")
    for path in (output_path, report_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Choose a new output path: {path}")


def save_results(files):
    """Serialize both payloads first and create a new file for each result."""
    contents = [
        (path, json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        for path, payload in files
    ]
    for path, _ in contents:
        path.parent.mkdir(parents=True, exist_ok=True)
    published = []
    try:
        for path, content in contents:
            with path.open("x", encoding="utf-8") as handle:
                published.append(path)
                handle.write(content)
    except BaseException:
        for path in reversed(published):
            path.unlink()
        raise


def parse_args(argv=None):
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--backend", choices=BACKENDS, default="auto",
                        help="Use the native search when a built library is available, or select a backend.")
    parser.add_argument("--native-library", type=Path, help="Path to a built C++ search library.")
    parser.add_argument("--method", choices=METHODS, default=defaults.method)
    parser.add_argument("--objective", choices=OBJECTIVES, default=defaults.objective)
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--reg", type=float, default=defaults.reg)
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations,
                        help="Total proposals across all caps, including rejected and no-op moves.")
    parser.add_argument("--time-budget", type=float, default=defaults.time_budget,
                        help="Cooperative wall-clock limit in seconds; 0 disables the deadline.")
    parser.add_argument("--layer-cap", type=int)
    parser.add_argument("--warm-start-extra-layers", choices=(0, 1, 2), type=int, default=0)
    parser.add_argument("--moves", choices=MOVES, nargs="+", default=MOVES)
    parser.add_argument("--optimize-free-edges", action=argparse.BooleanOptionalAction,
                        help="Refine unconstrained hyperedges (enabled by default for anneal).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings = vars(args).copy()
    for key in ("input", "output", "report", "backend", "native_library"):
        del settings[key]
    try:
        check_output_paths(args.input, args.output, args.report)
        config = Config(**{**settings, "moves": tuple(args.moves)})
        started_at = time.perf_counter()
        raw = args.input.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        output, report = solve(
            data, config, started_at=started_at, backend=args.backend, native_library=args.native_library,
        )
        report.update(input_path=str(args.input), input_sha256=sha256(raw).hexdigest())
        save_results(((args.output, output), (args.report, report)))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(
        f"{report['metrics']['used_layer_count']} layers; "
        f"objective {report['optimized_value']:.6g}; "
        f"{report['iterations']} proposals; {report['termination_reason']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
