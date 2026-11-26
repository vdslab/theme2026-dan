"""Cross-check the integer objective on existing four- to six-edge fixtures."""

from itertools import combinations, permutations
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from experiments.tree_optimization_comparison_01.review_results import evaluate, partitions
from experiments.visnotes_metrics import incidence_metrics, objective_value


def exhaustive_optimum(geometry, cap, objective, alpha, beta):
    best = None
    feasible_count = 0
    for blocks, used in partitions(geometry.size):
        if used > cap or any(blocks[i] == blocks[j] for i, j in geometry.overlaps):
            continue
        for ordering in permutations(range(used)):
            levels = tuple(ordering[block] for block in blocks)
            metrics = evaluate(geometry, levels)
            if objective == "degree-span":
                metrics.update(incidence_metrics(geometry, levels))
            value = objective_value(metrics, objective, alpha, beta)
            candidate = value, levels
            if best is None or candidate < best:
                best = candidate
            feasible_count += 1
    if best is None:
        raise ValueError(f"no feasible compact layout with cap {cap}")
    return {"feasible_count": feasible_count, "value": best[0], "layers": best[1]}


def milp_optimum(geometry, cap, objective, alpha, beta, time_budget):
    """Use assignment binaries and piecewise-linear integer square epigraphs."""
    size = geometry.size
    highest = cap - 1
    costs = [0.0] * (size + size * cap)
    upper = [highest] * size + [1.0] * (size * cap)
    integrality = [1] * len(costs)
    rows, columns, values = [], [], []
    lower_limits, upper_limits = [], []

    def add_row(coefficients, low=-math.inf, high=math.inf):
        for index, coefficient in coefficients:
            if coefficient:
                rows.append(len(lower_limits))
                columns.append(index)
                values.append(coefficient)
        lower_limits.append(low)
        upper_limits.append(high)

    def add_variable(bound, cost=0.0):
        index = len(costs)
        costs.append(cost)
        upper.append(bound)
        integrality.append(0)
        return index

    def assignment(edge, height):
        return size + edge * cap + height

    for i in range(size):
        add_row([(assignment(i, k), 1) for k in range(cap)], low=1, high=1)
        add_row([(i, -1), *((assignment(i, k), k) for k in range(cap))], low=0, high=0)
    for i, j in geometry.overlaps:
        for k in range(cap):
            add_row([(assignment(i, k), 1), (assignment(j, k), 1)], high=1)

    # Secants through adjacent integer points give the exact quadratic value
    # on the integer heights, including at either endpoint.
    squared_weight = 1.0 if objective == "squared" else alpha if objective == "hybrid" else 0.0
    if squared_weight:
        for i, j, shared in geometry.pair_counts:
            cost = 2.0 * geometry.scale * (shared + geometry.reg) * squared_weight
            if cost <= 0:
                continue
            square = add_variable(highest ** 2, cost)
            for gap in range(-highest, highest):
                slope = 2 * gap + 1
                add_row([(i, slope), (j, -slope), (square, -1)], high=gap * (gap + 1))

    if objective != "squared":
        for incident in geometry.incident:
            if len(incident) < 2:
                continue
            span = add_variable(highest, 1.0)
            for i, j in combinations(incident, 2):
                add_row([(i, 1), (j, -1), (span, -1)], high=0)
                add_row([(i, -1), (j, 1), (span, -1)], high=0)
            if objective == "degree-span" and beta:
                square = add_variable(highest ** 2, beta * len(incident))
                for gap in range(highest):
                    add_row([(span, 2 * gap + 1), (square, -1)], high=gap * (gap + 1))

    matrix = coo_matrix((values, (rows, columns)), shape=(len(lower_limits), len(costs))).tocsc()
    result = milp(
        costs, integrality=integrality, bounds=Bounds(np.zeros(len(costs)), upper),
        constraints=LinearConstraint(matrix, lower_limits, upper_limits),
        options={"time_limit": time_budget, "mip_rel_gap": 0.0},
    )
    info = {"status": int(result.status), "message": str(result.message)}
    dual_bound = getattr(result, "mip_dual_bound", None)
    gap = getattr(result, "mip_gap", None)
    info["lower_bound"] = float(dual_bound) if dual_bound is not None and math.isfinite(dual_bound) else None
    info["solver_relative_gap"] = float(gap) if gap is not None and math.isfinite(gap) else None
    if result.success:
        integer_levels = tuple(round(float(value)) for value in result.x[:size])
        if any(
            not 0 <= height < cap or abs(float(raw) - height) > 1e-6
            for height, raw in zip(integer_levels, result.x[:size])
        ) or any(integer_levels[i] == integer_levels[j] for i, j in geometry.overlaps):
            raise ValueError("MILP returned an invalid layout")
        ranks = {height: index for index, height in enumerate(sorted(set(integer_levels)))}
        levels = tuple(ranks[height] for height in integer_levels)
        metrics = evaluate(geometry, levels)
        if objective == "degree-span":
            metrics.update(incidence_metrics(geometry, levels))
        value = objective_value(metrics, objective, alpha, beta)
        if not math.isclose(value, float(result.fun), rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError("recomputed objective differs from MILP objective")
        info.update(value=value, layers=levels, reported_value=float(result.fun))
    return info


def verify_optimum(geometry, cap, objective, alpha, beta, time_budget):
    if not 1 <= geometry.size <= 6:
        raise ValueError("the exhaustive oracle is for layouts with at most six hyperedges")
    exact = exhaustive_optimum(geometry, cap, objective, alpha, beta)
    highs = milp_optimum(geometry, cap, objective, alpha, beta, time_budget)
    if highs["status"] not in (0, 1):
        raise ValueError(f"HiGHS failed for an exhaustive problem with known feasible assignments: {highs['message']}")
    if highs["status"] == 0 and not math.isclose(exact["value"], highs["value"], rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError(f"exhaustive enumeration and MILP disagree at cap {cap}")
    return {"enumeration": exact, "milp": highs}
