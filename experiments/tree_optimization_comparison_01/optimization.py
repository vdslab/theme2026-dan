"""Experiment-specific objectives and budgeted search using the tree solver."""

from datetime import datetime, timezone
import math
import random
from statistics import median
import time

from solve_layers_tree import (
    DeadlineExceeded,
    MOVE_TYPES,
    RepairFailed,
    accept_move,
    check_deadline,
    evaluate_proposal,
    propose,
    validate_state,
)


OBJECTIVES = ("squared", "weighted-l1", "pillar", "squared-plus-layers")
PENALTY = 100.0
RESTART_FRACTIONS = (1 / 3, 2 / 3)
RESTART_SEED_STEP = 0x9E3779B97F4A7C15


def objective_values(metrics, penalty=PENALTY):
    squared = metrics["l2_scaled_qp"]
    return {
        "squared": squared,
        "weighted-l1": metrics["l1_regularized"],
        "pillar": metrics["pillar_length"],
        "squared-plus-layers": squared + penalty * metrics["used_layer_count"],
    }


def calibrated_temperature(problem, initial, config, value, deadline):
    if config.method == "greedy":
        return 0.0, {"samples": 0, "positive_samples": 0, "source": "greedy"}

    rng = random.Random(config.seed ^ 0x75A94E)
    baseline = value(initial)
    positive = []
    evaluated = 0
    for _ in range(config.temperature_samples):
        check_deadline(deadline)
        proposal, _ = propose(problem, initial, rng, config)
        if proposal is None:
            continue
        try:
            candidate, _, _ = evaluate_proposal(
                problem, proposal, config, "temperature-sample", deadline
            )
        except RepairFailed:
            continue
        evaluated += 1
        delta = value(candidate) - baseline
        if delta > 0:
            positive.append(delta)
    if positive:
        temperature = -median(positive) / math.log(config.initial_acceptance)
        source = "sampled"
    else:
        temperature = max(1.0, abs(baseline) * 0.01)
        source = "fallback_no_positive_deltas"
    return temperature, {
        "samples": evaluated,
        "positive_samples": len(positive),
        "source": source,
    }


def run_search(
    problem,
    initial,
    config,
    optimized_objective,
    restart,
    case_start,
    deadline,
    checkpoint,
):
    config.validate()
    validate_state(problem, initial)

    def value(state):
        return objective_values(state.metrics)[optimized_objective]

    summary = {
        "attempts": 0,
        "accepted": 0,
        "uphill_accepted": 0,
        "equal_cost_tree_changes": 0,
        "best_improvements": 0,
        "repair_steps": 0,
        "repair_failures": 0,
        "returned_to_current": 0,
        "repair_returned_to_current": 0,
        "restarts": [],
        "temperature": {
            "source": "not_started",
            "initial": None,
            "samples": 0,
            "positive_samples": 0,
            "final_ratio": config.final_temperature_ratio,
        },
        "timing_seconds": dict.fromkeys(
            (
                "calibration",
                "generation",
                "repair",
                "rebuilding",
                "evaluation",
                "restarts",
                "saving",
                "search",
            ),
            0.0,
        ),
        "by_move": {
            kind: dict.fromkeys(
                ("attempts", "evaluated", "accepted", "improvements", "repair_steps"), 0
            )
            for kind in MOVE_TYPES
        },
    }
    timings = summary["timing_seconds"]
    current = best = initial
    current_value = best_value = value(initial)
    history = []

    def remember(attempt, kind):
        history.append(
            {
                "attempt": attempt,
                "move_type": kind,
                "elapsed_seconds": time.monotonic() - case_start,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "objective": best_value,
                "used_layer_count": best.metrics["used_layer_count"],
                "height_range": best.metrics["height_range"],
                "pillar_length": best.metrics["pillar_length"],
            }
        )

    remember(0, "initial")
    generation_rng = random.Random(config.seed)
    acceptance_rng = random.Random(config.seed ^ 0x128ABD)
    last_checkpoint = time.monotonic()
    restart_schedule = RESTART_FRACTIONS if restart else ()
    max_iterations = config.max_iterations
    stop_reason = "max_iterations_reached"
    search_started_at = None
    try:
        calibration_start = time.monotonic()
        temperature, temperature_info = calibrated_temperature(
            problem, initial, config, value, deadline
        )
        timings["calibration"] = time.monotonic() - calibration_start
        summary["temperature"] = {
            **temperature_info,
            "initial": temperature,
            "final_ratio": config.final_temperature_ratio,
        }
        search_started_at = time.monotonic()

        for attempt in range(1, max_iterations + 1):
            check_deadline(deadline)
            now = time.monotonic()
            if now - last_checkpoint >= config.checkpoint_interval:
                before = time.monotonic()
                checkpoint(best, summary, history)
                timings["saving"] += time.monotonic() - before
                last_checkpoint = time.monotonic()
                check_deadline(deadline)

            if deadline is None:
                fraction = (attempt - 1) / max_iterations
            else:
                fraction = (now - case_start) / (deadline - case_start)
            if restart_schedule and fraction >= restart_schedule[0]:
                before = time.monotonic()
                restart_number = len(summary["restarts"]) + 1
                restart_seed = config.seed + restart_number * RESTART_SEED_STEP
                generation_rng = random.Random(restart_seed)
                summary["restarts"].append(
                    {
                        "before_attempt": attempt,
                        "elapsed_seconds": before - case_start,
                        "seed": restart_seed,
                        "previous_current_objective": current_value,
                        "best_objective": best_value,
                    }
                )
                restart_schedule = restart_schedule[1:]
                current, current_value = initial, value(initial)
                timings["restarts"] += time.monotonic() - before

            summary["attempts"] = attempt
            before = time.monotonic()
            proposal, kind = propose(problem, current, generation_rng, config)
            timings["generation"] += time.monotonic() - before
            counts = summary["by_move"][kind]
            counts["attempts"] += 1
            if proposal is None:
                continue

            before = time.monotonic()
            try:
                candidate, repair, timing = evaluate_proposal(
                    problem, proposal, config, f"state-{attempt:06d}", deadline
                )
            except RepairFailed:
                summary["repair_failures"] += 1
                timings["repair"] += time.monotonic() - before
                continue
            for key, elapsed in timing.items():
                timings[key] += elapsed
            counts["evaluated"] += 1
            counts["repair_steps"] += len(repair.steps)
            summary["repair_steps"] += len(repair.steps)
            if candidate.layers == current.layers and candidate.tree == current.tree:
                summary["returned_to_current"] += 1
                summary["repair_returned_to_current"] += bool(repair.steps)
                continue

            progress = (attempt - 1) / max(1, max_iterations - 1)
            if deadline is not None:
                time_fraction = (time.monotonic() - search_started_at) / max(
                    deadline - search_started_at, 1e-12
                )
                progress = max(progress, min(1.0, time_fraction))
            current_temperature = temperature * config.final_temperature_ratio**progress
            candidate_value = value(candidate)
            delta = candidate_value - current_value
            if not accept_move(delta, current_temperature, acceptance_rng, config.method):
                continue

            summary["accepted"] += 1
            summary["uphill_accepted"] += delta > 0
            summary["equal_cost_tree_changes"] += candidate.layers == current.layers
            counts["accepted"] += 1
            counts["improvements"] += delta < 0
            current, current_value = candidate, candidate_value
            if current_value < best_value - max(1e-9, abs(best_value) * 1e-12):
                validate_state(problem, current)
                best, best_value = current, current_value
                summary["best_improvements"] += 1
                remember(attempt, kind)
    except DeadlineExceeded:
        stop_reason = "time_budget_exhausted"
    except KeyboardInterrupt:
        stop_reason = "interrupted"

    if search_started_at is not None:
        timings["search"] = time.monotonic() - search_started_at
    else:
        timings["calibration"] = time.monotonic() - calibration_start
    attempts = summary["attempts"]
    for counts in summary["by_move"].values():
        counts["acceptance_rate"] = counts["accepted"] / max(1, counts["attempts"])
    summary.update(
        stop_reason=stop_reason,
        initial_objective=value(initial),
        best_objective=best_value,
        current_objective=current_value,
        returned_to_current_ratio=summary["returned_to_current"] / max(1, attempts),
        repair_returned_to_current_ratio=(
            summary["repair_returned_to_current"] / max(1, attempts)
        ),
    )
    validate_state(problem, best)
    validate_state(problem, current)
    return best, summary, history
