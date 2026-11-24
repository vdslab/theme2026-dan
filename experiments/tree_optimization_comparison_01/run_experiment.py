"""Run the rooted-tree comparison sequentially from the repository root."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Configure numeric-library thread counts before imports initialize their runtimes.
THREAD_LIMITS = dict.fromkeys(
    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1"
)
os.environ.update(THREAD_LIMITS)

import numpy
import scipy
import shapely

from solve_layers_tree import (
    Config,
    atomic_write_json,
    check_deadline,
    initial_state,
    output_payload,
    prepare_problem,
)

from optimization import PENALTY, RESTART_FRACTIONS, RESTART_SEED_STEP, objective_values, run_search
from small_examples import exact_optima, fixtures


EXPERIMENT_ID = "tree_optimization_comparison_01"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path("experiments") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("results/experiments") / EXPERIMENT_ID
ORIGINAL = Path("original.json")
SEEDS = tuple(range(5))
MEASURE_SECONDS = 180.0
TOTAL_SECONDS = 7200.0
SMOKE_SECONDS = 5.0
SMALL_ITERATIONS = 250
TIME_CASE_ITERATION_GUARD = 1_000_000_000
FINALIZATION_SECONDS = 1.0
REG = 1e-5
ARMS = (
    ("stage1", "greedy", "squared"),
    ("stage1", "anneal", "squared"),
    ("stage1", "restart-greedy", "squared"),
    ("stage2", "anneal", "weighted-l1"),
    ("stage2", "anneal", "pillar"),
    ("stage2", "anneal", "squared-plus-layers"),
)
SOURCE_FILES = (
    SOURCE_DIR / "run_experiment.py",
    SOURCE_DIR / "optimization.py",
    SOURCE_DIR / "small_examples.py",
    Path("solve_layers_tree.py"),
    Path("solve_layers_dsatur_l1.py"),
    Path("solve_layers_dummy.py"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def output_lock(output):
    with (output / ".run.lock").open("ab+") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def git_bytes(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def provenance():
    paths = [path.as_posix() for path in (*SOURCE_FILES, ORIGINAL)]
    return {
        "git_commit": git_bytes("rev-parse", "HEAD").decode().strip(),
        "git_status": git_bytes("status", "--porcelain", "--", *paths).decode().splitlines(),
        "tracked_diff_sha256": hashlib.sha256(
            git_bytes("diff", "--binary", "HEAD", "--", *paths)
        ).hexdigest(),
        "source_sha256": {path.as_posix(): sha256(ROOT / path) for path in SOURCE_FILES},
        "environment": {
            "python_executable": sys.executable,
            "python_base_executable": sys._base_executable,
            "python_version": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "shapely": shapely.__version__,
            "geos": shapely.geos_version_string,
            "platform": platform.platform(),
            "thread_limits": THREAD_LIMITS,
        },
        "command": subprocess.list2cmdline([sys.executable, "-B", *sys.argv]),
        "working_directory": str(Path.cwd()),
    }


def case_record(phase, stage, fixture, method, objective, seed):
    arm = f"{method}-{objective}"
    folder = {
        "smoke": f"smoke/{arm}",
        "small": f"small/{fixture}/{arm}",
        "measure": f"{stage}/{arm}",
    }[phase]
    key = f"{folder}/seed-{seed}"
    return {
        "key": key,
        "phase": phase,
        "stage": stage,
        "fixture": fixture,
        "method": method,
        "objective": objective,
        "seed": seed,
        "path": key,
    }


def planned_cases():
    for stage, method, objective in ARMS:
        yield case_record("smoke", stage, "original", method, objective, 0)
    for fixture in fixtures():
        for stage, method, objective in ARMS:
            for seed in SEEDS:
                yield case_record("small", stage, fixture, method, objective, seed)
    for active_stage in ("stage1", "stage2"):
        for seed in SEEDS:
            for stage, method, objective in ARMS:
                if stage == active_stage:
                    yield case_record("measure", stage, "original", method, objective, seed)


def make_manifest():
    return {
        "experiment_id": EXPERIMENT_ID,
        "format_version": 1,
        "input_path": ORIGINAL.as_posix(),
        "input_base": "repository",
        "input_sha256": sha256(ROOT / ORIGINAL),
        "seeds": list(SEEDS),
        "reg": REG,
        "penalty": PENALTY,
        "penalty_rationale": (
            "100 units of normalized squared cost is about 2.9% of the previously "
            "reported 3493.207 squared cost. The term makes one fewer layer "
            "worth a squared-cost increase of up to 100."
        ),
        "objective_formulas": {
            "squared": "2 * scale * sum_{i<j} (shared(i,j) + reg) * (z_i-z_j)^2",
            "weighted-l1": "sum_{i<j} (shared(i,j) + reg) * abs(z_i-z_j)",
            "pillar": "sum_{v} (max_{i contains v} z_i - min_{i contains v} z_i)",
            "squared-plus-layers": "squared + penalty * len(set(z))",
        },
        "initialization": "Same deterministic BFS, squared-cost repair and compaction per input",
        "repair": "Existing squared-cost repair for all objectives",
        "solver_settings": asdict(Config()),
        "restart": {
            "fractions": list(RESTART_FRACTIONS),
            "max_restarts": len(RESTART_FRACTIONS),
            "seed_step": RESTART_SEED_STEP,
            "rule": (
                "Reset to the identical BFS + repair state and use seed + step * restart "
                "number at one-third and two-thirds of the case budget after the "
                "finalization reserve, or of attempts on small examples. Keep the global best."
            ),
        },
        "measure_case_budget_seconds": MEASURE_SECONDS,
        "measure_total_budget_seconds": TOTAL_SECONDS,
        "finalization_reserve_seconds": FINALIZATION_SECONDS,
        "timing_definition": (
            "The case clock starts before setup and stops after publishing report.json. "
            "Start and end of the setup, search and writes are measured on the same "
            "monotonic clock. The search stops one second before the case cap to reserve "
            "time for validating and publishing the final artifacts. "
            "Full case wall time is in progress.json."
        ),
        "thread_limits": THREAD_LIMITS,
        "time_case_iteration_guard": TIME_CASE_ITERATION_GUARD,
        "smoke_case_budget_seconds": SMOKE_SECONDS,
        "small_case_max_attempts": SMALL_ITERATIONS,
        "small_oracle": "small/oracle.json",
        "case_files": ["initial.json", "checkpoint.json", "layout.json", "report.json", "run.log"],
        "cases": list(planned_cases()),
        "reused_baselines": [
            {
                "phase": "measure",
                "stage": "stage2",
                "method": "anneal",
                "objective": "squared",
                "seed": seed,
                "source": f"stage1/anneal-squared/seed-{seed}",
                "source_case_key": f"stage1/anneal-squared/seed-{seed}",
            }
            for seed in SEEDS
        ],
        "historical_baselines": [
            {
                "report": "results/reports/layered_hypergraph_tree_report.json",
                "reused": False,
                "reason": "20 seconds / 1000 attempts / 4 calibration samples",
            },
            {
                "report": "results/reports/layers_hypergraph_report.json",
                "reused": False,
                "reason": "600 seconds / 20000 attempts",
            },
        ],
    }


def existing_or_write_plan(output, source):
    manifest_path = output / "manifest.json"
    manifest = make_manifest()
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved["plan"] != json.loads(json.dumps(manifest)):
            raise ValueError("Saved experiment conditions differ")
        if saved["provenance"]["source_sha256"] != source["source_sha256"]:
            raise ValueError("Experiment source differs from the saved manifest. Use --output.")
        return saved["plan"]
    atomic_write_json(manifest_path, {"plan": manifest, "provenance": source})
    return manifest


def prepare_small(output):
    oracle_path = output / "small" / "oracle.json"
    if oracle_path.exists():
        return {row["name"]: row for row in read_json(oracle_path)["fixtures"]}
    rows = {}
    for name, (data, expected_overlaps) in fixtures().items():
        # Exercise the same geometry path as the original input.
        problem = prepare_problem(data, REG)
        if list(problem.overlaps) != expected_overlaps:
            raise RuntimeError(f"{name} overlap geometry differs from the designed graph")
        record = Path("small") / name / "input.json"
        path = output / record
        if path.exists():
            if read_json(path) != data:
                raise ValueError(f"Existing fixture differs: {path}")
        else:
            atomic_write_json(path, data)
        rows[name] = {
            "name": name,
            "input_path": record.as_posix(),
            "input_sha256": sha256(path),
            "hyperedge_count": problem.n,
            "overlap_count": len(problem.overlaps),
            "overlaps": expected_overlaps,
            **exact_optima(data, expected_overlaps, REG, PENALTY),
        }
    atomic_write_json(
        oracle_path, {"reg": REG, "penalty": PENALTY, "fixtures": list(rows.values())}
    )
    return rows


def attempt_case(output, row, source, oracle, expected_sha256):
    before_case = time.monotonic()
    directory = output / row["path"]
    directory.mkdir(parents=True, exist_ok=False)
    phase = row["phase"]
    budget = {"smoke": SMOKE_SECONDS, "small": None, "measure": MEASURE_SECONDS}[phase]
    deadline = None if budget is None else before_case + budget - FINALIZATION_SECONDS
    config = Config(
        reg=REG,
        seed=row["seed"],
        method="anneal" if row["method"] == "anneal" else "greedy",
        max_iterations=SMALL_ITERATIONS if budget is None else TIME_CASE_ITERATION_GUARD,
        time_budget=budget or 0.0,
    )
    fixture = row["fixture"]
    input_path = Path(oracle[fixture]["input_path"]) if phase == "small" else ORIGINAL
    input_base = "output" if phase == "small" else "repository"
    input_file = (output if phase == "small" else ROOT) / input_path
    case_source = {
        **source,
        "input_base": input_base,
        "input_path": input_path.as_posix(),
        "input_sha256": sha256(input_file),
    }
    if case_source["input_sha256"] != expected_sha256:
        raise RuntimeError(f"Input changed since the conditions were recorded: {input_path}")

    settings = {
        **asdict(config),
        "penalty": PENALTY,
        "optimized_objective": row["objective"],
        "restart": row["method"] == "restart-greedy",
        "budget_seconds": budget,
        "finalization_reserve_seconds": FINALIZATION_SECONDS if budget else 0.0,
    }
    with (directory / "run.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                print(f"Started {utc_now()} {row['key']}", flush=True)
                prep_started = time.monotonic()
                problem = prepare_problem(read_json(input_file), REG, progress=print)
                expected_overlaps = oracle[fixture]["overlap_count"] if phase == "small" else 4431
                if len(problem.overlaps) != expected_overlaps:
                    raise RuntimeError("Unexpected number of overlap constraints")
                prep_seconds = time.monotonic() - prep_started
                check_deadline(deadline)
                before = time.monotonic()
                initial, initial_repair = initial_state(problem, config, deadline)
                init_seconds = time.monotonic() - before
                print(f"Initial metrics: {initial.metrics}", flush=True)

                def layout(best, summary, history):
                    return output_payload(
                        problem,
                        config,
                        initial,
                        best,
                        summary,
                        history,
                        provenance=case_source,
                    )

                def checkpoint(best, summary, history):
                    partial = layout(best, {**summary, "phase": "search"}, history)
                    atomic_write_json(directory / "checkpoint.json", partial)

                before = time.monotonic()
                initial_payload = layout(initial, {"phase": "initial"}, [])
                atomic_write_json(directory / "initial.json", initial_payload)
                atomic_write_json(directory / "checkpoint.json", initial_payload)
                initial_write_seconds = time.monotonic() - before
                best, search, history = run_search(
                    problem, initial, config, row["objective"], settings["restart"],
                    before_case, deadline, checkpoint,
                )
                exact = None if phase != "small" else oracle[fixture]["optima"][row["objective"]]["value"]
                best_values = objective_values(best.metrics)
                gap = None if exact is None else best_values[row["objective"]] - exact
                if gap is not None:
                    if gap < -1e-8:
                        raise RuntimeError(f"Result is below the exact minimum by {-gap}")
                    if abs(gap) < 1e-8:
                        gap = 0.0
                if best.metrics["overlap_violation_count"] != 0:
                    raise RuntimeError("Final case layout violates overlap constraints")
                before = time.monotonic()
                payload = layout(best, search, history)
                atomic_write_json(directory / "layout.json", payload)
                atomic_write_json(directory / "checkpoint.json", payload)
                final_write_seconds = time.monotonic() - before
                stop_reason = search["stop_reason"]
                print(f"Finished {utc_now()}. {stop_reason}, {search['attempts']} attempts.")
                print(f"Best metrics: {best.metrics}", flush=True)
            except (Exception, KeyboardInterrupt):
                traceback.print_exc(file=log)
                raise
    timing = {
        **search["timing_seconds"],
        "preprocessing": prep_seconds,
        "initialization": init_seconds,
        "initial_writes": initial_write_seconds,
        "final_layout_writes": final_write_seconds,
        "total": time.monotonic() - before_case,
    }
    report = {
        "case": {k: v for k, v in row.items() if k not in ("status", "previous_attempts")},
        "provenance": case_source,
        "settings": settings,
        "problem_signature": problem.signature,
        "overlap_count": len(problem.overlaps),
        "initial_metrics": initial.metrics,
        "initial_optimized_value": objective_values(initial.metrics)[row["objective"]],
        "initial_repair": {
            "initial_violations": initial_repair.initial_violations,
            "steps": initial_repair.steps,
        },
        "best_metrics": best.metrics,
        "all_objectives": best_values,
        "optimized_value": best_values[row["objective"]],
        "search": search,
        "timing_seconds": timing,
        "history": history,
        "small_exact_value": exact,
        "small_exact_gap": gap,
    }
    atomic_write_json(directory / "report.json", report)
    return stop_reason, search["attempts"], time.monotonic() - before_case


def load_progress(output, manifest, resume):
    path = output / "progress.json"
    if not path.exists():
        return {
            "experiment_id": EXPERIMENT_ID,
            "measurement_seconds": 0.0,
            "cases": [{**row, "status": "pending"} for row in manifest["cases"]],
        }
    progress = read_json(path)
    for row in progress["cases"]:
        if row["status"] != "running":
            continue
        if not resume:
            raise ValueError("An earlier invocation stopped inside a case. Use --resume.")
        row.update(status="interrupted", reason="previous_invocation_stopped_inside_case")
        if row["phase"] == "measure":
            # Charge the whole reserved budget when no final timing was persisted.
            progress["measurement_seconds"] += MEASURE_SECONDS
    return progress


def verified_cases(output, review_path):
    if review_path is None:
        return set()
    result = set()
    for row in read_json(review_path)["cases"]:
        if row["status"] != "verified":
            continue
        report = output / row["report"]
        layout = output / row["layout"]
        if (
            report.is_file()
            and layout.is_file()
            and row.get("report_sha256") == sha256(report)
            and row.get("layout_sha256") == sha256(layout)
        ):
            result.add((row["key"], row["path"]))
    return result


def choose_next_attempt(output, row):
    first = row["key"]
    if not (output / first).exists():
        return
    old = {k: row[k] for k in ("path", "status", "reason", "wall_seconds") if k in row}
    row.setdefault("previous_attempts", []).append(old)
    serial = 2
    while (output / first / f"attempt-{serial:02}").exists():
        serial += 1
    row["path"] = f"{first}/attempt-{serial:02}"
    for field in ("reason", "wall_seconds", "search_attempts"):
        row.pop(field, None)


def run_phase(phase, output, manifest, progress, source, resume, verified):
    rows = [row for row in progress["cases"] if row["phase"] == phase]
    if not resume and any(row["status"] != "pending" for row in rows):
        raise ValueError(f"{phase} already has attempts. Use --resume or a fresh --output.")
    if phase == "measure":
        preflight = [row for row in progress["cases"] if row["phase"] in ("smoke", "small")]
        if any(row["status"] != "completed" for row in preflight):
            raise ValueError("Complete the smoke run and small-case checks before measurement.")
    oracle = prepare_small(output) if phase == "small" else {}
    phase_start = time.monotonic()
    consumed = progress["measurement_seconds"]
    progress_path = output / "progress.json"

    def save():
        if phase == "measure":
            progress["measurement_seconds"] = consumed + time.monotonic() - phase_start
        progress["updated_at_utc"] = utc_now()
        atomic_write_json(progress_path, progress)

    for row in rows:
        if row["status"] == "completed" and (row["key"], row["path"]) in verified:
            continue
        remaining_budget = TOTAL_SECONDS - (consumed + time.monotonic() - phase_start)
        if phase == "measure" and remaining_budget < MEASURE_SECONDS:
            for remaining in rows:
                if remaining["status"] == "pending":
                    remaining.update(status="not_run", reason="insufficient_total_budget_for_full_case")
            progress["measurement_stop_reason"] = "total_budget_exhausted"
            save()
            return False
        choose_next_attempt(output, row)
        row["status"] = "running"
        save()
        print(f"[{utc_now()}] starting {row['path']}", flush=True)
        expected = (
            oracle[row["fixture"]]["input_sha256"] if phase == "small" else manifest["input_sha256"]
        )
        try:
            reason, attempts, elapsed = attempt_case(output, row, source, oracle, expected)
            finished = "interrupted" if reason == "interrupted" else "completed"
            if phase == "measure" and elapsed > MEASURE_SECONDS:
                finished, reason = "failed", "case_wall_clock_exceeded_cap"
            row.update(status=finished, reason=reason, search_attempts=attempts, wall_seconds=elapsed)
        except KeyboardInterrupt:
            row.update(status="interrupted", reason="keyboard_interrupt")
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        print(f"[{utc_now()}] {row['status']} {row['path']} ({row['reason']})", flush=True)
        save()
        if row["status"] == "interrupted" or (phase != "measure" and row["status"] == "failed"):
            return False
    if phase == "measure":
        progress["measurement_stop_reason"] = "completed_cases" if all(
            row["status"] == "completed" for row in rows
        ) else "cases_failed"
        save()
    return all(row["status"] == "completed" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "small", "measure", "all"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review", type=Path, help="Independent review with verified case checksums")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        source = provenance()
        manifest = existing_or_write_plan(output, source)
        progress = load_progress(output, manifest, args.resume)
        verified = verified_cases(output, args.review)
        phases = ("smoke", "small", "measure") if args.phase == "all" else (args.phase,)
        for phase in phases:
            if not run_phase(phase, output, manifest, progress, source, args.resume, verified):
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
