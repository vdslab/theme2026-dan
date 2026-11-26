"""Run the VisNotes remaining-validation plan with fixed, sequential budgets."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "experiments" / "visnotes_remaining_01"
THREAD_LIMITS = dict.fromkeys(
    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1",
)
XY_INPUTS = ("original", "paper", "scalable", "arranged")
OBJECTIVES = (
    "pillar", "squared", "degree-span", "hybrid-0.1", "hybrid-0.25", "hybrid-0.5", "hybrid-1",
    "hybrid-0.25-plus-1", "hybrid-0.25-plus-2",
)
BENCHMARKS = (
    "greedy-input", "greedy-size", "greedy-degree", "dsatur", "dsatur-free", "qp-l2", "lp-l1", "tree-sa",
    "swap", "swap-move", "swap-move-kempe", "full",
)


@dataclass(frozen=True)
class Phase:
    runner: str
    inputs: tuple
    total_budget: int
    case_budget: int | None = None
    conditions: tuple = ()
    seeds: tuple = (0, 1, 2, 3, 4)


PHASES = {
    "exp04": Phase("exp04_solver_benchmark", ("original",), 540, 35, BENCHMARKS),
    "exp05": Phase("exp05_xy_z_paradox_study", XY_INPUTS, 480, 35, ("full",)),
    "exp02": Phase("exp02_clean_benchmarks_prep", XY_INPUTS, 40),
    "exp03": Phase("exp03_objective_hub_pareto", ("original",), 720, 35, OBJECTIVES),
    "exp03-small": Phase(
        "exp03_objective_hub_pareto", ("four-path", "five-cycle", "six-mixed"),
        315, 35, OBJECTIVES, (0,),
    ),
    "exp04-legacy-reference": Phase("exp04_solver_benchmark", ("original",), 390, 180, ("qp-l2", "lp-l1")),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def command(phase, output):
    argv = [
        sys.executable, "-B", str(ROOT / "experiments" / phase.runner / "run_experiment.py"),
        "--output", str(output), "--total-budget", str(phase.total_budget), "--inputs", *phase.inputs,
    ]
    if phase.case_budget is not None:
        argv += [
            "--case-budget", str(phase.case_budget), "--max-iterations", "20000",
            "--seeds", *map(str, phase.seeds), "--conditions", *phase.conditions,
        ]
    return argv


def read_summary(directory):
    path = directory / "summary.json"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return None


def write_event(handle, event, **fields):
    payload = {"time_utc": utc_now(), "event": event, **fields}
    handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def phase_result(directory):
    summary = read_summary(directory)
    if summary is None:
        return {"status": "missing_summary"}
    return {
        "status": "complete" if summary["complete"] else "incomplete",
        "counts": summary["case_counts"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "error": summary.get("error"),
    }


def legacy_reference_conditions(output):
    summary = read_summary(output / "exp04")
    if summary is None:
        return None
    return tuple(
        case["condition"] for case in summary["cases"]
        if case["condition"] in PHASES["exp04-legacy-reference"].conditions and case["status"] != "verified"
    )


def run_phase(name, phase, output, handle):
    directory = output / name
    if directory.exists() or directory.is_symlink():
        result = phase_result(directory)
        write_event(handle, "existing", phase=name, **result)
        print(f"{name}: retaining existing output ({result['status']})", flush=True)
        return result["status"] == "complete"
    if name == "exp04-legacy-reference":
        wanted = legacy_reference_conditions(output)
        if not wanted:
            reason = "both_baselines_verified" if wanted == () else "requires_exp04_summary"
            write_event(handle, "skipped", phase=name, reason=reason)
            return wanted == ()
        phase = Phase(phase.runner, phase.inputs, phase.total_budget, phase.case_budget, wanted, phase.seeds)
    argv = command(phase, directory)
    write_event(handle, "started", phase=name, command=argv)
    print(f"Starting {name} at {directory}", flush=True)
    try:
        child_env = {**os.environ, **THREAD_LIMITS, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(argv, cwd=ROOT, env=child_env, check=False)
    except KeyboardInterrupt:
        write_event(handle, "interrupted", phase=name, **phase_result(directory))
        raise
    saved = phase_result(directory)
    write_event(handle, "finished", phase=name, exit_code=result.returncode, **saved)
    print(f"{name}: {saved['status']} (exit {result.returncode})", flush=True)
    if result.returncode == 130:
        raise KeyboardInterrupt
    return result.returncode == 0 and saved["status"] == "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory holding separate phase outputs")
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=tuple(PHASES))
    parser.add_argument("--plan-only", action="store_true", help="print the fixed commands without running")
    args = parser.parse_args(argv)
    if len(args.phases) != len(set(args.phases)):
        parser.error("phases must be unique")
    output = args.output.resolve()
    phases = {name: phase for name, phase in PHASES.items() if name in args.phases}
    plan = {
        "commands": {name: command(phase, output / name) for name, phase in phases.items()},
        "phase_order": list(phases),
        "thread_limits": THREAD_LIMITS,
        "article": "https://vdslab.esa.io/posts/9193",
        "launcher_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    journal = output / f"run-{stamp}.jsonl"
    with journal.open("x", encoding="utf-8", newline="\n") as handle:
        write_event(handle, "plan", **plan)
        complete = True
        try:
            for name, phase in phases.items():
                complete = run_phase(name, phase, output, handle) and complete
        except KeyboardInterrupt:
            write_event(handle, "stopped", reason="interrupted")
            return 130
        write_event(handle, "done", complete=complete)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
