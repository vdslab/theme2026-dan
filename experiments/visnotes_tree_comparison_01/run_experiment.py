"""Compare the rooted-tree and direct solvers under a common budget."""

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import (
    COMPACT, REG, Method, checked_compact_report, file_hash, make_geometry,
    manifest_sources, read_json, root_relative, save_json, solver_command, summarize,
)
from experiments.visnotes_metrics import check_report_metrics, projection_metrics, read_layout_metrics

EXPERIMENT = "visnotes_tree_comparison_01"
SOURCE = ROOT / "original.json"
METHODS = {
    "tree-squared": Method("solve_layers_tree.py", objective="squared"),
    "compact-squared": replace(COMPACT, objective="squared"),
    "compact-hybrid": COMPACT,
}
SOLVER_SECONDS = 25.0
WALL_SECONDS = 45.0
TOTAL_SECONDS = 480.0


def run_case(case, method, directory, geometry, trials, deadline):
    seed = case["seed"]
    directory.mkdir(parents=True)
    command = solver_command(method, SOURCE, directory, seed, trials, SOLVER_SECONDS)
    record = dict(case, command=command, case_directory=root_relative(directory))
    start = monotonic()
    try:
        with (directory / "run.log").open("wb") as log:
            subprocess.run(
                command, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=min(WALL_SECONDS, deadline - start),
            )
        elapsed = monotonic() - start
        layout = read_json(directory / "layout.json")
        levels, metrics = read_layout_metrics(layout, geometry, case["condition"])
        report = read_json(directory / "report.json")
        if method.solver == COMPACT.solver:
            stages, complete, _ = checked_compact_report(
                report, geometry, case, method, levels, metrics, deadline, {}, trials,
                SOLVER_SECONDS, elapsed,
            )
            if not complete:
                raise ValueError("Incomplete compact stages")
            history = stages[-1]["trajectory"]
            iterations, end = report["iterations"], report["termination_reason"]
        else:
            check_report_metrics(report["best_metrics"], metrics, case["condition"])
            settings = report["settings"]
            if (settings["method"], settings["seed"], settings["reg"], settings["time_budget"],
                settings["max_iterations"]) != ("anneal", seed, REG, SOLVER_SECONDS, trials):
                raise ValueError("Tree report settings differ from the case")
            if report["best"]["layers"] != list(levels) or report["provenance"]["input_sha256"] != geometry.hashes["input_sha256"]:
                raise ValueError("Tree report does not match the saved layout or input")
            history = report["history"]
            iterations, end = report["summary"]["attempts"], report["summary"]["stop_reason"]
        if not 0 <= iterations <= trials:
            raise ValueError("Reported iterations exceed the trial budget")
        record.update(
            status="verified", metrics=metrics, projection=projection_metrics(geometry, levels),
            solver_wall_seconds=elapsed, iterations=iterations, termination_reason=end, trajectory=history,
        )
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        record.update(
            status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
            error=str(error), solver_wall_seconds=monotonic() - start,
        )
    save_json(directory / "case.json", record)
    print(case["condition"], seed, record["status"], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    deadline = monotonic() + TOTAL_SECONDS
    seeds, trials = ((0,), 10) if args.smoke else (range(5), 20_000)
    cases = [
        {"input": "original", "condition": name, "seed": seed}
        for seed in seeds for name in METHODS
    ]
    output.mkdir(parents=True, exist_ok=False)
    geometry = make_geometry(SOURCE, REG)
    provenance = manifest_sources("benchmark", METHODS.values())
    provenance["source_sha256"][root_relative(Path(__file__).resolve())] = file_hash(Path(__file__))
    save_json(output / "manifest.json", {
        "experiment_id": EXPERIMENT, "input": root_relative(SOURCE), **geometry.hashes,
        "cases": cases, "methods": {name: asdict(method) for name, method in METHODS.items()},
        "trial_limit": trials, "regularization": REG, "solver_budget_seconds": SOLVER_SECONDS,
        "case_wall_limit_seconds": WALL_SECONDS, "total_budget_seconds": TOTAL_SECONDS,
        "provenance": provenance,
    })
    records = []
    try:
        for case in cases:
            if monotonic() >= deadline:
                break
            name = case["condition"]
            records.append(run_case(case, METHODS[name], output / name / f"seed-{case['seed']}", geometry, trials, deadline))
    except KeyboardInterrupt:
        pass
    records.extend(dict(case, status="pending") for case in cases[len(records):])
    unchanged = file_hash(SOURCE) == geometry.hashes["input_sha256"] and all(
        file_hash(ROOT / path) == digest for path, digest in provenance["source_sha256"].items()
    )
    complete = unchanged and all(record["status"] == "verified" for record in records)
    save_json(output / "summary.json", {
        "complete": complete, "sources_unchanged": unchanged, "cases": records,
        "comparisons": summarize(records) if unchanged else [],
    })
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
