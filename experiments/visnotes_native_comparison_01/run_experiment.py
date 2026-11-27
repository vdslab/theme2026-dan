"""Measure and verify the Python and native compact layer solver backends."""

import argparse
from dataclasses import asdict, replace
from itertools import pairwise
from math import isclose
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import (
    COMPACT, REG, checked_compact_report, file_hash, make_geometry,
    manifest_sources, read_json, root_relative, save_json, solver_command,
)
from experiments.visnotes_metrics import projection_metrics, read_layout_metrics

EXPERIMENT = "visnotes_native_comparison_01"
SOURCE = ROOT / "original.json"
PROFILES = {"fixed": COMPACT, "warm": replace(COMPACT, warm_start=2)}
SOLVER_SECONDS = 25.0
WALL_SECONDS = 45.0
TOTAL_SECONDS = 600.0


def run_case(case, directory, geometry, trials, library, deadline):
    directory.mkdir(parents=True)
    method = PROFILES[case["profile"]]
    backend = case["backend"]
    command = solver_command(method, SOURCE, directory, case["seed"], trials, SOLVER_SECONDS)
    command += ["--backend", backend]
    if backend == "native":
        command += ["--native-library", str(library)]
    record = dict(case, command=command, case_directory=root_relative(directory))
    started = monotonic()
    try:
        with (directory / "run.log").open("xb") as log:
            subprocess.run(
                command, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=max(0.001, min(WALL_SECONDS, deadline - started)),
            )
        elapsed = monotonic() - started
        layout = read_json(directory / "layout.json")
        levels, metrics = read_layout_metrics(layout, geometry, case["condition"])
        report = read_json(directory / "report.json")
        stages, all_caps, _ = checked_compact_report(
            report, geometry, case, method, levels, metrics, deadline, {}, trials, SOLVER_SECONDS, elapsed,
        )
        if report["backend"] != backend:
            raise ValueError("The reported backend differs from the requested backend")
        if not all_caps or report["iterations"] != trials or report["termination_reason"] != "iteration_budget":
            raise ValueError("The solver did not complete the planned trial budget and caps")
        for stage in report["stages"]:
            for stats in stage["moves"].values():
                if not 0 <= stats["accepted"] <= stats["evaluated"] <= stats["attempted"]:
                    raise ValueError("Invalid move counters")
            if sum(s["attempted"] for s in stage["moves"].values()) != stage["iterations"]:
                raise ValueError("Move counters do not sum to the number of attempts")
        record.update(
            status="verified", metrics=metrics, projection=projection_metrics(geometry, levels),
            solver_wall_seconds=elapsed, solver_seconds=report["elapsed_seconds"],
            preprocessing_seconds=report["preprocessing_seconds"],
            stage_seconds=sum(stage["elapsed_seconds"] for stage in stages),
            iterations=report["iterations"], termination_reason=report["termination_reason"],
            optimized_value=report["optimized_value"], stages=stages,
        )
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        record.update(status="failed", error=str(error), solver_wall_seconds=monotonic() - started)
        if isinstance(error, subprocess.TimeoutExpired):
            record["status"] = "timeout"
    save_json(directory / "case.json", record)
    print(case["condition"], case["seed"], record["status"], flush=True)
    return record


def compare(records):
    pairs = []
    for first, second in pairwise(records):
        if first["profile"] != second["profile"] or first["seed"] != second["seed"]:
            continue
        if first["status"] != "verified" or second["status"] != "verified":
            continue
        python_case, native_case = (first, second) if first["backend"] == "python" else (second, first)
        python, native = (
            read_json(ROOT / record["case_directory"] / "report.json")
            for record in (python_case, native_case)
        )
        exact_stages = [
            left["layers"] == right["layers"]
            for left, right in zip(python["stages"], native["stages"], strict=True)
        ]
        equal_objectives = [
            isclose(left["optimized_value"], right["optimized_value"], rel_tol=1e-9, abs_tol=1e-8)
            for left, right in zip(python["stages"], native["stages"], strict=True)
        ]
        pairs.append({
            "profile": first["profile"], "seed": first["seed"],
            "same_initial": python["initial_layers"] == native["initial_layers"],
            "same_caps": python["planned_caps"] == native["planned_caps"],
            "same_stage_layouts": exact_stages, "same_stage_objectives": equal_objectives,
            "same_stage_move_counts": [
                left["moves"] == right["moves"]
                for left, right in zip(python["stages"], native["stages"], strict=True)
            ],
            "native_minus_python_objective": native["optimized_value"] - python["optimized_value"],
            "wall_speedup": python_case["solver_wall_seconds"] / native_case["solver_wall_seconds"],
            "stage_speedup": python_case["stage_seconds"] / native_case["stage_seconds"],
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-library", required=True, type=Path)
    parser.add_argument("--compiler", default="g++", help="compiler identification in the manifest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    library = args.native_library.resolve(strict=True)
    library_hash = file_hash(library)
    trials, seeds = (100, (0,)) if args.smoke else (20_000, range(5))
    profiles = [("fixed", seed) for seed in seeds] + [("warm", 0)]
    cases = [
        {"input": "original", "condition": f"{profile}-{backend}", "seed": seed,
         "profile": profile, "backend": backend}
        for index, (profile, seed) in enumerate(profiles)
        for backend in (("python", "native") if index % 2 == 0 else ("native", "python"))
    ]
    output.mkdir(parents=True, exist_ok=False)
    deadline = monotonic() + TOTAL_SECONDS
    geometry = make_geometry(SOURCE, REG)
    provenance = manifest_sources("benchmark", (COMPACT,))
    extra_sources = [Path(__file__), *(ROOT / "native" / f"compact_search.{suffix}" for suffix in ("cpp", "h", "py"))]
    provenance["source_sha256"].update({root_relative(path.resolve()): file_hash(path) for path in extra_sources})
    provenance["relevant_git_status"] = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *provenance["source_sha256"]],
        cwd=ROOT, text=True,
    ).splitlines()
    save_json(output / "manifest.json", {
        "experiment_id": EXPERIMENT, "input": root_relative(SOURCE), **geometry.hashes,
        "cases": cases, "profiles": {name: asdict(method) for name, method in PROFILES.items()},
        "trial_limit": trials, "regularization": REG, "solver_budget_seconds": SOLVER_SECONDS,
        "case_wall_limit_seconds": WALL_SECONDS, "total_budget_seconds": TOTAL_SECONDS,
        "native_library": str(library), "native_library_sha256": library_hash,
        "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
        "provenance": provenance,
    })
    records = []
    try:
        for case in cases:
            if monotonic() >= deadline:
                break
            directory = output / case["condition"] / f"seed-{case['seed']}"
            records.append(run_case(case, directory, geometry, trials, library, deadline))
    except KeyboardInterrupt:
        pass
    records.extend(dict(case, status="pending") for case in cases[len(records):])
    unchanged = all(
        file_hash(ROOT / path) == digest for path, digest in provenance["source_sha256"].items()
    ) and file_hash(SOURCE) == geometry.hashes["input_sha256"] and file_hash(library) == library_hash
    pairs = compare(records) if unchanged else []
    complete = len(pairs) == len(profiles) and all(p["same_initial"] and p["same_caps"] for p in pairs)
    fixed = [pair for pair in pairs if pair["profile"] == "fixed"]
    save_json(output / "summary.json", {
        "complete": complete, "sources_unchanged": unchanged, "cases": records, "pairs": pairs,
        "median_fixed_wall_speedup": median(pair["wall_speedup"] for pair in fixed) if fixed else None,
        "median_fixed_stage_speedup": median(pair["stage_speedup"] for pair in fixed) if fixed else None,
    })
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
