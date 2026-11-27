"""Compare layer assignment on public inputs and separate the warm-start budgets."""

import argparse
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path
from time import monotonic
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import (
    COMPACT, CONDITIONS, OBJECTIVES, REG, cap_frontiers, file_hash, input_record,
    make_geometry, manifest_sources, read_json, root_relative, run_case, save_json, summarize,
)
from native.compact_search import DEFAULT_LIBRARY


EXPERIMENT = "visnotes_public_validation_01"
SEEDS = tuple(range(5))
BENCHMARK = {
    **CONDITIONS["benchmark"],
    **OBJECTIVES,
}
WARM = {
    "fixed": (COMPACT, 1),
    **{
        name: (replace(COMPACT, warm_start=extra) if name.startswith("warm") else COMPACT, factor)
        for extra in (1, 2)
        for name, factor in (
            (f"warm-fixed-total-{extra}", 1),
            (f"warm-extended-total-{extra}", extra + 1),
            (f"fixed-extended-total-{extra}", extra + 1),
        )
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("benchmark", "warm", "xy"), required=True)
    parser.add_argument("--input", action="append", nargs=2, metavar=("NAME", "JSON"), required=True)
    parser.add_argument("--input-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-budget", type=float, default=35)
    parser.add_argument("--total-budget", type=float, default=1800)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--condition", action="append", help="Run a subset of the selected phase.")
    args = parser.parse_args(argv)
    valid = "abcdefghijklmnopqrstuvwxyz0123456789-"
    names = [name for name, _ in args.input]
    if len(set(names)) != len(names) or any(
        not name or any(letter not in valid for letter in name) for name in names
    ):
        parser.error("Input names must be unique and contain only lowercase letters, digits and hyphens.")
    if any(not isfinite(limit) or limit <= 0 for limit in (args.case_budget, args.total_budget)):
        parser.error("Budgets must be finite and positive.")
    args.seed = tuple(args.seed) if args.seed is not None else SEEDS
    if min(args.seed) < 0 or len(set(args.seed)) != len(args.seed) or args.max_iterations < 1:
        parser.error("Enter unique nonnegative seeds and a positive iteration limit.")
    return args


def provenance_for(conditions):
    provenance = manifest_sources("benchmark", tuple(method for method, _ in conditions.values()))
    files = [
        Path(__file__).resolve(), ROOT / "native/compact_search.py", ROOT / "native/compact_search.cpp",
        ROOT / "native/compact_search.h",
    ]
    if DEFAULT_LIBRARY.is_file():
        files.append(DEFAULT_LIBRARY)
    provenance["source_sha256"].update({root_relative(file): file_hash(file) for file in files})
    return provenance


def run(args):
    conditions = {
        "benchmark": {label: (method, 1) for label, method in BENCHMARK.items()},
        "warm": WARM,
        "xy": {"full": (COMPACT, 1)},
    }[args.phase]
    if args.condition:
        if len(set(args.condition)) != len(args.condition) or not set(args.condition) <= conditions.keys():
            raise ValueError(f"Select distinct conditions from {', '.join(conditions)}.")
        conditions = {label: conditions[label] for label in args.condition}
    paths = {name: Path(path).resolve(strict=True) for name, path in args.input}
    input_manifests = {
        root_relative(path.resolve(strict=True)): file_hash(path) for path in args.input_manifest
    }
    provenance = provenance_for(conditions)
    cases = [
        {"input": name, "condition": label, "seed": seed, "trial_budget": args.max_iterations * factor}
        for name in paths
        for label, (method, factor) in conditions.items()
        for seed in ((None,) if method.deterministic else args.seed)
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    deadline = monotonic() + args.total_budget
    save_json(output / "manifest.json", {
        "experiment": EXPERIMENT, "phase": args.phase, "regularization": REG,
        "inputs": {name: {"path": root_relative(source), "sha256": file_hash(source)}
                   for name, source in paths.items()},
        "source_data_manifests": input_manifests,
        "conditions": {label: asdict(method) for label, (method, _) in conditions.items()},
        "cases": cases, "case_wall_limit_seconds": args.case_budget,
        "total_budget_seconds": args.total_budget, "provenance": provenance,
    })
    manifest = read_json(output / "manifest.json")
    input_reports = {}
    records = []
    geometry = None
    active = None
    oracles = {}
    interrupted = False
    try:
        for case in cases:
            if deadline - monotonic() < args.case_budget:
                break
            name = case["input"]
            source = paths[name]
            if active != name:
                geometry = make_geometry(source, REG)
                expected = manifest["inputs"][name]["sha256"]
                if geometry.hashes["input_sha256"] != expected:
                    raise ValueError(f"{name} changed after the manifest was saved.")
                input_reports[name] = input_record(geometry, source)
                active = name
            case_args = argparse.Namespace(**(vars(args) | {"max_iterations": case["trial_budget"]}))
            method = conditions[case["condition"]][0]
            mode = "benchmark" if args.phase == "benchmark" else "xy"
            record = run_case(case, case_args, mode, method, source, geometry, output, deadline, oracles)
            if method.solver == COMPACT.solver and record["status"] in ("verified", "partial"):
                report = read_json(ROOT / record["case_directory"] / "report.json")
                record["backend"] = report["backend"]
                if args.phase == "warm" and record["status"] == "verified" and (
                    record["iterations"] != case["trial_budget"]
                ):
                    record.update(status="partial", incomplete="warm_trial_budget")
            records.append(record)
    except KeyboardInterrupt:
        interrupted = True
    records.extend(
        dict(case, status="pending", termination_reason="interrupted" if interrupted else "total_budget")
        for case in cases[len(records):]
    )
    unchanged = (
        all(file_hash(paths[name]) == entry["sha256"] for name, entry in manifest["inputs"].items())
        and all(file_hash(ROOT / path) == digest for path, digest in input_manifests.items())
        and all(file_hash(ROOT / path) == digest for path, digest in provenance["source_sha256"].items())
    )
    complete = unchanged and all(record["status"] == "verified" for record in records)
    save_json(output / "summary.json", {
        "experiment": EXPERIMENT, "phase": args.phase,
        "complete": complete, "sources_unchanged": unchanged, "interrupted": interrupted,
        "inputs": input_reports, "cases": records, "comparisons": summarize(records) if unchanged else [],
        "observed_frontiers": (
            cap_frontiers(records, {label: method for label, (method, _) in conditions.items()})
            if unchanged and args.phase == "warm" else []
        ),
    })
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
