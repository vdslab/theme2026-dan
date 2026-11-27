"""Check the public benchmark solver against exact optima on small induced cases."""

import argparse
from collections import deque
from math import isclose
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.visnotes_common import REG, file_hash, make_geometry, save_json, stage_layout
from experiments.visnotes_exact import verify_optimum
from experiments.visnotes_metrics import check_report_metrics, read_layout_metrics
from solve_layers_compact import Config, solve


def incident_subgraph(geometry, size=6):
    neighbors = [set() for _ in range(geometry.size)]
    for left, right in geometry.overlaps:
        neighbors[left].add(right)
        neighbors[right].add(left)
    order = sorted(range(geometry.size), key=lambda edge: (-len(neighbors[edge]), edge))
    chosen = []
    queued = set()
    for start in order:
        if start in queued:
            continue
        pending = deque([start])
        queued.add(start)
        while pending and len(chosen) < size:
            edge = pending.popleft()
            chosen.append(edge)
            new = sorted(neighbors[edge] - queued)
            pending.extend(new)
            queued.update(new)
        if len(chosen) == size:
            break
    edges = [geometry.data["hyperedges"][index] for index in sorted(chosen)]
    vertices = {vertex for edge in edges for vertex in edge["nodes"]}
    return {
        "nodes": [node for node in geometry.data["nodes"] if node["id"] in vertices],
        "hyperedges": edges,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    sources = {source.stem: source.resolve(strict=True) for source in args.input}
    if len(sources) != len(args.input):
        parser.error("Input filenames must have unique stems.")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", {
        "experiment": "visnotes_public_validation_01",
        "sources": {name: {"path": str(source), "sha256": file_hash(source)}
                    for name, source in sources.items()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": file_hash(Path(__file__)),
        "selection": "breadth-first from highest overlap degree, tie by input order, six edges",
        "trial_budget": 20000, "solver_time_limit_seconds": 5, "milp_time_limit_seconds_per_cap": 15,
        "seeds": list(range(5)),
    })
    results = {}
    for name, source in sources.items():
        directory = output / name
        directory.mkdir()
        data = incident_subgraph(make_geometry(source, REG))
        filename = directory / "input.json"
        save_json(filename, data)
        geometry = make_geometry(filename, REG)
        independent = {}
        cases = []
        for seed in range(5):
            config = Config(seed=seed, warm_start_extra_layers=2, time_budget=5)
            layout, report = solve(data, config)
            _, measured = read_layout_metrics(layout, geometry, f"{name}/seed-{seed}")
            check_report_metrics(report["metrics"], measured, name, check_spans=True)
            stages = []
            if report["iterations"] != config.max_iterations:
                raise RuntimeError(f"{name}/seed-{seed}: trial budget did not complete.")
            for stage in report["stages"]:
                cap = stage["cap"]
                saved_stage = stage_layout(geometry, stage["layers"])
                _, stage_metrics = read_layout_metrics(saved_stage, geometry, f"{name}/cap-{cap}")
                check_report_metrics(stage["metrics"], stage_metrics, name, check_spans=True)
                if cap not in independent:
                    independent[cap] = verify_optimum(
                        geometry, cap, config.objective, config.alpha, config.beta, 15,
                    )
                target = independent[cap]["enumeration"]["value"]
                gap = stage["optimized_value"] - target
                if gap < 0 and not isclose(gap, 0, abs_tol=1e-8):
                    raise ValueError("Heuristic value is below an enumerated optimum.")
                stages.append({"cap": cap, "absolute_gap": max(0, gap)})
            case_dir = directory / f"seed-{seed}"
            case_dir.mkdir()
            save_json(case_dir / "layout.json", layout)
            save_json(case_dir / "solver.json", report)
            cases.append({"seed": seed, "backend": report["backend"], "stages": stages})
        record = {
            "input_sha256": file_hash(filename),
            "source_edge_ids": [edge["id"] for edge in data["hyperedges"]],
            "optima": independent, "cases": cases,
        }
        save_json(directory / "summary.json", record)
        results[name] = record
    save_json(output / "summary.json", {"complete": True, "datasets": results})


if __name__ == "__main__":
    main()
