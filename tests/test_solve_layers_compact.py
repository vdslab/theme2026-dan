"""Small independent checks for compact layer assignment."""

from contextlib import redirect_stderr, redirect_stdout
import io
from itertools import combinations
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import solve_layers_compact as solver


def polygons(*boundaries):
    coordinates = {}
    edges = []
    for index, boundary in enumerate(boundaries):
        ids = [coordinates.setdefault(point, len(coordinates)) for point in boundary]
        edges.append({"id": index, "nodes": ids, "name": f"E{index}"})
    return {
        "nodes": [{"id": index, "x": x, "y": y} for (x, y), index in coordinates.items()],
        "hyperedges": edges,
    }


def rectangle(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def shared_vertex_path():
    # The overlap graph is a path. The end faces share a point, but no area.
    return polygons(
        rectangle(-2, -2, 0, 0),
        rectangle(-1, -2, 2, -1),
        rectangle(1, -1.5, 2, 1),
        rectangle(0, 0, 2, 2),
    )


def independent_cost(data, layers, config):
    memberships = [set(edge["nodes"]) for edge in data["hyperedges"]]
    rows = [0.0] * len(layers)
    quadratic = 0.0
    for left, right in combinations(range(len(layers)), 2):
        weight = len(memberships[left] & memberships[right]) + config.reg
        quadratic += weight * (int(layers[left]) - int(layers[right])) ** 2
        rows[left] += weight
        rows[right] += weight
    quadratic = 2 * quadratic / (max(rows) or 1)
    pillar = degree_span = 0
    for node in data["nodes"]:
        incident = [int(height) for height, members in zip(layers, memberships) if node["id"] in members]
        span = max(incident) - min(incident) if incident else 0
        pillar += span
        degree_span += len(incident) * span ** 2
    return {
        "pillar": pillar, "squared": quadratic, "hybrid": pillar + config.alpha * quadratic,
        "degree-span": pillar + config.beta * degree_span,
    }[config.objective]


class CompactTests(unittest.TestCase):
    def test_first_fit_order_and_duplicates(self):
        boundaries = [rectangle(x, 0, x + 2, 2) for x in range(4)]
        for edge in (0, 3):
            boundaries[edge][1:1] = [boundaries[edge][0]] * 2
        data = polygons(*boundaries)
        data["hyperedges"][1], data["hyperedges"][3] = data["hyperedges"][3], data["hyperedges"][1]
        data["hyperedges"][2], data["hyperedges"][3] = data["hyperedges"][3], data["hyperedges"][2]
        self.assertEqual(
            solver.solve(data, solver.Config(method="greedy-input"))[1]["initial_color_count"], 3,
        )
        data["hyperedges"].sort(key=lambda edge: edge["id"])
        output, report = solver.solve(data, solver.Config(method="greedy-size"))
        self.assertEqual(report["initial_color_count"], 2)
        self.assertEqual(report["proven_minimum_layers"], 2)
        for output_edge, input_edge in zip(output["hyperedges"], data["hyperedges"]):
            self.assertEqual({k: v for k, v in output_edge.items() if k != "layer"}, input_edge)

    def test_coordinate_costs_include_shared_and_regularized_pairs(self):
        data = {
            "nodes": [{"id": index, "x": index, "y": 0} for index in range(4)],
            "hyperedges": [
                {"id": index, "nodes": members}
                for index, members in enumerate(([0, 1, 2], [0, 1, 2], [0, 0], [3]))
            ],
        }
        layers = np.array([0, 2, 1, 2])
        targets = np.arange(3)
        for name in solver.OBJECTIVES:
            with self.subTest(objective=name):
                config = solver.Config(objective=name, reg=0.125)
                problem = solver.prepare_problem(data, config.reg, overlaps=[(0, 1)])
                objective = solver.Objective(problem, config)
                baseline = independent_cost(data, layers, config)
                self.assertAlmostEqual(objective.value(layers), baseline)
                for free_edge in (2, 3):
                    expected = []
                    for target in targets:
                        candidate = layers.copy()
                        candidate[free_edge] = target
                        expected.append(independent_cost(data, candidate, config) - baseline)
                    np.testing.assert_allclose(objective.coordinate_deltas(layers, free_edge, targets), expected)
                metrics = objective.metrics(layers)
                self.assertEqual(metrics["pillar_length"], 6)
                self.assertEqual(metrics["pillar_square_sum"], 12)
                self.assertEqual(metrics["degree_span_square_sum"], 28)

    def test_all_neighborhoods_keep_overlap_separation(self):
        problem = solver.prepare_problem(shared_vertex_path())
        objective = solver.Objective(problem, solver.Config())
        rng = random.Random(17)
        for move in solver.MOVES:
            with self.subTest(move=move):
                layers = solver.initial_coloring(problem, "dsatur")
                changed = False
                for _ in range(40):
                    candidate = solver.propose(objective, layers, 4, move, rng)
                    if candidate is not None:
                        self.assertEqual(set(candidate), set(range(int(candidate.max()) + 1)))
                        self.assertLess(candidate.max(), 4)
                        self.assertTrue(all(candidate[left] != candidate[right] for left, right in problem.overlaps))
                        changed |= not np.array_equal(candidate, layers)
                        layers = candidate
                self.assertTrue(changed)

    def test_shift_aggregates_match_full_cost_for_every_insertion(self):
        data = {
            "nodes": [{"id": index, "x": index, "y": 0} for index in range(5)],
            "hyperedges": [
                {"id": index, "nodes": members}
                for index, members in enumerate((
                    [0, 1, 2], [0, 2], [1, 3], [0, 3], [1, 2, 3], [4], [0, 0, 2],
                ))
            ],
        }
        layers = np.array([0, 1, 2, 3, 4, 5, 2])
        layer_count = 6
        for reg in (0.0, 1e-5, 0.125):
            problem = solver.prepare_problem(data, reg, overlaps=[])
            for name in solver.OBJECTIVES:
                config = solver.Config(reg=reg, objective=name)
                objective = solver.Objective(problem, config)
                cached = solver.ShiftCosts(objective, layers, layer_count)
                for left, right in combinations(range(layer_count), 2):
                    for source, target in ((left, right), (right, left)):
                        with self.subTest(reg=reg, objective=name, source=source, target=target):
                            order = list(range(layer_count))
                            order.insert(target, order.pop(source))
                            candidate = np.array([order.index(level) for level in layers])
                            permutation = solver.insertion_permutation(layer_count, source, target)
                            np.testing.assert_array_equal(permutation[layers], candidate)
                            self.assertEqual(cached.value(permutation, source, target), objective.value(candidate))
                            self.assertAlmostEqual(objective.value(candidate), independent_cost(data, candidate, config))

    def test_shift_acceleration_keeps_seeded_search_with_unused_layers(self):
        data = shared_vertex_path()
        for name in solver.OBJECTIVES:
            with self.subTest(objective=name):
                config = solver.Config(
                    objective=name, max_iterations=360, seed=37, warm_start_extra_layers=2,
                )
                actual_layout, actual_report = solver.solve(data, config, backend="python")
                with patch.object(solver.ShiftCosts, "supported", return_value=False):
                    full_layout, full_report = solver.solve(data, config, backend="python")
                self.assertEqual(actual_layout, full_layout)
                for actual, full in zip(actual_report["stages"], full_report["stages"], strict=True):
                    for field in ("layers", "metrics", "moves", "optimized_value", "free_coordinate_updates"):
                        self.assertEqual(actual[field], full[field])
                    omit_time = lambda record: {k: v for k, v in record.items() if k != "elapsed_seconds"}
                    self.assertEqual(
                        [omit_time(record) for record in actual["trajectory"]],
                        [omit_time(record) for record in full["trajectory"]],
                    )

    def test_warm_start_can_find_shorter_pillars_with_one_more_layer(self):
        data = shared_vertex_path()
        config = solver.Config(max_iterations=210, warm_start_extra_layers=2)
        _, first = solver.solve(data, config)
        _, second = solver.solve(data, config)
        stages = first["stages"]
        self.assertEqual(first["proven_minimum_layers"], 2)
        self.assertEqual([stage["cap"] for stage in stages], [2, 3, 4])
        self.assertEqual(sum(stage["iterations"] for stage in stages), 210)
        self.assertEqual(stages[0]["metrics"]["pillar_length"], 1)
        self.assertEqual(stages[1]["metrics"]["pillar_length"], 0)
        self.assertEqual(stages[1]["metrics"]["used_layer_count"], 3)
        self.assertEqual(stages[0]["layers"], stages[1]["initial_layers"])
        self.assertEqual(stages[1]["layers"], stages[2]["initial_layers"])
        self.assertEqual([stage["layers"] for stage in stages],
                         [stage["layers"] for stage in second["stages"]])

    def test_coloring_bound_and_time_budget(self):
        data = shared_vertex_path()
        cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
        data["hyperedges"].append({"id": 4, "nodes": []})
        _, report = solver.solve(data, solver.Config(method="dsatur"), overlaps=cycle)
        self.assertEqual(report["clique_lower_bound"], 2)
        self.assertEqual(report["coloring_upper_bound"], 3)
        self.assertIsNone(report["proven_minimum_layers"])
        _, timed = solver.solve(data, solver.Config(max_iterations=60, time_budget=1e-9,
                                                   warm_start_extra_layers=2), overlaps=cycle)
        self.assertEqual(timed["iterations"], 0)
        self.assertEqual(timed["termination_reason"], "time_budget")
        self.assertEqual(len(timed["stages"]), 1)
        self.assertEqual(timed["metrics"]["overlap_violation_count"], 0)

    def test_cli_preserves_input_and_refuses_to_overwrite(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output, report = (folder / name for name in ("input.json", "output.json", "report.json"))
            original = json.dumps(shared_vertex_path()).encode("utf-8")
            source.write_bytes(original)
            args = [str(source), str(output), "--report", str(report), "--max-iterations", "10"]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(solver.main(args), 0)
                saved = [path.read_bytes() for path in (output, report)]
                self.assertEqual(solver.main(args), 1)
                self.assertEqual(solver.main([str(source), str(source), "--report", str(report)]), 1)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(saved, [path.read_bytes() for path in (output, report)])

    def test_second_output_collision_rolls_back_first_file(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            output = folder / "output.json"
            report = folder / "report.json"
            report.write_bytes(b"existing report")
            with self.assertRaises(FileExistsError):
                solver.save_results(((output, {}), (report, {})))
            self.assertFalse(output.exists())
            self.assertEqual(report.read_bytes(), b"existing report")


if __name__ == "__main__":
    unittest.main()
