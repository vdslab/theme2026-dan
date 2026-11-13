"""Independent invariant and numerical tests for the rooted-tree solver."""

from dataclasses import replace
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import solve_layers_tree as solver


def data_with_weights(n, weights=None):
    nodes, memberships = [], [[] for _ in range(n)]
    for (i, j), count in (weights or {}).items():
        for _ in range(count):
            node_id = len(nodes)
            nodes.append({"id": node_id, "x": float(node_id), "y": 0.0})
            memberships[i].append(node_id)
            memberships[j].append(node_id)
    return {"nodes": nodes, "hyperedges": [
        {"id": i, "nodes": row} for i, row in enumerate(memberships)]}


def tree_from_parents(parents):
    children = [[] for _ in range(len(parents) + 1)]
    for node, parent in enumerate(parents):
        children[parent].append(node)
    return solver.Tree(tuple(parents), tuple(tuple(row) for row in children))


def triangle():
    return solver.prepare_problem(
        data_with_weights(3, {(0, 1): 1, (1, 2): 1, (0, 2): 10}),
        overlaps=[(0, 1), (0, 2), (1, 2)])


def independent_cost(problem, layers):
    return sum(problem.regularized[i, j] * (layers[i] - layers[j]) ** 2
               for i in range(problem.n) for j in range(i + 1, problem.n))


class TreeTests(unittest.TestCase):
    def test_bfs_triangle_requires_and_receives_repair(self):
        problem = triangle()
        tree = solver.bfs_tree(problem)
        depths = solver.validate_tree(tree, 3)
        layers = tuple(d - 1 for d in depths[:-1])
        self.assertEqual(len(solver.violating_edges(problem, layers)), 1)
        state, repair = solver.initial_state(problem, solver.Config())
        self.assertEqual(repair.initial_violations, 1)
        self.assertEqual(len(set(state.layers)), 3)
        self.assertEqual(state.metrics["overlap_violation_count"], 0)

    def test_virtual_root_covers_components_and_isolated_nodes(self):
        problem = solver.prepare_problem(data_with_weights(5), overlaps=[(0, 1), (2, 3)])
        tree = solver.bfs_tree(problem)
        depths = solver.validate_tree(tree, 5)
        self.assertEqual(len(tree.children[5]), 3)
        self.assertEqual(depths[4], 1)
        self.assertEqual(set(solver.bfs_order(tree)), set(range(5)))

    def test_child_count_and_nonoverlap_parent_links_are_unrestricted(self):
        problem = solver.prepare_problem(data_with_weights(30), overlaps=[])
        star = tree_from_parents([30] * 30)
        self.assertEqual(solver.make_state(problem, star).layers, (0,) * 30)
        chain = tree_from_parents([30] + list(range(29)))
        self.assertEqual(solver.make_state(problem, chain).layers, tuple(range(30)))

    def test_cycles_duplicates_disconnected_nodes_and_bad_parents_are_rejected(self):
        bad = [
            solver.Tree((1, 0), ((1,), (0,), ())),
            solver.Tree((2, 2), ((), (), (0, 0))),
            solver.Tree((2, 2), ((), (), (0,))),
            solver.Tree((0, 2), ((0,), (), (1,))),
        ]
        for tree in bad:
            with self.subTest(tree=tree), self.assertRaises(ValueError):
                solver.validate_tree(tree, 2)

    def test_rebuild_preserves_compatible_parents_and_sibling_order(self):
        old = tree_from_parents([4, 4, 0, 1])
        tree, layers = solver.tree_for_layers((0, 0, 1, 0), old)
        self.assertEqual(layers, (0, 0, 1, 0))
        self.assertEqual(tree.parent, (4, 4, 0, 4))
        self.assertEqual(tree.children[4], (0, 1, 3))
        self.assertEqual(tree.children[0], (2,))
        self.assertEqual(old.parent, (4, 4, 0, 1))

    def test_subtree_move_preserves_relative_depths_and_detects_external_collision(self):
        problem = solver.prepare_problem(data_with_weights(5), overlaps=[(0, 1), (0, 4)])
        old = tree_from_parents([5, 0, 1, 5, 3])
        new = solver.reparent(old, 0, 3)
        before, after = solver.validate_tree(old, 5), solver.validate_tree(new, 5)
        self.assertEqual([after[i] - before[i] for i in (0, 1, 2)], [1, 1, 1])
        self.assertEqual(after[4], before[4])
        bad = solver.violating_edges(problem, [d - 1 for d in after[:-1]])
        self.assertEqual(bad.tolist(), [[0, 4]])
        for parent in (0, 1, 2):
            with self.assertRaises(ValueError):
                solver.reparent(old, 0, parent)

    def test_single_node_move_reattaches_children_without_moving_them(self):
        problem = solver.prepare_problem(data_with_weights(4), overlaps=[])
        old = tree_from_parents([4, 4, 0, 1])
        proposal = solver.Proposal("move", (1, 0, 1, 1), old, {"node": 0})
        state, _, _ = solver.evaluate_proposal(problem, proposal, solver.Config(), "test")
        self.assertEqual(state.layers, proposal.layers)
        self.assertEqual(state.tree.parent[3], old.parent[3])
        self.assertEqual(state.tree.parent[2], 1)
        self.assertEqual(state.tree.children[1][0], 3)


class RepairAndNumericalTests(unittest.TestCase):
    def test_repair_chooses_smallest_cost_on_either_side(self):
        for weights, expected_node in (
                ({(0, 2): 10, (1, 2): 1}, 0),
                ({(0, 2): 1, (1, 2): 10}, 1)):
            problem = solver.prepare_problem(data_with_weights(3, weights), overlaps=[(0, 1)])
            z = (0, 0, 1)
            result = solver.repair_layers(problem, z)
            self.assertEqual(result.steps[0]["node"], expected_node)
            brute = []
            for node in (0, 1):
                for target in range(-1, 3):
                    changed = list(z)
                    changed[node] = target
                    if target != z[node] and not len(solver.violating_edges(problem, changed)):
                        delta = independent_cost(problem, changed) - independent_cost(problem, z)
                        brute.append((delta, abs(target - z[node]), node, target))
            chosen = result.steps[0]
            actual = (chosen["delta"], abs(chosen["to"] - chosen["from"]),
                      chosen["node"], chosen["to"])
            self.assertAlmostEqual(actual[0], min(brute)[0])
            self.assertEqual(actual[2:], min(brute)[2:])

    def test_repair_always_decreases_collisions_and_can_create_layers(self):
        n = 7
        overlaps = [(i, j) for i in range(n) for j in range(i + 1, n)]
        problem = solver.prepare_problem(data_with_weights(n), overlaps=overlaps)
        original = [0] * n
        result = solver.repair_layers(problem, original)
        self.assertEqual(original, [0] * n)
        self.assertEqual(len(set(result.layers)), n)
        self.assertLessEqual(len(result.steps), len(overlaps))
        for step in result.steps:
            self.assertLess(step["violations_after"], step["violations_before"])

    def test_budget_failure_leaves_the_proposal_unchanged(self):
        problem = triangle()
        layers = [0, 1, 1]
        with self.assertRaises(solver.RepairFailed):
            solver.repair_layers(problem, layers, max_steps=0)
        self.assertEqual(layers, [0, 1, 1])

    def test_single_and_multiple_move_deltas_match_full_pairwise_cost(self):
        problem = triangle()
        before = (0, 1, 2)
        for node in range(3):
            for target in (-2, 0, 4):
                after = list(before)
                after[node] = target
                expected = independent_cost(problem, after) - independent_cost(problem, before)
                self.assertAlmostEqual(solver.move_delta(problem, before, node, target), expected)
        for after in ((0, 2, 1), (3, -1, 4), (4, 5, 6)):
            self.assertAlmostEqual(solver.changed_layers_delta(problem, before, after),
                                   independent_cost(problem, after) - independent_cost(problem, before))

    def test_legacy_qp_objective_matches_pairwise_cost_with_a_nonzero_anchor(self):
        problem = triangle()
        z = (17, 20, 18)
        expected = 2 * problem.scale * independent_cost(problem, z)
        self.assertAlmostEqual(solver.solve_layers_objective(problem, z), expected)
        self.assertAlmostEqual(solver.solve_layers_objective(problem, z),
                               solver.solve_layers_objective(problem, (0, 3, 1)))

    def test_compaction_preserves_feasibility_and_never_increases_squared_cost(self):
        problem = triangle()
        z = (-4, 2, 12)
        compact = solver.compact_layers(z)
        self.assertEqual(compact, (0, 1, 2))
        self.assertEqual(len(solver.violating_edges(problem, compact)), 0)
        self.assertLessEqual(independent_cost(problem, compact), independent_cost(problem, z))

    def test_random_graph_repairs_obey_invariants(self):
        rng = random.Random(104)
        for _ in range(24):
            n = rng.randrange(2, 11)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            overlaps = [pair for pair in pairs if rng.random() < 0.5]
            weights = {pair: rng.randrange(4) for pair in pairs}
            problem = solver.prepare_problem(data_with_weights(n, weights), overlaps=overlaps)
            z = [rng.randrange(-2, 3) for _ in range(n)]
            result = solver.repair_layers(problem, z)
            tree, compact = solver.tree_for_layers(result.layers, tree_from_parents([n] * n))
            state = solver.make_state(problem, tree)
            self.assertEqual(state.layers, compact)
            self.assertEqual(state.metrics["overlap_violation_count"], 0)
            self.assertAlmostEqual(state.metrics["l2_regularized"], independent_cost(problem, compact))

    def test_single_node_and_zero_regularization(self):
        problem = solver.prepare_problem(data_with_weights(1), reg=0, overlaps=[])
        result = solver.run_search(problem, solver.Config(reg=0, max_iterations=15, time_budget=0))
        self.assertEqual(result.best.layers, (0,))
        self.assertEqual(result.best.objective, 0)
        self.assertEqual(problem.scale, 1)

    def test_no_overlap_initially_collapses_to_one_layer(self):
        problem = solver.prepare_problem(data_with_weights(4, {(0, 3): 10}), overlaps=[])
        state, _ = solver.initial_state(problem, solver.Config())
        self.assertEqual(state.layers, (0, 0, 0, 0))
        self.assertEqual(state.objective, 0)

    def test_original_overlaps_need_not_have_shared_vertex_ids(self):
        data = {"nodes": [], "hyperedges": []}
        for edge, offset in enumerate((0.0, 0.5)):
            ids = []
            for x, y in ((offset, 0), (offset + 1, 0), (offset + 1, 1), (offset, 1)):
                node_id = len(data["nodes"])
                ids.append(node_id)
                data["nodes"].append({"id": node_id, "x": x, "y": y})
            data["hyperedges"].append({"id": edge, "nodes": ids})
        problem = solver.prepare_problem(data)
        self.assertEqual(problem.overlaps, ((0, 1),))
        self.assertEqual(problem.weights.sum(), 0)

    def test_invalid_input_and_config_are_rejected(self):
        bad = data_with_weights(2)
        bad["hyperedges"][0]["nodes"] = [999]
        with self.assertRaises(ValueError):
            solver.prepare_problem(bad)
        for config in (solver.Config(seed=-1), solver.Config(reg=float("nan")),
                       solver.Config(move_weights=(0, 0, 0, 0)),
                       solver.Config(initial_temperature=0)):
            with self.assertRaises(ValueError):
                config.validate()


class SearchTests(unittest.TestCase):
    def test_acceptance_uses_temperature_and_greedy_rejects_worse(self):
        self.assertTrue(solver.accept_move(1, 10, SimpleNamespace(random=lambda: 0.0)))
        self.assertFalse(solver.accept_move(1, 10, SimpleNamespace(random=lambda: 0.999)))
        self.assertFalse(solver.accept_move(1, 10, SimpleNamespace(random=lambda: 0.0), "greedy"))
        self.assertTrue(solver.accept_move(0, 0, SimpleNamespace(random=lambda: 1.0)))

    def test_best_survives_an_accepted_worsening_move(self):
        problem = triangle()
        poor_tree = tree_from_parents([3, 0, 1])
        start = solver.make_state(problem, poor_tree)
        proposals = [
            (solver.Proposal("swap", (0, 2, 1), poor_tree, {}), "swap"),
            (solver.Proposal("swap", (0, 1, 2), poor_tree, {}), "swap"),
        ]
        config = solver.Config(max_iterations=2, time_budget=0,
                               initial_temperature=1e12, final_temperature_ratio=1)
        saved = []
        with patch.object(solver, "propose", side_effect=proposals):
            result = solver.run_search(problem, config, starting_state=start,
                                       checkpoint=lambda initial, best, summary, history: saved.append(best))
        self.assertLess(result.best.objective, start.objective)
        self.assertEqual(result.current.layers, start.layers)
        self.assertEqual(result.summary["accepted"], 2)
        self.assertEqual(saved[-1], result.best)
        self.assertEqual(start.layers, (0, 1, 2))

    def test_equal_cost_different_parent_remains_an_explorable_state(self):
        problem = solver.prepare_problem(data_with_weights(3), overlaps=[(0, 2), (1, 2)])
        tree = tree_from_parents([3, 3, 0])
        start = solver.make_state(problem, tree)
        new = solver.reparent(tree, 2, 1)
        proposal = solver.Proposal("rewire", start.layers, new, {})
        with patch.object(solver, "propose", return_value=(proposal, "rewire")):
            result = solver.run_search(
                problem, solver.Config(max_iterations=1, time_budget=0, initial_temperature=1),
                starting_state=start)
        self.assertEqual(result.current.layers, start.layers)
        self.assertNotEqual(result.current.tree, start.tree)
        self.assertEqual(result.summary["equal_cost_tree_changes"], 1)

    def test_fixed_iterations_and_seed_are_reproducible(self):
        problem = triangle()
        config = solver.Config(max_iterations=50, time_budget=0, seed=23)
        a = solver.run_search(problem, config)
        b = solver.run_search(problem, config)
        self.assertEqual(a.best, b.best)
        self.assertEqual(a.current, b.current)
        self.assertEqual(a.history, b.history)
        self.assertEqual(a.summary["accepted"], b.summary["accepted"])

    def test_interrupt_preserves_a_feasible_checkpoint(self):
        problem = triangle()
        saved = []
        with patch.object(solver, "propose", side_effect=KeyboardInterrupt):
            result = solver.run_search(
                problem, solver.Config(max_iterations=2, time_budget=0, initial_temperature=1),
                checkpoint=lambda initial, best, summary, history: saved.append(best))
        self.assertEqual(result.summary["stop_reason"], "interrupted")
        self.assertEqual(saved[-1], result.best)
        solver.validate_state(problem, result.best)

    def test_deadline_during_search_preserves_the_initial_state(self):
        problem = triangle()
        with patch.object(solver, "propose", side_effect=solver.DeadlineExceeded):
            result = solver.run_search(
                problem, solver.Config(max_iterations=2, time_budget=0, initial_temperature=1))
        self.assertEqual(result.summary["stop_reason"], "time_budget_exhausted")
        self.assertEqual(result.best, result.initial)
        with self.assertRaises(solver.DeadlineExceeded):
            solver.check_deadline(time.monotonic() - 1)


class PersistenceTests(unittest.TestCase):
    @staticmethod
    def data():
        return {"custom_metadata": {"keep": True},
                "nodes": [{"id": 0, "x": 0, "y": 0}, {"id": 1, "x": 1, "y": 0},
                          {"id": 2, "x": 0, "y": 1}],
                "hyperedges": [{"id": i, "nodes": [0, 1, 2, 0], "label": str(i),
                                "color": {"r": 1, "g": 2, "b": 3}} for i in range(3)]}

    def test_cli_preserves_input_metadata_and_saves_consistent_feasible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, report = root / "input.json", root / "tree.json", root / "report.json"
            text = json.dumps(self.data(), ensure_ascii=False, indent=1)
            source.write_text(text, encoding="utf-8")
            code = solver.main([str(source), str(output), "--report", str(report),
                                "--max-iterations", "12", "--time-budget", "0", "--quiet"])
            self.assertEqual(code, 0)
            self.assertEqual(source.read_text(encoding="utf-8"), text)
            problem = solver.prepare_problem(self.data())
            final_checkpoint = json.loads((root / "tree_checkpoint.json").read_text(encoding="utf-8"))
            for target in (output, root / "tree_initial.json", root / "tree_checkpoint.json"):
                payload = json.loads(target.read_text(encoding="utf-8"))
                if target == output:
                    self.assertEqual(set(payload), {"nodes", "hyperedges", "overlaps"})
                    metadata = final_checkpoint["tree_layer_search"]
                else:
                    metadata = payload["tree_layer_search"]
                    self.assertEqual(payload["custom_metadata"], self.data()["custom_metadata"])
                saved = metadata["best"]
                tree = solver.Tree(tuple(saved["parent"]), tuple(map(tuple, saved["children"])))
                state = solver.make_state(problem, tree)
                self.assertEqual(list(state.layers), saved["layers"])
                self.assertEqual([float(z) for z in state.layers],
                                 [edge["layer"] for edge in payload["hyperedges"]])
                self.assertEqual(payload["nodes"], self.data()["nodes"])
                self.assertEqual(len(payload["hyperedges"]), 3)
                for actual, expected in zip(payload["hyperedges"], self.data()["hyperedges"]):
                    actual.pop("layer")
                    self.assertEqual(actual, expected)
                self.assertEqual(metadata["validation"]["overlap_violation_count"], 0)
            saved_report = json.loads(report.read_text())
            self.assertEqual(saved_report["summary"]["attempts"], 12)
            self.assertAlmostEqual(saved_report["objective_value_solve_layers"],
                                   saved_report["best_metrics"]["l2_scaled_qp"])
            self.assertFalse(list(root.glob("*.tmp")))

    def test_initial_only_and_comparison_geometry_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.json", root / "out.json"
            source.write_text(json.dumps(self.data()), encoding="utf-8")
            solver.main([str(source), str(output), "--report", str(root / "report.json"),
                         "--initial-only", "--time-budget", "0", "--quiet"])
            payload = json.loads((root / "out_checkpoint.json").read_text())
            self.assertEqual(payload["tree_layer_search"]["summary"]["attempts"], 0)
            problem = solver.prepare_problem(self.data())
            self.assertEqual(solver.comparison_metrics(problem, output)["metrics"]["overlap_violation_count"], 0)
            changed = copy.deepcopy(payload)
            changed["nodes"][0]["x"] += 1
            output.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                solver.comparison_metrics(problem, output)

    def test_default_filename_and_legacy_style_console_output(self):
        self.assertEqual(solver.parse_arguments([]).output_file, "layers_hypergraph.json")
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                source = Path("input.json")
                source.write_text(json.dumps(self.data()), encoding="utf-8")
                console = io.StringIO()
                with redirect_stdout(console):
                    solver.main([str(source), "--initial-only", "--time-budget", "0"])
                payload = json.loads(Path("layers_hypergraph.json").read_text())
                self.assertEqual(set(payload), {"nodes", "hyperedges", "overlaps"})
                self.assertTrue(Path("results/layers/layers_hypergraph_initial.json").is_file())
                self.assertTrue(Path("results/layers/layers_hypergraph_checkpoint.json").is_file())
                report = json.loads(Path("results/reports/layers_hypergraph_report.json").read_text())
                for message in ("Reading hypergraph data", "Loaded 3 nodes and 3 hyperedges",
                                "Computing overlap set C", "Computing weight matrix W",
                                "Objective scale factor", "Best objective value",
                                "Objective value (solve_layers.py)",
                                "All overlap constraints successfully satisfied",
                                "Coordinate range", "Preparing output JSON",
                                "Writing output to layers_hypergraph.json",
                                "Layer assignment complete!"):
                    self.assertIn(message, console.getvalue())
                self.assertIn(f"{report['objective_value_solve_layers']:.6f}", console.getvalue())
            finally:
                os.chdir(previous)

    def test_outputs_cannot_replace_input_or_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text(json.dumps(self.data()), encoding="utf-8")
            before = source.read_bytes()
            with self.assertRaises(ValueError):
                solver.main([str(source), str(source)])
            with self.assertRaises(ValueError):
                solver.main([str(source), str(Path(directory) / "out.json"),
                             "--report", str(source)])
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
