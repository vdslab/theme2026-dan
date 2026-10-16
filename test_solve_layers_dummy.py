"""Numerical and state-integrity tests for dummy-layer neighbourhood search.

Run with: python -m unittest test_solve_layers_dummy -v
The small synthetic overlap graphs keep these tests independent of original.json.
"""

import copy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import solve_layers_dummy as dummy


def weighted_data(count, pair_weights):
    """Realise integer edge weights as shared vertex IDs in a hypergraph."""
    memberships = [[] for _ in range(count)]
    nodes = []
    for (left, right), weight in pair_weights.items():
        for _ in range(weight):
            node_id = len(nodes)
            nodes.append({"id": node_id, "x": float(node_id), "y": 0.0})
            memberships[left].append(node_id)
            memberships[right].append(node_id)
    return {
        "nodes": nodes,
        "hyperedges": [
            {"id": index, "nodes": members}
            for index, members in enumerate(memberships)
        ],
    }


def triangle_problem():
    return dummy.prepare_problem(
        weighted_data(3, {(0, 1): 1, (1, 2): 1, (0, 2): 10}),
        overlaps=[(0, 1), (0, 2), (1, 2)],
    )


def unscaled_squared_objective(problem, z):
    differences = np.asarray(z)[:, None] - np.asarray(z)[None, :]
    return float(np.triu(problem.regularized * differences**2, 1).sum())


def generation_result(state, candidates):
    return candidates, {
        "base_state_id": state.state_id,
        "generated": len(candidates),
        "attempts": len(candidates),
        "by_type": {},
        "truncated": False,
    }


def preserve_candidate_order(problem, candidates, config):
    """Control batch boundaries without mocking numerical evaluation."""
    scores = {}
    for candidate in candidates:
        seed = dummy.make_feasible_seed(candidate.colors)
        scores[candidate.original.generation_index] = (
            2 * problem.scale * unscaled_squared_objective(problem, seed)
        )
    return list(candidates), scores


class NumericalLayoutTests(unittest.TestCase):
    def setUp(self):
        self.config = dummy.Config()

    def assert_feasible(self, problem, colors, z):
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertAlmostEqual(float(z[0]), 0.0, places=8)
        directed, _ = dummy.build_directed_constraints(problem, colors)
        for lower, upper in directed:
            self.assertGreaterEqual(
                float(z[upper] - z[lower]), 1.0 - self.config.constraint_tol
            )

    def test_triangle_swap_changes_order_and_reduces_objective(self):
        problem = triangle_problem()
        baseline = dummy.solve_heights_qp(problem, (0, 1, 2), self.config)
        swapped = dummy.solve_heights_qp(problem, (0, 2, 1), self.config)
        self.assert_feasible(problem, (0, 1, 2), baseline.z)
        self.assert_feasible(problem, (0, 2, 1), swapped.z)
        # The heavy 0--2 pair spans two steps initially, then only one.
        self.assertAlmostEqual(
            unscaled_squared_objective(problem, baseline.z), 42.00006, places=3
        )
        self.assertAlmostEqual(
            unscaled_squared_objective(problem, swapped.z), 15.00006, places=3
        )
        self.assertLess(
            swapped.metrics["l2_scaled_qp"], baseline.metrics["l2_scaled_qp"]
        )

    def test_swapped_anchor_can_be_in_upper_layer(self):
        problem = dummy.prepare_problem(
            weighted_data(2, {(0, 1): 1}), overlaps=[(0, 1)]
        )
        colors = (1, 0)
        np.testing.assert_array_equal(dummy.make_feasible_seed(colors), [0, -1])
        solution = dummy.solve_heights_qp(problem, colors, self.config)
        self.assert_feasible(problem, colors, solution.z)
        self.assertAlmostEqual(float(solution.z[1]), -1.0, places=5)

    def test_same_colour_does_not_force_same_continuous_height(self):
        problem = dummy.prepare_problem(
            weighted_data(4, {(0, 1): 1, (0, 2): 1, (1, 3): 10}),
            overlaps=[(0, 1), (0, 2)],
        )
        state = dummy.create_initial_state(problem, self.config)
        self.assertEqual(state.colors[0], state.colors[3])
        self.assert_feasible(problem, state.colors, state.z)
        self.assertGreater(abs(float(state.z[3] - state.z[0])), 0.5)

    def test_single_polygon_has_zero_height_and_no_candidates(self):
        problem = dummy.prepare_problem(weighted_data(1, {}), overlaps=[])
        state = dummy.create_initial_state(problem, self.config)
        self.assertEqual(tuple(state.colors), (0,))
        np.testing.assert_array_equal(state.z, [0.0])
        self.assertEqual(state.metrics["l2_scaled_qp"], 0.0)
        candidates, _ = dummy.generate_candidates(problem, state, self.config)
        self.assertEqual(candidates, [])

    def test_no_overlaps_collapses_all_heights(self):
        problem = dummy.prepare_problem(
            weighted_data(3, {(0, 1): 2, (1, 2): 1}), overlaps=[]
        )
        state = dummy.create_initial_state(problem, self.config)
        np.testing.assert_allclose(state.z, np.zeros(3), atol=1e-7)
        self.assertEqual(len(set(state.colors)), 1)
        self.assertAlmostEqual(state.metrics["l2_scaled_qp"], 0.0, places=7)

    def test_geometric_overlap_does_not_require_shared_vertex_ids(self):
        nodes = []
        hyperedges = []
        for index, (left, right) in enumerate(((0.0, 1.0), (0.5, 1.5), (1.5, 2.5))):
            ids = []
            for x, y in ((left, 0), (right, 0), (right, 1), (left, 1)):
                node_id = len(nodes)
                nodes.append({"id": node_id, "x": x, "y": y})
                ids.append(node_id)
            hyperedges.append({"id": index, "nodes": ids})
        problem = dummy.prepare_problem({"nodes": nodes, "hyperedges": hyperedges})
        self.assertEqual({tuple(pair) for pair in problem.overlaps}, {(0, 1)})
        np.testing.assert_array_equal(problem.weights, np.zeros((3, 3)))

    def test_absolute_objective_uses_regularized_weights(self):
        problem = triangle_problem()
        config = replace(self.config, objective="l1")
        solution = dummy.solve_heights_qp(problem, (0, 2, 1), config)
        self.assert_feasible(problem, (0, 2, 1), solution.z)
        self.assertAlmostEqual(solution.metrics["l1_regularized"], 13.00004, places=4)
        self.assertAlmostEqual(
            solution.solver_info["reported_objective"],
            solution.metrics["l1_regularized"],
            places=6,
        )

    def test_recomputed_objective_detects_wrong_solver_value(self):
        problem = triangle_problem()
        directed, _ = dummy.build_directed_constraints(problem, (0, 1, 2))
        with self.assertRaises(dummy.OptimizationError) as caught:
            dummy.checked_metrics(
                problem, np.array([0.0, 1.0, 2.0]), directed, self.config,
                reported_objective=0.0,
            )
        self.assertEqual(caught.exception.status, "objective_mismatch")

    def test_solver_runtime_error_becomes_recordable_optimization_failure(self):
        problem = triangle_problem()
        for objective, solver_name in (("l2", "minimize"), ("l1", "linprog")):
            with self.subTest(objective=objective):
                config = replace(self.config, objective=objective)
                with patch.object(
                    dummy, solver_name, side_effect=RuntimeError("Injected solver exception")
                ), self.assertRaises(dummy.OptimizationError) as caught:
                    dummy.solve_heights_qp(problem, (0, 1, 2), config)
                self.assertEqual(caught.exception.status, "solver_failed")
                self.assertIn("Injected solver exception", str(caught.exception))

    def test_solver_success_cannot_admit_nonfinite_coordinates(self):
        problem = triangle_problem()
        for objective, solver_name, variable_count in (
            ("l2", "minimize", 2), ("l1", "linprog", 6)
        ):
            with self.subTest(objective=objective):
                config = replace(self.config, objective=objective)
                result = SimpleNamespace(
                    success=True, x=np.full(variable_count, np.nan), fun=0.0,
                    nit=1, message="Misleading successful result",
                )
                with patch.object(
                    dummy, solver_name, return_value=result
                ), self.assertRaises(dummy.OptimizationError) as caught:
                    dummy.solve_heights_qp(problem, (0, 1, 2), config)
                self.assertEqual(caught.exception.status, "invalid_numeric")


class CandidateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.problem = triangle_problem()
        self.config = dummy.Config(max_candidates=30, max_attempts=1000)
        self.state = dummy.create_initial_state(self.problem, self.config)
        self.before = copy.deepcopy(self.state)
        self.candidates, _ = dummy.generate_candidates(
            self.problem, self.state, self.config
        )
        self.assertTrue(self.candidates, "The triangle has legal swaps.")
        self.candidate = self.candidates[0]

    def assert_original_unchanged(self):
        self.assertEqual(self.state.state_id, self.before.state_id)
        self.assertEqual(self.state.colors, self.before.colors)
        self.assertEqual(self.state.layer_members, self.before.layer_members)
        np.testing.assert_array_equal(self.state.z, self.before.z)
        np.testing.assert_array_equal(self.state.z_seed, self.before.z_seed)
        np.testing.assert_array_equal(self.state.directed, self.before.directed)
        self.assertEqual(self.state.metrics, self.before.metrics)

    def test_legal_swap_reverses_order_and_leaves_dummy_empty(self):
        result = dummy.validate_candidate(
            self.problem, self.state, self.candidate, self.config
        )
        self.assertEqual(result.status, "validated")
        validated = result.candidate
        self.assertTrue(validated.changes_overlap_order)
        self.assertNotEqual(tuple(validated.colors), tuple(self.state.colors))
        layer_count = len(self.state.layer_members)
        self.assertTrue(all(0 <= color < layer_count for color in validated.colors))
        self.assertEqual(
            sorted(index for members in validated.layer_members for index in members),
            list(range(self.problem.n)),
        )
        for left, right in self.problem.overlaps:
            self.assertNotEqual(validated.colors[left], validated.colors[right])
        self.assert_original_unchanged()

    def test_stale_candidate_is_not_applied(self):
        stale = replace(self.candidate, base_state_id="old-state")
        result = dummy.validate_candidate(self.problem, self.state, stale, self.config)
        self.assertEqual(result.status, "stale")
        self.assert_original_unchanged()

    def test_forged_colour_signature_is_rejected(self):
        invalid = replace(
            self.candidate, colors_signature=tuple(self.state.colors)
        )
        result = dummy.validate_candidate(self.problem, self.state, invalid, self.config)
        self.assertEqual(result.status, "malformed_candidate")
        self.assert_original_unchanged()

    def test_forged_direction_signature_is_rejected(self):
        invalid = replace(self.candidate, direction_signature=b"incorrect")
        result = dummy.validate_candidate(self.problem, self.state, invalid, self.config)
        self.assertEqual(result.status, "malformed_candidate")
        self.assert_original_unchanged()

    def test_incomplete_operation_log_is_rejected(self):
        invalid = replace(self.candidate, operations=self.candidate.operations[:-1])
        result = dummy.validate_candidate(self.problem, self.state, invalid, self.config)
        self.assertNotEqual(result.status, "validated")
        self.assert_original_unchanged()

    def test_duplicate_change_is_rejected(self):
        invalid = replace(
            self.candidate,
            changes=self.candidate.changes + (self.candidate.changes[0],),
        )
        result = dummy.validate_candidate(self.problem, self.state, invalid, self.config)
        self.assertEqual(result.status, "malformed_candidate")
        self.assert_original_unchanged()

    def test_candidate_cannot_move_into_an_overlapping_occupied_layer(self):
        source = self.state.colors[0]
        target = self.state.colors[1]
        dummy_layer = len(self.state.layer_members)
        colors = list(self.state.colors)
        colors[0] = target
        invalid = replace(
            self.candidate,
            move_type="move",
            changes=((0, source, target),),
            operations=((0, source, dummy_layer), (0, dummy_layer, target)),
            changed_polygons=(0,),
            colors_signature=tuple(colors),
        )
        result = dummy.validate_candidate(self.problem, self.state, invalid, self.config)
        self.assertNotEqual(result.status, "validated")
        self.assert_original_unchanged()

    def test_unmoved_blocker_prevents_one_to_one_swap(self):
        problem = dummy.prepare_problem(
            weighted_data(3, {(0, 1): 1, (1, 2): 1}),
            overlaps=[(0, 1), (1, 2)],
        )
        state = dummy.create_initial_state(problem, self.config)
        self.assertEqual(state.colors[0], state.colors[2])
        candidates, _ = dummy.generate_candidates(problem, state, self.config)
        # Exchanging the centre with either leaf would collide with the other leaf.
        self.assertEqual(candidates, [])

    def test_blocker_exchange_moves_all_blockers_when_enabled(self):
        problem = dummy.prepare_problem(
            weighted_data(3, {(0, 1): 1, (1, 2): 1}),
            overlaps=[(0, 1), (1, 2)],
        )
        config = replace(self.config, max_blockers=2)
        state = dummy.create_initial_state(problem, config)
        candidates, _ = dummy.generate_candidates(problem, state, config)
        exchanges = [
            candidate
            for candidate in candidates
            if candidate.move_type == "blocker_exchange"
        ]
        self.assertTrue(exchanges)
        for candidate in exchanges:
            result = dummy.validate_candidate(problem, state, candidate, config)
            self.assertEqual(result.status, "validated")
            self.assertEqual(len(result.candidate.original.changes), 3)
            self.assertTrue(result.candidate.changes_overlap_order)


class SearchTests(unittest.TestCase):
    def test_improvement_must_exceed_absolute_and_relative_threshold(self):
        config = dummy.Config(improvement_abs_tol=0.1, improvement_rel_tol=0.01)
        self.assertFalse(dummy.is_significant_improvement(100.0, 100.0, config))
        self.assertFalse(dummy.is_significant_improvement(100.0, 101.0, config))
        self.assertFalse(dummy.is_significant_improvement(100.0, 99.5, config))
        self.assertFalse(dummy.is_significant_improvement(100.0, 99.0, config))
        self.assertTrue(dummy.is_significant_improvement(100.0, 98.0, config))
        self.assertFalse(dummy.is_significant_improvement(0.0, 0.0, config))

    def test_search_finds_better_triangle_order_without_changing_initial(self):
        problem = triangle_problem()
        config = dummy.Config(
            max_candidates=30,
            max_attempts=1000,
            max_solver_calls_per_batch=3,
            max_solver_calls_total=9,
            max_batches=3,
            time_budget=30.0,
        )
        initial = dummy.create_initial_state(problem, config)
        original_colors = tuple(initial.colors)
        original_z = initial.z.copy()
        result = dummy.run_dummy_layer_search(problem, config, initial_state=initial)
        self.assertLess(
            result.best.metrics["l2_scaled_qp"],
            result.initial.metrics["l2_scaled_qp"] - 1.0,
        )
        self.assertTrue(result.history)
        self.assertNotEqual(result.best.state_id, result.initial.state_id)
        self.assertEqual(tuple(initial.colors), original_colors)
        np.testing.assert_array_equal(initial.z, original_z)
        self.assertEqual(tuple(result.initial.colors), original_colors)
        np.testing.assert_array_equal(result.initial.z, original_z)
        directed, _ = dummy.build_directed_constraints(problem, result.best.colors)
        for lower, upper in directed:
            self.assertGreaterEqual(
                float(result.best.z[upper] - result.best.z[lower]),
                1.0 - config.constraint_tol,
            )

    def test_next_batch_continues_after_first_batch_has_no_improvement(self):
        problem = triangle_problem()
        config = dummy.Config(
            max_solver_calls_per_batch=1, max_solver_calls_total=3, max_batches=2
        )
        initial = dummy.create_initial_state(problem, config)
        # Reversing all three preserves distances; the following swap improves.
        neutral = dummy.candidate_from_spec(
            problem, initial, "swap", (0, initial.colors[2], (2,)), 1
        )
        improving = dummy.candidate_from_spec(
            problem, initial, "swap", (1, initial.colors[2], (2,)), 2
        )
        with patch.object(
            dummy, "generate_candidates", return_value=generation_result(initial, [neutral, improving])
        ) as generate, patch.object(
            dummy, "prioritize_validated_candidates", side_effect=preserve_candidate_order
        ):
            result = dummy.run_dummy_layer_search(problem, config, initial_state=initial)
        self.assertEqual(generate.call_count, 1, "An unchanged state must retain its queue.")
        self.assertEqual(result.summary["batches"], 2)
        self.assertEqual(result.summary["solver_calls"], 2)
        self.assertEqual(result.summary["adoptions"], 1)
        self.assertEqual([log["batch"] for log in result.candidate_logs], [1, 2])
        self.assertFalse(result.candidate_logs[0]["accepted"])
        self.assertTrue(result.candidate_logs[1]["accepted"])

    def test_failed_solve_consumes_budget_without_replacing_valid_state(self):
        problem = triangle_problem()
        config = dummy.Config(max_solver_calls_total=1, max_solver_calls_per_batch=1)
        initial = dummy.create_initial_state(problem, config)
        with patch.object(
            dummy, "solve_heights_qp",
            side_effect=dummy.OptimizationError("solver_failed", "Injected failure"),
        ) as solve:
            result = dummy.run_dummy_layer_search(problem, config, initial_state=initial)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(result.summary["solver_calls"], 1)
        self.assertEqual(result.summary["solver_failures"], 1)
        self.assertEqual(result.summary["stop_reason"], "solver_budget_exhausted")
        self.assertEqual(result.history, [])
        self.assertEqual(result.best.colors, initial.colors)
        np.testing.assert_array_equal(result.best.z, initial.z)
        self.assertGreater(result.summary["known_unevaluated"], 0)

    def equivalent_direction_candidates(self, config):
        problem = dummy.prepare_problem(
            weighted_data(3, {(0, 1): 1}), overlaps=[(0, 1)]
        )
        # Keep one unused ordinary layer to permit different assignments of C.
        colors = (1, 2, 0)
        solution = dummy.solve_heights_qp(problem, colors, config)
        initial = dummy.assemble_state(problem, colors, 3, solution, config, "custom-initial")
        moved = dummy.candidate_from_spec(problem, initial, "move", (1, 0, ()), 1)
        swapped = dummy.candidate_from_spec(problem, initial, "swap", (0, 2, (1,)), 2)
        self.assertNotEqual(moved.colors_signature, swapped.colors_signature)
        self.assertEqual(moved.direction_signature, swapped.direction_signature)
        self.assertTrue(moved.changes_overlap_order)
        return problem, initial, [moved, swapped]

    def test_distinct_assignments_with_same_new_direction_reuse_success(self):
        config = dummy.Config(max_solver_calls_per_batch=1, max_solver_calls_total=3)
        problem, initial, candidates = self.equivalent_direction_candidates(config)
        solve_real = dummy.solve_heights_qp
        with patch.object(
            dummy, "generate_candidates", return_value=generation_result(initial, candidates)
        ), patch.object(
            dummy, "prioritize_validated_candidates", side_effect=preserve_candidate_order
        ), patch.object(dummy, "solve_heights_qp", wraps=solve_real) as solve:
            result = dummy.run_dummy_layer_search(problem, config, initial_state=initial)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(result.summary["solver_calls"], 1)
        self.assertEqual(result.summary["cache_hits"], 1)
        self.assertEqual(result.summary["evaluated"], 2)
        self.assertEqual(result.history, [])
        self.assertEqual(result.summary["stop_reason"], "no_improvement_in_evaluated_candidates")

    def test_distinct_assignments_with_same_failed_direction_are_not_retried(self):
        config = dummy.Config(max_solver_calls_per_batch=1, max_solver_calls_total=3)
        problem, initial, candidates = self.equivalent_direction_candidates(config)
        with patch.object(
            dummy, "generate_candidates", return_value=generation_result(initial, candidates)
        ), patch.object(
            dummy, "prioritize_validated_candidates", side_effect=preserve_candidate_order
        ), patch.object(
            dummy, "solve_heights_qp",
            side_effect=dummy.OptimizationError("solver_failed", "Injected failure"),
        ) as solve:
            result = dummy.run_dummy_layer_search(problem, config, initial_state=initial)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(result.summary["solver_failures"], 1)
        self.assertEqual(result.summary["failure_cache_hits"], 1)
        self.assertEqual(result.summary["stop_reason"], "evaluation_failed")
        self.assertEqual(result.best.colors, initial.colors)

    def test_unchanged_overlap_direction_reuses_initial_solution(self):
        problem = dummy.prepare_problem(
            weighted_data(4, {(0, 1): 1, (0, 2): 1, (1, 3): 10}),
            overlaps=[(0, 1), (0, 2)],
        )
        config = dummy.Config()
        initial = dummy.create_initial_state(problem, config)
        candidate = dummy.candidate_from_spec(
            problem, initial, "move", (3, initial.colors[1], ()), 1
        )
        self.assertFalse(candidate.changes_overlap_order)
        with patch.object(
            dummy, "generate_candidates", return_value=generation_result(initial, [candidate])
        ), patch.object(dummy, "solve_heights_qp") as solve:
            result = dummy.run_dummy_layer_search(problem, config, initial_state=initial)
        solve.assert_not_called()
        self.assertEqual(result.summary["cache_hits"], 1)
        self.assertEqual(result.summary["solver_calls"], 0)
        self.assertEqual(result.history, [])

    def test_interruption_retains_checkpoint_of_last_adopted_layout(self):
        problem = triangle_problem()
        config = dummy.Config(
            max_solver_calls_per_batch=1, max_solver_calls_total=10, max_batches=5
        )
        initial = dummy.create_initial_state(problem, config)
        solve_real = dummy.solve_heights_qp
        calls = []

        def interrupt_second_solve(problem, colors, config):
            calls.append(tuple(colors))
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return solve_real(problem, colors, config)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.json"

            def checkpoint(initial, best, history, summary):
                dummy.atomic_write_json(
                    checkpoint_path,
                    dummy.output_payload(problem, config, initial, best, history, summary),
                )

            with patch.object(dummy, "solve_heights_qp", side_effect=interrupt_second_solve):
                result = dummy.run_dummy_layer_search(
                    problem, config, initial_state=initial, checkpoint_callback=checkpoint
                )
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(result.summary["stop_reason"], "interrupted")
        self.assertEqual(result.summary["adoptions"], 1)
        self.assertEqual(saved["dummy_layer_search"]["best"]["state_id"], result.best.state_id)
        self.assertEqual(saved["dummy_layer_search"]["best"]["heights"], result.best.z.tolist())
        self.assertFalse(saved["dummy_layer_search"]["dummy_occupied"])
        self.assertEqual(len(saved["dummy_layer_search"]["history"]), 1)


class CliPersistenceTests(unittest.TestCase):
    def test_cli_saves_viewer_fields_initial_and_checkpoint_separately(self):
        data = {
            "nodes": [
                {"id": 0, "x": 0, "y": 0}, {"id": 1, "x": 1, "y": 0},
                {"id": 2, "x": 1, "y": 1}, {"id": 3, "x": 0, "y": 1},
            ],
            "hyperedges": [
                {"id": "edge-A", "nodes": [0, 1, 2, 3], "color": "#ff0000"},
                {"id": "edge-B", "nodes": [0, 1, 2, 3], "color": "#0000ff"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "result.json"
            input_text = json.dumps(data)
            input_path.write_text(input_text, encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                returncode = dummy.main([
                    str(input_path), str(output_path), "--max-batches", "0"
                ])
            self.assertEqual(returncode, 0)
            self.assertEqual(input_path.read_text(encoding="utf-8"), input_text)
            initial_path = Path(directory) / "layered_hypergraph_initial.json"
            checkpoint_path = Path(directory) / "result_checkpoint.json"
            final = json.loads(output_path.read_text(encoding="utf-8"))
            initial = json.loads(initial_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint, final)
            self.assertEqual(initial["dummy_layer_search"]["summary"]["stop_reason"], "initial_checkpoint")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
        self.assertEqual(final["nodes"], data["nodes"])
        # Existing solver/comparison JSON stores input indices, not hyperedge IDs.
        self.assertEqual(final["overlaps"], [[0, 1]])
        for source, saved in zip(data["hyperedges"], final["hyperedges"]):
            self.assertEqual(saved["id"], source["id"])
            self.assertEqual(saved["nodes"], source["nodes"])
            self.assertEqual(saved["color"], source["color"])
            self.assertIsInstance(saved["layer"], float)
            self.assertIsInstance(saved["logical_layer"], int)
        metadata = final["dummy_layer_search"]
        self.assertEqual(metadata["ordinary_layer_count"], 2)
        self.assertFalse(metadata["dummy_occupied"])
        self.assertEqual(metadata["summary"]["stop_reason"], "max_batches_reached")
        self.assertEqual(metadata["history"], [])

    def test_cli_refuses_input_output_path_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            text = json.dumps(weighted_data(1, {}))
            input_path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                dummy.main([str(input_path), str(input_path)])
            self.assertEqual(input_path.read_text(encoding="utf-8"), text)


if __name__ == "__main__":
    unittest.main()
