import unittest
from unittest.mock import patch

from native.compact_search import DEFAULT_LIBRARY
from solve_layers_compact import Config, OBJECTIVES, solve
from tests.test_solve_layers_compact import shared_vertex_path


@unittest.skipUnless(DEFAULT_LIBRARY.is_file(), "Build the native compact search library first.")
class NativeCompactSearchTests(unittest.TestCase):
    def test_warm_start_and_all_objectives_match_reference_on_small_input(self):
        data = shared_vertex_path()
        for objective in OBJECTIVES:
            with self.subTest(objective=objective):
                config = Config(
                    objective=objective, reg=0, max_iterations=360, seed=37,
                    warm_start_extra_layers=2, moves=("shift", "move", "swap", "kempe"),
                )
                layout, native = solve(data, config, backend="native")
                reference_layout, reference = solve(data, config, backend="python")
                self.assertEqual(layout, reference_layout)
                self.assertEqual(native["backend"], "native")
                for actual, expected in zip(native["stages"], reference["stages"], strict=True):
                    for field in (
                        "cap", "layers", "metrics", "moves", "optimized_value",
                        "iterations", "free_coordinate_updates",
                    ):
                        self.assertEqual(actual[field], expected[field], field)

    def test_deadline_at_first_stage_returns_feasible_initial_layout(self):
        _, report = solve(shared_vertex_path(), Config(time_budget=1e-9), backend="native")
        self.assertEqual(report["termination_reason"], "time_budget")
        self.assertEqual(report["iterations"], 0)
        self.assertEqual(report["metrics"]["overlap_violation_count"], 0)
        self.assertEqual(report["stages"][0]["layers"], report["initial_layers"])

    def test_zero_proposals_still_refines_free_coordinates(self):
        config = Config(max_iterations=0, optimize_free_edges=True)
        data = shared_vertex_path()
        native_layout, native = solve(data, config, backend="native")
        python_layout, reference = solve(data, config, backend="python")
        self.assertEqual(native_layout, python_layout)
        self.assertEqual(native["metrics"], reference["metrics"])
        self.assertEqual(native["stages"][0]["free_coordinate_updates"],
                         reference["stages"][0]["free_coordinate_updates"])

    def test_rng_cancellation_propagates_out_of_native_search(self):
        with patch("solve_layers_compact.random.Random") as factory:
            factory.return_value.randrange.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                solve(shared_vertex_path(), Config(max_iterations=10), backend="native")
