"""Regression checks for the independent validation of experiment reports."""

from dataclasses import replace
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.tree_optimization_comparison_01.review_results import make_geometry
from experiments.visnotes_common import COMPACT, FIXTURES, REG, checked_compact_report, run_case
from experiments.visnotes_exact import verify_optimum
from experiments.visnotes_metrics import read_layout_metrics
from solve_layers_compact import Config, solve


class ExperimentVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = make_geometry(FIXTURES["four-path"], REG)
        config = Config(max_iterations=4, warm_start_extra_layers=1)
        layout, report = solve(cls.geometry.data, config)
        report["input_sha256"] = cls.geometry.hashes["input_sha256"]
        cls.encoded_report = json.dumps(report)
        cls.levels, cls.metrics = read_layout_metrics(layout, cls.geometry, "fixture")

    def checked(self, report):
        return checked_compact_report(
            report, self.geometry, {"condition": "hybrid", "input": "original", "seed": 0},
            replace(COMPACT, warm_start=1), self.levels, self.metrics,
            monotonic() + 1, {}, 4, 0.0, 5.0,
        )

    def test_report_settings_cannot_relax_case_iteration_limit(self):
        report = json.loads(self.encoded_report)
        _, complete, _ = self.checked(report)
        self.assertTrue(complete)
        report["settings"]["max_iterations"] = 999
        with self.assertRaisesRegex(ValueError, "settings"):
            self.checked(report)

    def test_warm_start_cannot_claim_an_unrelated_initial_layout(self):
        report = json.loads(self.encoded_report)
        levels = report["stages"][1]["initial_layers"]
        report["stages"][1]["initial_layers"] = [1 - layer for layer in levels]
        with self.assertRaisesRegex(ValueError, "previous layout"):
            self.checked(report)

    def test_later_stage_cannot_backdate_its_trajectory(self):
        report = json.loads(self.encoded_report)
        for record in report["stages"][1]["trajectory"]:
            record["elapsed_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "trajectory"):
            self.checked(report)

    def test_custom_termination_reason_cannot_pass(self):
        report = json.loads(self.encoded_report)
        report["termination_reason"] = report["stages"][-1]["termination_reason"] = "unknown"
        with self.assertRaisesRegex(ValueError, "termination reason"):
            self.checked(report)

    def test_small_remaining_budget_does_not_shorten_solver_case(self):
        with patch("experiments.visnotes_common.subprocess.run") as subprocess_run:
            record = run_case(
                {"condition": "hybrid", "input": "original", "seed": 0},
                SimpleNamespace(case_budget=5.0), "objectives", COMPACT,
                None, None, Path("unused"), monotonic() + 1, {},
            )
        self.assertEqual(record["status"], "pending")
        subprocess_run.assert_not_called()

    def test_milp_infeasibility_contradicts_feasible_enumeration(self):
        with patch("experiments.visnotes_exact.milp_optimum", return_value={"status": 2, "message": "infeasible"}):
            with self.assertRaisesRegex(ValueError, "known feasible"):
                verify_optimum(self.geometry, 2, "pillar", 0.25, 0.005, 1.0)


if __name__ == "__main__":
    unittest.main()
