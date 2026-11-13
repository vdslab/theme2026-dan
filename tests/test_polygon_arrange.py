"""Actual intersections, shared coordinates, shape limits and native gradients."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import solve_polygon_layout as layout
from polygon_layout.polygon_arrange import PlacementAudit, PlacementEnergy, optimize_placement


def squares():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.],
                   [.6, .2], [1.6, .2], [1.6, 1.2], [.6, 1.2]])
    return {'nodes': [{'id': i, 'x': x, 'y': y} for i, (x, y) in enumerate(xy)],
            'hyperedges': [{'id': 'a', 'nodes': [0, 1, 2, 3]},
                           {'id': 'b', 'nodes': [4, 5, 6, 7]}]}


class PlacementTests(unittest.TestCase):
    def test_gradient_including_active_shape_limits(self):
        p = layout.PolygonProblem(squares())
        energy = PlacementEnergy(p, p.initial)
        x = (p.initial + np.random.default_rng(31).normal(0, .08, p.initial.shape)).ravel()
        x[0] -= .35
        value, analytic = energy.evaluate(x)
        self.assertGreater(energy.evaluate(x, details=True)[1]['shape_limit_penalty'], 0.)
        numeric = np.zeros_like(x)
        for i in range(len(x)):
            step = np.zeros_like(x); step[i] = 1e-6
            numeric[i] = (energy.evaluate(x+step)[0]-energy.evaluate(x-step)[0])/2e-6
        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-5)
        energy.close()
        with self.assertRaises(RuntimeError):
            energy.evaluate(x)

    def test_disjoint_members_can_translate_apart_without_shape_damage(self):
        data = squares()
        data['hyperedges'][0]['nodes'] += [0]
        data['hyperedges'][0]['layer'] = 4
        p = layout.PolygonProblem(data)
        xy, report = optimize_placement(p, iterations=180, rounds=2, progress=False)
        self.assertTrue(report['improved'])
        self.assertEqual(report['placement_after']['target_pairs'], 0)
        self.assertAlmostEqual(report['after']['mean_regularity_ratio'], 1., places=5)
        result = layout.make_output(p, xy, reorder=False)
        layout.assert_preserved(data, result)
        self.assertEqual(data['hyperedges'], result['hyperedges'])

    def test_shared_vertex_keeps_one_coordinate_and_can_pivot(self):
        # Two triangles share ID 0. Their boundary edges initially cross.
        xy = [[0., 0.], [1., 0.], [.5, .8660254038],
              [.94, .342], [.174, .985]]
        data = {'nodes': [{'id': i, 'x': x, 'y': y} for i, (x, y) in enumerate(xy)],
                'hyperedges': [{'id': 0, 'nodes': [0, 1, 2]}, {'id': 1, 'nodes': [0, 3, 4]}]}
        p = layout.PolygonProblem(data)
        out, report = optimize_placement(p, iterations=250, rounds=2, progress=False)
        self.assertTrue(report['improved'])
        self.assertEqual(out.shape, (5, 2))
        self.assertLess(report['placement_after']['target_area'], report['placement_before']['target_area'])
        layout.assert_preserved(data, layout.make_output(p, out, reorder=False))

    def test_three_shared_vertices_are_not_forced_apart(self):
        data = squares()
        data['hyperedges'] = [data['hyperedges'][0], {'id': 'b', 'nodes': [0, 1, 2, 3]}]
        p = layout.PolygonProblem(data)
        xy, report = optimize_placement(p, iterations=10, rounds=1, progress=False)
        self.assertFalse(report['improved'])
        self.assertEqual(report['placement_before']['target_pairs'], 0)
        self.assertEqual(report['before_overlaps']['overlapping_pairs'], 1)
        np.testing.assert_array_equal(xy, p.initial)

    def test_shape_guard_rejects_shrink_and_distortion(self):
        p = layout.PolygonProblem(squares())
        audit = PlacementAudit(p, p.initial)
        self.assertFalse(audit.shape_check(p.initial*.5)[0])
        bad = p.initial.copy(); bad[0] += [2., 1.]
        self.assertFalse(audit.shape_check(bad)[0])
        self.assertIsNone(audit.exact_overlap(bad))
        self.assertFalse(audit.shape_check(p.initial*np.nan)[0])
        coincident = p.initial.copy(); coincident[4] = coincident[0]
        self.assertFalse(audit.shape_check(coincident)[1]['node_distance_satisfied'])
        reversed_data = copy.deepcopy(p.data)
        for edge in reversed_data['hyperedges']:
            edge['nodes'].reverse()
        reverse = layout.PolygonProblem(reversed_data)
        a, b = PlacementEnergy(p, p.initial), PlacementEnergy(reverse, reverse.initial)
        self.assertAlmostEqual(a.evaluate(p.initial)[0], b.evaluate(p.initial)[0])

    def test_invalid_start_is_rejected_before_native_optimization(self):
        data = squares(); data['hyperedges'][0]['nodes'] = [0, 2, 1, 3]
        with self.assertRaisesRegex(ValueError, 'valid nondegenerate'):
            optimize_placement(layout.PolygonProblem(data), progress=False)

    def test_cli_replaces_with_backup_and_preserves_boundary_order(self):
        data = squares(); payload = json.dumps(data, indent=1).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'input.json'
            source.write_bytes(payload)
            argv = ['solve_polygon_layout.py', str(source), str(source), '--mode', 'arrange',
                    '--iterations', '180', '--max-cycles', '2', '--report', str(root/'report.json'),
                    '--preview', str(root/'preview.svg')]
            with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                layout.main()
            self.assertEqual((root/'input_before_polygon_arrangement.json').read_bytes(), payload)
            result = json.loads(source.read_text(encoding='utf-8'))
            self.assertEqual(result['hyperedges'], data['hyperedges'])
            self.assertTrue(json.loads((root/'report.json').read_text())['wrote_layout'])
            self.assertTrue((root/'preview.svg').is_file())

    def test_no_improvement_keeps_exact_input_bytes_and_no_backup(self):
        data = squares(); data['hyperedges'] = data['hyperedges'][:1]
        payload = json.dumps(data, indent=1).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'input.json'; source.write_bytes(payload)
            argv = ['solve_polygon_layout.py', str(source), str(source), '--mode', 'arrange',
                    '--iterations', '5', '--max-cycles', '1', '--report', str(root/'report.json'),
                    '--preview', str(root/'preview.svg')]
            with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                layout.main()
            self.assertEqual(source.read_bytes(), payload)
            self.assertFalse((root/'input_before_polygon_arrangement.json').exists())
            self.assertFalse(json.loads((root/'report.json').read_text())['wrote_layout'])

    def test_singleton_and_empty_are_safe_and_serializable(self):
        data = {'nodes': [{'id': 0, 'x': 3., 'y': -2.}],
                'hyperedges': [{'id': 0, 'nodes': [0]}, {'id': 1, 'nodes': []}]}
        p = layout.PolygonProblem(data)
        xy, report = optimize_placement(p, iterations=5, rounds=1, progress=False)
        self.assertFalse(report['improved'])
        np.testing.assert_array_equal(xy, p.initial)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
