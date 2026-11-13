"""Numerical, geometric and incidence-preservation checks for XY optimization."""

from collections import Counter
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint

import solve_polygon_layout as layout
from polygon_layout.polygon_paper import PaperEnergy, JointEnergy, TERMS, PAPER_WEIGHTS, hu_layout, find_sfdp
from polygon_layout.polygon_consensus import regular_polygon_consensus
from polygon_layout.polygon_monogons import orient_monogons


def fixture():
    points = [[0., 0.], [.08, .03], [1.3, .2], [.7, 1.8],
              [2.2, 1.1], [2.7, -.4], [-1.1, .9], [3.6, 2.4]]
    memberships = [[0, 1, 2, 3], [2, 3, 4, 5], [0, 4, 5],
                   [0, 1, 2, 4], [1, 5], [6, 7], [4], []]
    return {"nodes": [{"id": 10 + i, "label": f"node {i}", "x": p[0], "y": p[1]}
                      for i, p in enumerate(points)],
            "hyperedges": [{"id": 100 + i, "label": f"edge {i}",
                            "color": {"r": 100, "g": 160, "b": 200},
                            "nodes": [10 + j for j in members]}
                           for i, members in enumerate(memberships)],
            "metadata": {"source": "test fixture"}}


class PolygonLayoutTests(unittest.TestCase):
    def test_clique_expansion_shares_vertices_and_preserves_original_hyperedges(self):
        data = fixture()
        before = copy.deepcopy(data)
        problem = layout.PolygonProblem(data)
        adjacency = layout.clique_adjacency(problem)
        self.assertEqual(adjacency.shape, (len(data["nodes"]), len(data["nodes"])))
        self.assertTrue(np.array_equal(adjacency, adjacency.T))
        self.assertTrue(np.all(adjacency.diagonal() == 0))
        for edge in problem.members:
            for a in edge:
                for b in edge:
                    if a != b:
                        self.assertEqual(adjacency[a, b], 1.)
        self.assertEqual(adjacency[0, 7], 0.)
        self.assertEqual(data, before)
        try:
            find_sfdp()
        except RuntimeError:
            self.skipTest('Graphviz sfdp is not installed')
        xy, info = hu_layout(problem, representation='clique')
        self.assertEqual(xy.shape, problem.initial.shape)
        self.assertTrue(np.isfinite(xy).all())
        self.assertEqual(info['graph_vertices'], problem.n)
        self.assertEqual(info['graph_edges'], np.count_nonzero(adjacency)//2)

    def test_all_energy_gradients_against_independent_finite_differences(self):
        problem = layout.PolygonProblem(fixture())
        energy = PaperEnergy(problem)
        x = problem.initial.ravel()
        for index, term in enumerate(TERMS):
            with self.subTest(term=term):
                weights = np.zeros(len(TERMS))
                weights[index] = 1.
                value, analytic = energy.evaluate(x, weights)
                numeric = np.empty_like(x)
                for i in range(len(x)):
                    step = np.zeros_like(x)
                    step[i] = 2e-6
                    numeric[i] = (energy.evaluate(x + step, weights)[0]
                                  - energy.evaluate(x - step, weights)[0]) / (4e-6)
                self.assertTrue(np.isfinite(value))
                np.testing.assert_allclose(analytic, numeric, atol=3e-5, rtol=3e-4)

    def test_starrization_retains_interior_nodes_and_produces_simple_boundary(self):
        xy = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [.7, .8]])
        order = layout.starrize([2, 0, 4, 3, 1], xy)
        self.assertEqual(set(order), set(range(5)))
        self.assertTrue(Polygon(xy[order]).is_valid)
        self.assertGreater(layout.signed_area(xy[order]), 0)

    def test_output_preserves_shared_ids_duplicates_and_all_metadata(self):
        data = fixture()
        data["hyperedges"][0]["nodes"] += [10]
        data["hyperedges"][0]["layer"] = 2.5
        before = copy.deepcopy(data)
        problem = layout.PolygonProblem(data)
        output = layout.make_output(problem, problem.initial + [.5, -.2])
        layout.assert_preserved(data, output)
        self.assertEqual(data, before)
        self.assertEqual(Counter(output["hyperedges"][0]["nodes"]), Counter(data["hyperedges"][0]["nodes"]))
        self.assertEqual(len(output["nodes"]), len(data["nodes"]))
        self.assertIn(12, output["hyperedges"][0]["nodes"])
        self.assertIn(12, output["hyperedges"][1]["nodes"])
        output["hyperedges"][0]["nodes"].pop()
        with self.assertRaises(ValueError):
            layout.assert_preserved(data, output)

    def test_missing_duplicate_and_nonfinite_nodes_are_rejected(self):
        for mutation in (lambda d: d["nodes"].append(copy.deepcopy(d["nodes"][0])),
                         lambda d: d["hyperedges"][0]["nodes"].append(999),
                         lambda d: d["nodes"][0].update(x=float("nan"))):
            data = fixture()
            mutation(data)
            with self.assertRaises(ValueError):
                layout.PolygonProblem(data)

    def test_optimization_improves_irregular_face_without_losing_members(self):
        data = fixture()
        data["nodes"] = data["nodes"][:6]
        data["hyperedges"] = [data["hyperedges"][0]]
        data["hyperedges"][0]["nodes"] = [10, 11, 12, 13, 14, 15]
        problem = layout.PolygonProblem(data)
        with redirect_stdout(io.StringIO()):
            xy, report = layout.optimize(problem, iterations=90, max_cycles=2, initialization='input')
        self.assertLess(report["after"]["mean_regularity_ratio"], report["before"]["mean_regularity_ratio"])
        self.assertEqual(report["after"]["invalid_faces"], 0)
        self.assertEqual(report["after"]["degenerate_faces"], 0)
        self.assertGreater(report["after"]["minimum_node_distance"], 1e-6)
        layout.assert_preserved(data, layout.make_output(problem, xy))

    def test_raw_paper_equations_against_independent_geometry(self):
        problem = layout.PolygonProblem(fixture())
        xy = problem.initial

        def geometry(members):
            ids = list(members)
            if len(ids) >= 3:
                c = np.asarray(MultiPoint(xy[ids]).convex_hull.centroid.coords[0])
                ids.sort(key=lambda i: np.arctan2(*(xy[i]-c)[::-1]))
            points = xy[ids]
            lengths = np.linalg.norm(points-np.roll(points, -1, axis=0), axis=1)
            if len(ids) >= 3:
                polygon = Polygon(points)
                center = np.asarray(polygon.centroid.coords[0])
                regularity = lengths.sum()**2-4*len(ids)*np.tan(np.pi/len(ids))*polygon.area
            else:
                center, regularity = points.mean(axis=0), 0.
            return ids, lengths, center, regularity

        faces = {e: geometry(problem.members[e]) for e in problem.active}
        expected = np.zeros(4)
        for ids, lengths, center, regularity in faces.values():
            expected[0] += regularity
            expected[1] += np.sum((lengths-1)**2)
        for a, f in faces.items():
            for b, g in faces.items():
                if b <= a:
                    continue
                common = set(f[0]) & set(g[0]); n, m = len(f[0]), len(g[0])
                if len(common) >= 3:
                    expected[3] += geometry(common)[3]
                    for ids, lengths, _, _ in (f, g):
                        positions = [i for i, v in enumerate(ids) if v in common]
                        for pos, stop in zip(positions, positions[1:]+[positions[0]+len(ids)]):
                            arc = sum(lengths[i%len(ids)] for i in range(pos, stop))
                            expected[3] += (arc-len(ids)/len(common))**2
                else:
                    if len(common) == 1:
                        pivot = xy[next(iter(common))]
                        v, w = f[2]-pivot, g[2]-pivot
                        actual = np.arccos(np.clip(v@w/(np.linalg.norm(v)*np.linalg.norm(w)), -1, 1))
                        required = np.pi*((n-2)/(2*n)+(m-2)/(2*m))+.03
                    else:
                        actual = np.linalg.norm(f[2]-g[2])
                        required = (.5/np.sin(np.pi/n)+.5/np.sin(np.pi/m)+.1 if not common
                                    else .5/np.tan(np.pi/n)+.5/np.tan(np.pi/m))
                    expected[2] += max(0., required-actual)**2
        _, terms = PaperEnergy(problem).evaluate(xy, details=True)
        np.testing.assert_allclose([terms[t] for t in TERMS[:4]], expected, rtol=1e-10, atol=1e-9)

    def test_regular_161_gon_has_zero_shape_and_length_energy(self):
        n = 161
        angles = 2*np.pi*np.arange(n)/n
        xy = .5/np.sin(np.pi/n)*np.column_stack((np.cos(angles), np.sin(angles)))
        data = {'nodes': [{'id': i, 'x': x, 'y': y} for i, (x, y) in enumerate(xy)],
                'hyperedges': [{'id': 0, 'nodes': list(range(n))}]}
        problem = layout.PolygonProblem(data)
        energy = PaperEnergy(problem)
        w = np.array([.3, .16, 0., 0., 1., 0., 10.])
        value, gradient = energy.evaluate(xy, w)
        self.assertLess(abs(value), 1e-7)
        self.assertLess(np.max(np.abs(gradient)), 1e-7)
        # The high-cardinality energy must not accidentally be divided by n^2.
        xy[10] += [2., -.5]
        pts = xy[layout.starrize(problem.members[0], xy)]
        expected = np.linalg.norm(pts-np.roll(pts, 1, axis=0), axis=1).sum()**2
        expected -= 4*n*np.tan(np.pi/n)*Polygon(pts).area
        self.assertAlmostEqual(energy.evaluate(xy, details=True)[1]['regularity'], expected, places=7)

    def test_cached_full_swaps_match_energy_and_converge_on_fixture(self):
        problem = layout.PolygonProblem(fixture())
        energy = PaperEnergy(problem)
        x = problem.initial.copy()
        weights = np.array([.3, .16, .36, .18, 1., .05, 10.])
        all_pairs = {(int(a), int(b)) for m in problem.members for a in m for b in m if a < b}
        self.assertEqual(len(energy.swap_pairs)+energy.equivalent_pairs, len(all_pairs))
        for _ in range(50):
            before = energy.evaluate(x, weights)[0]
            x, info = energy.swap_sweep(x, weights)
            after = energy.evaluate(x, weights)[0]
            self.assertAlmostEqual(before-after, info['gain'], places=7)
            if not info['accepted']:
                break
        self.assertEqual(info['accepted'], 0)
        for a, b in all_pairs:
            candidate = x.copy(); candidate[[a, b]] = candidate[[b, a]]
            self.assertGreaterEqual(energy.evaluate(candidate, weights)[0], after-1e-7)

    def test_dual_gradient_and_cached_swaps(self):
        problem = layout.PolygonProblem(fixture())
        energy = JointEnergy(problem)
        x = np.vstack((problem.initial, np.random.default_rng(9).normal(size=(energy.m, 2)))).ravel()
        weights = np.array([.2, .2, .2, .2, .3, .05, 2.])
        value, analytic = energy.evaluate(x, weights)
        numeric = np.zeros_like(x)
        for i in range(len(x)):
            step = np.zeros_like(x); step[i] = 2e-6
            numeric[i] = (energy.evaluate(x+step, weights)[0]-energy.evaluate(x-step, weights)[0])/4e-6
        np.testing.assert_allclose(analytic, numeric, atol=3e-5, rtol=3e-4)
        after, info = energy.swap_sweep(x, weights)
        self.assertAlmostEqual(value-energy.evaluate(after, weights)[0], info['gain'], places=7)

    def test_consensus_and_monogons_preserve_shared_coordinates(self):
        data = fixture()
        problem = layout.PolygonProblem(data)
        x, info = regular_polygon_consensus(problem, problem.initial, iterations=20)
        self.assertEqual(x.shape, problem.initial.shape)
        self.assertTrue(np.isfinite(x).all())
        layout.assert_preserved(data, layout.make_output(problem, x))
        before = x.copy()
        report = orient_monogons(problem, x, PaperEnergy(problem))
        np.testing.assert_array_equal(x, before)
        self.assertEqual(len(report['glyphs']), 1)
        center = report['glyphs'][0]['center']
        self.assertAlmostEqual(np.linalg.norm(center-x[4]), report['distance'])

    def test_single_node_and_empty_edge_are_serializable(self):
        data = {'nodes': [{'id': 0, 'x': 3., 'y': -2.}],
                'hyperedges': [{'id': 0, 'nodes': [0]}, {'id': 1, 'nodes': []}]}
        p = layout.PolygonProblem(data)
        xy, report = layout.optimize(p, initialization='input', mode='paper', iterations=2,
                                     max_cycles=1, progress=False)
        layout.assert_preserved(data, layout.make_output(p, xy))
        json.dumps(report, allow_nan=False)

    def test_coincident_input_can_expand_to_a_valid_polygon(self):
        data = {'nodes': [{'id': i, 'x': 0., 'y': 0.} for i in range(5)],
                'hyperedges': [{'id': 0, 'nodes': list(range(5))}]}
        p = layout.PolygonProblem(data)
        xy, report = layout.optimize(p, initialization='input', mode='scalable', iterations=40,
                                     max_cycles=1, progress=False)
        self.assertEqual(report['after']['degenerate_faces'], 0)
        self.assertGreater(report['after']['minimum_node_distance'], .1)
        layout.assert_preserved(data, layout.make_output(p, xy))

    def test_default_cli_replaces_original_with_exact_byte_backup(self):
        data = fixture()
        data['nodes'] = data['nodes'][:6]
        data['hyperedges'] = [{'id': 1, 'nodes': list(range(10, 16))}]
        payload = json.dumps(data, indent=1).encode('utf-8')
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                Path('original.json').write_bytes(payload)
                argv = ['solve_polygon_layout.py', '--initialization', 'input',
                        '--iterations', '60', '--max-cycles', '1']
                with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                    layout.main()
                self.assertEqual(Path('original_before_polygon_layout.json').read_bytes(), payload)
                result = json.loads(Path('original.json').read_text(encoding='utf-8'))
                layout.assert_preserved(data, result)
                self.assertNotEqual(Path('original.json').read_bytes(), payload)
                self.assertTrue(Path('polygon_layout_comparison.svg').is_file())
                report = json.loads(Path('polygon_layout_report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['backup'], 'original_before_polygon_layout.json')
            finally:
                os.chdir(old_cwd)

    def test_same_file_shape_run_rejects_regression_without_changing_bytes(self):
        data = {'nodes': [{'id': i, 'x': x, 'y': y} for i, (x, y) in enumerate(
                    [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])],
                'hyperedges': [{'id': 0, 'nodes': [0, 1, 2, 3]}]}
        problem = layout.PolygonProblem(data)
        candidate = problem.initial*np.array([3., .3])
        # The write guard must measure actual geometry, even if an optimizer's
        # reported score erroneously claims an improvement.
        report = {'before': layout.measure(problem, problem.initial),
                  'after': layout.measure(problem, problem.initial),
                  'initial_energy': 10., 'paper_baseline_energy': 9., 'final_energy': 8.}
        payload = json.dumps(data, indent=1).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'original.json'
            source.write_bytes(payload)
            def run(destination):
                argv = ['solve_polygon_layout.py', str(source), str(destination),
                        '--report', str(root/'report.json'), '--preview', str(root/'preview.svg')]
                with patch('sys.argv', argv), patch.object(layout, 'optimize', return_value=(candidate, copy.deepcopy(report))), redirect_stdout(io.StringIO()):
                    layout.main()
                return json.loads((root/'report.json').read_text(encoding='utf-8'))
            rejected = run(source)
            self.assertEqual(source.read_bytes(), payload)
            self.assertFalse(rejected['wrote_layout'])
            self.assertFalse(rejected['inplace_replacement_allowed'])
            self.assertIn('mean_regularity_worsened', rejected['shape_replacement_check']['reasons'])
            self.assertFalse(list(root.glob('original_before_*.json')))
            # Explicit separate candidates remain available for experiments.
            separate = root/'candidate.json'
            inspected = run(separate)
            self.assertTrue(separate.is_file())
            self.assertTrue(inspected['wrote_layout'])
            self.assertFalse(inspected['inplace_replacement_allowed'])
            self.assertEqual(source.read_bytes(), payload)

    def test_shape_guard_rejects_local_damage_despite_better_mean(self):
        before = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.],
                           [4., 0.], [14., 0.], [14., 1.], [4., 1.]])
        after = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.],
                          [4., 0.], [5., 0.], [5., 1.], [4., 1.]])
        data = {'nodes': [{'id': i, 'x': x, 'y': y} for i, (x, y) in enumerate(before)],
                'hyperedges': [{'id': 1, 'nodes': [0, 1, 2, 3]}, {'id': 2, 'nodes': [4, 5, 6, 7]}]}
        p = layout.PolygonProblem(data)
        allowed, info = layout.shape_replacement_check(p, layout.make_output(p, after))
        self.assertFalse(allowed)
        self.assertLess(info['candidate']['mean_regularity_ratio'], info['before']['mean_regularity_ratio'])
        self.assertEqual(info['degraded_face_ids'], [1])

    def test_shape_guard_protects_large_faces_and_allows_geometry_repair(self):
        angle = np.linspace(0., 2*np.pi, 40, endpoint=False)
        circle = np.column_stack((np.cos(angle), np.sin(angle)))
        before = np.vstack((circle, [[4., 0.], [14., 0.], [14., 1.], [4., 1.]]))
        after = np.vstack((circle*[1.4, 1.], [[4., 0.], [5., 0.], [5., 1.], [4., 1.]]))
        data = {'nodes': [{'id': i, 'x': x, 'y': y} for i, (x, y) in enumerate(before)],
                'hyperedges': [{'id': 1, 'nodes': list(range(40))}, {'id': 2, 'nodes': list(range(40, 44))}]}
        p = layout.PolygonProblem(data)
        allowed, info = layout.shape_replacement_check(p, layout.make_output(p, after))
        self.assertFalse(allowed)
        self.assertIn('large_face_regularity_worsened', info['reasons'])
        self.assertNotIn('mean_regularity_worsened', info['reasons'])
        for n in data['nodes']:
            n['x'] = n['y'] = 0.
        broken = layout.PolygonProblem(data)
        self.assertTrue(layout.shape_replacement_check(broken, layout.make_output(broken, before))[0])


if __name__ == "__main__":
    unittest.main()
