"""Exhaustive judgments must detect a single damaged face, even below old 10%."""
from contextlib import redirect_stdout
import copy
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import solve_polygon_layout as layout
from polygon_layout.polygon_arrange import PlacementAudit
from polygon_layout.polygon_quality import all_face_judgment, write_quality_review


def squares(count=2):
    data = {'nodes': [], 'hyperedges': []}
    for edge in range(count):
        start = 4*edge
        for offset, (x, y) in enumerate(((0., 0.), (1., 0.), (1., 1.), (0., 1.))):
            data['nodes'].append({'id': start+offset, 'x': x+3*edge, 'y': y})
        data['hyperedges'].append({'id': edge, 'label': f'face {edge}', 'nodes': list(range(start, start+4))})
    return data


class QualityTests(unittest.TestCase):
    def test_one_slightly_damaged_small_face_cannot_hide_behind_better_mean(self):
        data = squares()
        data['nodes'][5]['x'] += 9.; data['nodes'][6]['x'] += 9.
        p = layout.PolygonProblem(data)
        xy = p.initial.copy()
        xy[1, 0] = xy[2, 0] = 1.05
        xy[5, 0] = xy[6, 0] = 4.
        output = layout.make_output(p, xy)
        allowed, check = layout.shape_replacement_check(p, output)
        self.assertLess(check['candidate']['mean_regularity_ratio'], check['before']['mean_regularity_ratio'])
        self.assertFalse(allowed)
        self.assertEqual(check['degraded_face_ids'], [0])
        self.assertTrue(layout.shape_replacement_check(p, output, relative_tolerance=.01)[0])

    def test_an_invalid_reference_face_does_not_exempt_other_valid_faces(self):
        data = squares(); data['hyperedges'][0]['nodes'] = [0, 2, 1, 3]
        p = layout.PolygonProblem(data)
        xy = p.initial.copy(); xy[5, 0] = xy[6, 0] = 5.
        allowed, check = layout.shape_replacement_check(p, layout.make_output(p, xy))
        self.assertFalse(allowed)
        self.assertEqual(check['all_face_judgment']['changes']['repaired'], 1)
        self.assertEqual(check['degraded_face_ids'], [1])

    def test_non_regression_is_not_a_claim_that_all_polygons_are_good(self):
        data = squares(1); data['nodes'][1]['x'] = data['nodes'][2]['x'] = 10.
        p = layout.PolygonProblem(data)
        result = all_face_judgment(p, p)
        self.assertTrue(result['all_faces_non_regressing'])
        self.assertFalse(result['all_faces_pass_quality'])
        self.assertEqual(result['quality_status'], 'incomplete')
        self.assertEqual(result['quality_fail_count'], 1)

    def test_all_hyperedges_are_accounted_for_and_glyphs_are_not_area_passes(self):
        data = squares(13)
        data['hyperedges'] += [{'id': 'binary', 'nodes': [0, 1]},
                               {'id': 'unary', 'nodes': [0]}, {'id': 'empty', 'nodes': []}]
        p = layout.PolygonProblem(data)
        result = all_face_judgment(p, p)
        self.assertEqual(result['checked_faces'], 13)
        self.assertEqual(result['total_hyperedges'], 16)
        self.assertTrue(result['all_hyperedges_accounted_for'])
        self.assertEqual(len(result['non_polygon_hyperedges']), 3)
        self.assertEqual({r['id'] for r in result['faces']}, set(range(13)))

    def test_geometric_transforms_do_not_change_dimensionless_quality(self):
        p = layout.PolygonProblem(squares())
        theta = .73
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        output = layout.make_output(p, p.initial@rotation.T*5+[8., -2.], reorder=False)
        result = all_face_judgment(p, layout.PolygonProblem(output))
        self.assertTrue(result['all_faces_non_regressing'])
        self.assertTrue(result['all_faces_pass_quality'])
        self.assertEqual(result['changes']['unchanged'], 2)

    def test_membership_change_is_rejected_instead_of_omitting_a_face(self):
        data = squares(); p = layout.PolygonProblem(data)
        changed = copy.deepcopy(data); changed['hyperedges'][0]['nodes'].pop()
        with self.assertRaisesRegex(ValueError, 'incidence'):
            all_face_judgment(p, layout.PolygonProblem(changed))

    def test_placement_checks_every_face_with_zero_default_tolerance(self):
        p = layout.PolygonProblem(squares())
        xy = p.initial.copy(); xy[1, 0] = xy[2, 0] = 1.05
        ok, check = PlacementAudit(p, p.initial).shape_check(xy)
        self.assertFalse(ok)
        self.assertEqual(check['checked_faces'], 2)
        self.assertEqual(check['regression_count'], 1)

    def test_full_review_contains_more_than_eight_faces_and_escapes_labels(self):
        data = squares(13); data['hyperedges'][0]['label'] = '<script>example</script>'
        p = layout.PolygonProblem(data); judgment = all_face_judgment(p, p)
        class CounterParser(HTMLParser):
            def __init__(self):
                super().__init__(); self.cards = 0; self.scripts = 0
            def handle_starttag(self, tag, attrs):
                self.cards += tag == 'article'
                self.scripts += tag == 'script'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'review.html'
            write_quality_review(path, p, p, judgment)
            document = path.read_text(encoding='utf-8')
            parser = CounterParser(); parser.feed(document)
            self.assertEqual(parser.cards, 13)
            self.assertEqual(parser.scripts, 1)
            self.assertIn('&lt;script&gt;example&lt;/script&gt;', document)

    def test_audit_only_does_not_optimize_write_layout_or_modify_reference(self):
        data = squares(); candidate = copy.deepcopy(data)
        candidate['nodes'][1]['x'] = candidate['nodes'][2]['x'] = 1.05
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'input.json'; reference = root/'reference.json'
            source.write_text(json.dumps(candidate, indent=1), encoding='utf-8')
            reference.write_text(json.dumps(data, indent=1), encoding='utf-8')
            snapshots = source.read_bytes(), reference.read_bytes()
            argv = ['solve_polygon_layout.py', str(source), str(root/'unused.json'), '--audit-only',
                    '--audit-reference', str(reference), '--report', str(root/'audit.json'),
                    '--preview', str(root/'all.html'), '--quality-regularity-max', '1.00001']
            with patch('sys.argv', argv), patch.object(layout, 'optimize', side_effect=AssertionError('Optimizer called')), redirect_stdout(io.StringIO()):
                layout.main()
            self.assertEqual((source.read_bytes(), reference.read_bytes()), snapshots)
            self.assertFalse((root/'unused.json').exists())
            self.assertFalse(list(root.glob('*before*.json')))
            report = json.loads((root/'audit.json').read_text(encoding='utf-8'))
            self.assertFalse(report['wrote_layout'])
            self.assertEqual(report['all_face_judgment']['regression_count'], 1)
            self.assertEqual(report['all_face_judgment']['quality_fail_count'], 1)
            self.assertEqual(report['all_face_judgment']['limits']['regularity_ratio'], 1.00001)
            self.assertTrue((root/'all.html').is_file())

    def test_normal_cli_defaults_to_zero_regression_for_every_face(self):
        data = squares(); data['nodes'][5]['x'] = data['nodes'][6]['x'] = 13.
        p = layout.PolygonProblem(data)
        xy = p.initial.copy(); xy[1, 0] = xy[2, 0] = 1.05; xy[5, 0] = xy[6, 0] = 4.
        result = {'before': layout.measure(p, p.initial), 'after': layout.measure(p, xy),
                  'initial_energy': 10., 'paper_baseline_energy': 9., 'final_energy': 8.}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'input.json'
            source.write_text(json.dumps(data, indent=1), encoding='utf-8'); snapshot = source.read_bytes()
            argv = ['solve_polygon_layout.py', str(source), str(source),
                    '--report', str(root/'report.json'), '--preview', str(root/'preview.svg')]
            with patch('sys.argv', argv), patch.object(layout, 'optimize', return_value=(xy, result)), redirect_stdout(io.StringIO()):
                layout.main()
            report = json.loads((root/'report.json').read_text(encoding='utf-8'))
            self.assertFalse(report['wrote_layout'])
            self.assertEqual(source.read_bytes(), snapshot)
            self.assertEqual(report['all_face_judgment']['regression_count'], 1)
            self.assertEqual(report['all_face_judgment']['relative_regularity_tolerance'], 0.)
            self.assertTrue((root/'polygon_layout_all_faces.html').is_file())


if __name__ == '__main__':
    unittest.main()
