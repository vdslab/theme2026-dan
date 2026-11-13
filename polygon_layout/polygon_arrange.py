"""Shape-preserving XY placement after polygon regularization.

This is a post-layout extension inspired by polygon separation, not a Z-layer
solver or a reproduction of the 2023 simplify/reverse framework. A conservative
support-interval surrogate supplies gradients; actual polygon intersections
and shape guards decide whether a candidate may replace the reference.
"""
import ctypes
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Polygon

from .polygon_paper import kernel_library
from solve_polygon_layout import signed_area, measure, face_metrics, overlap_statistics
from .polygon_quality import judge_face_metrics


class PlacementEnergy:
    def __init__(self, problem, reference, directions=64, shape_tolerance=0.):
        self.n = problem.n
        self.lib = kernel_library()
        ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.lib.placement_create.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, dp, ctypes.c_int]
        self.lib.placement_create.restype = ctypes.c_void_p
        self.lib.placement_destroy.argtypes = [ctypes.c_void_p]
        self.lib.placement_limits.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self.lib.placement_evaluate.argtypes = [ctypes.c_void_p, dp, ctypes.c_double, ctypes.c_double,
                                               ctypes.c_double, dp, dp]
        self.lib.placement_evaluate.restype = ctypes.c_double
        reference = self.checked_xy(reference).reshape(-1, 2)
        members = [m[::-1] if len(m) >= 3 and signed_area(reference[m]) < 0 else m
                   for m in problem.members]
        offsets = np.r_[0, np.cumsum([len(m) for m in members])].astype(np.int32)
        ids = np.concatenate(members).astype(np.int32) if offsets[-1] else np.empty(0, np.int32)
        if not isinstance(directions, int) or directions < 4:
            raise ValueError('At least four separation directions are required')
        if not np.isfinite(shape_tolerance) or not 0 <= shape_tolerance <= 1:
            raise ValueError('Shape tolerance must be between zero and one')
        self.handle = self.lib.placement_create(self.n, len(problem.members), offsets, ids,
                                                self.checked_xy(reference), directions)
        if not self.handle:
            raise RuntimeError('Could not initialize placement kernel')
        self.lib.placement_limits(self.handle, shape_tolerance)

    def checked_xy(self, xy):
        x = np.ascontiguousarray(xy, dtype=float).ravel()
        if len(x) != 2*self.n or not np.isfinite(x).all():
            raise ValueError('Expected one finite XY coordinate pair per original node')
        return x

    def evaluate(self, xy, shape_weight=1000., shared_weight=.25, margin=.01, details=False):
        if not self.handle:
            raise RuntimeError('Placement energy is closed')
        x = self.checked_xy(xy)
        if not np.isfinite([shape_weight, shared_weight, margin]).all() or min(shape_weight, shared_weight, margin) < 0:
            raise ValueError('Weights and margin must be finite and nonnegative')
        gradient = np.zeros_like(x); terms = np.zeros(4)
        value = self.lib.placement_evaluate(self.handle, x, shape_weight, shared_weight, margin, gradient, terms)
        if details:
            return value, dict(zip(('rigid_shape_residual', 'nonincident_penetration', 'incident_penetration', 'shape_limit_penalty'), terms.tolist()))
        return value, gradient

    def close(self):
        if getattr(self, 'handle', None):
            self.lib.placement_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


class PlacementAudit:
    def __init__(self, problem, reference):
        self.problem = problem
        self.reference = np.asarray(reference).copy()
        self.reference_faces = face_metrics(problem, reference, reorder=False)
        self.reference_by_id = {f['id']: f for f in self.reference_faces}
        self.faces = list(problem.faces)
        self.pairs = np.asarray([(i, j) for i in range(len(self.faces)) for j in range(i+1, len(self.faces))
                                 if len(set(problem.members[self.faces[i]]) & set(problem.members[self.faces[j]])) <= 2],
                                dtype=int).reshape(-1, 2)
        self.common = np.asarray([len(set(problem.members[self.faces[i]]) & set(problem.members[self.faces[j]]))
                                  for i, j in self.pairs])
        self.reference_areas = np.asarray([Polygon(reference[problem.members[e]]).area
                                           for e in self.faces])
        self.minimum_distance_floor = (float(cKDTree(reference).query(reference, k=2)[0][:, 1].min())*.25
                                       if problem.n > 1 else 0.)

    def exact_overlap(self, xy):
        if not np.isfinite(xy).all():
            return None
        polygons = np.asarray([Polygon(xy[self.problem.members[e]]) for e in self.faces], dtype=object)
        if not all(p.is_valid and p.area > 1e-12 for p in polygons):
            return None
        if not len(self.pairs):
            return {'target_pairs': 0, 'target_area': 0., 'normalized_target_area': 0.,
                    'nonincident_pairs': 0, 'nonincident_area': 0.}
        a, b = self.pairs.T
        areas = shapely.area(shapely.intersection(polygons[a], polygons[b]))
        denominators = np.maximum(np.minimum(self.reference_areas[a], self.reference_areas[b]), 1e-12)
        nonincident = self.common == 0
        return {'target_pairs': int(np.count_nonzero(areas > 1e-9)),
                'target_area': float(areas.sum()),
                'normalized_target_area': float(np.sum(areas/denominators)),
                'nonincident_pairs': int(np.count_nonzero(areas[nonincident] > 1e-9)),
                'nonincident_area': float(areas[nonincident].sum())}

    def shape_check(self, xy, tolerance=0.):
        if not np.isfinite(xy).all():
            return False, {'error': 'nonfinite coordinates'}
        metrics = face_metrics(self.problem, xy, reorder=False)
        judgment = judge_face_metrics(self.reference_faces, metrics, tolerance)
        bad = list(judgment['regressed_face_ids'])
        for f in metrics:
            old = self.reference_by_id[f['id']]
            if f['hull_deficit'] > old['hull_deficit']+.02+1e-9 and f['id'] not in bad:
                bad.append(f['id'])
        areas = np.asarray([Polygon(xy[self.problem.members[e]]).area for e in self.faces])
        area_ratios = areas/np.maximum(self.reference_areas, 1e-12)
        # No candidate may achieve its gain by collapsing or inflating faces.
        area_ok = bool(np.all((area_ratios >= .8) & (area_ratios <= 1.2)))
        mean_before = np.mean([f['regularity_ratio'] for f in self.reference_faces]) if metrics else 1.
        mean_after = np.mean([f['regularity_ratio'] for f in metrics]) if metrics else 1.
        distance = float(cKDTree(xy).query(xy, k=2)[0][:, 1].min()) if self.problem.n > 1 else None
        distance_ok = distance is None or distance >= self.minimum_distance_floor
        return bool(not bad and area_ok and distance_ok and mean_after <= mean_before*1.02+1e-9), {
            'violating_face_ids': bad, 'area_bounds_satisfied': area_ok,
            'checked_faces': judgment['checked_faces'], 'regression_count': judgment['regression_count'],
            'quality_fail_count': judgment['quality_fail_count'],
            'mean_regularity_ratio': float(mean_after),
            'minimum_node_distance': distance, 'node_distance_floor': self.minimum_distance_floor,
            'node_distance_satisfied': distance_ok,
            'area_ratio_min': float(area_ratios.min()) if len(area_ratios) else 1.,
            'area_ratio_max': float(area_ratios.max()) if len(area_ratios) else 1.}


def optimize_placement(problem, iterations=250, rounds=3, shape_tolerance=0.,
                       shape_weights=(1000., 300., 100., 30.), progress=True):
    started = time.monotonic()
    if iterations < 1 or rounds < 1 or not 0 <= shape_tolerance <= 1:
        raise ValueError('Invalid placement budget or shape tolerance')
    if not len(shape_weights) or not np.isfinite(shape_weights).all() or min(shape_weights) < 0:
        raise ValueError('Expected a nonempty sequence of finite nonnegative shape weights')
    reference = problem.initial.copy()
    audit = PlacementAudit(problem, reference)
    before = audit.exact_overlap(reference)
    if before is None:
        raise ValueError('Placement requires valid nondegenerate input faces')
    unit = problem.unit
    normalized = (reference-problem.origin)/unit
    energy = PlacementEnergy(problem, normalized, shape_tolerance=shape_tolerance)
    best = reference.copy(); best_metrics = before
    history = []
    for round_index in range(rounds):
        changed = False
        for weight in shape_weights:
            x = ((best-problem.origin)/unit).ravel()
            result = minimize(lambda y: energy.evaluate(y, shape_weight=weight), x,
                              jac=True, method='L-BFGS-B', options={'maxiter': iterations, 'maxls': 40,
                                                                 'maxcor': 20, 'ftol': 1e-11, 'gtol': 1e-7})
            proposal = result.x.reshape(-1, 2)*unit+problem.origin
            accepted = False
            trials = []
            for alpha in (1., .5, .25, .125, .0625, .03125, .015625):
                candidate = best+alpha*(proposal-best)
                shape_ok, shape = audit.shape_check(candidate, shape_tolerance)
                trial = {'step_fraction': alpha, 'shape_ok': shape_ok, 'shape_audit': shape}
                trials.append(trial)
                if not shape_ok:
                    trial['rejection'] = 'shape_or_node_distance_limits'
                    continue
                actual = audit.exact_overlap(candidate)
                trial['actual_overlap'] = actual
                if (actual is not None and actual['normalized_target_area'] < best_metrics['normalized_target_area']-1e-7
                        and actual['target_pairs'] <= best_metrics['target_pairs']
                        and actual['nonincident_pairs'] <= best_metrics['nonincident_pairs']
                        and actual['target_area'] <= best_metrics['target_area']+1e-10
                        and actual['nonincident_area'] <= best_metrics['nonincident_area']+1e-10):
                    best, best_metrics = candidate, actual
                    accepted = changed = True
                    trial['rejection'] = None
                    break
                trial['rejection'] = 'invalid_geometry' if actual is None else 'actual_overlap_not_improved'
            history.append({'round': round_index+1, 'shape_weight': weight, 'iterations': int(result.nit),
                            'solver_message': str(result.message), 'solver_success': bool(result.success),
                            'accepted': accepted, 'trials': trials,
                            'step_fraction': alpha if accepted else 0., 'actual_overlap': best_metrics.copy(),
                            'shape_audit': shape})
            if progress:
                print(f'placement {round_index+1}, shape weight {weight:g}: accepted={accepted}, '
                      f'pairs={best_metrics["target_pairs"]}, normalized area={best_metrics["normalized_target_area"]:.5g}', flush=True)
        if not changed:
            break
    energy.close()
    metrics = measure(problem, best, reorder=False)
    valid, shape = audit.shape_check(best, shape_tolerance)
    if not valid or metrics['invalid_faces'] or metrics['degenerate_faces']:
        raise RuntimeError('Final placement failed the shape/validity guard')
    report = {'mode': 'arrange', 'method': 'rigid shape preservation + directional support separation + exact intersection acceptance',
              'source': 'https://arxiv.org/html/2308.05043v1#S4.SS3',
              'polygon_separation_source': 'https://arxiv.org/html/2108.00671v1#S3.SS2',
              'reproduction_note': 'Post-layout extension, not the paper simplification/reversal pipeline',
              'nodes': problem.n, 'hyperedges': len(problem.members), 'incidence_preserved': True,
              'before': measure(problem, reference, reorder=False), 'after': metrics,
              'before_overlaps': overlap_statistics(problem, reference, reorder=False),
              'after_overlaps': overlap_statistics(problem, best, reorder=False),
              'placement_before': before, 'placement_after': best_metrics,
              'shape_tolerance': shape_tolerance, 'shape_audit': shape,
              'shape_weights': list(shape_weights), 'directions': 64,
              'boundary_order_preserved': True,
              'iteration_limit': iterations, 'round_limit': rounds,
              'shape_limits': {'individual_ratio_factor': 1+shape_tolerance, 'mean_ratio_factor': 1.02,
                               'area_ratio_bounds': [.8, 1.2], 'hull_deficit_increase': .02,
                               'minimum_node_distance': audit.minimum_distance_floor},
              'native_kernel_sha256': hashlib.sha256(Path(__file__).with_name('polygon_kernel.cpp').read_bytes()).hexdigest(),
              'history': history, 'elapsed_seconds': time.monotonic()-started,
              'improved': best_metrics['normalized_target_area'] < before['normalized_target_area']-1e-7,
              'face_metrics_before': face_metrics(problem, reference, reorder=False),
              'face_metrics_after': face_metrics(problem, best, reorder=False),
              'limitations': ['Shared-node constraints can prevent complete separation',
                              'Pairs sharing three or more vertices are not directly repelled',
                              'Directional convex-hull penetration is a conservative search surrogate']}
    return best, report
