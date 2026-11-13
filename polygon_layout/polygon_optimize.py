"""Alternating L-BFGS and exhaustive coordinate swaps for polygon layouts."""
import time

import numpy as np
from scipy.optimize import minimize

from .polygon_paper import PaperEnergy, JointEnergy, PAPER_WEIGHTS, TERMS, hu_layout


def optimize_reproduction(problem, iterations=600, seed=7, mode='scalable',
                          initialization=None, dual=False, max_cycles=6,
                          fit_weight=1., collision_weight=.05, convexity_weight=10., sfdp=None,
                          start_xy=None, progress=True):
    # Local import avoids coupling the numerical kernel to the file I/O layer.
    from solve_polygon_layout import measure, starrize, overlap_statistics, face_metrics
    if mode not in ('paper', 'scalable') or iterations < 1 or max_cycles < 1:
        raise ValueError('Invalid mode or optimization budget')
    if not np.isfinite([fit_weight, collision_weight, convexity_weight]).all() or min(fit_weight, collision_weight, convexity_weight) < 0:
        raise ValueError('Weights must be finite and nonnegative')
    started = time.monotonic()
    initialization = initialization or ('hu-clique' if mode == 'paper' else 'hu-bipartite')
    rng = np.random.default_rng(seed)
    initializer = {'algorithm': initialization}
    dual_xy = None
    if start_xy is not None:
        primal = np.asarray(start_xy).copy()
        initializer['algorithm'] = 'provided coordinates'
    elif initialization.startswith('hu-'):
        placed, initializer = hu_layout(problem, seed, initialization[3:], sfdp)
        primal = placed[:problem.n]
        if len(placed) > problem.n:
            dual_xy = placed[problem.n:]
    elif initialization == 'input':
        primal = problem.initial.copy()
    elif initialization in ('circular', 'random'):
        if initialization == 'circular':
            angles = 2*np.pi*rng.permutation(problem.n)/problem.n
            primal = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            primal = rng.normal(size=(problem.n, 2))
    else:
        raise ValueError('Unknown initialization')
    consensus_source = primal.copy()
    # Common scaling changes neither the incidence nor the initial ordering.
    lengths = []
    for edge in problem.active:
        p = primal[starrize(problem.members[edge], primal)]
        lengths.extend(np.linalg.norm(p-np.roll(p, -1, axis=0), axis=1))
    lengths = np.asarray(lengths)
    initial_scale = float(lengths.sum()/max(lengths@lengths, 1e-12)) if len(lengths) else 1.
    origin = primal.mean(axis=0)
    primal = (primal-origin)*initial_scale
    if dual_xy is not None:
        dual_xy = (dual_xy-origin)*initial_scale
    _, inverse, counts = np.unique(primal, axis=0, return_inverse=True, return_counts=True)
    repeated = counts[inverse] > 1
    primal[repeated] += rng.normal(0, .01, (int(repeated.sum()), 2))
    energy = JointEnergy(problem) if dual else PaperEnergy(problem)
    weights = np.array([.2, .2, .2, .2, 0., 0., 0.]) if dual else PAPER_WEIGHTS.copy()
    if dual:
        if dual_xy is None:
            dual_xy = energy.primal.centers(primal)
            dual_xy += rng.normal(0, .01, dual_xy.shape)
        xy = np.vstack((primal, dual_xy))
    else:
        xy = primal.copy()
    initial_xy = xy.copy()
    initial_energy, initial_terms = energy.evaluate(xy, weights, details=True)
    baseline = (problem.initial-problem.origin)/problem.unit
    baseline_for_energy = np.vstack((baseline, energy.primal.centers(baseline))) if dual else baseline
    source_energy = energy.evaluate(baseline_for_energy, weights)[0]
    stages, swaps = [], []
    # Fix two vertices during line searches as specified in section 3.3.
    # Swaps are discrete moves and may change their positions between searches.
    left = 0
    right = int(np.argmax(np.sum((xy-xy[left])**2, axis=1)))
    fixed = sorted(set([2*left, 2*left+1, 2*right, 2*right+1]))
    free = np.setdiff1d(np.arange(xy.size), fixed)

    def continuous(stage, budget, current_weights):
        nonlocal xy
        used, restarts = 0, 0
        start_value = energy.evaluate(xy, current_weights)[0]
        scale = max(start_value/1000, 1.)
        best_value = start_value
        stationary = False
        messages = []
        if not len(free):
            stages.append({'stage': stage, 'iterations': 0, 'restarts': 0,
                           'start_energy': start_value, 'final_energy': start_value,
                           'stationary': True, 'solver_messages': ['No free coordinates']})
            return
        while used < budget:
            template = xy.ravel().copy()

            def fun(vector):
                x = template.copy(); x[free] = vector
                value, gradient = energy.evaluate(x, current_weights)
                return value/scale, gradient[free]/scale

            result = minimize(fun, template[free], jac=True, method='L-BFGS-B',
                              options={'maxiter': min(150, budget-used), 'maxls': 50,
                                       'maxcor': 20, 'ftol': 1e-12, 'gtol': 1e-7})
            proposal = template.copy(); proposal[free] = result.x
            value = energy.evaluate(proposal, current_weights)[0]
            old_value = best_value
            if np.isfinite(value) and value <= best_value:
                xy = proposal.reshape(xy.shape)
                best_value = value
            used += max(int(result.nit), 1)
            messages.append(str(result.message))
            restarts += 1
            if progress:
                metrics = measure(problem, xy[:problem.n])
                print(f'  {stage} {used}: E={best_value:.6g}, regularity={metrics["mean_regularity_ratio"]:.4g}, concave={metrics["concave_faces"]}', flush=True)
            if old_value-best_value <= 1e-10*max(1., abs(old_value)):
                stationary = bool(result.success)
                break
        stages.append({'stage': stage, 'iterations': used, 'restarts': restarts,
                       'start_energy': start_value, 'final_energy': best_value,
                       'stationary': stationary, 'solver_messages': messages})

    def alternate(name, current_weights):
        for cycle in range(max_cycles):
            continuous(f'{name}_{cycle+1}', iterations, current_weights)
            before = energy.evaluate(xy, current_weights)[0]
            proposed, info = energy.swap_sweep(xy, current_weights)
            after = energy.evaluate(proposed, current_weights)[0]
            # Independently re-evaluate the full objective after the cached sweep.
            if after > before + 1e-7*max(1., abs(before)):
                raise RuntimeError('Cached swap sweep disagrees with the full objective')
            xy[:] = proposed
            info.update(stage=name, cycle=cycle+1, before=before, after=after)
            swaps.append(info)
            if progress:
                print(f'  {name} exhaustive swap sweep {cycle+1}: accepted={info["accepted"]}, E={after:.6g}', flush=True)
            if not info['accepted']:
                return True
        # A capped run must not finish immediately after a discrete move.
        continuous(f'{name}_final_refine', iterations, current_weights)
        return False

    paper_swaps_converged = alternate('paper', weights)
    paper_xy = xy.copy()
    paper_energy, paper_terms = energy.evaluate(xy, weights, details=True)
    extension_weights = weights.copy()
    extended_swaps_converged = None
    consensus_info = None
    if mode == 'scalable':
        from .polygon_consensus import regular_polygon_consensus
        extension_weights[4:] = [fit_weight, collision_weight, convexity_weight]
        # A second starting basin counteracts the collapsed large faces seen in
        # raw energy minimization. Retain the paper solution as a fallback.
        consensus, consensus_info = regular_polygon_consensus(problem, consensus_source)
        candidate = np.vstack((consensus, energy.primal.centers(consensus))) if dual else consensus
        previous = xy.copy()
        previous_value = energy.evaluate(previous, extension_weights)[0]
        xy = candidate
        # Free all coordinates for the extension: pinning two full positions
        # also fixes a distance, impeding expansion of high-cardinality faces.
        free = np.arange(xy.size)
        continuous('consensus_initial_refine', iterations, extension_weights)
        consensus_info['refined_energy'] = energy.evaluate(xy, extension_weights)[0]
        consensus_info['paper_solution_extended_energy'] = previous_value
        consensus_info['accepted'] = consensus_info['refined_energy'] < previous_value
        if not consensus_info['accepted']:
            xy = previous
        extended_swaps_converged = alternate('regular_polygon_fit', extension_weights)
    final_energy, final_terms = energy.evaluate(xy, extension_weights, details=True)
    final_paper_energy = energy.evaluate(xy, weights)[0]
    # Save the scale conversion rather than conflating resized display geometry
    # with the unit-side optimization objective.
    from .polygon_monogons import orient_monogons
    monogons = orient_monogons(problem, xy[:problem.n], energy.primal if dual else energy, seed)
    translation = xy[:problem.n].mean(axis=0)
    xy -= translation
    old_rms = np.sqrt(np.mean(np.sum((problem.initial-problem.origin)**2, axis=1)))
    new_rms = np.sqrt(np.mean(np.sum(xy[:problem.n]**2, axis=1)))
    world_scale = old_rms/new_rms if min(old_rms, new_rms) > 1e-12 else problem.unit
    output = xy[:problem.n]*world_scale + problem.origin
    for glyph in monogons['glyphs']:
        glyph['center'] = ((np.asarray(glyph['center'])-translation)*world_scale+problem.origin).tolist()
    monogons['distance_world'] = monogons['distance']*world_scale
    monogons['radius_world'] = monogons['radius']*world_scale
    after = measure(problem, output)
    if after['invalid_faces'] or after['degenerate_faces']:
        raise RuntimeError('Invalid final polygon; the input was not overwritten')
    report = {
        'mode': mode, 'method': 'Hu initialization + raw PR/PA/PS/PI + area centroids + L-BFGS + exhaustive pair swaps',
        'paper': 'https://arxiv.org/html/2108.00671v1',
        'context_paper': 'https://arxiv.org/html/2407.19621v1',
        'initializer': initializer, 'dual': dual, 'seed': seed,
        'nodes': problem.n, 'hyperedges': len(problem.members), 'incidence_preserved': True,
        'paper_weights': weights.tolist(), 'final_weights': extension_weights.tolist(),
        'energy_terms': list(TERMS),
        'energy_comparison_note': 'Initial and paper_baseline use paper_weights; final uses final_weights. Compare energies within the same weight set.',
        'source_energy_normalized_by_input_side_length': source_energy,
        'initial_energy': initial_energy, 'initial_terms': initial_terms,
        'paper_baseline_energy': paper_energy, 'paper_baseline_terms': paper_terms,
        'final_energy': final_energy, 'final_terms': final_terms,
        'final_paper_energy': final_paper_energy,
        'before': measure(problem, problem.initial, reorder=False),
        'initializer_metrics': measure(problem, initial_xy[:problem.n]),
        'paper_baseline_metrics': measure(problem, paper_xy[:problem.n]), 'after': after,
        'face_metrics_before': face_metrics(problem, problem.initial, reorder=False),
        'face_metrics_after': face_metrics(problem, output),
        'before_overlaps': overlap_statistics(problem, problem.initial, reorder=False),
        'after_overlaps': overlap_statistics(problem, output),
        'target_side_length_world': world_scale,
        'pair_sweeps': swaps, 'stages': stages,
        'paper_pair_search_converged': paper_swaps_converged,
        'extension_pair_search_converged': extended_swaps_converged,
        'max_cycles': max_cycles, 'iterations_per_cycle': iterations,
        'fixed_coordinate_indices_during_line_search': fixed,
        'extension_fixed_coordinates': [], 'consensus_initializer': consensus_info,
        'monogons': monogons,
        'extensions': ['regular n-gon consensus initialization', 'regular n-gon least-squares fit',
                       'short-range vertex repulsion', 'hull area deficit', 'free scale during extended line searches'] if mode == 'scalable' else [],
        'elapsed_seconds': time.monotonic()-started,
        'reproduction_limits': ['HGPolyVis source and original numerical defaults are not publicly provided',
                                'Unary glyph orientations are in the report; binary lens rendering is not part of XY layout',
                                'Empty input hyperedges are preserved and excluded from polygon energies',
                                '2023/2024 topology-changing hypergraph simplification is not applied'],
    }
    if dual:
        report['dual_coordinates'] = (xy[problem.n:]*world_scale+problem.origin).tolist()
    return output, report
