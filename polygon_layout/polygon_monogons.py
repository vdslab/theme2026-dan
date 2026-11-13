"""Post-layout monogon orientation optimization, Qu et al. equations (7)-(8).

The vertex XY coordinates stay fixed. Results are glyph metadata in the report;
the existing original.json schema and its incidence are not altered.
"""
import numpy as np
from scipy.optimize import minimize


def orient_monogons(problem, xy, energy, seed=7, distance=.5, radius=.15):
    if distance <= radius or radius <= 0:
        raise ValueError('Monogon center distance must exceed its positive radius')
    edges = [i for i, m in enumerate(problem.members) if len(m) == 1]
    if not edges:
        return {'glyphs': [], 'energy': 0., 'distance': distance, 'radius': radius}
    centers = energy.centers(xy)
    index = {edge: i for i, edge in enumerate(edges)}
    pivots = np.asarray([xy[problem.members[e][0]] for e in edges])
    pairs = []
    # Equation (8) is a directed sum over each monogon and its incident faces.
    # Thus an incident monogon-monogon pair appears twice, exactly as written.
    for edge in edges:
        vertex = problem.members[edge][0]
        for other, members in enumerate(problem.members):
            if edge != other and vertex in members:
                pairs.append((index[edge], other, .1 if other in index else 1.))

    def objective(angles):
        vectors = distance*np.column_stack((np.cos(angles), np.sin(angles)))
        tips = pivots+vectors
        derivatives = np.column_stack((-vectors[:, 1], vectors[:, 0]))
        gradient = np.zeros(len(edges)); value = 0.
        for i, other, weight in pairs:
            target = tips[index[other]] if other in index else centers[other]
            delta = tips[i]-target
            squared = max(float(delta@delta), 1e-16)
            value += weight/squared
            force = -2*weight*delta/(squared*squared)
            gradient[i] += force@derivatives[i]
            if other in index:
                gradient[index[other]] -= force@derivatives[index[other]]
        return value, gradient

    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, len(edges))
    result = minimize(objective, angles, jac=True, method='L-BFGS-B',
                      options={'maxiter': 600, 'maxls': 40, 'ftol': 1e-12})
    final = result.x if result.fun <= objective(angles)[0] else angles
    tips = pivots+distance*np.column_stack((np.cos(final), np.sin(final)))
    return {'glyphs': [{'hyperedge_id': problem.data['hyperedges'][e]['id'],
                        'angle': float(final[i]), 'center': tips[i].tolist()}
                       for i, e in enumerate(edges)],
            'energy': float(objective(final)[0]), 'distance': distance, 'radius': radius,
            'iterations': int(result.nit), 'converged': bool(result.success),
            'note': 'Post-layout glyph orientations; XY positions are fixed. Sizes are chosen defaults.'}
