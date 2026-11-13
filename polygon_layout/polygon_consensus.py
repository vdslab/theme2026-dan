"""Global regular-polygon projection used only by the scalable extension.

Each face proposes an ordered regular polygon; a sparse least-squares solve
reconciles all proposals into ONE XY pair per original node. No node is split.
This is a local/global initialization, not an algorithm from HGPolyVis.
"""
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import factorized


def regular_polygon_consensus(problem, initial, iterations=300, tether=.002):
    from solve_polygon_layout import starrize
    xy = np.asarray(initial, dtype=float).copy()
    origin = xy.mean(axis=0)
    xy -= origin
    source = xy.copy()
    rows, cols, values = [], [], []
    templates = []
    for edge in problem.active:
        members = problem.members[edge]
        n = len(members)
        radius = .5 / np.sin(np.pi/n)
        angles = 2*np.pi*np.arange(n)/n
        templates.append(radius*np.column_stack((np.cos(angles), np.sin(angles))))
        block = np.eye(n)-np.ones((n, n))/n
        rows.extend(np.repeat(members, n)); cols.extend(np.tile(members, n))
        values.extend(block.ravel())
    laplacian = coo_matrix((values, (rows, cols)), shape=(problem.n, problem.n)).tocsc()
    solve = factorized(laplacian+tether*eye(problem.n, format='csc'))
    for step in range(iterations):
        rhs = tether*source.copy()
        for edge, target in zip(problem.active, templates):
            ids = starrize(problem.members[edge], xy)
            points = xy[ids]-xy[ids].mean(axis=0)
            a = np.sum(target*points)
            b = np.sum(target[:, 0]*points[:, 1]-target[:, 1]*points[:, 0])
            angle = np.arctan2(b, a)
            co, si = np.cos(angle), np.sin(angle)
            rotation = np.array([[co, si], [-si, co]])
            rhs[ids] += target@rotation
        updated = solve(rhs)
        shift = np.max(np.linalg.norm(updated-xy, axis=1))
        xy = updated
        if shift < 1e-7:
            break
    return xy + origin, {'iterations': step+1, 'tether': tether,
                         'maximum_last_displacement': float(shift)}
