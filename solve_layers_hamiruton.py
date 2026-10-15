#!/usr/bin/env python3
import json
import argparse
import os
import sys
import time
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, LinearConstraint
from shapely.geometry import Polygon


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Assign layers to hyperedges in a hypergraph using a 2-phase QP approach."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="original.json",
        help="Path to the input JSON file (default: original.json)",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="layered_hypergraph.json",
        help="Path to the output JSON file (default: layered_hypergraph.json)",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-5,
        help="Regularization parameter to add to W elements to guarantee connectivity (default: 1e-5)",
    )
    return parser.parse_args()


def dsatur_coloring(N, C):
    """
    DSatur algorithm for graph coloring.
    N: number of vertices (0 to N-1)
    C: list of undirected edges (i, j) indicating overlap
    Returns a dictionary mapping vertex index to assigned color (integer >= 0).
    """
    adj = {i: [] for i in range(N)}
    for u, v in C:
        adj[u].append(v)
        adj[v].append(u)

    colors = {}
    degrees = {i: len(adj[i]) for i in range(N)}

    while len(colors) < N:
        # Find uncolored node with maximum saturation degree
        # Saturation degree is the number of distinct colors of already colored neighbors
        best_node = None
        max_sat = -1
        max_deg = -1

        for u in range(N):
            if u in colors:
                continue
            neighbor_colors = {colors[v] for v in adj[u] if v in colors}
            sat = len(neighbor_colors)
            deg = degrees[u]
            if sat > max_sat or (sat == max_sat and deg > max_deg):
                max_sat = sat
                max_deg = deg
                best_node = u

        # Color best_node with the smallest available color
        used_colors = {colors[v] for v in adj[best_node] if v in colors}
        color = 0
        while color in used_colors:
            color += 1
        colors[best_node] = color

    return colors


def solve_layer_assignment(input_path, output_path, reg_val):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading hypergraph data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    hyperedges = data.get("hyperedges", [])
    N = len(hyperedges)

    if N == 0:
        print("Error: No hyperedges found in input data.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(nodes)} nodes and {N} hyperedges.")

    # 1. Map nodes by ID for fast lookup
    node_by_id = {node["id"]: node for node in nodes}

    # 2. Compute polygon for each hyperedge using the node order
    print("Computing polygons from node order using Shapely...")
    polygons = []
    for he in hyperedges:
        he_nodes = he.get("nodes", [])
        pts = []
        for nid in he_nodes:
            if nid in node_by_id:
                pt = (node_by_id[nid]["x"], node_by_id[nid]["y"])
                # Deduplicate consecutive identical points
                if not pts or pts[-1] != pt:
                    pts.append(pt)
            else:
                print(
                    f"Warning: Node ID {nid} in hyperedge {he['id']} not found in nodes list.",
                    file=sys.stderr,
                )

        # Remove closing point if it matches the starting point (Shapely does it automatically)
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts.pop()

        if len(pts) >= 3:
            try:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                polygons.append(poly)
            except Exception as e:
                print(
                    f"Warning: Could not create valid polygon for hyperedge {he['id']}: {e}",
                    file=sys.stderr,
                )
                polygons.append(None)
        else:
            polygons.append(None)

    # 3. Compute overlap set C (intersection area > 0)
    print("Computing overlap set C...")
    C = []
    # Pre-extract bounds for fast bounding box filtering
    poly_bounds = []
    for poly in polygons:
        if poly is not None and not poly.is_empty:
            poly_bounds.append(poly.bounds)  # (minx, miny, maxx, maxy)
        else:
            poly_bounds.append(None)

    for i in range(N):
        poly_i = polygons[i]
        b_i = poly_bounds[i]
        if poly_i is None or b_i is None:
            continue
        minx_i, miny_i, maxx_i, maxy_i = b_i

        for j in range(i + 1, N):
            poly_j = polygons[j]
            b_j = poly_bounds[j]
            if poly_j is None or b_j is None:
                continue

            minx_j, miny_j, maxx_j, maxy_j = b_j
            # Bounding box intersection check (skip if disjoint)
            if maxx_i < minx_j or minx_i > maxx_j or maxy_i < miny_j or miny_i > maxy_j:
                continue

            # Check detailed intersection
            if poly_i.intersects(poly_j):
                inter = poly_i.intersection(poly_j)
                if inter.area > 1e-9:
                    C.append((i, j))

    print(f"Found {len(C)} overlapping pairs (C set).")

    # 4. Compute weight matrix W (shared nodes)
    print("Computing weight matrix W...")
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        nodes_i = set(hyperedges[i].get("nodes", []))
        for j in range(i + 1, N):
            nodes_j = set(hyperedges[j].get("nodes", []))
            shared_count = len(nodes_i.intersection(nodes_j))
            W[i, j] = shared_count
            W[j, i] = shared_count

    # 5. Regularize to ensure connectivity (guarantees unique Fiedler vector)
    print(f"Applying regularization (reg = {reg_val}) to ensure connectivity...")
    W_reg = W.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                W_reg[i, j] += reg_val

    # Compute Laplacian
    d_reg = np.sum(W_reg, axis=1)
    D_reg = np.diag(d_reg)
    L_G_reg = D_reg - W_reg

    dsatur_colors = dsatur_coloring(N, C)
    dsatur_num_colors = max(dsatur_colors.values()) + 1 if N > 0 else 0
    print(f"Graph coloring completed. Total colors/layers used: {dsatur_num_colors}")

    # 6. Phase 1: Graph Coloring Layout (DSatur)
    # print("Phase 1: Computing Graph Coloring using DSatur on overlap graph C...")
    # color_assignment = dsatur_coloring(N, C)
    # num_colors = max(color_assignment.values()) + 1 if N > 0 else 0
    # print(f"Graph coloring completed. Total colors/layers used: {num_colors}")

    # todo最短ハミルトン路
    # ポリゴンの重なりでできるグラフと、ポリゴンの共通する頂点でできるグラフの2種類ある
    # 最短ハミルトン路は共通する頂点でできるグラフを使う

    # 6. Phase 1: Shortest Hamiltonian Path Heuristic (Nearest Neighbor) per Connected Component
    print(
        "Phase 1: Computing Shortest Hamiltonian Path (Nearest Neighbor) per connected component..."
    )

    # 共通ノードでできるグラフ(W)ではなく、ポリゴンの重なり(C)に基づいて連結成分に分割する
    adj_C = {i: [] for i in range(N)}
    for u, v in C:
        adj_C[u].append(v)
        adj_C[v].append(u)

    visited = set()
    components = []
    for i in range(N):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj_C[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    print(f"Divided into {len(components)} connected component(s).")

    color_assignment = {}
    pos = {}

    for comp in components:
        if len(comp) == 1:
            color_assignment[comp[0]] = 0
            pos[comp[0]] = 0
            continue

        unvisited = set(comp)
        comp_path = []

        # 開始ノードは連結成分内で最も共有ノード数が多いハイパーエッジを選択
        best_start = None
        max_deg = -1
        for u in comp:
            deg = sum(W[u, v] for v in comp)
            if deg > max_deg:
                max_deg = deg
                best_start = u

        current = best_start
        comp_path.append(current)
        unvisited.remove(current)

        while unvisited:
            best_neighbor = None
            best_w = -1.0
            for v in unvisited:
                if W[current, v] > best_w:
                    best_w = W[current, v]
                    best_neighbor = v

            comp_path.append(best_neighbor)
            current = best_neighbor
            unvisited.remove(current)

        # 経路の順番をレイヤー（色）および相対順序として扱う
        for i, node in enumerate(comp_path):
            color_assignment[node] = i
            pos[node] = i

    print("Hamiltonian Path per component completed.")

    # 7. Phase 2: Convex QP Solver (Reduced strictly convex system)
    print(
        "Phase 2: Setting up strictly convex QP problem (eliminating anchor variable x_0)..."
    )

    # Scale factor for objective to ensure numerical stability in solver
    scale = 1.0 / np.max(np.abs(L_G_reg))
    print(f"Objective scale factor: {scale:.6e}")

    # L_yy is the submatrix of L_G_reg for variables x_1 to x_{N-1}
    L_yy = L_G_reg[1:, 1:]

    # Initial guess for reduced system variables: y_i = x_{i+1} (using color coordinates)
    color_anchor = color_assignment[0]
    y_init = np.array(
        [color_assignment[i] - color_anchor for i in range(1, N)], dtype=float
    )

    # Constraints matrix A_y (shape len(C) x (N - 1))
    # lb <= A_y * y <= ub where lb is 1.0 and ub is inf
    num_constraints = len(C)
    A_y = np.zeros((num_constraints, N - 1), dtype=float)
    lb_y = np.full(num_constraints, 1.0, dtype=float)
    ub_y = np.full(num_constraints, np.inf, dtype=float)

    # Formulate constraints: for overlap (i, j), if pos[i] < pos[j], then x_j - x_i >= 1
    # Recall: x_0 = 0 is constant, x_k = y[k-1] for k >= 1.
    for k, (i, j) in enumerate(C):
        is_i_less = pos[i] < pos[j]
        # x_right - x_left >= 1
        if is_i_less:
            left, right = i, j
        else:
            left, right = j, i

        # Coeffs: x_right has coefficient +1.0, x_left has coefficient -1.0
        # If right == 0: constraint is -x_left >= 1 (so left coeff is -1.0)
        # If left == 0: constraint is x_right >= 1 (so right coeff is +1.0)
        if left > 0:
            A_y[k, left - 1] = -1.0
        if right > 0:
            A_y[k, right - 1] = 1.0

    linear_constraint = LinearConstraint(A_y, lb_y, ub_y)

    # Objective function: 2 * scale * y^T * L_yy * y
    def objective(y):
        return 2.0 * scale * np.dot(y, np.dot(L_yy, y))

    # Gradient of objective: 4 * scale * L_yy * y
    def jacobian(y):
        return 4.0 * scale * np.dot(L_yy, y)

    print("Solving QP using scipy.optimize.minimize (SLSQP)...")
    start_time = time.time()
    res = minimize(
        fun=objective,
        jac=jacobian,
        x0=y_init,
        constraints=linear_constraint,
        method="SLSQP",
        options={"maxiter": 1000, "disp": True},
    )
    duration = time.time() - start_time

    if not res.success:
        print(f"Warning: Optimizer did not converge: {res.message}", file=sys.stderr)
    else:
        print(
            f"Optimization converged successfully in {res.nit} iterations ({duration:.3f} s)."
        )
        print(f"Optimal objective value: {res.fun:.6f}")

    y_opt = res.x

    # Reconstruct the full coordinates x (x_0 = 0)
    x_opt = np.zeros(N, dtype=float)
    x_opt[1:] = y_opt

    # 8. Post-verification of constraints
    violations = 0
    max_violation = 0.0
    for i, j in C:
        diff = abs(x_opt[i] - x_opt[j])
        if diff < 1.0 - 1e-5:
            violations += 1
            max_violation = max(max_violation, 1.0 - diff)

    if violations > 0:
        print(
            f"Warning: {violations} overlap constraints violated! Max violation: {max_violation:.6f}",
            file=sys.stderr,
        )
    else:
        print("All overlap constraints successfully satisfied (|x_i - x_j| >= 1).")

    print(f"Anchor coordinate x_0: {x_opt[0]:.6f}")
    print(f"Coordinate range: min={np.min(x_opt):.4f}, max={np.max(x_opt):.4f}")

    # 9. Format output and save
    print("Preparing output JSON...")
    # Add layer coordinate to each hyperedge
    for idx, he in enumerate(hyperedges):
        he["layer"] = float(x_opt[idx])

    # Store overlaps in output data for the viewer to use
    output_data = {"nodes": nodes, "hyperedges": hyperedges, "overlaps": C}

    print(f"Writing output to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Layer assignment complete!")


if __name__ == "__main__":
    args = parse_arguments()
    solve_layer_assignment(args.input_file, args.output_file, args.reg)
