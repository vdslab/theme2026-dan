#!/usr/bin/env python3
import json
import argparse
import os
import sys
import time
import numpy as np
import scipy.linalg as la
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from shapely.geometry import Polygon


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Assign layers to hyperedges in a hypergraph using a MILP approach."
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

    # 5. Regularize to ensure connectivity (minimal path instead of complete graph)
    print(f"Applying sparse regularization (reg = {reg_val}) to ensure connectivity...")
    W_reg = W.copy()
    for i in range(N - 1):
        if W_reg[i, i + 1] == 0:
            W_reg[i, i + 1] = reg_val
            W_reg[i + 1, i] = reg_val

    # ========== 5.1 繋がり(W)に基づくヒューリスティックな初期1次元配置の計算 ==========
    print(
        "Computing heuristic 1D layout based on connection weights (Spectral Layout)..."
    )
    # Laplacian行列を計算
    L = np.diag(np.sum(W_reg, axis=1)) - W_reg
    # 固有値計算 (対称行列なので eigh を使用)
    vals, vecs = la.eigh(L)
    # vals は昇順。2番目に小さい固有値の固有ベクトル (Fiedler vector) を使用
    x_init = vecs[:, 1] if N > 1 else np.zeros(N)

    # ========== 5.2 理想座標 x_init に基づくバケット(層)割り当て ==========
    print(
        "Assigning nodes to buckets (layers) to compress depth while preserving W-order..."
    )
    # x_init の昇順にノードをソート (W の希望する順序を優先)
    sorted_nodes = np.argsort(x_init)

    # 重なりグラフの隣接リストを作成
    adj = {i: set() for i in range(N)}
    for i, j in C:
        adj[i].add(j)
        adj[j].add(i)

    # バケット割り当て (同じバケット=同じ層に極力相乗りさせる)
    bucket_assignment = {}
    for node in sorted_nodes:
        used_buckets = [
            bucket_assignment[neighbor]
            for neighbor in adj[node]
            if neighbor in bucket_assignment
        ]
        if used_buckets:
            # 重なる下のノードたちの直上のバケットに入れる
            bucket_assignment[node] = max(used_buckets) + 1
        else:
            # 重なる下のノードがいなければ一番下のバケットに入れる
            bucket_assignment[node] = 0

    num_buckets = max(bucket_assignment.values()) + 1 if bucket_assignment else 0
    print(f"Nodes assigned to {num_buckets} discrete buckets (layers).")

    # ========== 6. Setup Linear Programming (LP) solver for L1 norm objective ==========
    print("Setting up Linear Programming (LP) problem for absolute value objective...")

    # Identify active pairs where objective weight > 0
    active_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0:
                active_pairs.append((i, j))

    M_pairs = len(active_pairs)

    # Total variables V = N (x_i) + M_pairs (t_ij)
    # y_k (ビッグMのスイッチ) は順序が固定されたため不要になりました
    V = N + M_pairs

    c = np.zeros(V, dtype=float)
    for pair_idx, (i, j) in enumerate(active_pairs):
        c[N + pair_idx] = W_reg[i, j]

    # Build sparse constraint matrix
    print("Building LP constraints...")
    rows = []
    cols = []
    data = []
    lb = []
    ub = []

    row_idx = 0

    # 重なり順序の制約 (Phase 1 のバケット結果で決定済み)
    for i, j in C:
        b_i = bucket_assignment[i]
        b_j = bucket_assignment[j]

        # 重なりグラフの辺なので必ず b_i != b_j
        if b_i < b_j:
            # j が上: x_j - x_i >= 1  =>  x_i - x_j <= -1
            rows.extend([row_idx, row_idx])
            cols.extend([i, j])
            data.extend([1.0, -1.0])
            lb.append(-np.inf)
            ub.append(-1.0)
            row_idx += 1
        else:
            # i が上: x_i - x_j >= 1  =>  x_j - x_i <= -1
            rows.extend([row_idx, row_idx])
            cols.extend([j, i])
            data.extend([1.0, -1.0])
            lb.append(-np.inf)
            ub.append(-1.0)
            row_idx += 1

    # Absolute value constraints for objective
    for pair_idx, (i, j) in enumerate(active_pairs):
        t_col = N + pair_idx

        # x_i - x_j - t_{ij} <= 0
        rows.extend([row_idx, row_idx, row_idx])
        cols.extend([i, j, t_col])
        data.extend([1.0, -1.0, -1.0])
        lb.append(-np.inf)
        ub.append(0.0)
        row_idx += 1

        # x_j - x_i - t_{ij} <= 0
        rows.extend([row_idx, row_idx, row_idx])
        cols.extend([j, i, t_col])
        data.extend([1.0, -1.0, -1.0])
        lb.append(-np.inf)
        ub.append(0.0)
        row_idx += 1

    if row_idx > 0:
        A = coo_matrix((data, (rows, cols)), shape=(row_idx, V)).tocsc()
        constraints = LinearConstraint(A, lb, ub)
    else:
        constraints = None

    # Variable bounds
    # x_0 is fixed to 0. x_i can be negative, t_ij >= 0
    lb_bounds = np.full(V, -np.inf)
    for pair_idx in range(M_pairs):
        lb_bounds[N + pair_idx] = 0.0  # t_ij >= 0
    lb_bounds[0] = 0.0

    ub_bounds = np.full(V, np.inf)
    ub_bounds[0] = 0.0  # Fix anchor x_0 to 0

    bounds = Bounds(lb_bounds, ub_bounds)

    # Integrality: すべて 0 (連続変数)
    # y_k がなくなったため純粋な LP として解かれます
    integrality = np.zeros(V, dtype=int)

    print(f"Solving LP with {V} continuous variables and {row_idx} constraints...")
    start_time = time.time()

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    duration = time.time() - start_time

    if not res.success:
        print(
            f"Warning: MILP optimization did not converge perfectly. Status: {res.status}",
            file=sys.stderr,
        )
        print(f"Message: {res.message}", file=sys.stderr)
        if res.x is not None:
            x_opt = res.x[:N]
            print(f"Best objective value found (L1 norm): {res.fun:.6f}")
        else:
            print("Error: No solution found by MILP solver.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Optimization converged successfully ({duration:.3f} s).")
        print(f"Optimal objective value (L1 norm): {res.fun:.6f}")
        x_opt = res.x[:N]

    # 7. Post-verification of constraints
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

    # Calculate Equivalent L2 Cost for comparison
    l2_cost = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            l2_cost += W_reg[i, j] * (x_opt[i] - x_opt[j]) ** 2

    print(f"Anchor coordinate x_0: {x_opt[0]:.6f}")
    print(f"Coordinate range: min={np.min(x_opt):.4f}, max={np.max(x_opt):.4f}")
    print(f"Equivalent L2 objective value (squared difference): {l2_cost:.6f}")

    # 8. Format output and save
    print("Preparing output JSON...")
    # Add layer coordinate to each hyperedge
    for idx, he in enumerate(hyperedges):
        he["layer"] = float(x_opt[idx])

    # Store overlaps in output data for the viewer to use
    output_data = {
        "nodes": nodes,
        "hyperedges": hyperedges,
        "overlaps": C,
        "objective_value": float(res.fun) if res.fun is not None else None,
        "objective_value_l2": float(l2_cost),
    }

    print(f"Writing output to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Layer assignment complete!")


if __name__ == "__main__":
    args = parse_arguments()
    solve_layer_assignment(args.input_file, args.output_file, args.reg)
