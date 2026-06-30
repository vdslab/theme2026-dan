#!/usr/bin/env python3
import json
import argparse
import os
import sys
import numpy as np
from scipy.sparse.csgraph import connected_components

def parse_arguments():
    parser = argparse.ArgumentParser(description="Check connected components of the shared node graph.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="original.json",
        help="Path to the input JSON file (default: original.json)"
    )
    return parser.parse_args()

def analyze_components(input_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading hypergraph data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hyperedges = data.get("hyperedges", [])
    N = len(hyperedges)

    if N == 0:
        print("Error: No hyperedges found in input data.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {N} hyperedges.")

    # 共通ノードに基づく隣接行列の作成
    print("Computing shared nodes graph...")
    W = np.zeros((N, N), dtype=int)
    for i in range(N):
        nodes_i = set(hyperedges[i].get("nodes", []))
        for j in range(i + 1, N):
            nodes_j = set(hyperedges[j].get("nodes", []))
            # 1つでもノードを共有していればエッジを張る
            if len(nodes_i.intersection(nodes_j)) > 0:
                W[i, j] = 1
                W[j, i] = 1

    # 各ハイパーエッジの連結数（次数）を計算
    degrees = np.sum(W, axis=1)

    # 連結成分の計算
    n_components, labels = connected_components(W, directed=False)
    unique, counts = np.unique(labels, return_counts=True)

    print(f"\nTotal connected components: {n_components}")

    # サイズが大きい順にコンポーネントをソート
    sizes = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)

    # 詳細な出力の作成（標準出力だと長すぎるためファイルに書き出す）
    output_filename = "components_detail.txt"
    print(f"\nWriting detailed results to {output_filename} ...")
    
    with open(output_filename, "w", encoding="utf-8") as out:
        out.write(f"Total connected components: {n_components}\n")
        out.write("="*60 + "\n\n")
        
        for c, count in sizes:
            out.write(f"--- Component {c} ({count} hyperedges) ---\n")
            # このコンポーネントに属するハイパーエッジのインデックスを取得
            indices = np.where(labels == c)[0]
            
            # 連結数（次数）が大きい順にソートして出力
            sorted_indices = sorted(indices, key=lambda idx: degrees[idx], reverse=True)
            
            for idx in sorted_indices:
                he = hyperedges[idx]
                he_id = he.get("id")
                he_label = he.get("label", "No Label")
                degree = degrees[idx]
                out.write(f"  - ID: {he_id:3d} | Connections: {degree:3d} | Label: {he_label}\n")
                
            out.write("\n")
            
    print(f"Done. Please check '{output_filename}' for details.")

if __name__ == "__main__":
    args = parse_arguments()
    analyze_components(args.input_file)
