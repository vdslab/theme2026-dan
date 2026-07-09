#!/usr/bin/env python3
import json
import argparse
import os
import sys
import numpy as np
from scipy.sparse.csgraph import connected_components
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Check connected components of the shared node graph."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="original.json",
        help="Path to the input JSON file (default: original.json)",
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
    nodes_data = data.get("nodes", [])
    N = len(hyperedges)

    if N == 0:
        print("Error: No hyperedges found in input data.", file=sys.stderr)
        sys.exit(1)

    # ノードIDからラベルへの辞書を作成
    node_id_to_label = {n["id"]: n.get("label", str(n["id"])) for n in nodes_data}

    print(f"Loaded {N} hyperedges and {len(nodes_data)} nodes.")

    # 共通ノードに基づく隣接行列の作成
    print("Computing shared nodes graph...")
    W = np.zeros((N, N), dtype=int)
    for i in range(N):
        nodes_i = set(hyperedges[i].get("nodes", []))
        for j in range(i + 1, N):
            nodes_j = set(hyperedges[j].get("nodes", []))
            if len(nodes_i.intersection(nodes_j)) > 0:
                W[i, j] = 1
                W[j, i] = 1

    degrees = np.sum(W, axis=1)
    n_components, labels = connected_components(W, directed=False)
    unique, counts = np.unique(labels, return_counts=True)
    sizes = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)

    output_filename = "components_detail.txt"
    print(f"\nWriting detailed results to {output_filename} ...")

    G = nx.Graph(W)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, with_labels=True)
    plt.show()

    with open(output_filename, "w", encoding="utf-8") as out:
        out.write(f"Total connected components: {n_components}\n")
        out.write("=" * 80 + "\n\n")

        for c, count in sizes:
            out.write(f"--- Component {c} ({count} 個のハイパーエッジ) ---\n")
            indices = np.where(labels == c)[0]

            # ノードIDをキーにして、それを含むハイパーエッジのリストを作成する
            node_to_hes = {}
            for idx in indices:
                he = hyperedges[idx]
                he_label = he.get("label", "No Label")

                for nid in he.get("nodes", []):
                    if nid not in node_to_hes:
                        node_to_hes[nid] = []
                    node_to_hes[nid].append(he_label)

            # ハイパーエッジを多く繋いでいる（共有度が高い）ノード順に並び替え
            sorted_nodes = sorted(
                node_to_hes.items(), key=lambda x: len(x[1]), reverse=True
            )

            out.write("  [共通頂点（ノード）と、それに含まれるハイパーエッジの一覧]\n")
            for nid, he_list in sorted_nodes:
                nlabel = node_id_to_label.get(nid, str(nid))
                freq = len(he_list)

                if freq >= 2:
                    out.write(f"  - 共通頂点: {nlabel} (ID:{nid}) | 結合数: {freq}\n")
                else:
                    out.write(f"  - 頂点: {nlabel} (ID:{nid}) | 結合数: 1 (単独)\n")

                # その頂点を共有しているハイパーエッジのリストをカンマ区切りで表示
                he_str = ", ".join(he_list)
                out.write(f"      ハイパーエッジ: [{he_str}]\n")

            out.write("\n")

    print(f"Done. Please check '{output_filename}' for details.")


if __name__ == "__main__":
    args = parse_arguments()
    analyze_components(args.input_file)
