import json
import numpy as np
import networkx as nx
from networkx.algorithms.approximation.traveling_salesman import traveling_salesman_problem, greedy_tsp

def test_tsp():
    with open('original.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    hyperedges = data.get('hyperedges', [])
    N = len(hyperedges)
    
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        nodes_i = set(hyperedges[i].get('nodes', []))
        for j in range(i + 1, N):
            nodes_j = set(hyperedges[j].get('nodes', []))
            shared_count = len(nodes_i.intersection(nodes_j))
            W[i, j] = shared_count
            W[j, i] = shared_count

    max_W = np.max(W)
    D = max_W - W
    
    # Add dummy node for Hamiltonian path
    G = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=D[i, j])
            
    dummy = N
    for i in range(N):
        G.add_edge(dummy, i, weight=0)

    print("Running greedy_tsp...")
    tour = traveling_salesman_problem(G, cycle=True, method=greedy_tsp)
    
    # tour is a cycle starting and ending at dummy (e.g., [dummy, a, b, ..., dummy])
    # The path is the tour without the dummy node
    idx = tour.index(dummy)
    path = tour[idx+1:] + tour[:idx]
    if path[-1] == dummy:
        path = path[:-1]
    # Remove duplicate dummy if it's there
    path = [node for node in path if node != dummy]
    
    print(f"Path length: {len(path)}")
    print("Path elements unique:", len(set(path)) == N)
    
if __name__ == '__main__':
    test_tsp()
