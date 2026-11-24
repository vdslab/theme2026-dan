"""Small polygon fixtures and a direct exhaustive oracle for compact layouts."""

from itertools import combinations, product
import math

from optimization import OBJECTIVES


def rectangle(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_input(polygons):
    representatives = {}
    nodes = []
    edges = []
    for polygon in polygons:
        order = []
        for x, y in polygon:
            if (x, y) not in representatives:
                number = len(nodes)
                representatives[(x, y)] = number
                nodes.append({"id": number, "x": x, "y": y})
            order.append(representatives[(x, y)])
        edges.append({"id": len(edges), "nodes": order})
    return {"nodes": nodes, "hyperedges": edges}


def radius_point(angle):
    radians = math.radians(angle)
    return round(2 * math.cos(radians), 8), round(2 * math.sin(radians), 8)


def fixtures():
    path = [rectangle(x, 0, x + 2, 2) for x in range(4)]
    cycle = [
        [(0, 0), radius_point(angle - 44), radius_point(angle + 44)]
        for angle in range(0, 360, 72)
    ]
    mixed = path + [rectangle(1, -1, 3, 1), [(0, 0), (5, 2)]]
    return {
        "four-path": (
            make_input(path),
            [(0, 1), (1, 2), (2, 3)],
        ),
        "five-cycle": (
            make_input(cycle),
            [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)],
        ),
        "six-mixed": (
            make_input(mixed),
            [(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4)],
        ),
    }


def exact_optima(data, overlaps, reg, penalty):
    memberships = [set(edge["nodes"]) for edge in data["hyperedges"]]
    n = len(memberships)
    pairs = [
        (i, j, len(memberships[i] & memberships[j]) + reg)
        for i, j in combinations(range(n), 2)
    ]
    degrees = [math.fsum(w for i, j, w in pairs if k in (i, j)) for k in range(n)]
    largest = max(degrees)
    scale = 1 / largest if largest else 1.0
    incidences = [
        [i for i, members in enumerate(memberships) if node["id"] in members]
        for node in data["nodes"]
    ]
    optima = {}
    feasible_count = 0
    for layers in product(range(n), repeat=n):
        heights = set(layers)
        if len(heights) != max(heights) + 1:
            continue
        if any(layers[i] == layers[j] for i, j in overlaps):
            continue
        feasible_count += 1
        raw_squared = math.fsum(w * (layers[i] - layers[j]) ** 2 for i, j, w in pairs)
        squared = 2 * scale * raw_squared
        weighted_l1 = math.fsum(w * abs(layers[i] - layers[j]) for i, j, w in pairs)
        pillars = sum(
            max(layers[i] for i in incidence) - min(layers[i] for i in incidence)
            for incidence in incidences
            if incidence
        )
        values = (squared, weighted_l1, pillars, squared + penalty * len(heights))
        for name, value in zip(OBJECTIVES, values):
            if name not in optima or value < optima[name]["value"]:
                optima[name] = {"value": value, "layers": layers}
    return {"feasible_count": feasible_count, "optima": optima}
