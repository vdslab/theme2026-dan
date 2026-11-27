"""Convert hash-pinned public sources into new VisNotes experiment inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess


HGPOLYVIS_COMMIT = "ba8868b1b002be5c32fb1e83548778fd7feed186"
DBLP_PATH = "Data/PaperAuthor_DBLP"
DBLP_DOCUMENTATION = f"https://github.com/peterdanieloliver/HGPolyVis/tree/{HGPOLYVIS_COMMIT}/{DBLP_PATH}"
DBLP_RAW = f"https://raw.githubusercontent.com/peterdanieloliver/HGPolyVis/{HGPOLYVIS_COMMIT}/{DBLP_PATH}"
MARSEILLE_GROUP_LIMIT = 126


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str
    documentation: str
    license: str


SOURCES = {
    "dblp": Source(
        f"{DBLP_RAW}/publishing_TVCG_2015_2017.er",
        "45bf87634f5ba0cb0b57bfa2ed5866b1c86459a4fa3ca5f79fcb1fafe846ea29",
        DBLP_DOCUMENTATION,
        "CC0 1.0 (DBLP bibliography; see HGPolyVis data README)",
    ),
    "dblp-medium": Source(
        f"{DBLP_RAW}/publishing_TPAMI_2013_2015.er",
        "8d208af874305c62cc578efa7969f7c97000f48695eeabed4a7eab86816c8032",
        DBLP_DOCUMENTATION,
        "CC0 1.0 (DBLP bibliography; see HGPolyVis data README)",
    ),
    "marseille": Source(
        "https://sociopatterns.org/assets/data/HighSchool2013_proximity_net.csv.gz",
        "ecf4a90312c8d72057f72e936c366b470e63876da7db846300044795f045e2ea",
        "https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/",
        "CC BY-NC-SA 3.0 (SocioPatterns)",
    ),
}


def dblp_input(raw):
    """Read the HGPolyVis 2D comma-delimited format with saved relationship colors."""
    lines = raw.decode("utf-8-sig").splitlines()
    if len(lines) < 4 or lines[:3] != [
        "Nary Relationship Data 2D", "Comma Delimiter", "Relationship Colors Saved",
    ]:
        raise ValueError("Unexpected HGPolyVis ER header")
    node_count, edge_count = map(int, lines[3].split(","))
    if node_count <= 0 or edge_count <= 0 or len(lines) != 4 + node_count + edge_count:
        raise ValueError("ER counts do not match the data rows")
    nodes = []
    for row in lines[4:4 + node_count]:
        identifier, payload = row.split(",", 1)
        label, x, y = payload.rsplit(",", 2)
        coordinates = float(x), float(y)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Non-finite ER coordinate")
        nodes.append({"id": int(identifier), "label": label, "x": coordinates[0], "y": coordinates[1]})
    known = {node["id"] for node in nodes}
    if len(known) != node_count:
        raise ValueError("Duplicate ER vertex ID")
    hyperedges = []
    for row in lines[4 + node_count:]:
        identifier, r, g, b, payload = row.split(",", 4)
        if any(not 0 <= int(channel) <= 255 for channel in (r, g, b)):
            raise ValueError("Invalid ER relationship color")
        fields = payload.split(",")
        angle = None
        for index in range(1, len(fields)):
            # Labels are unquoted and may include commas. The first matching
            # arity takes the longest structurally valid suffix of node IDs.
            suffix = fields[index + 1:]
            if fields[index] == str(len(suffix)):
                try:
                    members = list(map(int, suffix))
                except ValueError:
                    continue
            elif fields[index] == "1" and len(suffix) == 2:
                # HGPolyVis writes the original orientation after the member
                # of a one-vertex relationship when that orientation is saved.
                try:
                    members = [int(suffix[0])]
                    angle = float(suffix[1])
                except ValueError:
                    continue
                if not math.isfinite(angle):
                    raise ValueError(f"Invalid ER monogon orientation for edge {identifier}")
            else:
                continue
            if members:
                break
        else:
            raise ValueError(f"Invalid ER relationship membership for edge {identifier}")
        if not set(members) <= known or len(set(members)) != len(members):
            raise ValueError(f"Unknown or repeated ER relationship member for edge {identifier}")
        edge = {"id": int(identifier), "label": ",".join(fields[:index]), "nodes": members}
        if angle is not None:
            edge["source_monogon_angle"] = angle
        hyperedges.append(edge)
    if len({edge["id"] for edge in hyperedges}) != edge_count:
        raise ValueError("Duplicate ER relationship ID")
    return {"nodes": nodes, "hyperedges": hyperedges}


def maximal_groups(contacts):
    adjacency = defaultdict(set)
    for a, b in contacts:
        adjacency[a].add(b)
        adjacency[b].add(a)

    def cliques(members, candidates, excluded):
        if len(members) + len(candidates) < 3:
            return
        if not candidates and not excluded:
            yield tuple(sorted(members))
            return
        pivot = max(candidates | excluded, key=lambda node: (len(candidates & adjacency[node]), -node))
        for node in sorted(candidates - adjacency[pivot]):
            yield from cliques(members | {node}, candidates & adjacency[node], excluded & adjacency[node])
            candidates.remove(node)
            excluded.add(node)

    return sorted(cliques(set(), set(adjacency), set()))


def marseille_input(raw):
    windows = defaultdict(set)
    for row in gzip.decompress(raw).decode("ascii").splitlines():
        timestamp, a, b, _, _ = row.split()
        a, b = int(a), int(b)
        if a == b:
            raise ValueError("Self-contact in the proximity source")
        windows[int(timestamp)].add(tuple(sorted((a, b))))
    selected = {}
    for timestamp, contacts in sorted(windows.items()):
        for group in maximal_groups(contacts):
            selected.setdefault(group, timestamp)
            if len(selected) == MARSEILLE_GROUP_LIMIT:
                break
        if len(selected) == MARSEILLE_GROUP_LIMIT:
            break
    if len(selected) != MARSEILLE_GROUP_LIMIT:
        raise ValueError("Too few distinct higher-order contacts")

    adjacency = defaultdict(set)
    for group in selected:
        for a, b in combinations(group, 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
    remaining = set(adjacency)
    coordinates = {}
    while remaining:
        first = min(remaining)
        component = [first]
        remaining.remove(first)
        for node in component:
            neighbors = remaining & adjacency[node]
            component.extend(sorted(neighbors))
            remaining.difference_update(neighbors)
        # The source has no coordinates. Disjoint component circles put each
        # group in convex boundary order while keeping the geometry fixed.
        center = 3.0 * len(coordinates)
        for index, node in enumerate(sorted(component)):
            angle = math.tau * index / len(component)
            coordinates[node] = (round(center + math.cos(angle), 12), round(math.sin(angle), 12))
    data = {
        "nodes": [
            {"id": node, "label": f"Vertex {node}", "x": x, "y": y}
            for node, (x, y) in sorted(coordinates.items())
        ],
        "hyperedges": [
            {"id": index, "label": f"Group {index + 1}", "nodes": list(group)}
            for index, group in enumerate(selected)
        ],
    }
    return data, {
        "selection": "First 126 distinct maximal cliques of at least 3 vertices, ordered by onset timestamp and then ascending vertex ID tuple; first occurrence retained.",
        "coordinate_rule": "Disjoint unit circles per connected component of the selected groups, ordered by smallest vertex ID. Ascending vertex ID gives angular and polygon boundary order. Coordinates are synthesized.",
        "selection_first_timestamp": min(selected.values()),
        "selection_last_timestamp": max(selected.values()),
        "source_time_bin_seconds": 20,
        "source_citation": "Mastrandrea, Fournet & Barrat (2015), DOI 10.1371/journal.pone.0136497; SocioPatterns collaboration.",
    }


def encoded_json(data):
    return (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def prepare(name, output, source_dir):
    source = SOURCES[name]
    filename = source.url.rsplit("/", 1)[-1]
    raw = ((source_dir / filename).read_bytes() if source_dir else subprocess.check_output(
        ["curl", "--disable", "--fail", "--silent", "--show-error", "--location",
         "--max-time", "60", source.url],
    ))
    digest = sha256(raw).hexdigest()
    if digest != source.sha256:
        raise ValueError(f"Source SHA-256 mismatch for {name}: {digest}")
    if name == "marseille":
        data, rules = marseille_input(raw)
    else:
        data = dblp_input(raw)
        rules = {
            "source_commit": HGPOLYVIS_COMMIT,
            "selection": "Entire ER file.",
            "coordinate_rule": "Upstream XY and relationship member order retained; presentation colors omitted.",
        }
    payload = encoded_json(data)
    manifest = {
        "dataset_id": name,
        "source_url": source.url,
        "source_sha256": digest,
        "source_documentation": source.documentation,
        "source_license": source.license,
        **rules,
        "vertex_count": len(data["nodes"]),
        "hyperedge_count": len(data["hyperedges"]),
        "maximum_cardinality": max(len(edge["nodes"]) for edge in data["hyperedges"]),
        "layout_input_sha256": sha256(payload).hexdigest(),
    }
    paths = [output / f"{name}.json", output / f"{name}.manifest.json"]
    written = []
    try:
        for path, contents in zip(paths, (payload, encoded_json(manifest)), strict=True):
            with path.open("xb") as file:
                written.append(path)
                file.write(contents)
    except BaseException:
        for path in written:
            path.unlink()
        raise
    print(f"{name}: {manifest['vertex_count']} vertices, {manifest['hyperedge_count']} edges")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Use hash-pinned source files already downloaded here.")
    parser.add_argument("--dataset", choices=SOURCES, action="append", help="Repeat to select inputs; default is all.")
    args = parser.parse_args()
    names = list(dict.fromkeys(args.dataset or SOURCES))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in names:
        for suffix in (".json", ".manifest.json"):
            path = args.output / f"{name}{suffix}"
            if path.exists() or path.is_symlink():
                parser.error(f"Choose a new result path: {path}")
    for name in names:
        prepare(name, args.output, args.source_dir)


if __name__ == "__main__":
    main()
