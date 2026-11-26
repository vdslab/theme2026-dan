"""Bridge the compact solver's prepared problem to the C++ search stage."""

from ctypes import CDLL, CFUNCTYPE, POINTER, Structure, byref, c_double, c_int32, c_uint32
from itertools import accumulate, chain
from pathlib import Path
import sys
import time

import numpy as np


INT32_MAX = (1 << 31) - 1
LIBRARY_DIRECTORY = Path(__file__).with_name("build")
LIBRARY_NAME = {"win32": "compact_search.dll", "darwin": "libcompact_search.dylib"}.get(
    sys.platform, "libcompact_search.so",
)
DEFAULT_LIBRARY = LIBRARY_DIRECTORY / LIBRARY_NAME
IntPointer = POINTER(c_int32)
DoublePointer = POINTER(c_double)
RandBelow = CFUNCTYPE(c_uint32, c_uint32)
Random = CFUNCTYPE(c_double)
Record = CFUNCTYPE(None, c_int32, c_int32, c_double, c_double)


class SearchInput(Structure):
    _fields_ = [
        ("n", c_int32), ("cap", c_int32), ("budget", c_int32), ("flags", c_int32),
        ("quadratic_factor", c_double), ("reg", c_double), ("degree_factor", c_double),
        ("remaining_seconds", c_double), ("initial", IntPointer),
        ("adj_offsets", IntPointer), ("adj", IntPointer),
        ("group_offsets", IntPointer), ("incidence", IntPointer),
        ("group_count", c_int32), ("pair_count", c_int32),
        ("left", IntPointer), ("right", IntPointer), ("weights", DoublePointer),
        ("move_count", c_int32), ("moves", IntPointer),
        ("component_count", c_int32), ("component_offsets", IntPointer),
        ("free_edges", IntPointer), ("randbelow", RandBelow), ("random", Random),
        ("record", Record),
    ]


class SearchOutput(Structure):
    _fields_ = [
        ("layers", IntPointer), ("iterations", c_int32), ("updates", c_int32),
        ("timed_out", c_int32), ("value", c_double),
        ("attempted", IntPointer), ("evaluated", IntPointer), ("accepted", IntPointer),
    ]


def row_arrays(rows):
    offsets = np.fromiter(accumulate(map(len, rows), initial=0), dtype=np.int64)
    if offsets[-1] > INT32_MAX:
        raise ValueError("Native search index capacity exceeded.")
    entries = np.fromiter(chain.from_iterable(rows), dtype=np.int32, count=int(offsets[-1]))
    return offsets.astype(np.int32), entries


class NativeSearch:
    def __init__(self, objective, moves, library_path):
        path = Path(library_path).resolve(strict=True)
        self.library = CDLL(str(path))
        try:
            version = self.library.compact_search_abi_version
            self.search = self.library.compact_search
        except AttributeError as error:
            raise RuntimeError(f"Native search entry point is missing in {path}.") from error
        version.argtypes = ()
        version.restype = c_int32
        if version() != 1:
            raise RuntimeError(f"Unsupported native search ABI in {path}. Rebuild the library.")
        self.search.argtypes = (POINTER(SearchInput), POINTER(SearchOutput))
        self.search.restype = c_int32
        if max(objective.problem.n, len(objective.left)) > INT32_MAX:
            raise ValueError("Native search index capacity exceeded.")
        groups = tuple(objective.incident[index] for index in objective.shared_vertices)
        adj_offsets, adj = row_arrays(tuple(sorted(row) for row in objective.problem.adjacency))
        group_offsets, incidence = row_arrays(groups)
        component_offsets, free_edges = row_arrays(objective.free_components)
        left = np.ascontiguousarray(objective.left, dtype=np.int32)
        right = np.ascontiguousarray(objective.right, dtype=np.int32)
        weights = np.ascontiguousarray(objective.pair_weights, dtype=np.float64)
        self.move_names = moves
        self.selected_moves = objective.config.moves
        move_ids = np.array([moves.index(move) for move in self.selected_moves], dtype=np.int32)
        self.flags = int(objective.use_pillars) | (int(objective.config.refine_free_edges) << 1)
        self.factors = (objective.quadratic_factor, objective.config.reg, objective.degree_factor)
        self.graph = (
            adj_offsets, adj, group_offsets, incidence, len(groups), len(left), left, right,
            weights, len(move_ids), move_ids, len(objective.free_components),
            component_offsets, free_edges,
        )

    def run(self, initial, cap, budget, first_iteration, rng, started_at, deadline):
        stage_start = time.perf_counter()
        trace = []
        callback_errors = []

        @Record
        def record(iteration, layers, value, seconds):
            try:
                trace.append({
                    "iteration": first_iteration + iteration,
                    "elapsed_seconds": native_offset + seconds,
                    "optimized_value": value,
                    "used_layer_count": layers,
                })
            except BaseException as error:
                callback_errors.append(error)

        @RandBelow
        def randbelow(limit):
            try:
                if not callback_errors:
                    return rng.randrange(limit)
            except BaseException as error:
                callback_errors.append(error)
            return (1 << 32) - 1

        @Random
        def random():
            try:
                if not callback_errors:
                    return rng.random()
            except BaseException as error:
                callback_errors.append(error)
            return float("nan")
        source = np.ascontiguousarray(initial, dtype=np.int32)
        destination = np.empty_like(source)
        stats = np.zeros((3, len(self.move_names)), dtype=np.int32)
        graph = (
            entry.ctypes.data_as(DoublePointer if entry.dtype == np.float64 else IntPointer)
            if isinstance(entry, np.ndarray) else entry
            for entry in self.graph
        )
        request = SearchInput(
            len(source), cap, budget, self.flags, *self.factors, 0.0,
            source.ctypes.data_as(IntPointer), *graph, randbelow, random, record,
        )
        response = SearchOutput(
            destination.ctypes.data_as(IntPointer), 0, 0, 0, 0.0,
            *(row.ctypes.data_as(IntPointer) for row in stats),
        )
        native_start = time.perf_counter()
        native_offset = native_start - started_at
        request.remaining_seconds = -1.0 if deadline is None else max(0.0, deadline - native_start)
        code = self.search(byref(request), byref(response))
        if callback_errors:
            raise callback_errors[0]
        if code:
            raise RuntimeError(f"Native search failed with status {code}.")
        if not trace:
            raise RuntimeError("Native search returned no initial trajectory point.")
        move_stats = {
            move: dict(zip(("attempted", "evaluated", "accepted"),
                           stats[:, self.move_names.index(move)].tolist(), strict=True))
            for move in self.selected_moves
        }
        return destination.astype(np.int64), {
            "iterations": response.iterations,
            "free_coordinate_updates": response.updates,
            "trajectory": trace,
            "moves": move_stats,
            "elapsed_seconds": time.perf_counter() - stage_start,
        }, response.value, bool(response.timed_out)
