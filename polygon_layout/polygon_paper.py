"""Paper equations and Hu initialization, separate from legacy approximations.

The native kernel implements Qu et al. equations (1)-(6),(9)-(10), with
unnormalized sums, polygon area centroids, and analytical gradients. The last
three optional terms are explicitly extensions, disabled in paper mode.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

TERMS = ('regularity', 'length', 'separation', 'intersection', 'regular_fit', 'collision', 'convexity')
PAPER_WEIGHTS = np.array([.30, .16, .36, .18, 0., 0., 0.])
_DLL = None


def kernel_library():
    global _DLL
    if _DLL is not None:
        return _DLL
    source = Path(__file__).with_name('polygon_kernel.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = source.parent.parent / '.polygon_cache'
    cache.mkdir(exist_ok=True)
    library = cache / ('polygon_' + digest + ('.dll' if os.name == 'nt' else '.so'))
    if not library.exists():
        compiler = shutil.which('g++')
        if not compiler:
            raise RuntimeError('The polygon kernel needs g++ (C++17) for its first build.')
        command = [compiler, '-O3', '-std=c++17', '-shared', str(source), '-o', str(library)]
        command += ['-static-libgcc', '-static-libstdc++'] if os.name == 'nt' else ['-fPIC']
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Polygon kernel compilation failed: ' + result.stderr)
    lib = ctypes.CDLL(str(library))
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.polygon_create.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip]
    lib.polygon_create.restype = ctypes.c_void_p
    lib.polygon_destroy.argtypes = [ctypes.c_void_p]
    lib.polygon_buffers.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    lib.polygon_evaluate.argtypes = [ctypes.c_void_p, dp, dp, dp, dp]
    lib.polygon_evaluate.restype = ctypes.c_double
    lib.polygon_centers.argtypes = [ctypes.c_void_p, dp, dp]
    lib.polygon_centers_gradient.argtypes = [ctypes.c_void_p, dp, dp, dp]
    lib.polygon_swap_sweep.argtypes = [ctypes.c_void_p, dp, dp, ip, ctypes.c_int, dp, dp, ctypes.c_double, dp]
    lib.polygon_swap_sweep.restype = ctypes.c_int
    _DLL = lib
    return lib


class PaperEnergy:
    def __init__(self, problem, distance_buffer=.1, angle_buffer=.03):
        self.n = problem.n
        self.m = len(problem.members)
        self.nonempty = np.asarray([bool(len(m)) for m in problem.members])
        self.lib = kernel_library()
        offsets = np.r_[0, np.cumsum([len(m) for m in problem.members])].astype(np.int32)
        ids = np.concatenate(problem.members).astype(np.int32) if offsets[-1] else np.empty(0, np.int32)
        self.handle = self.lib.polygon_create(self.n, len(problem.members), offsets, ids)
        if not self.handle:
            raise RuntimeError('Could not initialize polygon energy kernel')
        self.lib.polygon_buffers(self.handle, distance_buffer, angle_buffer)
        self.evaluations = 0
        memberships = [set() for _ in range(self.n)]
        all_pairs = set()
        for edge, members in enumerate(problem.members):
            for a in members:
                memberships[a].add(edge)
                for b in members:
                    if a < b:
                        all_pairs.add((int(a), int(b)))
        # Equal incidence gives an identically zero swap gain. This is an exact
        # shortcut, not a vertex merge: every original vertex remains present.
        self.swap_pairs = np.asarray([p for p in sorted(all_pairs) if memberships[p[0]] != memberships[p[1]]], np.int32).reshape(-1, 2)
        self.equivalent_pairs = len(all_pairs) - len(self.swap_pairs)

    def evaluate(self, xy, weights=PAPER_WEIGHTS, details=False):
        x = self.checked_xy(xy)
        w = self.checked_weights(weights)
        gradient = np.zeros_like(x)
        terms = np.zeros(len(TERMS))
        value = self.lib.polygon_evaluate(self.handle, x, w, gradient, terms)
        self.evaluations += 1
        if details:
            return value, dict(zip(TERMS, map(float, terms)))
        return value, gradient

    def close(self):
        if getattr(self, 'handle', None):
            self.lib.polygon_destroy(self.handle)
            self.handle = None

    @staticmethod
    def checked_weights(weights):
        w = np.ascontiguousarray(weights, dtype=np.float64)
        if w.shape != (len(TERMS),) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError('Expected seven finite nonnegative energy weights')
        return w

    def checked_xy(self, xy):
        if not self.handle:
            raise ValueError('Polygon energy has been closed')
        x = np.ascontiguousarray(xy, dtype=np.float64).reshape(-1)
        if len(x) != self.n*2 or not np.isfinite(x).all():
            raise ValueError('Invalid coordinate vector')
        return x

    def __del__(self):
        self.close()

    def centers(self, xy):
        x = self.checked_xy(xy)
        centers = np.zeros((self.m, 2))
        self.lib.polygon_centers(self.handle, x, centers)
        return centers

    def centers_gradient(self, xy, center_gradient):
        x = self.checked_xy(xy)
        cg = np.ascontiguousarray(center_gradient, dtype=np.float64)
        if cg.shape != (self.m, 2) or not np.isfinite(cg).all():
            raise ValueError('Invalid center gradient')
        gradient = np.zeros_like(x)
        self.lib.polygon_centers_gradient(self.handle, x, cg, gradient)
        return gradient.reshape(self.n, 2)

    def swap_sweep(self, xy, weights=PAPER_WEIGHTS, vertex_targets=None, face_targets=None, dd_weight=0.):
        x = self.checked_xy(xy).copy()
        dummy = np.zeros(2)
        vt = dummy if vertex_targets is None else np.ascontiguousarray(vertex_targets, dtype=np.float64)
        ft = dummy if face_targets is None else np.ascontiguousarray(face_targets, dtype=np.float64)
        if not np.isfinite(dd_weight) or dd_weight < 0:
            raise ValueError('Invalid DD weight')
        if dd_weight and (vt.shape != (self.n, 2) or ft.shape != (self.m, 2)
                          or not np.isfinite(vt).all() or not np.isfinite(ft).all()):
            raise ValueError('Invalid DD targets')
        gain = np.zeros(1)
        accepted = self.lib.polygon_swap_sweep(self.handle, x, self.checked_weights(weights),
                    self.swap_pairs, len(self.swap_pairs), vt, ft, dd_weight, gain)
        return x.reshape(self.n, 2), {'tested': len(self.swap_pairs),
                'equivalent_zero_gain': self.equivalent_pairs, 'accepted': accepted, 'gain': float(gain[0])}


class JointEnergy:
    """Qu et al. section 4: both views plus equation (11), DD coordination."""
    def __init__(self, problem):
        from types import SimpleNamespace
        self.primal = PaperEnergy(problem)
        self.n, self.m = problem.n, len(problem.members)
        dual_members = [[] for _ in range(self.n)]
        for edge, members in enumerate(problem.members):
            for vertex in members:
                dual_members[vertex].append(edge)
        self.dual = PaperEnergy(SimpleNamespace(n=self.m, members=dual_members))
        self.nonisolated = np.asarray([bool(m) for m in dual_members])
        self.evaluations = 0

    def evaluate(self, xy, weights=None, details=False, dd_weight=.2):
        if weights is None:
            weights = np.array([.2, .2, .2, .2, 0., 0., 0.])
        x = np.asarray(xy).reshape(self.n + self.m, 2)
        p, d = x[:self.n], x[self.n:]
        vp, gp = self.primal.evaluate(p, weights)
        vd, gd = self.dual.evaluate(d, weights)
        cp, cd = self.primal.centers(p), self.dual.centers(d)
        rp, rd = (p-cd)*self.nonisolated[:, None], (d-cp)*self.primal.nonempty[:, None]
        dd = float(np.sum(rp*rp) + np.sum(rd*rd))
        gp = gp.reshape(self.n, 2) + dd_weight*(2*rp + self.primal.centers_gradient(p, -2*rd))
        gd = gd.reshape(self.m, 2) + dd_weight*(2*rd + self.dual.centers_gradient(d, -2*rp))
        self.evaluations += 1
        value = vp + vd + dd_weight*dd
        if details:
            return value, {'primal': self.primal.evaluate(p, weights, True)[1],
                           'dual': self.dual.evaluate(d, weights, True)[1], 'coordination': dd}
        return value, np.vstack((gp, gd)).ravel()

    def swap_sweep(self, xy, weights, dd_weight=.2):
        x = np.asarray(xy).reshape(self.n+self.m, 2).copy()
        p, d = x[:self.n], x[self.n:]
        p[:], a = self.primal.swap_sweep(p, weights, self.dual.centers(d), d, dd_weight)
        d[:], b = self.dual.swap_sweep(d, weights, self.primal.centers(p), p, dd_weight)
        return x, {'primal': a, 'dual': b, 'accepted': a['accepted']+b['accepted'], 'gain': a['gain']+b['gain']}


def find_sfdp(explicit=None):
    candidates = [explicit, shutil.which('sfdp')]
    if os.name == 'nt':
        for name in ('ProgramFiles', 'ProgramFiles(x86)'):
            if os.environ.get(name):
                candidates.append(str(Path(os.environ[name]) / 'Graphviz' / 'bin' / 'sfdp.exe'))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError('Hu initialization requires Graphviz sfdp. Install Graphviz or pass --sfdp PATH.')


def hu_layout(problem, seed=7, representation='clique', executable=None):
    """Call Graphviz's implementation of Hu's multilevel force-directed method."""
    sfdp = find_sfdp(executable)
    edges = set()
    count = problem.n
    if representation == 'clique':
        for members in problem.members:
            for a in members:
                for b in members:
                    if a < b:
                        edges.add((int(a), int(b)))
    elif representation == 'bipartite':
        count += len(problem.members)
        for edge, members in enumerate(problem.members):
            edges.update((int(vertex), problem.n + edge) for vertex in members)
    else:
        raise ValueError('Unknown Hu graph representation')
    lines = ['strict graph G {', f'graph [start="{seed}", overlap=true, splines=false];',
             'node [shape=point, width=0, height=0, label=""];']
    lines += [f'{i};' for i in range(count)]
    lines += [f'{a} -- {b};' for a, b in sorted(edges)]
    lines += ['}']
    result = subprocess.run([sfdp, '-Tplain'], input='\n'.join(lines), text=True,
                            capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError('sfdp failed: ' + result.stderr)
    xy = np.full((count, 2), np.nan)
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == 'node':
            xy[int(fields[1])] = [float(fields[2]), float(fields[3])]
    if not np.isfinite(xy).all():
        raise RuntimeError('sfdp did not return every graph vertex')
    version = subprocess.run([sfdp, '-V'], capture_output=True, text=True)
    return xy, {'algorithm': 'Hu multilevel force-directed (Graphviz sfdp)',
                'representation': representation, 'seed': seed, 'graph_vertices': count,
                'graph_edges': len(edges), 'executable': sfdp,
                'version': (version.stderr + version.stdout).strip()}
