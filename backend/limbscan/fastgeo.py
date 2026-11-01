"""Compiled (numba, multi-core) kernels for the per-point surface fits in postprocess.py.

Each point's fit only reads its own neighbourhood, so the parallel loops give exactly the same result on every
run (no shared sums). They are drop-in replacements for the numpy versions, which stay as the reference.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False)
def _median(a):
    return np.median(a)


@njit(parallel=True, cache=True)
def robust_planes(P, idx):
    """For each row of idx (k neighbour indices into P): robust local plane centre, normal, median height.
    Three reweighting rounds on the points within 0.6 mm of the median plane (hairs would tilt plain PCA)."""
    m, k = idx.shape
    C = np.empty((m, 3))
    N = np.empty((m, 3))
    H = np.empty(m)
    for i in prange(m):
        nb = np.empty((k, 3))
        for a in range(k):
            nb[a] = P[idx[i, a]]
        w = np.ones(k)
        c = np.zeros(3)
        n = np.zeros(3)
        med = 0.0
        for _ in range(3):
            ws = w.sum()
            c[:] = 0.0
            for a in range(k):
                c += nb[a] * w[a]
            c /= ws
            cov = np.zeros((3, 3))
            for a in range(k):
                x0, x1, x2 = nb[a, 0] - c[0], nb[a, 1] - c[1], nb[a, 2] - c[2]
                cov[0, 0] += w[a] * x0 * x0; cov[0, 1] += w[a] * x0 * x1; cov[0, 2] += w[a] * x0 * x2
                cov[1, 1] += w[a] * x1 * x1; cov[1, 2] += w[a] * x1 * x2; cov[2, 2] += w[a] * x2 * x2
            cov[1, 0] = cov[0, 1]; cov[2, 0] = cov[0, 2]; cov[2, 1] = cov[1, 2]
            cov /= ws
            n[:] = np.linalg.eigh(cov)[1][:, 0]
            off = np.empty(k)
            for a in range(k):
                off[a] = (nb[a, 0] - c[0]) * n[0] + (nb[a, 1] - c[1]) * n[1] + (nb[a, 2] - c[2]) * n[2]
            med = _median(off)
            for a in range(k):
                w[a] = (1.0 if abs(off[a] - med) < 0.6 else 0.0) + 1e-3
        C[i] = c
        N[i] = n
        H[i] = med
    return C, N, H


@njit(parallel=True, cache=True)
def quadric_offsets(V, ids, sup, nn):
    """Robust local quadric z = ax2+bxy+cy2+dx+ey+f in each point's tangent frame (6 rounds of hard rejection at
    2.5 robust sigmas); returns (normal, f) per point: the point moves by f along the normal."""
    m, k = nn.shape
    Nout = np.empty((m, 3))
    F = np.empty(m)
    for i in prange(m):
        q = V[ids[i]]
        X = np.empty((k, 3))
        for a in range(k):
            X[a] = sup[nn[i, a]] - q
        cov = X.T @ X
        ev = np.linalg.eigh(cov)[1]
        u, v, n = ev[:, 2].copy(), ev[:, 1].copy(), ev[:, 0].copy()
        A = np.empty((k, 6))
        z = np.empty(k)
        for a in range(k):
            xu = X[a, 0] * u[0] + X[a, 1] * u[1] + X[a, 2] * u[2]
            xv = X[a, 0] * v[0] + X[a, 1] * v[1] + X[a, 2] * v[2]
            z[a] = X[a, 0] * n[0] + X[a, 1] * n[1] + X[a, 2] * n[2]
            A[a, 0] = xu * xu; A[a, 1] = xu * xv; A[a, 2] = xv * xv; A[a, 3] = xu; A[a, 4] = xv; A[a, 5] = 1.0
        w = np.ones(k)
        coef = np.zeros(6)
        res = np.empty(k)
        for _ in range(6):
            M = np.zeros((6, 6))
            b = np.zeros(6)
            for a in range(k):
                for r in range(6):
                    war = w[a] * A[a, r]
                    b[r] += war * z[a]
                    for s in range(6):
                        M[r, s] += war * A[a, s]
            for r in range(6):
                M[r, r] += 1e-6
            coef = np.linalg.solve(M, b)
            for a in range(k):
                res[a] = z[a] - (A[a] * coef).sum()
            sc = 1.4826 * _median(np.abs(res)) + 1e-6
            for a in range(k):
                w[a] = (1.0 if abs(res[a]) < 2.5 * sc else 0.0) + 1e-4
        Nout[i] = n
        F[i] = coef[5]
    return Nout, F


@njit(cache=True)
def _propagate(N, order, pred):
    for t in range(1, len(order)):
        i = order[t]
        p = pred[i]
        if p >= 0 and N[i, 0] * N[p, 0] + N[i, 1] * N[p, 1] + N[i, 2] * N[p, 2] < 0:
            N[i] = -N[i]


def orient_normals(P, N, k=12):
    """Consistent normal signs (as open3d's orient_normals_consistent_tangent_plane, ~50x faster): minimum
    spanning tree of the k-NN graph weighted by 1 - |n_i.n_j|, signs propagated from a root along it, one tree
    per connected piece. The global sign (outward) is chosen by the caller."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
    from scipy.spatial import cKDTree
    N = np.array(N, dtype=np.float64)
    n = len(P)
    _, idx = cKDTree(P).query(P, k=k + 1, workers=-1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    w = 1.0 - np.abs((N[rows] * N[cols]).sum(1)) + 1e-6
    G = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)  # undirected
    T = minimum_spanning_tree(G)
    T = T + T.T
    ncomp, lab = connected_components(T, directed=False)
    first = np.unique(lab, return_index=True)[1]  # lowest index of each piece = its root (deterministic)
    for r in first:
        order, pred = breadth_first_order(T, int(r), directed=False, return_predecessors=True)
        _propagate(N, order.astype(np.int64), pred.astype(np.int64))
    return N
