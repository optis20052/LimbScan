"""Repeatable, fast SIFT matching for chosen photo pairs (replaces COLMAP's CPU matcher, whose multi-threaded
search returns slightly different matches on every run).

SIFT descriptors are uint8 vectors normalised to length 512, so every dot product is an integer below 2^24 and a
float32 matrix product computes it exactly, whatever the summation order or thread count: the matches are the
same on every run. Same rules as COLMAP: nearest neighbour by descriptor angle, ratio test 0.8, maximum distance
0.7, mutual (cross-check). Geometric verification is COLMAP's own (it is repeatable with a fixed RANSAC seed).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

MAX_RATIO = 0.8
MAX_DISTANCE = 0.7


@njit(cache=True, nogil=True)
def _mutual_matches(D, max_ratio, max_dist):
    """D: exact descriptor dot products (n1 x n2). Returns (k, 2) int64 index pairs passing COLMAP's tests."""
    n1, n2 = D.shape
    norm = 1.0 / (512.0 * 512.0)
    best12 = np.full(n1, -1, np.int64)
    for i in range(n1):
        b1, b2, bj = -1.0, -1.0, -1
        for j in range(n2):
            d = D[i, j]
            if d > b1:
                b2 = b1; b1 = d; bj = j
            elif d > b2:
                b2 = d
        if bj < 0:
            continue
        d1 = np.arccos(min(b1 * norm, 1.0))
        d2 = np.arccos(min(b2 * norm, 1.0)) if b2 >= 0 else np.pi
        if d1 > max_dist or d1 >= max_ratio * d2:
            continue
        best12[i] = bj
    best21 = np.full(n2, -1, np.int64)
    for j in range(n2):
        b1, b2, bi = -1.0, -1.0, -1
        for i in range(n1):
            d = D[i, j]
            if d > b1:
                b2 = b1; b1 = d; bi = i
            elif d > b2:
                b2 = d
        if bi < 0:
            continue
        d1 = np.arccos(min(b1 * norm, 1.0))
        d2 = np.arccos(min(b2 * norm, 1.0)) if b2 >= 0 else np.pi
        if d1 > max_dist or d1 >= max_ratio * d2:
            continue
        best21[j] = bi
    out = np.empty((n1, 2), np.int64)
    k = 0
    for i in range(n1):
        j = best12[i]
        if j >= 0 and best21[j] == i:
            out[k, 0] = i; out[k, 1] = j; k += 1
    return out[:k]


def match_pairs(db, pairs_path, verification_options, log=print):
    """Match every pair listed in pairs_path (two image names per line) and verify them geometrically."""
    import pycolmap
    from threadpoolctl import threadpool_limits
    d = pycolmap.Database.open(db)
    ids = {im.name: im.image_id for im in d.read_all_images()}
    desc = {i: np.asarray(d.read_descriptors(i).data, dtype=np.float32) for i in ids.values()}
    pairs = [tuple(ln.split()[:2]) for ln in open(pairs_path).read().splitlines() if ln.strip()]

    def one(p):
        a, b = ids[p[0]], ids[p[1]]
        A, B = desc[a], desc[b]
        if len(A) == 0 or len(B) == 0:
            return a, b, np.zeros((0, 2), np.uint32)
        return a, b, _mutual_matches(A @ B.T, MAX_RATIO, MAX_DISTANCE).astype(np.uint32)

    with threadpool_limits(1), ThreadPoolExecutor(os.cpu_count() or 4) as ex:  # one pair per core
        res = list(ex.map(one, pairs))
    n = 0
    for a, b, m in res:
        if len(m):
            d.write_matches(a, b, m)
            n += len(m)
    d.close()
    pycolmap.verify_matches(db, pairs_path, options=verification_options)
    log(f'[match] {len(pairs)} pairs, {n} mutual matches (exact, repeatable)')
