"""Low-pass the GPU step's correction: python tools/dispsmooth.py start.ply shape.ply out_prefix mu..."""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import trimesh

a = trimesh.load(sys.argv[1], process=False)
b = trimesh.load(sys.argv[2], process=False)
F = a.faces
n = len(a.vertices)
i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
A.data[:] = 1
L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
D = b.vertices - a.vertices
for mu in map(float, sys.argv[4:]):
    Ds = spl.splu((sp.eye(n) + mu * L).tocsc()).solve(D)
    trimesh.Trimesh(a.vertices + Ds, F, process=False).export(f'{sys.argv[3]}_mu{mu:g}.ply')
