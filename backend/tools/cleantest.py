"""Variants of the shape clean-up: python tools/cleantest.py scan_dir out_prefix"""
import sys
import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import trimesh
from scipy.spatial import cKDTree

d = sys.argv[1]
a = trimesh.load(f'{d}/shape/start.ply', process=False)
b = trimesh.load(f'{d}/shape/shape.ply', process=False)
F = a.faces
n = len(a.vertices)
i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
A.data[:] = 1
L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
D0 = b.vertices - a.vertices
for mu in (10, 40):
    D = spl.splu((sp.eye(n) + mu * L).tocsc()).solve(D0)
    m = trimesh.Trimesh(a.vertices + D, F, process=False)
    P, fi = trimesh.sample.sample_surface_even(m, 400000, radius=0.35, seed=0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(m.face_normals[fi])
    pm, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8, scale=1.1, linear_fit=True)
    out = trimesh.Trimesh(np.asarray(pm.vertices), np.asarray(pm.triangles), process=False)
    # keep the closed Poisson surface: cut only far-away pieces (no holes), largest component
    dd, _ = cKDTree(P).query(out.vertices)
    comps = out.split(only_watertight=False)
    out = max(comps, key=lambda x: len(x.faces))
    trimesh.smoothing.filter_taubin(out, iterations=10)
    out.export(f'{sys.argv[2]}_mu{mu}.ply')
    print(mu, len(out.faces), out.is_watertight, f'vol {out.volume/1000:.0f} ml')
