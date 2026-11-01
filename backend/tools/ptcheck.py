"""Accuracy of point clouds in a scan's frame against the reference placed in that frame (tools/refinscan.py):
python tools/ptcheck.py ref_in_scan.ply cloud.ply ..."""
import sys
import numpy as np
import open3d as o3d
import trimesh
ref = trimesh.load(sys.argv[1], process=False)
rs = o3d.t.geometry.RaycastingScene()
rs.add_triangles(o3d.core.Tensor(np.asarray(ref.vertices, np.float32)), o3d.core.Tensor(np.asarray(ref.faces, np.uint32)))
for p in sys.argv[2:]:
    g = trimesh.load(p, process=False)
    P = np.asarray(g.vertices if hasattr(g, 'vertices') else g, np.float32)
    d = rs.compute_distance(o3d.core.Tensor(P)).numpy()
    occ = rs.compute_occupancy(o3d.core.Tensor(P)).numpy() if ref.is_watertight else None
    near = d < 10
    s = np.where(occ > 0.5, -d, d) if occ is not None else d
    print(f'{p.split("/")[-1]:30s} n={near.sum():7d} |d| med {np.median(d[near]):.2f} p90 {np.percentile(d[near], 90):.2f}'
          f' signed med {np.median(s[near]):+.2f}  <1mm {(d[near] < 1).mean() * 100:.0f}%')
