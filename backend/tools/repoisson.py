"""Clean surface from a refined mesh: points + normals -> screened Poisson -> keep what is near the input:
python tools/repoisson.py in.ply out.ply [depth]"""
import sys
import numpy as np
import open3d as o3d
import trimesh

m = trimesh.load(sys.argv[1], process=False)
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 9
P, fi = trimesh.sample.sample_surface_even(m, 400000, radius=0.35, seed=0)
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
pc.normals = o3d.utility.Vector3dVector(m.face_normals[fi])
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, scale=1.1, linear_fit=True)
out = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
from scipy.spatial import cKDTree
d, _ = cKDTree(P).query(out.triangles_center)
out.update_faces(d < 2.0)
out.remove_unreferenced_vertices()
out = max(out.split(only_watertight=False), key=lambda x: len(x.faces))
out.export(sys.argv[2])
print(len(out.faces), out.is_watertight)
