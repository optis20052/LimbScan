"""Put the reference model into a scan's frame (inverse of the evalref alignment): python tools/refinscan.py ref scan out"""
import sys
import numpy as np
import trimesh
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from evalref import load, align

ref, scan = load(sys.argv[1]), load(sys.argv[2])
F = trimesh.sample.sample_surface(ref, 200000, seed=0)[0]
r = align(scan, ref, F)
ref.apply_transform(np.linalg.inv(r.transformation))
# keep the part the scan covers (plus margin): the reference's long forearm would hang outside every photo
from scipy.spatial import cKDTree
d, _ = cKDTree(scan.vertices).query(ref.triangles_center)
ref.update_faces(d < 15)
ref.remove_unreferenced_vertices()
ref.export(sys.argv[3])
print('fitness', r.fitness, len(ref.faces))
