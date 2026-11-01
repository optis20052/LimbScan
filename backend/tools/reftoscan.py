"""Reference model into a scan's frame (no cropping): python tools/reftoscan.py ref scan out"""
import sys
import numpy as np
import trimesh
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from evalref import load, align
ref, scan = load(sys.argv[1]), load(sys.argv[2])
F = trimesh.sample.sample_surface(ref, 200000, seed=0)[0]
r = align(scan, ref, F)
ref.apply_transform(np.linalg.inv(r.transformation))
ref.export(sys.argv[3])
print('fitness', r.fitness)
