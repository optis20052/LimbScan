"""Variants of a mesh for objective tests: python tools/meshvariants.py in.ply outprefix"""
import sys
import numpy as np
import trimesh
m = trimesh.load(sys.argv[1], process=False)
p = sys.argv[2]
s = m.copy(); trimesh.smoothing.filter_taubin(s, iterations=30); s.export(p + '_smooth.ply')
for o in (-0.7, 0.7):
    v = m.copy(); v.vertices = m.vertices + m.vertex_normals * o; v.export(p + f'_off{o:+.1f}.ply')
rng = np.random.default_rng(0)
# low-frequency bumps of 1 mm
c = m.vertices
f = np.sin(c[:, 0] / 6) * np.sin(c[:, 1] / 7) * np.sin(c[:, 2] / 5)
b = m.copy(); b.vertices = c + m.vertex_normals * f[:, None] * 1.0; b.export(p + '_bumps.ply')
