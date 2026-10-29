import sys, numpy as np
sys.path.insert(0, '.')
from limbscan.meshio_utils import *
A = load_mm('../Structre scanner.obj'); B = load_mm('../picture-scan.obj')
T, fit, rmse = align(B, A)
print('fitness', fit, 'rmse', rmse); print(T)
B2 = B.copy(); B2.apply_transform(T)
print('scale check (det^(1/3))', np.cbrt(np.linalg.det(T[:3,:3])))
pts, d = surface_distances(B2, A)
for q in (50, 75, 90, 95): print('B->A p%d' % q, np.percentile(d, q))
print('inlier (<10mm) mean', d[d<10].mean(), 'frac', (d<10).mean())
B2.export('/tmp/claude-1000/-home-ali-Downloads-Scan/dadd3a8e-3de1-44be-9ca9-2245841c965f/scratchpad/pic_aligned.ply')
np.save('/tmp/claude-1000/-home-ali-Downloads-Scan/dadd3a8e-3de1-44be-9ca9-2245841c965f/scratchpad/T_pic2struct.npy', T)
