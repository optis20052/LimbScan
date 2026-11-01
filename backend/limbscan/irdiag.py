"""Per-vertex best offsets on a mesh (no smoothing): python irdiag.py prep.npz mesh.ply level k step_px"""
import sys
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from irrefine import Views, vertex_normals, dev
from irdepth import visible, sweep, choose

z = np.load(sys.argv[1])
m = trimesh.load(sys.argv[2], process=False)
level, k, step_px = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
V = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
Fi = torch.tensor(m.faces, dtype=torch.int32, device=dev)
glctx = dr.RasterizeCudaContext()
vw = Views(z, level)
if len(sys.argv) > 6:
    vw.highpass(float(sys.argv[6]))
N = vertex_normals(V, Fi.long())
vis, cos = visible(vw, glctx, V, Fi, N)
Xc, _ = vw.cam(V[::50])
mm_px = (Xc[..., 2].median() / vw.f.mean()).item()
offs = np.arange(-3, 3.01, 0.25)
costs = sweep(vw, V, N, vis, cos, offs, mm_px * step_px, k)
best, conf, ok = choose(costs, offs)
C = costs.cpu().numpy()[:, ok]
print(f'L{level} k{k} step{step_px}: mm/px {mm_px:.2f}; mean cost curve:', ' '.join(f'{o:+.2f}:{c:.3f}' for o, c in zip(offs[::2], np.nanmean(C, 1)[::2])))
hc = conf[ok] > 0.2
print(f'  best offset: median {np.median(best[ok]):+.2f}, IQR {np.percentile(best[ok],25):+.2f}..{np.percentile(best[ok],75):+.2f}; '
      f'confident ({hc.mean()*100:.0f}%) median {np.median(best[ok][hc]):+.2f} IQR {np.percentile(best[ok][hc],25):+.2f}..{np.percentile(best[ok][hc],75):+.2f}')
