"""Pose check: warp photo A into photo B through a mesh (the true shape) and measure the leftover image shift
(optical flow) on the limb. Perfect cameras + shape: ~0 px. Usage: python irwarp.py prep.npz mesh.ply [out.jpg]"""
import sys
import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from irrefine import Views, sample, dev

z = np.load(sys.argv[1])
m = trimesh.load(sys.argv[2], process=False)
vw = Views(z, 0)
glctx = dr.RasterizeCudaContext()
V = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
F = torch.tensor(m.faces, dtype=torch.int32, device=dev)
clip, _ = vw.clip(V)
rast, _ = dr.rasterize(glctx, clip, F, (vw.H, vw.W))
X, _ = dr.interpolate(V[None].contiguous(), rast, F)  # world point per pixel per view
gray = (vw.gray[:, 0] * 255).clamp(0, 255).byte().cpu().numpy()
res = []
sheets = []
for b in range(0, vw.n, 3):
    # neighbour: the photo whose viewing direction is closest (10-30 degrees away)
    db = vw.R[b, 2]
    ang = torch.rad2deg(torch.acos((vw.R[:, 2] @ db).clamp(-1, 1)))
    ang[b] = 999
    cand = torch.where((ang > 8) & (ang < 30))[0]
    if len(cand) == 0:
        continue
    a = cand[ang[cand].argmin()].item()
    cov = (rast[b, ..., 3] > 0) & (vw.inner[b] > 0.5)
    P = X[b][cov]
    Xc, uv = vw.cam(P)
    # visible in A too
    za = torch.full((vw.H, vw.W), 1e9, device=dev)
    ra = rast[a]
    Xa, _ = dr.interpolate(V[None].contiguous(), rast[a:a + 1], F)
    zA = (vw.cam(Xa[0][ra[..., 3] > 0])[0][a, :, 2])
    za[ra[..., 3] > 0] = zA
    pa = uv[a].floor().long()
    okp = (pa[:, 0] >= 0) & (pa[:, 0] < vw.W) & (pa[:, 1] >= 0) & (pa[:, 1] < vw.H)
    pa = pa.clamp(min=0); pa[:, 0] = pa[:, 0].clamp(max=vw.W - 1); pa[:, 1] = pa[:, 1].clamp(max=vw.H - 1)
    okv = okp & (Xc[a, :, 2] < za[pa[:, 1], pa[:, 0]] + 1.0) & (vw.inner[a][pa[:, 1], pa[:, 0]] > 0.5)
    val = sample(vw.gray[a:a + 1], uv[a:a + 1], vw.W, vw.H)[0, :, 0]
    warped = torch.zeros(vw.H, vw.W, device=dev)
    msk = torch.zeros(vw.H, vw.W, dtype=torch.bool, device=dev)
    ys, xs = torch.nonzero(cov, as_tuple=True)
    warped[ys[okv], xs[okv]] = val[okv]
    msk[ys[okv], xs[okv]] = True
    W8 = (warped * 255).clamp(0, 255).byte().cpu().numpy()
    B8 = gray[b]
    mk = msk.cpu().numpy()
    W8 = np.where(mk, W8, B8)
    fl = cv2.calcOpticalFlowFarneback(B8, W8, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    er = cv2.erode(mk.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    mag = np.linalg.norm(fl[er], axis=1)
    if len(mag) < 1000:
        continue
    res.append((b, a, ang[a].item(), np.median(mag), np.median(fl[er], 0)))
    print(f'photo {b:2d} <- {a:2d} ({ang[a].item():4.1f} deg): leftover shift median {np.median(mag):.2f} px, '
          f'mean vector ({np.median(fl[er][:, 0]):+.2f}, {np.median(fl[er][:, 1]):+.2f})')
    if len(sheets) < 3:
        sheets.append(np.hstack([B8, W8, cv2.absdiff(B8, W8) * 4]))
print('ALL: median leftover shift %.2f px' % np.median([r[3] for r in res]))
if len(sys.argv) > 3 and sheets:
    h = max(s.shape[0] for s in sheets)
    cv2.imwrite(sys.argv[3], np.vstack([cv2.copyMakeBorder(s, 0, 0, 0, max(x.shape[1] for x in sheets) - s.shape[1], cv2.BORDER_CONSTANT) for s in sheets]))
