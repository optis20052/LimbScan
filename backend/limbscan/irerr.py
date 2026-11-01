"""Error map: a mesh coloured by signed distance to a reference in the same frame (blue inside, red outside,
+-3 mm), drawn from photo cameras next to the photo: python irerr.py prep.npz mesh.ply ref.ply out.jpg [views]"""
import sys
import cv2
import numpy as np
import open3d as o3d
import torch
import nvdiffrast.torch as dr
import trimesh
from irrefine import Views, vertex_normals, dev

z = np.load(sys.argv[1])
m = trimesh.load(sys.argv[2], process=False)
ref = trimesh.load(sys.argv[3], process=False)
views = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else [0, 13, 26, 39]
rs = o3d.t.geometry.RaycastingScene()
rs.add_triangles(o3d.core.Tensor(np.asarray(ref.vertices, np.float32)), o3d.core.Tensor(np.asarray(ref.faces, np.uint32)))
q = o3d.core.Tensor(np.asarray(m.vertices, np.float32))
d = rs.compute_distance(q).numpy()
# sign: outside if the vector to the closest point points against the reference normal
cp = rs.compute_closest_points(q)
pt, tid = cp['points'].numpy(), cp['primitive_ids'].numpy()
sgn = np.sign(((m.vertices - pt) * ref.face_normals[tid]).sum(1))
sd = d * sgn
t = np.clip(sd / 3.0, -1, 1)
col = np.stack([np.clip(t, 0, 1), 1 - np.abs(t), np.clip(-t, 0, 1)], 1).astype(np.float32)  # RGB
vw = Views(z, 0)
glctx = dr.RasterizeCudaContext()
V = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
F = torch.tensor(m.faces, dtype=torch.int32, device=dev)
N = vertex_normals(V, F.long())
clip, _ = vw.clip(V)
rast, _ = dr.rasterize(glctx, clip, F, (vw.H, vw.W))
C, _ = dr.interpolate(torch.tensor(col, device=dev)[None].contiguous(), rast, F)
Nn, _ = dr.interpolate(N[None].contiguous(), rast, F)
tiles = []
for i in views:
    vdir = -vw.R[i, 2]  # camera looks along +z: shade by normal facing the camera
    sh = (-(Nn[i] * vw.R[i, 2]).sum(-1)).clamp(0.2, 1)[..., None]
    img = (C[i] * (0.4 + 0.6 * sh)).cpu().numpy()
    cov = (rast[i, ..., 3] > 0).cpu().numpy()
    photo = vw.img[i].permute(1, 2, 0).cpu().numpy() ** (1 / 2.2)
    out = np.where(cov[..., None], img, photo * 0.5)
    tiles.append((np.hstack([photo, out]) * 255).astype(np.uint8))
h = max(x.shape[0] for x in tiles)
tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 4, cv2.BORDER_CONSTANT) for x in tiles]
cv2.imwrite(sys.argv[4], cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
inside = np.abs(sd) < 10
print(f'signed median {np.median(sd[inside]):+.2f}  |d| median {np.median(d[inside]):.2f}  >2mm outside {(sd > 2).mean() * 100:.1f}%  <-2mm {(sd < -2).mean() * 100:.1f}%')
