"""Fit a rigid mesh's pose and scale to the photos' outlines: python irfit.py prep.npz mesh.ply out.ply"""
import sys
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from irrefine import Views, dev
from irshape import outline


def rodrigues(w):
    th = w.norm() + 1e-12
    k = w / th
    K = torch.zeros(3, 3, device=dev)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=dev) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


z = np.load(sys.argv[1])
m = trimesh.load(sys.argv[2], process=False)
glctx = dr.RasterizeCudaContext()
V0 = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
Fi = torch.tensor(m.faces, dtype=torch.int32, device=dev)
c = V0.mean(0)
w = torch.zeros(3, device=dev, requires_grad=True)
t = torch.zeros(3, device=dev, requires_grad=True)
ls = torch.zeros(1, device=dev, requires_grad=True)
opt = torch.optim.Adam([w, t, ls], lr=0.002)
for level, n in ((2, 200), (1, 200), (0, 150)):
    vw = Views(z, level)
    for it in range(n):
        V = c + torch.exp(ls) * ((V0 - c) @ rodrigues(w).T) + t * 10
        sil, gV, _ = outline(vw, glctx, V.detach(), Fi)
        opt.zero_grad()
        V.backward(gV)
        opt.step()
    print(f'L{level} outline {sil.item() * 100:.4f} scale {torch.exp(ls).item():.4f} move {(t * 10).norm().item():.2f} mm '
          f'rot {np.degrees(w.norm().item()):.2f} deg')
V = (c + torch.exp(ls) * ((V0 - c) @ rodrigues(w).T) + t * 10).detach().cpu().numpy()
trimesh.Trimesh(V, m.faces, process=False).export(sys.argv[3])
