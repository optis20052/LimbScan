"""Objective along offsets / scalings of one mesh: python irsweep.py prep.npz mesh.ply"""
import sys
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from irrefine import Views, vertex_normals, ncc, dev

z = np.load(sys.argv[1])
m = trimesh.load(sys.argv[2], process=False)
glctx = dr.RasterizeCudaContext()
V0 = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
F = torch.tensor(m.faces, dtype=torch.int64, device=dev)
Fi = F.int().contiguous()
N0 = vertex_normals(V0, F)
c = V0.mean(0)
for level in (1, 0):
    vw = Views(z, level)
    for kind, vals in (('offset', (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5)), ('scale', (0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03))):
        row = []
        for v in vals:
            V = V0 + N0 * v if kind == 'offset' else c + (V0 - c) * v
            N = vertex_normals(V, F)
            clip, zv = vw.clip(V)
            rast, _ = dr.rasterize(glctx, clip, Fi, (vw.H, vw.W))
            sil = ((((rast[..., 3] > 0).float() - vw.mask) ** 2) * vw.valid).sum() / vw.valid.sum()
            zb, _ = dr.interpolate(zv[..., None].contiguous(), rast, Fi)
            zb = torch.where(rast[..., 3:] > 0, zb, torch.full_like(zb, 1e9))[..., 0]
            fa = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1).norm(dim=1)
            torch.manual_seed(0)
            tot = []
            for rep in range(4):
                fi = torch.multinomial(fa, 40000, replacement=True)
                b = torch.rand(40000, 2, device=dev); fl = b.sum(1) > 1; b[fl] = 1 - b[fl]
                bw = torch.stack([1 - b[:, 0] - b[:, 1], b[:, 0], b[:, 1]], 1)
                P = (V[F[fi]] * bw[..., None]).sum(1)
                Pn = torch.nn.functional.normalize((N[F[fi]] * bw[..., None]).sum(1), dim=1)
                Xc, uv = vw.cam(P)
                pix = uv.floor().long()
                inside = (pix[..., 0] >= 0) & (pix[..., 0] < vw.W) & (pix[..., 1] >= 0) & (pix[..., 1] < vw.H)
                px, py = pix[..., 0].clamp(0, vw.W - 1), pix[..., 1].clamp(0, vw.H - 1)
                ar = torch.arange(vw.n, device=dev)[:, None]
                vis = inside & (Xc[..., 2] < zb[ar, py, px] + 1.5) & (vw.valid[ar, py, px] > 0) & (vw.mask[ar, py, px] > 0.5)
                L = vw.C[:, None] - P[None]
                cos = (L / L.norm(dim=-1, keepdim=True) * Pn[None]).sum(-1, keepdim=True)
                vis = vis & (cos[..., 0] > 0.25)
                loss, ok = ncc(vw, P, Pn, vis, cos, level, 1.5)
                tot.append(loss.item())
            row.append(f'{v:+.2f}: ncc {np.mean(tot) * 10:.3f} sil {sil.item() * 100:.2f}')
        print(f'L{level} {kind}:  ' + ' | '.join(row))
