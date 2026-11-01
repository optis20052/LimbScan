"""Score meshes with the refinement objective (no optimisation): python irscore.py prep.npz mesh.ply ..."""
import sys
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from irrefine import Views, vertex_normals, ncc, dev
from irshape import outline

z = np.load(sys.argv[1])
glctx = dr.RasterizeCudaContext()
torch.manual_seed(0)
for level in (2, 1, 0):
    vw = Views(z, level)
    for p in sys.argv[2:]:
        m = trimesh.load(p, process=False)
        V = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
        F = torch.tensor(m.faces, dtype=torch.int64, device=dev)
        Fi = F.int().contiguous()
        N = vertex_normals(V, F)
        sil, _, zb = outline(vw, glctx, V, Fi)
        fa = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1).norm(dim=1)
        tot = []
        for rep in range(5):
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
        print(f'L{level} {p.split("/")[-1]:28s} outline {sil.item()*100:.4f}  ncc {np.mean(tot)*10:.4f}')
