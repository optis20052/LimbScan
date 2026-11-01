"""Shading cost of meshes (light fitted per mesh): python irshtest.py prep.npz level mesh.ply ..."""
import sys
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from irrefine import Views, vertex_normals, dev
from irdepth import visible
from irshade import Light, cost

z = np.load(sys.argv[1])
level = int(sys.argv[2])
vw = Views(z, level)
glctx = dr.RasterizeCudaContext()
for p in sys.argv[3:]:
    m = trimesh.load(p, process=False)
    V = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
    Fi = torch.tensor(m.faces, dtype=torch.int32, device=dev)
    F = Fi.long()
    N = vertex_normals(V, F)
    vis, _ = visible(vw, glctx, V, Fi, N, min_cos=0.0)
    light = Light(vw.n)
    opt = torch.optim.Adam(light.parameters(), lr=0.03)
    for it in range(300):
        c, ok = cost(vw, light, V, N, vis)
        opt.zero_grad(); c.backward(); opt.step()
    print(f'{p.split("/")[-1]:24s} L{level} shading cost {c.item()*100:.3f}  torch {torch.nn.functional.softplus(light.torch_ + 3).item():.2f} '
          f'shine {torch.exp(light.ks).item():.3f} p {torch.exp(light.kp).item():.1f} room {torch.nn.functional.softplus(light.room[:, 0]).detach().cpu().numpy().round(3)}  points {ok.float().mean().item()*100:.0f}%')
