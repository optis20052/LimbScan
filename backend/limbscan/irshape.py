"""Shape from the torch shading + outlines (GPU, ~1 min): the starting surface (outline hull) is moved until
  - drawn from every camera, it covers the limb's mask in that photo (outline term), and
  - every surface point, with the torch at the camera (cos / distance^2), the room light and each photo's
    exposure accounted for, shows one albedo in all photos that see it (shading term, irshade.cost).
The shading term ranks the true shape of the printed hand above shifted, bumpy and smoothed copies of it and far
above the outline hull, so minimising it carves the hollows the outlines cannot see (palm, knuckles, thumb web).
Vertex updates go through "large steps" smoothing (Nicolet et al. 2021), coarse to fine.

Usage: python irshape.py <prep.npz> <in.ply> <out.ply> [key=value ...]
"""
import sys
import time

import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import from_differential, to_differential

from irrefine import Views, vertex_normals, dev, sample
import torch.nn.functional as tF
from irshade import Light, cost


def surface_samples(V, F, n):
    fa = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1).norm(dim=1).detach()
    fi = torch.multinomial(fa, n, replacement=True)
    b = torch.rand(n, 2, device=dev)
    fl = b.sum(1) > 1
    b[fl] = 1 - b[fl]
    return fi, torch.stack([1 - b[:, 0] - b[:, 1], b[:, 0], b[:, 1]], 1)


def outline(vw, glctx, V, Fi, batch=12):
    """Outline mismatch (mean squared, coverage vs mask), its gradient at the vertices, and the depth buffers for
    visibility - drawn a few photos at a time so full-size photos fit in GPU memory."""
    Vg = V.clone().requires_grad_(True)
    zbs, tot = [], 0.0
    norm = vw.valid.sum()
    for s in range(0, vw.n, batch):
        e = min(vw.n, s + batch)
        sub = _Sub(vw, s, e)
        clip, zv = sub.clip(Vg)
        rast, _ = dr.rasterize(glctx, clip, Fi, (vw.H, vw.W))
        cover = dr.antialias((rast[..., 3:] > 0).float(), rast, clip, Fi)[..., 0]
        sil = (((cover - vw.mask[s:e]) ** 2) * vw.valid[s:e]).sum() / norm
        sil.backward()
        tot += sil.item()
        with torch.no_grad():
            zb, _ = dr.interpolate(zv[..., None].contiguous(), rast, Fi)
            zbs.append(torch.where(rast[..., 3:] > 0, zb, torch.full_like(zb, 1e9))[..., 0])
        del rast, cover, clip, zv
    return torch.tensor(tot), Vg.grad, torch.cat(zbs)


def rodrigues(w):
    """Rotation matrices (n, 3, 3) of axis-angle vectors w (n, 3), differentiable."""
    th = w.norm(dim=1, keepdim=True).clamp(min=1e-9)
    k = w / th
    K = torch.zeros(len(w), 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=w.device)[None] + s * K + (1 - c) * (K @ K)


class _Sub:
    """A slice of the photos with the same projection helpers; with per-photo pose corrections (vw.dw, vw.dt) when
    present, each slice building its own graph to them (the outline is drawn and differentiated slice by slice)."""

    def __init__(self, vw, s, e):
        self.f, self.c, self.R, self.t = vw.f[s:e], vw.c[s:e], vw.R[s:e], vw.t[s:e]
        if getattr(vw, 'dw', None) is not None:
            self.R = rodrigues(vw.dw[s:e]) @ vw.R0[s:e]
            self.t = vw.t0[s:e] + vw.dt[s:e]
        self.H, self.W = vw.H, vw.W

    cam = Views.cam
    clip = Views.clip


_FLIP = None
WV = None


def normal_cost(vw, P, Pn, vis, wp=None):
    """The surface's direction against the MoGe-2 normals of the photos that see it (inside the cut-out, facing the
    camera): mean of 1 - cos. The photos' palm, knuckles and creases the outlines cannot carve. MoGe's axis
    convention is found once (the sign flips that best match the current surface)."""
    global _FLIP
    _, uv = vw.cam(P)
    Nm = sample(vw.nrm, uv, vw.W, vw.H)  # (n, N, 3)
    Nc = torch.einsum('nij,pj->npi', vw.R, Pn)  # surface normals in each camera's frame
    with torch.no_grad():
        if not hasattr(vw, 'deep'):
            # well inside the cut-out only (~12 px at full size from its edge): the palm and the back of the hand,
            # not the fingers - MoGe's normals are too coarse across a finger 15-20 px wide (1 Nov: fingers 96 -> 88 %
            # on the printed hand with them, palm 92 -> 97 %)
            k = 2 * max(2, 12 >> getattr(vw, 'level', 0)) + 1
            vw.deep = -tF.max_pool2d(-vw.mask[:, None], k, 1, k // 2)[:, 0]
        inn = sample(vw.deep[:, None], uv, vw.W, vw.H)[..., 0] > 0.5
        ok = vis & inn & (Nm.norm(dim=-1) > 0.5) & (Nc[..., 2] < -0.2)  # facing the camera (OpenCV: -z)
        if _FLIP is None:
            best = None
            for fx in (1, -1):
                for fy in (1, -1):
                    for fz in (1, -1):
                        f = torch.tensor([fx, fy, fz], device=dev, dtype=torch.float32)
                        c = (tF.normalize(Nm * f, dim=-1) * Nc).sum(-1)[ok].mean().item()
                        if best is None or c > best[0]:
                            best = (c, f)
            _FLIP = best[1]
            print(f'[shape] MoGe normals: axis signs {_FLIP.tolist()}, initial agreement {best[0]:.3f}', flush=True)
    Nm = tF.normalize(Nm * _FLIP, dim=-1)
    c = (Nm * Nc).sum(-1)
    w = ok.float()
    norm = w.sum().clamp(min=1)  # (normalised over all seen points: gating a region must not make it pull harder)
    if wp is not None:  # per surface point: where the normals may act (palm and back of the hand, see shape.py)
        w = w * wp[None]
    return ((1 - c) * w).sum() / norm


def run(prep, src, dst, levels=(2, 1, 0), iters=(150, 150, 100), lam=30.0, lr=0.03, w_sil=1.0, w_shade=1.0,
        n_pts=150000, pose=0.0, w_fold=0.0, fold_cos=0.5, w_norm=0.0, log=print):
    """pose > 0: the photos' cameras are corrected too (a small rotation and shift each, at the coarse levels): a
    camera placed slightly off, or a limb that moved a little between photos, is lined up with the others instead of
    carving its error into the shape. pose = learning-rate scale (1: ~0.03 degree, 0.03 mm per step)."""
    t0 = time.time()
    z = np.load(prep)
    m = trimesh.load(src, process=False)
    V0 = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
    global WV
    WV = torch.tensor(z['w_norm_v'], dtype=torch.float32, device=dev) if 'w_norm_v' in z else None
    F = torch.tensor(m.faces, dtype=torch.int64, device=dev)
    Fi = F.int().contiguous()
    M = compute_matrix(V0, F, lam)
    # w_fold > 0: a face whose normal turns more than acos(fold_cos) from where it started is penalised - the surface
    # may move to match the photos but not fold over itself (31 Oct: a flap in the thumb web of a printed hand where
    # the step pulled a saddle inward further than its curvature allows)
    FN0 = torch.nn.functional.normalize(torch.cross(V0[F[:, 1]] - V0[F[:, 0]], V0[F[:, 2]] - V0[F[:, 0]], dim=1), dim=1)
    u = to_differential(M, V0).clone().requires_grad_(True)
    opt = AdamUniform([u], lr=lr)
    glctx = dr.RasterizeCudaContext()
    light = Light(len(z['names']))
    lopt = torch.optim.Adam(light.parameters(), lr=0.03)
    nv = len(z['names'])
    dw = torch.zeros(nv, 3, device=dev, requires_grad=True)
    dt = torch.zeros(nv, 3, device=dev, requires_grad=True)
    popt = torch.optim.Adam([{'params': [dw], 'lr': 5e-4 * pose}, {'params': [dt], 'lr': 0.03 * pose}]) if pose else None
    for level, n_it in zip(levels, iters):
        vw = Views(z, level)
        vw.level = level
        refine_pose = popt is not None and (level != levels[-1] or len(levels) == 1)
        vw.R0, vw.t0 = vw.R.clone(), vw.t.clone()
        if popt is not None:
            vw.dw, vw.dt = (dw, dt) if refine_pose else (dw.detach(), dt.detach())
        for it in range(-60 if level == levels[0] else 0, n_it):  # negative: fit the light alone first
            V = from_differential(M, u, 'Cholesky')
            N = vertex_normals(V, F)
            if popt is not None:
                with torch.no_grad():  # the shading below sees the corrected cameras (without a gradient to them)
                    vw.R = rodrigues(dw) @ vw.R0
                    vw.t = vw.t0 + dt
                    vw.C = -torch.einsum('nji,nj->ni', vw.R, vw.t)
                popt.zero_grad()
            sil, gV, zb = outline(vw, glctx, V.detach(), Fi)
            fi, bw = surface_samples(V, F, n_pts)
            P = (V[F[fi]] * bw[..., None]).sum(1)
            Pn = torch.nn.functional.normalize((N[F[fi]] * bw[..., None]).sum(1), dim=1)
            with torch.no_grad():
                Xc, uv = vw.cam(P)
                pix = uv.floor().long()
                inside = (pix[..., 0] >= 0) & (pix[..., 0] < vw.W) & (pix[..., 1] >= 0) & (pix[..., 1] < vw.H)
                px, py = pix[..., 0].clamp(0, vw.W - 1), pix[..., 1].clamp(0, vw.H - 1)
                ar = torch.arange(vw.n, device=dev)[:, None]
                vis = inside & (Xc[..., 2] < zb[ar, py, px] + 1.0) & (vw.valid[ar, py, px] > 0)
            sh, ok = cost(vw, light, P, Pn, vis)
            if w_norm > 0 and vw.nrm is not None:
                wp = None
                if WV is not None:
                    wp = (WV[F[fi]] * bw).sum(1)
                sh = sh + w_norm * normal_cost(vw, P, Pn, vis, wp)
            if w_fold > 0:
                FN = torch.nn.functional.normalize(torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1), dim=1)
                sh = sh + w_fold * torch.relu(fold_cos - (FN * FN0).sum(1)).pow(2).sum() / 1000.0
            opt.zero_grad()
            lopt.zero_grad()
            # the outline gradient was computed photo batch by photo batch (memory): injected at the vertices
            torch.autograd.backward([w_shade * sh * 100, V], [None, gV * (w_sil * 100)])
            if it >= 0:
                opt.step()
                if refine_pose:
                    popt.step()
                    with torch.no_grad():  # no common motion of all cameras (that is the whole model moving)
                        dw -= dw.mean(0)
                        dt -= dt.mean(0)
            lopt.step()
            if it % 50 == 0 or it == n_it - 1:
                mv = (V - V0).norm(dim=1)
                cams = ''
                if popt is not None:
                    a = dw.detach().norm(dim=1) * 57.2958
                    b = dt.detach().norm(dim=1)
                    cams = f'  cameras turned med {a.median().item():.2f} max {a.max().item():.2f} deg, ' \
                           f'moved med {b.median().item():.2f} max {b.max().item():.2f} mm'
                log(f'[shape] L{level} it {it:4d}  outline {sil.item() * 100:.3f}  shading {sh.item() * 100:.3f}  '
                    f'moved med {mv.median().item():.2f} p95 {mv.quantile(0.95).item():.2f} mm{cams}  {time.time() - t0:.0f}s')
        del vw
    V = from_differential(M, u, 'Cholesky').detach().cpu().numpy()
    trimesh.Trimesh(V, m.faces, process=False).export(dst)
    if popt is not None:  # the corrected cameras: R' = rot(dw) R, t' = t + dt (camera frame, mm)
        np.savez(str(dst) + '.poses.npz', names=z['names'], dw=dw.detach().cpu().numpy(), dt=dt.detach().cpu().numpy())
    log(f'[shape] done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    kw = {}
    for a in sys.argv[4:]:
        k, v = a.split('=')
        kw[k] = tuple(int(x) for x in v.split(',')) if ',' in v else float(v)
    if 'n_pts' in kw:
        kw['n_pts'] = int(kw['n_pts'])
    for k in ('levels', 'iters'):  # a single level / iteration count
        if k in kw and not isinstance(kw[k], tuple):
            kw[k] = (int(kw[k]),)
    run(sys.argv[1], sys.argv[2], sys.argv[3], **kw)
