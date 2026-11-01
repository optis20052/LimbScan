"""Carve the hollows the outlines cannot see (cupped palm, between knuckles, thumb web): every surface point tries
depths along its normal, mostly inward (the outline hull is an upper bound of the limb), and keeps the depth where
all photos that see it agree best (patch matching, normalized cross-correlation). The chosen depths are then
smoothed with their confidence (a sparse linear solve), so noise and textureless spots follow their neighbours.

Usage: python irdepth.py <prep.npz> <in.ply> <out.ply>
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch
import torch.nn.functional as tF
import nvdiffrast.torch as dr
import trimesh

from irrefine import Views, vertex_normals, sample, dev


def tangent(N):
    a = torch.where(N[:, :1].abs() < 0.9, torch.tensor([[1.0, 0, 0]], device=dev), torch.tensor([[0, 1.0, 0]], device=dev))
    t1 = tF.normalize(torch.cross(N, a, dim=1), dim=1)
    return t1, torch.cross(N, t1, dim=1)


def visible(vw, glctx, V, Fi, N, min_cos=0.35):
    """(n_views, n_verts) mask of the photos that see each vertex (on the current surface), and its cosines."""
    clip, zv = vw.clip(V)
    rast, _ = dr.rasterize(glctx, clip, Fi, (vw.H, vw.W))
    zb, _ = dr.interpolate(zv[..., None].contiguous(), rast, Fi)
    zb = torch.where(rast[..., 3:] > 0, zb, torch.full_like(zb, 1e9))[..., 0]
    Xc, uv = vw.cam(V)
    pix = uv.floor().long()
    inside = (pix[..., 0] >= 0) & (pix[..., 0] < vw.W) & (pix[..., 1] >= 0) & (pix[..., 1] < vw.H)
    px, py = pix[..., 0].clamp(0, vw.W - 1), pix[..., 1].clamp(0, vw.H - 1)
    ar = torch.arange(vw.n, device=dev)[:, None]
    L = vw.C[:, None] - V[None]
    cos = (tF.normalize(L, dim=-1) * N[None]).sum(-1)
    vis = inside & (Xc[..., 2] < zb[ar, py, px] + 1.0) & (vw.inner[ar, py, px] > 0.5) & (cos > min_cos)
    return vis, cos


@torch.no_grad()
def sweep(vw, V, N, vis, cos, offsets, step_mm, k, chunk=8000):
    """Matching cost (n_offsets, n_verts) of each vertex moved by each offset along its normal."""
    t1, t2 = tangent(N)
    o = torch.arange(k, device=dev, dtype=torch.float32) - (k - 1) / 2
    oa, ob = torch.meshgrid(o, o, indexing='ij')
    oa, ob = oa.reshape(1, -1, 1) * step_mm, ob.reshape(1, -1, 1) * step_mm
    out = torch.full((len(offsets), len(V)), float('nan'), device=dev)
    for s in range(0, len(V), chunk):
        e = min(len(V), s + chunk)
        Vc, Nc, T1, T2 = V[s:e], N[s:e], t1[s:e], t2[s:e]
        vc, cc = vis[:, s:e], cos[:, s:e]
        nv = e - s
        # one fixed set of photos per point: the patch inside the limb's outline at every depth tried (patches that
        # touch the background near the outline get better as the point moves inward: a bias towards shrinking)
        for off in (offsets[0], 0.0, offsets[-1]):
            P = Vc + Nc * off
            Q = P[:, None] + oa * T1[:, None] + ob * T2[:, None]
            _, uvq = vw.cam(Q.reshape(-1, 3))
            im = sample(vw.inner[:, None], uvq, vw.W, vw.H)[..., 0].reshape(vw.n, nv, k * k).min(-1).values
            vc = vc & (im > 0.5)
        for j, off in enumerate(offsets):
            P = Vc + Nc * off
            Q = P[:, None] + oa * T1[:, None] + ob * T2[:, None]
            _, uvq = vw.cam(Q.reshape(-1, 3))
            g = sample(vw.gray, uvq, vw.W, vw.H)[..., 0].reshape(vw.n, nv, k * k)
            mu, sd = g.mean(-1, keepdim=True), g.std(-1, keepdim=True)
            gn = (g - mu) / (sd + 0.004)
            w = vc.float() * cc.clamp(min=0) ** 2 * (sd[..., 0] ** 2 / (sd[..., 0] ** 2 + 0.01 ** 2))
            ws = w.sum(0)
            tm = (w[..., None] * gn).sum(0) / ws.clamp(min=1e-6)[:, None]
            c = (w * ((gn - tm[None]) ** 2).mean(-1)).sum(0) / ws.clamp(min=1e-6)
            c[(vc.sum(0) < 3) | (ws < 0.05)] = float('nan')
            out[j, s:e] = c
    return out


def laplacian(F, n):
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return sp.diags(np.asarray(A.sum(1)).ravel()) - A


def choose(costs, offsets):
    """Best offset per vertex (sub-step parabola) and a confidence 0..1 (how clearly it beats the other depths)."""
    C = costs.cpu().numpy()
    ok = ~np.isnan(C).any(0)
    Cn = np.where(np.isnan(C), np.inf, C)
    b = Cn.argmin(0)
    n = len(offsets)
    idx = np.arange(C.shape[1])
    lo, hi = np.clip(b - 1, 0, n - 1), np.clip(b + 1, 0, n - 1)
    c0, cl, ch = Cn[b, idx], Cn[lo, idx], Cn[hi, idx]
    den = cl - 2 * c0 + ch
    sub = np.where((b > 0) & (b < n - 1) & np.isfinite(den) & (den > 1e-9), 0.5 * (cl - ch) / np.where(den > 1e-9, den, 1), 0)
    step = offsets[1] - offsets[0]
    best = offsets[b] + np.clip(sub, -0.5, 0.5) * step
    med = np.median(np.where(np.isfinite(Cn), Cn, np.nan), 0)
    conf = np.clip((med - c0) / np.maximum(med, 1e-6), 0, 1)
    conf[~ok] = 0
    conf[(b == 0) | (b == n - 1)] *= 0.2  # the best depth lies at the end of the search: unreliable
    return best, conf, ok


def solve(F, n, best, conf, mu):
    """Smooth offsets: min sum conf (o - best)^2 + mu |grad o|^2."""
    L = laplacian(F, n)
    A = (sp.diags(conf + 1e-4) + mu * L).tocsc()
    return spl.spsolve(A, conf * best)


def run(prep, src, dst, log=print):
    t0 = time.time()
    z = np.load(prep)
    m = trimesh.load(src, process=False)
    Fn = np.asarray(m.faces)
    Fi = torch.tensor(Fn, dtype=torch.int32, device=dev)
    V = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
    glctx = dr.RasterizeCudaContext()
    passes = (  # (pyramid level, offsets mm, patch k, patch step px, smoothing)
        (1, np.arange(-6.0, 1.01, 0.25), 7, 1.0, 3.0),
        (0, np.arange(-1.5, 1.51, 0.1), 7, 1.0, 1.0),
    )
    for level, offsets, k, step_px, mu in passes:
        vw = Views(z, level)
        N = vertex_normals(V, Fi.long())
        vis, cos = visible(vw, glctx, V, Fi, N)
        with torch.no_grad():
            Xc, _ = vw.cam(V[::50])
            mm_px = (Xc[..., 2].median() / vw.f.mean()).item()
        costs = sweep(vw, V, N, vis, cos, offsets, mm_px * step_px, k)
        best, conf, ok = choose(costs, offsets)
        o = solve(Fn, V.shape[0], best, conf, mu)
        V = V + N * torch.tensor(o, dtype=torch.float32, device=dev)[:, None]
        log(f'[depth] L{level}: {ok.mean() * 100:.0f}% of points seen by 3+ photos, confidence median {np.median(conf[ok]):.2f}; '
            f'moved median {np.median(o):+.2f} mm, 5% {np.percentile(o, 5):+.2f}, 95% {np.percentile(o, 95):+.2f}  '
            f'{time.time() - t0:.0f}s')
        del vw
    trimesh.Trimesh(V.cpu().numpy(), Fn, process=False).export(dst)


if __name__ == '__main__':
    run(*sys.argv[1:4])
