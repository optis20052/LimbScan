"""GPU surface refinement: move the starting surface (the outline hull) until it explains every photo.

Two terms, on the GPU (nvdiffrast rasteriser, PyTorch):
  outline  - the surface drawn from each camera covers exactly the limb's mask in that photo;
  shading  - every surface point, seen in several photos, must have one colour (its albedo) once the light is
             accounted for: the phone's torch at the camera (cos / distance^2, moves with the phone) plus the room
             light (fixed in the room, smooth in the normal) and each photo's exposure. Texture aligns the surface
             like stereo does; on plain skin or plastic the torch shading gives the normals (photometric stereo).
The vertex updates go through Nicolet et al.'s "large steps" smoothing (the surface stays smooth and does not
tangle) and run coarse to fine over an image pyramid.

Usage: python irrefine.py <prep.npz> <out.ply> [iterations per level]
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as tF
import nvdiffrast.torch as dr
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import from_differential, to_differential

dev = 'cuda'


def vertex_normals(V, F):
    fn = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1)
    n = torch.zeros_like(V)
    for k in range(3):
        n.index_add_(0, F[:, k], fn)
    return tF.normalize(n, dim=1)


class Views:
    """All photos at one pyramid level, padded to one size (the rasteriser draws all views in one batch)."""

    def __init__(self, z, level):
        n = len(z['names'])
        s = 2 ** level
        imgs, masks = [], []
        for i in range(n):
            im = z[f'img_{i}'].astype(np.float32) / 255.0
            m = z[f'mask_{i}'].astype(np.float32) / 255.0
            h, w = m.shape
            h2, w2 = h // s, w // s
            im = im[:h2 * s, :w2 * s].reshape(h2, s, w2, s, 3).mean((1, 3))
            m = m[:h2 * s, :w2 * s].reshape(h2, s, w2, s).mean((1, 3))
            imgs.append(im)
            masks.append(m)
        H = max(m.shape[0] for m in masks)
        W = max(m.shape[1] for m in masks)
        H, W = -(-H // 8) * 8, -(-W // 8) * 8
        self.img = torch.zeros(n, 3, H, W)
        self.mask = torch.zeros(n, H, W)
        self.valid = torch.zeros(n, H, W)
        for i, (im, m) in enumerate(zip(imgs, masks)):
            h, w = m.shape
            self.img[i, :, :h, :w] = torch.from_numpy(im).permute(2, 0, 1) ** 2.2  # sRGB -> linear (approx.)
            self.mask[i, :h, :w] = torch.from_numpy(m)
            self.valid[i, :h, :w] = 1
        self.img, self.mask, self.valid = self.img.to(dev), self.mask.to(dev), self.valid.to(dev)
        # MoGe-2 surface normals per photo (camera frame), when computed: averaged down to this level
        self.nrm = None
        if any(f'nrm_{i}' in z for i in range(n)):
            self.nrm = torch.zeros(n, 3, H, W)
            for i in range(n):
                if f'nrm_{i}' not in z:
                    continue
                q = z[f'nrm_{i}'].astype(np.float32)
                v = q / 127.5 - 1.0
                v[(q == 0).all(-1)] = 0
                h, w = v.shape[:2]
                h2, w2 = h // s, w // s
                v = v[:h2 * s, :w2 * s].reshape(h2, s, w2, s, 3).mean((1, 3))
                self.nrm[i, :, :h2, :w2] = torch.from_numpy(v).permute(2, 0, 1)
            self.nrm = self.nrm.to(dev)
        er = 2 * max(1, 4 >> level) + 1  # mask edge pixels are uncertain: 4 px at full size
        self.inner = -tF.max_pool2d(-self.mask[:, None], er, 1, er // 2)[:, 0]
        self.gray = (self.img ** (1 / 2.2) * torch.tensor([0.3, 0.59, 0.11], device=dev)[:, None, None]).sum(1, keepdim=True)
        K = torch.tensor(z['K'], dtype=torch.float32)
        self.f = torch.stack([K[:, 0, 0], K[:, 1, 1]], 1).to(dev) / s
        self.c = torch.stack([K[:, 0, 2], K[:, 1, 2]], 1).to(dev) / s
        self.R = torch.tensor(z['R'], dtype=torch.float32, device=dev)
        self.t = torch.tensor(z['t'], dtype=torch.float32, device=dev)
        self.C = -torch.einsum('nji,nj->ni', self.R, self.t)  # camera centres
        self.H, self.W, self.n = H, W, n

    def highpass(self, sigma):
        """Fine texture only: the grey image minus its blur (the torch moves with the phone, so the smooth shading
        differs from photo to photo; pores, creases and sculpted detail stay put)."""
        r = int(3 * sigma + 1)
        x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        b = tF.conv2d(tF.pad(self.gray, (r, r, 0, 0), mode='replicate'), g.view(1, 1, 1, -1))
        b = tF.conv2d(tF.pad(b, (0, 0, r, r), mode='replicate'), g.view(1, 1, -1, 1))
        self.gray = self.gray - b

    def cam(self, X):
        """Camera-frame points (n, N, 3) and pixels (n, N, 2) of world points X (N, 3)."""
        Xc = torch.einsum('nij,pj->npi', self.R, X) + self.t[:, None]
        z = Xc[..., 2:3].clamp(min=1.0)
        uv = Xc[..., :2] / z * self.f[:, None] + self.c[:, None]
        return Xc, uv

    def clip(self, X, near=50.0, far=2000.0):
        Xc, uv = self.cam(X)
        z = Xc[..., 2]
        x = (2 * uv[..., 0] / self.W - 1) * z
        y = (2 * uv[..., 1] / self.H - 1) * z
        zc = (z * (far + near) - 2 * far * near) / (far - near)
        return torch.stack([x, y, zc, z], -1).contiguous(), z


def sample(img, uv, W, H):
    """Bilinear samples of img (n, C, H, W) at pixels uv (n, N, 2) -> (n, N, C)."""
    g = torch.stack([2 * uv[..., 0] / W - 1, 2 * uv[..., 1] / H - 1], -1)[:, None]
    return tF.grid_sample(img, g, mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0].permute(0, 2, 1)


def ncc(vw, P, Pn, vis, cos, level, patch_px, k=5, min_cos=0.4):
    """Patch matching: a small square on the surface's tangent plane at every point, sampled in every photo that
    sees it; the zero-mean, unit-variance patches must agree (insensitive to the photos' brightness and to smooth
    shading). Returns (loss, points used)."""
    n_pts = len(P)
    a = torch.where((Pn[:, :1].abs() < 0.9), torch.tensor([[1.0, 0, 0]], device=dev), torch.tensor([[0, 1.0, 0]], device=dev))
    t1 = tF.normalize(torch.cross(Pn, a, dim=1), dim=1)
    t2 = torch.cross(Pn, t1, dim=1)
    with torch.no_grad():
        Xc, _ = vw.cam(P[:2000].detach())
        mm_per_px = (Xc[..., 2].median() / vw.f.mean()).item()
    step = mm_per_px * patch_px
    o = torch.arange(k, device=dev, dtype=torch.float32) - (k - 1) / 2
    oa, ob = torch.meshgrid(o, o, indexing='ij')
    Q = P[:, None] + (oa.reshape(1, -1, 1) * t1[:, None] + ob.reshape(1, -1, 1) * t2[:, None]) * step
    _, uvq = vw.cam(Q.reshape(-1, 3))
    g = sample(vw.gray, uvq, vw.W, vw.H)[..., 0].reshape(vw.n, n_pts, k * k)
    with torch.no_grad():  # the whole patch on the limb in that photo (no background pixels at the outline)
        inmask = sample(vw.inner[:, None], uvq, vw.W, vw.H)[..., 0].reshape(vw.n, n_pts, k * k).min(-1).values > 0.5
        vis = vis & inmask & (cos[..., 0] > min_cos)
    mu = g.mean(-1, keepdim=True)
    sd = g.std(-1, keepdim=True)
    gn = (g - mu) / (sd + 0.004)
    w = (vis.float() * cos[..., 0].detach().clamp(min=0) ** 2 * (sd[..., 0] ** 2 / (sd[..., 0] ** 2 + 0.01 ** 2)).detach())
    ws = w.sum(0)
    ok = (vis.sum(0) >= 2) & (ws > 1e-4)
    tmpl = (w[..., None] * gn).sum(0) / ws.clamp(min=1e-6)[:, None]
    e = ((gn - tmpl[None]) ** 2).mean(-1)
    return (w * e)[:, ok].sum() / (w[:, ok].sum() + 1e-6), ok


def refine(prep, out, iters=(150, 150, 120), lam=20.0, lr=0.08, w_sil=1.0, w_photo=1.0, n_pts=200000, mode='ncc',
           patch_px=1.5, log=print):
    n_pts = int(n_pts)
    z = np.load(prep)
    V0 = torch.tensor(z['V'], dtype=torch.float32, device=dev)
    F = torch.tensor(z['F'], dtype=torch.int64, device=dev)
    Fi = F.int().contiguous()
    M = compute_matrix(V0, F, lam)
    u = to_differential(M, V0).clone().requires_grad_(True)
    opt = AdamUniform([u], lr=lr)
    glctx = dr.RasterizeCudaContext()
    n_views = len(z['names'])
    # light: torch gain, room light (SH order 1 per channel), per-photo exposure (log, per channel)
    p_torch = torch.zeros(1, device=dev, requires_grad=True)
    p_amb = torch.tensor([[-1.0, 0, 0, 0]] * 3, device=dev).requires_grad_(True)
    p_exp = torch.zeros(n_views, 3, device=dev, requires_grad=True)
    lopt = torch.optim.Adam([p_torch, p_amb, p_exp], lr=0.02)
    t0 = time.time()
    for level, n_it in zip((2, 1, 0), iters):
        vw = Views(z, level)
        for it in range(n_it):
            V = from_differential(M, u, 'Cholesky')
            N = vertex_normals(V, F)
            clip, zv = vw.clip(V)
            rast, _ = dr.rasterize(glctx, clip, Fi, (vw.H, vw.W))
            cover = dr.antialias((rast[..., 3:] > 0).float(), rast, clip, Fi)[..., 0]
            sil = (((cover - vw.mask) ** 2) * vw.valid).sum() / vw.valid.sum()
            # depth buffers for visibility (no gradient)
            with torch.no_grad():
                zb, _ = dr.interpolate(zv[..., None].contiguous(), rast, Fi)
                zb = torch.where(rast[..., 3:] > 0, zb, torch.full_like(zb, 1e9))[..., 0]
            # surface samples, area-weighted
            fa = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1).norm(dim=1).detach()
            fi = torch.multinomial(fa, n_pts, replacement=True)
            b = torch.rand(n_pts, 2, device=dev)
            flip = b.sum(1) > 1
            b[flip] = 1 - b[flip]
            bw = torch.stack([1 - b[:, 0] - b[:, 1], b[:, 0], b[:, 1]], 1)
            P = (V[F[fi]] * bw[..., None]).sum(1)
            Pn = tF.normalize((N[F[fi]] * bw[..., None]).sum(1), dim=1)
            Xc, uv = vw.cam(P)
            with torch.no_grad():
                pix = uv.floor().long()
                inside = (pix[..., 0] >= 0) & (pix[..., 0] < vw.W) & (pix[..., 1] >= 0) & (pix[..., 1] < vw.H)
                px, py = pix[..., 0].clamp(0, vw.W - 1), pix[..., 1].clamp(0, vw.H - 1)
                ar = torch.arange(vw.n, device=dev)[:, None]
                zbuf = zb[ar, py, px]
                vis = inside & (Xc[..., 2] < zbuf + 1.5) & (vw.valid[ar, py, px] > 0)
                vis &= vw.mask[ar, py, px] > 0.5
            L = vw.C[:, None] - P[None]
            d = L.norm(dim=-1, keepdim=True)
            cos = (L / d * Pn[None]).sum(-1, keepdim=True)
            vis = vis & (cos[..., 0].detach() > 0.25)
            if mode == 'ncc':
                photo, ok = ncc(vw, P, Pn, vis, cos, level, patch_px)
            else:
                I = sample(vw.img, uv, vw.W, vw.H)  # (n, N, 3) linear
                vis = vis & (I.detach().max(-1).values < 0.9)  # saturated highlights carry no shading
                shade_t = torch.nn.functional.softplus(p_torch + 3) * cos.clamp(min=0.05) / (d / 350.0) ** 2
                sh = torch.cat([torch.ones_like(Pn[:, :1]), Pn], 1)  # (N, 4)
                amb = torch.nn.functional.softplus(sh @ p_amb.T)  # (N, 3)
                shade = shade_t + amb[None]
                r = torch.log(I + 0.005) - torch.log(shade) - p_exp[:, None]
                w = vis.float() * cos[..., 0].detach().clamp(min=0) ** 2
                ws = w.sum(0)
                ok = (vis.sum(0) >= 2) & (ws > 0)
                a = (w[..., None] * r).sum(0) / ws.clamp(min=1e-6)[:, None]
                e = r - a[None]
                rob = torch.sqrt(e ** 2 + 0.02 ** 2) - 0.02  # Charbonnier
                photo = (w[..., None] * rob)[:, ok].sum() / (w[:, ok].sum() * 3 + 1e-6)
            loss = w_sil * sil * 100 + w_photo * photo * 10
            opt.zero_grad()
            lopt.zero_grad()
            loss.backward()
            opt.step()
            lopt.step()
            with torch.no_grad():
                p_exp -= p_exp.mean(0, keepdim=True)
            if it % 30 == 0 or it == n_it - 1:
                mv = (V - V0).norm(dim=1)
                log(f'[refine] L{level} it {it:3d}  outline {sil.item() * 100:.4f}  shading {photo.item() * 10:.4f}  '
                    f'points {ok.float().mean().item() * 100:.0f}% seen twice  moved med {mv.median().item():.2f} '
                    f'max {mv.max().item():.2f} mm  torch {torch.nn.functional.softplus(p_torch + 3).item():.2f}  '
                    f'{time.time() - t0:.0f}s')
        del vw
    V = from_differential(M, u, 'Cholesky').detach().cpu().numpy()
    import trimesh
    trimesh.Trimesh(V, z['F'], process=False).export(out)
    log(f'[refine] done in {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    its = tuple(int(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 else (150, 150, 120)
    kw = {}
    for a in sys.argv[4:]:
        k, v = a.split('=')
        try:
            kw[k] = float(v)
        except ValueError:
            kw[k] = v
    refine(sys.argv[1], sys.argv[2], its, **kw)
