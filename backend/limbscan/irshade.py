"""Torch shading model: I = exposure_j * albedo(x) * (torch * cos / (d/350)^2 + room(n)). Albedo is eliminated
per point (mean of log over the photos that see it). Shared by the refinement and the tests."""
import torch
import torch.nn.functional as tF

from irrefine import sample, dev


class Light(torch.nn.Module):
    def __init__(self, n_views):
        super().__init__()
        self.torch_ = torch.nn.Parameter(torch.zeros(1, device=dev))
        self.room = torch.nn.Parameter(torch.tensor([[-2.0, 0, 0, 0]] * 3, device=dev))
        self.exp = torch.nn.Parameter(torch.zeros(n_views, 3, device=dev))
        self.ks = torch.nn.Parameter(torch.tensor([-3.0], device=dev))   # shine (log), the same colour as the light
        self.kp = torch.nn.Parameter(torch.tensor([2.5], device=dev))    # shine sharpness (log of the exponent)

    def shade(self, cos, d, Pn):
        t = tF.softplus(self.torch_ + 3) * cos.clamp(min=0.02) / (d / 350.0) ** 2
        sh = torch.cat([torch.ones_like(Pn[:, :1]), Pn], 1)
        return t + tF.softplus(sh @ self.room.T)[None]


def cost(vw, light, P, Pn, vis, min_cos=0.3, spec=True):
    if spec:
        return cost_spec(vw, light, P, Pn, vis, min_cos)
    return cost_log(vw, light, P, Pn, vis, min_cos)


def _inputs(vw, P, Pn, vis, min_cos):
    L = vw.C[:, None] - P[None]
    d = L.norm(dim=-1, keepdim=True)
    cos = (L / d * Pn[None]).sum(-1, keepdim=True)
    _, uv = vw.cam(P)
    I = sample(vw.img, uv, vw.W, vw.H)
    with torch.no_grad():
        inn = sample(vw.inner[:, None], uv, vw.W, vw.H)[..., 0] > 0.5
        v = vis & inn & (cos[..., 0] > min_cos) & (I.max(-1).values < 0.92) & (I.min(-1).values > 0.003)
    return d, cos, I, v


def cost_spec(vw, light, P, Pn, vis, min_cos=0.3):
    """The log model with the plastic's/skin's shine removed first: log(I/exposure - shine) - log(diffuse light)
    must be one albedo per point (the torch is at the camera: the shine peaks where the surface faces the phone)."""
    d, cos, I, v = _inputs(vw, P, Pn, vis, min_cos)
    T = tF.softplus(light.torch_ + 3)
    dd = (d / 350.0) ** 2
    sh = torch.cat([torch.ones_like(Pn[:, :1]), Pn], 1)
    D = T * cos.clamp(min=0.02) / dd + tF.softplus(sh @ light.room.T)[None]
    S = torch.exp(light.ks) * T * cos.clamp(min=0) ** torch.exp(light.kp) / dd
    Ie = I * torch.exp(-light.exp[:, None])
    r = torch.log((Ie - S).clamp(min=1e-3)) - torch.log(D)
    w = v.float()
    ws = w.sum(0)
    ok = v.sum(0) >= 3
    a = (w[..., None] * r).sum(0) / ws.clamp(min=1)[:, None]
    e = r - a[None]
    rob = torch.sqrt(e ** 2 + 0.03 ** 2) - 0.03
    return (w[..., None] * rob)[:, ok].sum() / (w[:, ok].sum() * 3 + 1e-6), ok


def cost_log(vw, light, P, Pn, vis, min_cos=0.3):
    """Robust disagreement of the albedo seen by each photo (log space). vis: (n, N) bool."""
    L = vw.C[:, None] - P[None]
    d = L.norm(dim=-1, keepdim=True)
    cos = (L / d * Pn[None]).sum(-1, keepdim=True)
    I = sample(vw.img, vw.cam(P)[1], vw.W, vw.H)
    with torch.no_grad():
        _, uv = vw.cam(P)
        inn = sample(vw.inner[:, None], uv, vw.W, vw.H)[..., 0] > 0.5
        v = vis & inn & (cos[..., 0] > min_cos) & (I.max(-1).values < 0.92) & (I.min(-1).values > 0.003)
    r = torch.log(I.clamp(min=1e-4)) - torch.log(light.shade(cos, d, Pn)) - light.exp[:, None]
    w = v.float()
    ws = w.sum(0)
    ok = (v.sum(0) >= 3)
    a = (w[..., None] * r).sum(0) / ws.clamp(min=1)[:, None]
    e = r - a[None]
    rob = torch.sqrt(e ** 2 + 0.03 ** 2) - 0.03
    return (w[..., None] * rob)[:, ok].sum() / (w[:, ok].sum() * 3 + 1e-6), ok
