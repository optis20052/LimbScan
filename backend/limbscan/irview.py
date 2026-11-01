"""Shaded renders of meshes (one row per mesh, 4 views around the long axis): python irview.py out.png mesh ...
Each mesh is centred and turned so its long axis is vertical (PCA). --same: use the first mesh's frame for all
(meshes already aligned)."""
import sys
import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh

dev = 'cuda'


def load(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    if m.extents.max() < 5:
        m.apply_scale(1000.0)
    return m


def frame(V):
    c = V.mean(0)
    _, _, vt = np.linalg.svd(V - c, full_matrices=False)
    R = np.stack([vt[1], vt[0], np.cross(vt[1], vt[0])])  # long axis -> y
    return c, R


def render(m, c, R, S=420, half=None):
    V = (m.vertices - c) @ R.T
    half = half or np.abs(V).max() * 1.05
    glctx = dr.RasterizeCudaContext()
    F = torch.tensor(m.faces, dtype=torch.int32, device=dev)
    out = []
    for ang in (0, 90, 180, 270):
        a = np.deg2rad(ang)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        W = V @ Ry.T
        v = torch.tensor(W, dtype=torch.float32, device=dev)
        clip = torch.cat([v[:, :1] / half, -v[:, 1:2] / half, v[:, 2:3] / (half * 2) * -1, torch.ones_like(v[:, :1])], 1)[None]
        rast, _ = dr.rasterize(glctx, clip.contiguous(), F, (S, S))
        fn = torch.tensor(trimesh.Trimesh(W, m.faces, process=False).face_normals, dtype=torch.float32, device=dev)
        tri = rast[0, ..., 3].long() - 1
        n = fn[tri.clamp(min=0)]
        sh = (n @ torch.tensor([0.3, 0.4, 0.86], device=dev)).abs() * 0.8 + 0.2
        img = torch.where(tri[..., None] >= 0, sh[..., None].repeat(1, 1, 3), torch.full((S, S, 3), 0.12, device=dev))
        out.append((img.cpu().numpy() * 255).astype(np.uint8))
    return np.hstack(out), half


if __name__ == '__main__':
    a = sys.argv[1:]
    same = '--same' in a
    a = [x for x in a if x != '--same']
    rows, ref = [], None
    for p in a[1:]:
        m = load(p)
        if ref is None or not same:
            ref = frame(m.vertices) + (None,)
        img, h = render(m, ref[0], ref[1], half=ref[2])
        if same:
            ref = (ref[0], ref[1], h)
        cv2.putText(img, p.split('/')[-1][:40], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(img)
    cv2.imwrite(a[0], np.vstack(rows))
