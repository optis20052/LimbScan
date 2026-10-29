"""Photo colour for the mesh: every vertex takes its colour from the photos that see it most frontally.

Visibility is tested by ray casting against the mesh itself (a finger must not take the colour of the palm
behind it). Weights favour frontal, close, central views; only the best few views are blended so slightly
misaligned photos don't blur the skin. Each photo's exposure / white balance is equalised first (the phone's
auto-exposure changes as the background changes), so the skin colour is even all the way round.
"""
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def colorize(mesh, model_dir, image_dir, top_k=3, log=print):
    """mesh: trimesh in the model's metric frame; model_dir: undistorted (PINHOLE) COLMAP model."""
    import pycolmap
    rec = pycolmap.Reconstruction(model_dir)
    V = np.asarray(mesh.vertices, np.float64)
    Nrm = np.asarray(mesh.vertex_normals, np.float64)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.asarray(mesh.faces)))))
    n = len(V)
    views = []  # per photo: vertex indices, sampled colours, weights
    for im in rec.images.values():
        p = Path(image_dir) / im.name
        img = cv2.imread(str(p))
        if img is None:
            continue
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world().matrix()
        C = im.cam_from_world().inverse().translation
        Xc = V @ T[:3, :3].T + T[:3, 3]
        front = Xc[:, 2] > 1.0
        if front.sum() == 0:
            continue
        px = np.full((n, 2), -1.0)
        px[front] = cam.img_from_cam(Xc[front])
        h, w = img.shape[:2]
        inside = front & (px[:, 0] >= 1) & (px[:, 0] < w - 2) & (px[:, 1] >= 1) & (px[:, 1] < h - 2)
        if inside.sum() == 0:
            continue
        idx = np.where(inside)[0]
        dirs = V[idx] - C
        dist = np.linalg.norm(dirs, axis=1)
        dirs /= dist[:, None]
        rays = np.c_[np.repeat(C[None], len(idx), 0), dirs].astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        vis = np.abs(hit - dist) < np.maximum(1.0, 0.004 * dist)  # nothing in front of the vertex
        idx, dirs, dist = idx[vis], dirs[vis], dist[vis]
        cosang = -(Nrm[idx] * dirs).sum(1)
        ok = cosang > 0.25
        idx, cosang, dist = idx[ok], cosang[ok], dist[ok]
        if len(idx) == 0:
            continue
        uv = px[idx]
        centre = 1.0 - 0.5 * np.clip(np.hypot((uv[:, 0] - w / 2) / (w / 2), (uv[:, 1] - h / 2) / (h / 2)), 0, 1)
        wgt = cosang ** 4 * centre / (dist / 300.0) ** 2
        col = cv2.remap(img, uv[:, 0].astype(np.float32).reshape(1, -1), uv[:, 1].astype(np.float32).reshape(1, -1),
                        cv2.INTER_LINEAR).reshape(-1, 3)[:, ::-1].astype(np.float64)
        # skip the grey-painted background (silhouette edge pixels) when images were masked
        grey = (np.abs(col - 127).max(1) < 20) & ((col.max(1) - col.min(1)) < 12)
        wgt[grey] = 0
        views.append((idx, col.astype(np.float32), wgt))

    def blend():
        best_w = np.zeros((n, top_k)); best_c = np.zeros((n, top_k, 3))
        for (idx, col, wgt), g in zip(views, gains):
            c = col * g
            slot = np.argmin(best_w[idx], axis=1)  # keep the top_k weights per vertex
            better = wgt > best_w[idx, slot]
            ii, ss = idx[better], slot[better]
            best_w[ii, ss] = wgt[better]; best_c[ii, ss] = c[better]
        wsum = best_w.sum(1)
        ok = wsum > 0
        ref = np.zeros((n, 3))
        ref[ok] = (best_c[ok] * best_w[ok, :, None]).sum(1) / wsum[ok, None]
        return ref, ok

    # per-photo RGB gain: the median ratio between the blended colour and what this photo saw (2 rounds)
    gains = [np.ones(3, np.float32) for _ in views]
    for _ in range(2):
        ref, ok = blend()
        for k, (idx, col, wgt) in enumerate(views):
            sel = ok[idx] & (col.min(1) > 15) & (wgt > 0)
            if sel.sum() > 200:
                r = ref[idx[sel]] / (col[sel] * gains[k])
                gains[k] = (gains[k] * np.clip(np.median(r, axis=0), 0.6, 1.6)).astype(np.float32)
    ref, colored = blend()
    used = len(views)
    if views:
        gl = np.array([g.mean() for g in gains])
        log(f'[texture] exposure equalised: photo gains {gl.min():.2f}..{gl.max():.2f}')
    out = np.asarray(mesh.visual.vertex_colors[:, :3], np.float64).copy()
    out[colored] = ref[colored]
    mesh.visual.vertex_colors = np.c_[np.clip(out, 0, 255), np.full(n, 255)].astype(np.uint8)
    log(f'[texture] colours from {used} photos for {100 * colored.mean():.0f}% of vertices')
    return mesh
