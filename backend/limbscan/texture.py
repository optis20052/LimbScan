"""Photo colour for the mesh: every vertex takes its colour from the photos that see it most frontally.

Visibility is tested by ray casting against the mesh itself (a finger must not take the colour of the palm
behind it). Weights favour frontal, close, central views; only the best few views are blended so slightly
misaligned photos don't blur the skin. Each photo's exposure / white balance is equalised first (the phone's
auto-exposure changes as the background changes), so the skin colour is even all the way round.
"""
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


EDGE_PX = 0.006  # cut-out edge band (share of the photo's height): no colour there, the background bleeds in
FULL_PX = 0.02  # full weight this far inside the cut-out
OUTLIER = 45.0
TOP_K = int(os.environ.get('LIMBSCAN_TEX_K', '3'))
POW = float(os.environ.get('LIMBSCAN_TEX_POW', '4'))  # a view whose colour is this far (RGB) from the others' median is left out of the blend


def colorize(mesh, model_dir, image_dir, mask_dir=None, top_k=None, log=print):
    """mesh: trimesh in the model's metric frame; model_dir: COLMAP model of the photos in image_dir (any camera
    model). mask_dir: the limb cut-out per photo (same names, .png): colour only from well inside the cut-out, so
    background (clothes, the floor, the grey paint of masked photos) never reaches the skin."""
    import pycolmap
    top_k = top_k or TOP_K
    rec = pycolmap.Reconstruction(model_dir)
    V = np.asarray(mesh.vertices, np.float64)
    Nrm = np.asarray(mesh.vertex_normals, np.float64)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.asarray(mesh.faces)))))
    n = len(V)
    names = []
    views = []  # per photo: vertex indices, sampled colours, weights
    seen = np.zeros(n, np.int32)  # evidence: photos that saw each vertex within 60 degrees of head-on
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
        inner = None
        if mask_dir is not None:
            mp = Path(mask_dir) / (Path(im.name).stem + '.png')
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            d = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5)
            inner = np.clip((d - EDGE_PX * h) / ((FULL_PX - EDGE_PX) * h), 0, 1).astype(np.float32)
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
        seen[idx[cosang > 0.5]] += 1
        ok = cosang > 0.25
        idx, cosang, dist = idx[ok], cosang[ok], dist[ok]
        if len(idx) == 0:
            continue
        uv = px[idx]
        centre = 1.0 - 0.5 * np.clip(np.hypot((uv[:, 0] - w / 2) / (w / 2), (uv[:, 1] - h / 2) / (h / 2)), 0, 1)
        wgt = cosang ** POW * centre / (dist / 300.0) ** 2
        col = cv2.remap(img, uv[:, 0].astype(np.float32).reshape(1, -1), uv[:, 1].astype(np.float32).reshape(1, -1),
                        cv2.INTER_LINEAR).reshape(-1, 3)[:, ::-1].astype(np.float64)
        if inner is not None:
            wgt = wgt * cv2.remap(inner, uv[:, 0].astype(np.float32).reshape(1, -1),
                                  uv[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).reshape(-1)
        else:
            # skip the grey-painted background (silhouette edge pixels) when images were masked
            grey = (np.abs(col - 127).max(1) < 20) & ((col.max(1) - col.min(1)) < 12)
            wgt[grey] = 0
        keep = wgt > 0
        idx, col, wgt = idx[keep], col[keep], wgt[keep]
        views.append((idx, col.astype(np.float32), wgt))
        names.append(im.name)

    def blend():
        best_w = np.zeros((n, top_k)); best_c = np.zeros((n, top_k, 3))
        for (idx, col, wgt), g in zip(views, gains):
            c = col * g
            slot = np.argmin(best_w[idx], axis=1)  # keep the top_k weights per vertex
            better = wgt > best_w[idx, slot]
            ii, ss = idx[better], slot[better]
            best_w[ii, ss] = wgt[better]; best_c[ii, ss] = c[better]
        # a view that disagrees with the others (a shadow, a reflection, a slightly wrong camera at an edge) is
        # left out rather than smeared in
        used = best_w > 0
        with warnings.catch_warnings():  # vertices no photo saw: all-NaN rows
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(np.where(used[..., None], best_c, np.nan), axis=1)
        far = used & (np.linalg.norm(best_c - med[:, None], axis=2) > OUTLIER) & (used.sum(1) >= 3)[:, None]
        best_w = np.where(far, 0.0, best_w)
        wsum = best_w.sum(1)
        ok = wsum > 0
        ref = np.zeros((n, 3))
        ref[ok] = (best_c[ok] * best_w[ok, :, None]).sum(1) / wsum[ok, None]
        return ref, ok

    # per-photo RGB gain (exposure and white balance): every pair of photos that sees the same skin gives the
    # colour ratio between them there (median over the shared vertices); all ratios are solved together in the
    # least-squares sense, anchored so the typical photo keeps its colour. A blend-based reference doesn't work: a
    # run of photos in warm window light from similar angles is its own reference and stays warm. Clipped pixels
    # (any channel >= 250) and dark ones say nothing about the colour and are not used.
    gains = [np.ones(3, np.float32) for _ in views]
    m = len(views)
    if m >= 2:
        L = np.full((m, n, 3), np.nan, np.float32)
        for k, (idx, col, wgt) in enumerate(views):
            good = (col.min(1) > 15) & (col.max(1) < 250) & (wgt >= np.percentile(wgt, 30))
            L[k, idx[good]] = np.log(col[good])
        seen_k = ~np.isnan(L[..., 0])
        rows, rhs, wts = [], [], []
        for a in range(m):
            for b in range(a + 1, m):
                both = seen_k[a] & seen_k[b]
                c = int(both.sum())
                if c < 300:
                    continue
                rows.append((a, b)); rhs.append(np.median(L[a, both] - L[b, both], axis=0)); wts.append(np.sqrt(c))
        if rows:
            A = np.zeros((len(rows) + 1, m)); B = np.zeros((len(rows) + 1, 3))
            for r, ((a, b), d, w_) in enumerate(zip(rows, rhs, wts)):
                A[r, a], A[r, b], B[r] = w_, -w_, -w_ * d  # log g_a + log c_a = log g_b + log c_b
            A[-1] = 1.0  # mean log gain 0
            lg = np.linalg.lstsq(A, B, rcond=None)[0]
            linked = np.zeros(m, bool)
            for a, b in rows:
                linked[a] = linked[b] = True
            lg -= np.median(lg[linked], axis=0)
            gains = [np.exp(np.clip(g, np.log(0.4), np.log(2.5))).astype(np.float32) if linked[k]
                     else np.ones(3, np.float32) for k, g in enumerate(lg)]
        del L
    ref, colored = blend()
    used = len(views)
    if views:
        gl = np.array([g.mean() for g in gains])
        log(f'[texture] exposure equalised: photo gains {gl.min():.2f}..{gl.max():.2f}')
        if os.environ.get('LIMBSCAN_TEX_DEBUG'):
            for nm, g, (idx, col, wgt) in zip(names, gains, views):
                log(f'  {nm} gain {np.round(g, 2)} raw median {np.round(np.median(col[wgt > np.median(wgt)], 0))} n {len(idx)}')
    out = np.asarray(mesh.visual.vertex_colors[:, :3], np.float64).copy()
    out[colored] = ref[colored]
    if (~colored).any() and colored.any():
        # parts no photo saw well enough (between fingers, under the thumb) take the colour of the skin around them
        # instead of a flat grey: spread from coloured neighbours over the mesh
        import scipy.sparse as sp
        E = mesh.edges_unique
        A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        known = colored.copy()
        col = np.where(known[:, None], out, 0.0)
        for _ in range(200):
            s_ = A @ (col * known[:, None])
            c_ = A @ known.astype(float)
            new = (~known) & (c_ > 0)
            if not new.any():
                break
            col[new] = s_[new] / c_[new][:, None]
            known |= new
        out[~colored] = col[~colored]
    mesh.visual.vertex_colors = np.c_[np.clip(out, 0, 255), np.full(n, 255)].astype(np.uint8)
    log(f'[texture] colours from {used} photos for {100 * colored.mean():.0f}% of vertices')
    mesh.metadata['seen'] = seen
    return mesh


EVIDENCE_COLOURS = {'well': (63, 185, 80), 'weak': (210, 153, 34), 'unseen': (248, 81, 73)}


def evidence(mesh, seen, good=4, weak=2):
    """The surface coloured by how many photos saw each part head-on (>= good: green, weak..good-1: amber,
    fewer: red, shape there comes from its surroundings only), and the area shares. Nothing is filled in or
    smoothed on the strength of it: it is a report of the evidence behind each part of the model."""
    import trimesh
    seen = np.asarray(seen)
    cls = np.where(seen >= good, 0, np.where(seen >= weak, 1, 2))
    col = np.array([EVIDENCE_COLOURS[k] for k in ('well', 'weak', 'unseen')], np.uint8)[cls]
    m = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
    m.visual.vertex_colors = np.c_[col, np.full(len(col), 255, np.uint8)]
    fa = mesh.area_faces
    fc = cls[mesh.faces].max(1)  # a face is as weak as its weakest corner
    tot = fa.sum()
    share = {k: round(float(100 * fa[fc == i].sum() / tot), 1) for i, k in enumerate(('well', 'weak', 'unseen'))}
    return m, {'pct_well_seen': share['well'], 'pct_weak': share['weak'], 'pct_unseen': share['unseen'],
               'good_views': good, 'weak_views': weak}
