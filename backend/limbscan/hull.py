"""The limb's shape from its outlines (visual hull): deterministic and independent of surface texture.

Every photo has an exact camera (SfM, metric) and a clean limb mask. A point in space belongs to the limb only if
it falls inside the limb's outline in every photo that sees it. Intersecting all the outlines gives a closed solid
that is exact wherever the surface is convex as seen from the camera path (forearm, wrist, the back of the hand)
and only too big in hollows no outline can see into (the cupped palm, the gaps between closed fingers). Stereo
points from the photos' texture then pull the surface into those hollows (see refine()).

Texture-based stereo alone fails on smooth, even surfaces (plain skin, the printed test model): lumps and holes.
"""
import os
from pathlib import Path

import cv2
import numpy as np


def _views(model_dir, mask_dir, grow_px):
    import pycolmap
    rec = pycolmap.Reconstruction(str(model_dir))
    views = []
    for im in rec.images.values():
        m = cv2.imread(str(Path(mask_dir) / (Path(im.name).stem + '.png')), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 127).astype(np.uint8)
        if grow_px:  # a mask edge a pixel or two inside the real outline must not shave the limb
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)))
        T = im.cam_from_world()
        cam = rec.cameras[im.camera_id]
        views.append((T.rotation.matrix().astype(np.float32), np.asarray(T.translation, np.float32), cam, m.astype(bool)))
    return views


def _project(cam, Xc):
    """Pixel coordinates of camera-frame points (N, 3) for PINHOLE / OPENCV cameras (lens distortion included)."""
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    p = np.asarray(cam.params, np.float64)
    if cam.model_name == 'OPENCV':
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        x, y = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    elif cam.model_name == 'PINHOLE':
        fx, fy, cx, cy = p
    else:
        raise ValueError(f'camera model {cam.model_name}')
    return fx * x + cx, fy * y + cy


def mask_agreement(views, points):
    """Per view: the share of the (stereo) limb points seen in the photo that fall inside its mask. A mask that
    missed part of the limb (a wrist cut off, a finger lost) scores low."""
    out = []
    for R, t, cam, m in views:
        # a few pixels of slack: stereo points on the limb's rim land right on the outline
        tol = max(3, int(0.008 * m.shape[1]))
        m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))) > 0
        Xc = points @ R.T + t
        front = Xc[:, 2] > 1.0
        u, v = _project(cam, np.where(front[:, None], Xc, 1.0))
        ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        ins = front & (ui >= 0) & (vi >= 0) & (ui < m.shape[1]) & (vi < m.shape[0])
        out.append(m[vi[ins], ui[ins]].mean() if ins.sum() > 100 else 0.0)
    return np.array(out)


MISS_SHARE = float(os.environ.get('LIMBSCAN_HULL_MISS_SHARE', '0.1'))
MISS_MIN = int(os.environ.get('LIMBSCAN_HULL_MISS_MIN', '2'))
CREST_DEG = float(os.environ.get('LIMBSCAN_HULL_CREST_DEG', '12'))  # round off crests sharper than this (degrees; 1 Nov: thumb-web fold gone, 5 mold scans 95-98 %)
FINE_MM = float(os.environ.get('LIMBSCAN_HULL_FINE_MM', '0.5'))  # >0: the hand carved on a grid this fine
SEPARATE = os.environ.get('LIMBSCAN_HULL_SEPARATE', '0') == '1'  # fingers separated where the photos show it
WEB_MM = float(os.environ.get('LIMBSCAN_HULL_WEB_MM', '12'))  # from this far beyond the knuckle joint
FINE_FROM = float(os.environ.get('LIMBSCAN_HULL_FINE_FROM', '55'))  # mm beyond the wrist (not the palm)
FINE_FACES = int(os.environ.get('LIMBSCAN_HULL_FINE_FACES', '180000'))
FINE_SMOOTH = float(os.environ.get('LIMBSCAN_HULL_FINE_SMOOTH', '1.2'))  # (fine voxels)
SLIT_PX = int(os.environ.get('LIMBSCAN_HULL_SLIT_PX', '15'))  # >0: slits narrower than this inside a cut-out carve with one photo
ESCAPE = float(os.environ.get('LIMBSCAN_HULL_ESCAPE', '0.15'))  # >0: fill funnel pits open in less than this share of directions
PIT_MM = float(os.environ.get('LIMBSCAN_HULL_PIT_MM', '0'))  # >0: fill pockets up to this size beyond the wrist
CLOSE_MM = float(os.environ.get('LIMBSCAN_HULL_CLOSE_MM', '2.5'))
OPEN_MM = float(os.environ.get('LIMBSCAN_HULL_OPEN_MM', '3'))  # remove bridges thinner than ~2x this beyond the wrist (31 Oct: renders of 4 mold scans cleaner, scores equal)
HAND_SHARE = float(os.environ.get('LIMBSCAN_HULL_HAND_SHARE', '0.02'))  # carving rule beyond the wrist (finger gaps; 31 Oct: 16 printed-hand scans 93.6 -> 94.1 %, real arms unchanged)
SMOOTH = float(os.environ.get('LIMBSCAN_HULL_SMOOTH', '1.5'))
VOXEL = float(os.environ.get('LIMBSCAN_HULL_VOXEL', '1.0'))
# stereo in the hull's unpinned regions tore the palm on plain surfaces; the GPU shape step carves those hollows
BLEND = os.environ.get('LIMBSCAN_HULL_BLEND', '0') == '1'
CHECK = os.environ.get('LIMBSCAN_MASK_CHECK', '1') == '1'  # drop cut-outs that disagree with the others


def carve(model_dir, mask_dir, lo, hi, voxel=1.0, min_views=6, miss_share=None, grow_px=2, points=None,
          min_agreement=0.9, log=print, keep=None, hand=None):
    """Occupancy grid over the box lo..hi (mm): 1 where a point is inside the limb outline in (nearly) every photo
    seeing it, seen by >= `min_views` photos. Up to `miss_share` of those photos (at least 2) may disagree, and masks
    that disagree with the stereo `points` (< `min_agreement` of them inside) are left out: a wrong mask must not
    cut a piece off the limb. hand = (wrist, unit direction to the fingertips, share): beyond the wrist the rule is
    `share` instead (see HAND_SHARE). Returns (occupancy float32 (nx, ny, nz), origin, voxel, the views used)."""
    miss_share = MISS_SHARE if miss_share is None else miss_share
    views = _views(model_dir, mask_dir, grow_px)
    if keep is not None:
        views = [v for v, k in zip(views, keep) if k]
    slits = None
    agreement = None
    if points is not None and len(points):
        P = np.asarray(points, np.float32)
        P = P[np.random.default_rng(0).choice(len(P), min(len(P), 30000), replace=False)]
        agree = mask_agreement(views, P)
        bad = agree < min_agreement
        if bad.sum() < len(views) - 8:
            views = [v for v, b in zip(views, bad) if not b]
        agreement = float(np.median(agree))
        log(f'[hull] {bad.sum()} of {len(agree)} masks disagree with the stereo surface (left out); '
            f'median agreement {np.median(agree) * 100:.1f}%')
    if hand is not None and SLIT_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLIT_PX, SLIT_PX))
        slits = []
        for _, _, _, m in views:
            mm = m.astype(np.uint8)
            s_ = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, k).astype(bool) & ~m
            slits.append(cv2.erode(s_.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))  # its core only
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 1, 2)
    xs = (lo[0] + voxel * np.arange(n[0])).astype(np.float32)
    ys = (lo[1] + voxel * np.arange(n[1])).astype(np.float32)
    zs = (lo[2] + voxel * np.arange(n[2])).astype(np.float32)
    occ = np.zeros(n, np.float32)
    gy, gz = np.meshgrid(ys, zs, indexing='ij')
    for i, x in enumerate(xs):  # one slab at a time keeps memory small
        X = np.stack([np.full(gy.size, x, np.float32), gy.ravel(), gz.ravel()], 1)
        seen = np.zeros(len(X), np.int16)
        miss = np.zeros(len(X), np.int16)
        slit_hit = np.zeros(len(X), bool)
        for vi_, (R, t, cam, m) in enumerate(views):
            Xc = X @ R.T + t
            front = Xc[:, 2] > 1.0
            u, v = _project(cam, np.where(front[:, None], Xc, 1.0))
            ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
            inside = front & (ui >= 0) & (vi >= 0) & (ui < m.shape[1]) & (vi < m.shape[0])
            seen += inside
            hit = np.zeros(len(X), bool)
            hit[inside] = m[vi[inside], ui[inside]]
            miss += inside & ~hit
            if slits is not None:
                sh = np.zeros(len(X), bool)
                sh[inside] = slits[vi_][vi[inside], ui[inside]]
                slit_hit |= sh
        share = miss_share
        if hand is not None:
            share = np.where((X - hand[0]) @ hand[1] > 10.0, hand[2], miss_share)
        keep_v = (seen >= min_views) & (miss <= np.maximum(MISS_MIN, share * seen))
        if slits is not None and hand is not None:
            # a narrow slit of background inside the hand's outline (the gap between two fingers, seen straight
            # through by one or two photos only): direct evidence, one photo is enough - beyond the wrist only
            keep_v &= ~(slit_hit & ((X - hand[0]) @ hand[1] > 10.0))
        occ[i] = keep_v.reshape(n[1], n[2])
    log(f'[hull] {len(views)} outlines carved a {n[0]}x{n[1]}x{n[2]} grid at {voxel:g} mm: '
        f'{occ.sum() * voxel ** 3 / 1000:.0f} ml inside')
    return occ, lo, voxel, views, agreement


def surface(occ, origin, voxel, smooth=None):
    """Closed triangle mesh of the largest solid in the occupancy grid (marching cubes on a lightly smoothed grid:
    sub-voxel surface instead of voxel steps)."""
    import trimesh
    from scipy import ndimage
    from skimage import measure
    smooth = SMOOTH if smooth is None else smooth
    lab, k = ndimage.label(occ > 0.5)
    if k == 0:
        return None
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    solid = ndimage.binary_fill_holes(lab == big).astype(np.float32)
    f = ndimage.gaussian_filter(np.pad(solid, 2), smooth) if smooth else np.pad(solid, 2)
    v, faces, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 2) * voxel + np.asarray(origin)
    mesh = trimesh.Trimesh(v, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()
    if CREST_DEG > 0:
        mesh = round_crests(mesh, CREST_DEG)
    return mesh


def fill_funnels(occ, origin, vx, hand, share, reach=14.0, near=5.0, log=print):
    """Funnel pits at the ends of finger gaps filled (1 Nov, user: a pit between index and middle finger, already in
    the outline model, open toward the gap so neither opening nor small-pocket filling reaches it). For every empty
    voxel near the surface beyond the wrist: the share of 26 directions in which a straight line leaves the hand within
    `reach` mm without touching it. A gap between fingers is open along itself and to the front and back; a funnel pit
    only through its small mouth: voxels open in less than `share` of the directions are filled, repeatedly, from the
    bottom of the pit up."""
    from scipy import ndimage
    solid = occ > 0.5
    dirs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)],
                    float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    nsteps = int(reach / vx)
    shape = np.array(solid.shape)
    total = 0
    for _ in range(4):
        dist = ndimage.distance_transform_edt(~solid) * vx
        cand = (~solid) & (dist <= near)
        idx = np.argwhere(cand)
        if not len(idx):
            break
        P = idx.astype(np.float32) * vx + np.asarray(origin, np.float32)
        keep = (P - hand[0]) @ hand[1] > 10.0
        idx = idx[keep]
        if not len(idx):
            break
        free = np.zeros(len(idx), np.int16)
        for d in dirs:
            blocked = np.zeros(len(idx), bool)
            for st in range(1, nsteps + 1):
                q = np.round(idx + d * st).astype(np.int64)
                ins = ((q >= 0) & (q < shape)).all(1)
                hit = np.zeros(len(idx), bool)
                hit[ins] = solid[q[ins, 0], q[ins, 1], q[ins, 2]]
                blocked |= hit
            free += ~blocked
        fill = idx[free < share * len(dirs)]
        if not len(fill):
            break
        solid[fill[:, 0], fill[:, 1], fill[:, 2]] = True
        total += len(fill)
    log(f'[hull] {total * vx ** 3:.0f} mm3 of funnel pits filled beyond the wrist')
    return solid.astype(np.float32)


_FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def separate_fingers(occ, origin, voxel, zone, model_dir, web_mm=None, margin=0.6, log=print):
    """Neighbouring fingers made separate where the photos show they are. The outlines only carve along the few
    lines of sight that pass straight through the gap between two fingers, so a thin wall stays between their round
    sides (1 Nov, user: from an angle the index-middle gap looked closed though a photo shows it open down to the
    knuckles). Each finger's axis comes from its joints found in the photos (triangulated, the model's frame), re-centred
    on the model; its radius is measured on the model where that finger is already free (between its middle and end
    joints). Material between two neighbouring fingers farther than its radius from both axes is removed, from
    `web_mm` beyond the knuckle onward. Only measured quantities: a finger the photos do not show (a patient without
    it) has no joints and nothing is done for it."""
    import json
    from scipy.spatial import cKDTree
    from .handarm import landmarks_in_model
    web_mm = WEB_MM if web_mm is None else web_mm
    lmf = Path(model_dir).parent.parent / 'hand_landmarks.json'
    if not lmf.exists():
        return occ
    L = landmarks_in_model(model_dir, json.loads(lmf.read_text()))
    idx = np.argwhere(occ & zone)
    if len(idx) < 1000:
        return occ
    P = idx.astype(np.float64) * voxel + origin
    tree = cKDTree(P)
    axes = {}
    for fi, f in enumerate(_FINGERS):
        if not all(k in L for k in f):
            continue
        J = [np.asarray(L[k], float) for k in f]
        pts, arc = [], []
        a0 = 0.0
        for a, b in zip(J[:-1], J[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.5))
            for t in np.linspace(0, 1, n, endpoint=False):
                pts.append(a + t * (b - a)); arc.append(a0 + t * np.linalg.norm(b - a))
            a0 += np.linalg.norm(b - a)
        pts, arc = np.array(pts), np.array(arc)
        pip = np.linalg.norm(J[1] - J[0]); dip = pip + np.linalg.norm(J[2] - J[1])
        # re-centre on the model between the middle and end joints (the finger is free there); radius there too
        cen, rad = pts.copy(), []
        for i in np.where((arc >= pip) & (arc <= dip))[0]:
            t_ = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]); t_ /= np.linalg.norm(t_) + 1e-12
            near = tree.query_ball_point(pts[i], 12.0)
            if len(near) < 20:
                continue
            Q = P[near]
            sl_ = np.abs((Q - pts[i]) @ t_) < voxel
            if sl_.sum() < 10:
                continue
            c = Q[sl_].mean(0)
            cen[i] = c
            rq = Q[sl_] - c
            rq -= np.outer(rq @ t_, t_)
            rad.append(np.percentile(np.linalg.norm(rq, axis=1), 95))
        if len(rad) < 5:
            continue
        mid = (arc >= pip) & (arc <= dip)
        off = (cen[mid] - pts[mid]).mean(0)
        cen[~mid] = pts[~mid] + off  # the axis shifted like its re-centred middle part
        axes[fi] = (cen, arc, float(np.median(rad)))
    if len(axes) < 2:
        return occ
    D = np.full((len(P), 4), np.inf)
    A = np.zeros((len(P), 4))
    for fi, (cen, arc, r) in axes.items():
        d, j = cKDTree(cen).query(P)
        D[:, fi], A[:, fi] = d, arc[j]
    order = np.argsort(D, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    d1, d2 = D[np.arange(len(P)), i1], D[np.arange(len(P)), i2]
    rr = np.array([axes[k][2] if k in axes else 0 for k in range(4)])
    neighbours = np.abs(i1 - i2) == 1
    beyond = A[np.arange(len(P)), i1] >= web_mm
    between = d1 + d2 < rr[i1] + rr[i2] + 8.0
    cut = neighbours & beyond & between & (d1 > rr[i1] + margin) & (d2 > rr[i2] + margin) & np.isfinite(d2)
    out = occ.copy()
    ci = idx[cut]
    out[ci[:, 0], ci[:, 1], ci[:, 2]] = False
    log(f'[hull] fingers separated: radii {", ".join(f"{axes[k][2]:.1f}" for k in sorted(axes))} mm, '
        f'{cut.sum() * voxel ** 3:.0f} mm3 between them removed')
    return out


def surface_mixed(occ, origin, voxel, zone, smooth_out, smooth_in):
    """surface() with less smoothing inside `zone` (voxels): the hand keeps its narrow finger gaps, the forearm stays
    as smooth as before."""
    import trimesh
    from scipy import ndimage
    from skimage import measure
    lab, k = ndimage.label(occ)
    if k == 0:
        return None
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    solid = np.pad(ndimage.binary_fill_holes(lab == big).astype(np.float32), 2)
    z = ndimage.gaussian_filter(np.pad(zone, 2).astype(np.float32), smooth_out)  # soft seam
    f = (1 - z) * ndimage.gaussian_filter(solid, smooth_out) + z * ndimage.gaussian_filter(solid, smooth_in)
    v, faces, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 2) * voxel + np.asarray(origin)
    mesh = trimesh.Trimesh(v, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()
    if CREST_DEG > 0:
        mesh = round_crests(mesh, CREST_DEG)
    return mesh


def round_crests(mesh, deg, rings=3, iters=30):
    """Sharp convex crests rounded off. Where outlines from different directions meet in a hollow (the thumb web, the
    base of a finger gap) the carved solid has a ridge along their intersection - a limb has none (1 Nov: a flap the
    user saw in the thumb web of a printed hand, already in the outline model). Vertices along edges bending more than
    `deg` degrees convexly, and a few rings around them, are smoothed (Laplacian, only those vertices); rounding only
    removes material, and the outline model bounds the limb from outside, so it moves toward the real surface."""
    import scipy.sparse as sp
    import trimesh
    ang = mesh.face_adjacency_angles
    convex = mesh.face_adjacency_convex
    edges = mesh.face_adjacency_edges[(ang > np.radians(deg)) & convex]
    if len(edges) == 0:
        return mesh
    n = len(mesh.vertices)
    sel = np.zeros(n, bool)
    sel[edges.ravel()] = True
    E = mesh.edges_unique
    A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    for _ in range(rings):
        sel = sel | (A @ sel.astype(float) > 0)
    deg_ = np.asarray(A.sum(1)).ravel()
    V = mesh.vertices.copy()
    for _ in range(iters):
        avg = (A @ V) / np.maximum(deg_, 1)[:, None]
        V[sel] = 0.5 * V[sel] + 0.5 * avg[sel]
    return trimesh.Trimesh(V, mesh.faces, process=False)


def refine(mesh, points, max_in=12.0, radius=2.5, min_pts=6, spread=1.5, min_depth=3.0, log=print):
    """Pull the hull into hollows the outlines cannot see (cupped palm, between the fingers): each vertex moves
    inward along its normal onto the stereo surface where the stereo points around it agree (median depth, spread
    within `spread` mm). The hull is never pushed outward (it already bounds the limb) and stereo that disagrees
    with itself (lumps, holes on smooth surfaces) is ignored, and so is anything shallower than `min_depth`: on the
    convex limb the hull is the better surface (on the printed test model it is +0.6 mm, stereo -0.9 mm), so only
    real hollows the outlines cannot see are taken from stereo. The displacement is then smoothed over the mesh."""
    import scipy.sparse as sp
    import trimesh
    from scipy.spatial import cKDTree
    P = np.asarray(points, float)
    V, N = mesh.vertices, mesh.vertex_normals
    d0 = np.zeros(len(V))
    ok = np.zeros(len(V), bool)
    if len(P):
        tree = cKDTree(P)
        near = tree.query_ball_point(V - N * (max_in / 2), np.hypot(max_in / 2, radius))
        for k, idx in enumerate(near):
            if len(idx) < min_pts:
                continue
            q = P[idx] - V[k]
            depth = -(q @ N[k])
            lateral = np.linalg.norm(q + np.outer(depth, N[k]), axis=1)
            sel = (lateral < radius) & (depth > -2.0) & (depth < max_in)
            if sel.sum() < min_pts:
                continue
            dd = depth[sel]
            # the outermost consistent layer: stereo points are on the visible surface, the hull is outside it
            med = np.median(dd)
            if np.median(np.abs(dd - med)) > spread:
                continue
            if med >= min_depth:
                d0[k], ok[k] = med, True
    # smooth: stereo-supported vertices keep their depth (weight 1), the rest stay on the hull (weak weight),
    # neighbours pull each other together (no steps at the edge of a supported patch)
    e = mesh.edges_unique
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V),) * 2).tocsr()
    deg = np.asarray(A.sum(1)).ravel() + 1e-9
    c = np.where(ok, 1.0, 0.2)
    d = d0.copy()
    for _ in range(40):
        d = (c * d0 + (A @ d) / deg) / (c + 1.0)
    out = trimesh.Trimesh(V - N * d[:, None], mesh.faces, process=False)
    log(f'[hull] stereo supports {ok.mean() * 100:.0f}% of the hull surface; moved inward median '
        f'{np.median(d0[ok]) if ok.any() else 0:.1f} mm, max {d.max():.1f} mm')
    return out


def silhouette_support(mesh, views, edge_px=3.0):
    """Per vertex: in how many photos it lies on the limb's outline (within `edge_px` of the mask edge). Only there
    does an outline pin the hull to the real surface; elsewhere (an arm's underside on a table, any side no photo
    saw edge-on) the hull is just the boundary of what was never seen, and can be far too big."""
    V = mesh.vertices.astype(np.float32)
    count = np.zeros(len(V), np.int16)
    for R, t, cam, m in views:
        edge = (m.astype(np.uint8) - cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))) > 0
        dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
        Xc = V @ R.T + t
        front = Xc[:, 2] > 1.0
        u, v = _project(cam, np.where(front[:, None], Xc, 1.0))
        ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        ins = front & (ui >= 0) & (vi >= 0) & (ui < m.shape[1]) & (vi < m.shape[0])
        on = np.zeros(len(V), bool)
        on[ins] = dist[vi[ins], ui[ins]] <= edge_px
        count += on
    return count


def blend(mesh, support, stereo_closed, min_support=2, min_region_mm2=400.0, log=print):
    """Keep the hull where outlines pin it; elsewhere move it inward onto the (closed) stereo surface; smooth the
    displacement so the two meet without a step."""
    import scipy.sparse as sp
    import trimesh
    V, N = mesh.vertices, mesh.vertex_normals
    pinned = support >= min_support
    # only large unpinned regions (an arm's underside on a table, cm across) follow stereo; small gaps between pinned
    # outlines (between fingers, in a crease) keep the hull: stereo there is the hole-ridden part that made dents
    from scipy.sparse.csgraph import connected_components
    free_e = mesh.edges_unique[~pinned[mesh.edges_unique].any(1)]
    G = sp.coo_matrix((np.ones(len(free_e)), (free_e[:, 0], free_e[:, 1])), shape=(len(V),) * 2)
    _, lab = connected_components(G, directed=False)
    va = np.zeros(len(V))
    np.add.at(va, mesh.faces.ravel(), np.repeat(mesh.area_faces / 3, 3))
    region = np.bincount(lab[~pinned], weights=va[~pinned], minlength=lab.max() + 1)
    small = ~pinned & (region[lab] < min_region_mm2)
    pinned = pinned | small
    d0 = np.zeros(len(V))
    free = np.where(~pinned)[0]
    if len(free):
        cp, _, _ = trimesh.proximity.closest_point(stereo_closed, V[free])
        d0[free] = np.clip(((V[free] - cp) * N[free]).sum(1), 0.0, None)  # inward only: the hull bounds the limb
    e = mesh.edges_unique
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V),) * 2).tocsr()
    deg = np.asarray(A.sum(1)).ravel() + 1e-9
    d = d0.copy()
    for _ in range(30):
        d = (d0 + (A @ d) / deg) / 2.0
    out = trimesh.Trimesh(V - N * d[:, None], mesh.faces, process=False)
    log(f'[hull] outlines pin {pinned.mean() * 100:.0f}% of the surface; the rest follows stereo '
        f'(moved inward median {np.median(d0[free]) if len(free) else 0:.1f} mm, max {d.max():.1f} mm)')
    return out


def near_stereo(stereo, occ, origin, voxel, margin):
    """The part of the carved solid near the stereo limb: within `margin` mm of the stereo surface (an arm's inside
    is within its radius, < 45 mm, of its skin). Outlines can keep parts the stereo limb does not have (a neck or
    shoulder the skin mask included in a few photos, seen by too few photos to be carved away): those are not the
    limb. Distance only, no filling: the stereo surface has holes, so it does not enclose anything reliably."""
    from scipy import ndimage
    shape = occ.shape
    n = int(min(4e6, stereo.area / (0.25 * voxel * voxel)))
    pts, _ = trimesh_sample(stereo, n)
    ijk = np.round((pts - origin) / voxel).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
    shell = np.zeros(shape, bool)
    shell[tuple(ijk[ok].T)] = True
    return ((occ > 0.5) & (ndimage.distance_transform_edt(~shell) * voxel <= margin)).astype(np.float32)


def trimesh_sample(mesh, n):
    import trimesh
    return trimesh.sample.sample_surface(mesh, n, seed=0)[:2]


def mask_consistency(occ, origin, voxel, views, log=print):
    """Per photo, how its cut-out disagrees with the solid all photos agree on: 'bite' = share of that solid's
    footprint the mask leaves out (a cut-out biting into a finger or the palm: a dark or backlit photo, a skin-coloured
    background), 'spill' = share of the mask outside the footprint (the cut-out ran onto the stand or the room).
    Per-view (n, 2) array."""
    idx = np.argwhere(occ > 0.5)
    if len(idx) > 400000:
        idx = idx[np.random.default_rng(0).choice(len(idx), 400000, replace=False)]
    P = (np.asarray(origin, np.float32) + idx.astype(np.float32) * voxel)
    out = []
    for R, t, cam, m in views:
        Xc = P @ R.T + t
        front = Xc[:, 2] > 1.0
        u, v = _project(cam, np.where(front[:, None], Xc, 1.0))
        ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        ins = front & (ui >= 0) & (vi >= 0) & (ui < m.shape[1]) & (vi < m.shape[0])
        f = np.zeros(m.shape, np.uint8)
        f[vi[ins], ui[ins]] = 1
        px = max(2, int(np.ceil(voxel * float(np.asarray(cam.params)[0]) / max(1.0, np.median(Xc[front, 2])))))
        f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))
        core = cv2.erode(f, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        wide = cv2.dilate(f, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
        bite = float((core & ~m).sum()) / max(1, core.sum())
        spill = float((m & ~wide).sum()) / max(1, m.sum())
        out.append((bite, spill))
    return np.array(out)


def limb_hull(model_dir, mask_dir, stereo_surface, stereo_closed, voxel=None, log=print):
    """The limb as a closed surface: its outlines in every photo (texture-free) wherever an outline saw that part
    of the surface edge-on, the closed stereo surface elsewhere. None when outlines and stereo disagree (bad masks
    or cameras): the caller keeps the stereo surface."""
    from scipy.spatial import cKDTree
    voxel = VOXEL if voxel is None else voxel
    lo, hi = stereo_surface.bounds[0] - 20, stereo_surface.bounds[1] + 20
    # first a lenient carve (a part stays unless 30 % of the photos say otherwise) as the consensus solid; photos
    # whose cut-out bites into it or spills far outside are left out of the real carve (one bad cut-out per few
    # photos took a piece out of the little finger)
    occ0, origin0, vx0, views0, _ = carve(model_dir, mask_dir, lo, hi, voxel=2.0, grow_px=0, miss_share=0.3,
                                          log=lambda *a: None)
    keep = None
    # (tracked cut-outs are consistent from photo to photo: the check, relative to the scan's own median, then only
    # removed good photos from the sides the photos see least - 31 Oct, 4 of 5 scans better without it)
    tracked = (Path(mask_dir).parent / 'masks_tracked').exists()
    if CHECK and len(views0) >= 12 and occ0.sum() > 0:
        bs = mask_consistency(occ0, origin0, vx0, views0)
        b, sp = bs[:, 0], bs[:, 1]
        bite_bad = b > max(0.01, 3 * np.median(b))
        # a bite only counts against a photo when photos from nearly the same direction (within 25 degrees) do not
        # bite: the only photo from one side (thumb side on a short walk) is outvoted by the lenient consensus and
        # removing it loses that side (31 Oct: three good thumb-side photos dropped, fingers merged)
        cen = np.asarray(origin0, float) + np.argwhere(occ0 > 0.5).mean(0) * vx0
        dirs = np.array([-(R.T @ t) - cen for R, t, _, _ in views0])
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        near = (dirs @ dirs.T) > np.cos(np.radians(25))
        np.fill_diagonal(near, False)
        backed = np.array([(near[i] & ~bite_bad).sum() >= 2 for i in range(len(bs))])
        bad = (bite_bad & backed) | (sp > max(0.03, 3 * np.median(sp)))
        if tracked:  # tracked cut-outs: only spills (bites there were good photos of rarely seen sides)
            bad = sp > max(0.03, 3 * np.median(sp))
        # direct evidence: the hand landmarker found the hand in the photo and the cut-out misses part of it
        import json
        cf = Path(mask_dir).parent / 'hand_cover.json'
        if os.environ.get('LIMBSCAN_HAND_COVER', '0') == '1' and cf.exists():  # off: see segment.FILL
            import pycolmap
            cov = json.loads(cf.read_text())
            names0 = [im.name for im in pycolmap.Reconstruction(str(model_dir)).images.values()
                      if (Path(mask_dir) / (Path(im.name).stem + '.png')).exists()]
            miss = np.array([(cov.get(n) is not None and cov[n] < 0.85) for n in names0])
            if len(miss) == len(bad):
                bad |= miss
        if bad.sum() <= len(bs) - 12:
            keep = ~bad
            # the same photos are left out of the GPU shape step (irprep reads this list)
            import json
            import pycolmap
            rec = pycolmap.Reconstruction(str(model_dir))
            names = [im.name for im in rec.images.values()
                     if (Path(mask_dir) / (Path(im.name).stem + '.png')).exists()]
            (Path(mask_dir).parent / 'bad_masks.json').write_text(json.dumps([n for n, k in zip(names, keep) if not k]))
        log(f'[hull] {bad.sum()} of {len(bs)} cut-outs disagree with the others (bite median {np.median(b) * 100:.1f}%, '
            f'spill median {np.median(sp) * 100:.1f}%): left out')
    hand = None
    if HAND_SHARE > 0:
        # the gaps between fingers that nearly touch are seen through by only a few photos: there, a few photos say
        # so. Only beyond the wrist: on the forearm (hair, a limb that sways a little) a few photos must not carve
        # (31 Oct: the same rule over the whole limb cut real forearm away, 0.8 -> 6-16 mm vs the Structure scan)
        lmf = Path(model_dir).parent.parent / 'hand_landmarks.json'
        if lmf.exists():
            import json
            from .handarm import arm_in_model
            try:
                arm = arm_in_model(model_dir, json.loads(lmf.read_text()), log=lambda *a: None)
            except Exception:  # noqa: BLE001
                arm = None
            if arm is not None:
                W = np.asarray(arm['wrist'], float)
                u = np.asarray(arm['b'], float) - W
                hand = (W.astype(np.float32), (u / np.linalg.norm(u)).astype(np.float32), HAND_SHARE)
    occ, origin, vx, views, agreement = carve(model_dir, mask_dir, lo, hi, voxel=voxel, grow_px=0, points=stereo_surface.vertices,
                                   log=log, keep=keep, hand=hand)
    if len(views) < 12:
        log(f'[hull] only {len(views)} usable outlines: keeping the stereo surface')
        return None
    occ = near_stereo(stereo_surface, occ, origin, vx, margin=45.0)
    if OPEN_MM > 0 and hand is not None:
        # thin bridges and folds left where the cuts from the back and from the palm side of a finger gap did not quite
        # meet (a pit with a bridge between index and middle finger, a fold in the thumb web): anything thinner than
        # ~2 x OPEN_MM removed - beyond the wrist only; fingers and the thumb web are far thicker
        from scipy import ndimage
        r = max(1, int(round(OPEN_MM / vx)))
        zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
        ball = (xx ** 2 + yy ** 2 + zz ** 2) <= r * r
        opened = ndimage.binary_opening(occ > 0.5, structure=ball)
        ii = np.indices(occ.shape).reshape(3, -1).T.astype(np.float32) * vx + np.asarray(origin, np.float32)
        beyond = ((ii - hand[0]) @ hand[1] > 10.0).reshape(occ.shape)
        n0 = int((occ > 0.5).sum())
        occ = np.where(beyond, opened, occ > 0.5).astype(np.float32)
        log(f'[hull] thin bridges beyond the wrist removed: {(n0 - int(occ.sum())) * vx ** 3 / 1000:.1f} ml')
        if PIT_MM > 0:
            # blind pits at the bottom of finger gaps (a few photos saw a little way into the web from the back and the
            # loose carving there dug a hole; 1 Nov, user: a pit between index and middle finger): the dents a ball of
            # radius CLOSE_MM cannot enter, kept only where they form a small compact pocket (at most PIT_MM across);
            # the gaps between fingers are long slots and are never filled
            rc = max(1, int(round(CLOSE_MM / vx)))
            zz, yy, xx = np.mgrid[-rc:rc + 1, -rc:rc + 1, -rc:rc + 1]
            ballc = (xx ** 2 + yy ** 2 + zz ** 2) <= rc * rc
            solid = occ > 0.5
            closed = ndimage.binary_closing(np.pad(solid, rc + 1), structure=ballc)[rc + 1:-rc - 1, rc + 1:-rc - 1, rc + 1:-rc - 1]
            add = closed & ~solid & beyond
            lab, k = ndimage.label(add)
            filled = 0
            if k:
                sl = ndimage.find_objects(lab)
                for i, sli in enumerate(sl, 1):
                    ext = max(sx.stop - sx.start for sx in sli) * vx
                    if ext <= PIT_MM:
                        m_ = lab[sli] == i
                        solid[sli][m_] = True
                        filled += int(m_.sum())
            occ = solid.astype(np.float32)
            log(f'[hull] {filled * vx ** 3:.0f} mm3 of small pits filled beyond the wrist')
    if ESCAPE > 0 and hand is not None:
        occ = fill_funnels(occ, origin, vx, hand, ESCAPE, log=log)
    if FINE_MM > 0 and hand is not None and vx / FINE_MM >= 1.5:
        # the hand on a finer grid: a gap between two fingers ~1 mm wide at its base is thinner than one 1 mm
        # cell and the smoothing closed it (1 Nov: the index-middle slit a photo shows to the knuckles stopped a third
        # of the way down). Carved again at FINE_MM beyond the wrist (the coarse result bounds it: fine carving only
        # removes), the whole grid upsampled, and smoothed less on the hand.
        r = int(round(vx / FINE_MM))
        fv = vx / r
        up = np.repeat(np.repeat(np.repeat(occ > 0.5, r, 0), r, 1), r, 2)
        P = np.argwhere(occ > 0.5).astype(np.float32) * vx + np.asarray(origin, np.float32)
        hb = P[(P - hand[0]) @ hand[1] > FINE_FROM - 5.0]
        if len(hb):
            lo_h, hi_h = hb.min(0) - 3.0, hb.max(0) + 3.0
            occ_h, org_h, _, _, _ = carve(model_dir, mask_dir, lo_h, hi_h, voxel=fv, grow_px=0, log=log, keep=keep,
                                          hand=hand)
            i0 = np.round((org_h - np.asarray(origin)) / fv).astype(int)
            sl = tuple(slice(max(0, a), min(up.shape[d], a + occ_h.shape[d])) for d, a in enumerate(i0))
            sub = tuple(slice(sl[d].start - i0[d], sl[d].stop - i0[d]) for d in range(3))
            G = np.indices(occ_h.shape).reshape(3, -1).T.astype(np.float32) * fv + org_h.astype(np.float32)
            beyond_h = ((G - hand[0]) @ hand[1] > FINE_FROM).reshape(occ_h.shape)  # the fingers and their bases
            fine = np.where(beyond_h, occ_h > 0.5, True)[sub]
            up[sl] &= fine
            hand_zone = np.zeros(up.shape, bool)
            hand_zone[sl] = beyond_h[sub]
            if SEPARATE:
                up = separate_fingers(up, np.asarray(origin, float), fv, hand_zone, model_dir, log=log)
            mesh = surface_mixed(up, origin, fv, hand_zone, SMOOTH * r, FINE_SMOOTH)
            if mesh is not None and len(mesh.faces) > FINE_FACES:
                # the same density as before where it matters for the GPU step (it moves a 0.5 mm mesh 4x slower and
                # left the palm full): quadric decimation keeps the narrow gaps, the forearm gets coarse triangles
                import open3d as o3d
                import trimesh
                om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
                om = om.simplify_quadric_decimation(FINE_FACES)
                om.remove_non_manifold_edges()
                om.remove_degenerate_triangles()
                om.remove_unreferenced_vertices()
                dec = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=True)
                if not dec.is_watertight:
                    trimesh.repair.fill_holes(dec)
                if dec.is_watertight:
                    mesh = dec
                else:
                    # (decimation can leave a hole; a hole in the outline model makes the GPU step's result
                    # non-watertight and rejected - 1 Nov, a real hand showed the raw outline with a hole in the palm)
                    from skimage import measure
                    from scipy import ndimage
                    lab, _ = ndimage.label(up)
                    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
                    solid = np.pad(ndimage.binary_fill_holes(lab == big).astype(np.float32), 2)
                    z = ndimage.gaussian_filter(np.pad(hand_zone, 2).astype(np.float32), SMOOTH * r)
                    f = (1 - z) * ndimage.gaussian_filter(solid, SMOOTH * r) + z * ndimage.gaussian_filter(solid, FINE_SMOOTH)
                    v, fc, _, _ = measure.marching_cubes(f, 0.5, step_size=2)
                    mesh = trimesh.Trimesh((v - 2) * fv + np.asarray(origin), fc[:, ::-1], process=True)
                    if mesh.volume < 0:
                        mesh.invert()
                    if CREST_DEG > 0:
                        mesh = round_crests(mesh, CREST_DEG)
                    log(f'[hull] decimation left holes: fine surface sampled at {2 * fv:g} mm instead '
                        f'(watertight {mesh.is_watertight})')
        else:
            mesh = surface(occ, origin, vx)
    else:
        mesh = surface(occ, origin, vx)
    if mesh is None:
        return None
    if BLEND:
        mesh = blend(mesh, silhouette_support(mesh, views), stereo_closed, log=log)
    # agreement check: the stereo surface lies on or just inside the result nearly everywhere
    S = stereo_surface.vertices[::max(1, len(stereo_surface.vertices) // 50000)]
    d, _ = cKDTree(mesh.vertices).query(S)
    near = (d < 4.0).mean()
    log(f'[hull] {mesh.volume / 1000:.0f} ml closed surface; {near * 100:.0f}% of the stereo surface within 4 mm of it')
    # outlines that agree with the stereo limb photo by photo (nearly all stereo points inside every mask) make a
    # trustworthy hull even when the stereo surface itself is noisy (pits, lumps: a moving torch, plain surfaces)
    trusted = agreement is not None and agreement >= 0.95 and len(views) >= 24
    if near < (0.5 if trusted else 0.8):
        log('[hull] outlines and stereo disagree: keeping the stereo surface')
        return None
    return mesh
