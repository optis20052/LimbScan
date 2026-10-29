"""Metric scale + table-frame recovery from ArUco scale sheets seen in a COLMAP reconstruction.

Photogrammetry is scale-free; the printed marker sheets make it metric. We triangulate every
marker corner from all views where it was detected (using the SfM poses + calibrated intrinsics),
then estimate scale two ways:
  * sheet fit: Umeyama similarity between the triangulated corners of a sheet and the known
    printed layout (long baselines -> ~0.1% precision)
  * edge fit: median of 40 mm marker edges (independent check)
"""
import numpy as np
import cv2
from . import markers


def _detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 4
    p.cornerRefinementMinAccuracy = 0.01
    p.cornerRefinementMaxIterations = 60
    return cv2.aruco.ArucoDetector(markers.dictionary(), p)


def detect(image_path):
    return detect_gray(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE))


def detect_gray(g):
    corners, ids, _ = _detector().detectMarkers(g)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def _triangulate(obs):
    """obs: list of (P 3x4 normalized camera matrix, xy normalized). Linear DLT + Gauss-Newton."""
    A = []
    for P, (x, y) in obs:
        A.append(x * P[2] - P[0]); A.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1, :3] / vt[-1, 3]
    for _ in range(10):
        J, r = [], []
        for P, (x, y) in obs:
            h = P[:3, :3] @ X + P[:, 3]
            r += [h[0] / h[2] - x, h[1] / h[2] - y]
            J.append((P[0, :3] * h[2] - P[2, :3] * h[0]) / h[2] ** 2)
            J.append((P[1, :3] * h[2] - P[2, :3] * h[1]) / h[2] ** 2)
        J, r = np.asarray(J), np.asarray(r)
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        X = X + dx
        if np.linalg.norm(dx) < 1e-12:
            break
    return X, np.abs(r)


def umeyama(src, dst):
    """Similarity dst ~ s R src + t."""
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3); E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / (S ** 2).sum() * len(src)
    return s, R, md - s * R @ ms


def recover(rec, image_dir, log=print):
    """Return dict with scale (mm per SfM unit), sim3 (s,R,t) mapping SfM->table frame mm, stats."""
    obs = {}  # (mid, k) -> list of (P, xy_norm, focal)
    nimg = 0
    for img in rec.images.values():
        det = detect(f'{image_dir}/{img.name}')
        if not det:
            continue
        nimg += 1
        cam = rec.cameras[img.camera_id]
        T = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        P = T.matrix()
        f = cam.mean_focal_length()
        for mid, c in det.items():
            xy = cam.cam_from_img(c)
            for k in range(4):
                obs.setdefault((mid, k), []).append((P, xy[k], f))
    corners3d = {}
    errs = []
    for key, o in obs.items():
        if len(o) < 3:
            continue
        X, r = _triangulate([(P, xy) for P, xy, _ in o])
        rpx = (r.reshape(-1, 2) * np.array([[f] for *_, f in o])).max(1)
        keep = [oo for oo, e in zip(o, rpx) if e < 2.0]
        if len(keep) < 3:
            continue
        if len(keep) < len(o):
            X, r = _triangulate([(P, xy) for P, xy, _ in keep])
        corners3d[key] = (X, len(keep))
        errs.append(np.median(rpx))
    mids = sorted({m for m, _ in corners3d})
    log(f'[scale] markers seen in {nimg} images, {len(mids)} markers / {len(corners3d)} corners triangulated')
    if len(mids) < 2:
        raise RuntimeError('Not enough scale markers reconstructed. Make sure the printed sheets are visible in many photos.')
    # edge-based scale
    edges = []
    for m in mids:
        if all((m, k) in corners3d for k in range(4)):
            C = [corners3d[(m, k)][0] for k in range(4)]
            edges += [np.linalg.norm(C[k] - C[(k + 1) % 4]) for k in range(4)]
    s_edge = markers.MARKER_MM / np.median(edges) if edges else np.nan
    # sheet layout fits
    fits = []
    world_pts, table_pts = [], []
    for sheet, ids in enumerate(markers.SHEET_IDS):
        lay = {mid: (x, y) for mid, x, y in markers.sheet_layout(sheet)}
        src, dst = [], []
        for mid in ids:
            if mid not in lay:
                continue
            x, y = lay[mid]
            # corner order from ArUco: TL, TR, BR, BL ; page y grows downward
            pc = [(x, y), (x + markers.MARKER_MM, y), (x + markers.MARKER_MM, y + markers.MARKER_MM), (x, y + markers.MARKER_MM)]
            for k in range(4):
                if (mid, k) in corners3d:
                    src.append(corners3d[(mid, k)][0]); dst.append([pc[k][0], -pc[k][1], 0.0])
        if len(src) >= 8:
            src, dst = np.asarray(src), np.asarray(dst)
            s, R, t = umeyama(src, dst)
            res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
            fits.append({'sheet': sheet, 'scale': s, 'R': R, 't': t, 'n': len(src), 'rms_mm': float(np.sqrt((res ** 2).mean())),
                         'src': src})
    if fits:
        w = np.array([f['n'] / max(f['rms_mm'], 0.05) ** 2 for f in fits])
        s_sheet = float((w * np.array([f['scale'] for f in fits])).sum() / w.sum())
    else:
        s_sheet = np.nan
    scale = s_sheet if np.isfinite(s_sheet) else s_edge
    # table frame: plane through all corners, z up towards cameras, origin at corner centroid
    allc = np.array([v[0] for v in corners3d.values()])
    cen = allc.mean(0)
    _, _, vt = np.linalg.svd(allc - cen)
    n = vt[2]
    camc = np.array([(img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world).inverse().translation
                     for img in rec.images.values()])
    if ((camc - cen) @ n).mean() < 0:
        n = -n
    xax = vt[0] - (vt[0] @ n) * n; xax /= np.linalg.norm(xax)
    R = np.stack([xax, np.cross(n, xax), n])
    if np.linalg.det(R) < 0:
        R[1] *= -1
    t = -scale * R @ cen
    planarity = float(np.abs((allc - cen) @ n).max() * scale)
    stats = {
        'scale_mm_per_unit': float(scale),
        'scale_sheet': float(s_sheet) if np.isfinite(s_sheet) else None,
        'scale_edge': float(s_edge) if np.isfinite(s_edge) else None,
        'edge_vs_sheet_pct': float(100 * (s_edge / s_sheet - 1)) if fits and edges else None,
        'sheet_fits': [{'sheet': f['sheet'] + 1, 'corners': f['n'], 'rms_mm': f['rms_mm'], 'scale': float(f['scale'])} for f in fits],
        'markers': len(mids), 'corners': len(corners3d), 'images_with_markers': nimg,
        'median_reproj_px': float(np.median(errs)) if errs else None,
        'mat_planarity_mm': planarity,
        'edge_lengths_mm': (np.array(edges) * scale).round(3).tolist(),
    }
    log(f"[scale] sheet scale {stats['scale_sheet']}, edge scale {stats['scale_edge']}, diff {stats['edge_vs_sheet_pct']} %")
    return scale, R, t, stats
