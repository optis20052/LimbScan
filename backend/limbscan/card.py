"""Metric scale from an ISO/IEC 7810 ID-1 card (bank / ID / health-insurance card: 85.60 x 53.98 mm)
lying flat near the limb. No printed markers needed.

1. per image: find convex quadrilateral contours; refine each side with a line fit that ignores the
   rounded corners, intersect the lines -> sub-pixel virtual corners of the card outline.
2. across images: triangulate quad hypotheses from image pairs with the SfM poses, try all corner
   correspondences, keep 3D quads that are rectangles with the ID-1 aspect ratio, and vote: project
   each into every image and count views where a detected quad matches.
3. refine the winner with all its views (multi-view triangulation per corner) -> scale + table plane.
"""
import itertools

import cv2
import numpy as np

from .scale import _triangulate

CARD_W, CARD_H = 85.60, 53.98
ASPECT = CARD_W / CARD_H


def _fit_side(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([vx, vy]), np.array([x0, y0])


def _intersect(l1, l2):
    (d1, p1), (d2, p2) = l1, l2
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def _subpix_side(gx, gy, a0, a1, n_samples=48, reach=5.0):
    """Re-locate one card side at the gradient maximum along its normal (sub-pixel), return line."""
    d = a1 - a0; L = np.linalg.norm(d); d /= L
    nrm = np.array([-d[1], d[0]])
    offs = np.arange(-reach, reach + 1e-6, 0.25)
    pts = []
    for t in np.linspace(0.15, 0.85, n_samples):
        base = a0 + t * L * d
        q = base[None] + offs[:, None] * nrm[None]
        mx, my = q[:, 0].astype(np.float32).reshape(1, -1), q[:, 1].astype(np.float32).reshape(1, -1)
        g = cv2.remap(gx, mx, my, cv2.INTER_LINEAR).ravel() * nrm[0] + cv2.remap(gy, mx, my, cv2.INTER_LINEAR).ravel() * nrm[1]
        g = np.abs(g)
        k = int(np.argmax(g))
        if k == 0 or k == len(g) - 1 or g[k] < 4:
            continue
        den = g[k - 1] - 2 * g[k] + g[k + 1]
        sub = 0.5 * (g[k - 1] - g[k + 1]) / den if den != 0 else 0.0
        pts.append(base + (offs[k] + sub * 0.25) * nrm)
    if len(pts) < 12:
        return None
    return _fit_side(np.array(pts))


def detect_quads(gray, max_quads=12):
    """Candidate card outlines in one image: list of (4,2) float corner arrays (convex, ordered)."""
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = list(cnts)
    # second candidate source: regions clearly darker / lighter than their surroundings (a dark card on skin
    # or wood, a light card on a dark table). Robust where an edge merges with a shadow line.
    blk = int(max(31, (min(h, w) // 12) | 1))
    for inv in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, inv, blk, 12)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # fill logos / embossing
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        c2, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts += list(c2)
    out = []
    gxy = None
    amin, amax = 0.0015 * w * h, 0.25 * w * h
    for c in cnts:
        if len(c) < 60:
            continue
        a = cv2.contourArea(c)
        if not (amin < a < amax):
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        # rounded corners / a touching edge give 5-6 polygon vertices: simplify harder until 4 remain
        ap = None
        for eps in (0.02, 0.03, 0.045, 0.06, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) <= 4:
                break
        if len(ap) != 4 or not cv2.isContourConvex(ap):
            continue
        if cv2.contourArea(hull) / max(cv2.contourArea(ap), 1) < 0.90:
            continue
        q = ap.reshape(4, 2).astype(np.float64)
        # refine: fit each side to contour points in its middle 70% (skip rounded corners)
        pts = c.reshape(-1, 2).astype(np.float64)
        lines = []
        ok = True
        for k in range(4):
            a0, a1 = q[k], q[(k + 1) % 4]
            d = a1 - a0; L = np.linalg.norm(d); d /= L
            nrm = np.array([-d[1], d[0]])
            t = (pts - a0) @ d
            off = np.abs((pts - a0) @ nrm)
            sel = (t > 0.15 * L) & (t < 0.85 * L) & (off < max(3.0, 0.03 * L))
            if sel.sum() < 10:
                ok = False; break
            lines.append(_fit_side(pts[sel]))
        if not ok:
            continue
        cs = [_intersect(lines[k - 1], lines[k]) for k in range(4)]
        if any(x is None for x in cs):
            continue
        cs = np.array(cs)
        if np.abs(cs - q).max() > 0.08 * np.sqrt(a) + 6:
            continue
        # sub-pixel: move every side onto the true intensity edge (Canny/dilation bias ~1 px outward)
        if gxy is None:
            gf = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
            gxy = (cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3))
        sl = [_subpix_side(*gxy, cs[k], cs[(k + 1) % 4]) for k in range(4)]
        if any(x is None for x in sl):
            continue
        cs2 = [_intersect(sl[k - 1], sl[k]) for k in range(4)]
        if any(x is None for x in cs2) or np.abs(np.array(cs2) - cs).max() > 6:
            continue
        cs = np.array(cs2)
        # dedupe (inner/outer edge of the same outline)
        if any(np.abs(cs - o).max() < 6 for o in out):
            continue
        out.append(cs)
    out.sort(key=lambda q: -cv2.contourArea(q.astype(np.float32)))
    return out[:max_quads]


def plausible_card(q):
    """Card seen in perspective: side ratio roughly 1.2-2.4 and opposite sides similar (not a long window strip)."""
    s = [np.linalg.norm(q[(k + 1) % 4] - q[k]) for k in range(4)]
    a, b = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    r = max(a, b) / max(min(a, b), 1)
    return 1.2 < r < 2.4 and abs(s[0] - s[2]) / max(s[0], s[2]) < 0.35 and abs(s[1] - s[3]) / max(s[1], s[3]) < 0.35


def _rect_score(X):
    """How well 4 ordered 3D points form an ID-1 rectangle; returns (err, scale_units_per_mm)."""
    s = [np.linalg.norm(X[(k + 1) % 4] - X[k]) for k in range(4)]
    if min(s) <= 0:
        return 9, None
    long = (s[0] + s[2]) / 2 if s[0] + s[2] > s[1] + s[3] else (s[1] + s[3]) / 2
    short = (s[1] + s[3]) / 2 if s[0] + s[2] > s[1] + s[3] else (s[0] + s[2]) / 2
    asp = long / short
    opp = max(abs(s[0] - s[2]) / max(s[0], s[2]), abs(s[1] - s[3]) / max(s[1], s[3]))
    diag = abs(np.linalg.norm(X[2] - X[0]) - np.linalg.norm(X[3] - X[1])) / np.linalg.norm(X[2] - X[0])
    n = np.cross(X[1] - X[0], X[3] - X[0]); n /= np.linalg.norm(n) + 1e-12
    flat = abs((X[2] - X[0]) @ n) / long
    err = abs(asp / ASPECT - 1) + opp + diag + flat
    return err, (long / CARD_W + short / CARD_H) / 2


def recover_card(rec, image_dir, log=print):
    """Return (scale mm/unit, R, t, stats) like scale.recover, using an ID-1 card."""
    imgs = list(rec.images.values())
    data = []
    for im in imgs:
        g = cv2.imread(f'{image_dir}/{im.name}', cv2.IMREAD_GRAYSCALE)
        quads = detect_quads(g)
        cam = rec.cameras[im.camera_id]
        P = im.cam_from_world().matrix()
        qn = [cam.cam_from_img(q) for q in quads]
        C = im.cam_from_world().inverse().translation
        data.append({'im': im, 'cam': cam, 'P': P, 'C': C, 'quads': quads, 'qn': qn, 'f': cam.mean_focal_length()})
    nq = sum(len(d['quads']) for d in data)
    log(f'[card] {nq} quad candidates in {len(data)} images')
    # hypotheses from image pairs with a useful baseline
    centers = np.array([d['C'] for d in data])
    med = np.median(np.linalg.norm(centers - centers.mean(0), axis=1))
    hyps = []
    rng = np.random.default_rng(0)
    idx = [i for i, d in enumerate(data) if d['quads']]
    for i in idx:
        others = [j for j in idx if j != i and 0.15 * med < np.linalg.norm(centers[i] - centers[j]) < 1.2 * med]
        for j in rng.permutation(others)[:6]:
            for qa in data[i]['qn']:
                for qb in data[j]['qn']:
                    for shift, rev in itertools.product(range(4), (False, True)):
                        qb2 = qb[::-1] if rev else qb
                        qb2 = np.roll(qb2, shift, axis=0)
                        X = []
                        good = True
                        for k in range(4):
                            Xk, r = _triangulate([(data[i]['P'], qa[k]), (data[j]['P'], qb2[k])])
                            if r.max() * data[i]['f'] > 3.0:
                                good = False; break
                            X.append(Xk)
                        if not good:
                            continue
                        X = np.array(X)
                        # in front of both cameras
                        if any((d['P'][:3, :3] @ X.T + d['P'][:, 3:])[2].min() <= 0 for d in (data[i], data[j])):
                            continue
                        err, s = _rect_score(X)
                        if err < 0.12:
                            hyps.append((err, X))
    log(f'[card] {len(hyps)} rectangle hypotheses')
    if not hyps:
        raise RuntimeError('Could not find the reference card. Place a bank/ID card flat near the limb, fully visible in many photos.')

    def support(X, tol=4.0):
        obs = []
        for d in data:
            h = d['P'][:3, :3] @ X.T + d['P'][:, 3:]
            if (h[2] <= 0).any():
                continue
            xy = (h[:2] / h[2]).T
            px = d['cam'].img_from_cam(np.c_[xy, np.ones(len(xy))])
            best = None
            for qi, q in enumerate(d['quads']):
                for sh in range(4):
                    for rev in (False, True):
                        qq = np.roll(q[::-1] if rev else q, sh, axis=0)
                        e = np.linalg.norm(qq - px, axis=1).max()
                        if e < tol and (best is None or e < best[0]):
                            best = (e, d['cam'].cam_from_img(qq))
            if best:
                obs.append((d, best[1]))
        return obs

    hyps.sort(key=lambda h: h[0])
    best = None
    for err, X in hyps[:60]:
        ob = support(X)
        if best is None or len(ob) > len(best[1]):
            best = (X, ob)
    X, obs = best
    if len(obs) < 3:
        raise RuntimeError('Reference card seen in too few photos. Keep the card visible during the scan.')
    # refine each corner with all supporting views
    Xr, reps = [], []
    for k in range(4):
        Xk, r = _triangulate([(d['P'], xy[k]) for d, xy in obs])
        Xr.append(Xk); reps.append(np.median(r) * obs[0][0]['f'])
    Xr = np.array(Xr)
    err, s_units_per_mm = _rect_score(Xr)
    scale = 1.0 / s_units_per_mm
    sides = [np.linalg.norm(Xr[(k + 1) % 4] - Xr[k]) * scale for k in range(4)]
    # table frame from the card plane
    cen = Xr.mean(0)
    n = np.cross(Xr[1] - Xr[0], Xr[3] - Xr[0]); n /= np.linalg.norm(n)
    if ((centers - cen) @ n).mean() < 0:
        n = -n
    xax = Xr[1] - Xr[0]; xax -= (xax @ n) * n; xax /= np.linalg.norm(xax)
    R = np.stack([xax, np.cross(n, xax), n])
    t = -scale * R @ cen
    long = sorted(sides)[2:]; short = sorted(sides)[:2]
    stats = {
        'reference': 'ID-1 card 85.60 x 53.98 mm',
        'scale_mm_per_unit': float(scale), 'scale_sheet': None, 'scale_edge': None,
        'card_views': len(obs), 'card_sides_mm': [round(float(x), 2) for x in sides],
        'card_view_names': sorted(d['im'].name for d, _ in obs),
        'card_corners_sfm': Xr.tolist(),  # SfM frame; pipeline maps them to the metric frame
        'card_shape_error': float(err),
        # independent check: aspect ratio from the reconstruction vs ISO (scale-free)
        'edge_vs_sheet_pct': float(100 * ((np.mean(long) / np.mean(short)) / ASPECT - 1)),
        'sheet_fits': [], 'markers': 0, 'corners': 4, 'images_with_markers': len(obs),
        'median_reproj_px': float(np.median(reps)), 'mat_planarity_mm': None,
    }
    log(f"[card] scale {scale:.5f} mm/unit from {len(obs)} views, sides {stats['card_sides_mm']} mm, aspect check {stats['edge_vs_sheet_pct']:+.2f}%")
    return scale, R, t, stats



def recover_card_pnp(rec, image_dir, log=print, min_ratios=15, min_views=2):
    """Single-view fallback when the card was seen without enough parallax for triangulation (phone barely
    moved, or far away). Per photo: the card's metric pose from its 4 corners (planar PnP, known 85.60 x 53.98
    mm) gives the true depth of the card plane; SfM points seen on / right at the card give the same depth in
    model units. scale = mm / unit, robust median over all photos and points."""
    obj = np.array([[0, 0, 0], [CARD_W, 0, 0], [CARD_W, CARD_H, 0], [0, CARD_H, 0]], np.float64)
    ratios, per_view, used = [], [], []
    for im in rec.images.values():
        g = cv2.imread(f'{image_dir}/{im.name}', cv2.IMREAD_GRAYSCALE)
        quads = [q for q in detect_quads(g) if plausible_card(q)]
        if not quads:
            continue
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world().matrix()
        pts = []
        for p2d in im.points2D:
            if p2d.has_point3D():
                X = rec.points3D[p2d.point3D_id].xyz
                pts.append((p2d.xy, (T @ np.r_[X, 1.0])[2]))
        if not pts:
            continue
        xy = np.array([p[0] for p in pts]); zs = np.array([p[1] for p in pts])
        for q in quads:
            qn = cam.cam_from_img(q)  # normalized, undistorted image coordinates
            best = None
            for sh in range(4):  # corner order: long side first
                qq = np.roll(qn, sh, axis=0)
                if np.linalg.norm(qq[1] - qq[0]) < np.linalg.norm(qq[2] - qq[1]):
                    continue
                ok, rv, tv = cv2.solvePnP(obj, qq.astype(np.float64), np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
                if not ok:
                    continue
                pr, _ = cv2.projectPoints(obj, rv, tv, np.eye(3), None)
                e = np.abs(pr.reshape(4, 2) - qq).max() * cam.mean_focal_length()
                if best is None or e < best[0]:
                    best = (e, rv, tv)
            if best is None or best[0] > 1.5:
                continue
            _, rv, tv = best
            Rm = cv2.Rodrigues(rv)[0]
            n, c0 = Rm[:, 2], tv.ravel()  # card plane in camera coords (mm)
            # SfM observations on the card (or within a few px of its outline: the fingers holding it)
            poly = q.astype(np.float32).reshape(-1, 1, 2)
            dist = np.array([cv2.pointPolygonTest(poly, tuple(map(float, p)), True) for p in xy])
            sel = dist > -6
            if sel.sum() < 2:
                continue
            for p, z in zip(xy[sel], zs[sel]):
                r = cam.cam_from_img(np.array([p]))[0]
                ray = np.r_[r, 1.0]
                z_mm = (n @ c0) / (n @ ray)  # depth where the pixel ray meets the card plane
                if z > 0 and z_mm > 0:
                    ratios.append(z_mm / z)
            per_view.append(float(np.median([z_mm / z for z in zs[sel] if z > 0])) if (zs[sel] > 0).any() else None)
            used.append(im.name)
    ratios = np.array(ratios)
    if len(ratios) < min_ratios or len(used) < min_views:
        raise RuntimeError('Reference card not usable: hold the card flat against the arm, 25-35 cm from the phone, '
                           'and move the phone left and right during calibration.')
    med = np.median(ratios)
    good = np.abs(ratios / med - 1) < 0.05
    scale = float(np.median(ratios[good]))
    spread = float(100 * np.median(np.abs(ratios[good] / scale - 1)))
    log(f'[card-pnp] scale {scale:.4f} mm/unit from {len(used)} photos / {good.sum()} points, spread {spread:.2f}%')
    stats = {'reference': 'ID-1 card (single-view PnP)', 'scale_mm_per_unit': scale, 'scale_sheet': None, 'scale_edge': None,
             'card_views': len(used), 'card_view_names': sorted(set(used)), 'card_sides_mm': [CARD_W, CARD_H, CARD_W, CARD_H],
             'edge_vs_sheet_pct': spread,  # QA: robust spread of the per-point scale estimates
             'sheet_fits': [], 'markers': 0, 'corners': 4, 'images_with_markers': len(used),
             'median_reproj_px': None, 'mat_planarity_mm': None, 'method': 'pnp'}
    return scale, np.eye(3), np.zeros(3), stats
