"""Clinical limb measurements: centreline, tape-measure girths, lengths, volume."""
import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union


def _basis(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _despike(pts, nbins=72, win=3, tol=3.0):
    """A tape slides over thin stray fragments (hair-like flaps of reconstruction noise): drop points that
    stick out more than `tol` mm beyond the outer radius of their angular neighbourhood."""
    if len(pts) < 30:
        return pts
    c = np.median(pts, 0)
    d = pts - c
    r = np.hypot(d[:, 0], d[:, 1])
    b = ((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    rmax = np.full(nbins, np.nan)
    for k in np.unique(b):
        rmax[k] = np.percentile(r[b == k], 90)
    ref = np.array([np.nanmedian(np.take(rmax, range(k - win, k + win + 1), mode='wrap')) for k in range(nbins)])
    ref = np.where(np.isnan(ref), np.nanmax(rmax), ref)
    keep = r <= ref[b] + tol
    return pts[keep] if keep.sum() > 0.8 * len(pts) else pts


def _complete_polar(pts, nbins=72, order=4):
    """Angular coverage of a cross-section and a completed closed curve: unseen arcs are filled with a
    periodic Fourier fit r(theta) to the seen part (limb sections are smooth and near-convex)."""
    c = pts.mean(0)
    for _ in range(3):  # re-centre on the curve, not the point mass (robust to one-sided data)
        th = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        r = np.linalg.norm(pts - c, axis=1)
        b = ((th + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
        rb = np.full(nbins, np.nan)
        for k in np.unique(b):
            rb[k] = np.median(r[b == k])
        thb = (np.arange(nbins) + 0.5) / nbins * 2 * np.pi - np.pi
        ok = ~np.isnan(rb)
        A = np.column_stack([np.ones(nbins)] + [f(m * thb) for m in range(1, order + 1) for f in (np.cos, np.sin)])
        coef = np.linalg.lstsq(A[ok], rb[ok], rcond=None)[0]
        # first harmonic = centre offset
        dc = np.array([coef[1], coef[2]])
        c = c + dc
        if np.linalg.norm(dc) < 0.05:
            break
    cov = ok.mean()
    rfit = A @ coef
    fill = np.c_[c[0] + rfit[~ok] * np.cos(thb[~ok]), c[1] + rfit[~ok] * np.sin(thb[~ok])]
    return cov, fill


def section_at(mesh, origin, normal, near=None, max_dist=60.0, complete=True):
    """Cut the mesh with a plane. Returns the tape-measure girth (convex-hull perimeter) of the limb
    cross-section nearest `near`, its area, contour perimeter and 3D centroid; None if no limb there."""
    segs = trimesh.intersections.mesh_plane(mesh, normal, origin)
    if len(segs) < 4:
        return None
    u, v = _basis(normal)
    B = np.stack([u, v], 1)
    s2 = (segs - origin) @ B  # (n,2,2)
    c0 = ((origin if near is None else near) - origin) @ B
    mid = s2.mean(1)
    keep = np.linalg.norm(mid - c0, axis=1) < max_dist + 40
    if keep.sum() < 4:
        return None
    s2 = s2[keep]
    pts = _despike(s2.reshape(-1, 2))
    cov, fill = 1.0, np.zeros((0, 2))
    if complete:
        cov, fill = _complete_polar(pts)
        if cov < 0.97 and cov > 0.5:
            pts = np.vstack([pts, fill])
    hull = MultiPoint([tuple(x) for x in pts]).convex_hull
    if hull.geom_type != 'Polygon':
        return None
    groups = []
    try:
        path = trimesh.load_path(s2)
        from shapely.geometry import Point
        polys = [p for p in path.polygons_closed if p is not None and p.is_valid and p.area > 4.0]
        if polys:
            un = unary_union(polys)
            groups = [g for g in (un.geoms if hasattr(un, 'geoms') else [un]) if g.distance(Point(c0)) < max_dist]
    except Exception:
        groups = []
    if groups:
        un = unary_union(groups)
        area = un.area if un.area > 0.85 * hull.area else hull.area
        cen2 = np.asarray(un.centroid.coords[0])
        perim = sum(g.exterior.length for g in groups)
    else:
        area, cen2, perim = hull.area, np.asarray(hull.centroid.coords[0]), hull.length
    cen3 = origin + B @ cen2
    if cov < 0.97 and len(fill):
        area = hull.area  # open section: closed-loop area unavailable, use completed hull
    return {'girth': hull.length, 'perimeter': perim, 'area': area, 'centroid': cen3, 'n_loops': len(groups), 'seen': cov,
            'hull2d': np.asarray(hull.exterior.coords), 'basis': (origin, u, v)}


def _march(mesh, p0, t0, step, max_steps=200, max_turn_deg=55, follow_bends=False):
    """Track the limb from p0 along direction t0: section, re-centre, update tangent, repeat."""
    pts, tans, girths = [], [], []
    t0 = t0 / np.linalg.norm(t0)
    p, t, prev = p0.copy(), t0.copy(), None
    for _ in range(max_steps):
        s = section_at(mesh, p, t, near=p)
        if s is None:
            break
        g = s['girth']
        if girths and g > 1.3 * girths[-1]:
            break  # abrupt jump between two 5 mm steps: elbow/shoulder debris or the body (the hand widens gradually)
        c = s['centroid']
        if prev is not None and np.dot(c - prev, t if follow_bends else t0) < 0.3 * step:
            break  # not advancing along the limb any more (looping in debris)
        if np.dot(t, t0) < np.cos(np.deg2rad(max_turn_deg)):
            break  # limb bends away (elbow): stop, the segment of interest ends here
        if prev is not None:
            d = c - prev
            if np.linalg.norm(d) > 1e-6:
                t = 0.75 * t + 0.25 * d / np.linalg.norm(d)
                t /= np.linalg.norm(t)
        pts.append(c); tans.append(t.copy()); girths.append(g)
        prev = c
        p = c + t * step
    return np.array(pts).reshape(-1, 3), np.array(tans).reshape(-1, 3)


def centreline(mesh, step=5.0):
    """Smooth centreline tracked in both directions from the middle of the limb."""
    v = mesh.vertices
    c = v.mean(0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    ax = vt[0]
    t = (v - c) @ ax
    # start at the median-girth single-loop section near the middle of the long axis
    best = None
    for f in np.linspace(-0.25, 0.25, 11):
        p = c + ax * (np.median(t) + f * (t.max() - t.min()))
        s = section_at(mesh, p, ax, near=p)
        if s is not None and s['n_loops'] == 1:
            if best is None or abs(f) < abs(best[0]):
                best = (f, s['centroid'])
    p0 = best[1] if best else c
    a_pts, a_t = _march(mesh, p0, ax, step)
    b_pts, b_t = _march(mesh, p0 - ax * step, -ax, step)
    pts = np.vstack([b_pts[::-1], a_pts])
    tang = np.vstack([-b_t[::-1], a_t])
    if len(pts) < 5:
        return pts, tang
    # smooth + resample at uniform arc length
    k = 3
    Ps = np.array([pts[max(0, i - k):i + k + 1].mean(0) for i in range(len(pts))])
    Ps[0], Ps[-1] = pts[0], pts[-1]
    seg = np.linalg.norm(np.diff(Ps, axis=0), axis=1)
    sarc = np.r_[0, np.cumsum(seg)]
    ss = np.arange(0, sarc[-1], step)
    out = np.stack([np.interp(ss, sarc, Ps[:, j]) for j in range(3)], 1)
    tg = np.gradient(out, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12
    return out, tg


def measure(mesh, step=5.0):
    """Girth profile along the limb from the distal tip. Distal end = narrower/branching (hand) end."""
    pts, tang = centreline(mesh, step=step)
    rows = []
    for p, n in zip(pts, tang):
        s = section_at(mesh, p, n, near=p)
        rows.append(s)
    ok = [i for i, r in enumerate(rows) if r is not None]
    pts, tang, rows = pts[ok], tang[ok], [rows[i] for i in ok]
    loops = np.array([r['n_loops'] for r in rows])
    area = np.array([r['area'] for r in rows])
    # orient so index 0 is the distal end: hand sections are flat (aspect ~2-3), forearm sections round (~1.3)
    def flat(r):
        h = r['hull2d'] - r['hull2d'].mean(0)
        ev = np.linalg.eigvalsh(h.T @ h)
        return float(np.sqrt(ev[1] / max(ev[0], 1e-9)))
    asp = np.array([flat(r) for r in rows])
    q = max(len(rows) // 4, 1)
    if np.median(asp[-q:]) > np.median(asp[:q]):
        pts, tang, rows, loops, area = pts[::-1], -tang[::-1], rows[::-1], loops[::-1], area[::-1]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    girth_raw = np.array([r['girth'] for r in rows])
    # landmarks use a lightly smoothed profile (5-sample median) so single noisy sections can't move them
    from scipy.ndimage import median_filter
    girth = median_filter(girth_raw, size=5, mode='nearest')
    L = arc[-1]
    lm = {}
    # wrist: walk from the proximal (elbow) end toward the hand; girth falls to the wrist, then rises
    # sharply into the hand. The wrist is the valley before the first >15% rise (independent of the hand).
    n = len(girth)
    seen = np.array([r['seen'] for r in rows])
    # skip the proximal end: the last 40 mm (cut, flaps) and any poorly seen stretch there (the elbow end is
    # often only partly photographed; the closed mesh tapers and wobbles, which looks like a "rise")
    j0 = n - 1
    while j0 > 0 and (arc[-1] - arc[j0] < 40 or (seen[j0] < 0.6 and arc[-1] - arc[j0] < 0.35 * L)):
        j0 -= 1
    m_i, i_stop = j0, None
    for k in range(j0 - 1, -1, -1):
        if girth[k] < girth[m_i]:
            m_i = k
        elif girth[k] > 1.12 * girth[m_i] and arc[m_i] - arc[k] > 10:
            i_stop = k
            break
    if i_stop is None and arc[m_i] > 10 and girth[0] > 1.04 * girth[m_i]:
        i_stop = 0  # scan ends shortly after the wrist (hand only partly captured)
    if i_stop is not None and arc[-1] - arc[m_i] > 40:
        gmin = girth[m_i]
        tol = 0.02 * gmin  # valley = girth within 2% of the minimum; take its centre (flat, so argmin is noisy)
        lo = hi = m_i
        while lo - 1 > i_stop and girth[lo - 1] < gmin + tol: lo -= 1
        while hi + 1 < n and girth[hi + 1] < gmin + tol: hi += 1
        # wrist position = vertex of a parabola fitted to the valley (+-20 mm): it moves continuously with the
        # data, unlike snapping to the nearest 5 mm sample, so tiny mesh differences can't jump it (and with it
        # every station and the volume zone) by a whole sample
        sw = _valley_vertex(arc, girth, m_i, lo, hi, i_stop)
        g_at = lambda s_: float(np.interp(s_, arc, girth_raw))
        lm['wrist'] = {'s_mm': sw, 'girth_mm': g_at(sw), 'min_girth_mm': float(gmin)}
        i = int(np.argmin(np.abs(arc - sw)))
        for dd in (50, 100, 150):  # standard forearm stations above the wrist, interpolated at exactly +dd
            if sw + dd <= arc[-1] - 2:
                lm[f'wrist_plus_{dd}'] = {'s_mm': sw + dd, 'girth_mm': g_at(sw + dd), 'from_wrist_mm': dd}
        # hand girth (MCP / knuckles, incl. thumb) = the FIRST peak walking from the wrist toward the fingers:
        # further on, the tape-measure hull can span spread fingers and read far more than the hand
        j, best = None, 0.0
        for k in range(i - 1, -1, -1):
            if arc[i] - arc[k] < 15:
                continue
            if arc[i] - arc[k] > 120:
                break
            if girth[k] > best:
                j, best = k, girth[k]
            elif girth[k] < 0.93 * best:  # past the knuckles (finger gaps)
                break
        if j is not None:
            lm['hand_max'] = {'s_mm': float(arc[j]), 'girth_mm': float(girth_raw[j])}
    # clinical zone: wrist -> wrist + 150 mm (forearm segment volumetry, as in lymphoedema practice)
    z0 = lm['wrist']['s_mm'] if 'wrist' in lm else 0.0
    zend = z0 + 150 if 'wrist' in lm else L
    zone = (arc >= z0) & (arc <= zend)
    # integrate with exact (interpolated) ends, not the nearest samples
    za = np.r_[z0, arc[zone], min(zend, L)]
    vol = float(np.trapezoid(np.interp(za, arc, area), za)) / 1000.0  # cm^3 (ml)
    return {
        'length_mm': float(L),
        'volume_ml': vol,
        'landmarks': lm,
        'section_seen_pct': float(100 * seen[zone].mean()) if zone.any() else None,
        'zone_mm': [float(z0), float(min(zend, L))],
        'profile': [{'s_mm': round(float(a), 1), 'girth_mm': round(float(g), 2), 'area_mm2': round(float(ar), 1), 'loops': int(l), 'seen': round(float(sn), 2)}
                    for a, g, ar, l, sn in zip(arc, girth_raw, area, loops, seen)],
        '_centreline': pts, '_tangents': tang,
    }


def _valley_vertex(arc, girth, m_i, lo, hi, i_stop):
    """Sub-sample wrist position: least-squares parabola through the valley samples within 20 mm of the band
    centre; falls back to the band centre if the fit is not a proper (convex, in-range) valley."""
    c = 0.5 * (arc[lo] + arc[hi])
    sel = (np.abs(arc - c) <= 20) & (np.arange(len(arc)) > (i_stop if i_stop is not None else -1))
    if sel.sum() >= 5:
        a, b, _ = np.polyfit(arc[sel] - c, girth[sel], 2)
        if a > 1e-6:
            v = c - b / (2 * a)
            if arc[lo] - 5 <= v <= arc[hi] + 5:
                return float(v)
    return float(c)


def extend_past_elbow(mesh, cl, step=5.0):
    """Continue the centreline beyond the measured segment, following the limb around the elbow bend, until the
    data ends. Returns the extra points (may be empty)."""
    if len(cl) < 6:
        return np.zeros((0, 3))
    t = cl[-1] - cl[-6]; t /= np.linalg.norm(t)
    pts, _ = _march(mesh, cl[-1] + t * step, t, step, max_turn_deg=170, follow_bends=True)
    return pts
