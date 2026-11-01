"""Finger gaps the outline cannot see: where two neighbouring fingers touch in a photo, no background shows between
them, but the crease between them is still there as a dark line from the end of the visible gap toward the knuckles.
That line is the photo's evidence of where one finger ends and the next begins. It is traced here and cut into the
photo's cut-out as a thin gap, so the outline model can open the gap wherever several photos agree (hull.py: a few
photos are enough beyond the wrist).

Only in the corridor between two neighbouring fingers (from their middle joints down to the knuckle line, from the
hand landmarks found in the photo), only where the line is clearly darker than both fingers beside it, and never past
the knuckles: a palm crease or fingers that really are joined (no dark line) are left alone.
"""
import cv2
import numpy as np

MAX_T = 0.7  # how far from the middle joints toward the knuckles a crease may be cut (0..1)
PAIRS = [((5, 6), (9, 10)), ((9, 10), (13, 14)), ((13, 14), (17, 18))]  # (MCP, PIP) of neighbouring fingers


def _trace(gray, a0, a1, b0, b1, steps=48, lateral=0.22, smooth=3.0):
    """The darkest continuous path between two finger axes (a0->a1 and b0->b1: knuckle -> middle joint, pixels), from
    the middle joints (t=0) to the knuckles (t=1). Returns points (steps, 2) and per-point contrast (crease darkness
    relative to the fingers on either side: >0 darker)."""
    h, w = gray.shape
    ts = np.linspace(0, 1, steps)
    A = a1 + (a0 - a1) * ts[:, None]  # middle joint -> knuckle
    B = b1 + (b0 - b1) * ts[:, None]
    M = (A + B) / 2
    side = B - A
    width = np.linalg.norm(side, axis=1)
    n = side / np.maximum(width, 1e-6)[:, None]
    K = 9
    offs = np.linspace(-1, 1, 2 * K + 1)
    pts = M[:, None, :] + (offs[None, :, None] * lateral * width[:, None, None]) * n[:, None, :]
    xs = np.clip(pts[..., 0], 0, w - 1).astype(np.float32)
    ys = np.clip(pts[..., 1], 0, h - 1).astype(np.float32)
    I = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR)  # (steps, 2K+1)
    # dynamic programming: darkest path, lateral jumps penalised
    cost = I.copy()
    back = np.zeros(I.shape, np.int32)
    jump = smooth * np.abs(np.arange(2 * K + 1)[:, None] - np.arange(2 * K + 1)[None, :])
    for s in range(1, steps):
        c = cost[s - 1][None, :] + jump  # (to, from)
        back[s] = np.argmin(c, axis=1)
        cost[s] = I[s] + c[np.arange(2 * K + 1), back[s]]
    k = int(np.argmin(cost[-1]))
    path = [k]
    for s in range(steps - 1, 0, -1):
        k = back[s, k]
        path.append(k)
    path = path[::-1]
    P = np.array([pts[s, path[s]] for s in range(steps)])
    line = np.array([I[s, path[s]] for s in range(steps)])
    # the fingers on either side: half way from the path to each finger's axis
    fa = cv2.remap(gray, ((P[:, 0] + A[:, 0]) / 2).astype(np.float32)[None], ((P[:, 1] + A[:, 1]) / 2).astype(np.float32)[None],
                   cv2.INTER_LINEAR)[0]
    fb = cv2.remap(gray, ((P[:, 0] + B[:, 0]) / 2).astype(np.float32)[None], ((P[:, 1] + B[:, 1]) / 2).astype(np.float32)[None],
                   cv2.INTER_LINEAR)[0]
    contrast = 1.0 - line / np.maximum(np.minimum(fa, fb), 1.0)
    return P, contrast, width


def cut(bgr, mask, lm, min_contrast=0.12, run=3, gap_px=None):
    """The cut-out with the creases between touching fingers cut in as thin gaps. lm: 21 hand landmarks (0..1).
    Returns (mask, number of pixels cut)."""
    if lm is None or len(lm) < 21:
        return mask, 0
    h, w = mask.shape[:2]
    L = np.array([[p[0] * w, p[1] * h] for p in lm[:21]], np.float32)
    if (L < 0).any() or (L[:, 0] >= w).any() or (L[:, 1] >= h).any():
        return mask, 0
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    out = mask.copy()
    hand = mask > 127
    total = 0
    for (a0i, a1i), (b0i, b1i) in PAIRS:
        P, con, width = _trace(gray, L[a0i], L[a1i], L[b0i], L[b1i])
        # from the middle joints toward the knuckles: cut while the crease is clearly there (dark for `run` points in
        # a row); stop for good where it fades (the web, or fingers really joined)
        dark = con > min_contrast
        stop = len(P)
        for s in range(len(P) - run):
            if not dark[s:s + run].any():
                stop = s
                break
        # never down to the knuckles: on the palm side the web is half way down the first finger bone, and below it
        # the palm's own creases are dark too (31 Oct: notches cut into the palm with 0.92)
        stop = min(stop, int(MAX_T * len(P)))
        g = gap_px or max(2, int(round(0.08 * float(np.median(width)))))
        line = np.zeros_like(out)
        for s in range(stop - 1):
            if dark[s] and dark[s + 1]:
                cv2.line(line, tuple(np.round(P[s]).astype(int)), tuple(np.round(P[s + 1]).astype(int)), 255, g)
        cutpx = (line > 0) & hand
        total += int(cutpx.sum())
        out[cutpx] = 0
    return out, total


def _ray(meta, u, v):
    """Camera centre and unit direction (AR world) of the view ray through (u, v) (0..1 of the stored photo), the
    inverse of segment._image_point."""
    P = np.asarray(meta['pose'], float)
    fx, fy, cx, cy = meta['K']
    w, h = meta.get('stored_w', meta.get('w')), meta.get('stored_h', meta.get('h'))
    s = w / float(meta.get('w', w))
    yz = (u * w / s - cx) / fx
    xz = (v * h / s - cy) / fy
    d = xz * P[0:3] + yz * P[4:7] - P[8:11]
    return P[12:15], d / np.linalg.norm(d)


def _triangulate(rays):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ C
    return np.linalg.solve(A, b)


def back_views(metas, dets):
    """Names of the photos that look at the back of the hand (the crease between touching fingers is clean there; on
    the palm side the palm's own creases and the finger pads' shadows are dark too). The wrist, the index and little
    finger knuckles and the thumb are triangulated from all photos with the phone's poses; the thumb lies on the palm
    side of the wrist-knuckle plane, which gives the palm's direction; a photo whose camera is on the other side sees
    the back. None when it cannot be told (too few photos, no AR poses)."""
    idx = (0, 5, 17, 2, 3, 4)
    rays = {i: [] for i in idx}
    for n, lm in dets.items():
        m = metas.get(n)
        if not m or 'pose' not in m or not m.get('K') or not lm or len(lm) < 21:
            continue
        for i in idx:
            rays[i].append(_ray(m, lm[i][0], lm[i][1]))
    if min(len(r) for r in rays.values()) < 6:
        return None
    X = {i: _triangulate(rays[i]) for i in idx}
    c = (X[0] + X[5] + X[17]) / 3
    nrm = np.cross(X[5] - X[0], X[17] - X[0])
    nrm /= np.linalg.norm(nrm)
    thumb = (X[2] + X[3] + X[4]) / 3 - c
    if abs(thumb @ nrm) < 0.003:  # thumb in the plane (3 mm): cannot tell
        return None
    palm = nrm * np.sign(thumb @ nrm)
    out = set()
    for n in dets:
        m = metas.get(n)
        if m and 'pose' in m and (np.asarray(m['pose'], float)[12:15] - c) @ palm < 0:
            out.add(n)
    return out
