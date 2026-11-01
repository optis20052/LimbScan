"""The arm's position from the hand in every photo: MediaPipe's hand landmarker per photo, each landmark's viewing
ray from the photo's AR camera pose, and the rays of all photos intersected (least squares, outliers dropped).

Deterministic and texture-free, and it does not depend on the cylinder the phone placed while scanning (a guess
from one view can be tilted towards or away from that view). The result is the cylinder the masks are cut with.
"""
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / 'models' / 'hand_landmarker.task'
_det = None


def _detector():
    global _det
    if _det is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as V
        _det = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)), num_hands=2,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4))
    return _det


def _ray(meta, u, v, w, h):
    """Viewing ray (origin, unit direction) through (u, v) (0..1 of the stored portrait photo); inverse of
    segment._image_point."""
    P = np.asarray(meta['pose'], float)
    fx, fy, cx, cy = meta['K']
    s = w / float(meta.get('w', w))
    ys = (u * w / s - cx) / fx  # sensor y / z
    xs = (v * h / s - cy) / fy  # sensor x / z
    d = P[0:3] * xs + P[4:7] * ys - P[8:11]
    return P[12:15].copy(), d / np.linalg.norm(d)


def _meet(rays):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ o
    return np.linalg.solve(A, b)


def _off(x, ray):
    o, d = ray
    q = x - o
    return np.linalg.norm(q - d * (q @ d))


def triangulate(rays, tol=0.02):
    """The point nearest to most rays: start from every pair's meeting point, keep the one most rays pass within
    `tol` (same unit as the rays' origins) of (another hand in some photos, a wrong detection), refine on those. (point, inliers) or None."""
    if len(rays) < 4:
        return None
    best, best_in = None, []
    n = len(rays)
    step = max(1, n // 25)
    for i in range(0, n, step):
        for j in range(i + 1, n, step):
            if abs(rays[i][1] @ rays[j][1]) > np.cos(np.deg2rad(10)):
                continue  # nearly parallel: depth undefined
            try:
                x = _meet([rays[i], rays[j]])
            except np.linalg.LinAlgError:
                continue
            inl = [r for r in rays if _off(x, r) < tol]
            if len(inl) > len(best_in):
                best, best_in = x, inl
    if best is None or len(best_in) < 4:
        return None
    x = _meet(best_in)
    return x, [r for r in rays if _off(x, r) < tol]


def detect(image_dir, metas, near=None):
    """The scanned hand's 21 landmarks per photo (0..1 of the photo): {name: [[x, y], ...]}. `near`: the phone's arm
    {a, b, r} (AR world), to pick the right hand when two are in a photo; else the biggest hand."""
    import mediapipe as mp
    det = _detector()
    out = {}
    for p in sorted(Path(image_dir).glob('*.jpg')):
        m = metas.get(p.name)
        if not m or 'pose' not in m or 'K' not in m:
            continue
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
        if not r.hand_landmarks:
            continue
        cands = r.hand_landmarks
        if near is not None and len(cands) > 1:
            from .segment import _image_point
            c = _image_point(m, (np.asarray(near['a']) + np.asarray(near['b'])) / 2, w, h)
            if c is not None:
                cands = [min(cands, key=lambda l: (l[0].x - c[0]) ** 2 + (l[0].y - c[1]) ** 2)]
        if len(cands) > 1:  # the biggest hand: the one being scanned
            cands = [max(cands, key=lambda l: (l[0].x - l[9].x) ** 2 + (l[0].y - l[9].y) ** 2)]
        out[p.name] = [[float(q.x), float(q.y)] for q in cands[0]]
    return out


_K = (0, 5, 8, 9, 12, 13, 16, 17)


def _arm(rays, unit, seen, forearm, log):
    """{a, b, r, wrist} from per-landmark rays; lengths in `unit` per metre (1 = AR metres, 1000 = model mm)."""
    pts = {}
    for k in _K:
        t = triangulate(rays[k], tol=0.02 * unit)
        if t is not None:
            pts[k] = t
    if not all(k in pts for k in (0, 5, 9, 17)):
        log(f'[arm] hand found in {seen} photos: too few agreeing views to place the arm')
        return None
    W, M, I5, P17 = (pts[k][0] for k in (0, 9, 5, 17))
    L, palm = np.linalg.norm(W - M) / unit, np.linalg.norm(I5 - P17) / unit
    if not (0.05 < L < 0.14 and 0.04 < palm < 0.11):
        log(f'[arm] triangulated hand implausible (wrist-knuckle {L * 1000:.0f} mm, palm {palm * 1000:.0f} mm)')
        return None
    u = (W - M) / (L * unit)
    tip = max([L] + [-(pts[k][0] - W) @ u / unit for k in (8, 12, 16) if k in pts])
    tip = min(tip, 0.24)
    r = float(np.clip(0.65 * palm, 0.03, 0.06))
    log(f'[arm] hand found in {seen} photos, wrist from {len(pts[0][1])} of them; wrist-knuckle {L * 1000:.0f} mm, '
        f'palm {palm * 1000:.0f} mm, radius {r * 1000:.0f} mm')
    return {'a': (W + u * forearm * unit).tolist(), 'b': (W - u * (tip + 0.01) * unit).tolist(), 'r': r * unit,
            'wrist': W.tolist()}


def arm_from_photos(image_dir, metas, near=None, forearm=0.22, log=print, dets=None):
    """{a, b, r, wrist} (AR world, metres) from the hand in the photos and their AR poses, or None. Also returns the
    detections (for arm_in_model)."""
    dets = detect(image_dir, metas, near) if dets is None else dets
    rays = {k: [] for k in _K}
    for name, lm in dets.items():
        m = metas[name]
        w, h = m.get('stored_w', m.get('w', 1080)), m.get('stored_h', m.get('h', 1920))
        for k in _K:
            rays[k].append(_ray(m, lm[k][0], lm[k][1], w, h))
    return _arm(rays, 1.0, len(dets), forearm, log), dets


def arm_in_model(model_dir, dets, forearm=0.22, log=print):
    """The same hand triangulated with the reconstruction's own cameras (metric, mm): in the frame of the 3D model,
    so it lines up with the mesh even when the arm swayed relative to the room (AR poses follow the room)."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(model_dir))
    rays = {k: [] for k in _K}
    n = 0
    for im in rec.images.values():
        lm = dets.get(im.name)
        if lm is None:
            continue
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        R = T.rotation.matrix()
        C = T.inverse().translation
        px = np.array([[x * cam.width, y * cam.height] for x, y in lm], float)
        xy = cam.cam_from_img(px)
        n += 1
        for k in _K:
            d = R.T @ np.array([xy[k][0], xy[k][1], 1.0])
            rays[k].append((np.asarray(C, float), d / np.linalg.norm(d)))
    return _arm(rays, 1000.0, n, forearm, log)


def landmarks_in_model(model_dir, dets, tol_mm=10.0, min_views=6):
    """All 21 hand landmarks triangulated with the reconstruction's own cameras (mm, the model's frame): {index:
    point}. A landmark is kept only when at least `min_views` photos agree on it within `tol_mm` (a finger the
    landmarker guessed - hidden, or missing on a patient's hand - is left out, not invented)."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(model_dir))
    rays = {k: [] for k in range(21)}
    for im in rec.images.values():
        lm = dets.get(im.name)
        if lm is None or len(lm) < 21:
            continue
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        R = T.rotation.matrix()
        C = np.asarray(T.inverse().translation, float)
        xy = cam.cam_from_img(np.array([[p[0] * cam.width, p[1] * cam.height] for p in lm[:21]], float))
        for k in range(21):
            d = R.T @ np.array([xy[k][0], xy[k][1], 1.0])
            rays[k].append((C, d / np.linalg.norm(d)))
    out = {}
    for k, r in rays.items():
        t = triangulate(r, tol=tol_mm)
        if t is not None and len(t[1]) >= min_views:
            out[k] = t[0]
    return out
