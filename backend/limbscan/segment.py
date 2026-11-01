"""Limb segmentation (MediaPipe selfie-multiclass 'body-skin' class) + scale-card regions.

Masks drive the arm-only reconstruction: features, camera poses and dense stereo come from the limb's
own skin, so the limb may sway relative to the room without breaking the scan.
"""
import os
from pathlib import Path

import cv2
import numpy as np

# off: tested 31 Oct, no gain on two scans and one much worse (more photos dropped, shape step rejected)
FILL = os.environ.get('LIMBSCAN_FILL_HOLES', '0') == '1'
FINGERS = os.environ.get('LIMBSCAN_FINGERS', '0') == '1'  # recover fingers the cut-out missed (off: added background)
MODEL = Path(__file__).resolve().parent.parent / 'models' / 'selfie_multiclass.tflite'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'


def _segmenter():
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    if not MODEL.exists():
        import urllib.request
        MODEL.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=str(MODEL)),
                                        output_category_mask=False, output_confidence_masks=True)
    return mp, vision.ImageSegmenter.create_from_options(opts)


def skin_mask(seg, mp, bgr, thr=0.5):
    """Binary limb mask at full resolution: body-skin probability, cleaned, largest regions kept."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    prob = r.confidence_masks[2].numpy_view()  # 2 = body skin (3 = face skin: excluded on purpose)
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    m = (prob > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.where(areas >= max(0.15 * areas.max(), 0.002 * h * w))[0]
        m = np.isin(lab, keep).astype(np.uint8)
    return m


_local = None

TOUCH_MODEL = Path(__file__).resolve().parent.parent / 'models' / 'interactive_segmentation.task'
TOUCH_URL = 'https://storage.googleapis.com/mediapipe-models/interactive_segmenter_v2/magic_touch/int8/latest/interactive_segmentation.task'


def touch_mask(bgr, seg):
    """Test-model mode (e.g. a 3D-printed hand): MediaPipe's interactive segmenter (MagicTouch) at the point the
    phone tracked inside the object, then only the connected region there, holes filled."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision as V
    global _local
    import threading
    if _local is None:
        _local = threading.local()
    if not hasattr(_local, 'touch'):
        if not TOUCH_MODEL.exists():
            import urllib.request
            urllib.request.urlretrieve(TOUCH_URL, TOUCH_MODEL)
        _local.touch = V.InteractiveSegmenter.create_from_options(
            V.InteractiveSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=str(TOUCH_MODEL))))
    h, w = bgr.shape[:2]
    sx, sy = seg.get('seed', [0.5, 0.5])
    pts = [p for p in (seg.get('pts') or [[sx, sy]]) if 0 <= p[0] < 1 and 0 <= p[1] < 1] or [[sx, sy]]
    _local.touch.set_image(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
    stroke = V.InteractiveSegmenterStroke(brush_mode=V.InteractiveSegmenterBrushMode.POSITIVE,
                                          points=[V.InteractiveSegmenterStrokePoint(float(x), float(y)) for x, y in pts],
                                          is_completed=True)
    prob = _local.touch.segment([stroke]).numpy_view().squeeze()
    prob = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = (prob > 0.5).astype(np.uint8)
    n, lab_, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)
    ks = set()
    for x, y in pts:  # the regions under the points (the limb), or the nearest region when a point is just off it
        px, py = int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))
        k = lab_[py, px]
        if k == 0:
            k = min(range(1, n), key=lambda i: (cen[i][0] - px) ** 2 + (cen[i][1] - py) ** 2)
        ks.add(int(k))
    m = np.isin(lab_, list(ks)).astype(np.uint8)
    inv = (1 - m).copy()
    cv2.floodFill(inv, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 0)  # fill holes
    return m | inv


_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
          (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def _hand_px(seg, w, h):
    """The hand landmarks found in this photo (seg['hand'], 0..1) in pixels, or None."""
    lm = (seg or {}).get('hand')
    if not lm or len(lm) < 21:
        return None
    return np.array([[p[0] * w, p[1] * h] for p in lm[:21]], np.float32)


def hand_cover(m, seg, tol=6):
    """Share of the hand's landmarks (those inside the photo) that lie on the cut-out (within tol px), or None."""
    h, w = m.shape[:2]
    P = _hand_px(seg, w, h)
    if P is None:
        return None
    ins = (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
    if ins.sum() < 15:
        return None
    md = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1)))
    Pi = P[ins].astype(int)
    return float(md[Pi[:, 1], Pi[:, 0]].mean())


def recover_fingers(bgr, m, seg):
    """Fingers the hand landmarker found in the photo but the cut-out left out (shine, light from behind, a finger
    the colour of the wall: 31 Oct, three fingers and the thumb missing in one photo carved the ring finger away)
    cut out again from their own joints, kept only along those fingers."""
    h, w = m.shape[:2]
    P = _hand_px(seg, w, h)
    if P is None:
        return m
    inside = lambda q: 0 <= q[0] < w and 0 <= q[1] < h and m[int(q[1]), int(q[0])] > 0  # noqa: E731
    pts, miss = [], np.zeros((h, w), np.uint8)
    palm = float(np.linalg.norm(P[5] - P[17]))
    width = max(6, int(0.42 * palm))  # a finger is about a quarter of the palm wide: generous, then cut by the image
    for a, b in _BONES:
        if a == 0 and b not in (1, 5, 17):
            continue
        for t in (0.5, 1.0):
            q = P[a] + t * (P[b] - P[a])
            if not inside(q) and 0 <= q[0] < w and 0 <= q[1] < h:
                pts.append([q[0] / w, q[1] / h])
                cv2.line(miss, tuple(P[a].astype(int)), tuple(P[b].astype(int)), 1, width)
    if not pts:
        return m
    t = touch_mask(bgr, {'seed': pts[0], 'pts': pts})
    return (m | (t & miss)).astype(np.uint8)


def _roi(shape, seg):
    """The placed arm's outline in this photo (convex hull of the phone's projected cylinder points), or None."""
    pts = (seg or {}).get('roi')
    if not pts or len(pts) < 6:
        return None
    h, w = shape[:2]
    P = np.array([[x * w, y * h] for x, y in pts], np.float32)
    P = P[(P[:, 0] > -2 * w) & (P[:, 0] < 3 * w) & (P[:, 1] > -2 * h) & (P[:, 1] < 3 * h)]
    if len(P) < 6:
        return None
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(P).astype(np.int32), 1)
    return m


def limb_mask(bgr, seg=None):
    """The limb in one photo, kept inside the arm the user placed in AR when the phone sent its outline ('roi'):
    a cut-out that ran onto a body, a chair or a table outside it is dropped."""
    m = _crop_mask(bgr, seg) if CROP else None
    if m is None:
        m = _limb_mask(bgr, seg)
    if m is not None and FINGERS and (seg or {}).get('hand'):
        m = recover_fingers(bgr, m, seg)
    roi = _roi(bgr.shape, seg)
    if roi is None or m is None:
        return m
    clipped = m & roi
    # the connected part through the seed (what is left of a spill can be a separate scrap inside the outline)
    if seg.get('seed') and clipped.any():
        h, w = clipped.shape
        n, lab, st, cen = cv2.connectedComponentsWithStats(clipped, 8)
        px, py = int(np.clip(seg['seed'][0] * w, 0, w - 1)), int(np.clip(seg['seed'][1] * h, 0, h - 1))
        k = lab[py, px]
        if k == 0 and n > 1:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if k:
            clipped = (lab == k).astype(np.uint8)
    return clipped


CROP = __import__('os').environ.get('LIMBSCAN_SEG_CROP', '1') == '1'


def _crop_mask(bgr, seg):
    """The cut-out run on a crop around the placed arm (its projected outline plus a margin), pasted back: the
    segmentation models work at ~256-512 px, so on the whole photo a 3 mm gap between two fingers (~13 px) is 2-3
    model pixels and closes; on the crop it is resolved. None without an outline."""
    if not seg or not seg.get('roi') or not seg.get('seed'):
        return None
    h, w = bgr.shape[:2]
    r = np.clip(np.asarray(seg['roi'], float), 0, 1)
    x0, x1 = r[:, 0].min() * w, r[:, 0].max() * w
    y0, y1 = r[:, 1].min() * h, r[:, 1].max() * h
    mx, my = 0.08 * (x1 - x0) + 16, 0.08 * (y1 - y0) + 16
    x0, x1 = int(max(0, x0 - mx)), int(min(w, x1 + mx))
    y0, y1 = int(max(0, y0 - my)), int(min(h, y1 + my))
    if x1 - x0 < 64 or y1 - y0 < 64 or (x1 - x0) * (y1 - y0) > 0.8 * w * h:
        return None
    cw, ch = x1 - x0, y1 - y0
    sub = dict(seg)
    sub['seed'] = [(seg['seed'][0] * w - x0) / cw, (seg['seed'][1] * h - y0) / ch]
    if not (0 <= sub['seed'][0] < 1 and 0 <= sub['seed'][1] < 1):
        return None
    sub.pop('roi', None)
    if sub.get('pts'):
        sub['pts'] = [[(x * w - x0) / cw, (y * h - y0) / ch] for x, y in sub['pts']]
    mc = _limb_mask(np.ascontiguousarray(bgr[y0:y1, x0:x1]), sub)
    if mc is None:
        return None
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = mc
    return m


def _classes(bgr, ids):
    """Max probability over selfie-multiclass classes `ids` (0 background, 1 hair, 2 body skin, 3 face skin,
    4 clothes, 5 other) at full resolution."""
    global _local
    import threading
    if _local is None:
        _local = threading.local()
    if not hasattr(_local, 'seg'):
        _local.mp, _local.seg = _segmenter()
    h, w = bgr.shape[:2]
    r = _local.seg.segment(_local.mp.Image(image_format=_local.mp.ImageFormat.SRGB,
                                           data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
    p = np.max([r.confidence_masks[i].numpy_view() for i in ids], axis=0)
    return cv2.resize(p.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _limb_mask(bgr, seg=None):
    """The limb in one photo. seg (from the phone) = {mode: auto, seed: [x, y]}: a point on the limb from the arm the
    user placed in AR. Skin regions touching that point are the limb (other skin, a skin-coloured chair, is dropped);
    when there is no skin at the point (a printed model, gloves) the interactive segmenter cuts out what is there.
    mode touch = always the interactive segmenter; no seg = all large skin regions."""
    global _local
    import threading
    if _local is None:
        _local = threading.local()
    if seg and seg.get('mode') == 'touch':
        return touch_mask(bgr, seg)
    if not hasattr(_local, 'seg'):
        _local.mp, _local.seg = _segmenter()
    m = skin_mask(_local.seg, _local.mp, bgr)
    if not (seg and seg.get('mode') == 'auto' and seg.get('seed')):
        return m
    if seg.get('pts'):
        # the placed arm's axis in this photo (palm, wrist, forearm): the interactive cut-out along all of it does not
        # depend on skin colour (a hairy forearm, a sleeve edge); skin regions on those points add what it missed
        # (fingers). A spill onto a body or a chair is clipped to the arm's outline afterwards (limb_mask).
        t = touch_mask(bgr, seg)
        h, w = m.shape
        # the cut-out can run onto the face, hair or the shirt behind the arm: the same body model labels those
        not_limb = _classes(bgr, (3, 4)) > 0.5  # face skin, clothes (not hair: a hairy forearm reads as hair)
        not_limb = cv2.morphologyEx(not_limb.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
        t = t & ~not_limb
        _, lab, _, _ = cv2.connectedComponentsWithStats(m, 8)
        ks = {int(lab[int(np.clip(y * h, 0, h - 1)), int(np.clip(x * w, 0, w - 1))]) for x, y in seg['pts'] if 0 <= x < 1 and 0 <= y < 1}
        ks.discard(0)
        return (t | np.isin(lab, list(ks))).astype(np.uint8)
    h, w = m.shape
    px, py = int(np.clip(seg['seed'][0] * w, 0, w - 1)), int(np.clip(seg['seed'][1] * h, 0, h - 1))
    r = max(8, int(0.04 * w))
    y0, y1, x0, x1 = max(0, py - r), min(h, py + r + 1), max(0, px - r), min(w, px + r + 1)
    if m[y0:y1, x0:x1].mean() < 0.3:
        return touch_mask(bgr, seg)
    _, lab, _, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.unique(lab[y0:y1, x0:x1][m[y0:y1, x0:x1] > 0])
    return np.isin(lab, keep[keep > 0]).astype(np.uint8)


def masks_for(bgr, cards=True, seg=None):
    """One photo -> (SfM feature mask, MVS mask, limb share of the frame, card outlines found), 0/1 uint8 masks.
    Thread-safe: one segmenter per thread."""
    global _local
    import threading
    from .card import detect_quads, plausible_card
    if _local is None:
        _local = threading.local()
    if not hasattr(_local, 'seg'):
        _local.mp, _local.seg = _segmenter()
    m = limb_mask(bgr, seg)
    # SfM: eroded limb (silhouette edges mix in background) + any card outline (scale calibration)
    sfm = cv2.erode(m, np.ones((9, 9), np.uint8))
    mvs = m.copy()  # exact silhouette: growing it bleeds background into stereo, shrinking it loses real skin
    n_cards = 0
    for q in (detect_quads(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)) if cards else ()):
        if not plausible_card(q):
            continue
        poly = q.astype(np.int32)
        # only cards touching / overlapping the limb count (card held against the arm)
        ring = np.zeros_like(m); cv2.fillConvexPoly(ring, poly, 1)
        if (cv2.dilate(ring, np.ones((41, 41), np.uint8)) & m).any():
            cv2.fillConvexPoly(sfm, poly, 1)
            n_cards += 1
    return sfm, mvs, float(m.mean()), n_cards




def fill_holes(m):
    """The cut-out with every hole inside it filled. A limb's outline has no holes, but the colour cut-out leaves some
    where a finger or the back of the hand looks like the background (light, shine, shadow); the outline model then
    carves through the finger there (31 Oct printed hand: the ring finger cut away)."""
    m = (m > 0).astype(np.uint8)
    h, w = m.shape
    ff = np.ones((h + 2, w + 2), np.uint8)  # background, with a 1-pixel frame joining all of the photo's border
    ff[1:-1, 1:-1] = 1 - m
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)  # the background reachable from the border
    return (m | (ff[1:-1, 1:-1] == 1)).astype(np.uint8)


def build_masks(image_dir, out_sfm, out_mvs, log=print, cards=True, cached=None, seg=None):
    """Write COLMAP feature masks (<name>.png, limb + card candidates) and MVS masks (limb). Frames where the
    card is confirmed in 3D are later excluded from stereo by the pipeline (see card_view_names).
    Runs on all cores (one segmenter per thread: identical masks). With cards=False (photos carry AR tracking,
    so no card is needed for scale) the card search is skipped."""
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    Path(out_sfm).mkdir(parents=True, exist_ok=True)
    Path(out_mvs).mkdir(parents=True, exist_ok=True)

    def one(p):
        c = cached(p.name) if cached else None  # done while uploading (prefetch): same masks, already on disk
        if c is not None:
            shutil.copyfile(c[0], f'{out_sfm}/{p.name}.png')
            shutil.copyfile(c[1], f'{out_mvs}/{p.stem}.png')
            return c[2], c[3]
        sg = seg(p.name) if seg else None
        sfm, mvs, frac, n_cards = masks_for(cv2.imread(str(p)), cards, sg)
        if FILL:
            mvs = fill_holes(mvs)
        cover[p.name] = hand_cover(mvs, sg)
        cv2.imwrite(f'{out_sfm}/{p.name}.png', sfm * 255)
        cv2.imwrite(f'{out_mvs}/{p.stem}.png', mvs * 255)
        return frac, n_cards

    paths = sorted(Path(image_dir).glob('*.jpg'))
    cover = {}  # share of the hand's landmarks on each cut-out (hull.py leaves out photos where fingers are missing)
    with ThreadPoolExecutor(min(8, os.cpu_count() or 4)) as ex:
        res = list(ex.map(one, paths))
    import json
    (Path(out_mvs).parent / 'hand_cover.json').write_text(json.dumps(cover))
    low = sorted(n for n, c in cover.items() if c is not None and c < 0.85)
    if low:
        log(f'[segment] {len(low)} cut-outs miss part of the hand found in the photo: {", ".join(low)}')
    fracs = [c for c, _ in res]
    card_frames = [p.name for p, (_, n) in zip(paths, res) if n]
    n_cards = sum(n for _, n in res)
    log(f'[segment] {len(fracs)} masks, limb covers {100 * np.median(fracs):.0f}% of the frame (median), {n_cards} card outlines on the limb')
    return {'limb_frac_median': float(np.median(fracs)), 'card_outlines': n_cards, 'card_frames': card_frames}


def apply_mask(image_dir, mask_dir, out_dir, gray=127):
    """Copy images with everything outside the MVS mask painted flat grey (stereo ignores it)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def one(p):
        im = cv2.imread(str(p))
        m = cv2.imread(f'{mask_dir}/{p.stem}.png', cv2.IMREAD_GRAYSCALE)
        im[m < 128] = gray
        cv2.imwrite(f'{out_dir}/{p.name}', im, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        list(ex.map(one, sorted(Path(image_dir).glob('*.jpg'))))


def _image_point(meta, p, w, h):
    """World point -> (u, v) in 0..1 of the stored portrait photo, from its AR pose (camera -> world, column-major)
    and portrait intrinsics K (same convention as the app's imagePoint)."""
    P = np.asarray(meta['pose'], float)
    d = np.asarray(p, float) - P[12:15]
    x, y, z = d @ P[0:3], d @ P[4:7], -(d @ P[8:11])
    if z < 0.02:
        return None
    fx, fy, cx, cy = meta['K']
    s = w / float(meta.get('w', w))
    return [(cx + fx * y / z) * s / w, (cy + fy * x / z) * s / h]


def hand_offsets(metas, arm, dets, names):
    """Per photo (du, dv) in 0..1 image units: where the hand really is in the photo (its detected landmarks) minus
    where the arm's projection through the phone's AR pose puts it. AR tracking drifts by centimetres over a scan
    (printed hand, 31 Oct: 230-380 px on the first 12 photos), which put the cut-out's outline and seed on the
    curtains instead of the hand. The constant part (palm centre vs the axis point used) is removed; photos without
    a detected hand take the nearest photos' drift (it changes slowly)."""
    a, b = np.asarray(arm['a'], float), np.asarray(arm['b'], float)
    u = (a - b) / (np.linalg.norm(a - b) + 1e-12)  # fingertips -> elbow
    wrist = np.asarray(arm['wrist'], float) if 'wrist' in arm else b + u * 0.19
    palm = wrist - u * 0.04
    raw = {}
    for n in names:
        m, lm = metas.get(n), dets.get(n)
        if not m or 'pose' not in m or 'K' not in m or not lm or len(lm) < 18:
            continue
        w, h = m.get('stored_w', m.get('w', 1080)), m.get('stored_h', m.get('h', 1920))
        p = _image_point(m, palm, w, h)
        if p is None:
            continue
        c = np.asarray(lm, float)[[0, 5, 9, 13, 17], :2].mean(0)
        raw[n] = c - np.asarray(p)
    if len(raw) < 5:
        return {}
    bias = np.median(np.array(list(raw.values())), axis=0)
    order = [n for n in names if n in metas]
    have = [i for i, n in enumerate(order) if n in raw]
    out = {}
    for i, n in enumerate(order):
        near = sorted(have, key=lambda j: abs(j - i))[:5]  # median of the 5 nearest photos in capture order
        out[n] = np.median(np.array([raw[order[j]] for j in near]), axis=0) - bias
    return out


def shift_seg(seg, off):
    """seg (seed + roi) moved by off (0..1 image units)."""
    if not seg or off is None:
        return seg
    dx, dy = float(off[0]), float(off[1])
    out = dict(seg)
    if seg.get('seed'):
        out['seed'] = [seg['seed'][0] + dx, seg['seed'][1] + dy]
    if seg.get('roi'):
        out['roi'] = [[x + dx, y + dy] for x, y in seg['roi']]
    return out


def arm_seg(meta, arm, w, h):
    """seed + outline ('roi') of the placed arm cylinder {a, b, r} in one photo (see limb_mask)."""
    a, b, r = np.asarray(arm['a'], float), np.asarray(arm['b'], float), float(arm['r'])
    ax = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    e1 = np.cross(ax, [0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.cross(ax, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    roi = []
    L = np.linalg.norm(b - a)
    hand = min(0.2, L / 2)  # the hand end (b): spread fingers and the thumb reach well past the palm's width
    for t in np.linspace(0, 1, 12):
        c = a + (b - a) * t
        rr = r + (0.045 if (1 - t) * L < hand + 0.02 else 0.02)
        for k in range(12):
            q = c + rr * (np.cos(k * np.pi / 6) * e1 + np.sin(k * np.pi / 6) * e2)
            uv = _image_point(meta, q, w, h)
            if uv:
                roi.append(uv)
    u = (a - b) / (L + 1e-12)  # fingertips -> elbow
    seed = _image_point(meta, b + u * min(0.09, L / 2), w, h) or _image_point(meta, (a + b) / 2, w, h)
    wrist = np.asarray(arm['wrist'], float) if 'wrist' in arm else b + u * min(0.19, L / 2)
    along = [b + u * min(0.06, L / 3), b + u * min(0.09, L / 2), wrist] + [wrist + u * d for d in (0.04, 0.08, 0.12)
                                                                           if (wrist + u * d - a) @ u < 0]
    pts = [q for q in (_image_point(meta, p, w, h) for p in along) if q and 0.01 < q[0] < 0.99 and 0.01 < q[1] < 0.99]
    # (the cut-out along these points spreads to whatever the axis runs into: a suitcase under a printed model,
    # a shirt behind the arm; measured worse on every validated scan, so they are not used: 'pts' stays off)
    return {'mode': 'auto', 'seed': seed, 'roi': roi} if seed else None
