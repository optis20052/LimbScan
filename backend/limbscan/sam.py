"""Limb masks from SAM 2 on the GPU (limbscan/samseg.py in the `limbgpu` image), replacing the colour-based
cut-out: on a beige printed hand in front of a cream sofa, or skin against skin-coloured backgrounds, the colour
cut-out bites into fingers and palm and fills the gaps between fingers; SAM follows the object's real outline.

Prompts per photo: the hand's 21 landmarks (handarm.detect) and points along the first centimetres of the forearm
(positive), inside the projected arm outline (box).
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .shape import CODE, available as _gpu
from .segment import _image_point, arm_seg

SAM_IMAGE = os.environ.get('LIMBSCAN_SAM_IMAGE', 'limbsam')
CLEAN = os.environ.get('LIMBSCAN_SAM_CLEAN', '1') == '1'  # tracked cut-outs: only the limb (see _only_limb)
CREASE = os.environ.get('LIMBSCAN_CREASE', '0') == '1'  # creases between touching fingers cut in (creases.py). OFF: better scores on 16 scans, but on some it cut nubs and mangled the palm (31 Oct 22:58 scan)
GAP_PX = int(os.environ.get('LIMBSCAN_SAM_GAP_PX', '1'))  # >1: slivers narrower than this cut out where the colour cut-out says background (off: on a real arm a strip the colour cut-out missed split the forearm, volume -60 %)  # own image: SAM 2 needs a newer PyTorch
CKPT = Path(os.environ.get('LIMBSCAN_SAM_CKPT', Path.home() / 'ls2' / 'models' / 'sam2.1_hiera_large.pt'))


def available():
    # off by default: on the printed hand SAM's outlines were no better than the colour cut-out and the hull came
    # out thicker (wrist 49 vs 42 mm deep); kept for testing with LIMBSCAN_SAM=1
    return os.environ.get('LIMBSCAN_SAM', '0') == '1' and _gpu() and CKPT.exists()


def prompts(metas, arm, dets, names, w_of):
    a, b = np.asarray(arm['a'], float), np.asarray(arm['b'], float)
    u = (a - b) / (np.linalg.norm(a - b) + 1e-12)  # fingertips -> elbow
    wrist = np.asarray(arm['wrist'], float) if 'wrist' in arm else b + u * 0.19
    out = {}
    for n in names:
        m = metas.get(n)
        if not m or 'pose' not in m or 'K' not in m:
            continue
        w, h = w_of(m)
        pos = []
        lm = dets.get(n)
        if lm:
            pos += [p for p in lm if 0.005 < p[0] < 0.995 and 0.005 < p[1] < 0.995]
        else:  # no hand found in this photo: points along the hand's axis
            pos += [q for q in (_image_point(m, b + u * d, w, h) for d in (0.06, 0.09, 0.12)) if q]
        # the forearm just above the wrist (not further: under a printed model the axis runs into its stand)
        pos += [q for q in (_image_point(m, wrist + u * d, w, h) for d in (0.02, 0.05)) if q]
        pos = [p for p in pos if 0.005 < p[0] < 0.995 and 0.005 < p[1] < 0.995]
        if not pos:
            continue
        seg = arm_seg(m, arm, w, h)
        box = None
        if seg and seg.get('roi'):
            r = np.clip(np.asarray(seg['roi'], float), 0, 1)
            box = [float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 0].max()), float(r[:, 1].max())]
        out[n] = {'pos': pos, 'box': box}
    return out


def masks(job, image_dir, arm, metas, dets, out_mvs, log=print):
    """Write SAM masks over the colour masks used for the shape (out_mvs/<stem>.png). True when done. The feature
    masks for the camera poses stay the colour ones: their looser edge keeps more features (SAM's tight outline
    placed only 29 of 56 photos on the printed hand)."""
    t0 = time.time()
    names = sorted(p.name for p in Path(image_dir).glob('*.jpg'))
    pr = prompts(metas, arm, dets, names, lambda m: (m.get('stored_w', m.get('w', 1080)), m.get('stored_h', m.get('h', 1920))))
    if len(pr) < 0.8 * len(names):
        log(f'[sam] prompts for only {len(pr)}/{len(names)} photos: keeping the colour cut-out')
        return False
    work = job.root / 'sam'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    (work / 'prompts.json').write_text(json.dumps(pr))
    cache = Path.home() / '.cache' / 'lgcache'
    cmd = ['podman', 'run', '--rm', '--device', 'nvidia.com/gpu=all', '--security-opt', 'label=disable',
           '-v', f'{cache}:/root/.cache', '-v', f'{CODE}:/code:ro', '-v', f'{Path(image_dir).resolve()}:/img:ro',
           '-v', f'{work}:/work', '-v', f'{CKPT.parent}:/models:ro', '-w', '/code', SAM_IMAGE,
           'python', 'samseg.py', '/work/prompts.json', '/img', '/work/masks', f'/models/{CKPT.name}']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('[sam]')]
    if r.returncode != 0 or not lines:
        log(f'[sam] failed ({" | ".join((r.stdout + r.stderr).strip().splitlines()[-3:])}): keeping the colour cut-out')
        return False
    log(lines[-1])
    n = 0
    for p in (work / 'masks').glob('*.png'):
        name = next((x for x in names if Path(x).stem == p.stem), None)
        if name is None:
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        old = cv2.imread(str(Path(out_mvs) / p.name), cv2.IMREAD_GRAYSCALE)
        if old is not None and old.shape != m.shape:
            m = cv2.resize(m, old.shape[::-1], interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(Path(out_mvs) / p.name), m)
        n += 1
    log(f'[sam] {n} photos cut out by SAM 2 in {time.time() - t0:.0f}s')
    return n > 0


def _only_limb(m, bgr, lm):
    """The tracked cut-out of a person's limb, without what it ran onto (1 Nov, a real hand held in front of the body:
    the face, the shirt and the other hand were cut out with it in 10 of 54 photos and left lumps in the palm). Only
    the piece(s) holding the hand found in this photo are kept; face skin and clothes (the body-part segmenter) are
    removed except around the hand itself, where a hand in front of the face stays."""
    mm = (m > 127).astype(np.uint8)
    h, w = mm.shape
    P = None
    if lm and len(lm) >= 21:
        P = np.array([[x * w, y * h] for x, y in [q[:2] for q in lm[:21]]], np.float32)
        P = P[(P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)]
        # the hand found in the photo must be the tracked one (the landmarker sometimes picks the other hand, resting
        # in the lap, when the scanned hand is edge-on in front of the face)
        if len(P) < 5 or mm[P[:, 1].astype(int), P[:, 0].astype(int)].mean() < 0.5:
            P = None
        else:
            # ... and on the main piece of the cut-out (the limb being scanned is the one nearest the phone)
            k0, lab0 = cv2.connectedComponents(mm, connectivity=8)
            if k0 > 2:
                big = 1 + int(np.argmax(np.bincount(lab0.ravel())[1:]))
                if (lab0[P[:, 1].astype(int), P[:, 0].astype(int)] == big).mean() < 0.5:
                    P = None
    if bgr is not None:
        from .segment import _classes
        other = _classes(bgr, (3, 4)) > 0.5  # face skin, clothes
        if P is not None and len(P) >= 5:
            near = np.zeros((h, w), np.uint8)
            r = int(0.35 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1])))
            cv2.fillConvexPoly(near, cv2.convexHull(P.astype(np.int32)), 1)
            near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            other &= near == 0
        other = cv2.morphologyEx((other & (mm > 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        # only large regions (a face, a shirt): small specks are the hair on a forearm read as something else
        k2, lab2, st2, _ = cv2.connectedComponentsWithStats(other, 8)
        for i in range(1, k2):
            if st2[i, cv2.CC_STAT_AREA] > 0.004 * h * w:
                mm[lab2 == i] = 0
    k, lab = cv2.connectedComponents(mm, connectivity=8)
    if k > 2:
        if P is not None and len(P):
            pi = P.astype(int)
            keep = set(lab[pi[:, 1], pi[:, 0]].tolist()) - {0}
        else:
            keep = set()
        if not keep:
            keep = {1 + int(np.argmax(np.bincount(lab.ravel())[1:]))}
        mm = np.isin(lab, list(keep)).astype(np.uint8)
    return (mm * 255).astype(np.uint8)


def video_available():
    # on: 31 Oct, 5 printed-hand scans within 2 mm of the STL 93/85/92/86/93 % (colour cut-outs) -> 97/91/94/95/90 %;
    # backlit photos and a real arm in front of the body are where the colour cut-out fails and tracking does not
    return os.environ.get('LIMBSCAN_SAM_VIDEO', '1') == '1' and _gpu() and CKPT.exists()


def video_prompts(dets, names, every=6):
    """Points on the limb in a photo every few (capture order), from the hand landmarks found in that photo: the palm's
    centre, the middle of each finger, the wrist, and the forearm continuing the hand's line past the wrist. Only
    what the photo shows: nothing from the AR placement (it drifts)."""
    out, last = {}, -every
    for i, n in enumerate(names):
        lm = dets.get(n)
        if not lm or len(lm) < 21 or i - last < every:
            continue
        P = np.array([p[:2] for p in lm[:21]], float)
        pos = [P[[0, 5, 9, 13, 17]].mean(0), P[9], P[0]]
        pos += [(P[a] + P[b]) / 2 for a, b in ((5, 6), (9, 10), (13, 14), (17, 18), (2, 3))]  # finger middles
        down = P[0] - P[9]  # wrist away from the middle knuckle: along the forearm
        pos += [P[0] + k * down for k in (0.5, 1.0)]
        pos = [p.tolist() for p in pos if 0.01 < p[0] < 0.99 and 0.01 < p[1] < 0.99]
        if len(pos) >= 4:
            out[n] = {'pos': pos}
            last = i
    return out


def video_masks(job, image_dir, dets, out_mvs, log=print):
    """Cut-outs of every photo by tracking the limb through the scan (samvideo.py), written over out_mvs/<stem>.png.
    True when done."""
    t0 = time.time()
    names = sorted(p.name for p in Path(image_dir).glob('*.jpg'))
    pr = video_prompts(dets, names)
    if len(pr) < 3:
        log(f'[samvideo] the hand was found in too few photos ({len(pr)} prompts): keeping the colour cut-out')
        return False
    work = job.root / 'samvideo'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    hands = {n: [p[:2] for p in dets[n][:21]] for n in names if dets.get(n) and len(dets[n]) >= 21}
    if os.environ.get('LIMBSCAN_SAM_CROP', '1') != '1':
        hands = {}  # (no second, high-resolution pass on the hand)
    (work / 'prompts.json').write_text(json.dumps({'frames': names, 'prompts': pr, 'hands': hands}))
    cache = Path.home() / '.cache' / 'lgcache'
    cmd = ['podman', 'run', '--rm', '--device', 'nvidia.com/gpu=all', '--security-opt', 'label=disable',
           '-v', f'{cache}:/root/.cache', '-v', f'{CODE}:/code:ro', '-v', f'{Path(image_dir).resolve()}:/img:ro',
           '-v', f'{work}:/work', '-v', f'{CKPT.parent}:/models:ro', '-w', '/code', SAM_IMAGE,
           'python', 'samvideo.py', '/work/prompts.json', '/img', '/work/masks', f'/models/{CKPT.name}']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('[samvideo]')]
    if r.returncode != 0 or not lines:
        log(f'[samvideo] failed ({" | ".join((r.stdout + r.stderr).strip().splitlines()[-3:])}): keeping the colour cut-out')
        return False
    log(lines[-1])
    n = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GAP_PX, GAP_PX))
    for p in (work / 'masks').glob('*.png'):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        old = cv2.imread(str(Path(out_mvs) / p.name), cv2.IMREAD_GRAYSCALE)
        if old is not None and old.shape != m.shape:
            m = cv2.resize(m, old.shape[::-1], interpolation=cv2.INTER_NEAREST)
        if CLEAN:
            name = next((x for x in names if Path(x).stem == p.stem), None)
            m = _only_limb(m, cv2.imread(str(Path(image_dir) / name)) if name else None, dets.get(name) if name else None)
        if old is not None and GAP_PX > 1:
            # SAM 2's outline comes from a 256 x 256 grid (~7 px per cell on a phone photo): the narrow gaps
            # between fingers fill in, and the model grows webbing between the fingers (31 Oct printed hand).
            # The colour cut-out (run on a crop of the hand) resolves them. Thin slivers inside SAM's outline that
            # the colour cut-out calls background are cut out again; large areas are not (that is the colour
            # cut-out failing in backlight, exactly where SAM is trusted).
            d = ((m > 127) & (old < 128)).astype(np.uint8)
            thin = d & (1 - cv2.morphologyEx(d, cv2.MORPH_OPEN, k))
            m = np.where(thin > 0, 0, m).astype(np.uint8)
        cv2.imwrite(str(Path(out_mvs) / p.name), m)
        n += 1
    log(f'[samvideo] {n} photos cut out by tracking the limb in {time.time() - t0:.0f}s')
    if CREASE and n:
        # where neighbouring fingers touch in a photo, the dark crease between them cut in as a thin gap
        from .creases import back_views, cut
        k = px = 0
        mf = job.root / 'ar_meta.json'
        # only photos of the back of the hand: on the palm side the palm's creases and the finger pads' shadows are
        # dark too and were cut in as notches (31 Oct, the palm looked 'fake'); a gap opened from the back runs
        # through the whole hand anyway
        backs = back_views(json.loads(mf.read_text()), dets) if mf.exists() else None
        for nm in [x for x in names if backs and x in backs]:
            mp = Path(out_mvs) / (Path(nm).stem + '.png')
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            lm = dets.get(nm)
            if m is None or not lm:
                continue
            img = cv2.imread(str(Path(image_dir) / nm))
            m2, c = cut(img, m, lm)
            if c:
                cv2.imwrite(str(mp), m2)
                k += 1
                px += c
        log(f'[creases] finger creases cut into {k} of {len(backs or [])} photos of the back of the hand ({px} px)')
    if n > 0:
        (Path(out_mvs).parent / 'masks_tracked').write_text(str(n))  # hull.py: no drop check on tracked cut-outs
    return n > 0
