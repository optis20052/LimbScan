"""Limb cut-outs for a whole scan with SAM 2's video tracker (runs in the `limbsam` image, GPU).

The photos of one walk round the limb are one video: the tracker is told where the limb is on a few photos (points
on the hand and forearm from the hand landmarker, found in the photo itself) and follows that same object through
every photo, with memory of the frames before. One consistent object instead of a separate colour guess per photo,
which grabbed curtains, lost fingers in shine or backlight and ran onto the body behind a real arm.

Usage: python samvideo.py <prompts.json> <image_dir> <out_dir> <checkpoint>
prompts.json: {"frames": [name, ...] (capture order), "prompts": {name: {"pos": [[x, y], ...], "neg": [...]}} (0..1)}
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


CROP = 1024  # side of the hand crops of the second pass (0: off)


def track(pred, fdir, names, prompts, w, h):
    """Masks {frame index: bool array} of the limb through a folder of <int>.jpg frames, from point prompts given in
    0..1 of (w, h) on some frames; both directions, each frame from the nearer prompt."""
    masks = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = pred.init_state(video_path=str(fdir), offload_video_to_cpu=True)
        prompted = []
        for i, n in enumerate(names):
            p = prompts.get(n)
            if not p:
                continue
            pts = [[x * w, y * h] for x, y in p['pos']] + [[x * w, y * h] for x, y in p.get('neg', [])]
            labels = [1] * len(p['pos']) + [0] * len(p.get('neg', []))
            pred.add_new_points_or_box(state, frame_idx=i, obj_id=1, points=np.array(pts, np.float32),
                                       labels=np.array(labels, np.int32))
            prompted.append(i)
        if not prompted:
            return masks, prompted
        passes = {}
        for reverse in (False, True):
            passes[reverse] = {fi: (logits[0, 0] > 0).cpu().numpy()
                               for fi, _, logits in pred.propagate_in_video(state, reverse=reverse)}
    for fi in range(len(names)):
        # frames between two prompts are reached by both passes: keep the one that came from the nearer prompt
        before = [p for p in prompted if p <= fi]
        after = [p for p in prompted if p >= fi]
        use_fwd = bool(before) and (not after or fi - before[-1] <= after[0] - fi)
        m = passes[not use_fwd].get(fi)
        if m is None:
            m = passes[use_fwd].get(fi)
        if m is not None:
            masks[fi] = m
    return masks, prompted


def sharpen_hand(pred, work, image_dir, names, prompts, hands, masks, w, h):
    """Second pass on a square crop around the hand in every photo (from its landmarks; photos without take the
    nearest photo's box), scaled to the tracker's input size: SAM 2's outline comes from a 256 x 256 grid, ~7 px
    per cell on the whole photo, which fills the 3-8 px gaps between fingers near their base (webbing in the model).
    On the crop the hand fills the grid (~2-3 px per cell). The hand's cut-out replaces the first one inside the
    crop; the forearm outside keeps the first."""
    boxes = {}
    for n in names:
        lm = hands.get(n)
        if lm and len(lm) >= 21:
            P = np.array([[p[0] * w, p[1] * h] for p in lm[:21]])
            c = (P.min(0) + P.max(0)) / 2
            side = max(320.0, 1.45 * max(P.max(0) - P.min(0)))
            boxes[n] = (c[0], c[1], side)
    have = [i for i, n in enumerate(names) if n in boxes]
    if len(have) < 3:
        return masks
    cdir = work / 'crops'
    cdir.mkdir(exist_ok=True)
    geo = {}
    for i, n in enumerate(names):
        j = min(have, key=lambda k: abs(k - i))
        cx, cy, side = boxes[names[j]]
        x0, y0 = cx - side / 2, cy - side / 2
        img = cv2.imread(str(Path(image_dir) / n))
        A = np.array([[CROP / side, 0, -x0 * CROP / side], [0, CROP / side, -y0 * CROP / side]], np.float32)
        crop = cv2.warpAffine(img, A, (CROP, CROP), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(cdir / f'{i:05d}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        geo[n] = (x0, y0, side, A)
    cp = {}
    for n, p in prompts.items():
        x0, y0, side, _ = geo[n]
        q = [[(x * w - x0) / side, (y * h - y0) / side] for x, y in p['pos']]
        q = [v for v in q if 0.01 < v[0] < 0.99 and 0.01 < v[1] < 0.99]
        if len(q) >= 3:
            cp[n] = {'pos': q}
    cm, prompted = track(pred, cdir, names, cp, CROP, CROP)
    if len(prompted) < 2:
        return masks
    out = {}
    for fi, n in enumerate(names):
        m = masks.get(fi)
        c = cm.get(fi)
        if m is None or c is None:
            out[fi] = m
            continue
        x0, y0, side, A = geo[n]
        back = cv2.warpAffine(c.astype(np.uint8), A, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255) > 0
        inside = cv2.warpAffine(np.ones((CROP, CROP), np.uint8), A, (w, h), flags=cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        inside = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        out[fi] = np.where(inside, back, m.astype(bool))
    return out


def main(prompts, image_dir, out_dir, ckpt):
    from sam2.build_sam import build_sam2_video_predictor
    t0 = time.time()
    pr = json.loads(Path(prompts).read_text())
    names = pr['frames']
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # the predictor reads a folder of <int>.jpg in order
    fdir = out.parent / 'frames'
    fdir.mkdir(exist_ok=True)
    for i, n in enumerate(names):
        link = fdir / f'{i:05d}.jpg'
        if not link.exists():
            link.symlink_to(Path(image_dir).resolve() / n)
    h, w = cv2.imread(str(Path(image_dir) / names[0])).shape[:2]
    pred = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml', ckpt, device='cuda')
    prompts = {n: p for n, p in pr['prompts'].items() if p.get('pos')}
    masks, prompted = track(pred, fdir, names, prompts, w, h)
    if not prompted:
        print('[samvideo] no prompts')
        return
    hands = pr.get('hands') or {}
    if CROP and hands:
        masks = sharpen_hand(pred, out.parent, image_dir, names, prompts, hands, masks, w, h)
    t1 = time.time()
    areas = []
    for fi, n in enumerate(names):
        m = masks.get(fi)
        if m is None:
            continue
        m = m.astype(np.uint8)
        # the limb: the biggest piece and any piece at least a tenth of it (a finger cut off by the frame edge)
        k, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if k > 2:
            a = st[1:, cv2.CC_STAT_AREA]
            m = np.isin(lab, 1 + np.where(a >= 0.1 * a.max())[0]).astype(np.uint8)
        areas.append(m.mean())
        cv2.imwrite(str(out / (Path(n).stem + '.png')), m * 255)
    print(f'[samvideo] {len(areas)} of {len(names)} photos tracked from {len(prompted)} prompted, '
          f'{t1 - t0:.0f}s tracking, limb {100 * np.median(areas):.0f}% of the frame (median)')


if __name__ == '__main__':
    main(*sys.argv[1:5])
