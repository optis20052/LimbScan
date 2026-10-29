"""Limb segmentation (MediaPipe selfie-multiclass 'body-skin' class) + scale-card regions.

Masks drive the arm-only reconstruction: features, camera poses and dense stereo come from the limb's
own skin, so the limb may sway relative to the room without breaking the scan.
"""
from pathlib import Path

import cv2
import numpy as np

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


def build_masks(image_dir, out_sfm, out_mvs, log=print):
    """Write COLMAP feature masks (<name>.png, limb + card candidates) and MVS masks (limb). Frames where the
    card is confirmed in 3D are later excluded from stereo by the pipeline (see card_view_names)."""
    from .card import detect_quads, plausible_card
    mp, seg = _segmenter()
    Path(out_sfm).mkdir(parents=True, exist_ok=True)
    Path(out_mvs).mkdir(parents=True, exist_ok=True)
    cover, cards, card_frames = [], 0, []
    for p in sorted(Path(image_dir).glob('*.jpg')):
        bgr = cv2.imread(str(p))
        m = skin_mask(seg, mp, bgr)
        # SfM: eroded limb (silhouette edges mix in background) + any card outline (scale calibration)
        sfm = cv2.erode(m, np.ones((9, 9), np.uint8))
        mvs = m.copy()  # exact silhouette: growing it bleeds background into stereo, shrinking it loses real skin
        for q in detect_quads(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)):
            if not plausible_card(q):
                continue
            poly = q.astype(np.int32)
            # only cards touching / overlapping the limb count (card held against the arm)
            ring = np.zeros_like(m); cv2.fillConvexPoly(ring, poly, 1)
            if (cv2.dilate(ring, np.ones((41, 41), np.uint8)) & m).any():
                cv2.fillConvexPoly(sfm, poly, 1)
                cards += 1
                if p.name not in card_frames:
                    card_frames.append(p.name)
        cv2.imwrite(f'{out_sfm}/{p.name}.png', sfm * 255)
        cv2.imwrite(f'{out_mvs}/{p.stem}.png', mvs * 255)
        cover.append(m.mean())
    log(f'[segment] {len(cover)} masks, limb covers {100 * np.median(cover):.0f}% of the frame (median), {cards} card outlines on the limb')
    return {'limb_frac_median': float(np.median(cover)), 'card_outlines': cards, 'card_frames': card_frames}


def apply_mask(image_dir, mask_dir, out_dir, gray=127):
    """Copy images with everything outside the MVS mask painted flat grey (stereo ignores it)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(image_dir).glob('*.jpg')):
        im = cv2.imread(str(p))
        m = cv2.imread(f'{mask_dir}/{p.stem}.png', cv2.IMREAD_GRAYSCALE)
        im[m < 128] = gray
        cv2.imwrite(f'{out_dir}/{p.name}', im, [cv2.IMWRITE_JPEG_QUALITY, 95])
