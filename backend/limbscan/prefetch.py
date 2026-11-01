"""Work done while the phone is still uploading: each photo is prepared, segmented and its limb features extracted
as soon as it arrives, so after "Finish" the pipeline only has to put the photos together.

The cache (<scan>/cache/<photo stem>.*) holds exactly what the pipeline itself would compute, made by the same
functions from the same files: whether a photo was prefetched or not, the result is identical. Every file is
written under a temporary name and renamed, so the pipeline never reads a half-written entry; anything missing is
simply computed by the pipeline.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np


def _cache(root):
    return Path(root) / 'cache'


def cached_image(root, photo):
    p = _cache(root) / f'{Path(photo).stem}.jpg'
    return p if p.exists() else None


def _stem(root, name):
    """Pipeline photo name (p0012.jpg) -> uploaded photo stem, from the map prepare_frames writes."""
    m = Path(root) / 'names.json'
    if not m.exists():
        return None
    return json.loads(m.read_text()).get(name)


def cached_masks(root, name):
    stem = _stem(root, name)
    if stem is None:
        return None
    c = _cache(root)
    sfm, mvs, info = c / f'{stem}.sfm.png', c / f'{stem}.mvs.png', c / f'{stem}.seg.json'
    if not (sfm.exists() and mvs.exists() and info.exists()):
        return None
    i = json.loads(info.read_text())
    return sfm, mvs, i['frac'], i['cards']


def cached_sift(root, name):
    stem = _stem(root, name)
    p = _cache(root) / f'{stem}.sift.npz' if stem else None
    if p is None or not p.exists():
        return None
    z = np.load(p)
    return z['kp'], z['desc']


def _write(path, writer):
    tmp = path.with_name('.tmp-' + path.name)
    writer(tmp)
    os.replace(tmp, path)


def process(root, photo):
    """Prepare one uploaded photo (resize, limb masks, limb SIFT) into the cache. Safe to call more than once."""
    from . import pipeline
    from .segment import masks_for
    root, photo = Path(root), Path(photo)
    if photo.suffix.lower() not in pipeline.IMAGE_EXT:
        return
    c = _cache(root)
    c.mkdir(exist_ok=True)
    stem = photo.stem
    img_p = c / f'{stem}.jpg'
    if img_p.exists() and img_p.stat().st_mtime < photo.stat().st_mtime:  # re-uploaded photo: start over
        for f in c.glob(f'{stem}.*'):
            f.unlink()
    if not img_p.exists():
        img = pipeline._read_any(photo)
        if img is None:
            return
        small = pipeline._resize_max(img, pipeline.MAX_SIDE)
        _write(img_p, lambda t: cv2.imwrite(str(t), small, [cv2.IMWRITE_JPEG_QUALITY, 95]))
    if not pipeline.ARM_MODE:
        return
    sfm_p, mvs_p, info_p = c / f'{stem}.sfm.png', c / f'{stem}.mvs.png', c / f'{stem}.seg.json'
    if not info_p.exists():
        bgr = cv2.imread(str(img_p))  # the stored JPEG, exactly as the pipeline reads it
        # photos with AR tracking need no card for scale: same rule as the pipeline
        meta = json.loads(photo.with_suffix('.json').read_text()) if photo.with_suffix('.json').exists() else {}
        sfm, mvs, frac, cards = masks_for(bgr, cards=not meta, seg=meta.get('seg'))
        _write(sfm_p, lambda t: cv2.imwrite(str(t), sfm * 255))
        _write(mvs_p, lambda t: cv2.imwrite(str(t), mvs * 255))
        _write(info_p, lambda t: t.write_text(json.dumps({'frac': frac, 'cards': cards})))
    sift_p = c / f'{stem}.sift.npz'
    if not sift_p.exists() and not pipeline.USE_GPU:  # GPU servers extract on the GPU after upload instead
        n = f'{stem}.jpg'
        feats = pipeline.sift_cropped([(n, img_p, sfm_p)], c / f'.sift-{stem}')
        kp, desc = feats[n]
        _write(sift_p, lambda t: np.savez(open(t, 'wb'), kp=kp, desc=desc))
