"""Inputs for the GPU surface refinement (limbscan/irrefine.py): every photo undistorted and cropped to the limb, its
mask, pinhole intrinsics and pose (metric model, mm), plus the starting surface. One .npz file.

Usage: python -m limbscan.irprep <scan_dir> <out.npz> [start_mesh]
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def prepare(scan, out, start=None, margin=48, model=None):
    import pycolmap
    import trimesh
    scan = Path(scan)
    rec = pycolmap.Reconstruction(str(model or scan / 'colmap' / 'metric'))
    mask_dir = scan / 'colmap' / 'mask_mvs'
    import json
    bad_file = scan / 'colmap' / 'bad_masks.json'
    bad = set(json.loads(bad_file.read_text())) if bad_file.exists() else set()
    views = {}
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        if im.name in bad:
            continue
        img = cv2.imread(str(scan / 'images' / im.name))
        m = cv2.imread(str(mask_dir / (Path(im.name).stem + '.png')), cv2.IMREAD_GRAYSCALE)
        nrm = cv2.imread(str(scan / 'moge' / (Path(im.name).stem + '.png')))  # MoGe-2 normals, when computed
        if img is None or m is None:
            continue
        cam = rec.cameras[im.camera_id]
        p = np.asarray(cam.params, float)
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
        h, w = img.shape[:2]
        if cam.width != w:  # the model was built on resized photos
            s = w / cam.width
            K[:2] *= s
        if cam.model_name == 'OPENCV':
            dist = p[4:8]
            img = cv2.undistort(img, K, dist)
            m = cv2.undistort(m, K, dist)
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        if nrm is not None and nrm.shape[:2] != (h, w):
            nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(m > 127)
        if len(xs) < 500:
            continue
        x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin + 1)
        y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin + 1)
        Kc = K.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        T = im.cam_from_world()
        views[im.name] = dict(img=img[y0:y1, x0:x1, ::-1].copy(), mask=m[y0:y1, x0:x1].copy(), K=Kc,
                              nrm=None if nrm is None else nrm[y0:y1, x0:x1, ::-1].copy(),
                              R=T.rotation.matrix(), t=np.asarray(T.translation, float), full=(w, h), off=(x0, y0))
    mesh = trimesh.load(start or scan / 'result' / 'limb.ply', process=False)
    names = sorted(views)
    np.savez(out, names=np.array(names), V=np.asarray(mesh.vertices, np.float32), F=np.asarray(mesh.faces, np.int32),
             K=np.stack([views[n]['K'] for n in names]), R=np.stack([views[n]['R'] for n in names]),
             t=np.stack([views[n]['t'] for n in names]), off=np.array([views[n]['off'] for n in names]),
             **{f'img_{i}': views[n]['img'] for i, n in enumerate(names)},
             **{f'mask_{i}': views[n]['mask'] for i, n in enumerate(names)},
             **{f'nrm_{i}': views[n]['nrm'] for i, n in enumerate(names) if views[n]['nrm'] is not None})
    return len(names)


if __name__ == '__main__':
    n = prepare(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print(f'{n} views')
