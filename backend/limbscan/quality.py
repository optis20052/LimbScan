"""Does the finished model explain its own photos? The model projected into every photo (its outline) against the
limb's cut-out there. Needs no reference scan. On good scans (printed hand, real hands with the elbow resting) the
median overlap is 0.95-0.98; when the hand moved or the fingers changed shape during the scan it falls to 0.8-0.89
(1 Nov, two scans of a hand held up in the air with the fingers moving)."""
from pathlib import Path

import cv2
import numpy as np


def photo_agreement(model_dir, mask_dir, mesh, n=200000):
    """Median and per-photo overlap (IoU) of the mesh's outline with the cut-outs."""
    import pycolmap
    import trimesh
    rec = pycolmap.Reconstruction(str(model_dir))
    P = trimesh.sample.sample_surface(mesh, n, seed=0)[0]
    out = {}
    for im in rec.images.values():
        mk = cv2.imread(str(Path(mask_dir) / (Path(im.name).stem + '.png')), cv2.IMREAD_GRAYSCALE)
        if mk is None:
            continue
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        Xc = P @ T.rotation.matrix().T + T.translation
        Xc = Xc[Xc[:, 2] > 1]
        uv = cam.img_from_cam(Xc)
        sx, sy = mk.shape[1] / cam.width, mk.shape[0] / cam.height
        u, v = (uv[:, 0] * sx).astype(int), (uv[:, 1] * sy).astype(int)
        ok = (u >= 0) & (v >= 0) & (u < mk.shape[1]) & (v < mk.shape[0])
        s = np.zeros(mk.shape, np.uint8)
        s[v[ok], u[ok]] = 1
        s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        a = mk > 127
        out[im.name] = float((a & s).sum() / max(1, (a | s).sum()))
    return (float(np.median(list(out.values()))) if out else None), out
