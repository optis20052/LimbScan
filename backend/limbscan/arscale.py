"""Metric scale from the phone's AR tracking (WebXR/ARCore on Android, ARKit on iOS). No card needed.

Each uploaded photo carries the AR camera pose (camera-to-world, metres). The arm-only SfM gives the camera
path relative to the limb in arbitrary units; a robust similarity fit between the two paths gives mm/unit.
The limb may sway a little relative to the room (AR tracks the room): that shows up as path residual, but
over a ~1 m camera path the scale stays stable (validated on MyFit/ARKit frames: +-0.33 %).
"""
import numpy as np

from .scale import umeyama


def ar_position_mm(meta):
    """Camera centre in mm from a WebXR-style column-major 4x4 `pose` (or a row-major `matrix`)."""
    if 'pose' in meta:
        M = np.array(meta['pose'], float).reshape(4, 4).T  # column-major -> row-major
    else:
        M = np.array(meta['matrix'], float).reshape(4, 4)
    return M[:3, 3] * 1000.0


def intrinsics(meta, width, height):
    """(fx, fy, cx, cy) in pixels of the stored image from a WebXR projection matrix, if present."""
    if meta.get('K'):  # native app (ARCore / ARKit): exact intrinsics of the stored image
        fx, fy, cx, cy = meta['K']
        s = width / float(meta.get('w', width))
        return fx * s, fy * s, cx * s, cy * s
    p = meta.get('proj')
    if not p:
        return None
    w0, h0 = meta.get('w', width), meta.get('h', height)
    fx = p[0] * w0 / 2.0
    fy = p[5] * h0 / 2.0
    cx = w0 * (1.0 - p[8]) / 2.0
    cy = h0 * (1.0 + p[9]) / 2.0
    s = width / float(w0)
    return fx * s, fy * s, cx * s, cy * s


def recover_ar(rec, metas, log=print, min_frames=12):
    """Return (scale mm/unit, R, t, stats). R, t map SfM -> AR world (mm, gravity-aligned)."""
    src, dst, names = [], [], []
    for im in rec.images.values():
        m = metas.get(im.name)
        if m is None:
            continue
        src.append(im.cam_from_world().inverse().translation)
        dst.append(ar_position_mm(m))
        names.append(im.name)
    if len(src) < min_frames:
        raise RuntimeError(f'Only {len(src)} photos with AR tracking were reconstructed (need {min_frames}).')
    src, dst = np.array(src), np.array(dst)
    keep = np.ones(len(src), bool)
    for _ in range(4):  # trimmed refit: drop frames where tracking jumped or the limb moved most
        s, R, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        thr = max(3.0 * np.median(res[keep]), 15.0)
        new = res < thr
        if new.sum() < min_frames or (new == keep).all():
            break
        keep = new
    s, R, t = umeyama(src[keep], dst[keep])
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    rng = np.random.default_rng(0)
    idx = np.where(keep)[0]
    boot = [umeyama(src[k], dst[k])[0] for k in (rng.choice(idx, max(len(idx) // 2, 6), replace=False) for _ in range(200))]
    spread = float(100 * np.std(boot) / s)
    extent = float(np.linalg.norm(dst[keep].max(0) - dst[keep].min(0)))
    log(f'[ar] scale {s:.4f} mm/unit from {keep.sum()}/{len(src)} frames, path {extent:.0f} mm, '
        f'residual median {np.median(res[keep]):.1f} mm, scale spread {spread:.2f} %')
    stats = {
        'reference': 'phone AR tracking', 'method': 'ar',
        'scale_mm_per_unit': float(s), 'ar_frames': int(keep.sum()), 'ar_frames_total': len(src),
        'ar_path_mm': extent, 'ar_residual_median_mm': float(np.median(res[keep])),
        'ar_scale_spread_pct': spread,
        # generic QA fields used by the report
        'edge_vs_sheet_pct': spread, 'scale_sheet': None, 'scale_edge': None, 'sheet_fits': [], 'markers': 0,
        'corners': 0, 'images_with_markers': 0, 'median_reproj_px': None, 'mat_planarity_mm': None,
    }
    return s, R, t, stats
