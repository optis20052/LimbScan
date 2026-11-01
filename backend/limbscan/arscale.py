"""Metric scale from the phone's AR tracking (WebXR/ARCore on Android, ARKit on iOS). No card needed.

Each uploaded photo carries the AR camera pose (camera-to-world, metres). The arm-only SfM gives the camera
path relative to the limb in arbitrary units; a robust similarity fit between the two paths gives mm/unit.
The limb may sway a little relative to the room (AR tracks the room): that shows up as path residual, but
over a ~1 m camera path the scale stays stable (validated on MyFit/ARKit frames: +-0.33 %).
"""
import os

import numpy as np

from .scale import umeyama

# (with the cameras started from the phone's own poses, LIMBSCAN_MAPPER=ar, the photos and the phone agree and the
# whole-path fit is used)
LOCAL = os.environ.get('LIMBSCAN_AR_LOCAL', '0' if os.environ.get('LIMBSCAN_MAPPER') == 'ar' else '1') == '1'


def ar_pose(meta):
    """Camera-to-world 4x4 (row-major, metres) from a WebXR/ARCore column-major `pose` or a row-major `matrix`."""
    if 'pose' in meta:
        return np.array(meta['pose'], float).reshape(4, 4).T  # column-major -> row-major
    return np.array(meta['matrix'], float).reshape(4, 4)


def ar_position_mm(meta):
    """Camera centre in mm."""
    return ar_pose(meta)[:3, 3] * 1000.0


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


def recover_ar(rec, metas, log=print, min_frames=12, local=None):
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
    s_path = s
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)  # (the checks on the residual stay on the path fit)
    if (LOCAL if local is None else local):
        # the scale from short moves between photos taken close together (median ratio of AR / SfM displacement):
        # AR tracking drifts slowly over a scan, and one similarity over the whole path absorbs that drift as scale
        # (printed hand, 31 Oct: drift 50-60 mm on 13 photos, whole path 3.5 % small, short moves 0.9 %)
        # (14 printed-hand scans of known size, 31 Oct: mean |error| 1.25 %, worst 4.1 %; whole path 1.39 % / 5.1 %;
        # combining both was 1.14 % on average but worse on single scans, so not used)
        order = np.argsort(names)
        r = [np.linalg.norm(dst[j] - dst[i]) / np.linalg.norm(src[j] - src[i])
             for k, i in enumerate(order) for j in order[k + 1:k + 6]
             if np.linalg.norm(dst[j] - dst[i]) > 30 and np.linalg.norm(src[j] - src[i]) > 0]
        if len(r) >= 3 * min_frames:
            s = float(np.median(r))
            ms, md = src[keep].mean(0), dst[keep].mean(0)
            t = md - s * R @ ms
    rng = np.random.default_rng(0)
    idx = np.where(keep)[0]
    boot = [umeyama(src[k], dst[k])[0] for k in (rng.choice(idx, max(len(idx) // 2, 6), replace=False) for _ in range(200))]
    spread = float(100 * np.std(boot) / s)
    extent = float(np.linalg.norm(dst[keep].max(0) - dst[keep].min(0)))
    log(f'[ar] scale {s:.4f} mm/unit from {keep.sum()}/{len(src)} frames ({100 * (s / s_path - 1):+.1f} % vs the whole-path '
        f'fit), path {extent:.0f} mm, '
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


ROT_OFFSET = float(os.environ.get('LIMBSCAN_ROT_OFFSET', '0.0028'))


def rotation_ratio(rec, metas, window=8, min_deg=3.0):
    """How much larger the photos' camera turns are than the phone's own (gyro) turns between nearby photos, rotation
    about the optical axis left out (it does not depend on the lens). On 16 printed-hand scans of known size (1 Nov) it
    tracked the size error: photos turning 2-3.5 % more than the gyro went with models 2-4 % too small (the lens the
    phone reports drifts between sessions, 1286-1320 px). None when too few photos."""
    from scipy.spatial.transform import Rotation as Ro
    ims = [i for i in sorted(rec.images.values(), key=lambda i: i.name) if i.name in metas and 'pose' in metas[i.name]]
    if len(ims) < 12:
        return None
    Rs = [i.cam_from_world().rotation.matrix() for i in ims]
    Ra = []
    for i in ims:
        M = ar_pose(metas[i.name])
        Ra.append(np.stack([M[:3, 1], M[:3, 0], -M[:3, 2]]))
    a_s, a_a = [], []
    for i in range(len(ims)):
        for j in range(i + 1, min(len(ims), i + window)):
            a_s.append(np.linalg.norm(Ro.from_matrix(Rs[j] @ Rs[i].T).as_rotvec()[:2]))
            a_a.append(np.linalg.norm(Ro.from_matrix(Ra[j] @ Ra[i].T).as_rotvec()[:2]))
    a_s, a_a = np.array(a_s), np.array(a_a)
    k = a_a > np.radians(min_deg)
    if k.sum() < 20:
        return None
    return float(np.sum(a_s[k] * a_a[k]) / np.sum(a_a[k] ** 2))
