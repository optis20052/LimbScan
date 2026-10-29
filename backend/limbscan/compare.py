"""Compare a limb scan against a reference scan (e.g. Structure Sensor) in millimetres.

Fingers/hand posture changes between sessions, so the headline metric is the forearm zone
(wrist -> wrist + 180 mm along the reference centreline), where the anatomy is rigid.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from .meshio_utils import load_mm, align, point_to_mesh
from .measure import centreline, section_at, measure

FOREARM_MM = 170.0
ARTEFACT_MM = 25.0  # a girth this far off means the section hit scan debris, not limb


def signed_dist(ref, pts):
    cp, d, tri = trimesh.proximity.closest_point(ref, pts)
    s = np.sign(((pts - cp) * ref.face_normals[tri]).sum(1))
    return d * np.where(s == 0, 1, s)


def _refine_on_zone(scan, ref, zone_pts):
    """Second ICP restricted to the rigid forearm zone of the reference."""
    sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(trimesh.sample.sample_surface(scan, 80000, seed=3)[0]))
    tp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(zone_pts))
    tp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6, max_nn=30))
    T = np.eye(4)
    for d in (10.0, 5.0, 3.0):
        r = o3d.pipelines.registration.registration_icp(
            sp, tp, d, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T = r.transformation
    return T


def compare(scan_path, ref_path, out_dir=None):
    scan, ref = load_mm(scan_path), load_mm(ref_path)
    T, fit, rmse = align(scan, ref)
    scan.apply_transform(T)
    rm = measure(ref)
    cl, tg = rm['_centreline'], rm['_tangents']  # oriented distal -> proximal
    lm = rm['landmarks']
    if 'wrist' not in lm:
        raise RuntimeError('reference has no detectable wrist')
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))]
    w = lm['wrist']['s_mm']
    zone = (arc >= w) & (arc <= w + FOREARM_MM)
    # reference zone surface points: vertices whose nearest centreline sample lies in the zone
    from scipy.spatial import cKDTree
    _, ci = cKDTree(cl).query(ref.vertices)
    zmask = zone[ci]
    T2 = _refine_on_zone(scan, ref, ref.vertices[zmask])
    scan.apply_transform(T2)
    # distances, zone only (scan points near the zone)
    sp, _ = trimesh.sample.sample_surface(scan, 150000, seed=5)
    _, si = cKDTree(cl).query(sp)
    sz = sp[zone[si]]
    sd = signed_dist(ref, sz)
    ud = np.abs(sd)
    rp, _ = trimesh.sample.sample_surface(ref, 150000, seed=6)
    _, ri = cKDTree(cl).query(rp)
    rz = rp[zone[ri]]
    back = point_to_mesh(scan, rz)  # completeness: ref zone covered by scan?
    # girths along the same planes
    girths = []
    for i in np.where(zone)[0][::2]:
        a = section_at(ref, cl[i], tg[i], near=cl[i])
        b = section_at(scan, cl[i], tg[i], near=cl[i])
        if a and b:
            girths.append({'s_from_wrist_mm': round(float(arc[i] - w), 1), 'ref_mm': round(a['girth'], 2), 'scan_mm': round(b['girth'], 2),
                           'diff_mm': round(b['girth'] - a['girth'], 2), 'ref_area': a['area'], 'scan_area': b['area']})
    n_art = sum(abs(g['diff_mm']) > ARTEFACT_MM for g in girths)
    girths = [g for g in girths if abs(g['diff_mm']) <= ARTEFACT_MM]
    gd = np.array([g['diff_mm'] for g in girths]) if girths else np.array([np.nan])
    # zone volume via section areas (5 mm spacing *2)
    step = np.median(np.diff(arc)) * 2
    vr = sum(g['ref_area'] for g in girths) * step / 1000
    vs = sum(g['scan_area'] for g in girths) * step / 1000
    for g in girths:
        g.pop('ref_area'); g.pop('scan_area')
    res = {
        'reference': Path(ref_path).name, 'scan': Path(scan_path).name,
        'zone': f'forearm: wrist -> wrist+{FOREARM_MM:.0f} mm',
        'surface_mean_abs_mm': float(ud.mean()), 'surface_median_abs_mm': float(np.median(ud)),
        'surface_p90_abs_mm': float(np.percentile(ud, 90)), 'surface_rms_mm': float(np.sqrt((sd ** 2).mean())),
        'surface_signed_mean_mm': float(sd.mean()),
        'coverage_pct_within_3mm': float(100 * (back < 3).mean()),
        'girth_mean_diff_mm': float(np.nanmean(gd)), 'girth_mean_abs_diff_mm': float(np.nanmean(np.abs(gd))),
        'girth_median_abs_diff_mm': float(np.nanmedian(np.abs(gd))),
        'girth_max_abs_diff_mm': float(np.nanmax(np.abs(gd))), 'girth_sections': len(girths), 'girth_artefact_sections': int(n_art),
        'zone_volume_ref_ml': vr, 'zone_volume_scan_ml': vs, 'zone_volume_diff_pct': 100 * (vs / vr - 1) if vr else None,
        'ref_wrist_girth_mm': lm['wrist']['girth_mm'],
        'girths': girths,
    }
    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        # heat-map coloured scan (blue=-3mm, green=0, red=+3mm)
        d_all = signed_dist(ref, scan.vertices)
        x = np.clip((d_all + 3) / 6, 0, 1)
        colors = np.c_[np.clip(2 * x - 1, 0, 1), 1 - np.abs(2 * x - 1), np.clip(1 - 2 * x, 0, 1)]
        colors[np.abs(d_all) > 10] = [0.6, 0.6, 0.6]
        h = scan.copy(); h.visual.vertex_colors = (np.c_[colors, np.ones(len(colors))] * 255).astype(np.uint8)
        h.export(out / f'heatmap_{Path(ref_path).stem}.glb')
        refc = ref.copy(); refc.visual.vertex_colors = [180, 180, 190, 255]
        refc.export(out / f'ref_{Path(ref_path).stem}.glb')
        (out / f'compare_{Path(ref_path).stem}.json').write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    r = compare(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print(json.dumps({k: v for k, v in r.items() if k != 'girths'}, indent=1))
    for g in r['girths'][::3]:
        print(g)
