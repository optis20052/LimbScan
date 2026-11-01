"""Compare scans with a reference model of the same hand (e.g. the printed model's STL), region by region.

Each scan is aligned to the reference (coarse search + ICP, scan -> reference, so a shorter scan is fine), then:
  acc  = scan -> reference distance (is every scanned point on the real surface?), signed: + = scan outside (fat)
  comp = reference -> scan distance, only where the scan reaches (is every real detail there, e.g. separate fingers?)
Regions by distance from the fingertips along the reference's long axis: fingers, palm, wrist/forearm.

Usage: python tools/evalref.py <reference> <scan> [<scan> ...] [--save-aligned DIR]
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation as Rot

SCALE = False
BANDS = (('fingers', 0, 85), ('palm', 85, 170), ('wrist', 170, 260))


def load(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    if m.extents.max() < 5:  # metres
        m.apply_scale(1000.0)
    return m


def pcd(P):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float)))


def align(scan, ref, F):
    A = trimesh.sample.sample_surface(scan, 30000, seed=1)[0]
    sa, tf = pcd(A), pcd(F)
    c, ca = F.mean(0), A.mean(0)
    ax = np.linalg.svd(F - c, full_matrices=False)[2][0]
    axs = np.linalg.svd(A - ca, full_matrices=False)[2][0]
    t = (F - c) @ ax
    best = None
    for end in (1, -1):
        target = F[(t * end) > np.percentile(t * end, 50)].mean(0)
        for flip in (1, -1):
            R0 = _rot_between(axs * flip, ax * end)
            for spin in range(0, 360, 30):
                R = Rot.from_rotvec(np.deg2rad(spin) * ax * end).as_matrix() @ R0
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = target - R @ ca
                r = o3d.pipelines.registration.registration_icp(sa, tf, 12, T)
                if best is None or r.fitness > best.fitness + 0.02 or (
                        abs(r.fitness - best.fitness) <= 0.02 and r.inlier_rmse < best.inlier_rmse):
                    best = r
    # also a feature-based global registration (FPFH + RANSAC): for references that cover more of the arm
    # (whole arm, bent elbow) the axis-based start above can miss
    reg = o3d.pipelines.registration
    def feat(pc):
        d = pc.voxel_down_sample(2.5)
        d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=7, max_nn=30))
        return d, reg.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=100))
    (ds, fs), (dt, ft) = feat(sa), feat(tf)
    for seed in range(3):
        o3d.utility.random.seed(seed)
        g = reg.registration_ransac_based_on_feature_matching(
            ds, dt, fs, ft, True, 6, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(6)],
            reg.RANSACConvergenceCriteria(200000, 0.999))
        r = reg.registration_icp(sa, tf, 12, g.transformation)
        if r.fitness > best.fitness + 0.02 or (abs(r.fitness - best.fitness) <= 0.02 and r.inlier_rmse < best.inlier_rmse):
            best = r
    for d in (6, 3, 1.5):
        best = o3d.pipelines.registration.registration_icp(sa, tf, d, best.transformation)
    if SCALE:  # also fit the scan's overall scale (a metric scale error shows up as this factor)
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        for d in (3, 1.5, 1.5, 1.5):
            best = o3d.pipelines.registration.registration_icp(sa, tf, d, best.transformation, est)
        print(f'  scale factor {np.cbrt(np.linalg.det(best.transformation[:3, :3])):.4f}')
    return best


def _rot_between(a, b):
    v = np.cross(a, b)
    c = a @ b
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else Rot.from_rotvec(np.pi * np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.pi * np.array([0, 1.0, 0])).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def report(ref_path, scans, save=None):
    ref = load(ref_path)
    F = trimesh.sample.sample_surface(ref, 200000, seed=0)[0]
    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(o3d.core.Tensor(np.asarray(ref.vertices, np.float32)), o3d.core.Tensor(np.asarray(ref.faces, np.uint32)))
    out = {}
    for s in scans:
        m = load(s)
        r = align(m, ref, F)
        m.apply_transform(r.transformation)
        if save:
            Path(save).mkdir(parents=True, exist_ok=True)
            m.export(Path(save) / (Path(s).parent.parent.name + '_' + Path(s).stem + '_aligned.ply'))
        S = trimesh.sample.sample_surface(m, 200000, seed=2)[0]
        sd = rs.compute_signed_distance(o3d.core.Tensor(S.astype(np.float32))).numpy()
        sd = -sd if _outside_negative(rs, ref) else sd
        # fingertip end: the reference end the scan covers
        c = F.mean(0)
        ax = np.linalg.svd(F - c, full_matrices=False)[2][0]
        if ((S - c) @ ax).mean() < 0:
            ax = -ax
        tipF = ((F - c) @ ax).max()
        dS, dF = tipF - (S - c) @ ax, tipF - (F - c) @ ax
        reach = np.percentile(dS, 99.5) - 15  # leave out the scan's cut end (a cap where the reference goes on)
        dc = np.asarray(pcd(F).compute_point_cloud_distance(pcd(S)))
        rows = {}
        for name, a, b in BANDS + (('all', 0, 1e9),):
            sel, selF = (dS >= a) & (dS < min(b, reach)), (dF >= a) & (dF < min(b, reach))
            if sel.sum() < 100:
                continue
            ad = np.abs(sd[sel])
            rows[name] = dict(n=int(sel.sum()), signed=float(np.median(sd[sel])), med=float(np.median(ad)),
                              p90=float(np.percentile(ad, 90)), w1=float((ad < 1).mean() * 100),
                              w2=float((ad < 2).mean() * 100),
                              comp2=float((dc[selF] < 2).mean() * 100) if selF.sum() else float('nan'))
        out[s] = rows
        print(f'\n{s}  (ICP fitness {r.fitness:.2f}, rmse {r.inlier_rmse:.2f} mm)')
        print(f'  {"region":8s} {"fat(+)":>7s} {"|d|med":>7s} {"p90":>6s} {"<1mm":>6s} {"<2mm":>6s} {"complete<2mm":>12s}')
        for k, v in rows.items():
            print(f'  {k:8s} {v["signed"]:+7.2f} {v["med"]:7.2f} {v["p90"]:6.2f} {v["w1"]:5.0f}% {v["w2"]:5.0f}% {v["comp2"]:11.0f}%')
    return out


def _outside_negative(rs, ref):
    """Open3D's sign convention: negative inside. Check with a point far outside."""
    far = ref.bounds[1] + 100
    return rs.compute_signed_distance(o3d.core.Tensor(far[None].astype(np.float32))).numpy()[0] < 0


if __name__ == '__main__':
    args = sys.argv[1:]
    save = None
    if '--scale' in args:
        args.remove('--scale')
        SCALE = True
    if '--save-aligned' in args:
        i = args.index('--save-aligned')
        save = args[i + 1]
        args = args[:i] + args[i + 2:]
    report(args[0], args[1:], save)
