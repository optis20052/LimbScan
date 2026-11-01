"""Isolate the limb from the reconstructed scene, build clean + watertight meshes, measure, export."""
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from .measure import measure

TABLE_CLEAR_MM = 12.0   # everything closer than this to the marker plane is table/mat


QUALITY_GATE = os.environ.get('LIMBSCAN_QUALITY_GATE', '1') == '1'
QUALITY_MIN = float(os.environ.get('LIMBSCAN_QUALITY_MIN', '0.9'))


def _cameras(dense):
    import pycolmap
    rec = pycolmap.Reconstruction(Path(dense) / 'sparse')
    C, D = [], []
    for im in rec.images.values():
        T = im.cam_from_world()
        R = T.rotation.matrix()
        C.append(T.inverse().translation); D.append(R[2])
    return np.array(C), np.array(D)


def focus_point(C, D):
    """Least-squares point closest to all optical axes."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, d in zip(C, D):
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ c
    return np.linalg.solve(A, b)


def isolate(points, colors, normals, C, D, log=print, table=True):
    f = focus_point(C, D)
    R = 0.9 * np.median(np.linalg.norm(C - f, axis=1))
    if table:
        m = (points[:, 2] > TABLE_CLEAR_MM) & (np.linalg.norm(points - f, axis=1) < R)
    else:  # arm mode: background was masked out before stereo; just drop far-away strays
        m = np.linalg.norm(points - np.median(points, 0), axis=1) < 2.5 * R
    log(f'[isolate] focus {f.round(1)}, radius {R:.0f} mm, {m.sum()}/{len(points)} candidate points')
    P = points[m]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    ds = pc.voxel_down_sample(3.0)
    lab = np.array(ds.cluster_dbscan(eps=7.0, min_points=6))
    if lab.max() < 0:
        raise RuntimeError('Could not find the limb in the reconstruction.')
    big = np.bincount(lab[lab >= 0]).argmax()
    keep_v = np.asarray(ds.points)[lab == big]
    d, _ = cKDTree(keep_v).query(P, distance_upper_bound=5.0)
    sel = np.where(m)[0][np.isfinite(d)]
    log(f'[isolate] limb cluster: {len(sel)} points')
    if len(sel) < 30000:
        raise RuntimeError('The arm was not reconstructed: it most likely moved during the scan (the room and card were '
                           'reconstructed, the arm was not). Support the arm so it cannot sway, e.g. let it rest on the '
                           'table or a pillow, and scan again.')
    return sel, f


def _largest_component(mesh):
    parts = mesh.split(only_watertight=False)
    return max(parts, key=lambda p: p.area) if len(parts) else mesh


def _robust_planes(P, Q, k):
    """For each query point: robust local plane (centre c, normal n, median height) from its k neighbours in P
    (compiled, multi-core; identical to _robust_planes_np)."""
    from .fastgeo import robust_planes
    _, idx = cKDTree(P).query(Q, k=k, workers=-1)
    return robust_planes(np.ascontiguousarray(P, dtype=np.float64), idx)


def _robust_planes_np(P, Q, k, chunk=20000):
    """Reference numpy version of _robust_planes."""
    tree = cKDTree(P)
    C = np.empty_like(Q); Nn = np.empty_like(Q); H = np.empty(len(Q))
    for a in range(0, len(Q), chunk):
        q = Q[a:a + chunk]
        _, idx = tree.query(q, k=k)
        nb = P[idx]
        w = np.ones(nb.shape[:2])
        for _ in range(3):  # refit on the skin points only: hairs would tilt a plain PCA plane
            ws = w.sum(1, keepdims=True)
            c = (nb * w[:, :, None]).sum(1) / ws
            X = nb - c[:, None]
            cov = np.einsum('mk,mki,mkj->mij', w, X, X) / ws[:, :, None]
            n = np.linalg.eigh(cov)[1][:, :, 0]
            off = np.einsum('mki,mi->mk', X, n)
            med = np.median(off, axis=1)
            w = (np.abs(off - med[:, None]) < 0.6).astype(float) + 1e-3
        C[a:a + chunk], Nn[a:a + chunk], H[a:a + chunk] = c, n, med
    return C, Nn, H


def skin_surface(P, k_wide=120, k=48, hair_mm=0.8):
    """Hair-robust skin. (1) In a wide neighbourhood skin points are the clear majority: drop points more than
    `hair_mm` off the robust skin plane (hairs, stray strands). (2) Project the rest onto the robust local
    plane (median height) -> smooth skin without the outward bias a mean / least-squares fit would get."""
    C, Nn, H = _robust_planes(P, P, k_wide)
    own = np.einsum('mi,mi->m', P - C, Nn) - H
    keep = np.abs(own) < hair_mm
    P = P[keep]
    if os.environ.get('LIMBSCAN_SKIN_FIT', 'plane') == 'quadric':
        # robust local quadric instead of a plane: a plane sits below a convex surface's apex (r^2 / 4R, ~0.15 mm
        # at the wrist), which shrank girths; the quadric follows the curvature
        from .fastgeo import quadric_offsets
        P = np.ascontiguousarray(P, dtype=np.float64)
        _, nn = cKDTree(P).query(P, k=k, workers=-1)
        n, f = quadric_offsets(P, np.arange(len(P)), P, nn)
        return P + n * f[:, None], keep
    C, Nn, H = _robust_planes(P, P, k)
    own = np.einsum('mi,mi->m', P - C, Nn) - H
    return P - Nn * own[:, None], keep


def build_meshes(job, limb_pts, limb_col, C):
    """Screened-Poisson watertight mesh from the isolated dense limb cloud."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(limb_pts))
    pc.colors = o3d.utility.Vector3dVector(limb_col / 255.0)
    pc = pc.voxel_down_sample(0.7)
    pc, _ = pc.remove_statistical_outlier(20, 2.0)
    if os.environ.get('LIMBSCAN_SKIN_SMOOTH', '1') == '1':
        Ps, keep = skin_surface(np.asarray(pc.points))
        cols = np.asarray(pc.colors)[keep]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Ps)); pc.colors = o3d.utility.Vector3dVector(cols)
        job.log(f'[mesh] hair-robust skin surface: {(~keep).sum()} hair/stray points removed, rest projected to skin')
    # surfaces have ~60 neighbours within 3 mm at 0.7 mm spacing, noise strands/spikes ~10: drop them
    pc, _ = pc.remove_radius_outlier(nb_points=22, radius=3.0)
    lab = np.array(pc.cluster_dbscan(eps=2.5, min_points=8))
    if lab.max() >= 0:
        pc = pc.select_by_index(np.where(lab == np.bincount(lab[lab >= 0]).argmax())[0])
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=40))
    from .fastgeo import orient_normals
    Q = np.asarray(pc.points)
    N = orient_normals(Q, np.asarray(pc.normals))  # consistent signs (open3d's tangent-plane MST is 12x slower)
    pc.normals = o3d.utility.Vector3dVector(N)
    _, ci = cKDTree(C).query(Q)  # global sign: surface faces the cameras
    if (((C[ci] - Q) * N).sum(1) > 0).mean() < 0.5:
        pc.normals = o3d.utility.Vector3dVector(-N)
    job.log(f'[mesh] {len(Q)} oriented points')
    # single-threaded: the multi-threaded solver sums in a different order each run, so identical photos would not
    # give an identical mesh (a few seconds slower, but repeatable)
    pm, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=int(os.environ.get('LIMBSCAN_POISSON_DEPTH', 8)), scale=1.1, linear_fit=True, n_threads=1)
    tree = cKDTree(Q)
    closed = _largest_component(trimesh.Trimesh(np.asarray(pm.vertices), np.asarray(pm.triangles)))
    w = trimesh.Trimesh(np.asarray(pm.vertices), np.asarray(pm.triangles))
    d, _ = tree.query(w.vertices)
    w.update_faces((d < 10.0)[w.faces].all(1)); w.remove_unreferenced_vertices()
    w = _largest_component(w)
    # "surface" = only where real data exists (no hole filling), for inspection
    surf = w.copy()
    d, _ = tree.query(surf.vertices)
    surf.update_faces((d < 1.5)[surf.faces].all(1)); surf.remove_unreferenced_vertices()
    cols = np.asarray(pc.colors)
    for m in (w, surf):
        _, idx = tree.query(m.vertices)
        m.visual.vertex_colors = np.c_[cols[idx] * 255, np.full(len(idx), 255)].astype(np.uint8)
    job.log(f'[mesh] {len(w.faces)} faces, watertight={w.is_watertight}')
    return surf, w, closed


def _quadric_denoise(V, mask, k=110, iters=4, chunk=8000, support_mm=2.0):
    """Robust local quadric fit (z = ax2+bxy+cy2+dx+ey+f in each point's tangent frame, Huber-reweighted) and
    projection of the point onto it. Removes bumps a few mm wide (hair clumps, stereo noise) while keeping the
    limb's curvature, so girths don't shrink the way a plane-based smoother would make them."""
    from .fastgeo import quadric_offsets
    V = np.array(V, dtype=np.float64)
    idx_all = np.where(mask > 0.05)[0]
    for _ in range(iters):
        # support set at ~2 mm spacing, so k neighbours always span a fixed ~12 mm patch, whatever the mesh density
        sup = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V)).voxel_down_sample(support_mm).points)
        _, nn = cKDTree(sup).query(V[idx_all], k=k, workers=-1)
        n, f = quadric_offsets(V, idx_all, sup, nn)  # compiled, multi-core; same maths as _quadric_denoise_np
        V[idx_all] = V[idx_all] + n * (f * mask[idx_all])[:, None]
    return V


def _quadric_denoise_np(V, mask, k=110, iters=4, chunk=8000, support_mm=2.0):
    """Reference numpy version of _quadric_denoise."""
    V = V.copy()
    idx_all = np.where(mask > 0.05)[0]
    for _ in range(iters):
        sup = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V)).voxel_down_sample(support_mm).points)
        tree = cKDTree(sup)
        for a in range(0, len(idx_all), chunk):
            ids = idx_all[a:a + chunk]
            q = V[ids]
            _, nn = tree.query(q, k=k)
            X = sup[nn] - q[:, None]                     # neighbours relative to the point
            cov = np.einsum('mki,mkj->mij', X, X)
            ev = np.linalg.eigh(cov)[1]                  # columns: tangent u, tangent v, normal
            u, v, n = ev[:, :, 2], ev[:, :, 1], ev[:, :, 0]
            xu = np.einsum('mki,mi->mk', X, u); xv = np.einsum('mki,mi->mk', X, v); z = np.einsum('mki,mi->mk', X, n)
            A = np.stack([xu * xu, xu * xv, xv * xv, xu, xv, np.ones_like(xu)], -1)   # (m, k, 6)
            w = np.ones_like(z)
            for _ in range(6):  # iterative hard rejection: bumps are dropped from the fit entirely
                Aw = A * w[:, :, None]
                coef = np.linalg.solve(np.einsum('mki,mkj->mij', Aw, A) + 1e-6 * np.eye(6),
                                       np.einsum('mki,mk->mi', Aw, z)[:, :, None])[:, :, 0]
                res = z - np.einsum('mki,mi->mk', A, coef)
                sc = 1.4826 * np.median(np.abs(res), axis=1, keepdims=True) + 1e-6
                w = (np.abs(res) < 2.5 * sc).astype(float) + 1e-4
            V[ids] = q + n * (coef[:, 5] * mask[ids])[:, None]
    return V


def _smooth_forearm(m, cl, wrist_s, iters=int(os.environ.get("LIMBSCAN_FOREARM_SMOOTH", 40)), lam=0.5, mu=-0.53, ramp=25.0):
    """Taubin smoothing restricted to the forearm/elbow skin (proximal to the wrist), which is smooth at the
    5-10 mm scale where hair clumps and stereo noise make bumps; the hand and fingers keep their detail.
    Same triangles -> stays watertight; Taubin barely changes volume."""
    if wrist_s is None or len(cl) < 2:
        return
    import scipy.sparse as sp
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))]
    _, ci = cKDTree(cl).query(m.vertices)
    wgt = np.clip((arc[ci] - (wrist_s - 10.0)) / ramp, 0, 1)  # 0 on the hand, ramps to 1 above the wrist
    if wgt.max() == 0:
        return
    e = m.edges_unique
    n = len(m.vertices)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    W = sp.diags(1.0 / deg) @ A
    V = m.vertices.copy()
    V = _quadric_denoise(V, wgt)          # remove 3-10 mm bumps, keep curvature
    for i in range(iters):                # then a light Taubin pass to even out what is left
        f = lam if i % 2 == 0 else mu
        V = V + (f * wgt)[:, None] * (W @ V - V)
    m.vertices = V


def clean_model(job, closed, surf, cl, faces=400000, wrist_s=None):
    """Closed Poisson solid -> flat cut across the limb at the proximal end of the measured centreline
    (removes the ragged elbow end / Poisson balloon), capped, decimated, lightly smoothed. Watertight."""
    m = decimate(closed, faces * 2) if len(closed.faces) > faces * 2 else closed
    wt_steps = [('poisson', m.is_watertight)]
    from .measure import extend_past_elbow
    ext = extend_past_elbow(surf, cl)
    if len(ext) and len(cl) >= 6:
        # the extension may follow the elbow's bend, but never turn back toward the hand: on a real arm it once did,
        # the cuts below then faced the wrong way and the exported model kept only the elbow end (1 Nov, scan a065)
        t_cl = cl[-1] - cl[-6]; t_cl /= np.linalg.norm(t_cl)
        steps = np.diff(np.vstack([cl[-1:], ext]), axis=0)
        fwd = (steps @ t_cl) > 0.5 * np.linalg.norm(steps, axis=1)
        bad = np.where(~fwd)[0]
        if len(bad):
            job.log(f'[model] extension past the measured limb turned back after {bad[0] * 5} mm: stopped there')
            ext = ext[:bad[0]]
    full_cl = np.vstack([cl, ext]) if len(ext) else cl
    job.log(f'[model] limb followed {len(ext) * 5} mm past the measured segment (elbow / upper arm)')
    if len(full_cl) >= 6:
        # the line followed along the limb can stop short of the end of the real surface (31 Oct, a real arm: 55 mm
        # short - the exported model lost that much forearm): continue it straight to where the real data ends, so
        # both cuts below happen there
        t = full_cl[-1] - full_cl[-6]; t /= np.linalg.norm(t)
        S = np.asarray(surf.vertices)
        rel = S - full_cl[-1]
        along = rel @ t
        near = np.linalg.norm(rel - along[:, None] * t, axis=1) < 80.0
        if (near & (along > 0)).sum() > 200:
            reach = float(np.percentile(along[near & (along > 0)], 99))
            if reach > 15.0:
                job.log(f'[model] real surface continues {reach:.0f} mm past the followed line: line extended')
                full_cl = np.vstack([full_cl, full_cl[-1] + np.arange(5.0, reach + 1e-6, 5.0)[:, None] * t])
    if len(full_cl) >= 6:
        t = full_cl[-1] - full_cl[-6]; t /= np.linalg.norm(t)
        if len(cl) >= 6:
            t_cl = cl[-1] - cl[-6]; t_cl /= np.linalg.norm(t_cl)
            if t @ t_cl < 0.5:  # the cut always faces away from the hand
                t = t_cl
        origin = full_cl[-1] - 8.0 * t
        try:
            cut = m.slice_plane(origin, -t, cap=True)
            if cut is not None and len(cut.faces) > 1000:
                m = _largest_component(cut)
            wt_steps.append(('cut', m.is_watertight))
        except Exception as e:
            job.log(f'[model] proximal cut skipped: {e}')
    # the Poisson solid closes the open proximal end with an invented 'balloon': cut straight across the limb
    # just before the first place where the surface leaves the real data, and cap it flat
    try:
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(full_cl, axis=0), axis=1))]
        _, ci = cKDTree(full_cl).query(m.vertices)
        dd, _ = cKDTree(surf.vertices).query(m.vertices)
        end_zone = arc[ci] > arc[-1] - 80.0
        balloon = end_zone & (dd > 4.0)
        if balloon.sum() > 20:
            # never into the measured limb: a few stray points near the end moved this cut 55 mm into the forearm
            # (31 Oct, a real arm: same measurements, but the exported model lost its forearm)
            a0 = max(np.percentile(arc[ci][balloon], 2) - 5.0, arc[len(cl) - 1])
            k = int(np.clip(np.searchsorted(arc, a0), 3, len(full_cl) - 2))
            t = full_cl[k + 1] - full_cl[k - 3]; t /= np.linalg.norm(t)
            if len(cl) >= 6:
                t_cl = cl[-1] - cl[-6]; t_cl /= np.linalg.norm(t_cl)
                if t @ t_cl < 0.5:
                    t = t_cl
            cut = m.slice_plane(full_cl[k], -t, cap=True)
            if cut is not None and len(cut.faces) > 1000:
                cut = _largest_component(cut)
                if cut.is_watertight:
                    m = cut
        wt_steps.append(('end cut', m.is_watertight))
    except Exception as e:
        job.log(f'[model] end cut skipped: {e}')
    trimesh.smoothing.filter_taubin(m, iterations=10)  # in place: same triangles, so it stays closed
    _smooth_forearm(m, full_cl, wrist_s)
    wt_steps.append(('smooth', m.is_watertight))
    trimesh.repair.fill_holes(m)
    m = decimate_plain(m, faces)
    wt_steps.append(('decimate', m.is_watertight))
    job.log(f'[model] watertight by step: {wt_steps}')
    _, idx = cKDTree(surf.vertices).query(m.vertices)
    m.visual.vertex_colors = surf.visual.vertex_colors[idx]
    job.log(f'[model] clean model: {len(m.faces)} faces, watertight={m.is_watertight}, volume {m.volume / 1000:.0f} ml')
    return m


def decimate_plain(mesh, faces):
    if len(mesh.faces) <= faces:
        return mesh
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    o = o.simplify_quadric_decimation(faces)
    return trimesh.Trimesh(np.asarray(o.vertices), np.asarray(o.triangles), process=True)


def _display_mesh(wt, surf, faces=150000):
    """What the clinician sees: the watertight surface trimmed to 4 mm around real data (no frayed membranes),
    small fragments dropped, lightly smoothed. Measurements and exports are unaffected."""
    m = decimate(wt, faces * 2)
    d, _ = cKDTree(surf.vertices).query(m.vertices)
    m.update_faces((d < 4.0)[m.faces].all(1)); m.remove_unreferenced_vertices()
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        big = max(p.area for p in parts)
        m = trimesh.util.concatenate([p for p in parts if p.area > 0.05 * big])
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    o.vertex_colors = o3d.utility.Vector3dVector(m.visual.vertex_colors[:, :3] / 255.0)
    o = o.filter_smooth_taubin(number_of_iterations=40)  # display only: hides sub-mm stereo noise
    return decimate(trimesh.Trimesh(np.asarray(o.vertices), np.asarray(o.triangles),
                                    vertex_colors=(np.asarray(o.vertex_colors) * 255).astype(np.uint8)), faces)


def decimate(mesh, faces):
    if len(mesh.faces) <= faces:
        return mesh
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    m.vertex_colors = o3d.utility.Vector3dVector(mesh.visual.vertex_colors[:, :3] / 255.0)
    m = m.simplify_quadric_decimation(faces)
    out = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles),
                          vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8))
    return out


def _not_background(col):
    """Points coloured like the grey-painted background, or a skin/grey blend at the silhouette, are stereo
    bleed: near-neutral colour (low chroma) and close to the grey level."""
    c = col.astype(int)
    chroma = c.max(1) - c.min(1)
    near_grey = np.abs(c - 127).max(1) < 40
    return ~((chroma < 14) | (near_grey & (chroma < 30)))


def _read_cloud(path, arm):
    pcd = o3d.io.read_point_cloud(str(path))
    P = np.asarray(pcd.points)
    col = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else np.full((len(P), 3), 200, np.uint8)
    N = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(P)
    if arm:
        keep = _not_background(col)
        P, col, N = P[keep], col[keep], N[keep]
    return P, col, N


def merge_pieces(job, P, col, N, scale_stats):
    """Extra SfM pieces were placed in the metric frame from AR poses (cm-level). Lock each onto the main
    surface with trimmed point-to-plane ICP; keep only pieces that fit tightly (same limb shape)."""
    names = scale_stats.get('pieces', [])
    if not names:
        return P, col, N
    main = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(1.5)
    main.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    for name in names:
        f = job.root / f'dense_{name}' / 'scene_dense.ply'
        if not f.exists():
            continue
        Q, qc, qn = _read_cloud(f, scale_stats.get('mode') == 'arm')
        if len(Q) < 5000:
            job.log(f'[merge] {name}: too few points ({len(Q)})'); continue
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q)).voxel_down_sample(1.5)
        T = np.eye(4); r = None
        for d in (30.0, 12.0, 5.0, 2.5):
            r = o3d.pipelines.registration.registration_icp(
                src, main, d, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            T = r.transformation
        # the piece's own AR scale comes from a short path (+-2-5 %): correct it against the main surface
        rs = o3d.pipelines.registration.registration_icp(
            src, main, 2.5, T, o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
        sc = np.cbrt(np.linalg.det(rs.transformation[:3, :3]))
        if abs(sc - 1) < 0.03 and rs.fitness >= r.fitness * 0.9:
            T, r = rs.transformation, rs
            job.log(f'[merge] {name}: scale corrected by {100 * (sc - 1):+.1f} %')
        # how far the fit moved the piece's own centre (AR placement is good to a few cm: more = wrong local fit)
        qc0 = Q.mean(0)
        moved = np.linalg.norm((T[:3, :3] @ qc0 + T[:3, 3]) - qc0)
        if moved > 60.0:
            job.log(f'[merge] {name}: rejected (fit moved it {moved:.0f} mm from its AR position: wrong match)'); continue
        if r.fitness < 0.25 or r.inlier_rmse > 1.5:
            job.log(f'[merge] {name}: rejected (overlap {r.fitness:.2f}, rmse {r.inlier_rmse:.2f} mm)'); continue
        Qt = (T[:3, :3] @ Q.T).T + T[:3, 3]
        # only add what is new: points farther than 2 mm from the main surface
        d, _ = cKDTree(np.asarray(main.points)).query(Qt, distance_upper_bound=2.0)
        new = ~np.isfinite(d)
        job.log(f'[merge] {name}: fitted (overlap {r.fitness:.2f}, rmse {r.inlier_rmse:.2f} mm, shift {moved:.1f} mm), +{new.sum()} new points')
        P = np.vstack([P, Qt[new]]); col = np.vstack([col, qc[new]]); N = np.vstack([N, qn[new]])
    return P, col, N


def finalize(job, dense, scale_stats):
    dense = Path(dense)
    out = job.root / 'result'
    out.mkdir(exist_ok=True)
    pcd = o3d.io.read_point_cloud(str(dense / 'scene_dense.ply'))
    P = np.asarray(pcd.points)
    col = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else np.full((len(P), 3), 200, np.uint8)
    N = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(P)
    if scale_stats.get('mode') == 'arm':  # stereo bleed from the grey-painted background at the silhouette
        keep = _not_background(col)
        job.log(f'[isolate] dropped {(~keep).sum()} background-coloured points')
        P, col, N = P[keep], col[keep], N[keep]
    P, col, N = merge_pieces(job, P, col, N, scale_stats)
    C, D = _cameras(dense)
    sel, focus = isolate(P, col, N, C, D, log=job.log, table=scale_stats.get('mode') != 'arm')
    surf, full, closed = build_meshes(job, P[sel], col[sel], C)
    surf = decimate(surf, 600000)
    masks = job.root / 'colmap' / 'mask_mvs'
    if scale_stats.get('mode') == 'arm' and masks.exists() and os.environ.get('LIMBSCAN_HULL', '1') == '1':
        # the shape from the limb's outlines in every photo (texture-free, closed), stereo only for hollows:
        # stereo alone is lumpy and full of holes on smooth skin / plain surfaces
        from .hull import limb_hull
        try:
            h = limb_hull(job.root / 'colmap' / 'metric', masks, surf, closed, log=job.log)
        except Exception as e:  # noqa: BLE001
            job.log(f'[hull] failed ({e}): keeping the stereo surface')
            h = None
        if h is None and QUALITY_GATE and (job.root / 'colmap' / 'masks_tracked').exists():
            # the photos do not agree on one shape: the limb moved or the fingers changed during the scan. A model
            # from the stereo surface alone would look plausible and be wrong - say so instead (medical use)
            raise RuntimeError('The hand moved or the fingers changed position during the scan: the photos do not '
                               'show one shape. Scan again with the elbow resting on a table, the fingers spread and '
                               'kept completely still.')
        model = None
        if h is not None:
            from . import shape
            if shape.available() and shape.CAMFIX:
                # cameras lined up with the outlines first, then the outline model carved again with them
                try:
                    model = shape.fix_cameras(job, h, log=job.log)
                    if model is not None:
                        h2 = limb_hull(model, masks, surf, closed, log=job.log)
                        if h2 is not None:
                            h = h2
                        else:
                            model = None
                except Exception as e:  # noqa: BLE001
                    job.log(f'[camfix] failed ({e}): keeping the cameras')
                    model = None
        if h is not None:
            surf = closed = h
            # the hollows the outlines cannot see (palm, knuckles, thumb web) from the torch shading, on the GPU
            from . import shape
            if shape.available():
                try:
                    r = shape.refine(job, h, log=job.log, model=model)
                except Exception as e:  # noqa: BLE001
                    job.log(f'[shape] failed ({e}): keeping the outline hull')
                    r = None
                if r is not None:
                    h.export(out / 'limb_hull.ply')
                    surf = closed = r
    surf.export(out / 'limb_surface.ply')
    if QUALITY_GATE and (job.root / 'colmap' / 'masks_tracked').exists():
        from .quality import photo_agreement
        med, per = photo_agreement(job.root / 'colmap' / 'metric', masks, surf)
        if med is not None:
            job.log(f'[quality] the model matches its photos: median overlap {med:.3f}, worst {min(per.values()):.3f}')
            if med < QUALITY_MIN:
                raise RuntimeError(f'The model does not match its own photos (overlap {med:.2f}, good scans 0.95+): the '
                                   'hand moved or the fingers changed position during the scan. Scan again with the '
                                   'elbow resting on a table, the fingers spread and kept completely still.')
    # measure on real data only; unseen arcs are completed per section (see measure._complete_polar)
    tip = axis = None
    lmf, model = job.root / 'hand_landmarks.json', job.root / 'colmap' / 'metric'
    if lmf.exists() and model.exists():
        # the hand triangulated with the model's own cameras (mm, the mesh's frame): measure along the real hand ->
        # forearm axis, distal end at the fingertips
        from .handarm import arm_in_model
        try:
            arm = arm_in_model(model, json.loads(lmf.read_text()), log=job.log)
        except Exception as e:  # noqa: BLE001
            job.log(f'[arm] triangulation in the model failed: {e}')
            arm = None
        if arm is not None:
            tip = np.asarray(arm['b'], float)
            axis = (np.asarray(arm['wrist'], float), np.asarray(arm['a'], float) - np.asarray(arm['b'], float))
    meas = measure(surf, tip=tip, axis=axis)
    if meas['length_mm'] < 120:
        raise RuntimeError(f"Only a small part of the limb could be reconstructed ({meas['length_mm']:.0f} mm long). "
                           'Usually the arm or hand moved during the scan, or the light was behind the arm. '
                           'Scan again: arm still and supported, light in front of the arm, shorter capture.')
    if 'wrist' not in meas['landmarks']:
        raise RuntimeError(f"The limb was reconstructed ({meas['length_mm']:.0f} mm) but the wrist could not be found. "
                           'Make sure the whole hand and the wrist are in the scan, fingers relaxed and still.')
    coverage = meas.get('section_seen_pct') or 0.0
    cl = meas.pop('_centreline'); meas.pop('_tangents')
    np.save(out / 'centreline.npy', cl)
    # the exported model: closed, watertight, cut flat at the proximal end of the measured limb (like clinical
    # scanners), lightly smoothed, with real photo colour
    wt = clean_model(job, closed, surf, cl, wrist_s=meas['landmarks'].get('wrist', {}).get('s_mm'))
    wt.export(out / 'limb.ply')
    wt.export(out / 'limb.stl')
    wt.export(out / 'limb.obj')
    disp = decimate(wt, 150000)
    try:
        from .texture import colorize
        model, masks = job.root / 'colmap' / 'metric', job.root / 'colmap' / 'mask_mvs'
        if model.exists() and masks.exists() and (job.root / 'images').exists():
            # the full-size photos (the stereo copies are small and grey-painted around the limb)
            disp = colorize(disp, model, job.root / 'images', masks, log=job.log)
        else:
            disp = colorize(disp, dense / 'sparse', dense / 'images', log=job.log)
    except Exception as e:
        job.log(f'[texture] skipped: {e}')
    seen = disp.metadata.pop('seen', None)  # (not into the exported files)
    disp.export(out / 'limb.glb')
    disp.export(out / 'limb_textured.ply')
    if seen is not None:
        from .texture import evidence
        ev, meas['evidence'] = evidence(disp, seen)
        ev.export(out / 'evidence.glb')
        job.log(f"[evidence] {meas['evidence']['pct_well_seen']}% of the surface seen head-on by 4+ photos, "
                f"{meas['evidence']['pct_weak']}% by 2-3, {meas['evidence']['pct_unseen']}% by fewer")
    meas['scale'] = {k: v for k, v in scale_stats.items() if k != 'edge_lengths_mm'}
    meas['mesh'] = {'faces': int(len(wt.faces)), 'watertight': bool(wt.is_watertight),
                    'surface_area_cm2': float(wt.area / 100), 'bbox_mm': wt.extents.round(1).tolist(),
                    'data_coverage_pct': coverage}
    (out / 'measurements.json').write_text(json.dumps(meas, indent=1))
    lm = meas['landmarks']
    summary = {'length_mm': round(meas['length_mm'], 1), 'volume_ml': round(meas['volume_ml'], 1),
               **{f'{k}_girth_mm': round(v['girth_mm'], 1) for k, v in lm.items()},
               'scale_check_pct': scale_stats.get('edge_vs_sheet_pct'), 'data_coverage_pct': round(coverage, 1)}
    job.log(f'[result] {summary}')
    return summary
