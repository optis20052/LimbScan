"""Mesh loading, unit normalisation, alignment and limb measurement utilities."""
import numpy as np
import trimesh
import open3d as o3d


def load_mm(path):
    """Load a mesh and return it in millimetres (auto-detects metres)."""
    m = trimesh.load(path, force='mesh', process=True)
    if m.extents.max() < 5.0:  # metres
        m.apply_scale(1000.0)
    elif m.extents.max() < 60.0:  # centimetres
        m.apply_scale(10.0)
    return m


def to_pcd(mesh, n=60000, seed=0):
    pts, fi = trimesh.sample.sample_surface(mesh, n, seed=seed)
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts)))
    p.normals = o3d.utility.Vector3dVector(mesh.face_normals[fi])
    return p


def pca_frame(points):
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    return c, vt  # rows = principal axes


def align(source, target, max_iter=200, trim_mm=8.0):
    """Global (PCA flips) + multi-scale point-to-plane ICP. Returns 4x4 transform source->target, fitness, rmse."""
    sp, tp = to_pcd(source), to_pcd(target)
    cs, Rs = pca_frame(np.asarray(sp.points))
    ct, Rt = pca_frame(np.asarray(tp.points))
    best = None
    flips = [np.diag(d) for d in ([1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1])]
    # different hand/wrist poses bias the PCA frame: also try spins about the limb axis and small tilts of it
    def rot(ax, deg):
        a = np.deg2rad(deg); c, s_ = np.cos(a), np.sin(a)
        return {0: np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]), 2: np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])}[ax]
    starts = [F @ rot(0, roll) @ rot(2, tilt) for F in flips for roll in (0, 60, 120, 180, 240, 300) for tilt in (-15, 0, 15)]
    for F in starts:
        R = Rt.T @ F @ Rs
        T0 = np.eye(4); T0[:3, :3] = R; T0[:3, 3] = ct - R @ cs
        T = T0
        r = o3d.pipelines.registration.registration_icp(
            sp, tp, 25.0, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15))
        if best is not None and r.fitness < 0.5 * best[2]:
            continue  # clearly worse start: don't spend the full refinement on it
        T = r.transformation
        for d in (25.0, 12.0, trim_mm):
            r = o3d.pipelines.registration.registration_icp(
                sp, tp, d, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
            T = r.transformation
        score = r.fitness - 0.01 * r.inlier_rmse
        if best is None or score > best[0]:
            best = (score, T, r.fitness, r.inlier_rmse)
    return best[1], best[2], best[3]


def surface_distances(a, b, n=80000):
    """Unsigned distances from samples of a to surface b (mm)."""
    pts, _ = trimesh.sample.sample_surface(a, n, seed=1)
    return pts, point_to_mesh(b, pts)


def point_to_mesh(mesh, pts):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(pts, dtype=np.float32))).numpy()


def limb_axis(mesh):
    """Return (origin, unit axis) of the dominant (long) axis."""
    c, R = pca_frame(mesh.vertices)
    return c, R[0]


def slice_girths(mesh, origin, axis, positions):
    """Tape-measure style girth (convex-hull perimeter of the largest section loop) at each axial position.
    Also returns section area. NaN where the section doesn't exist."""
    from shapely.geometry import MultiPoint, Polygon
    out = []
    for t in positions:
        o = origin + axis * t
        sec = mesh.section(plane_origin=o, plane_normal=axis)
        if sec is None:
            out.append((np.nan, np.nan, np.nan)); continue
        p2, _ = sec.to_planar(normal=axis) if hasattr(sec, 'to_planar') else sec.to_2D()
        # choose loop closest to the axis (the limb, not stray fragments)
        best = None
        for poly in p2.polygons_closed if len(p2.polygons_closed) else []:
            if poly is None or poly.area < 50: continue
            if best is None or poly.area > best.area: best = poly
        pts = np.vstack([np.asarray(e) for e in p2.discrete]) if best is None else np.asarray(best.exterior.coords)
        if len(pts) < 10:
            out.append((np.nan, np.nan, np.nan)); continue
        hull = MultiPoint([tuple(x) for x in pts]).convex_hull
        area = best.area if best is not None else hull.area
        out.append((hull.length, area, 1.0 if best is not None else 0.0))
    return np.array(out)
