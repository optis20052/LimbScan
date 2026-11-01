"""Detail pass: palm and fingers from 2D Gaussian Splatting (2DGS) on the GPU PC, merged into the outline hull.

The outline hull (limbscan/hull.py) is exact where the limb is convex, but it cannot see into hollows: a cupped
palm fills in and touching fingers grow webbing. 2DGS fits a surface to the colour and shading of all photos at once
(with our cameras, and a mask loss so the background stays empty), which does show the hollows and the gaps
between fingers. It is noisier and slightly biased, so it is only used where it is clearly deeper than the hull:

  1. dataset: undistorted, background-grey photos, principal points centred (2DGS assumes it), scene in decimetres
  2. 2DGS training on the GPU PC (ssh host LIMBSCAN_GS_HOST, ~15 min), depth maps rendered for every photo
  3. TSDF fusion of those depths, only inside each photo's cut-out -> 2DGS surface (mm)
  4. calibration: shifted along its normals onto the hull where the two agree (the convex limb)
  5. volumetric intersection: inside the hull AND within `tol` of the inside of the 2DGS surface

Measurements stay those of the hull (tape girths go round the outside and do not see hollows); the exported models
(limb.ply / .stl / .obj / .glb) become the detailed surface.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import trimesh

HOST = os.environ.get('LIMBSCAN_GS_HOST', '')
SCALE = 0.01  # mm -> decimetres: 2DGS's defaults suit scenes about one unit across


def available():
    return bool(HOST)


def prepare(job, out):
    """2DGS dataset from the scan's metric model and background-grey photos."""
    import pycolmap
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    uo = pycolmap.UndistortCameraOptions()
    uo.max_image_size = 1920
    pycolmap.undistort_images(str(out / 'und'), str(job.root / 'colmap' / 'metric'), str(job.root / 'images_masked'),
                              output_type='COLMAP', undistort_options=uo)
    d = out / 'data'
    s = d / 'sparse' / '0'
    s.mkdir(parents=True)
    shutil.move(str(out / 'und' / 'images'), str(d / 'images'))
    pycolmap.Reconstruction(str(out / 'und' / 'sparse')).write_text(str(s))
    for f in ('frames.txt', 'rigs.txt'):
        (s / f).unlink(missing_ok=True)
    shutil.rmtree(out / 'und')
    # principal point to the image centre (crop), poses to decimetres, no 2D points
    cams, lines = {}, []
    for ln in (s / 'cameras.txt').read_text().splitlines():
        if ln.startswith('#') or not ln.strip():
            continue
        p = ln.split()
        cid, W, H = int(p[0]), int(p[2]), int(p[3])
        fx, fy, cx, cy = map(float, p[4:8])
        w2 = int(2 * min(cx, W - cx)) // 2 * 2
        h2 = int(2 * min(cy, H - cy)) // 2 * 2
        x0, y0 = int(round(cx - w2 / 2)), int(round(cy - h2 / 2))
        cams[cid] = (x0, y0, w2, h2)
        lines.append(f'{cid} PINHOLE {w2} {h2} {fx!r} {fy!r} {cx - x0!r} {cy - y0!r}')
    (s / 'cameras.txt').write_text('\n'.join(lines) + '\n')
    out_lines, k = [], 0
    for ln in (s / 'images.txt').read_text().splitlines():
        if ln.startswith('#'):
            continue
        if k % 2 == 0:
            p = ln.split()
            p[5:8] = [repr(float(x) * SCALE) for x in p[5:8]]
            out_lines.append(' '.join(p))
            x0, y0, w2, h2 = cams[int(p[8])]
            im = cv2.imread(str(d / 'images' / p[9]))
            cv2.imwrite(str(d / 'images' / p[9]), im[y0:y0 + h2, x0:x0 + w2], [cv2.IMWRITE_JPEG_QUALITY, 97])
        else:
            out_lines.append('')
        k += 1
    (s / 'images.txt').write_text('\n'.join(out_lines) + '\n')
    (s / 'points3D.txt').write_text('')
    # start the Gaussians on the limb (the hull surface), not on the room's sparse points
    m = trimesh.load(job.root / 'result' / 'limb.ply')
    P, fi = trimesh.sample.sample_surface(m, 60000, seed=0)
    N = m.face_normals[fi]
    v = np.zeros(len(P), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v['x'], v['y'], v['z'] = (P * SCALE).T
    v['nx'], v['ny'], v['nz'] = N.T
    v['red'], v['green'], v['blue'] = 200, 180, 160
    hdr = ('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % len(P)
           + ''.join(f'property float {c}\n' for c in ('x', 'y', 'z', 'nx', 'ny', 'nz'))
           + ''.join(f'property uchar {c}\n' for c in ('red', 'green', 'blue')) + 'end_header\n')
    (s / 'points3D.ply').write_bytes(hdr.encode() + v.tobytes())
    return d


def train_remote(local_data, name, log=print):
    """Copy the dataset to the GPU PC, train + render depths there (gs_jobs/run.sh), fetch the depth maps."""
    remote = f'gs_jobs/{name}'
    subprocess.run(['ssh', HOST, f'rm -rf ~/{remote} && mkdir -p ~/{remote}'], check=True)
    subprocess.run(['rsync', '-a', str(local_data), f'{HOST}:~/{remote}/'], check=True)
    log(f'[detail] training 2DGS on {HOST} ...')
    r = subprocess.run(['ssh', HOST, f'~/gs_jobs/run.sh {name} && test -f ~/{remote}/depth/cams.json'], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'2DGS on {HOST} failed: {(r.stdout + r.stderr)[-500:]}')
    depth = Path(local_data).parent / 'depth'
    depth.mkdir(exist_ok=True)
    subprocess.run(['rsync', '-az', '--include=*_depth.npy', '--include=*_alpha.npy', '--include=cams.json', '--exclude=*',
                    f'{HOST}:~/{remote}/depth/', f'{depth}/'], check=True)
    subprocess.run(['ssh', HOST, f'rm -rf ~/{remote}'])
    return depth


def fuse(depth_dir, image_dir, bounds, vox=0.5):
    """TSDF fusion of the 2DGS depth maps (decimetres) inside each photo's cut-out -> closed mesh in mm."""
    from skimage import measure
    cams = json.loads((Path(depth_dir) / 'cams.json').read_text())
    trunc = 4 * vox
    lo, hi = np.asarray(bounds[0]) - 15, np.asarray(bounds[1]) + 15
    n = np.ceil((hi - lo) / vox).astype(int)
    ys, zs = lo[1] + vox * np.arange(n[1]), lo[2] + vox * np.arange(n[2])
    tsdf = np.ones(n, np.float32)
    wsum = np.zeros(n, np.float32)
    outside = np.zeros(n, np.int16)
    gy, gz = np.meshgrid(ys, zs, indexing='ij')
    views = []
    for i, c in enumerate(cams):
        d = np.load(Path(depth_dir) / f'{i:03d}_depth.npy') / SCALE
        H, W = d.shape
        im = cv2.resize(cv2.imread(str(Path(image_dir) / f"{c['name']}.jpg")), (W, H))
        fg = (np.abs(im.astype(np.int16) - 127).max(2) > 10) & (im.sum(2) > 30)
        inner = cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # the rim mixes with the background
        E = np.array(c['w2c'], float)
        E[:3, 3] /= SCALE
        views.append((d, fg, inner, E[:3, :3].astype(np.float32), E[:3, 3].astype(np.float32), c['fx'], c['fy'], W, H))
    for ix in range(n[0]):
        X = np.stack([np.full(gy.size, lo[0] + vox * ix), gy.ravel(), gz.ravel()], 1).astype(np.float32)
        acc = np.zeros(len(X), np.float32)
        w = np.zeros(len(X), np.float32)
        out_n = np.zeros(len(X), np.int16)
        for d, fg, inner, R, t, fx, fy, W, H in views:
            Pc = X @ R.T + t
            z = Pc[:, 2]
            ok = z > 50
            zz = np.where(ok, z, 1)
            u = np.round(Pc[:, 0] / zz * fx + W / 2).astype(np.int64)
            v = np.round(Pc[:, 1] / zz * fy + H / 2).astype(np.int64)
            ok &= (u >= 0) & (v >= 0) & (u < W) & (v < H)
            uu, vv = u[ok], v[ok]
            o = np.zeros(len(X), bool)
            o[ok] = ~fg[vv, uu]
            out_n += o
            dd = np.zeros(len(X), np.float32)
            dd[ok] = np.where(inner[vv, uu], d[vv, uu], 0)
            sdf = dd - z
            use = ok & (dd > 0) & (sdf > -trunc)
            acc[use] += np.clip(sdf[use] / trunc, -1, 1)
            w[use] += 1
        tsdf[ix] = np.where(w > 0, acc / np.maximum(w, 1), 1).reshape(n[1], n[2])
        wsum[ix] = w.reshape(n[1], n[2])
        outside[ix] = out_n.reshape(n[1], n[2])
    # measured where 2DGS saw the surface; empty where outlines say so; anything unseen counts as solid (the hull
    # bounds it later): a gap in the 2DGS depths must not be cut out of the limb (a flat patch in the palm)
    field = np.where(outside > 2, 1.0, np.where(wsum >= 2, tsdf, -1.0))
    V, F, _, _ = measure.marching_cubes(field, 0.0)
    m = trimesh.Trimesh(V * vox + lo, F, process=True)
    return max(m.split(only_watertight=False), key=lambda p: p.area)


def calibrate(gs, hull, log=print):
    """Shift the 2DGS surface along its normals onto the hull where the two agree (its depth sits slightly inside)."""
    V, N = gs.vertices, gs.vertex_normals
    sub = np.arange(0, len(V), 10)
    cp, d, tri = trimesh.proximity.closest_point(hull, V[sub])
    s = d * np.sign(((V[sub] - cp) * hull.face_normals[tri]).sum(1))
    near = np.abs(s) < 4.0
    off = -float(np.median(s[near]))
    log(f'[detail] 2DGS surface shifted {off:+.2f} mm onto the hull ({near.mean() * 100:.0f}% agree within 4 mm)')
    m = trimesh.Trimesh(V + N * off, gs.faces, process=False)
    trimesh.smoothing.filter_taubin(m, iterations=10)
    return m, float(near.mean())


def combine(hull, gs, tol=1.0, vox=0.5):
    """Inside the hull AND within `tol` mm of the inside of the 2DGS surface (signed distances on a grid)."""
    import open3d as o3d
    from skimage import measure

    def scene(m):
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))))
        return sc
    sh, sg = scene(hull), scene(gs)
    lo, hi = hull.bounds[0] - 3, hull.bounds[1] + 3
    n = np.ceil((hi - lo) / vox).astype(int)
    f = np.empty(n, np.float32)
    gy, gz = np.meshgrid(lo[1] + vox * np.arange(n[1]), lo[2] + vox * np.arange(n[2]), indexing='ij')
    for i in range(n[0]):
        t = o3d.core.Tensor(np.stack([np.full(gy.size, lo[0] + vox * i), gy.ravel(), gz.ravel()], 1).astype(np.float32))
        f[i] = np.maximum(sh.compute_signed_distance(t).numpy(), sg.compute_signed_distance(t).numpy() - tol).reshape(n[1], n[2])
    V, F, _, _ = measure.marching_cubes(f, 0.0)
    m = trimesh.Trimesh(V * vox + lo, F[:, ::-1], process=True)
    m = max(m.split(only_watertight=False), key=lambda p: p.area)
    trimesh.smoothing.filter_taubin(m, iterations=10)
    if m.volume < 0:
        m.invert()
    return m


def run(job, log=print):
    """The detail pass for a finished scan: replaces the exported models, keeps the measurements."""
    from .postprocess import decimate
    out = job.root / 'result'
    hull = trimesh.load(out / 'limb.ply')
    work = job.root / 'detail'
    data = prepare(job, work)
    depth = train_remote(data, job.root.name, log=log)
    gs = fuse(depth, data / 'images', hull.bounds)
    gs, agree = calibrate(gs, hull, log=log)
    if agree < 0.5:
        log('[detail] 2DGS and the hull disagree: keeping the hull model')
        return False
    m = combine(hull, gs)
    log(f'[detail] detailed model: {len(m.faces)} faces, watertight {m.is_watertight}, '
        f'{m.volume / 1000:.0f} ml (hull {hull.volume / 1000:.0f} ml)')
    shutil.copy(out / 'limb.ply', out / 'limb_hull.ply')
    m.export(out / 'limb.ply')
    m.export(out / 'limb.stl')
    m.export(out / 'limb.obj')
    disp = decimate(m, 150000)
    try:
        from .texture import colorize
        disp = colorize(disp, job.root / 'dense' / 'sparse', job.root / 'dense' / 'images', log=log)
    except Exception as e:  # noqa: BLE001
        log(f'[detail] texture skipped: {e}')
    disp.export(out / 'limb.glb')
    disp.export(out / 'limb_textured.ply')
    mf = out / 'measurements.json'
    if mf.exists():
        meas = json.loads(mf.read_text())
        meas.setdefault('mesh', {})['detail'] = '2dgs'
        mf.write_text(json.dumps(meas, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    return True


if __name__ == '__main__':
    import sys
    from .pipeline import Job
    j = Job(sys.argv[1])
    ok = run(j, log=j.log)
    print('detail', 'done' if ok else 'skipped')
