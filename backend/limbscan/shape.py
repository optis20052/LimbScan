"""The final shape on the GPU (limbscan/irshape.py, ~1 min on the RTX 4070 Ti SUPER): the outline hull moved until
it matches every photo's outline and the torch shading. Runs in the podman image `limbgpu` (PyTorch + nvdiffrast).

Validated on the printed hand (two torch scans) against its STL: 95 % of the surface within 2 mm and 89 % within
1 mm (hull: 82 % / 71 %), fingers and palm 91-95 % within 2 mm.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

import trimesh

IMAGE = os.environ.get('LIMBSCAN_GPU_IMAGE', 'limbgpu')
CODE = Path(__file__).resolve().parent
ARGS = os.environ.get('LIMBSCAN_SHAPE_ARGS', 'lr=0.25 w_sil=300 iters=300,200,100 n_pts=100000').split()


def available():
    return os.environ.get('LIMBSCAN_SHAPE', '1') == '1' and shutil.which('podman') is not None \
        and shutil.which('nvidia-smi') is not None


def clean(start, shaped, mu=10.0, depth=8):
    """The GPU step's correction without its wrinkle noise: the per-vertex correction low-passed on the surface
    (sparse solve), then a fresh watertight surface (screened Poisson) - no folded triangles. Same accuracy on the
    printed hand (94 % within 2 mm), far less 'crumpled' (2-5 mm wrinkles, speckle)."""
    import numpy as np
    import open3d as o3d
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from scipy.spatial import cKDTree
    F = start.faces
    n = len(start.vertices)
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    D0 = np.asarray(shaped.vertices) - start.vertices
    D = spl.splu((sp.eye(n) + mu * L).tocsc()).solve(D0)
    if UNFOLD:
        # where the GPU step folded the surface over (its normal turned more than ~90 degrees from the outline model's:
        # the thumb web, the base of a finger gap - 31 Oct, a fold the user saw in the thumb web of a printed hand),
        # the correction there and a few rings around it is replaced by a much smoother one: a smooth hollow instead
        # of a crease. Elsewhere the detail stays.
        folded = (trimesh.Trimesh(start.vertices + D, F, process=False).vertex_normals * start.vertex_normals).sum(1) \
            < UNFOLD_COS
        if folded.any():
            zone = folded.astype(float)
            for _ in range(UNFOLD_RINGS):
                zone = np.minimum(1.0, zone + A @ zone)
            Ds = spl.splu((sp.eye(n) + UNFOLD_MU * L).tocsc()).solve(D0)
            w = spl.splu((sp.eye(n) + 2.0 * L).tocsc()).solve(zone)[:, None].clip(0, 1)  # soft edge
            D = D * (1 - w) + Ds * w
    m = trimesh.Trimesh(start.vertices + D, F, process=False)
    P, fi = trimesh.sample.sample_surface_even(m, 400000, radius=0.35, seed=0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(m.face_normals[fi])
    pm, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, scale=1.1, linear_fit=True)
    full = trimesh.Trimesh(np.asarray(pm.vertices), np.asarray(pm.triangles), process=False)
    out = full.copy()
    d, _ = cKDTree(P).query(out.triangles_center)
    out.update_faces(d < 2.0)
    out.remove_unreferenced_vertices()
    out = max(out.split(only_watertight=False), key=lambda x: len(x.faces))
    if out.is_watertight:
        return out
    if m.is_watertight:
        return m
    # neither is closed: the Poisson surface uncropped (closed by construction), its largest piece
    full = max(full.split(only_watertight=False), key=lambda x: len(x.faces))
    return full


UNFOLD = os.environ.get('LIMBSCAN_UNFOLD', '0') == '1'  # smooth out folds left by the GPU step (see clean)
UNFOLD_MU = float(os.environ.get('LIMBSCAN_UNFOLD_MU', '300'))
UNFOLD_COS = float(os.environ.get('LIMBSCAN_UNFOLD_COS', '0'))  # surface turned more than acos(this) from the outline model
UNFOLD_RINGS = int(os.environ.get('LIMBSCAN_UNFOLD_RINGS', '6'))
CAMFIX = os.environ.get('LIMBSCAN_CAMFIX', '0') == '1'
CAMFIX_ARGS = 'levels=2 iters=150 lr=0.25 w_sil=300 w_shade=0 n_pts=50000 pose=1'.split()


def _gpu_run(work, args):
    cache = Path.home() / '.cache' / 'lgcache'
    cache.mkdir(parents=True, exist_ok=True)
    return ['podman', 'run', '--rm', '--device', 'nvidia.com/gpu=all', '--security-opt', 'label=disable',
            '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True', '-v', f'{cache}:/root/.cache',
            '-v', f'{CODE}:/code:ro', '-v', f'{work}:/work', '-w', '/code', IMAGE,
            'python', 'irshape.py', '/work/prep.npz', '/work/start.ply', '/work/shape.ply', *args]


NORMALS = os.environ.get('LIMBSCAN_NORMALS', '1') == '1'  # MoGe normals on the palm and back of the hand
NORM_THICK = float(os.environ.get('LIMBSCAN_NORM_THICK', '24'))
W_NORM = float(os.environ.get('LIMBSCAN_W_NORM', '50'))
MOGE_IMAGE = os.environ.get('LIMBSCAN_MOGE_IMAGE', 'limbmoge3')


def normals_start(job):
    """Start the MoGe normals in the background as soon as the photos are ready: they run on the GPU while the
    cameras, stereo and outlines are worked out (keeps the whole scan within ~3 min). normals() collects them."""
    if not (NORMALS and available()):
        return
    out = job.root / 'moge'
    shutil.rmtree(out, ignore_errors=True)
    cache = Path.home() / '.cache' / 'lgcache'
    cmd = ['podman', 'run', '--rm', '--device', 'nvidia.com/gpu=all', '--security-opt', 'label=disable',
           '-v', f'{cache}:/root/.cache', '-v', f'{CODE}:/code:ro', '-v', f'{(job.root / "images").resolve()}:/img:ro',
           '-v', f'{job.root.resolve()}:/scan', '-w', '/code', MOGE_IMAGE, 'python', 'mogerun.py', '/img', '/scan/moge']
    job.moge = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def normals(job, log=print):
    """Surface normals of every photo (MoGe-3, mogerun.py) into <scan>/moge: the palm's relief, the knuckles and the
    finger sides the outlines cannot carve and skin stereo is too noisy for (1 Nov, a real hand: the palm came out
    flat with a false ridge; with the normals it has its thumb mound and hollow)."""
    out = job.root / 'moge'
    if getattr(job, 'moge', None) is None:
        normals_start(job)
    p = getattr(job, 'moge', None)
    if p is None:
        return False
    t0 = time.time()
    text, _ = p.communicate(timeout=600)
    job.moge = None
    line = [x for x in text.splitlines() if x.startswith('[moge]')]
    if p.returncode != 0 or not line:
        log(f'[moge] failed: {text.strip().splitlines()[-1:]}')
        shutil.rmtree(out, ignore_errors=True)
        return False
    log(f'{line[-1]} (waited {time.time() - t0:.0f}s for them)')
    return True


def _normal_weights(job, hull, prep, model=None, log=print):
    """Where the photos' normals may shape the surface: the palm and the back of the hand only. Beyond the wrist
    (the hand found in the photos), and where the hand is thick (> NORM_THICK mm through, measured along the inward
    normal): not the fingers (MoGe is too coarse across a finger: 1 Nov, printed-hand fingers 96 -> 90 %), not the
    forearm (hair; a real arm got worse with them, 0.51 -> 0.71 mm). Stored in prep.npz as 'w_norm_v'."""
    import json
    import numpy as np
    V, N = np.asarray(hull.vertices), np.asarray(hull.vertex_normals)
    w = np.ones(len(V), np.float32)
    lmf = job.root / 'hand_landmarks.json'
    arm = None
    if lmf.exists():
        from .handarm import arm_in_model
        try:
            arm = arm_in_model(model or job.root / 'colmap' / 'metric', json.loads(lmf.read_text()), log=lambda *a: None)
        except Exception:  # noqa: BLE001
            arm = None
    if arm is None:
        log('[shape] hand not found in 3D: no photo normals')
        w[:] = 0
    else:
        W = np.asarray(arm['wrist'], float)
        u = np.asarray(arm['b'], float) - W
        u /= np.linalg.norm(u)
        w *= ((V - W) @ u > 15.0)
        import open3d as o3d
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(np.asarray(hull.faces, np.uint32)))
        rays = np.c_[V - N * 0.5, -N].astype(np.float32)
        thick = sc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy() + 0.5  # inf where nothing is hit
        w *= thick > NORM_THICK
    z = dict(np.load(prep))
    z['w_norm_v'] = w
    np.savez(prep, **z)
    log(f'[shape] photo normals act on {100 * w.mean():.0f}% of the surface (palm and back of the hand)')


def fix_cameras(job, hull, log=print):
    """Every photo's camera lined up with the limb's outlines in all the others (a small turn and shift each, on the
    GPU, ~15 s): a camera placed a little off, or a limb that moved a little between photos (a hand held up without
    support), otherwise carves its error into the outline model - the corrections come before the carving. Writes
    the corrected cameras to colmap/metric_fixed and returns that model, or None."""
    import numpy as np
    import pycolmap
    from .irprep import prepare
    work = job.root / 'camfix'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    t0 = time.time()
    hull.export(work / 'start.ply')
    prepare(job.root, work / 'prep.npz', work / 'start.ply')
    r = subprocess.run(_gpu_run(work, CAMFIX_ARGS), capture_output=True, text=True, timeout=600)
    pf = work / 'shape.ply.poses.npz'
    if r.returncode != 0 or not pf.exists():
        log(f'[camfix] failed: {(r.stdout + r.stderr).strip().splitlines()[-1:]}')
        return None
    z = np.load(pf)
    rec = pycolmap.Reconstruction(str(job.root / 'colmap' / 'metric'))
    by = {im.name: im for im in rec.images.values()}
    turned = []
    for name, w, dt in zip(z['names'], z['dw'], z['dt']):
        im = by.get(str(name))
        if im is None:
            continue
        th = float(np.linalg.norm(w))
        k = w / th if th > 1e-12 else np.zeros(3)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        dR = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        T = im.cam_from_world()
        R, t = dR @ T.rotation.matrix(), np.asarray(T.translation) + dt
        rec.frames[im.frame_id].rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        turned.append(np.degrees(th))
    out = job.root / 'colmap' / 'metric_fixed'
    out.mkdir(exist_ok=True)
    rec.write(str(out))
    log(f'[camfix] {len(turned)} cameras corrected: turned median {np.median(turned):.2f} max {np.max(turned):.2f} deg, '
        f'moved median {np.median(np.linalg.norm(z["dt"], axis=1)):.1f} max {np.linalg.norm(z["dt"], axis=1).max():.1f} '
        f'mm, {time.time() - t0:.0f}s')
    return out


def refine(job, hull, log=print, model=None):
    """The refined closed surface, or None (keep the hull) when the GPU step fails or its result looks wrong."""
    from .irprep import prepare
    work = job.root / 'shape'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    t0 = time.time()
    hull.export(work / 'start.ply')
    args = list(ARGS)
    if NORMALS:
        normals(job, log=log)
        if (job.root / 'moge').exists():
            args.append(f'w_norm={W_NORM}')
    n = prepare(job.root, work / 'prep.npz', work / 'start.ply', model=model)
    if NORMALS and any('w_norm' in a for a in args):
        _normal_weights(job, hull, work / 'prep.npz', model, log=log)
    cmd = _gpu_run(work, args)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith('[shape]'):
            log(line)
        elif line:
            tail = (tail + [line])[-8:]
    if p.wait(timeout=600) != 0 or not (work / 'shape.ply').exists():
        log(f'[shape] GPU step failed: {" | ".join(tail[-3:])}: keeping the outline hull')
        return None
    m = clean(trimesh.load(work / 'start.ply', process=False), trimesh.load(work / 'shape.ply', process=False),
              mu=float(os.environ.get('LIMBSCAN_SHAPE_MU', '10')), depth=int(os.environ.get('LIMBSCAN_SHAPE_DEPTH', '8')))
    ratio = m.volume / hull.volume if hull.volume > 0 else 0
    # the hull bounds the limb: the refined surface is the same size or smaller (hollows carved; a hull from few
    # directions is fat, and the photos' outlines and shading take that off: 31 Oct, a result at 83 % of its hull was
    # 89 % within 2 mm of the STL while the hull kept instead was 64 %). Below 75 % the step has collapsed.
    if not (0.75 < ratio < 1.03) or not m.is_watertight:
        log(f'[shape] result rejected (volume {ratio * 100:.0f} % of the hull, watertight {m.is_watertight}): '
            'keeping the outline hull')
        return None
    log(f'[shape] {n} photos, volume {ratio * 100:.1f} % of the hull, {time.time() - t0:.0f}s in all')
    (work / 'prep.npz').unlink(missing_ok=True)
    return m
