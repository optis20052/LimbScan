"""LimbScan reconstruction pipeline.

input photos / video  ->  frame selection  ->  COLMAP SfM (pycolmap)  ->  ArUco metric scale
->  OpenMVS dense MVS + mesh refinement (docker)  ->  limb isolation  ->  watertight mesh
->  clinical measurements.

Run:  python -m limbscan.pipeline <job_dir>      (job_dir/input holds the uploaded media)
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {'.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi', '.3gp'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.tif', '.tiff'}
OPENMVS_IMAGE = os.environ.get('LIMBSCAN_OPENMVS_IMAGE', '')
MAX_SIDE = int(os.environ.get('LIMBSCAN_MAX_SIDE', '2400'))
TARGET_FRAMES = int(os.environ.get('LIMBSCAN_TARGET_FRAMES', '110'))
ARM_MODE = os.environ.get('LIMBSCAN_MODE', 'arm') == 'arm'
# NVIDIA GPU present -> CUDA COLMAP (container) for features, matching and dense stereo
USE_GPU = os.environ.get('LIMBSCAN_GPU', 'auto') == '1' or (os.environ.get('LIMBSCAN_GPU', 'auto') == 'auto' and shutil.which('nvidia-smi') is not None)
COLMAP_IMAGE = os.environ.get('LIMBSCAN_COLMAP_IMAGE', 'docker.io/colmap/colmap:latest')  # 'arm': reconstruct from the limb only; 'scene': whole scene


class Job:
    def __init__(self, root):
        self.root = Path(root)
        self.status_path = self.root / 'status.json'
        self.log_path = self.root / 'log.txt'
        self.t0 = time.time()

    def log(self, msg):
        line = f'[{time.time() - self.t0:7.1f}s] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')

    def status(self, stage, progress, **kw):
        s = {'stage': stage, 'progress': progress, 'time': time.time(), **kw}
        tmp = self.status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(s))
        tmp.replace(self.status_path)
        self.log(f'== {stage} ({progress:.0%})')


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (0, 0), fx=0.5, fy=0.5)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def _resize_max(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def _read_any(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # HEIC etc.
        try:
            from PIL import Image
            import pillow_heif
            pillow_heif.register_heif_opener()
            img = cv2.cvtColor(np.asarray(Image.open(path).convert('RGB')), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    else:
        # honour EXIF orientation
        try:
            from PIL import Image, ImageOps
            im = ImageOps.exif_transpose(Image.open(path))
            if im.size != (img.shape[1], img.shape[0]):
                img = cv2.cvtColor(np.asarray(im.convert('RGB')), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    return img


def prepare_frames(job):
    """Collect photos and/or extract the sharpest, evenly spaced frames from videos."""
    inp, out = job.root / 'input', job.root / 'images'
    out.mkdir(exist_ok=True)
    files = sorted(p for p in inp.iterdir() if p.is_file())
    vids = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    imgs = [p for p in files if p.suffix.lower() in IMAGE_EXT]
    n = 0
    metas = {}  # per-photo AR tracking data uploaded alongside the image (<photo>.json)
    for p in imgs:
        img = _read_any(p)
        if img is None:
            job.log(f'skip unreadable {p.name}')
            continue
        name = f'p{n:04d}.jpg'
        small = _resize_max(img, MAX_SIDE)
        cv2.imwrite(str(out / name), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mp = p.with_suffix('.json')
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
                m['stored_w'], m['stored_h'] = small.shape[1], small.shape[0]
                metas[name] = m
            except Exception:
                job.log(f'bad AR metadata {mp.name}')
        n += 1
    if metas:
        (job.root / 'ar_meta.json').write_text(json.dumps(metas))
        job.log(f'{len(metas)} photos carry AR tracking data')
    for vi, v in enumerate(vids):
        tmp = job.root / f'_frames{vi}'
        tmp.mkdir(exist_ok=True)
        dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(v)],
                                   capture_output=True, text=True).stdout.strip() or 30)
        want = max(TARGET_FRAMES // max(len(vids), 1), 20)
        fps = min(30.0, 4.0 * want / max(dur, 1))  # 4 candidates per kept frame
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(v), '-vf', f'fps={fps:.4f}', '-q:v', '2', str(tmp / 'f%05d.jpg')], check=True)
        cand = sorted(tmp.glob('*.jpg'))
        scores = []
        for c in cand:
            scores.append(sharpness(cv2.imread(str(c))))
        # pick sharpest frame in each of `want` equal windows
        for w in np.array_split(np.arange(len(cand)), min(want, len(cand))):
            if len(w) == 0:
                continue
            best = w[int(np.argmax([scores[i] for i in w]))]
            img = cv2.imread(str(cand[best]))
            cv2.imwrite(str(out / f'v{vi}_{n:04d}.jpg'), _resize_max(img, MAX_SIDE), [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
        shutil.rmtree(tmp)
    if n < 15:
        raise RuntimeError(f'Only {n} usable images; need at least 15 (ideally 60-120).')
    # drop only grossly blurred photos (relative to the set); low-texture views are not blur
    paths = sorted(out.glob('*.jpg'))
    sc = np.array([sharpness(cv2.imread(str(p))) for p in paths])
    bad = sc < 0.08 * np.median(sc)
    for p, b in zip(paths, bad):
        if b:
            p.unlink()
    job.log(f'{n} frames prepared, {bad.sum()} rejected as blurry')
    return out


def run_sfm(job, image_dir, mask_dir=None):
    """Camera poses. With mask_dir, features come only from the limb (+ calibration card), so the pose is
    relative to the limb and the limb may sway relative to the room."""
    import pycolmap
    ws = job.root / 'colmap'
    ws.mkdir(exist_ok=True)
    db = ws / 'database.db'
    if db.exists():
        db.unlink()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = 'OPENCV'
    reader.default_focal_length_factor = 0.75  # phone main cameras (browser frames carry no EXIF focal)
    if mask_dir:
        reader.mask_path = str(mask_dir)
    metas = _ar_metas(job)
    if metas:  # lens from the AR session as the starting calibration (bundle adjustment still refines it)
        from .arscale import intrinsics
        m = next(iter(metas.values()))
        k = intrinsics(m, m['stored_w'], m['stored_h'])
        if k:
            reader.camera_params = ','.join(f'{v:.4f}' for v in (*k, 0, 0, 0, 0))
            job.log(f'[ar] initial intrinsics fx={k[0]:.1f} fy={k[1]:.1f} cx={k[2]:.1f} cy={k[3]:.1f}')
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = min(MAX_SIDE, 2000)
    ext.sift.max_num_features = 8000
    ext.sift.peak_threshold = 0.0025  # skin is low-contrast: keep fainter features
    ext.sift.estimate_affine_shape = False
    ext.sift.domain_size_pooling = True
    if USE_GPU:
        a = ['feature_extractor', '--database_path', _in_job(job, db), '--image_path', _in_job(job, image_dir),
             '--ImageReader.camera_model', 'OPENCV', '--ImageReader.single_camera', '1',
             '--ImageReader.default_focal_length_factor', '0.75',
             '--FeatureExtraction.use_gpu', '1', '--FeatureExtraction.max_image_size', str(ext.max_image_size),
             '--SiftExtraction.max_num_features', '8000', '--SiftExtraction.peak_threshold', '0.0025']
        # (domain-size pooling is CPU-only in COLMAP: leaving it off keeps SIFT on the GPU)
        if mask_dir:
            a += ['--ImageReader.mask_path', _in_job(job, mask_dir)]
        if reader.camera_params:
            a += ['--ImageReader.camera_params', reader.camera_params]
        colmap(job, a)
    else:
        pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ext)
    job.status('matching', 0.15)
    mo = pycolmap.FeatureMatchingOptions()
    mo.guided_matching = True
    n_img = len(list(Path(image_dir).glob('*.jpg')))
    # fixed RANSAC seed for the pair verification too (default -1 = random per run): repeatable results
    vo = pycolmap.TwoViewGeometryOptions()
    vo.ransac.random_seed = 0
    if USE_GPU:
        colmap(job, ['exhaustive_matcher', '--database_path', _in_job(job, db), '--FeatureMatching.use_gpu', '1',
                     '--FeatureMatching.guided_matching', '1', '--TwoViewGeometry.random_seed', '0'])
    elif n_img <= 220:  # limb-only features are few, so exhaustive matching is fast and closes every loop
        pycolmap.match_exhaustive(db, matching_options=mo, verification_options=vo)
    else:  # long ordered sequences (video / live capture): windowed + quadratic overlap
        so = pycolmap.SequentialPairingOptions()
        so.overlap = 30
        so.quadratic_overlap = True
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=so, verification_options=vo)
    job.status('camera poses (SfM)', 0.25)
    # deterministic: the same photos must always give the same model (COLMAP's default seed is random per run)
    pycolmap.set_random_seed(0)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.random_seed = 0
    opts.num_threads = 1  # multi-threaded bundle adjustment is not repeatable across runs; costs only ~10 %
    opts.ba_refine_principal_point = False
    opts.multiple_models = True               # pieces the main model can't absorb are kept (merged via AR later)
    opts.min_model_size = 5
    opts.mapper.init_min_tri_angle = 4.0      # small object, short baselines
    opts.mapper.init_max_forward_motion = 0.99
    opts.init_num_trials = 400
    maps = pycolmap.incremental_mapping(db, image_dir, ws / 'sparse', options=opts)
    if not maps:
        raise RuntimeError('Camera pose estimation failed: images do not overlap enough.')
    models = sorted(maps.values(), key=lambda r: -r.num_reg_images())
    rec = models[0]
    job.pieces = models[1:]
    if job.pieces:
        job.log(f'SfM: {len(job.pieces)} extra piece(s) with {[m.num_reg_images() for m in job.pieces]} photos')
    n_in = len(list(Path(image_dir).glob('*.jpg')))
    job.log(f'SfM registered {rec.num_reg_images()}/{n_in} images, {rec.num_points3D()} points, '
            f'mean reproj {rec.compute_mean_reprojection_error():.3f}px')
    job.log(f'camera: {rec.cameras[next(iter(rec.cameras))].summary() if hasattr(rec.cameras[next(iter(rec.cameras))], "summary") else rec.cameras[next(iter(rec.cameras))]}')
    return rec


def _ar_metas(job):
    p = job.root / 'ar_meta.json'
    return json.loads(p.read_text()) if p.exists() else {}


def apply_metric(job, rec, image_dir):
    """Metric scale: phone AR tracking if the photos carry it, else printed sheets, else the ID-1 card."""
    import pycolmap
    from .scale import recover
    from .card import recover_card, recover_card_pnp
    from .arscale import recover_ar
    metas = _ar_metas(job)
    try:
        if not metas:
            raise RuntimeError('no AR tracking data')
        s, R, t, stats = recover_ar(rec, metas, log=job.log)
    except RuntimeError as e:
        if metas:
            job.log(f'[ar] AR scale unavailable ({e}); trying printed sheets / card')
        s, R, t, stats = _scale_from_objects(job, rec, image_dir, recover, recover_card, recover_card_pnp)
    sim = pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t)
    rec.transform(sim)
    if 'card_corners_sfm' in stats:
        X = np.array(stats.pop('card_corners_sfm'))
        stats['card_corners_mm'] = ((s * (R @ X.T)).T + t).tolist()
    out = job.root / 'colmap' / 'metric'
    out.mkdir(exist_ok=True)
    rec.write(out)
    stats['sfm_images'] = rec.num_reg_images()
    stats['sfm_reproj_px'] = rec.compute_mean_reprojection_error()
    return rec, stats


def place_pieces(job, pieces):
    """Put extra SfM pieces into the same metric (AR world) frame as the main model, using their photos' AR
    poses. Their dense clouds are fitted onto the main surface afterwards (postprocess.merge_pieces)."""
    import pycolmap
    from .arscale import recover_ar
    metas = _ar_metas(job)
    placed = []
    if not metas:
        return placed
    for k, rec in enumerate(pieces):
        try:
            s, R, t, st = recover_ar(rec, metas, log=lambda m: job.log(f'[piece {k}] {m}'), min_frames=6)
        except RuntimeError as e:
            job.log(f'[piece {k}] skipped: {e}')
            continue
        if st['ar_scale_spread_pct'] > 3.0 or st['ar_path_mm'] < 120:
            job.log(f'[piece {k}] skipped: AR fit too weak (spread {st["ar_scale_spread_pct"]:.1f}%, path {st["ar_path_mm"]:.0f} mm)')
            continue
        rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
        out = job.root / 'colmap' / f'piece_{k}'
        out.mkdir(parents=True, exist_ok=True)
        rec.write(out)
        placed.append(out)
    return placed


def _scale_from_objects(job, rec, image_dir, recover, recover_card, recover_card_pnp):
    try:
        s, R, t, stats = recover(rec, image_dir, log=job.log)  # printed marker sheets (if present)
        stats['reference'] = 'printed ArUco sheets'
    except RuntimeError as e:
        job.log(f'[scale] no marker sheets ({e}); using ID-1 card')
        try:
            try:
                s, R, t, stats = recover_card(rec, image_dir, log=job.log)
            except RuntimeError as e2:
                job.log(f'[card] multi-view card failed ({e2}); trying single-view PnP')
                s, R, t, stats = recover_card_pnp(rec, image_dir, log=job.log)
        except RuntimeError:
            reg = {im.name for im in rec.images.values()}
            first = sorted(p.name for p in Path(image_dir).glob('*.jpg'))[:12]
            if first and sum(n in reg for n in first) < len(first) // 3:
                raise RuntimeError('The size-calibration photos could not be linked to the rest of the scan: the arm '
                                   'probably changed position after calibration (e.g. lying on the table, then raised). '
                                   'Hold the card against the arm in the SAME position you scan in, and keep the arm there.')
            raise
    return s, R, t, stats


def _container_run(host_dir, workdir, gpu=False):
    """`docker run` (laptop) or rootless `podman run` (Fedora/Bazzite PC: keep-id user mapping, SELinux-safe
    volume, NVIDIA via CDI) with the job folder mounted at /job."""
    engine = os.environ.get('LIMBSCAN_CONTAINER') or ('docker' if shutil.which('docker') else 'podman')
    if engine == 'podman':
        cmd = ['podman', 'run', '--rm', '--userns=keep-id', '--security-opt', 'label=disable',
               '-v', f'{host_dir}:/job', '-w', workdir]
        if gpu:
            cmd += ['--device', 'nvidia.com/gpu=all']
    else:
        cmd = ['docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}', '-v', f'{host_dir}:/job', '-w', workdir]
        if gpu:
            cmd += ['--gpus', 'all']
    return cmd


def colmap(job, args):
    """Run the CUDA COLMAP CLI in a container; host paths under job.root are visible as /job/..."""
    cmd = _container_run(job.root.resolve(), '/job', gpu=True) + [COLMAP_IMAGE, 'colmap'] + args
    job.log('$ colmap ' + ' '.join(args[:1]))
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(job.root / 'colmap_cli.log', 'a') as f:
        f.write(p.stdout[-20000:] + p.stderr[-20000:])
    if p.returncode != 0:
        raise RuntimeError(f'COLMAP {args[0]} failed:\n' + (p.stdout + p.stderr)[-1500:])


def _in_job(job, path):
    return '/job/' + str(Path(path).resolve().relative_to(job.root.resolve()))


def _docker_image():
    # The 2019 OpenMVS release is the validated engine (v2.3 crashes with FORTIFY on Ubuntu 24.04)
    return OPENMVS_IMAGE or 'openmvs/openmvs-ubuntu:latest'


def mvs(job, args, cwd):
    """Run an OpenMVS tool inside docker with the job directory mounted at /job."""
    img = _docker_image()
    cmd = _container_run(job.root.resolve(), f'/job/{cwd}') + [img] + args
    job.log('$ ' + ' '.join(args))
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(job.root / 'mvs.log', 'a') as f:
        f.write(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f'OpenMVS {args[0]} failed:\n' + (p.stdout + p.stderr)[-2000:])


def _mask_card(model, card_mm, mask_dir, margin_mm=6.0):
    """The calibration card moves with the limb: project its 3D outline (enlarged, both faces) into every
    photo and remove it from the stereo masks, so it can't be rebuilt as a bump on the skin."""
    import pycolmap
    C = np.array(card_mm)
    c = C.mean(0)
    n = np.cross(C[1] - C[0], C[3] - C[0]); n /= np.linalg.norm(n)
    big = c + (C - c) * (1 + 2 * margin_mm / np.linalg.norm(C[2] - C[0]))
    slab = np.vstack([big + n * 2.0, big - n * 4.0])  # card + the gap to the skin under it
    rec = pycolmap.Reconstruction(model)
    for im in rec.images.values():
        mp = Path(mask_dir) / f'{Path(im.name).stem}.png'
        if not mp.exists():
            continue
        Xc = (im.cam_from_world().matrix() @ np.c_[slab, np.ones(len(slab))].T).T
        if (Xc[:, 2] <= 1).any():
            continue
        px = rec.cameras[im.camera_id].img_from_cam(Xc)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        cv2.fillConvexPoly(m, cv2.convexHull(px.astype(np.int32)), 0)
        cv2.imwrite(str(mp), m)


def run_mvs(job, image_dir, mvs_mask_dir=None, skip_images=(), card_mm=None, model=None, dense_name='dense'):
    """Dense stereo. With masks, the background is painted flat grey (no depth there: fast, limb only);
    frames where the card covers the limb are left out (they only serve the scale)."""
    import pycolmap
    dense = job.root / dense_name
    if dense.exists():
        shutil.rmtree(dense)
    src = image_dir
    if mvs_mask_dir:
        from .segment import apply_mask
        if card_mm is not None:
            _mask_card(job.root / 'colmap' / 'metric', card_mm, mvs_mask_dir)
        src = job.root / 'images_masked'
        if not src.exists():
            apply_mask(image_dir, mvs_mask_dir, src)
    uo = pycolmap.UndistortCameraOptions()
    uo.max_image_size = MAX_SIDE
    model = Path(model) if model else job.root / 'colmap' / 'metric'
    if skip_images:  # drop card-calibration frames from the stereo model (they only fix the scale)
        rec = pycolmap.Reconstruction(model)
        for im in list(rec.images.values()):
            if im.name in set(skip_images) and im.has_pose:
                rec.deregister_frame(im.frame_id)
        model = job.root / 'colmap' / 'metric_stereo'
        model.mkdir(exist_ok=True)
        rec.write(model)
    pycolmap.undistort_images(dense, model, src, output_type='COLMAP', undistort_options=uo,
                              num_patch_match_src_images=12)  # GPU stereo: each depth map uses its 12 best neighbours
    # text model for OpenMVS' COLMAP importer
    r = pycolmap.Reconstruction(dense / 'sparse')
    r.write_text(dense / 'sparse')
    job.status('dense reconstruction', 0.45)
    if USE_GPU and os.environ.get('LIMBSCAN_GPU_STEREO') == '1':  # CUDA PatchMatch (full-res, slower than OpenMVS level 1 here)
        colmap(job, ['patch_match_stereo', '--workspace_path', _in_job(job, dense), '--workspace_format', 'COLMAP',
                     '--PatchMatchStereo.geom_consistency', 'true', '--PatchMatchStereo.max_image_size', '2000',
                     '--PatchMatchStereo.filter_min_ncc', '0.1'])
        colmap(job, ['stereo_fusion', '--workspace_path', _in_job(job, dense), '--workspace_format', 'COLMAP',
                     '--input_type', 'geometric', '--output_path', _in_job(job, dense / 'scene_dense.ply'),
                     '--StereoFusion.min_num_pixels', '3'])
        return dense
    new = 'openmvs-ubuntu' not in _docker_image()
    mvs(job, ['InterfaceCOLMAP', '-i', '.', '-o', 'scene.mvs', '--image-folder', 'images'], dense_name)
    dargs = ['DensifyPointCloud', 'scene.mvs', '--resolution-level', os.environ.get('LIMBSCAN_MVS_LEVEL', '1'), '--min-resolution', '640',
             '--number-views', '5', '--number-views-fuse', '3', '--estimate-normals', '0', '-v', '1']
    if new:
        dargs += ['--filter-point-cloud', '1', '--tower-mode', '0']
    mvs(job, dargs, dense_name)
    return dense


def run(job_dir, resume=False):
    job = Job(job_dir)
    try:
        metric = job.root / 'colmap' / 'metric'
        job.pieces = []
        if resume and metric.exists() and (job.root / 'scale.json').exists():
            image_dir = job.root / 'images'
            scale_stats = json.loads((job.root / 'scale.json').read_text())
            job.log('resuming from metric SfM model')
        else:
            job.status('preparing frames', 0.02)
            image_dir = prepare_frames(job)
            mask_dir = None
            if ARM_MODE:
                job.status('finding the limb in each photo', 0.05)
                from .segment import build_masks
                seg = build_masks(image_dir, job.root / 'colmap' / 'mask_sfm', job.root / 'colmap' / 'mask_mvs', log=job.log)
                (job.root / 'segment.json').write_text(json.dumps(seg))
                mask_dir = job.root / 'colmap' / 'mask_sfm'
            job.status('feature extraction', 0.12)
            rec = run_sfm(job, image_dir, mask_dir)
            job.status('metric scale', 0.35)
            rec, scale_stats = apply_metric(job, rec, image_dir)
            scale_stats['mode'] = 'arm' if ARM_MODE else 'scene'
            if job.pieces and scale_stats.get('method') == 'ar':
                scale_stats['pieces'] = [p.name for p in place_pieces(job, job.pieces)]
            (job.root / 'scale.json').write_text(json.dumps(scale_stats, indent=1))
        dense = job.root / 'dense'
        if not (resume and (dense / 'scene_dense.ply').exists()):
            seg = json.loads((job.root / 'segment.json').read_text()) if (job.root / 'segment.json').exists() else {}
            skip = scale_stats.get('card_view_names', []) if scale_stats.get('mode') == 'arm' else []  # card on the limb
            card_mm = scale_stats.get('card_corners_mm') if scale_stats.get('mode') == 'arm' else None
            dense = run_mvs(job, image_dir, job.root / 'colmap' / 'mask_mvs' if seg else None, skip, card_mm)
            for name in scale_stats.get('pieces', []):
                job.status(f'dense reconstruction ({name})', 0.8)
                try:
                    run_mvs(job, image_dir, job.root / 'colmap' / 'mask_mvs' if seg else None,
                            model=job.root / 'colmap' / name, dense_name=f'dense_{name}')
                except RuntimeError as e:
                    job.log(f'[{name}] stereo failed: {str(e)[-200:]}')
        job.status('isolating limb + measurements', 0.9)
        from .postprocess import finalize
        summary = finalize(job, dense, scale_stats)
        job.status('done', 1.0, summary=summary)
    except Exception as e:
        import traceback
        job.log(traceback.format_exc())
        job.status('failed', 1.0, error=str(e))
        raise


if __name__ == '__main__':
    run(sys.argv[1], resume='--resume' in sys.argv)
