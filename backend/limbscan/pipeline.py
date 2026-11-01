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
SEG_SHIFT = os.environ.get('LIMBSCAN_SEG_SHIFT', '1') == '1'  # cut-out outline moved onto each photo's detected hand
# NVIDIA GPU present -> CUDA COLMAP (container) for features, matching and dense stereo
USE_GPU = os.environ.get('LIMBSCAN_GPU', 'auto') == '1' or (os.environ.get('LIMBSCAN_GPU', 'auto') == 'auto' and shutil.which('nvidia-smi') is not None)
COLMAP_IMAGE = os.environ.get('LIMBSCAN_COLMAP_IMAGE', 'docker.io/colmap/colmap:latest')  # 'arm': reconstruct from the limb only; 'scene': whole scene
# camera poses: 'global' (GLOMAP-style, multi-core and repeatable, ~4x faster) or the older 'incremental' mapper
MAPPER = os.environ.get('LIMBSCAN_MAPPER', 'global')
ROT_MAX = float(os.environ.get('LIMBSCAN_ROT_MAX', '1.04'))  # larger photo/gyro turn ratios: no size correction
ROT_SCALE = os.environ.get('LIMBSCAN_ROT_SCALE', '1') == '1'  # size corrected from photo vs gyro turns
RESCUE = os.environ.get('LIMBSCAN_AR_RESCUE', '1') == '1'  # cameras from the phone when the photos place < 90 %
PRIOR_MM = float(os.environ.get('LIMBSCAN_AR_PRIOR_MM', '10'))  # how far (1 sigma) the ar mapper may move a camera from the phone's pose
# dense stereo: 'gpu' (our OpenCL PatchMatch, limbscan/gpumvs.py) when an OpenCL GPU exists, else OpenMVS
MVS_ENGINE = os.environ.get('LIMBSCAN_MVS_ENGINE', 'gpu')
THREADS = os.cpu_count() or 4


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
    from concurrent.futures import ThreadPoolExecutor

    from .prefetch import cached_image

    def load(i_p):  # decode + resize + encode on all cores; names are given in order afterwards
        i, p = i_p
        c = cached_image(job.root, p)  # already prepared while uploading
        if c is not None:
            shutil.copyfile(c, out / f'_tmp{i:05d}.jpg')
            from PIL import Image
            w, h = Image.open(c).size
            return (h, w, 3)
        img = _read_any(p)
        if img is None:
            return None
        small = _resize_max(img, MAX_SIDE)
        cv2.imwrite(str(out / f'_tmp{i:05d}.jpg'), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return small.shape

    with ThreadPoolExecutor(THREADS) as ex:
        shapes = list(ex.map(load, enumerate(imgs)))
    names = {}
    for i, (p, shape) in enumerate(zip(imgs, shapes)):
        if shape is None:
            job.log(f'skip unreadable {p.name}')
            continue
        name = f'p{n:04d}.jpg'
        (out / f'_tmp{i:05d}.jpg').rename(out / name)
        names[name] = p.stem
        mp = p.with_suffix('.json')
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
                m['stored_w'], m['stored_h'] = shape[1], shape[0]
                metas[name] = m
            except Exception:
                job.log(f'bad AR metadata {mp.name}')
        n += 1
    (job.root / 'names.json').write_text(json.dumps(names))  # photo -> upload, for the prefetch cache
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
    with ThreadPoolExecutor(THREADS) as ex:
        sc = np.array(list(ex.map(lambda p: sharpness(cv2.imread(str(p))), paths)))
    bad = sc < 0.08 * np.median(sc)
    for p, b in zip(paths, bad):
        if b:
            p.unlink()
    job.log(f'{n} frames prepared, {bad.sum()} rejected as blurry')
    return out


def run_sfm(job, image_dir, mask_dir=None, mapper=None):
    """Camera poses. With mask_dir, features come only from the limb (+ calibration card), so the pose is
    relative to the limb and the limb may sway relative to the room."""
    import pycolmap
    ws = job.root / 'colmap'
    ws.mkdir(exist_ok=True)
    db = ws / 'database.db'
    if db.exists():
        db.unlink()
    mapper = mapper or MAPPER
    if mapper == 'ar' and any('pose' in m for m in _ar_metas(job).values()):
        # the cameras start where the phone was, so the room cannot split the scan: its features (plenty, unlike
        # plain plastic or smooth skin) only correct the phone's 1-2 degree errors on every photo
        mask_dir = None
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
    ext = sift_options()
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
    elif mask_dir:
        _extract_cropped(job, db, image_dir, mask_dir, reader, ext)
    else:
        pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ext)
    job.status('matching', 0.15)
    pairs = _ar_pairs(job, db, metas) if metas and os.environ.get('LIMBSCAN_AR_PAIRS', '1') == '1' else None
    mo = pycolmap.FeatureMatchingOptions()
    # guided (epipolar) re-matching costs 4x the matching time; with AR-chosen pairs the plain matches suffice
    mo.guided_matching = pairs is None
    n_img = len(list(Path(image_dir).glob('*.jpg')))
    # fixed RANSAC seed for the pair verification too (default -1 = random per run): repeatable results
    vo = pycolmap.TwoViewGeometryOptions()
    vo.ransac.random_seed = 0
    if USE_GPU and pairs is not None:
        colmap(job, ['matches_importer', '--database_path', _in_job(job, db), '--match_list_path', _in_job(job, pairs),
                     '--match_type', 'pairs', '--FeatureMatching.use_gpu', '1', '--TwoViewGeometry.random_seed', '0'])
    elif USE_GPU:
        colmap(job, ['exhaustive_matcher', '--database_path', _in_job(job, db), '--FeatureMatching.use_gpu', '1',
                     '--FeatureMatching.guided_matching', '1', '--TwoViewGeometry.random_seed', '0'])
    elif pairs is not None:  # exact, repeatable matcher (COLMAP's multi-threaded CPU matcher varies run to run)
        from .matching import match_pairs
        match_pairs(db, pairs, vo, log=job.log)
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
    # the phone's factory lens calibration is exact; refining it on the photos drifts by 1.5-3.6 % (seen: fy +3.6 %
    # from room features, fx +1.6 % and invented distortion from limb features), which scales or stretches the limb
    # (printed hand: 2 % too small). Always fixed when the phone sent its calibration.
    fix_k = bool(reader.camera_params) and os.environ.get('LIMBSCAN_FIX_K', '1') != '0'
    if fix_k:
        job.log('[ar] lens fixed to the phone calibration')
    if mapper == 'ar' and metas and any('pose' in m for m in metas.values()):
        maps = _ar_mapping(db, image_dir, ws / 'sparse', metas, fix_k, log=job.log)
        job.ar_mapped = True
    elif mapper == 'global':
        maps = _global_mapping(db, image_dir, ws / 'sparse', fix_k)
    else:
        maps = _incremental_mapping(db, image_dir, ws / 'sparse', fix_k)
    if not maps:
        raise RuntimeError('Camera pose estimation failed: images do not overlap enough.')
    models = sorted(maps.values(), key=lambda r: -r.num_reg_images())
    rec = models[0]
    job.pieces = models[1:]
    if job.pieces:
        job.log(f'SfM: {len(job.pieces)} extra piece(s) with {[m.num_reg_images() for m in job.pieces]} photos')
    n_in = len(list(Path(image_dir).glob('*.jpg')))
    job.log(f'SfM ({MAPPER}) registered {rec.num_reg_images()}/{n_in} images, {rec.num_points3D()} points, '
            f'mean reproj {rec.compute_mean_reprojection_error():.3f}px')
    cam = rec.cameras[next(iter(rec.cameras))]
    job.log(f'camera: {cam.model_name} {cam.width}x{cam.height} params {[round(v, 3) for v in cam.params]}')
    return rec


def _global_mapping(db, image_dir, out, fix_k=False):
    """Global SfM (rotation averaging + global positioning + bundle adjustment): all photos at once, multi-core.
    Every random choice is seeded, so the same photos give the same cameras whatever the thread count."""
    import pycolmap
    out.mkdir(parents=True, exist_ok=True)
    o = pycolmap.GlobalPipelineOptions()
    o.random_seed = 0
    o.min_model_size = 5
    g = o.mapper
    g.random_seed = g.global_positioning.random_seed = g.retriangulation.random_seed = 0
    g.rotation_averaging.random_seed = 0
    g.keep_max_num_tracks = 6000  # the bundle adjustment needs a few thousand good tracks, not all of them
    # re-adding every track and adjusting again doubles the time for no gain in accuracy (synthetic truth test)
    g.skip_retriangulation = os.environ.get('LIMBSCAN_RETRIANGULATE', '0') != '1'
    if fix_k:
        b = g.bundle_adjustment
        b.refine_focal_length = b.refine_principal_point = b.refine_extra_params = False
    return pycolmap.global_mapping(db, image_dir, out, options=o)


def _ar_mapping(db, image_dir, out, metas, fix_k=True, log=print):
    """Cameras from the phone's AR tracking, refined on the photos (LIMBSCAN_MAPPER=ar). Every photo starts where the
    phone was (mm, the AR world): 3D points are triangulated with those poses, then a bundle adjustment moves the
    cameras until the points line up (the phone's few-mm tracking error, a limb that sways a little), and the points
    are triangulated again. No photo can be left out or split off: a photo with few matches keeps the phone's pose
    (31 Oct printed hand: the global solve split 40 photos 27 + 9, losing the whole palm side)."""
    import pycolmap
    from .arscale import ar_pose
    out.mkdir(parents=True, exist_ok=True)
    d = pycolmap.Database.open(str(db))
    cams, ims = d.read_all_cameras(), d.read_all_images()
    d.close()
    txt = out / 'ar_init'
    txt.mkdir(exist_ok=True)
    with open(txt / 'cameras.txt', 'w') as f:
        for c in cams:
            f.write(f'{c.camera_id} {c.model_name} {c.width} {c.height} {" ".join(f"{v:.6f}" for v in c.params)}\n')
    n = 0
    with open(txt / 'images.txt', 'w') as f:
        for im in ims:
            m = metas.get(im.name)
            if not m or 'pose' not in m:
                continue
            M = ar_pose(m)
            # AR camera axes (x, y, -z = forward) -> the stored portrait photo: u runs along the AR y axis, v along x
            R = np.stack([M[:3, 1], M[:3, 0], -M[:3, 2]])
            t = -R @ (M[:3, 3] * 1000.0)
            q = pycolmap.Rotation3d(R).quat  # x, y, z, w
            f.write(f'{im.image_id} {q[3]:.10f} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} '
                    f'{im.camera_id} {im.name}\n\n')
            n += 1
    (txt / 'points3D.txt').write_text('')
    rec = pycolmap.Reconstruction(str(txt))

    def tri(rec, px, deg):
        # the phone's camera directions are 0-2 degrees off (20-40 px at a hand 30 cm away): the first points are
        # accepted with a loose tolerance, the cameras adjusted on them, then the tolerance tightened
        o = pycolmap.IncrementalPipelineOptions()
        o.ba_refine_principal_point = o.ba_refine_focal_length = o.ba_refine_extra_params = False
        t = o.triangulation
        t.create_max_angle_error = t.continue_max_angle_error = deg
        t.merge_max_reproj_error = t.complete_max_reproj_error = px
        o.mapper.filter_max_reproj_error = px
        return pycolmap.triangulate_points(rec, db, image_dir, out / 'tri', clear_points=True, options=o)

    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_focal_length = ba.refine_principal_point = ba.refine_extra_params = False  # the phone lens is exact
    ba.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    centre = {}
    for im in ims:
        m = metas.get(im.name)
        if m and 'pose' in m:
            centre[im.image_id] = ar_pose(m)[:3, 3] * 1000.0

    def adjust(rec):
        # every camera tied to where the phone was (+-10 mm, robust): the adjustment can turn the cameras and move
        # them a little, but not stretch or shrink the whole scene (a free adjustment changed the scale by up to
        # 6 % on the printed hand)
        cfg = pycolmap.BundleAdjustmentConfig()
        priors = []
        for im in rec.images.values():
            cfg.add_image(im.image_id)
            c = centre.get(im.image_id)
            if c is not None:
                pp = pycolmap.PosePrior(position=c, position_covariance=np.eye(3) * PRIOR_MM ** 2,
                                        coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN)
                pp.corr_data_id = im.data_id
                priors.append(pp)
        for cid in rec.cameras:
            cfg.set_constant_cam_intrinsics(cid)
        po = pycolmap.PosePriorBundleAdjustmentOptions()
        po.alignment_ransac.max_error = 50.0
        po.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.CAUCHY
        pycolmap.create_pose_prior_bundle_adjuster(ba, po, cfg, priors, rec).solve()

    steps = [(40.0, 8.0, 8.0), (12.0, 4.0, 4.0), (4.0, 2.0, 2.0)]
    for k, (px, deg, loss) in enumerate(steps):
        rec = tri(rec, px, deg)
        if k == 0:
            e0, p0 = rec.compute_mean_reprojection_error(), rec.num_points3D()
        ba.ceres.loss_function_scale = loss
        if PRIOR_MM > 0:
            adjust(rec)
        else:  # free: the scale comes afterwards from short moves between photos (arscale.LOCAL)
            pycolmap.bundle_adjustment(rec, ba)
    rec = tri(rec, 4.0, 2.0)
    seen = sum(1 for im in rec.images.values() if im.num_points3D >= 20)
    log(f'[ar-sfm] {n} photos placed by the phone ({seen} with 20+ points), {rec.num_points3D()} points; reprojection '
        f'{e0:.1f} px from the phone poses ({p0} points), {rec.compute_mean_reprojection_error():.2f} px after adjusting')
    return {0: rec}


def _incremental_mapping(db, image_dir, out, fix_k=False):
    import pycolmap
    opts = pycolmap.IncrementalPipelineOptions()
    opts.random_seed = 0
    opts.num_threads = 1  # multi-threaded bundle adjustment is not repeatable across runs; costs only ~10 %
    opts.ba_refine_principal_point = False
    if fix_k:
        opts.ba_refine_focal_length = opts.ba_refine_extra_params = False
    opts.multiple_models = True               # pieces the main model can't absorb are kept (merged via AR later)
    opts.min_model_size = 5
    opts.mapper.init_min_tri_angle = 4.0      # small object, short baselines
    opts.mapper.init_max_forward_motion = 0.99
    opts.init_num_trials = 400
    out.mkdir(parents=True, exist_ok=True)
    return pycolmap.incremental_mapping(db, image_dir, out, options=opts)


def sift_options():
    import pycolmap
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = min(MAX_SIDE, int(os.environ.get('LIMBSCAN_SFM_MAX_SIDE', 2000)))
    ext.sift.max_num_features = 8000
    ext.sift.peak_threshold = 0.0025  # skin is low-contrast: keep fainter features
    ext.sift.estimate_affine_shape = False
    # domain-size pooling: 2.4x slower extraction for no measurable gain on limb scans (same registration and
    # accuracy vs Structure on the AR validation set)
    ext.sift.domain_size_pooling = os.environ.get('LIMBSCAN_SIFT_DSP') == '1'
    return ext


def sift_cropped(items, tmp, ext=None):
    """SIFT only where the limb is. items: [(name, image path, SfM mask path)]. Every photo is cropped to its
    mask's bounding box (about half the pixels) and the keypoints are moved back to full-photo coordinates:
    same features as extracting on the full photo with the mask, 1.7x faster. Returns {name: (keypoints, descriptors)}."""
    import pycolmap
    from concurrent.futures import ThreadPoolExecutor
    tmp = Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / 'img').mkdir(parents=True)
    (tmp / 'mask').mkdir()

    def crop(item):
        n, ip, mp = item
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        im = cv2.imread(str(ip))
        ys, xs = np.nonzero(m) if m is not None else ((), ())
        if len(xs) == 0:
            x0, y0, x1, y1 = 0, 0, im.shape[1], im.shape[0]
            m = np.zeros(im.shape[:2], np.uint8) if m is None else m
        else:
            x0, y0 = max(int(xs.min()) - 24, 0), max(int(ys.min()) - 24, 0)
            x1, y1 = min(int(xs.max()) + 25, im.shape[1]), min(int(ys.max()) + 25, im.shape[0])
        cv2.imwrite(str(tmp / 'img' / n), im[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(str(tmp / 'mask' / f'{n}.png'), m[y0:y1, x0:x1])
        return n, (x0, y0)

    with ThreadPoolExecutor(THREADS) as ex:
        off = dict(ex.map(crop, items))
    r = pycolmap.ImageReaderOptions()
    r.camera_model = 'SIMPLE_PINHOLE'
    r.mask_path = str(tmp / 'mask')
    tdb = tmp / 'crops.db'
    pycolmap.extract_features(tdb, tmp / 'img', camera_mode=pycolmap.CameraMode.PER_IMAGE, reader_options=r,
                              extraction_options=ext or sift_options())
    out = {}
    src = pycolmap.Database.open(tdb)
    for im in src.read_all_images():
        kp = np.array(src.read_keypoints(im.image_id), dtype=np.float32)
        kp[:, :2] += np.array(off[im.name], np.float32)
        out[im.name] = (kp, np.array(src.read_descriptors(im.image_id).data, dtype=np.uint8))
    src.close()
    shutil.rmtree(tmp)
    return out


def _extract_cropped(job, db, image_dir, mask_dir, reader, ext):
    """Limb-only SIFT for every photo into the database; photos already done while uploading (prefetch cache) are
    not recomputed."""
    import pycolmap
    from .prefetch import cached_sift
    pycolmap.Database.open(db).close()  # creates the empty database
    pycolmap.import_images(db, image_dir, pycolmap.CameraMode.SINGLE, options=reader)
    names = sorted(p.name for p in Path(image_dir).glob('*.jpg'))
    # features made while uploading used the masks of that moment: not valid once the photos were cut out anew
    feats = {n: (None if getattr(job, 'fresh_masks', False) else cached_sift(job.root, n)) for n in names}
    todo = [(n, Path(image_dir) / n, Path(mask_dir) / f'{n}.png') for n in names if feats[n] is None]
    if todo:
        feats.update(sift_cropped(todo, job.root / 'colmap' / 'crops', ext))
    if len(todo) < len(names):
        job.log(f'[features] {len(names) - len(todo)} photos were already done while uploading')
    dst = pycolmap.Database.open(db)
    for im in dst.read_all_images():
        kp, desc = feats[im.name]
        dst.write_keypoints(im.image_id, kp)
        dst.write_descriptors(im.image_id, pycolmap.FeatureDescriptors(pycolmap.FeatureExtractorType.SIFT,
                                                                       np.ascontiguousarray(desc, dtype=np.uint8)))
    dst.close()


def _ar_pairs(job, db, metas, k=12, k_side=12):
    """Which photos to match, from the AR camera path: for each photo the k nearest cameras looking the same way,
    plus the k_side cameras that see the same side of the arm from any distance (direction from the arm centre),
    plus its neighbours in capture order. Far fewer pairs than all-against-all, and robust to photos taken from very
    different distances."""
    import pycolmap
    d = pycolmap.Database.open(db)
    names = sorted(im.name for im in d.read_all_images())
    d.close()
    names = [n for n in names if n in metas]
    if len(names) < 20:
        return None
    from .arscale import ar_pose
    M = [ar_pose(metas[n]) for n in names]
    C = np.array([m[:3, 3] for m in M])
    F = np.array([-m[:3, 2] for m in M])  # ARKit / ARCore cameras look down their -Z axis
    # arm centre: the point closest to all viewing rays
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, f in zip(C, F):
        P = np.eye(3) - np.outer(f, f); A += P; b += P @ c
    centre = np.linalg.lstsq(A, b, rcond=None)[0]
    side = C - centre
    side /= np.linalg.norm(side, axis=1)[:, None] + 1e-9
    pairs = set()
    for i in range(len(names)):
        score = np.linalg.norm(C - C[i], axis=1) + 0.3 * (1 - F @ F[i])  # metres + view-direction penalty
        for j in np.argsort(score, kind='stable')[1:k + 1]:
            pairs.add((min(i, int(j)), max(i, int(j))))
        for j in np.argsort(-(side @ side[i]), kind='stable')[1:k_side + 1]:
            pairs.add((min(i, int(j)), max(i, int(j))))
        for j in (i + 1, i + 2):
            if j < len(names):
                pairs.add((i, j))
    path = job.root / 'colmap' / 'pairs.txt'
    path.write_text('\n'.join(f'{names[a]} {names[b]}' for a, b in sorted(pairs)) + '\n')
    job.log(f'[match] {len(pairs)} photo pairs chosen from the AR camera path (all pairs: {len(names) * (len(names) - 1) // 2})')
    return path


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
        # cameras placed from the phone's poses: photos and phone agree by construction, the whole path gives the scale
        s, R, t, stats = recover_ar(rec, metas, log=job.log, local=False if getattr(job, 'ar_mapped', False) else None)
        if ROT_SCALE and not getattr(job, 'ar_mapped', False) and all('pose' in m and m.get('K') for m in metas.values()):
            # size corrected by how much the photos' camera turns exceed the gyro's (see arscale.rotation_ratio):
            # 16 printed-hand scans, mean size error 1.24 % -> 0.5 %, worst 4.3 % -> 1.6 %
            from .arscale import ROT_OFFSET, rotation_ratio
            r = rotation_ratio(rec, metas)
            if r is not None and 0.97 < r < ROT_MAX:
                # only within the range validated against the printed hand's true size (photo turns up to +3.5 %); a
                # larger value is not a lens error: on a real hand at +5.1 % the uncorrected wrist matched the tape
                # exactly (162 mm) and the corrected one was 3.6 % too big (1 Nov)
                c = r - ROT_OFFSET
                s *= c
                t = t * c
                stats['scale_mm_per_unit'] = float(s)
                stats['rotation_ratio'] = r
                job.log(f'[ar] photos turn {100 * (r - 1):+.1f} % vs the gyro: size corrected by {100 * (c - 1):+.1f} %')
            elif r is not None:
                job.log(f'[ar] photos turn {100 * (r - 1):+.1f} % vs the gyro: outside the validated range, no correction')
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
    level = int(os.environ.get('LIMBSCAN_MVS_LEVEL', '1'))
    cropped = bool(mvs_mask_dir) and os.environ.get('LIMBSCAN_MVS_CROP', '1') == '1'
    if cropped:  # stereo only on the limb's bounding box, at the stereo resolution: same result, less work
        _crop_workspace(dense, 0.5 ** level)
        job.log(f'[dense] photos cropped to the limb and scaled to {100 * 0.5 ** level:.0f}%')
    job.status('dense reconstruction', 0.45)
    if USE_GPU and os.environ.get('LIMBSCAN_GPU_STEREO') == '1':  # CUDA PatchMatch (full-res, slower than OpenMVS level 1 here)
        colmap(job, ['patch_match_stereo', '--workspace_path', _in_job(job, dense), '--workspace_format', 'COLMAP',
                     '--PatchMatchStereo.geom_consistency', 'true', '--PatchMatchStereo.max_image_size', '2000',
                     '--PatchMatchStereo.filter_min_ncc', '0.1'])
        colmap(job, ['stereo_fusion', '--workspace_path', _in_job(job, dense), '--workspace_format', 'COLMAP',
                     '--input_type', 'geometric', '--output_path', _in_job(job, dense / 'scene_dense.ply'),
                     '--StereoFusion.min_num_pixels', '3'])
        return dense
    if cropped and MVS_ENGINE != 'openmvs':
        from . import gpumvs
        if gpumvs.available():
            gpumvs.densify(dense, log=job.log)
            return dense
        job.log('[dense] no OpenCL GPU: using OpenMVS')
    new = 'openmvs-ubuntu' not in _docker_image()
    mvs(job, ['InterfaceCOLMAP', '-i', '.', '-o', 'scene.mvs', '--image-folder', 'images'], dense_name)
    dargs = ['DensifyPointCloud', 'scene.mvs', '--resolution-level', '0' if cropped else str(level),
             '--min-resolution', '160' if cropped else '640',
             '--number-views', os.environ.get('LIMBSCAN_MVS_VIEWS', '5'), '--number-views-fuse', '3',
             '--estimate-normals', '0', '-v', '1']
    if new:
        dargs += ['--filter-point-cloud', '1', '--tower-mode', '0']
    mvs(job, dargs, dense_name)
    return dense


def _crop_workspace(dense, scale):
    """Crop every undistorted (background-greyed) photo of a COLMAP dense workspace to the limb's bounding box and
    scale it, with one PINHOLE camera per photo (principal point shifted, focal scaled). Rewrites the text model
    (2D observations moved too); binary/rig files are removed so every reader uses the cropped text model."""
    from concurrent.futures import ThreadPoolExecutor
    sp = dense / 'sparse'
    cams = {}
    for line in (sp / 'cameras.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        assert p[1] == 'PINHOLE', p[1]
        cams[int(p[0])] = tuple(map(float, p[4:8]))
    lines = [ln for ln in (sp / 'images.txt').read_text().splitlines() if not ln.startswith('#')]
    heads, obs = lines[0::2], lines[1::2]

    def crop(h):
        name = h.split()[9]
        path = dense / 'images' / name
        im = cv2.imread(str(path))
        d = np.abs(im.astype(np.int16) - 127).max(2)
        m = ((d > 10) & (im.sum(2) > 30)).astype(np.uint8)  # not the grey background, not the black border
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(m)
        H, W = m.shape
        if len(xs) == 0:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, y0 = max(int(xs.min()) - 16, 0), max(int(ys.min()) - 16, 0)
            x1, y1 = min(int(xs.max()) + 17, W), min(int(ys.max()) + 17, H)
        c = im[y0:y1, x0:x1]
        if scale != 1.0:
            c = cv2.resize(c, (max(1, round(c.shape[1] * scale)), max(1, round(c.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return x0, y0, c.shape[1], c.shape[0]

    with ThreadPoolExecutor(THREADS) as ex:
        boxes = list(ex.map(crop, heads))
    cl, il = [], []
    for k, (h, ob, (x0, y0, w, hh)) in enumerate(zip(heads, obs, boxes)):
        p = h.split()
        fx, fy, cx, cy = cams[int(p[8])]
        cl.append(f'{k + 1} PINHOLE {w} {hh} {fx * scale!r} {fy * scale!r} {(cx - x0) * scale!r} {(cy - y0) * scale!r}')
        p[8] = str(k + 1)
        q = ob.split()
        for j in range(0, len(q), 3):
            q[j] = repr((float(q[j]) - x0) * scale)
            q[j + 1] = repr((float(q[j + 1]) - y0) * scale)
        il += [' '.join(p), ' '.join(q)]
    (sp / 'cameras.txt').write_text('\n'.join(cl) + '\n')
    (sp / 'images.txt').write_text('\n'.join(il) + '\n')
    for f in list(sp.glob('*.bin')) + [sp / 'rigs.txt', sp / 'frames.txt']:
        f.unlink(missing_ok=True)


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
                for stale in ('bad_masks.json', 'masks_tracked', 'hand_cover.json'):  # from an earlier run of this scan
                    (job.root / 'colmap' / stale).unlink(missing_ok=True)
                shutil.rmtree(job.root / 'thumbs', ignore_errors=True)  # photo previews with the old cut-outs
                from .segment import build_masks
                from .prefetch import cached_masks
                arm_file = job.root / 'arm.json'
                arm = json.loads(arm_file.read_text()) if arm_file.exists() else None
                metas = _ar_metas(job)
                if metas and any('pose' in m for m in metas.values()):
                    # the arm from the hand in every photo, triangulated with the photos' AR poses: does not depend
                    # on the phone's live placement (one view cannot see the arm's tilt towards it)
                    from .handarm import arm_from_photos
                    try:
                        tri, dets = arm_from_photos(image_dir, metas, near=arm, log=job.log)
                        (job.root / 'hand_landmarks.json').write_text(json.dumps(dets))
                    except Exception as e:  # noqa: BLE001
                        job.log(f'[arm] hand triangulation failed: {e}')
                        tri = None
                    if tri is not None:
                        if arm is not None:
                            d = np.linalg.norm((np.asarray(tri['a']) + tri['b']) / 2 - (np.asarray(arm['a']) + arm['b']) / 2)
                            job.log(f'[arm] the phone placed it {d * 1000:.0f} mm from where the photos put it')
                        arm = tri
                        (job.root / 'arm_photos.json').write_text(json.dumps(arm))
                if arm is not None:
                    # cut every photo out with the final arm, not with the guess each photo was sent with (so the
                    # masks made during the upload are not used)
                    from .segment import arm_seg, hand_offsets, shift_seg
                    lmf = job.root / 'hand_landmarks.json'
                    dets = json.loads(lmf.read_text()) if lmf.exists() else {}
                    offs = hand_offsets(metas, arm, dets, sorted(p.name for p in Path(image_dir).glob('*.jpg'))) \
                        if SEG_SHIFT else {}
                    if offs:
                        big = [n for n, o in offs.items() if np.hypot(o[0] * 1080, o[1] * 1920) > 150]
                        job.log(f'[segment] outline moved onto the hand found in each photo (AR drift over 150 px in '
                                f'{len(big)} photos)')

                    def seg_of(n):
                        m = metas.get(n)
                        if not m or 'pose' not in m:
                            return None
                        w, h = m.get('stored_w', m.get('w', 1080)), m.get('stored_h', m.get('h', 1920))
                        s = arm_seg(m, arm, w, h)
                        s = shift_seg(s, offs.get(n)) if s else m.get('seg')
                        if s is not None and dets.get(n):
                            s = {**s, 'hand': dets[n]}  # the hand found in this photo: fingers the cut-out missed
                        return s
                    cached = None
                    job.fresh_masks = True
                    job.log(f'[segment] cut out with the final arm placement (r {arm["r"] * 1000:.0f} mm)')
                else:
                    seg_of = lambda n: _ar_metas(job).get(n, {}).get('seg')  # noqa: E731
                    cached = lambda n: cached_masks(job.root, n)  # noqa: E731
                seg = build_masks(image_dir, job.root / 'colmap' / 'mask_sfm', job.root / 'colmap' / 'mask_mvs', log=job.log,
                                  cards=not _ar_metas(job), cached=cached, seg=seg_of)
                lmf = job.root / 'hand_landmarks.json'
                if lmf.exists():
                    # the limb tracked through the whole scan (SAM 2 video, GPU) over the colour cut-outs
                    from . import sam
                    if sam.video_available():
                        try:
                            sam.video_masks(job, image_dir, json.loads(lmf.read_text()),
                                            job.root / 'colmap' / 'mask_mvs', log=job.log)
                        except Exception as e:  # noqa: BLE001
                            job.log(f'[samvideo] failed ({e}): keeping the colour cut-out')
                if arm is not None and lmf.exists():
                    # the real outline of the hand and forearm (SAM 2 on the GPU) over the colour cut-out
                    from . import sam
                    if sam.available():
                        try:
                            sam.masks(job, image_dir, arm, metas, json.loads(lmf.read_text()),
                                      job.root / 'colmap' / 'mask_mvs', log=job.log)
                        except Exception as e:  # noqa: BLE001
                            job.log(f'[sam] failed ({e}): keeping the colour cut-out')
                (job.root / 'segment.json').write_text(json.dumps(seg))
                # the photo normals on the GPU in the background, after the cut-outs (which use the GPU too) and while
                # the cameras, stereo and outlines are worked out on the CPU
                from . import shape as _shape
                _shape.normals_start(job)
                mask_dir = job.root / 'colmap' / 'mask_sfm'
            job.status('feature extraction', 0.12)
            rec = run_sfm(job, image_dir, mask_dir)
            n_in = len(list(Path(image_dir).glob('*.jpg')))
            # (only when the limb alone places too few photos: a person's arm sways a little relative to the room, which
            # the limb-only cameras follow and room-based cameras would not; a still object has no such problem)
            still = False
            if mask_dir and 0.5 * n_in <= rec.num_reg_images() < 0.9 * n_in:
                # some photos missing: if the limb held still relative to the room (limb-only cameras agree with the
                # phone's tracking), the room places every photo and costs nothing in accuracy (printed hand: 49/65
                # placed from the limb alone gave a fat outline model)
                try:
                    from .arscale import recover_ar
                    still = recover_ar(rec, _ar_metas(job), log=lambda *a: None)[3]['ar_residual_median_mm'] < 12
                except Exception:  # noqa: BLE001
                    still = False
            if os.environ.get('LIMBSCAN_LIMB_ONLY') == '1':  # (experiment: never fall back to the room)
                still = False
            if mask_dir and (rec.num_reg_images() < 0.5 * n_in or still):
                # too few features on the limb alone (plain plastic, smooth skin): place the cameras from the whole
                # photo (the room around the still limb); the limb masks still cut the dense model
                job.log(f'[sfm] only {rec.num_reg_images()}/{n_in} photos placed from the limb alone: using the whole photos')
                first, first_pieces = rec, job.pieces
                job.status('camera poses from the whole photos', 0.2)
                rec = run_sfm(job, image_dir, None)
                if rec.num_reg_images() <= first.num_reg_images():
                    rec, job.pieces = first, first_pieces
            # (native app only: its poses and lens 'K' follow the convention _ar_mapping assumes; the old browser
            # capture's poses do not, and the rescue built garbage from them)
            if RESCUE and rec.num_reg_images() < 0.9 * n_in and _ar_metas(job) \
                    and all('pose' in m and m.get('K') for m in _ar_metas(job).values()):
                # the photos alone could not place every photo (the scan split: 31 Oct printed hand, 27 + 9 of 40,
                # the whole palm side lost): every camera from the phone's pose instead, refined on the photos
                job.log(f'[sfm] only {rec.num_reg_images()}/{n_in} photos placed: cameras from the phone instead')
                job.pieces = []
                rec = run_sfm(job, image_dir, mask_dir, mapper='ar')
            job.status('metric scale', 0.35)
            rec, scale_stats = apply_metric(job, rec, image_dir)
            scale_stats['mode'] = 'arm' if ARM_MODE else 'scene'
            if scale_stats.get('method') == 'ar' and scale_stats.get('ar_residual_median_mm', 0) > 50:
                # the photos' camera path and the phone's own tracking disagree by centimetres: the limb moved
                # during the scan (or the photos matched wrongly). Any measurement from this would be made up.
                raise RuntimeError(f"The arm moved during the scan (photos and phone tracking differ by "
                                   f"{scale_stats['ar_residual_median_mm']:.0f} mm). Scan again with the elbow resting "
                                   "on a table and the arm completely still.")
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
                except Exception as e:  # noqa: BLE001  an extra piece that fails (no points) must not fail the scan
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
