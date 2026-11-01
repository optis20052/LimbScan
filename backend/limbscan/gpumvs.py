"""GPU multi-view stereo for the limb (OpenCL: Intel / AMD / NVIDIA GPUs, or any OpenCL CPU driver).

PatchMatch stereo in the Gipuma / ACMH family, on the cropped, background-greyed workspace written by
pipeline._crop_workspace:

1. Every photo gets a depth + normal per limb pixel. Each pixel's hypothesis is a small slanted plane, scored by
   bilateral-weighted NCC over an 11x11 window (step 2) against its 5 best neighbour photos (the 3 best views
   count), improved by red-black propagation from neighbouring pixels and random refinement.
2. Fusion: a pixel's 3D point is kept when at least 2 neighbour depth maps agree with it (depth within 1 %,
   reprojection within 1 px, normals within 30 degrees); the agreeing points are averaged.

Randomness comes from a hash of (photo, pixel, iteration), never from a live generator, so the same photos always
give the same cloud. Replaces OpenMVS DensifyPointCloud (CPU, ~35 s) with a few seconds on a laptop iGPU.
"""
import time
from pathlib import Path

import cv2
import numpy as np

NSRC = 5          # neighbour photos per depth map
TOPK = 3          # best views that count in the cost
import os
ITERS_COARSE = int(os.environ.get('GPUMVS_ITERS_COARSE', 3))  # red + black sweeps at half the stereo resolution
ITERS = int(os.environ.get('GPUMVS_ITERS', 2))                # then at the stereo resolution
COST_MAX = float(os.environ.get('GPUMVS_COST_MAX', 0.55))   # 1 - NCC: weaker matches are not used in fusion
MIN_AGREE = 2     # other depth maps that must agree (with the reference: 3 views, like OpenMVS fuse 3)
VOXEL = 0.4       # mm: duplicate merging of the fused cloud

KERNELS = r"""
#define NSRC 5
#ifndef BILAT
#define BILAT 12.0f
#endif
__constant sampler_t S = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__constant sampler_t SN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

uint hash(uint x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
float rnd(uint seed, uint a, uint b, uint c) {
    return (float)(hash(seed ^ hash(a ^ hash(b ^ hash(c)))) & 0xFFFFFF) / 16777216.0f;
}

typedef struct { float R[9]; float t[3]; float K[4]; float wh[2]; } View;  // ref -> src transform, src intrinsics

float3 ray(float2 p, float4 K) { return (float3)((p.x - K.z) / K.x, (p.y - K.w) / K.y, 1.0f); }

float3 random_normal(uint seed, uint a, uint b, uint c, float3 r0) {
    float u = rnd(seed, a, b, c), v = rnd(seed, a, b, c + 101u);
    float z = 2.0f * u - 1.0f, phi = 6.2831853f * v, s = sqrt(max(0.0f, 1.0f - z * z));
    float3 n = (float3)(s * cos(phi), s * sin(phi), z);
    if (dot(n, r0) > 0.0f) n = -n;  // faces the camera
    return n;
}

// 1 - NCC of the plane's window between the reference and one source photo (2 = unusable)
float view_cost(image2d_t ref, image2d_t src, __constant View *vw, float4 K, float2 p, float3 n, float q, float c0) {
    float sw = 0.0f, sr = 0.0f, ss = 0.0f, srr = 0.0f, sss = 0.0f, srs = 0.0f;
    int bad = 0;
    for (int dy = -5; dy <= 5; dy += 2) {
        for (int dx = -5; dx <= 5; dx += 2) {
            float2 pp = p + (float2)(dx, dy);
            float4 rp = read_imagef(ref, S, pp);
            if (rp.y < 0.5f) continue;              // background in the reference: not part of the patch
            float cr = rp.x;
            float w = native_exp(-fabs(cr - c0) * BILAT - (dx * dx + dy * dy) * 0.02f);
            float3 rr = ray(pp, K);
            float3 X = rr * (q / dot(n, rr));
            float3 Y = (float3)(vw->R[0] * X.x + vw->R[1] * X.y + vw->R[2] * X.z + vw->t[0],
                                vw->R[3] * X.x + vw->R[4] * X.y + vw->R[5] * X.z + vw->t[1],
                                vw->R[6] * X.x + vw->R[7] * X.y + vw->R[8] * X.z + vw->t[2]);
            float iz = native_recip(Y.z);
            float2 u = (float2)(vw->K[0] * Y.x * iz + vw->K[2], vw->K[1] * Y.y * iz + vw->K[3]);
            float4 cs = read_imagef(src, S, u);
            if (Y.z <= 1e-3f || cs.y < 0.5f || u.x < 1.0f || u.y < 1.0f || u.x > vw->wh[0] - 1.0f || u.y > vw->wh[1] - 1.0f) {
                bad++;
                continue;
            }
            sw += w; sr += w * cr; ss += w * cs.x;
            srr += w * cr * cr; sss += w * cs.x * cs.x; srs += w * cr * cs.x;
        }
    }
    if (bad > 9 || sw <= 0.0f) return 2.0f;   // > 25 % of the window off the limb
    float mr = sr / sw, ms = ss / sw;
    float vr = srr / sw - mr * mr, vs = sss / sw - ms * ms, cv = srs / sw - mr * ms;
    if (vr < 1e-6f || vs < 1e-6f) return 2.0f;  // textureless
    return clamp(1.0f - cv * native_rsqrt(vr * vs), 0.0f, 2.0f);
}

// mean cost of the TOPK (3) best of the source views for the plane (n, depth at pixel p)
float plane_cost(image2d_t ref, image2d_t s0, image2d_t s1, image2d_t s2, image2d_t s3, image2d_t s4,
                 __constant View *V, int nsrc, float4 K, float2 p, float3 n, float depth) {
    float3 r0 = ray(p, K);
    if (dot(n, r0) > -1e-4f) return 2.0f;
    float q = dot(n, r0 * depth);
    float c0 = read_imagef(ref, S, p).x;
    float c[5] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    c[0] = view_cost(ref, s0, &V[0], K, p, n, q, c0);
    if (nsrc > 1) c[1] = view_cost(ref, s1, &V[1], K, p, n, q, c0);
    if (nsrc > 2) c[2] = view_cost(ref, s2, &V[2], K, p, n, q, c0);
    if (nsrc > 3) c[3] = view_cost(ref, s3, &V[3], K, p, n, q, c0);
    if (nsrc > 4) c[4] = view_cost(ref, s4, &V[4], K, p, n, q, c0);
    // sum of the 3 smallest (sorting network on 5 values)
    float a0 = c[0], a1 = c[1], a2 = c[2], a3 = c[3], a4 = c[4], t;
#define SW(x, y) if (x > y) { t = x; x = y; y = t; }
    SW(a0, a1); SW(a3, a4); SW(a2, a4); SW(a2, a3); SW(a0, a3); SW(a0, a2); SW(a1, a4); SW(a1, a3); SW(a1, a2);
    if (a0 >= 2.0f) return 2.0f;
    if (a1 >= 2.0f) return 0.5f * (a0 + 2.0f);          // a single usable view: penalised
    if (a2 >= 2.0f) return 0.5f * (a0 + a1);
    return (a0 + a1 + a2) * (1.0f / 3.0f);
}

__kernel void init(image2d_t ref, image2d_t s0, image2d_t s1, image2d_t s2, image2d_t s3, image2d_t s4,
                   __constant View *V, int nsrc, float4 K, float dmin, float dmax, uint seed,
                   int W, int H, __global float4 *plane, __global float *cost) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int i = y * W + x;
    float2 p = (float2)(x + 0.5f, y + 0.5f);
    if (read_imagef(ref, SN, p).y < 0.5f) { plane[i] = (float4)(0, 0, -1, 0); cost[i] = 2.0f; return; }
    float3 r0 = ray(p, K);
    float d = dmin + (dmax - dmin) * rnd(seed, x, y, 1u);
    float3 n = random_normal(seed, x, y, 7u, r0);
    plane[i] = (float4)(n, d);
    cost[i] = plane_cost(ref, s0, s1, s2, s3, s4, V, nsrc, K, p, n, d);
}

__kernel void sweep(image2d_t ref, image2d_t s0, image2d_t s1, image2d_t s2, image2d_t s3, image2d_t s4,
                    __constant View *V, int nsrc, float4 K, float dmin, float dmax, uint seed, int colour, int iter,
                    int W, int H, __global float4 *plane, __global float *cost) {
    int y = get_global_id(1);
    int x = 2 * get_global_id(0) + ((y + colour) & 1);   // only this colour's pixels are launched
    if (x >= W || y >= H) return;
    int i = y * W + x;
    if (cost[i] >= 2.0f && plane[i].w == 0.0f) return;  // not limb
    float2 p = (float2)(x + 0.5f, y + 0.5f);
    float3 r0 = ray(p, K);
    float4 best = plane[i];
    float bc = cost[i];
    const int ox[8] = {-1, 1, 0, 0, -5, 5, 0, 0};
    const int oy[8] = {0, 0, -1, 1, 0, 0, -5, 5};
    for (int k = 0; k < 8; k++) {   // propagation: the neighbour's plane, extended to this pixel
        int xx = x + ox[k], yy = y + oy[k];
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        float4 pn = plane[yy * W + xx];
        if (pn.w <= 0.0f) continue;
        float3 n = pn.xyz;
        float3 rn = ray((float2)(xx + 0.5f, yy + 0.5f), K);
        float den = dot(n, r0);
        if (den >= -1e-4f) continue;
        float d = dot(n, rn * pn.w) / den;
        if (d < dmin || d > dmax) continue;
        float c = plane_cost(ref, s0, s1, s2, s3, s4, V, nsrc, K, p, n, d);
        if (c < bc) { bc = c; best = (float4)(n, d); }
    }
    // refinement: random plane, then perturbations that shrink with the iteration
    float sc = pow(0.5f, (float)iter);
    for (int k = 0; k < 4; k++) {
        float3 n = best.xyz;
        float d = best.w;
        uint tag = (uint)(iter * 16 + k) * 2u + 100u;
        if (k == 0) {
            d = dmin + (dmax - dmin) * rnd(seed, x, y, tag);
            n = random_normal(seed, x, y, tag + 1u, r0);
        } else {
            if (k != 2) d = d * (1.0f + 0.02f * sc * (2.0f * rnd(seed, x, y, tag) - 1.0f));
            if (k != 1) {
                float3 e = (float3)(rnd(seed, x, y, tag + 3u), rnd(seed, x, y, tag + 5u), rnd(seed, x, y, tag + 7u));
                n = normalize(n + (e * 2.0f - 1.0f) * 0.25f * sc);
                if (dot(n, r0) > 0.0f) n = -n;
            }
        }
        if (d < dmin || d > dmax) continue;
        float c = plane_cost(ref, s0, s1, s2, s3, s4, V, nsrc, K, p, n, d);
        if (c < bc) { bc = c; best = (float4)(n, d); }
    }
    plane[i] = best;
    cost[i] = bc;
}

// coarse-to-fine: start each fine pixel from the plane found at half resolution (same plane, re-intersected)
__kernel void upsample(image2d_t ref, image2d_t s0, image2d_t s1, image2d_t s2, image2d_t s3, image2d_t s4,
                       __constant View *V, int nsrc, float4 K, float dmin, float dmax, uint seed,
                       int W, int H, __global const float4 *coarse, int Wc, int Hc,
                       __global float4 *plane, __global float *cost) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int i = y * W + x;
    float2 p = (float2)(x + 0.5f, y + 0.5f);
    if (read_imagef(ref, SN, p).y < 0.5f) { plane[i] = (float4)(0, 0, -1, 0); cost[i] = 2.0f; return; }
    float3 r0 = ray(p, K);
    int cx = min(x / 2, Wc - 1), cy = min(y / 2, Hc - 1);
    float4 pc = coarse[cy * Wc + cx];
    float3 n = pc.xyz;
    float d = -1.0f;
    if (pc.w > 0.0f) {
        float4 Kc = K * 0.5f;
        float3 Xc = ray((float2)(cx + 0.5f, cy + 0.5f), Kc) * pc.w;   // same camera frame at both scales
        float den = dot(n, r0);
        if (den < -1e-4f) d = dot(n, Xc) / den;
    }
    if (d < dmin || d > dmax) {
        d = dmin + (dmax - dmin) * rnd(seed, x, y, 3u);
        n = random_normal(seed, x, y, 9u, r0);
    }
    plane[i] = (float4)(n, d);
    cost[i] = plane_cost(ref, s0, s1, s2, s3, s4, V, nsrc, K, p, n, d);
}
"""


def _read_workspace(dense):
    import pycolmap
    rec = pycolmap.Reconstruction(Path(dense) / 'sparse')
    views = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        T = im.cam_from_world()
        views.append({'name': im.name, 'id': im.image_id, 'K': np.array([fx, fy, cx, cy], np.float64),
                      'R': T.rotation.matrix(), 't': np.asarray(T.translation, np.float64),
                      'w': cam.width, 'h': cam.height})
    return rec, views


def _neighbours(rec, views):
    """For each photo, the NSRC photos sharing the most sparse points at a useful baseline (5-45 degrees)."""
    index = {v['id']: k for k, v in enumerate(views)}
    C = np.array([-v['R'].T @ v['t'] for v in views])
    n = len(views)
    score = np.zeros((n, n))
    for p in rec.points3D.values():
        ks = sorted({index[e.image_id] for e in p.track.elements if e.image_id in index})
        if len(ks) < 2:
            continue
        X = p.xyz
        d = C[ks] - X
        d /= np.linalg.norm(d, axis=1)[:, None]
        ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
        w = np.exp(-((ang - 15.0) / np.where(ang < 15.0, 7.0, 20.0)) ** 2) * (ang > 3.0)
        score[np.ix_(ks, ks)] += w
    np.fill_diagonal(score, 0)
    return [list(np.argsort(-score[i], kind='stable')[:NSRC][score[i][np.argsort(-score[i], kind='stable')[:NSRC]] > 1.0])
            for i in range(n)]


def _depth_range(rec, v):
    zs = []
    for p2 in rec.images[v['id']].points2D:
        if p2.has_point3D():
            X = rec.points3D[p2.point3D_id].xyz
            zs.append((v['R'] @ X + v['t'])[2])
    if len(zs) < 10:
        return None
    zs = np.array(zs)
    return float(np.percentile(zs, 1) * 0.8), float(np.percentile(zs, 99) * 1.2)


def _load(dense, v):
    bgr = cv2.imread(str(Path(dense) / 'images' / v['name']))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    d = np.abs(bgr.astype(np.int16) - 127).max(2)
    m = ((d > 10) & (bgr.astype(np.int32).sum(2) > 30)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32)
    img = np.zeros(gray.shape + (4,), np.float32)
    img[..., 0], img[..., 1] = gray, m
    return bgr, img


def densify(dense, log=print, seed=12345):
    """Depth maps for every photo of the cropped workspace on the GPU, fused into dense/scene_dense.ply."""
    import open3d as o3d
    import pyopencl as cl
    t0 = time.time()
    dense = Path(dense)
    rec, views = _read_workspace(dense)
    nbrs = _neighbours(rec, views)
    dev = _gpu()
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNELS).build(options='-cl-fast-relaxed-math -cl-mad-enable -DBILAT=' + os.environ.get('GPUMVS_BILAT', '12.0') + 'f')
    t_build = time.time() - t0
    k_init, k_sweep, k_up = cl.Kernel(prg, 'init'), cl.Kernel(prg, 'sweep'), cl.Kernel(prg, 'upsample')
    fmt = cl.ImageFormat(cl.channel_order.RG, cl.channel_type.HALF_FLOAT)  # (grey, limb mask), 4 bytes/pixel
    mf = cl.mem_flags
    imgs, cols, gimg, gimg_c = [], [], [], []
    keep = []  # host copies stay alive until the GPU is done (drivers may copy lazily from host memory)
    for v in views:
        bgr, img = _load(dense, v)
        cols.append(bgr)
        imgs.append(img)
        rg = np.ascontiguousarray(img[..., :2].astype(np.float16))
        keep.append(rg)
        gimg.append(cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, shape=(img.shape[1], img.shape[0]), hostbuf=rg))
        h2, w2 = (img.shape[0] + 1) // 2, (img.shape[1] + 1) // 2
        small = cv2.resize(img[..., :2], (w2, h2), interpolation=cv2.INTER_AREA)
        small[..., 1] = (small[..., 1] > 0.75).astype(np.float32)   # coarse mask: mostly limb
        small = np.ascontiguousarray(small.astype(np.float16))
        keep.append(small)
        gimg_c.append(cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, shape=(w2, h2), hostbuf=small))
    t_load = time.time() - t0
    up = lambda n, m: (n + m - 1) // m * m
    depth, normal, costs = [None] * len(views), [None] * len(views), [None] * len(views)
    vdt = [('R', np.float32, 9), ('t', np.float32, 3), ('K', np.float32, 4), ('wh', np.float32, 2)]
    for r, v in enumerate(views):
        rng = _depth_range(rec, v)
        src = nbrs[r]
        if rng is None or len(src) < 2:
            continue
        H, W = imgs[r].shape[:2]
        Hc, Wc = (H + 1) // 2, (W + 1) // 2
        arr, arr_c = np.zeros(NSRC, dtype=vdt), np.zeros(NSRC, dtype=vdt)
        for k, j in enumerate(src):
            Rr, tr, Rj, tj = v['R'], v['t'], views[j]['R'], views[j]['t']
            Rrel = Rj @ Rr.T
            Kj = views[j]['K']
            arr[k] = (Rrel.ravel(), tj - Rrel @ tr, Kj, (views[j]['w'], views[j]['h']))
            arr_c[k] = (Rrel.ravel(), tj - Rrel @ tr, Kj * 0.5, ((views[j]['w'] + 1) // 2, (views[j]['h'] + 1) // 2))
        Vb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)
        Vc = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_c)
        keep += [arr, arr_c]
        pad = [gimg[r]] * (NSRC - len(src))
        seed_r = np.uint32(seed + 7919 * r)
        rng32 = (np.float32(rng[0]), np.float32(rng[1]))
        # half resolution: random start, full search
        Kc = cl.cltypes.make_float4(*[float(x) * 0.5 for x in v['K']])
        com_c = (gimg_c[r], *[gimg_c[j] for j in src], *[gimg_c[r]] * (NSRC - len(src)), Vc, np.int32(len(src)), Kc,
                 *rng32, seed_r)
        plane_c = cl.Buffer(ctx, mf.READ_WRITE, Wc * Hc * 16)
        cost_c = cl.Buffer(ctx, mf.READ_WRITE, Wc * Hc * 4)
        k_init(q, (up(Wc, 16), up(Hc, 8)), (16, 8), *com_c, np.int32(Wc), np.int32(Hc), plane_c, cost_c)
        for it in range(ITERS_COARSE):
            for colour in (0, 1):
                k_sweep(q, (up((Wc + 1) // 2, 16), up(Hc, 8)), (16, 8), *com_c, np.int32(colour), np.int32(it),
                        np.int32(Wc), np.int32(Hc), plane_c, cost_c)
        # stereo resolution: start from the coarse planes, finer perturbations
        K = cl.cltypes.make_float4(*[float(x) for x in v['K']])
        com = (gimg[r], *[gimg[j] for j in src], *pad, Vb, np.int32(len(src)), K, *rng32, seed_r)
        plane = cl.Buffer(ctx, mf.READ_WRITE, W * H * 16)
        cost = cl.Buffer(ctx, mf.READ_WRITE, W * H * 4)
        k_up(q, (up(W, 16), up(H, 8)), (16, 8), *com, np.int32(W), np.int32(H), plane_c, np.int32(Wc), np.int32(Hc),
             plane, cost)
        for it in range(ITERS):
            for colour in (0, 1):
                k_sweep(q, (up((W + 1) // 2, 16), up(H, 8)), (16, 8), *com, np.int32(colour),
                        np.int32(ITERS_COARSE + it), np.int32(W), np.int32(H), plane, cost)
        pl = np.empty((H, W, 4), np.float32)
        cs = np.empty((H, W), np.float32)
        cl.enqueue_copy(q, pl, plane)
        cl.enqueue_copy(q, cs, cost)
        ok = (cs < COST_MAX) & (pl[..., 3] > 0) & (imgs[r][..., 1] > 0.5)
        depth[r] = np.where(ok, pl[..., 3], 0).astype(np.float32)
        normal[r] = pl[..., :3]
        costs[r] = cs
    q.finish()
    del keep
    t_dm = time.time() - t0 - t_load
    t1 = time.time()
    P, Ccol = _fuse(views, nbrs, depth, normal, cols)
    t_fuse = time.time() - t1
    n_fused = len(P)
    P, Ccol = _voxel_merge(P, Ccol, VOXEL)  # the same skin seen in many photos: merge duplicates (post uses 0.7 mm)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(Ccol[:, ::-1] / 255.0)
    o3d.io.write_point_cloud(str(dense / 'scene_dense.ply'), pc)
    log(f'[gpu-mvs] {dev.name.strip()}: {len(views)} depth maps in {t_dm:.1f}s (load {t_load:.1f}s), '
        f'{n_fused} fused points -> {len(P)} (fusion {t_fuse:.1f}s, build {t_build:.1f}s), total {time.time() - t0:.1f}s')
    return dense / 'scene_dense.ply'


def _fuse(views, nbrs, depth, normal, cols):
    """Keep pixels whose 3D point is confirmed by >= MIN_AGREE neighbour depth maps; average the agreeing points.
    One photo per thread (numpy releases the GIL); results are concatenated in photo order (repeatable)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        res = list(ex.map(lambda r: _fuse_one(r, views, nbrs, depth, normal, cols), range(len(views))))
    res = [x for x in res if x is not None]
    return np.vstack([a for a, _ in res]), np.vstack([b for _, b in res])


def _voxel_merge(P, C, size):
    """Average the points (and colours) in each voxel, in a fixed order (open3d's version sums in a varying order,
    so the cloud would differ in the last bits between runs)."""
    key = np.floor(P / size).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    order = np.argsort(lin, kind='stable')
    lin = lin[order]
    start = np.r_[0, np.nonzero(np.diff(lin))[0] + 1]
    cnt = np.diff(np.r_[start, len(lin)])[:, None]
    return np.add.reduceat(P[order], start) / cnt, np.add.reduceat(C[order], start) / cnt


def _mul(X, M):
    """X @ M for (n, 3) x (3, 3) with a fixed summation order (BLAS may split the work differently per run and
    change the last bit, which flips borderline points: the fused cloud would not be repeatable)."""
    return X[:, 0:1] * M[0] + X[:, 1:2] * M[1] + X[:, 2:3] * M[2]


def _fuse_one(r, views, nbrs, depth, normal, cols):
    v = views[r]
    if True:
        D = depth[r]
        if D is None:
            return None
        ys, xs = np.nonzero(D)
        if len(xs) == 0:
            return None
        z = D[ys, xs].astype(np.float64)
        K = v['K']
        Xc = np.stack([(xs + 0.5 - K[2]) / K[0] * z, (ys + 0.5 - K[3]) / K[1] * z, z], 1)
        Xw = _mul(Xc - v['t'], v['R'])           # R^T (Xc - t)
        Nw = _mul(normal[r][ys, xs].astype(np.float64), v['R'])
        acc = Xw.copy()
        cnt = np.ones(len(xs))
        for j in nbrs[r]:
            Dj = depth[j]
            if Dj is None:
                continue
            vj = views[j]
            Y = _mul(Xw, vj['R'].T) + vj['t']
            front = Y[:, 2] > 1e-3
            u = np.where(front, vj['K'][0] * Y[:, 0] / np.maximum(Y[:, 2], 1e-3) + vj['K'][2], -1)
            w_ = np.where(front, vj['K'][1] * Y[:, 1] / np.maximum(Y[:, 2], 1e-3) + vj['K'][3], -1)
            ui, vi = np.floor(u).astype(int), np.floor(w_).astype(int)
            inside = (ui >= 0) & (vi >= 0) & (ui < Dj.shape[1]) & (vi < Dj.shape[0])
            zj = np.zeros(len(xs))
            zj[inside] = Dj[vi[inside], ui[inside]]
            good = inside & (zj > 0) & (np.abs(Y[:, 2] - zj) < 0.01 * zj)
            # the neighbour's own point, back in world coordinates
            Kj = vj['K']
            Xj = np.stack([(ui + 0.5 - Kj[2]) / Kj[0] * zj, (vi + 0.5 - Kj[3]) / Kj[1] * zj, zj], 1)
            Xjw = _mul(Xj - vj['t'], vj['R'])
            # reprojection back into the reference within 1 px
            Yr = _mul(Xjw, v['R'].T) + v['t']
            ur = K[0] * Yr[:, 0] / np.maximum(Yr[:, 2], 1e-3) + K[2] - (xs + 0.5)
            vr = K[1] * Yr[:, 1] / np.maximum(Yr[:, 2], 1e-3) + K[3] - (ys + 0.5)
            good &= (ur * ur + vr * vr) < 1.0
            Nj = _mul(normal[j][np.clip(vi, 0, Dj.shape[0] - 1), np.clip(ui, 0, Dj.shape[1] - 1)].astype(np.float64), vj['R'])
            good &= (Nw * Nj).sum(1) > np.cos(np.radians(30))
            acc[good] += Xjw[good]
            cnt[good] += 1
        keep = cnt >= 1 + MIN_AGREE
        return acc[keep] / cnt[keep][:, None], cols[r][ys[keep], xs[keep]].astype(np.float64)


def _gpu():
    import pyopencl as cl
    devs = [d for p in cl.get_platforms() for d in p.get_devices() if d.type & cl.device_type.GPU]
    if not devs:
        raise RuntimeError('no OpenCL GPU')
    return devs[0]


def available():
    try:
        _gpu()
        return True
    except Exception:
        return False
