"""Render a synthetic, metrically-known photogrammetry capture of a reference limb mesh.

Usage: python tools/synth.py <reference.obj> <out_dir> [n_per_ring]
Writes out_dir/images/*.jpg, out_dir/gt_mesh.ply (mm, scene frame), out_dir/gt_cameras.json
"""
import sys, os, json
import numpy as np, cv2, trimesh, open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from limbscan.meshio_utils import load_mm, pca_frame
from limbscan import markers

W, H, F = 1920, 1440, 1450.0
TEX = float(os.environ.get('SYNTH_TEX', 1.0))      # skin texture contrast multiplier
NOISE = float(os.environ.get('SYNTH_NOISE', 0.008))
BLUR = float(os.environ.get('SYNTH_BLUR', 0.6))
JPEGQ = int(os.environ.get('SYNTH_JPEG', 92))
REF = os.environ.get('SYNTH_REF', 'sheets')  # 'sheets' (printed ArUco) or 'card' (ID-1 card on a wooden table)
CARD = (-60.0, -165.0, 85.60, 53.98, 3.18)  # x0, y0, w, h, corner radius (mm), on the table
rng = np.random.default_rng(7)
PERM = rng.permutation(4096)


def vnoise(p, scale):
    """3D value noise in [-1,1] at points p (N,3) with feature size `scale` mm."""
    q = p / scale
    i = np.floor(q).astype(np.int64); f = q - i
    f = f * f * (3 - 2 * f)
    def h(ix, iy, iz):
        return PERM[(PERM[(PERM[ix & 4095] + iy) & 4095] + iz) & 4095] / 2047.5 - 1.0
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * h(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
    return out


def skin(p):
    n = TEX * (0.10 * vnoise(p, 14) + 0.07 * vnoise(p + 31, 5) + 0.06 * vnoise(p + 77, 1.6) + 0.04 * vnoise(p + 5, 0.7))
    spots = np.clip((vnoise(p + 113, 2.2) - 0.55) * 4, 0, 1) * 0.18 * TEX  # freckles / pores
    base = np.array([0.82, 0.62, 0.52])
    return np.clip(base[None] * (1 + n[:, None]) * (1 - spots[:, None]), 0, 1)


def table_card(p):
    """Wood table with an ISO ID-1 card (blue bank-card look, rounded corners) lying on it."""
    x, y = p[:, 0], p[:, 1]
    grain = 0.5 + 0.5 * np.sin(x * 0.35 + 6 * vnoise(p * np.array([0.2, 1, 1]), 9))
    col = np.array([0.55, 0.37, 0.22])[None] * (0.8 + 0.25 * grain[:, None] + 0.12 * vnoise(p, 3)[:, None])
    x0, y0, w, h, r = CARD
    u, v = x - x0, y - y0
    dx = np.maximum(np.maximum(r - u, u - (w - r)), 0); dy = np.maximum(np.maximum(r - v, v - (h - r)), 0)
    inside = (u >= 0) & (u <= w) & (v >= 0) & (v <= h) & (dx ** 2 + dy ** 2 <= r ** 2)
    cc = np.tile(np.array([0.10, 0.24, 0.58]), (inside.sum(), 1))
    ui, vi = u[inside], v[inside]
    cc[(vi > 30) & (vi < 37)] = [0.85, 0.88, 0.95]                                  # stripe
    cc[(ui > 8) & (ui < 20) & (vi > 12) & (vi < 22)] = [0.85, 0.70, 0.30]           # chip
    txt = (ui > 8) & (ui < 70) & (vi > 42) & (vi < 47) & (np.sin(ui * 2.2) > 0.2)    # embossed digits
    cc[txt] = [0.9, 0.9, 0.9]
    col[inside] = cc
    return np.clip(col, 0, 1)


def look_at(c, target, upref):
    f = target - c; f /= np.linalg.norm(f)
    r = np.cross(f, upref); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])  # world->cam rows (x right, y down, z forward)
    return R, -R @ c


SWAY = float(os.environ.get('SYNTH_SWAY', 0))         # deg: limb rotates relative to the room (patient can't stay still)
SHIFT = float(os.environ.get('SYNTH_SHIFT', 0))       # mm: limb translation amplitude
CARD_ON_ARM = int(os.environ.get('SYNTH_CARD_ON_ARM', 0))  # first N frames: ID-1 card resting on the forearm


def _rot(ax, deg):
    ax = np.asarray(ax, float); ax /= np.linalg.norm(ax); a = np.deg2rad(deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def arm_pose(f, pivot):
    """Smooth pseudo-random rigid motion of the limb at frame f (4x4, arm-local -> world)."""
    R = _rot([1, 0, 0], SWAY * np.sin(0.13 * f + 0.5)) @ _rot([0, 1, 0], SWAY * np.sin(0.07 * f + 1.3)) @ _rot([0, 0, 1], SWAY * np.sin(0.11 * f + 2.0))
    t = SHIFT * np.array([np.sin(0.05 * f), np.sin(0.09 * f + 1), np.sin(0.06 * f + 2)])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pivot - R @ pivot + t
    return T


def card_on_forearm(arm):
    """Card corners (arm-local) resting on the top of the mid-forearm, long side along the limb."""
    v, n = arm.vertices, arm.vertex_normals
    x0 = np.percentile(v[:, 0], 40)
    cand = np.where((np.abs(v[:, 0] - x0) < 15) & (n[:, 2] > 0.7))[0]
    p0 = v[cand[np.argmax(v[cand, 2])]]
    u = np.array([1.0, 0, 0]); w = np.array([0, 1.0, 0])
    c = p0 + np.array([0, 0, 1.2])  # rigid card touches the top of the curved forearm
    hw, hh = CARD[2] / 2, CARD[3] / 2
    return np.array([c - hw * u - hh * w, c + hw * u - hh * w, c + hw * u + hh * w, c - hw * u + hh * w])


def card_texture(uv):
    """uv in mm from the card's corner (u along 85.6, v along 53.98); same look as the table card."""
    x0, y0, w, h, r = CARD
    fake = np.c_[uv[:, 0] + x0, uv[:, 1] + y0, np.zeros(len(uv))]
    return table_card(fake)


def main(ref, out, n_per_ring=32):
    os.makedirs(f'{out}/images', exist_ok=True)
    arm = load_mm(ref)
    c, R = pca_frame(arm.vertices)
    arm.apply_translation(-c); arm.apply_transform(np.vstack([np.c_[R, [0, 0, 0]], [0, 0, 0, 1]]))
    arm.apply_translation([0, 0, 200.0])
    arm.export(f'{out}/gt_mesh.ply')
    pivot = arm.vertices[np.argmax(arm.vertices[:, 0])]
    card3d = card_on_forearm(arm) if CARD_ON_ARM else None
    sheets = []
    for k, (ox, oy) in enumerate([(-315, -148.5), (-105, -148.5), (105, -148.5)]):
        sheets.append((k, ox, oy, markers.render_sheet(k, dpmm=8)))
    mat_v = np.array([[-315, -148.5, 0], [315, -148.5, 0], [315, 148.5, 0], [-315, 148.5, 0]], float)
    if REF == 'card':
        mat_v = np.array([[-550, -420, 0], [550, -420, 0], [550, 420, 0], [-550, 420, 0]], float)
    bg = o3d.geometry.TriangleMesh.create_sphere(4000, 40); bg.compute_vertex_normals()
    tri = lambda V, Fc: o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(Fc)))
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dirs_c = np.stack([(uu - W / 2) / F, (vv - H / 2) / F, np.ones_like(uu)], -1).reshape(-1, 3)
    sun = np.array([0.3, -0.4, 1.0]); sun /= np.linalg.norm(sun)
    # camera plan: optional calibration views from above (card on the arm), then 3 rings around the limb
    views = []
    for k in range(CARD_ON_ARM):
        phi = np.deg2rad(-35 + 70 * k / max(CARD_ON_ARM - 1, 1))
        views.append((np.array([-40 + 50 * np.sin(k), 330 * np.sin(phi), 200 + 330 * np.cos(phi)]), np.array([-40, 0, 200.0]), True))
    span = 115 if REF == 'card' else 135  # with a real table, the phone can't go below its surface
    for ring, (xoff, rad) in enumerate([(-130, 400), (0, 360), (130, 400)]):
        for k in range(n_per_ring):
            phi = np.deg2rad(-span + 2 * span * k / (n_per_ring - 1)) + rng.normal(0, 0.03)
            views.append((np.array([xoff + rng.normal(0, 8), rad * np.sin(phi), 200 + rad * np.cos(phi)]), np.array([xoff * 0.5, 0, 200]), False))
    cams = []
    for idx, (ctr, tgt, with_card) in enumerate(views):
        T = arm_pose(idx, pivot) if (SWAY or SHIFT) else np.eye(4)
        Ti = np.linalg.inv(T)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tri((arm.vertices @ T[:3, :3].T + T[:3, 3]), arm.faces))          # id 0 limb
        scene.add_triangles(tri(mat_v, np.array([[0, 1, 2], [0, 2, 3]])))                     # id 1 table
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(bg))                       # id 2 room
        if with_card:
            cw = card3d @ T[:3, :3].T + T[:3, 3]
            scene.add_triangles(tri(cw, np.array([[0, 1, 2], [0, 2, 3]])))                    # id 3 card on arm
        tgt = tgt + rng.normal(0, 10, 3)
        Rw, t = look_at(ctr, tgt, np.array([1.0, 0, 0]) + rng.normal(0, 0.05, 3))
        dw = dirs_c @ Rw
        rays = np.c_[np.repeat(ctr[None], len(dw), 0), dw].astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        th = ans['t_hit'].numpy(); gid = ans['geometry_ids'].numpy(); nrm = ans['primitive_normals'].numpy()
        P = ctr[None] + dw * th[:, None]
        col = np.zeros((len(P), 3))
        m0 = gid == 0
        col[m0] = skin(P[m0] @ Ti[:3, :3].T + Ti[:3, 3])  # texture moves with the limb
        m1 = gid == 1
        if m1.any() and REF == 'card':
            col[m1] = table_card(P[m1]) if not CARD_ON_ARM else table_card(P[m1] + np.array([0, 2000, 0]))  # no table card
        elif m1.any():
            pc = np.full(m1.sum(), 0.95)
            if ctr[2] > 0:
                Pm = P[m1]
                for (_, ox, oy, img) in sheets:
                    sel = (Pm[:, 0] >= ox) & (Pm[:, 0] < ox + 210)
                    px = ((Pm[sel, 0] - ox) * 8).astype(int).clip(0, img.shape[1] - 1)
                    py = ((oy + 297 - Pm[sel, 1]) * 8).astype(int).clip(0, img.shape[0] - 1)
                    pc[sel] = img[py, px] / 255.0 * 0.93 + 0.03
            col[m1] = pc[:, None] * np.array([1, 1, 0.97])
        m2 = gid == 2
        g = 0.45 + 0.2 * vnoise(P[m2], 120) + 0.1 * vnoise(P[m2], 25)
        col[m2] = np.clip(g, 0, 1)[:, None] * np.array([0.8, 0.85, 0.9])
        m3 = gid == 3
        if m3.any():
            cw = card3d @ T[:3, :3].T + T[:3, 3]
            eu = (cw[1] - cw[0]) / np.linalg.norm(cw[1] - cw[0]); ev = (cw[3] - cw[0]) / np.linalg.norm(cw[3] - cw[0])
            uv = np.c_[(P[m3] - cw[0]) @ eu, (P[m3] - cw[0]) @ ev]
            col[m3] = card_texture(uv)
        nn = nrm.copy(); flip = (nn * dw).sum(1) > 0; nn[flip] *= -1
        lam = 0.35 + 0.45 * np.clip(nn @ sun, 0, 1) + 0.25 * np.clip(-(nn * dw).sum(1), 0, 1)
        lam[m2] = 1.0
        img = (col * lam[:, None]).reshape(H, W, 3)
        img = cv2.GaussianBlur(img, (0, 0), BLUR) + rng.normal(0, NOISE, img.shape)
        bgr = (np.clip(img, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)
        name = f'img_{idx:03d}.jpg'
        cv2.imwrite(f'{out}/images/{name}', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEGQ])
        cams.append({'name': name, 'R': Rw.tolist(), 't': t.tolist(), 'K': K.tolist(), 'arm_T': T.tolist(), 'card': bool(with_card)})
        print(name, flush=True)
    json.dump(cams, open(f'{out}/gt_cameras.json', 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32)
