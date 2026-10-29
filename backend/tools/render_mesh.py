"""Quick shaded render of a mesh from several angles (for checking what the clinician sees)."""
import sys, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
m = trimesh.load(sys.argv[1], force='mesh')
if not hasattr(m.visual, 'vertex_colors') or m.visual.kind != 'vertex':
    m = trimesh.Trimesh(m.vertices, m.faces, vertex_colors=np.tile([205, 170, 150, 255], (len(m.vertices), 1)))
if len(m.faces) > 40000:
    import open3d as o3d
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    o.vertex_colors = o3d.utility.Vector3dVector(m.visual.vertex_colors[:, :3] / 255.0)
    o = o.simplify_quadric_decimation(40000)
    m = trimesh.Trimesh(np.asarray(o.vertices), np.asarray(o.triangles), vertex_colors=(np.asarray(o.vertex_colors) * 255).astype(np.uint8))
c = m.vertices.mean(0); _, _, vt = np.linalg.svd(m.vertices - c, full_matrices=False)
V = (m.vertices - c) @ vt.T
fc = m.visual.vertex_colors[m.faces][:, :, :3].mean(1) / 255.0
n = np.cross(V[m.faces[:, 1]] - V[m.faces[:, 0]], V[m.faces[:, 2]] - V[m.faces[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
fig = plt.figure(figsize=(24, 8))
for k, (el, az) in enumerate([(10, -90), (10, 0), (60, -60)]):
    ax = fig.add_subplot(1, 3, k + 1, projection='3d'); ax.set_facecolor('k')
    e, a = np.deg2rad(el), np.deg2rad(az); light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    sh = np.clip(np.abs(n @ light), 0.25, 1)[:, None]
    ax.add_collection3d(Poly3DCollection(V[m.faces], facecolors=np.clip(fc * sh, 0, 1), linewidths=0))
    r = np.percentile(np.abs(V), 99.5) * 0.62; ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r); ax.view_init(el, az); ax.axis('off')
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=60, facecolor='k')
