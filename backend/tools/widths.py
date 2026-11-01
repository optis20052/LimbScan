"""Widths, thicknesses and girths of a reference and an aligned scan at set distances from the fingertips:
python tools/widths.py reference scan_aligned_to_reference"""
import sys
import numpy as np
import trimesh
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from evalref import load

ref, scan = load(sys.argv[1]), trimesh.load(sys.argv[2], process=False)
F = ref.vertices
c = F.mean(0)
ax = np.linalg.svd(F - c, full_matrices=False)[2][0]
if ((scan.vertices - c) @ ax).mean() < 0:
    ax = -ax
tip = ((F - c) @ ax).max()
Pd = F[np.abs((tip - (F - c) @ ax) - 100) < 3]
Q = Pd - Pd.mean(0)
Q -= np.outer(Q @ ax, ax)
w_dir = np.linalg.svd(Q, full_matrices=False)[2][0]
t_dir = np.cross(ax, w_dir)


def loops(m, d):
    o = c + ax * (tip - d)
    p = m.section(plane_origin=o, plane_normal=ax)
    if p is None:
        return []
    out = []
    for e in p.discrete:
        e = np.asarray(e)
        if len(e) < 8:
            continue
        a, b = (e - o) @ w_dir, (e - o) @ t_dir
        out.append((a.min(), a.max(), b.max() - b.min(), np.linalg.norm(np.diff(e, axis=0), axis=1).sum()))
    return sorted(out)


def fmt(L):
    if not L:
        return '-'
    if len(L) == 1:
        return f'width {L[0][1] - L[0][0]:5.1f}  thick {L[0][2]:5.1f}  girth {L[0][3]:5.1f}'
    return 'pieces ' + ' '.join(f'{x[1] - x[0]:4.1f}' for x in L) + f'  span {L[-1][1] - L[0][0]:5.1f}'


print(f'{"from tip":>9s} | {"STL (truth)":44s} | new scan')
for d in (20, 40, 60, 80, 100, 120, 150, 175, 195):
    print(f'{d:6d} mm | {fmt(loops(ref, d)):44s} | {fmt(loops(scan, d))}')
