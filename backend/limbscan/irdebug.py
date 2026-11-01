"""Debug sheet: photos with the mask outline (green) and mesh outlines (red, blue, ...) drawn on top.

Usage: python irdebug.py <prep.npz> <out.jpg> <mesh.ply> [<mesh.ply> ...] [--views 0,10,20]
"""
import sys

import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh

from irrefine import Views, dev

COLS = [(255, 0, 0), (0, 120, 255), (255, 0, 255), (255, 255, 0)]


def main(prep, out, meshes, views):
    z = np.load(prep)
    vw = Views(z, 0)
    glctx = dr.RasterizeCudaContext()
    covers = []
    for p in meshes:
        m = trimesh.load(p, process=False)
        V = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
        F = torch.tensor(m.faces, dtype=torch.int32, device=dev)
        clip, _ = vw.clip(V)
        rast, _ = dr.rasterize(glctx, clip, F, (vw.H, vw.W))
        covers.append((rast[..., 3] > 0).cpu().numpy().astype(np.uint8))
    tiles = []
    for i in views:
        im = (vw.img[i].permute(1, 2, 0).cpu().numpy() ** (1 / 2.2) * 255).astype(np.uint8).copy()
        m = (vw.mask[i].cpu().numpy() > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, (0, 255, 0), 1)
        for c, col in zip(covers, COLS):
            cs, _ = cv2.findContours(c[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, col, 1)
        tiles.append(im)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(out, cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    a = sys.argv[1:]
    views = [0, 13, 26, 39]
    if '--views' in a:
        i = a.index('--views')
        views = [int(x) for x in a[i + 1].split(',')]
        a = a[:i] + a[i + 2:]
    main(a[0], a[1], a[2:], views)
