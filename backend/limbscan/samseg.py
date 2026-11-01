"""Limb masks with SAM 2 (Segment Anything 2, Meta) on the GPU, prompted per photo with the hand's landmarks and
points along the forearm (positive) inside the projected arm outline (box). Runs in the `limbgpu` image.

Usage: python samseg.py <prompts.json> <image_dir> <out_dir> <checkpoint>
prompts.json: {name: {"pos": [[x, y], ...] (0..1), "box": [x0, y0, x1, y1] (0..1)}}
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main(prompts, image_dir, out_dir, ckpt):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    t0 = time.time()
    model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', ckpt, device='cuda')
    pred = SAM2ImagePredictor(model)
    pr = json.loads(Path(prompts).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for name, p in sorted(pr.items()):
            img = cv2.imread(str(Path(image_dir) / name))
            if img is None or not p.get('pos'):
                continue
            h, w = img.shape[:2]
            pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pts = np.array([[x * w, y * h] for x, y in p['pos']], np.float32)
            box = np.array([p['box'][0] * w, p['box'][1] * h, p['box'][2] * w, p['box'][3] * h], np.float32) \
                if p.get('box') else None
            masks, scores, _ = pred.predict(point_coords=pts, point_labels=np.ones(len(pts), np.int32), box=box,
                                            multimask_output=False)
            m = masks[0] > 0
            # keep the piece(s) the prompt points are on
            n, lab = cv2.connectedComponents(m.astype(np.uint8))
            keep = {lab[int(min(h - 1, max(0, y))), int(min(w - 1, max(0, x)))] for x, y in pts} - {0}
            m = np.isin(lab, list(keep)) if keep else m
            cv2.imwrite(str(out / (Path(name).stem + '.png')), m.astype(np.uint8) * 255)
            stats[name] = float(scores[0])
    (out / 'sam_scores.json').write_text(json.dumps(stats))
    print(f'[sam] {len(stats)} masks in {time.time() - t0:.0f}s, score median {np.median(list(stats.values())):.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:5])
