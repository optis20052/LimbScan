"""High-resolution cut-out of the hand (BiRefNet, MIT code) on a square crop around it in every photo, for the
narrow gaps between fingers that SAM 2's 256-cell grid fills in. Runs in the `limbbiref` image (GPU).
Usage: python birefrun.py <boxes.json> <image_dir> <out_dir>
boxes.json: {name: [x0, y0, side] (pixels)} -> out_dir/<stem>.png: probability 0..255 of the crop (side x side)."""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main(boxes, image_dir, out_dir):
    from transformers import AutoModelForImageSegmentation
    t0 = time.time()
    name = os.environ.get('BIREF_MODEL', 'ZhengPeng7/BiRefNet_HR')
    size = int(os.environ.get('BIREF_SIZE', '2048'))
    m = AutoModelForImageSegmentation.from_pretrained(name, trust_remote_code=True).cuda().eval().half()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    bx = json.loads(Path(boxes).read_text())
    for n, (x0, y0, side) in bx.items():
        img = cv2.imread(str(Path(image_dir) / n))
        if img is None:
            continue
        A = np.array([[size / side, 0, -x0 * size / side], [0, size / side, -y0 * size / side]], np.float32)
        crop = cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        t = torch.tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32, device='cuda')
        t = ((t.permute(2, 0, 1)[None] - mean) / std).half()
        with torch.no_grad():
            p = m(t)[-1].sigmoid().float()[0, 0].cpu().numpy()
        cv2.imwrite(str(out / (Path(n).stem + '.png')), (p * 255).astype(np.uint8))
    print(f'[biref] {len(bx)} hand crops at {size} px in {time.time() - t0:.0f}s ({name})')


if __name__ == '__main__':
    main(*sys.argv[1:4])
