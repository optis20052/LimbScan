"""Surface normals of every photo with MoGe-2 (Microsoft, MIT licence), in the `limbmoge` image (GPU).
Usage: python mogerun.py <image_dir> <out_dir> [names...]   -> out_dir/<stem>.png: normal (camera frame) as
uint8 RGB = (n + 1) * 127.5, 0 where MoGe has no valid point."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main(image_dir, out_dir, names):
    import os
    t0 = time.time()
    ver = os.environ.get('MOGE_MODEL', 'Ruicheng/moge-3-vitl')
    if 'moge-3' in ver:
        from moge.model.v3 import MoGeModel
    else:
        from moge.model.v2 import MoGeModel
    m = MoGeModel.from_pretrained(ver).cuda().eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = names or sorted(p.name for p in Path(image_dir).glob('*.jpg'))
    for n in names:
        img = cv2.imread(str(Path(image_dir) / n))
        if img is None:
            continue
        t = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32, device='cuda').permute(2, 0, 1)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            r = m.infer(t)
        nm = r['normal'].float().cpu().numpy()
        q = np.clip((nm + 1) * 127.5, 0, 255).astype(np.uint8)
        q[~r['mask'].cpu().numpy()] = 0
        cv2.imwrite(str(out / (Path(n).stem + '.png')), q[..., ::-1])
    print(f'[moge] normals of {len(names)} photos in {time.time() - t0:.0f}s ({ver})')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
