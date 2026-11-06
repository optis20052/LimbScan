"""LimbScan web server: mobile capture UI + job queue + results/compare API.

Run:  .venv/bin/python -m server.app          (serves https://<lan-ip>:8443)
"""
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('LIMBSCAN_DATA', ROOT / 'data'))
REFS = Path(os.environ.get('LIMBSCAN_REFS', ROOT.parent / 'test-data' / 'references'))  # reference .obj/.stl/.ply scans
PY = sys.executable
(DATA / 'scans').mkdir(parents=True, exist_ok=True)

app = FastAPI(title='LimbScan')
jobs_q: 'queue.Queue[str]' = queue.Queue()
# photos are prepared (resize, limb masks, limb features) while the phone is still uploading: see limbscan/prefetch.py
prefetch_q: 'queue.Queue[tuple[str, Path]]' = queue.Queue()
prefetch_pending: dict[str, int] = {}
prefetch_lock = threading.Lock()
PREFETCH = os.environ.get('LIMBSCAN_PREFETCH', '1') == '1'


def scan_dir(sid):
    if not sid.replace('-', '').isalnum():
        raise HTTPException(400, 'bad id')
    d = DATA / 'scans' / sid
    if not d.exists():
        raise HTTPException(404, 'no such scan')
    return d


def references():
    out = []
    for p in sorted(REFS.iterdir()) if REFS.exists() else []:
        if p.suffix.lower() in ('.obj', '.stl', '.ply') and p.is_file():
            out.append(p)
    return out


def prefetcher():
    sys.path.insert(0, str(ROOT))
    from limbscan import prefetch
    while True:
        sid, photo = prefetch_q.get()
        try:
            prefetch.process(DATA / 'scans' / sid, photo)
        except Exception as e:  # the pipeline computes anything missing itself
            print('prefetch failed', photo, e)
        finally:
            with prefetch_lock:
                prefetch_pending[sid] = prefetch_pending.get(sid, 1) - 1


def _wait_prefetch(sid, limit=30.0):
    """Let the last photos finish preparing before the pipeline starts (else both would do the same work)."""
    t = time.time()
    while time.time() - t < limit:
        with prefetch_lock:
            if prefetch_pending.get(sid, 0) <= 0:
                return
        time.sleep(0.1)


def worker():
    while True:
        sid = jobs_q.get()
        _wait_prefetch(sid)
        d = DATA / 'scans' / sid
        meta = json.loads((d / 'meta.json').read_text())
        meta['started'] = time.time(); (d / 'meta.json').write_text(json.dumps(meta))
        with open(d / 'pipeline.out', 'a') as f:
            subprocess.run([PY, '-m', 'limbscan.pipeline', str(d)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        st = json.loads((d / 'status.json').read_text()) if (d / 'status.json').exists() else {}
        meta = json.loads((d / 'meta.json').read_text())  # read again: the scan may have been renamed meanwhile
        meta['finished'] = time.time(); (d / 'meta.json').write_text(json.dumps(meta))
        if st.get('stage') == 'done' and os.environ.get('LIMBSCAN_GS_HOST'):
            detail_q.put(d)  # palm + fingers from 2DGS on the GPU PC (~15 min): the quick result is already there
        if st.get('stage') == 'done' and os.environ.get('LIMBSCAN_AUTO_COMPARE') == '1':
            # off by default: comparing every scan with every reference keeps all CPU cores busy for minutes; run
            # limbscan.compare by hand for validation scans
            compare_q.put(d)


compare_q: 'queue.Queue[Path]' = queue.Queue()
detail_q: 'queue.Queue[Path]' = queue.Queue()


def detailer():
    """One detail pass at a time (limbscan/detail.py); the scan's status says 'detail': running / done / failed."""
    while True:
        d = detail_q.get()
        def mark(state):
            p = d / 'status.json'
            st = json.loads(p.read_text()) if p.exists() else {}
            st['detail'] = state
            p.write_text(json.dumps(st))
        mark('running')
        with open(d / 'detail.out', 'a') as f:
            r = subprocess.run([PY, '-m', 'limbscan.detail', str(d)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        mark('done' if r.returncode == 0 else 'failed')

_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')


def comparer():
    while True:
        d = compare_q.get()
        for ref in references():
            try:
                # lowest priority and a few cores: a research side job must not load the PC (or slow the next scan)
                subprocess.run(['nice', '-n', '19', PY, '-m', 'limbscan.compare', str(d / 'result' / 'limb_surface.ply'), str(ref),
                                str(d / 'compare')], cwd=ROOT, stdout=open(d / 'compare.out', 'a'), stderr=subprocess.STDOUT,
                               timeout=900, env={**os.environ, **{k: '2' for k in _THREAD_VARS}})
            except Exception as e:  # noqa
                print('compare failed', e)


threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=comparer, daemon=True).start()
threading.Thread(target=detailer, daemon=True).start()
if PREFETCH:  # a few in parallel: a photo takes ~0.8 s to prepare, the phone sends one every ~0.5 s
    for _ in range(int(os.environ.get('LIMBSCAN_PREFETCH_THREADS', 4))):
        threading.Thread(target=prefetcher, daemon=True).start()


@app.get('/', response_class=HTMLResponse)
def index():
    return (ROOT / 'server' / 'static' / 'index.html').read_text()


@app.get('/app.apk')
def android_app():
    """Latest LimbScan Android build, for installing straight from the phone's browser."""
    p = DATA / 'limbscan.apk'
    if not p.exists():
        raise HTTPException(404, 'no app build uploaded yet')
    return FileResponse(p, media_type='application/vnd.android.package-archive', filename='LimbScan.apk')


@app.get('/markers.pdf')
def markers_pdf():
    p = DATA / 'limbscan_markers.pdf'
    if not p.exists():
        from limbscan.markers import write_pdf
        write_pdf(str(p))
    return FileResponse(p, media_type='application/pdf', filename='limbscan_markers.pdf')


@app.post('/api/scans')
def create_scan(name: str = Form('')):
    sid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    d = DATA / 'scans' / sid
    (d / 'input').mkdir(parents=True)
    (d / 'meta.json').write_text(json.dumps({'id': sid, 'name': name or sid, 'created': time.time()}))
    return {'id': sid}


@app.post('/api/scans/{sid}/files')
async def upload(sid: str, files: list[UploadFile] = File(...), meta: str = Form('')):
    """Photos (live capture / upload). `meta` = JSON with the phone's AR pose for a single uploaded photo."""
    d = scan_dir(sid)
    n = 0
    for f in files:
        ext = Path(f.filename or 'x.jpg').suffix.lower() or '.jpg'
        name = Path(f.filename or '').name
        if re.fullmatch(r'[A-Za-z0-9_.-]{1,80}\.(jpg|jpeg|png)', name, re.I) and len(files) == 1:
            dst = d / 'input' / name  # the app's own name: a retried upload replaces itself, no duplicates
        else:
            dst = d / 'input' / f'{int(time.time() * 1000)}_{n:03d}{ext}'
        with open(dst, 'wb') as o:
            shutil.copyfileobj(f.file, o, 1 << 20)
        if meta and len(files) == 1:
            json.loads(meta)  # validate
            dst.with_suffix('.json').write_text(meta)
        if PREFETCH:
            with prefetch_lock:
                prefetch_pending[sid] = prefetch_pending.get(sid, 0) + 1
            prefetch_q.put((sid, dst))
        n += 1
    return {'received': n, 'total': len([p for p in (d / 'input').iterdir() if p.suffix != '.json'])}


@app.post('/api/scans/{sid}/chunk')
async def upload_chunk(sid: str, name: str = Form(...), index: int = Form(...), total: int = Form(...), chunk: UploadFile = File(...)):
    """Large files arrive in pieces (tunnels/proxies cap request size at ~100 MB)."""
    d = scan_dir(sid)
    safe = ''.join(c for c in Path(name).name if c.isalnum() or c in '._-') or 'upload.bin'
    part = d / 'input' / f'.{safe}.part'
    with open(part, 'ab' if index > 0 else 'wb') as o:
        shutil.copyfileobj(chunk.file, o, 1 << 20)
    if index == total - 1:
        part.rename(d / 'input' / f'{int(time.time() * 1000)}_{safe}')
    return {'ok': True}


@app.post('/api/scans/{sid}/start')
def start(sid: str, arm: str = Form('')):
    d = scan_dir(sid)
    if not any((d / 'input').iterdir()):
        raise HTTPException(400, 'no images uploaded')
    if arm:  # the arm as the phone finally placed it (refined while scanning): every photo is cut out with it
        a = json.loads(arm)
        (d / 'arm.json').write_text(json.dumps({k: a[k] for k in ('a', 'b', 'r')}))
    (d / 'status.json').write_text(json.dumps({'stage': 'queued', 'progress': 0, 'position': jobs_q.qsize() + 1}))
    jobs_q.put(sid)
    return {'ok': True}


@app.post('/api/log')
def device_log(session: str = Form(...), text: str = Form(...)):
    """The phone's own log for one capture session (sent during and after a scan, also when it is discarded)."""
    d = DATA / 'logs'
    d.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', session)[:80] or 'unknown'
    (d / f'{name}.txt').write_text(text[-600000:])
    sd = DATA / 'scans' / name
    if sd.is_dir():
        (sd / 'device_log.txt').write_text(text[-600000:])
    return {'ok': True}


@app.post('/api/crash')
def crash_report(text: str = Form(...)):
    d = DATA / 'crashes'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{time.strftime("%Y%m%d-%H%M%S")}.txt').write_text(text[:200000])
    return {'ok': True}


@app.post('/api/check')
async def check_frame(frame: UploadFile = File(...)):
    """Live setup/capture check on a small preview frame: scale markers, exposure, sharpness."""
    import cv2
    import numpy as np
    from limbscan.scale import detect_gray
    g = cv2.imdecode(np.frombuffer(await frame.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise HTTPException(400, 'bad image')
    det = detect_gray(g)
    from limbscan.card import detect_quads, plausible_card
    quads = detect_quads(g)
    sheets = sorted({int(m) // 12 + 1 for m in det if m < 36})
    mean = float(g.mean())
    sharp = float(cv2.Laplacian(cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1]))), cv2.CV_64F).var())
    return {'markers': len(det), 'sheets': sheets, 'cards': int(sum(bool(plausible_card(q)) for q in quads)), 'brightness': mean,
            'exposure': 'dark' if mean < 70 else 'bright' if mean > 200 else 'ok', 'sharpness': sharp}


@app.get('/api/scans')
def list_scans():
    out = []
    for d in sorted((DATA / 'scans').iterdir(), reverse=True):
        try:
            meta = json.loads((d / 'meta.json').read_text())
            st = json.loads((d / 'status.json').read_text()) if (d / 'status.json').exists() else {'stage': 'capturing'}
            out.append({**meta, 'stage': st.get('stage'), 'progress': st.get('progress'), 'summary': st.get('summary')})
        except Exception:
            continue
    return out


@app.get('/api/scans/{sid}')
def get_scan(sid: str):
    d = scan_dir(sid)
    meta = json.loads((d / 'meta.json').read_text())
    st = json.loads((d / 'status.json').read_text()) if (d / 'status.json').exists() else {'stage': 'capturing', 'progress': 0}
    res = {'meta': meta, 'status': st, 'n_input': len([p for p in (d / 'input').iterdir() if p.suffix != '.json'])}
    m = d / 'result' / 'measurements.json'
    if m.exists():
        res['measurements'] = json.loads(m.read_text())
    cmp = []
    for c in sorted((d / 'compare').glob('compare_*.json')) if (d / 'compare').exists() else []:
        cmp.append({**json.loads(c.read_text()), 'key': c.stem[len('compare_'):]})
    res['compare'] = cmp
    if (d / 'log.txt').exists():
        res['log_tail'] = (d / 'log.txt').read_text().splitlines()[-12:]
    return res


@app.get('/api/scans/{sid}/file/{kind}/{name}')
def get_file(sid: str, kind: str, name: str):
    d = scan_dir(sid)
    if kind not in ('result', 'compare') or '/' in name or '..' in name:
        raise HTTPException(400)
    p = d / kind / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=name)


@app.get('/api/scans/{sid}/photos')
def list_photos(sid: str):
    """The photos the server used (capture order), and which were left out of the model."""
    d = scan_dir(sid)
    names = sorted(p.name for p in (d / 'images').glob('*.jpg')) if (d / 'images').exists() else []
    bad = json.loads((d / 'colmap' / 'bad_masks.json').read_text()) if (d / 'colmap' / 'bad_masks.json').exists() else []
    return {'photos': names, 'left_out': bad}


@app.get('/api/scans/{sid}/photo/{name}')
def get_photo(sid: str, name: str, size: int = 360, outline: int = 1):
    """One photo as a JPEG (thumbnail when size < full width), with the limb cut-out drawn in red when outline=1.
    Cached on disk next to the scan."""
    import cv2
    import numpy as np
    d = scan_dir(sid)
    if not re.fullmatch(r'[\w.-]+\.jpg', name) or '..' in name:
        raise HTTPException(400)
    src = d / 'images' / name
    if not src.exists():
        raise HTTPException(404)
    size = max(120, min(int(size), 2000))
    cache = d / 'thumbs' / f'{Path(name).stem}_{size}_{int(bool(outline))}.jpg'
    if not cache.exists():
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        if outline:
            m = cv2.imread(str(d / 'colmap' / 'mask_mvs' / (Path(name).stem + '.png')), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                c, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, c, -1, (40, 40, 255), max(2, w // 360))
        if size < w:
            img = cv2.resize(img, (size, int(h * size / w)), interpolation=cv2.INTER_AREA)
        cache.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(cache), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return FileResponse(cache, media_type='image/jpeg', headers={'Cache-Control': 'max-age=3600'})


@app.post('/api/scans/{sid}/name')
def rename_scan(sid: str, name: str = Form(...)):
    name = ' '.join(name.split())[:80]
    if not name:
        raise HTTPException(400, 'empty name')
    d = scan_dir(sid)
    meta = json.loads((d / 'meta.json').read_text())
    meta['name'] = name
    tmp = d / 'meta.json.tmp'
    tmp.write_text(json.dumps(meta)); tmp.replace(d / 'meta.json')
    return {'name': name}


@app.delete('/api/scans/{sid}')
def delete_scan(sid: str):
    shutil.rmtree(scan_dir(sid))
    return {'ok': True}


app.mount('/static', StaticFiles(directory=ROOT / 'server' / 'static'), name='static')
app.mount('/models', StaticFiles(directory=ROOT / 'models'), name='models')  # on-device limb segmentation model


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1)); return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


CERT_VERSION = '2'  # leaf cert with serverAuth EKU, <=397 days: required by iOS/macOS Safari


def ensure_cert(ip):
    cdir = DATA / 'cert'; cdir.mkdir(exist_ok=True)
    key, crt = cdir / 'key.pem', cdir / 'cert.pem'
    stamp = cdir / 'ip.txt'
    want = f'{ip} v{CERT_VERSION}'
    if not crt.exists() or not stamp.exists() or stamp.read_text() != want:
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-sha256', '-nodes', '-days', '397',
                        '-keyout', str(key), '-out', str(crt), '-subj', '/CN=LimbScan local',
                        '-addext', f'subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost',
                        '-addext', 'basicConstraints=critical,CA:FALSE',
                        '-addext', 'keyUsage=critical,digitalSignature,keyEncipherment',
                        '-addext', 'extendedKeyUsage=serverAuth'],
                       check=True, capture_output=True)
        stamp.write_text(want)
    return str(key), str(crt)


async def _serve_both(key, crt, https_port, http_port):
    import asyncio
    import uvicorn
    servers = [uvicorn.Server(uvicorn.Config(app, host='0.0.0.0', port=https_port, ssl_keyfile=key, ssl_certfile=crt, log_level='warning')),
               uvicorn.Server(uvicorn.Config(app, host='0.0.0.0', port=http_port, log_level='warning'))]
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == '__main__':
    import asyncio
    ip = lan_ip()
    key, crt = ensure_cert(ip)
    port = int(os.environ.get('PORT', 8443))
    http_port = int(os.environ.get('HTTP_PORT', 8080))
    print(f'''
  LimbScan running — phone on the same Wi-Fi:
    https://{ip}:{port}      live camera capture (accept the certificate warning once)
    http://{ip}:{http_port}       upload video/photos only (browsers block the camera on plain http)
''')
    asyncio.run(_serve_both(key, crt, port, http_port))
