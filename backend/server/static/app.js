import { $, api, fmt, ui } from './common.js';
import { S } from './sketches.js';
import { scanPage, disposeResults } from './results.js?v=4';

const app = $('#app');
const store = { get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { } } };

// ---------------- routing ----------------
function route() {
  clearInterval(ui.pollTimer); disposeResults(); stopPreview();
  const h = location.hash.slice(1) || '/';
  if (h.startsWith('/setup')) return setup(+(h.split('/')[2] || 0));
  if (h.startsWith('/scan/')) return scanPage(h.split('/')[2]);
  return home();
}
addEventListener('hashchange', route);

// ---------------- home ----------------
async function home() {
  const done = store.get('setupDone');
  app.innerHTML = `
  <div class="card hero"><h2>New limb scan</h2>
    <p class="mut small">About 3 minutes: a quick setup, then 4 guided passes around the arm.</p>
    ${XR.supported ? `<button id="goar" class="big">📱 Start AR scan <span style="font-weight:400">(no card)</span></button><p class="small mut" style="text-align:center;margin:6px 0 10px">Uses your phone's AR tracking for real size.</p>` : ''}
    <button id="go" class="big ${XR.supported ? 'sec' : ''}">📷 ${XR.supported ? 'Scan with a card instead' : 'Start guided scan'}</button>
    ${done ? `<p class="small mut" style="text-align:center;margin:8px 0 0">Setup already done before? <a href="#" id="skip" style="color:var(--acc)">Skip to camera check</a></p>` : ''}
    <details style="margin-top:12px"><summary class="small mut">Other options</summary>
      <div class="row" style="margin-top:10px"><label class="btn sec">⬆️ Upload video / photos<input id="file" type="file" accept="image/*,video/*" multiple hidden></label>
      <a class="btn sec" href="/markers.pdf">🖨 Printed sheets (optional, most precise)</a></div>
      <p class="small mut">Upload gives the best quality: record a slow 4K video with the phone's camera app following the same 4 passes, with the card in view.</p></details>
    <div id="upl" style="display:none"><h3>Uploading…</h3><div class="bar"><i id="ubar"></i></div><div id="utxt" class="small mut"></div></div>
  </div>
  <div class="card"><h2>Scans</h2><div class="list" id="list"><span class="mut">Loading…</span></div></div>`;
  $('#go').onclick = () => { store.set('mode', 'card'); location.hash = '/setup/0'; };
  $('#goar') && ($('#goar').onclick = () => { store.set('mode', 'ar'); location.hash = '/setup/0'; });
  $('#skip') && ($('#skip').onclick = e => { e.preventDefault(); store.set('mode', 'card'); STEPS = stepsList(); location.hash = '/setup/' + (STEPS.length - 1); });
  $('#file').onchange = e => uploadFiles([...e.target.files]);
  const scans = await api('/api/scans');
  $('#list').innerHTML = scans.length ? scans.map(s => `
    <div class="item" data-id="${s.id}"><div><div>${s.name}</div><div class="small mut">${new Date(s.created * 1000).toLocaleString()}${s.summary ? ` · wrist ${fmt(s.summary.wrist_girth_mm)} mm` : ''}</div></div>
    <span class="pill ${s.stage === 'done' ? 'ok' : s.stage === 'failed' ? 'bad' : 'run'}">${s.stage || '…'}</span></div>`).join('') : '<span class="mut">No scans yet.</span>';
  document.querySelectorAll('.item').forEach(e => e.onclick = () => location.hash = '/scan/' + e.dataset.id);
}

async function newScan() { const fd = new FormData(); fd.append('name', 'Scan ' + new Date().toLocaleString()); return (await api('/api/scans', { method: 'POST', body: fd })).id; }

function xhrUpload(url, fd, onp) {
  return new Promise((res, rej) => { const x = new XMLHttpRequest(); x.open('POST', url); x.upload.onprogress = e => onp && onp(e.loaded / e.total);
    x.onload = () => x.status < 300 ? res(JSON.parse(x.responseText)) : rej(new Error(x.responseText)); x.onerror = () => rej(new Error('network')); x.send(fd); });
}

async function uploadFiles(files) {
  if (!files.length) return;
  $('#upl').style.display = 'block';
  const id = await newScan();
  const total = files.reduce((a, f) => a + f.size, 0); let done = 0;
  const CH = 40e6; // stay well under proxy/tunnel request limits
  const prog = p => { $('#ubar').style.width = (100 * (done + p) / total) + '%'; $('#utxt').textContent = `${((done + p) / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`; };
  for (const f of files) {
    const n = Math.max(1, Math.ceil(f.size / CH));
    for (let i = 0; i < n; i++) {
      const fd = new FormData(); fd.append('name', f.name); fd.append('index', i); fd.append('total', n);
      fd.append('chunk', f.slice(i * CH, Math.min(f.size, (i + 1) * CH)), f.name);
      for (let attempt = 0; ; attempt++) {
        try { await xhrUpload(`/api/scans/${id}/chunk`, fd, p => prog(i * CH + p * Math.min(CH, f.size - i * CH))); break; }
        catch (e) { if (attempt >= 3) throw e; await new Promise(r => setTimeout(r, 1500)); }
      }
    }
    done += f.size;
  }
  await api(`/api/scans/${id}/start`, { method: 'POST' });
  location.hash = '/scan/' + id;
}

// ---------------- setup wizard ----------------
const CARD_STEPS = [
  { t: 'How will the arm be placed?', pose: true },
  { t: 'Have a card ready', sk: S.calib, body: `<ol><li>Take any <b>standard-size card</b>: bank card, ID card, health insurance card (Carte Vitale), driving licence. It gives the scan its real size.</li>
     <li>At the <b>start of the scan</b> you hold it <b>flat against the arm</b> for a few seconds while the phone moves left and right. The app says <b>“✓ size locked”</b>, then you put it away and scan normally.</li>
     <li><b>Important:</b> calibrate with the arm already in its scanning position, and <b>don't move the arm afterwards</b>. If the arm changes position (e.g. lying flat, then raised), the size can't be transferred.</li>
     <li>All 4 edges must be visible; avoid glare on shiny cards.</li></ol>`,
    check: 'I have a standard card in my hand' },
  { t: 'Get comfortable and still', sk: () => (store.get('pose') === 'resting' ? S.support() : S.poseUpright()), body: `<ol><li>Upright: rest the <b>elbow on a folded towel</b> on the table, forearm vertical, fingers relaxed and slightly apart. Resting: let the forearm <b>lie on a pillow</b>, with the hand free if possible.</li>
     <li>Roll the sleeve above the elbow. Remove watch, rings and bandages if allowed.</li>
     <li><b>Keep the arm completely still</b> for about 1–2 minutes and breathe normally. Only the phone moves.</li></ol>
     <p class="small mut">Areas touching the support can't be seen. They are filled in smoothly and shown as “not seen” in the report.</p>`,
    check: 'Patient is comfortable and still' },
  { t: 'Light', sk: S.light, body: `<ol><li>Bright, <b>even</b> light on the arm. The window or lamp should be <b>behind the person holding the phone</b>, never behind the arm: a bright window behind the hand turns it into a dark silhouette and the scan fails.</li><li>No flash, no direct sun, no hard shadows on the arm.</li>
     <li>Phone camera at <b>1×</b>. Don't zoom. iPhone Pro: Settings → Camera → turn off <b>Macro Control</b>.</li></ol>`,
    check: 'Light is even, zoom is 1×' },
  { t: 'Camera check', live: true },
];

const XR = { supported: false };
const AR_START = { t: 'Start the AR scan', ar: true };
const stepsList = () => store.get('mode') === 'ar'
  ? CARD_STEPS.filter(s => s.pose || s.t.startsWith('Get comfortable') || s.t === 'Light').concat([AR_START])
  : CARD_STEPS;
let STEPS = stepsList();

function arStart(i) {
  app.innerHTML = `<div class="card wiz">${dots(i)}<div class="step">Step ${i + 1} of ${STEPS.length}</div><h2>Start the AR scan</h2>
    <ol><li>Tap <b>Start</b>. Chrome asks to use the camera and AR: allow both.</li>
    <li>Point the phone at the arm and <b>move it slowly for 2 seconds</b> so the phone can lock its position. No card needed: the phone measures its own movement.</li>
    <li>Then 2 short circles around the hand and the forearm (under a minute). The phone guides you: it only takes photos where the arm is visible and new.</li></ol>
    <div class="row nav"><button class="sec" id="back">Back</button><button id="go">Start AR scan</button></div></div>`;
  $('#back').onclick = () => location.hash = '/setup/' + (i - 1);
  $('#go').onclick = () => startXR().catch(e => alert('AR could not start: ' + e.message + '\nUse the standard scan instead.'));
}

function poseStep(i) {
  const cur = store.get('pose') || 'upright';
  app.innerHTML = `<div class="card wiz">${dots(i)}<div class="step">Step ${i + 1} of ${STEPS.length}</div><h2>How will the arm be placed?</h2>
    <div class="poses">
      <button class="pose ${cur === 'upright' ? 'sel' : ''}" data-p="upright"><div class="skwrap">${S.poseUpright()}</div><b>Upright</b><span>Elbow on the table, forearm up, fingers up. Best for scanning <b>by yourself</b>: you circle the phone around it.</span></button>
      <button class="pose ${cur === 'resting' ? 'sel' : ''}" data-p="resting"><div class="skwrap">${S.poseResting()}</div><b>Resting</b><span>Arm lying on a pillow or the table. For patients who can't hold the arm up. Easier with a second person.</span></button>
    </div>
    <div class="row nav"><button class="sec" id="back">Back</button><button id="next">Next</button></div></div>`;
  document.querySelectorAll('.pose').forEach(b => b.onclick = () => { store.set('pose', b.dataset.p); document.querySelectorAll('.pose').forEach(x => x.classList.toggle('sel', x === b)); });
  if (!store.get('pose')) store.set('pose', 'upright');
  $('#back').onclick = () => location.hash = '/';
  $('#next').onclick = () => location.hash = '/setup/' + (i + 1);
}

function dots(i) { return `<div class="dots">${STEPS.map((_, k) => `<i class="${k < i ? 'done' : k === i ? 'on' : ''}"></i>`).join('')}</div>`; }

function setup(i) {
  STEPS = stepsList();
  const st = STEPS[i]; if (!st) return (location.hash = '/');
  if (st.live) return cameraCheck(i);
  if (st.ar) return arStart(i);
  if (st.pose) return poseStep(i);
  app.innerHTML = `<div class="card wiz">${dots(i)}<div class="step">Step ${i + 1} of ${STEPS.length}</div><h2>${st.t}</h2>
    <div class="skwrap">${st.sk()}</div>${st.body}
    <label class="chk"><input type="checkbox" id="ok"> ${st.check}</label>
    <div class="row nav"><button class="sec" id="back">Back</button><button id="next" disabled>Next</button></div></div>`;
  $('#ok').onchange = e => $('#next').disabled = !e.target.checked;
  $('#back').onclick = () => location.hash = i ? '/setup/' + (i - 1) : '/';
  $('#next').onclick = () => location.hash = '/setup/' + (i + 1);
}

// ---------------- camera + live check ----------------
let preview = null;
function stopPreview() { if (preview) { clearInterval(preview.timer); preview.stream?.getTracks().forEach(t => t.stop()); preview = null; } }

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera needs a secure (https) address. Use the https:// link, or Upload.');
  const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 30 } } });
  const track = stream.getVideoTracks()[0];
  try { const c = track.getCapabilities?.() || {}; const adv = {}; if (c.focusMode?.includes('continuous')) adv.focusMode = 'continuous'; if (c.exposureMode?.includes('continuous')) adv.exposureMode = 'continuous'; if (c.zoom) adv.zoom = Math.max(c.zoom.min, Math.min(1, c.zoom.max)); await track.applyConstraints({ advanced: [adv] }); } catch (e) { }
  return stream;
}

async function askSensors() {
  try { if (typeof DeviceOrientationEvent !== 'undefined' && DeviceOrientationEvent.requestPermission) { await DeviceOrientationEvent.requestPermission(); await DeviceMotionEvent.requestPermission(); } } catch (e) { }
}

const small = document.createElement('canvas');
async function checkFrame(v) {
  if (!v.videoWidth) return null;
  const w = Math.min(1920, v.videoWidth), h = Math.round(w * v.videoHeight / v.videoWidth); small.width = w; small.height = h;
  small.getContext('2d').drawImage(v, 0, 0, w, h);
  const b = await new Promise(r => small.toBlob(r, 'image/jpeg', 0.8));
  const fd = new FormData(); fd.append('frame', b, 'f.jpg');
  try { return await api('/api/check', { method: 'POST', body: fd }); } catch (e) { return null; }
}

function cameraCheck(i) {
  app.innerHTML = `<div class="card wiz">${dots(i)}<div class="step">Step ${i + 1} of ${STEPS.length}</div><h2>Camera check</h2>
    <p class="small mut">Point the phone at the arm from 30–40 cm, holding the card against the arm.</p>
    <div class="pv"><video id="pv" playsinline muted autoplay></video><div class="pvframe"></div></div>
    <div class="checks"><div id="c1" class="ck">⏳ Size reference (card)</div><div id="c2" class="ck">⏳ Light</div><div id="c3" class="ck">⏳ Motion sensors</div></div>
    <div class="row nav"><button class="sec" id="back">Back</button><button id="start">Start camera</button></div>
    <p class="small mut" id="skipw" style="display:none;text-align:center">Checks won't pass? <a href="#" id="skipc" style="color:var(--acc)">Start anyway</a></p></div>`;
  $('#back').onclick = () => location.hash = '/setup/' + (i - 1);
  const go = () => { removeEventListener('deviceorientation', onOri); store.set('setupDone', '1'); const s = preview.stream; clearInterval(preview.timer); preview = null; startCapture(s); };
  let sensors = false;
  const onOri = e => { if (e.beta != null) sensors = true; };
  $('#start').onclick = async () => {
    await askSensors(); addEventListener('deviceorientation', onOri);
    let stream; try { stream = await openCamera(); } catch (e) { alert(e.message); return; }
    const v = $('#pv'); v.srcObject = stream; await v.play();
    preview = { stream };
    $('#start').textContent = 'Checking…'; $('#start').disabled = true;
    let good = 0;
    setTimeout(() => { if ($('#skipw')) { $('#skipw').style.display = 'block'; $('#skipc').onclick = e => { e.preventDefault(); if (preview) go(); }; } }, 8000);
    preview.timer = setInterval(async () => {
      const r = await checkFrame(v); if (!r || !preview) return;
      const mk = r.markers >= 6 || r.cards > 0, lt = r.exposure === 'ok';
      $('#c1').className = 'ck ' + (mk ? 'ok' : 'bad');
      $('#c1').textContent = r.markers >= 6 ? `✓ ${r.markers} printed markers seen` : r.cards ? '✓ card found' : '✗ card not found: show the whole card, flat, all 4 edges';
      $('#c2').className = 'ck ' + (lt ? 'ok' : 'bad'); $('#c2').textContent = lt ? '✓ light OK' : r.exposure === 'dark' ? '✗ too dark: add light' : '✗ too bright: avoid direct sun';
      $('#c3').className = 'ck ' + (sensors ? 'ok' : 'warn'); $('#c3').textContent = sensors ? '✓ motion sensors on' : '△ no motion sensors (guidance limited)';
      good = mk && lt ? good + 1 : 0;
      if (good >= 2) { $('#start').disabled = false; $('#start').textContent = 'Looks good: start scanning ▸'; $('#start').onclick = go; }
    }, 1200);
  };
}

// ---------------- guided capture ----------------
// camera elevation in degrees from W3C device orientation: 0 = level, + looking up, - looking down
const elevation = o => o ? Math.asin(Math.max(-1, Math.min(1, -Math.cos(o[1] * Math.PI / 180) * Math.cos(o[2] * Math.PI / 180)))) * 180 / Math.PI : null;
// Passes per arm pose. `band` = suggested camera elevation (deg); it only drives a gentle hint, never blocks capture.
const PHASE_SETS = {
  upright: [
    { k: 'cal', sk: 'cal', t: 'Size · Calibration', d: 'Put the arm in its <b>scanning position now</b> (elbow on the table, forearm up) and <b>keep it there for the whole scan</b>. Pinch the card between thumb and index of the scanned hand, <b>flat side to the phone</b>. Move the phone slowly left ↔ right until “✓ size locked”, then just let the card drop: <b>don\'t move the arm</b>.', band: null, min: 10, needCards: 6 },
    { k: 'hand', sk: 'up', t: 'Circle 1 · Around the hand & wrist', d: 'Phone at <b>hand height</b>, 30–40 cm away. Go slowly <b>all the way around the hand and wrist</b>: palm side, thumb side, back, little-finger side. Light should come from <b>behind you</b>.', band: [-40, 30], min: 24, cov: 0.4, covAR: 0.7 },
    { k: 'forearm', sk: 'up', t: 'Circle 2 · Around the forearm', d: 'Lower the phone to the <b>forearm</b>, tilt slightly down, and circle once more down to the elbow. Then you are done.', band: [-60, 10], min: 20, cov: 0.4, covAR: 0.6 },
  ],
  resting: [
    { k: 'cal', sk: 'cal', t: 'Size · Calibration', d: 'With the arm already <b>resting where it will stay</b>, lay the card flat on top of the forearm. Move the phone slowly left ↔ right, 25–35 cm above it, until “✓ size locked”. Then lift the card off <b>without moving the arm</b>.', band: null, min: 10, needCards: 6 },
    { k: 'above', sk: 'high', t: 'Circle 1 · From above', d: 'Phone 30–50 cm <b>above</b> the arm, looking down. Move all the way around the arm. Keep the card in view now and then.', band: [-90, -30], min: 28, cov: 0.6 },
    { k: 'side', sk: 'level', t: 'Circle 2 · From the sides, lower', d: 'Lower the phone to <b>table height</b> and look at the sides of the arm. Get as low as possible to see under the forearm.', band: [-50, 15], min: 26, cov: 0.5 },
    { k: 'close', sk: 'hand', t: 'Close-ups · Hand', d: 'Slow close passes 20–30 cm from the hand: back of the hand, thumb, between the fingers.', band: null, min: 18 },
  ],
};
let PHASES = PHASE_SETS.upright;
const sketchFor = ph => ph.sk === 'cal' ? S.calib() : ph.sk === 'up' ? S.uprightPass(ph.k) : S.orbit(ph.sk);
const BINS = 24;
const cap = {};

function speak(t) { if (!cap.voice || !window.speechSynthesis) return; speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(t.replace(/<[^>]+>/g, '')); u.rate = 1.05; speechSynthesis.speak(u); }

async function startCapture(stream) {
  PHASES = PHASE_SETS[store.get('pose') || 'upright'].map(p => ({ ...p }));
  Object.assign(cap, { cardHits: 0, offBandSince: 0, on: true, paused: true, stream, id: await newScan(), n: 0, up: 0, inflight: 0, queue: [], last: 0, lastOri: null, ori: null, rot: 0, sharpRef: 0,
    phase: 0, pn: PHASES.map(() => 0), bins: PHASES.map(() => new Array(BINS).fill(0)), markers: null, hintT: 0, hint: '', voice: store.get('voice') === '1', sensors: false });
  const c = $('#cap'); c.style.display = 'block';
  loadLimbModel();
  const v = $('#vid'); v.srcObject = stream; await v.play();
  try { cap.wake = await navigator.wakeLock?.request('screen'); } catch (e) { }
  addEventListener('deviceorientation', onOri); addEventListener('devicemotion', onMot);
  $('#voice').textContent = cap.voice ? '🔊' : '🔇';
  intro(0);
  cap.mtimer = setInterval(async () => { if (!cap.paused) { const r = await checkFrame(v); if (r) cap.markers = r; } }, 2500);
  loop();
}
function onOri(e) { if (e.alpha != null) { cap.ori = [e.alpha, e.beta, e.gamma]; cap.sensors = true; } }
function onMot(e) { const r = e.rotationRate; if (r && r.alpha != null) cap.rot = Math.hypot(r.alpha, r.beta, r.gamma); }
function angDiff(a, b) { if (!a || !b) return 999; const d = i => { let x = Math.abs(a[i] - b[i]) % 360; return x > 180 ? 360 - x : x; }; return Math.hypot(d(0), d(1), d(2)); }

function intro(p) {
  cap.phase = p; cap.paused = true;
  const ph = PHASES[p];
  $('#intro').innerHTML = `<div class="icard"><div class="step">Pass ${p + 1} of ${PHASES.length}</div><h2>${ph.t}</h2><div class="skwrap">${sketchFor(ph)}</div><p>${ph.d}</p>
    <p class="small mut">The phone takes photos by itself. Keep the arm <b>and fingers</b> exactly where they are: don't tap with the scanned hand.</p><button id="ibtn" class="big">Start pass ${p + 1}</button>
    <div class="row" style="margin-top:8px">${p ? '<button class="sec" id="iprev">◂ Previous pass</button>' : ''}<button class="sec" id="icancel">Cancel scan</button></div></div>`;
  $('#intro').style.display = 'flex';
  $('#mini').innerHTML = sketchFor(ph);
  speak(ph.t + '. ' + ph.d);
  const go = () => { clearInterval(cap.autoT); $('#intro').style.display = 'none'; cap.paused = false; cap.lastOri = null; };
  $('#ibtn').onclick = go;
  clearInterval(cap.autoT);
  if (p > 0) {  // later passes start by themselves: no tap needed, so the scanned arm and hand don't move
    let n = 6;
    cap.autoT = setInterval(() => {
      n--; if (!$('#ibtn')) return clearInterval(cap.autoT);
      $('#ibtn').textContent = n > 0 ? `Starting in ${n}… (tap to start now)` : 'Go';
      if (n === 3) speak('Three, two, one');
      if (n <= 0) go();
    }, 1000);
  }
  $('#icancel').onclick = () => { clearInterval(cap.autoT); cancelCapture(); };
  $('#iprev') && ($('#iprev').onclick = () => { clearInterval(cap.autoT); intro(p - 1); });
  renderHud();
}

function phaseProgress(p) {
  const ph = PHASES[p];
  if (cap.xr) {  // AR: sides of the arm really covered (needs a stable arm-centre estimate) + enough photos
    if (!cap.xr.centre || cap.xr.shots.length < 6) return Math.min(0.15, cap.pn[p] / ph.min);
    const need = ph.band ? (ph.covAR || 0.7) : 0;
    const cov = cap.bins[p].filter(b => b > 0).length / BINS;
    // scanning alone you can't reach all the way round: done when ~12 photos in a row added no new side
    const lastNew = (cap.newSideAt || [])[p] || 0;
    if (cap.pn[p] >= ph.min && cov >= 0.3 && cap.pn[p] - lastNew >= 12) { cap.reached = true; return 1; }
    return Math.min(need ? cov / need : 1, cap.pn[p] / ph.min, 1);
  }
  if (ph.needCards) return Math.min(cap.cardHits / ph.needCards, cap.pn[p] / ph.min, 1);
  const need = ph.cov || 0.6;
  const cov = cap.sensors && ph.band ? cap.bins[p].filter(b => b > 0).length / BINS : need * Math.min(1, cap.pn[p] / ph.min);
  return Math.min(cov / need, cap.pn[p] / ph.min, 1);
}

const scv = document.createElement('canvas'); scv.width = 240; scv.height = 180; const sctx = scv.getContext('2d', { willReadFrequently: true });
function sharpness(v) {
  sctx.drawImage(v, 0, 0, 240, 180); const d = sctx.getImageData(0, 0, 240, 180).data; const W = 240, H = 180; const g = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) g[i] = d[4 * i] * .3 + d[4 * i + 1] * .59 + d[4 * i + 2] * .11;
  let s = 0, s2 = 0, n = 0;
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) { const i = y * W + x; const l = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - W] - g[i + W]; s += l; s2 += l * l; n++; }
  return s2 / n - (s / n) ** 2;
}

function setHint(t, cls = '') {
  const now = performance.now();
  if (t !== cap.hint && now - cap.hintT > 1500) { cap.hint = t; cap.hintT = now; $('#hint').innerHTML = t; $('#hint').className = cls; if (cls === 'warn') speak(t); }
}

function loop() {
  if (!cap.on || cap.xr) return;
  const v = $('#vid'), now = performance.now();
  if (!cap.paused && v.videoWidth && now - (cap.lastEval || 0) > 120) {
    cap.lastEval = now;
    const ph = PHASES[cap.phase];
    const s = sharpness(v); cap.sharpRef = Math.max(cap.sharpRef * 0.995, s);
    const sharp = s > 0.45 * cap.sharpRef, steady = cap.rot < 90;  // deg/s: normal hand-held motion is fine
    const el = elevation(cap.ori);
    const inBand = !ph.band || el == null || (el >= ph.band[0] - 10 && el <= ph.band[1] + 10);
    cap.offBandSince = inBand ? 0 : (cap.offBandSince || now);
    const moved = cap.ori ? angDiff(cap.ori, cap.lastOri) > 3 : now - cap.last > 700;
    if (ph.needCards) {
      const left = Math.max(0, ph.needCards - cap.cardHits);
      const seenNow = cap.markers && cap.markers.cards > 0;
      setHint(!left ? '✓ Size locked' : seenNow ? `Card seen ✓ — keep moving left ↔ right (${left} more)` : 'Show the whole card, flat against the arm, all 4 edges visible', seenNow || !left ? 'ok' : 'warn');
    }
    else if (!steady) setHint('A bit slower', 'warn');
    else if (!sharp) setHint('Hold still a moment: focusing…', '');
    else if (cap.offBandSince && now - cap.offBandSince > 4000) setHint(el < ph.band[0] ? 'Tip: tilt the phone up a little' : 'Tip: tilt the phone down a little', '');
    else setHint(cap.pn[cap.phase] < 3 ? 'Good. Start moving around slowly' : 'Keep going, smooth and steady', 'ok');
    const limb = sharp && steady && moved && now - cap.last > 300 ? limbInFrame(v) : null;
    if (limb && limb.frac <= 0.04) setHint('Point the phone at the arm — it is not in the picture', 'warn');
    else if (sharp && steady && moved && now - cap.last > 300) grab(v);
    $('#qs').className = 'q ' + (sharp ? 'ok' : 'bad'); $('#qm').className = 'q ' + (steady ? 'ok' : 'warn');
    const mk = cap.markers; const ref = PHASES[cap.phase].needCards ? mk && mk.cards > 0 : cap.cardHits >= 3; $('#qk').className = 'q ' + (ref ? 'ok' : 'warn'); $('#tk').textContent = PHASES[cap.phase].needCards ? `card ${cap.cardHits}/${PHASES[cap.phase].needCards}` : 'card ✓';
    drawTilt(el, ph.band);
    if (phaseProgress(cap.phase) >= 1 && !cap.paused) phaseDone();
  }
  drawRing();
  requestAnimationFrame(loop);
}

function phaseDone() {
  cap.paused = true; navigator.vibrate?.([40, 60, 40]);
  if (cap.phase < PHASES.length - 1) {
    const cal = PHASES[cap.phase].needCards;
    const msg = cal ? '✓ Size locked — put the card away' : cap.reached ? '✓ Covered what you can reach' : '✓ Pass complete';
    cap.reached = false;
    setHint(msg, 'ok'); speak(cal ? 'Size locked. Put the card away.' : 'Pass complete');
    setTimeout(() => intro(cap.phase + 1), cal ? 1800 : 900);
  }
  else { $('#intro').innerHTML = `<div class="icard"><h2>✓ All passes done</h2><p>${cap.n} photos captured. Keep the arm still until the upload finishes.</p><button id="ibtn" class="big">Finish &amp; process</button><button id="more" class="sec" style="margin-top:8px;width:100%">Add more photos</button></div>`;
    $('#intro').style.display = 'flex'; speak('All passes done. Tap finish.'); $('#ibtn').onclick = finish; $('#more').onclick = () => { $('#intro').style.display = 'none'; cap.paused = false; PHASES[cap.phase].min += 15; }; }
}

function grab(v) {
  cap.last = performance.now(); cap.lastOri = cap.ori ? [...cap.ori] : null;
  if (cap.ori) cap.bins[cap.phase][Math.floor((((cap.ori[0] % 360) + 360) % 360) / (360 / BINS))]++;
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext('2d').drawImage(v, 0, 0);
  cap.n++; cap.pn[cap.phase]++; navigator.vibrate?.(12);
  const f = $('#flash'); f.style.opacity = .3; setTimeout(() => f.style.opacity = 0, 80);
  c.toBlob(b => { cap.queue.push(b); pump(); }, 'image/jpeg', 0.93);
  if (PHASES[cap.phase].needCards) calibCheck(c);
  renderHud();
}
async function calibCheck(c) {
  const w = Math.min(1920, c.width), h = Math.round(w * c.height / c.width);
  const s = document.createElement('canvas'); s.width = w; s.height = h; s.getContext('2d').drawImage(c, 0, 0, w, h);
  const b = await new Promise(r => s.toBlob(r, 'image/jpeg', 0.85));
  const fd = new FormData(); fd.append('frame', b, 'f.jpg');
  try { const r = await api('/api/check', { method: 'POST', body: fd }); cap.markers = r; if (r.cards > 0) { cap.cardHits++; navigator.vibrate?.(25); } } catch (e) { }
  renderHud();
}

async function pump() {
  while (cap.inflight < 2 && cap.queue.length) {
    const item = cap.queue.shift(); cap.inflight++;
    const b = item.blob || item;
    const fd = new FormData(); fd.append('files', b, `live_${Date.now()}.jpg`);
    if (item.meta) fd.append('meta', JSON.stringify(item.meta));
    xhrUpload(`/api/scans/${cap.id}/files`, fd).then(() => { cap.up++; }).catch(() => cap.queue.push(item)).finally(() => { cap.inflight--; renderHud(); pump(); });
  }
}
function renderHud() {
  $('#big').textContent = cap.n;
  $('#ptitle').textContent = `${cap.phase + 1}/${PHASES.length} · ${PHASES[cap.phase].t.split('·')[1].trim()}`;
  $('#upbar').style.width = (100 * cap.up / Math.max(cap.n, 1)) + '%';
  $('#steps').innerHTML = PHASES.map((p, k) => `<i class="${k < cap.phase ? 'done' : k === cap.phase ? 'on' : ''}" style="--p:${Math.round(100 * phaseProgress(k))}%"></i>`).join('');
}
function drawRing() {
  const c = $('#ring'), x = c.getContext('2d'); x.clearRect(0, 0, 220, 220);
  const bins = cap.bins[cap.phase] || [];
  for (let i = 0; i < BINS; i++) { const a0 = (i / BINS) * 2 * Math.PI - Math.PI / 2, a1 = ((i + 1) / BINS) * 2 * Math.PI - Math.PI / 2;
    x.beginPath(); x.arc(110, 110, 90, a0 + .03, a1 - .03); x.lineWidth = 24; x.strokeStyle = bins[i] ? 'rgba(63,185,80,.95)' : 'rgba(255,255,255,.2)'; x.stroke(); }
  if (cap.ori) { const a = (cap.ori[0] / 360) * 2 * Math.PI - Math.PI / 2; x.beginPath(); x.arc(110 + 90 * Math.cos(a), 110 + 90 * Math.sin(a), 11, 0, 7); x.fillStyle = '#fff'; x.fill(); }
  x.fillStyle = '#fff'; x.font = 'bold 40px sans-serif'; x.textAlign = 'center'; x.fillText(Math.round(100 * phaseProgress(cap.phase)) + '%', 110, 124);
}
function drawTilt(el, band) {
  const c = $('#tilt'), x = c.getContext('2d'), H = 240, Y = d => H / 2 - d * (H / 2 - 10) / 90; x.clearRect(0, 0, 60, H);
  x.fillStyle = 'rgba(255,255,255,.15)'; x.fillRect(22, 10, 16, H - 20);
  if (band) { x.fillStyle = 'rgba(63,185,80,.6)'; x.fillRect(22, Y(band[1]), 16, Y(band[0]) - Y(band[1])); }
  if (el != null) { x.fillStyle = '#fff'; x.beginPath(); x.moveTo(8, Y(el)); x.lineTo(20, Y(el) - 8); x.lineTo(20, Y(el) + 8); x.fill(); x.fillRect(20, Y(el) - 2, 22, 4); }
  x.fillStyle = '#fff'; x.font = '18px sans-serif'; x.textAlign = 'center'; x.fillText('▲', 30, 9 + 12); x.fillText('▼', 30, H - 4);
}
function stopCapture() {
  cap.on = false; clearInterval(cap.mtimer); removeEventListener('deviceorientation', onOri); removeEventListener('devicemotion', onMot);
  cap.stream?.getTracks().forEach(t => t.stop()); cap.wake?.release?.(); speechSynthesis?.cancel(); $('#cap').style.display = 'none';
  if (cap.xr) { try { cap.xr.session.end(); } catch (e) { } cap.xr = null; $('#cap').classList.remove('xr'); }
}
async function finish() {
  if (cap.n < 20 && !confirm(`Only ${cap.n} photos: results may fail. Process anyway?`)) return;
  cap.paused = true; $('#intro').innerHTML = `<div class="icard"><h2>Uploading…</h2><p id="ut"></p></div>`; $('#intro').style.display = 'flex';
  while (cap.queue.length || cap.inflight) { $('#ut').textContent = `${cap.up} / ${cap.n} photos sent`; await new Promise(r => setTimeout(r, 300)); }
  stopCapture(); await api(`/api/scans/${cap.id}/start`, { method: 'POST' }); location.hash = '/scan/' + cap.id;
}
$('#cpause').onclick = () => { if ($('#intro').style.display === 'flex') return; cap.paused = !cap.paused; $('#cpause').textContent = cap.paused ? '▶︎' : '⏸'; setHint(cap.paused ? 'Paused' : 'Resumed', ''); };
$('#cnext').onclick = () => { if (cap.phase < PHASES.length - 1) intro(cap.phase + 1); else phaseDone(); };
$('#cdone').onclick = finish;
async function cancelCapture() { if (!confirm('Discard this scan?')) return; stopCapture(); await fetch(`/api/scans/${cap.id}`, { method: 'DELETE' }); location.hash = '/'; route(); }
$('#ccancel').onclick = cancelCapture;
$('#voice').onclick = () => { cap.voice = !cap.voice; store.set('voice', cap.voice ? '1' : '0'); $('#voice').textContent = cap.voice ? '🔊' : '🔇'; if (cap.voice) speak('Voice guidance on'); };

// ---------------- WebXR (ARCore) capture: metric phone pose with every photo, no card ----------------
async function detectXR() {
  try { XR.supported = !!(window.isSecureContext && navigator.xr && await navigator.xr.isSessionSupported('immersive-ar')); } catch (e) { XR.supported = false; }
}

async function startXR() {
  const root = $('#cap');
  loadLimbModel();
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2', { xrCompatible: true }) || canvas.getContext('webgl', { xrCompatible: true });
  const session = await navigator.xr.requestSession('immersive-ar', {
    requiredFeatures: ['camera-access'], optionalFeatures: ['dom-overlay', 'local'], domOverlay: { root } });
  await gl.makeXRCompatible?.();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  const ref = await session.requestReferenceSpace('local');
  const binding = new XRWebGLBinding(session, gl);
  PHASES = PHASE_SETS[store.get('pose') || 'upright'].filter(p => !p.needCards).map(p => ({ ...p }));
  Object.assign(cap, { cardHits: 0, newSideAt: [], reached: false, offBandSince: 0, on: true, paused: true, stream: null, id: await newScan(), n: 0, up: 0, inflight: 0, queue: [],
    last: 0, lastOri: null, ori: null, rot: 0, sharpRef: 0, phase: 0, pn: PHASES.map(() => 0), bins: PHASES.map(() => new Array(BINS).fill(0)),
    markers: null, hintT: 0, hint: '', voice: store.get('voice') === '1', sensors: true,
    xr: { session, gl, ref, binding, fb: gl.createFramebuffer(), lastPos: null, lastFwd: null, prevFwd: null, prevT: 0, tracking: false,
          shots: [], centre: null, stillSince: 0 } });
  root.classList.add('xr'); root.style.display = 'block';
  root.addEventListener('beforexrselect', e => e.preventDefault());  // taps on our buttons are not AR "select"
  session.addEventListener('end', () => { if (cap.on) { cap.on = false; root.classList.remove('xr'); root.style.display = 'none'; } });
  try { cap.wake = await navigator.wakeLock?.request('screen'); } catch (e) { }
  $('#voice').textContent = cap.voice ? '🔊' : '🔇';
  intro(0);
  session.requestAnimationFrame(xrFrame);
}

// ---- on-device limb detection: MediaPipe selfie-multiclass "body skin" (the model the server also uses)
const LIMB = { seg: null, loading: null };
async function loadLimbModel() {
  if (LIMB.seg || LIMB.loading) return LIMB.loading;
  LIMB.loading = (async () => {
    const V = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
    const { FilesetResolver, ImageSegmenter } = await import(V + '/vision_bundle.mjs');
    const fs = await FilesetResolver.forVisionTasks(V + '/wasm');
    LIMB.seg = await ImageSegmenter.createFromOptions(fs, { baseOptions: { modelAssetPath: '/models/selfie_multiclass.tflite', delegate: 'GPU' },
      runningMode: 'IMAGE', outputCategoryMask: false, outputConfidenceMasks: true });
  })().catch(e => { console.warn('limb model unavailable', e); LIMB.failed = true; });
  return LIMB.loading;
}
const segCanvas = document.createElement('canvas');
function limbInFrame(src) {
  // returns {frac, edge} or null if the model isn't ready: frac = share of the frame that is limb skin,
  // edge = limb touches the top/bottom/side border (hand or arm cut off)
  if (!LIMB.seg) return null;
  const sw = src.videoWidth || src.width, sh = src.videoHeight || src.height;
  const w = 192, h = Math.round(192 * sh / sw); segCanvas.width = w; segCanvas.height = h;
  segCanvas.getContext('2d').drawImage(src, 0, 0, w, h);
  const r = LIMB.seg.segment(segCanvas);
  const m = r.confidenceMasks[2].getAsFloat32Array(); const mw = r.confidenceMasks[2].width, mh = r.confidenceMasks[2].height;
  // luminance of limb vs background (backlight: dark silhouette in front of a window = no skin detail)
  const px = segCanvas.getContext('2d').getImageData(0, 0, w, h).data;
  let n = 0, e = 0, lumL = 0, lumB = 0, nb = 0;
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) {
    const sx = Math.min(w - 1, Math.floor(x * w / mw)), sy = Math.min(h - 1, Math.floor(y * h / mh)), i = 4 * (sy * w + sx);
    const lum = 0.3 * px[i] + 0.59 * px[i + 1] + 0.11 * px[i + 2];
    if (m[y * mw + x] > 0.5) { n++; lumL += lum; if (y < 2 || x < 2 || x >= mw - 2) e++; } else { lumB += lum; nb++; }
  }
  r.close?.();
  return { frac: n / (mw * mh), edge: e / Math.max(n, 1), limbLum: lumL / Math.max(n, 1), bgLum: lumB / Math.max(nb, 1) };
}

// Least-squares point closest to all camera viewing rays = where the arm is (the phone always looks at it).
function rayCentre(shots) {
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  for (const { pos: c, fwd: d } of shots) {
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { const m = (i === j ? 1 : 0) - d[i] * d[j]; A[i][j] += m; b[i] += m * c[j]; }
  }
  const det = M => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const D = det(A); if (Math.abs(D) < 1e-9) return null;
  return [0, 1, 2].map(k => det(A.map((row, i) => row.map((v, j) => j === k ? b[i] : v))) / D);
}
const azimuthAround = (c, p) => ((Math.atan2(p[0] - c[0], p[2] - c[2]) * 180 / Math.PI) % 360 + 360) % 360;
function isDuplicate(shots, pos, fwd) {  // a photo from (almost) the same place and angle adds nothing
  for (const s of shots) {
    const dp = Math.hypot(pos[0] - s.pos[0], pos[1] - s.pos[1], pos[2] - s.pos[2]);
    const da = Math.acos(Math.min(1, fwd[0] * s.fwd[0] + fwd[1] * s.fwd[1] + fwd[2] * s.fwd[2])) * 180 / Math.PI;
    if (dp < 0.035 && da < 7) return true;
  }
  return false;
}
function nextDirection(bins, cur, right, c, pos) {
  // which way around the arm is the nearest uncovered side? compare with the phone's right-hand direction
  const nb = bins.length, ci = Math.floor(cur / (360 / nb)) % nb;
  let best = null;
  for (let k = 1; k <= nb / 2; k++) {
    if (!bins[(ci + k) % nb]) { best = +1; break; }
    if (!bins[(ci - k + nb) % nb]) { best = -1; break; }
  }
  if (best === null) return null;
  const azR = azimuthAround(c, [pos[0] + 0.05 * right[0], pos[1] + 0.05 * right[1], pos[2] + 0.05 * right[2]]);
  let dR = azR - cur; dR = ((dR + 540) % 360) - 180;  // + if moving right increases azimuth
  return best * Math.sign(dR || 1) > 0 ? 'right' : 'left';
}

function readCamera(x, cam) {
  const gl = x.gl, w = cam.width, h = cam.height;
  const tex = x.binding.getCameraImage(cam);
  gl.bindFramebuffer(gl.FRAMEBUFFER, x.fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const px = new Uint8ClampedArray(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, x.session.renderState.baseLayer.framebuffer);
  const t = document.createElement('canvas'); t.width = w; t.height = h; t.getContext('2d').putImageData(new ImageData(px, w, h), 0, 0);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d'); ctx.translate(0, h); ctx.scale(1, -1); ctx.drawImage(t, 0, 0);  // GL rows are bottom-up
  return c;
}

function xrFrame(t, frame) {
  const x = cap.xr; if (!cap.on || !x) return;
  x.session.requestAnimationFrame(xrFrame);
  const gl = x.gl;
  gl.bindFramebuffer(gl.FRAMEBUFFER, x.session.renderState.baseLayer.framebuffer);
  gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  const pose = frame.getViewerPose(x.ref);
  drawRing();
  if (!pose || pose.emulatedPosition) { x.tracking = false; setHint('Move the phone slowly so it can track…', 'warn'); return; }
  x.tracking = true;
  const view = pose.views[0], m = view.transform.matrix;
  const pos = [m[12], m[13], m[14]], fwd = [-m[8], -m[9], -m[10]];
  const az = Math.atan2(fwd[0], -fwd[2]) * 180 / Math.PI, el = Math.asin(Math.max(-1, Math.min(1, fwd[1]))) * 180 / Math.PI;
  cap.ori = [((az % 360) + 360) % 360, 0, 0];
  if (x.prevFwd) { const dt = (t - x.prevT) / 1000; const d = Math.acos(Math.min(1, fwd[0] * x.prevFwd[0] + fwd[1] * x.prevFwd[1] + fwd[2] * x.prevFwd[2])) * 180 / Math.PI; if (dt > 0) cap.rot = 0.7 * cap.rot + 0.3 * d / dt; }
  x.prevFwd = fwd; x.prevT = t;
  const now = performance.now();
  if (cap.paused || now - (cap.lastEval || 0) < 120) return;
  cap.lastEval = now;
  const ph = PHASES[cap.phase];
  const steady = cap.rot < 90;
  const right = [m[0], m[1], m[2]];
  if (x.shots.length >= 4) x.centre = rayCentre(x.shots) || x.centre;
  const c = x.centre;
  if (c && x.shots.length >= 6) {  // recount sides of the arm for this pass with the current (improving) centre
    const b = new Array(BINS).fill(0);
    for (const s of x.shots) if (s.phase === cap.phase) b[Math.floor(azimuthAround(c, s.pos) / (360 / BINS)) % BINS]++;
    cap.bins[cap.phase] = b;
  }
  const dist = c ? Math.hypot(pos[0] - c[0], pos[1] - c[1], pos[2] - c[2]) : null;
  const azArm = c ? azimuthAround(c, pos) : cap.ori[0];
  cap.ori = [azArm, 0, 0];                       // the ring now shows sides of the ARM, not phone heading
  const inBand = !ph.band || (el >= ph.band[0] - 10 && el <= ph.band[1] + 10);
  cap.offBandSince = inBand ? 0 : (cap.offBandSince || now);
  const dup = isDuplicate(x.shots, pos, fwd);
  x.stillSince = dup ? (x.stillSince || now) : 0;
  const dir = c ? nextDirection(cap.bins[cap.phase], azArm, right, c, pos) : null;
  if (!steady) setHint('A bit slower', 'warn');
  else if (dist && dist > 0.6) setHint('Move closer to the arm (30–40 cm)', 'warn');
  else if (dist && dist < 0.18) setHint('Too close: move back a little', 'warn');
  else if (x.stillSince && now - x.stillSince > 1500) setHint(dir ? `Got this view ✓ Now move ${dir === 'left' ? '⟵ to your LEFT' : 'to your RIGHT ⟶'}` : 'Got this view ✓ Move to a new spot', 'ok');
  else if (cap.offBandSince && now - cap.offBandSince > 4000) setHint(el < ph.band[0] ? 'Tip: tilt the phone up a little' : 'Tip: tilt the phone down a little', '');
  else if (dir && cap.pn[cap.phase] > 4) setHint(`Good — keep going ${dir === 'left' ? '⟵ left' : 'right ⟶'} around the arm`, 'ok');
  else setHint(cap.pn[cap.phase] < 3 ? 'Good. Start moving around the arm slowly' : 'Keep going, smooth and steady', 'ok');
  $('#qm').className = 'q ' + (steady ? 'ok' : 'warn'); $('#qk').className = 'q ' + (cap.noLimb && now - cap.noLimb < 1500 ? 'bad' : 'ok'); $('#tk').textContent = 'arm';
  drawTilt(el, ph.band);
  if (steady && !dup && now - cap.last > 250 && view.camera && (!dist || (dist < 0.65 && dist > 0.15))) {
    const cimg = readCamera(x, view.camera);
    // AR cameras are fixed-focus on most phones: judge blur against the TYPICAL recent frame, not the best ever
    const s = sharpness(cimg);
    (x.sharpHist = x.sharpHist || []).push(s); if (x.sharpHist.length > 25) x.sharpHist.shift();
    const med = [...x.sharpHist].sort((p, q) => p - q)[Math.floor(x.sharpHist.length / 2)];
    const sharp = x.sharpHist.length < 5 || s >= 0.7 * med;
    const limb = limbInFrame(cimg);
    const backlit = limb && limb.frac > 0.04 && (limb.limbLum < 55 || limb.bgLum > 1.7 * limb.limbLum);
    const limbOk = (!limb || limb.frac > 0.04) && !backlit;
    $('#qs').className = 'q ' + (sharp ? 'ok' : 'bad');
    if (backlit) { cap.noLimb = now; setHint('Light is behind the arm — turn so the window / lamp is behind YOU', 'warn'); }
    else if (!limbOk) { cap.noLimb = now; setHint('Point the phone at the arm — it is not in the picture', 'warn'); }
    else if (limb && limb.edge > 0.25 && limb.frac > 0.35) setHint('Move back a little: part of the hand is cut off', '');
    if (sharp && limbOk) {
      x.lastPos = pos; x.lastFwd = fwd; cap.last = now;
      x.shots.push({ pos, fwd, phase: cap.phase });
      const before = cap.bins[cap.phase].filter(v => v > 0).length;
      cap.bins[cap.phase][Math.floor(azArm / (360 / BINS)) % BINS]++;
      cap.n++; cap.pn[cap.phase]++; navigator.vibrate?.(12);
      if (cap.bins[cap.phase].filter(v => v > 0).length > before) (cap.newSideAt = cap.newSideAt || [])[cap.phase] = cap.pn[cap.phase];
      const meta = { pose: Array.from(m), proj: Array.from(view.projectionMatrix), w: cimg.width, h: cimg.height, t: now, src: 'webxr' };
      cimg.toBlob(b => { cap.queue.push({ blob: b, meta }); pump(); }, 'image/jpeg', 0.93);
      renderHud();
    }
  }
  if (phaseProgress(cap.phase) >= 1 && !cap.paused) phaseDone();
}

window.__limbscan = { startCapture, openCamera, startXR, loadLimbModel, limbInFrame }; // test hook
detectXR().then(route);
