// Results / processing page: status polling, 3D viewer, measurements, reference comparison.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { $, api, fmt, ui } from './common.js';
let viewer = null;
export function disposeResults() { if (viewer) { viewer.dispose(); viewer = null; } }

// ---------------- scan page ----------------
export async function scanPage(id) {
  const app = $('#app');
  app.innerHTML = `<div id="sp"><div class="card"><span class="mut">Loading…</span></div></div>`;
  let shown = null;
  const tick = async () => {
    let s; try { s = await api('/api/scans/' + id); } catch (e) { app.innerHTML = '<div class="card">Scan not found</div>'; clearInterval(ui.pollTimer); return; }
    const st = s.status.stage;
    if (st === 'done' && s.measurements) { if (shown !== 'done' || (s.compare.length && !shown_cmp)) { shown = 'done'; shown_cmp = s.compare.length > 0; renderResult(id, s); } if (shown_cmp) clearInterval(ui.pollTimer); return; }
    shown = st;
    $('#sp').innerHTML = `<div class="card"><h2>${s.meta.name}</h2>
      <div class="row" style="align-items:center"><div style="flex:3"><div class="bar"><i style="width:${100 * (s.status.progress || 0)}%"></i></div></div><span class="pill ${st === 'failed' ? 'bad' : 'run'}" style="flex:0">${st}</span></div>
      <p class="small mut">${s.n_input} file(s) uploaded. ${st === 'failed' ? '<b style="color:var(--bad)">' + (s.status.error || '') + '</b>' : 'Processing takes ~10–30 min on this computer. You can close this page.'}</p>
      <pre>${(s.log_tail || []).join('\n')}</pre>
      ${st === 'capturing' ? `<button id="go">Start processing</button>` : ''}
      ${st === 'failed' ? `<button class="sec" id="retry">Retry</button>` : ''}
      <button class="sec" id="del" style="margin-top:8px">Delete</button></div>`;
    $('#go') && ($('#go').onclick = () => api(`/api/scans/${id}/start`, { method: 'POST' }));
    $('#retry') && ($('#retry').onclick = () => api(`/api/scans/${id}/start`, { method: 'POST' }));
    $('#del').onclick = async () => { if (confirm('Delete scan?')) { await fetch('/api/scans/' + id, { method: 'DELETE' }); location.hash = '/'; } };
  };
  let shown_cmp = false;
  await tick(); ui.pollTimer = setInterval(tick, 3000);
}

function qa(m) {
  const s = m.scale || {}; const e = Math.abs(s.edge_vs_sheet_pct ?? 99);
  if (s.method === 'ar') {  // phone AR tracking (WebXR / ARKit): no card
    const ok = s.ar_scale_spread_pct < 0.8 && s.ar_frames >= 20 && s.ar_path_mm > 300;
    return { ok, html: `<span class="pill ${ok ? 'ok' : 'bad'}">${ok ? 'Scale verified (phone AR)' : 'Scale check FAILED'}</span>
      <span class="small mut"> ${s.ar_frames}/${s.ar_frames_total} AR frames · camera path ${fmt(s.ar_path_mm / 10, 0)} cm · scale consistency ±${fmt(s.ar_scale_spread_pct, 2)}% · ${s.sfm_images} views
      ${ok ? '' : '<br><b style="color:var(--bad)">Rescan moving the phone further around the arm (at least 30 cm path).</b>'}</span>` };
  }
  if (s.card_views != null) {  // ID-1 card reference
    const ok = e < 0.6 && s.card_views >= 8 && (s.median_reproj_px ?? 9) < 1.5;
    return { ok, html: `<span class="pill ${ok ? 'ok' : 'bad'}">${ok ? 'Scale verified (card)' : 'Scale check FAILED'}</span>
      <span class="small mut"> card seen in ${s.card_views} photos · sides ${(s.card_sides_mm || []).map(x => fmt(x, 1)).join(' / ')} mm · shape check ${fmt(s.edge_vs_sheet_pct, 2)}% · reproj ${fmt(s.median_reproj_px, 2)} px · ${s.sfm_images} views
      ${ok ? '' : '<br><b style="color:var(--bad)">Rescan with the card fully visible in more photos.</b>'}</span>` };
  }
  const ok = e < 0.5 && (s.sheet_fits || []).length >= 1 && (s.median_reproj_px ?? 9) < 1.5;
  return { ok, html: `<span class="pill ${ok ? 'ok' : 'bad'}">${ok ? 'Scale verified (sheets)' : 'Scale check FAILED'}</span>
    <span class="small mut"> sheets ${(s.sheet_fits || []).length} · markers ${s.markers} · fit RMS ${fmt((s.sheet_fits || [])[0]?.rms_mm, 2)} mm · edge check ${fmt(s.edge_vs_sheet_pct, 2)}% · reproj ${fmt(s.median_reproj_px, 2)} px · ${s.sfm_images} views</span>` };
}

function renderResult(id, s) {
  const m = s.measurements, L = m.landmarks || {};
  const f = n => `/api/scans/${id}/file/result/${n}`;
  const q = qa(m);
  const cmpRows = s.compare.map(c => `<tr><td>${c.reference}</td><td>${fmt(c.surface_median_abs_mm, 2)}</td><td>${fmt(c.girth_mean_abs_diff_mm, 1)}</td><td>${fmt(c.girth_mean_diff_mm, 1)}</td><td>${fmt(c.zone_volume_diff_pct, 1)}%</td><td>${fmt(c.coverage_pct_within_3mm, 0)}%</td></tr>`).join('');
  $('#sp').innerHTML = `
  <div class="card"><h2>${s.meta.name}</h2>${q.html}
    <div style="margin-top:6px"><span class="pill ${m.mesh.data_coverage_pct >= 90 ? 'ok' : 'bad'}">Surface seen: ${fmt(m.mesh.data_coverage_pct, 0)}%</span>
    <span class="small mut">${m.mesh.data_coverage_pct >= 90 ? 'complete capture' : 'parts of the limb were not seen and are interpolated — rescan with more views of the grey areas / better light'}</span></div></div>
  <div class="card" style="padding:8px"><div id="viewer"></div>
    <div class="row" style="padding:8px 4px 2px" id="layers"><button class="sec" data-l="limb">Scan</button>${s.compare.map(c => `<button class="sec" data-l="heat:${c.key}">Δ vs ${c.key.slice(0, 14)}</button>`).join('')}</div>
    <div class="small mut" style="padding:0 6px 6px" id="legend"></div></div>
  <div class="card"><h2>Measurements</h2><div class="grid">
    <div class="stat"><div class="v">${fmt(L.wrist?.girth_mm)}</div><div class="k">Wrist girth (mm)</div></div>
    <div class="stat"><div class="v">${fmt(L.wrist_plus_50?.girth_mm)}</div><div class="k">Girth wrist +50 mm</div></div>
    <div class="stat"><div class="v">${fmt(L.wrist_plus_100?.girth_mm)}</div><div class="k">Girth wrist +100 mm</div></div>
    <div class="stat"><div class="v">${fmt(L.wrist_plus_150?.girth_mm)}</div><div class="k">Girth wrist +150 mm</div></div>
    <div class="stat"><div class="v">${fmt(L.hand_max?.girth_mm)}</div><div class="k">Hand girth incl. thumb (mm)</div></div>
    <div class="stat"><div class="v">${fmt(m.length_mm, 0)}</div><div class="k">Scanned length (mm)</div></div>
    <div class="stat"><div class="v">${fmt(m.volume_ml, 0)}</div><div class="k">Forearm volume wrist→+150 (ml)</div></div>
    <div class="stat"><div class="v">${fmt(m.mesh.surface_area_cm2, 0)}</div><div class="k">Surface (cm²)</div></div></div>
    <h3>Girth profile (from fingertips)</h3><canvas id="chart"></canvas></div>
  ${s.compare.length ? `<div class="card"><h2>Accuracy vs reference scans</h2><p class="small mut">Forearm zone (wrist → +170 mm), after rigid alignment. Hand/fingers excluded because posture differs between sessions.</p>
    <table><tr><th>Reference</th><th>Surface median |Δ| mm</th><th>Girth mean |Δ| mm</th><th>Girth bias mm</th><th>Volume Δ</th><th>Coverage</th></tr>${cmpRows}</table></div>` : '<div class="card small mut">Comparing with reference scans…</div>'}
  <div class="card"><h2>Export</h2><div class="row"><a class="btn sec" href="${f('limb.stl')}">STL</a><a class="btn sec" href="${f('limb.obj')}">OBJ</a><a class="btn sec" href="${f('limb.ply')}">PLY (colour)</a><a class="btn sec" href="${f('limb_surface.ply')}">Raw surface</a><a class="btn sec" href="${f('measurements.json')}">Measurements JSON</a></div></div>`;
  viewer = makeViewer($('#viewer'));
  viewer.load(f('limb.glb'));
  document.querySelectorAll('#layers button').forEach(b => b.onclick = () => {
    const l = b.dataset.l;
    if (l === 'limb') { viewer.load(f('limb.glb')); $('#legend').textContent = ''; }
    else { const k = l.split(':')[1]; viewer.load(`/api/scans/${id}/file/compare/${encodeURIComponent("heatmap_" + k + ".glb")}`); $('#legend').innerHTML = 'Deviation from reference: <b style="color:#58f">−3 mm</b> · <b style="color:#3f3">0</b> · <b style="color:#f55">+3 mm</b> · grey = outside ±10 mm'; }
  });
  drawChart($('#chart'), m, s.compare[0]);
}

function drawChart(c, m, cmp) {
  const dpr = devicePixelRatio || 1; c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr; const x = c.getContext('2d'); x.scale(dpr, dpr);
  const W = c.clientWidth, H = c.clientHeight, P = 34; const wl = m.landmarks.wrist; const pr = m.profile.filter(p => !wl || p.s_mm <= wl.s_mm + 190);
  const xs = pr.map(p => p.s_mm), ys = pr.map(p => p.girth_mm);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys) * .9, y1 = Math.max(...ys) * 1.05;
  const X = v => P + (v - x0) / (x1 - x0) * (W - P - 8), Y = v => H - 22 - (v - y0) / (y1 - y0) * (H - 30);
  x.strokeStyle = '#30363d'; x.fillStyle = '#8b949e'; x.font = '11px sans-serif';
  for (let g = Math.ceil(y0 / 50) * 50; g < y1; g += 50) { x.beginPath(); x.moveTo(P, Y(g)); x.lineTo(W, Y(g)); x.stroke(); x.fillText(g, 2, Y(g) + 4); }
  for (let g = 0; g < x1; g += 50) { x.fillText(g, X(g) - 8, H - 6); }
  const line = (a, b, col, w) => { x.beginPath(); a.forEach((v, i) => i ? x.lineTo(X(v), Y(b[i])) : x.moveTo(X(v), Y(b[i]))); x.strokeStyle = col; x.lineWidth = w; x.stroke(); };
  line(xs, ys, '#2f81f7', 2.5);
  if (cmp && m.landmarks.wrist) { const w = m.landmarks.wrist.s_mm; line(cmp.girths.map(g => w + g.s_from_wrist_mm), cmp.girths.map(g => g.ref_mm), '#d29922', 1.5); x.fillStyle = '#d29922'; x.fillText('reference ' + cmp.reference, W - 200, 14); }
  for (const [k, v] of Object.entries(m.landmarks).filter(([k, v]) => v.s_mm <= x1)) { x.fillStyle = '#3fb950'; x.beginPath(); x.arc(X(v.s_mm), Y(v.girth_mm), 4, 0, 7); x.fill(); x.fillText(k.replace('wrist_plus_', 'W+').replace('_max', ''), X(v.s_mm) + 5, Y(v.girth_mm) - 6); }
}

function makeViewer(el) {
  const r = new THREE.WebGLRenderer({ antialias: true }); r.setPixelRatio(devicePixelRatio); r.setSize(el.clientWidth, el.clientHeight); el.appendChild(r.domElement);
  const sc = new THREE.Scene(); sc.background = new THREE.Color(0x0b0f14);
  const cam = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, 1, 10000);
  sc.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.6)); const dl = new THREE.DirectionalLight(0xffffff, 1.2); cam.add(dl); sc.add(cam);
  const ctl = new OrbitControls(cam, r.domElement); ctl.enableDamping = true;
  let obj = null, alive = true; const loader = new GLTFLoader();
  const onR = () => { cam.aspect = el.clientWidth / el.clientHeight; cam.updateProjectionMatrix(); r.setSize(el.clientWidth, el.clientHeight); }; addEventListener('resize', onR);
  (function an() { if (!alive) return; ctl.update(); r.render(sc, cam); requestAnimationFrame(an); })();
  return {
    load(url) { loader.load(url, g => { if (obj) sc.remove(obj); obj = g.scene;
      obj.traverse(o => { if (o.isMesh) { o.geometry.computeVertexNormals(); o.material = new THREE.MeshStandardMaterial({ vertexColors: !!o.geometry.attributes.color, roughness: .75, metalness: 0, side: THREE.DoubleSide }); } });
      sc.add(obj); const b = new THREE.Box3().setFromObject(obj), c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3()).length();
      ctl.target.copy(c); if (!this._placed) { cam.position.copy(c).add(new THREE.Vector3(0, -s * 1.3, s * .8)); cam.up.set(0, 0, 1); this._placed = true; } cam.near = s / 100; cam.far = s * 20; cam.updateProjectionMatrix(); }); },
    dispose() { alive = false; removeEventListener('resize', onR); r.dispose(); el.innerHTML = ''; }
  };
}

