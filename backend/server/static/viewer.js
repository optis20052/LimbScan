// 3D viewer for a scan: the limb stood upright (fingertips up, palm or back of the hand facing you), orbiting around the arm's own axis,
// view presets, clay / colour / wireframe shading, girth rings at the measured landmarks. Renders only while
// something moves (no background GPU/CPU use when idle).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const RING_COLS = { wrist: 0x2f81f7, wrist_plus_50: 0xa371f7, wrist_plus_100: 0xdb61a2, wrist_plus_150: 0xf0883e, hand_max: 0x3fb950 };
export const ringColour = k => '#' + (RING_COLS[k] ?? 0xd29922).toString(16).padStart(6, '0');

// .npy (float64, C order) -> array of [x, y, z]
async function loadNpy(url) {
  const b = await (await fetch(url)).arrayBuffer();
  const hl = new DataView(b).getUint16(8, true), off = 10 + hl;
  const a = new Float64Array(b.slice(off));
  const out = []; for (let i = 0; i + 2 < a.length; i += 3) out.push(new THREE.Vector3(a[i], a[i + 1], a[i + 2]));
  return out;
}

// centreline point and direction (distal -> proximal) at arc length s (mm from the fingertips)
function along(cl, s, win = 20) {
  const arc = [0]; for (let i = 1; i < cl.length; i++) arc.push(arc[i - 1] + cl[i].distanceTo(cl[i - 1]));
  const at = v => { v = Math.max(0, Math.min(arc.at(-1), v)); let i = 1; while (i < arc.length - 1 && arc[i] < v) i++;
    const t = (v - arc[i - 1]) / Math.max(1e-9, arc[i] - arc[i - 1]); return cl[i - 1].clone().lerp(cl[i], t); };
  return { p: at(s), t: at(s + win).sub(at(s - win)).normalize() };
}

// plane section of an indexed mesh -> line segment positions
function section(geo, p, n, rmax) {
  const P = geo.attributes.position.array, I = geo.index ? geo.index.array : null, nv = P.length / 3;
  const d = new Float32Array(nv);
  for (let i = 0; i < nv; i++) d[i] = (P[3 * i] - p.x) * n.x + (P[3 * i + 1] - p.y) * n.y + (P[3 * i + 2] - p.z) * n.z;
  const out = [], nt = I ? I.length / 3 : nv / 3, r2 = rmax * rmax;
  const cut = (u, v) => { const t = d[u] / (d[u] - d[v]); return [P[3 * u] + t * (P[3 * v] - P[3 * u]), P[3 * u + 1] + t * (P[3 * v + 1] - P[3 * u + 1]), P[3 * u + 2] + t * (P[3 * v + 2] - P[3 * u + 2])]; };
  for (let f = 0; f < nt; f++) {
    const a = I ? I[3 * f] : 3 * f, b = I ? I[3 * f + 1] : 3 * f + 1, c = I ? I[3 * f + 2] : 3 * f + 2;
    const sa = d[a] > 0, sb = d[b] > 0, sc = d[c] > 0;
    if (sa === sb && sb === sc) continue;
    const q = []; if (sa !== sb) q.push(cut(a, b)); if (sb !== sc) q.push(cut(b, c)); if (sc !== sa) q.push(cut(c, a));
    if (q.length !== 2) continue;
    if ((q[0][0] - p.x) ** 2 + (q[0][1] - p.y) ** 2 + (q[0][2] - p.z) ** 2 > r2) continue;
    out.push(...q[0], ...q[1]);
  }
  return new Float32Array(out);
}

export function makeViewer(el, { centrelineUrl, landmarks } = {}) {
  el.style.position = 'relative';
  const r = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  r.setPixelRatio(Math.min(devicePixelRatio, 2)); r.setSize(el.clientWidth, el.clientHeight);
  r.toneMapping = THREE.ACESFilmicToneMapping; r.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(r.domElement);
  const sc = new THREE.Scene(); sc.background = new THREE.Color(0x0b0f14);
  const pm = new THREE.PMREMGenerator(r); sc.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture; pm.dispose();
  const cam = new THREE.PerspectiveCamera(35, el.clientWidth / el.clientHeight, 1, 10000);
  const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(0.4, 0.6, 1); cam.add(key); sc.add(cam);

  const ctl = new OrbitControls(cam, r.domElement);
  Object.assign(ctl, { enableDamping: true, dampingFactor: 0.12, rotateSpeed: 0.7, zoomSpeed: 1.1, panSpeed: 0.8,
    screenSpacePanning: true, zoomToCursor: true, autoRotateSpeed: 3 });

  // model frame -> viewer frame: arm axis = +Y (fingertips up), palm normal = +Z (towards the default camera)
  const root = new THREE.Group(); sc.add(root);
  const model = new THREE.Group(); root.add(model);
  const rings = new THREE.Group(); rings.visible = false; root.add(rings);
  let cl = null, aligned = false, mode = 'clay', wire = false, size = 300, alive = true, anim = null, running = false;
  const loader = new GLTFLoader(), meshes = [];

  const clay = new THREE.MeshStandardMaterial({ color: 0xd9cfc4, roughness: 0.55, metalness: 0, side: THREE.DoubleSide, envMapIntensity: 0.7, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  const colour = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, metalness: 0, side: THREE.DoubleSide, envMapIntensity: 0.5, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  const wireMat = new THREE.MeshBasicMaterial({ color: 0x0b0f14, wireframe: true, transparent: true, opacity: 0.16 });

  function frame() {
    const changed = ctl.update();
    if (anim) {
      const t = Math.min(1, (performance.now() - anim.t0) / anim.ms), e = 1 - (1 - t) ** 3;
      cam.position.copy(anim.p0).lerp(anim.p1, e); ctl.target.copy(anim.q0).lerp(anim.q1, e);
      cam.up.copy(anim.u0).lerp(anim.u1, e).normalize(); cam.lookAt(ctl.target);
      if (t >= 1) anim = null;
    }
    r.render(sc, cam);
    if (alive && (changed || anim || ctl.autoRotate)) requestAnimationFrame(frame); else running = false;
  }
  const kick = () => { if (!running && alive) { running = true; requestAnimationFrame(frame); } };
  ctl.addEventListener('change', kick); ctl.addEventListener('start', kick);
  const ro = new ResizeObserver(() => { cam.aspect = el.clientWidth / el.clientHeight; cam.updateProjectionMatrix(); r.setSize(el.clientWidth, el.clientHeight); kick(); });
  ro.observe(el);

  function align() {
    // world bounding box of the model in its own frame, then a rotation that stands it up
    const box = new THREE.Box3().setFromObject(model), c = box.getCenter(new THREE.Vector3());
    let Y = new THREE.Vector3(0, 1, 0), Z = new THREE.Vector3(0, 0, 1);
    if (cl && cl.length > 4) {
      const len = cl.reduce((a, p, i) => i ? a + p.distanceTo(cl[i - 1]) : 0, 0);
      const tip = cl[0], hand = along(cl, Math.min(80, len / 3)).p;
      Y = tip.clone().sub(along(cl, Math.min(160, len * 0.7)).p).normalize();  // proximal -> fingertips
      // palm normal: the thinnest direction of the hand across the axis
      const pts = [], v = new THREE.Vector3();
      for (const m of meshes) { const P = m.geometry.attributes.position.array;
        for (let i = 0; i < P.length; i += 3 * 7) { v.set(P[i], P[i + 1], P[i + 2]); const s = v.clone().sub(tip).dot(Y);
          if (s < -15 && s > -110) pts.push(v.clone().sub(hand).addScaledVector(Y, -v.clone().sub(hand).dot(Y))); } }
      const e1 = Math.abs(Y.x) < 0.9 ? new THREE.Vector3(1, 0, 0).cross(Y).normalize() : new THREE.Vector3(0, 1, 0).cross(Y).normalize();
      const e2 = Y.clone().cross(e1);
      let a = 0, b = 0, cc = 0; for (const q of pts) { const x = q.dot(e1), y = q.dot(e2); a += x * x; b += x * y; cc += y * y; }
      const th = 0.5 * Math.atan2(2 * b, a - cc);  // major axis angle (palm width)
      const W = e1.clone().multiplyScalar(Math.cos(th)).addScaledVector(e2, Math.sin(th));
      Z = Y.clone().cross(W).normalize();
      c.copy(along(cl, len * 0.45).p);
    }
    const X = Y.clone().cross(Z).normalize(); Z.copy(X.clone().cross(Y));
    const R = new THREE.Matrix4().makeBasis(X, Y, Z).transpose();
    root.matrixAutoUpdate = false;
    root.matrix.copy(R).multiply(new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
    root.updateMatrixWorld(true);
    const b2 = new THREE.Box3().setFromObject(model); size = b2.getSize(new THREE.Vector3()).length();
    root.matrix.premultiply(new THREE.Matrix4().makeTranslation(0, -b2.getCenter(new THREE.Vector3()).y, 0)); root.updateMatrixWorld(true);
    cam.near = size / 200; cam.far = size * 30; cam.updateProjectionMatrix();
    ctl.minDistance = size * 0.15; ctl.maxDistance = size * 6;
  }

  function buildRings() {
    rings.clear();
    if (!cl || !landmarks || !meshes.length) return;
    for (const [k, v] of Object.entries(landmarks)) {
      if (v.s_mm == null) continue;
      const { p, t } = along(cl, v.s_mm);
      const pos = meshes.map(m => section(m.geometry, p, t, 120));
      const all = new Float32Array(pos.reduce((a, x) => a + x.length, 0)); let o = 0; for (const x of pos) { all.set(x, o); o += x.length; }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(all, 3));
      const ln = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: RING_COLS[k] ?? 0xd29922, depthTest: false, transparent: true, opacity: 0.95 }));
      ln.renderOrder = 2; ln.scale.setScalar(1.004); ln.position.copy(p).multiplyScalar(-0.004);  // just outside the skin
      rings.add(ln);
    }
  }

  function applyMaterials() {
    for (const m of meshes) {
      const hasCol = !!m.geometry.attributes.color;
      m.material = m.userData.heat ? m.userData.heatMat : (mode === 'colour' && hasCol ? colour : clay);
      m.children.forEach(ch => ch.visible = wire);
    }
    kick();
  }

  const views = { front: [0, 0, 1], back: [0, 0, -1], left: [-1, 0, 0], right: [1, 0, 0], tips: [0, 1, 0.3], top: [0.5, 0.45, 1] };
  function view(name, ms = 600) {
    const d = new THREE.Vector3(...views[name]).normalize();
    const dist = size / (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2)) * (cam.aspect < 1 ? 0.8 / cam.aspect : 0.85) * (name === 'tips' ? 0.8 : 1);
    const up = new THREE.Vector3(0, 1, 0);  // OrbitControls keeps the up it started with
    anim = { t0: performance.now(), ms, p0: cam.position.clone(), p1: d.multiplyScalar(dist), q0: ctl.target.clone(), q1: new THREE.Vector3(),
      u0: cam.up.clone(), u1: up };
    if (!ms) { cam.position.copy(anim.p1); ctl.target.set(0, 0, 0); cam.up.copy(up); anim = null; }
    kick();
  }

  // double click / double tap: orbit around that point
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  r.domElement.addEventListener('dblclick', ev => {
    const b = r.domElement.getBoundingClientRect(); ndc.set((ev.clientX - b.left) / b.width * 2 - 1, -(ev.clientY - b.top) / b.height * 2 + 1);
    ray.setFromCamera(ndc, cam); const hit = ray.intersectObjects(meshes, false)[0];
    const q1 = hit ? hit.point : new THREE.Vector3();
    anim = { t0: performance.now(), ms: 450, p0: cam.position.clone(), p1: cam.position.clone().add(q1.clone().sub(ctl.target)).lerp(q1, hit ? 0.35 : 0),
      q0: ctl.target.clone(), q1, u0: cam.up.clone(), u1: cam.up.clone() };
    kick();
  });

  const api = {
    async load(url, { heat = false } = {}) {
      if (centrelineUrl && !cl) { try { cl = await loadNpy(centrelineUrl); } catch (e) { cl = null; } }
      const g = await loader.loadAsync(url);
      g.scene.updateMatrixWorld(true);
      model.clear(); meshes.length = 0;
      g.scene.traverse(o => { if (!o.isMesh) return;
        const geo = o.geometry.clone().applyMatrix4(o.matrixWorld); geo.computeVertexNormals();
        const m = new THREE.Mesh(geo); m.userData.heat = heat;
        if (heat) m.userData.heatMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7, metalness: 0, side: THREE.DoubleSide, envMapIntensity: 0.4 });
        m.add(new THREE.Mesh(geo, wireMat)); meshes.push(m); model.add(m); });
      if (!aligned) { align(); aligned = true; view('front', 0); }
      buildRings(); applyMaterials();
      return meshes.some(m => !!m.geometry.attributes.color);
    },
    view, setMode(m) { mode = m; applyMaterials(); }, setWire(w) { wire = w; applyMaterials(); },
    setRings(on) { rings.visible = on; kick(); }, setSpin(on) { ctl.autoRotate = on; kick(); },
    snapshot() { r.render(sc, cam); return r.domElement.toDataURL('image/png'); },
    dispose() { alive = false; ro.disconnect(); ctl.dispose(); r.dispose(); el.innerHTML = ''; }
  };
  return api;
}
