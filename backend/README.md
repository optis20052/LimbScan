# LimbScan: metric 3D limb scanning from a phone

Mobile web app (iPhone / Android / iPad, any browser) + reconstruction server.
You capture a guided photo sequence (or upload a video/photos from the native camera app). The server
returns a millimetre-accurate, watertight 3D mesh of the limb, clinical girth measurements, and an
automatic accuracy comparison against your reference scans (Structure Sensor, other apps).

## Scale: phone AR tracking first (no card)
On **Android Chrome** the app runs a **WebXR / ARCore** session: every photo is uploaded with the phone's metric
camera pose and lens data. The server fits the arm-only SfM camera path to the AR path (robust similarity), which
gives mm per unit with no reference object (`limbscan/arscale.py`).
Validated on MyFit's iPhone 13 Pro frames with ARKit poses: scale consistent to ±0.33%; against the Structure scan,
forearm volume −0.09% and girth +1.7 mm.
Fallbacks when no AR data is available: printed sheets, then an ID-1 card (taped on the back of the hand / held flat).

## How it works (arm mode, default)
1. **Segmentation**: MediaPipe body-skin segmentation finds the limb in every photo (about 0.15 s each).
2. **Arm-only camera tracking**: SIFT features only on the limb, plus the calibration card, so the camera poses are
   relative to the limb itself. The patient may sway relative to the room (validated at ±4° / ±12 mm).
3. **Card calibration**: for the first seconds the card is held flat against the arm; its 4 corners are
   triangulated and give the metric scale. Its 3D outline is projected into every photo and masked out of stereo.
4. **Dense stereo on the limb only** (background painted grey), followed by Poisson meshing and tape-measure girths.

Speed on this laptop (CPU): about 1 min for 90 synthetic photos up to scale, about 3 min end-to-end; about 5 min for 133 real phone photos.

## Why this design

| Problem | Solution |
|---|---|
| Photogrammetry has **no scale** (the #1 error source for "phone scanning") | **Default: any ISO ID-1 card** (bank / ID / health card, 85.60 × 53.98 mm) lying flat near the limb. No printing. Card outlines are found in every photo with sub-pixel edge fitting, verified in 3D across views, and the 4 corners are triangulated from all views (synthetic test: 0.05% scale error). The card plane also defines the table. **Optional: printed ArUco sheets** (similarity fit to the printed layout, 0.03%), used automatically when present. |
| Phone depth sensors are coarse (LiDAR: about 5–10 mm) and not available to web apps | Dense multi-view stereo on full-resolution images (COLMAP SfM + OpenMVS PatchMatch + photometric mesh refinement), which gives sub-mm surface detail. |
| Lens distortion and focal length vary by phone | Self-calibration in bundle adjustment (OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2). |
| Background, table, mat | Cleanup: drop everything near the marker plane, crop around the cameras' focus point, keep the largest connected cluster. |
| Clinicians need numbers, not only a mesh | Centreline estimation; tape-measure girths (convex-hull perimeter) every 5 mm; wrist, max forearm and hand girth; length; volume. Exports STL/OBJ/PLY plus JSON. |
| "Is it accurate?" | Built-in comparison to reference scans: rigid ICP on the forearm, signed surface deviation heatmap, girth differences along identical planes, volume difference. |

## Run

```bash
./run.sh                       # https://<your-lan-ip>:8443  (phone on same Wi-Fi, accept the cert warning)
```
On the phone: **Guide** (read once), print **marker sheets** at 100%, then **Guided live capture** or
**Upload video / photos**. Processing takes about 15–30 min on this laptop (CPU only; an NVIDIA GPU server takes about 2–4 min).

Reference scans: every `.obj/.stl/.ply` in the parent folder (`LIMBSCAN_REFS`) is compared automatically.

CLI:
```bash
.venv/bin/python -m limbscan.pipeline <job_dir>          # job_dir/input/* = photos/videos
.venv/bin/python -m limbscan.compare scan.ply ref.obj out/
.venv/bin/python tools/synth.py "../Structre scanner.obj" /tmp/synth   # synthetic, metrically known capture
```

## Capture protocol (summary)
1. Lay a standard bank/ID/health card flat on the table, 5–30 cm from the limb, fully visible.
2. The forearm may rest on a pillow (contact areas are interpolated and flagged). Keep it still for about 2 min.
3. Diffuse light. Circle the limb at 3 heights (level, above, as low as possible), 30–50 cm away, about 100 photos.
4. Best quality: native camera 4K video or photos, then Upload.

## Layout
```
limbscan/segment.py      MediaPipe limb masks (SfM + stereo), card regions
limbscan/arscale.py      metric scale from phone AR camera poses (WebXR/ARKit), default when present
limbscan/card.py         ID-1 card detection, 3D verification / single-view PnP, metric scale (fallback)
limbscan/markers.py      printable ArUco sheets (vector PDF) + layout (optional)
limbscan/scale.py        marker triangulation, sheet similarity fit, metric/table frame
limbscan/pipeline.py     frames -> SfM -> scale -> OpenMVS (docker) -> postprocess
limbscan/postprocess.py  limb isolation, cropped refined surface, Poisson watertight mesh, exports
limbscan/measure.py      centreline, girths, landmarks, volume
limbscan/compare.py      accuracy vs reference scans (+ heatmap GLB)
server/app.py            FastAPI + HTTPS + job queue
server/static/index.html mobile UI: guided capture (sharpness/motion gating, coverage ring), upload, 3D viewer, report
tools/synth.py           photoreal-ish synthetic capture renderer for end-to-end metric validation
docker/                  OpenMVS build
```
