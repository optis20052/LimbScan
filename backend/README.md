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

Speed (laptop: Intel Core Ultra X7 358H + Arc B390 iGPU, no NVIDIA): **about 30 s from "Finish" to the result**
for 84 phone photos with AR tracking (was 167 s). What makes it fast:

| Stage | How | Time |
|---|---|---|
| Photo prep, limb masks, limb SIFT | done **while the phone is still uploading** (`limbscan/prefetch.py`, 4 server threads); masks on all cores; SIFT on limb crops only, without domain-size pooling | ~1 s after Finish |
| Matching | only the ~12 nearest photos per photo, chosen from the AR camera path (7x fewer pairs); exact uint8 dot products (`limbscan/matching.py`, repeatable) | 3.5 s |
| Camera poses | global SfM (rotation averaging + global positioning + BA, all seeded, multi-core), 6000 tracks, no retriangulation | 4.5 s |
| Dense stereo | **our OpenCL PatchMatch on the GPU** (`limbscan/gpumvs.py`): limb crops at the stereo resolution, coarse-to-fine, deterministic | 5 s |
| Post-processing | per-point surface fits compiled with numba (multi-core), fast MST normal orientation | 13 s |

Every stage is repeatable: the same photos give the same numbers. Without a GPU, OpenMVS is used automatically
(`LIMBSCAN_MVS_ENGINE=openmvs` forces it). Accuracy: see Validation below.

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
**Upload video / photos**. Processing takes about 30 s after the last photo on this laptop.

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

## Validation (fast pipeline, 2026-10-29)

| Test | Surface error (median) | Girth error (mean abs) | Volume |
|---|---|---|---|
| Synthetic capture of the Structure mesh, true shape known (`tools/synth.py`, exact AR poses) | 0.066 mm | 0.31 mm | +0.00 % |
| MyFit's 84 ARKit photos vs the Structure scan, GPU stereo | 0.33 mm | 1.96 mm | -0.33 % |
| same photos, OpenMVS stereo (`LIMBSCAN_MVS_ENGINE=openmvs`) | 0.34 mm | 1.77 mm | -0.15 % |
| MyFit's own scan vs Structure (for reference) | 0.66 mm | 2.7 mm | -1.38 % |

On the true-shape test the GPU stereo is more accurate than OpenMVS (surface 0.066 vs 0.082 mm, volume +0.0 vs
+0.2 %). On real skin the GPU cloud lies ~0.3 mm deeper than OpenMVS's; the wrist reads ~3 mm smaller in girth
against Structure (-3.7 vs -0.6 mm). Which one is right on real skin is open (Structure has its own error): check
with tape measurements / more reference scans. Repeat runs on the same photos give identical measurements.

## Layout
```
limbscan/segment.py      MediaPipe limb masks (SfM + stereo), card regions
limbscan/arscale.py      metric scale from phone AR camera poses (WebXR/ARKit), default when present
limbscan/card.py         ID-1 card detection, 3D verification / single-view PnP, metric scale (fallback)
limbscan/markers.py      printable ArUco sheets (vector PDF) + layout (optional)
limbscan/scale.py        marker triangulation, sheet similarity fit, metric/table frame
limbscan/pipeline.py     frames -> SfM -> scale -> OpenMVS (docker) -> postprocess
limbscan/prefetch.py     per-photo prep / masks / limb SIFT while the phone is still uploading (server threads)
limbscan/matching.py     exact, repeatable SIFT matching of AR-chosen photo pairs
limbscan/gpumvs.py       OpenCL PatchMatch multi-view stereo + fusion (GPU), OpenMVS is the fallback
limbscan/fastgeo.py      numba kernels: robust skin planes, quadric smoothing, normal orientation
limbscan/postprocess.py  limb isolation, cropped refined surface, Poisson watertight mesh, exports
limbscan/measure.py      centreline, girths, landmarks, volume
limbscan/compare.py      accuracy vs reference scans (+ heatmap GLB)
server/app.py            FastAPI + HTTPS + job queue
server/static/index.html mobile UI: guided capture (sharpness/motion gating, coverage ring), upload, 3D viewer, report
tools/synth.py           photoreal-ish synthetic capture renderer for end-to-end metric validation
docker/                  OpenMVS build
```
