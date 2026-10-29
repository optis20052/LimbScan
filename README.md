# LimbScan

Medical-grade 3D limb scanning with a phone: guided capture in a native app, reconstruction on a GPU server.

```
LimbScan/
├── app/                      Flutter app (Android first, iOS next): one Dart codebase
│   ├── lib/
│   │   ├── main.dart         app entry
│   │   ├── config.dart       server address, pose, voice (persisted)
│   │   ├── api/              REST client for the LimbScan server
│   │   ├── capture/          guidance (arm centre, coverage, duplicates, passes) + background upload queue
│   │   ├── screens/          home, setup, capture (AR + HUD), result (server page), settings
│   │   └── widgets/          coverage ring
│   ├── packages/limb_capture/  our native AR capture plugin
│   │   ├── android/          Kotlin + ARCore 1.57: autofocus, largest CPU image, pose + intrinsics per photo
│   │   └── ios/              Swift + ARKit (next milestone)
│   └── test/                 unit tests (guidance maths)
├── backend/                  reconstruction server (Python glue around CUDA COLMAP, OpenMVS, Open3D, MediaPipe)
│   ├── limbscan/             pipeline: segmentation, SfM, AR/card scale, stereo, meshing, measurements
│   ├── server/               FastAPI + web app
│   ├── tools/                synthetic test renderer, helpers
│   └── pc.sh                 manage the GPU PC server (url | status | restart | deploy | logs)
├── docs/                     notes; docs/research = analysis of the MyFit app
└── test-data/                reference scans (Structure Sensor, MyFit) + MyFit iPhone frames with ARKit poses
```

## Build the Android app
```bash
export PATH=~/development/flutter/bin:$PATH JAVA_HOME=~/Android/jdk ANDROID_HOME=~/Android/sdk
cd app && flutter build apk --release --target-platform android-arm64
```
Install on the phone: open `http://192.168.50.223:8080/app.apk` (same Wi-Fi), or `adb install`.

## Server
Runs on the GPU PC (`ssh limbpc`) as systemd user services. From `backend/`: `./pc.sh status`, `./pc.sh deploy`.
