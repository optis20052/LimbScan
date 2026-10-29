# My3D Scanner — Capture & Calibration: the actual code

Companion to [`architecture.md`](./architecture.md). This file contains the **real implemented logic**
for camera setup, calibration, frame triggering, pixel extraction and JPEG encoding — quoted from the
decompiled classes, plus a few clearly-marked reconstructions where a Kotlin coroutine decompiles into
unreadable state machines.

> **Provenance.** This is **not** the vendor's original Kotlin source. It is the result of
> `jadx 1.5.3` on `fr.myned.my3dScanner` **26.8.11** (`com.myfit.sdk`, R8-shrunk, debug info stripped).
> Names like `C01521`, `AnonymousClass2`, `fArr2`, `i21`, `L$0` are decompiler artefacts, not vendor code.
> The vendor's Kotlin sources were **not** published; everything here is reconstructed from bytecode.
> Behaviour is faithful; formatting is not.

**Where the full tree lives in this repo**

| What | Path |
|---|---|
| All decompiled Java (12 340 files) | `apk/jadx_out/sources/` |
| SDK package | `apk/jadx_out/sources/com/myfit/sdk/` |
| App package (Flutter host + activities) | `apk/jadx_out/sources/fr/myned/my3dScanner/` |
| Dart AOT dump (Blutter) | `apk/dart_out/asm/` |
| Resources / manifest | `apk/apktool_out/` |

**Regenerate**

```bash
jadx -d apk/jadx_out --show-bad-code apk/x/fr.myned.my3dScanner.apk
# or browse: jadx-gui apk/x/fr.myned.my3dScanner.apk
```

**Files that matter (this document follows call order)**

| # | Class | Role |
|---|---|---|
| 1 | `utils/arcore/ArCore.java` | ARCore session + camera configuration |
| 2 | `utils/arcore/FrameCallback.java` | per-vsync update loop |
| 3 | `fragments/ARFragment.java` | wires ARCore + Filament + listeners |
| 4 | `controllers/CalibrationDetector.java` | the calibration algorithm |
| 5 | `controllers/CalibrationFrameMonitor.java` | feeds frames into the detector |
| 6 | `controllers/MotionBasedFrameCapture.java` | decides *when* to take a picture |
| 7 | `controllers/FrameDataExtractor.java` | pulls image / point cloud / matrices out of ARCore |
| 8 | `services/application/processors/RawImageProcessor.java` | YUV → crop → JPEG → rotate → JPEG |
| 9 | `services/application/processors/CenteredCrop.java` | crop rectangle maths |
| 10 | `services/application/processors/FeaturePointProcessor.java` | 3D → 2D pixel projection |
| 11 | `services/application/processors/CameraMatrixProcessor.java` | pose matrix flattening |
| 12 | `services/application/FrameProcessService.java` | worker that persists image + metadata |
| 13 | `services/application/repository/FSLocalFrameRepository.java` | file naming on disk |
| 14 | `controllers/ScanController.java` | step machine that drives all of the above |

---

## 1. Session & camera configuration

`apk/jadx_out/sources/com/myfit/sdk/utils/arcore/ArCore.java`

Everything that defines *which* sensor stream is used is in this constructor — note
`LATEST_CAMERA_IMAGE`, `AUTO` focus, and **`DepthMode.DISABLED`** (no depth sensor is involved).


```java
    public ArCore(Activity activity, Filament filament, View view) {
        Intrinsics.checkNotNullParameter(activity, "activity");
        Intrinsics.checkNotNullParameter(filament, "filament");
        Intrinsics.checkNotNullParameter(view, "view");
        this.activity = activity;
        this.filament = filament;
        this.view = view;
        int iCreateExternalTextureId = OpenGLKt.createExternalTextureId();
        this.cameraStreamTextureId = iCreateExternalTextureId;
        Session session = new Session(activity);
        session.setCameraConfig(findAppropriateCameraConfig(session));
        Config config = session.getConfig();
        config.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE);
        config.setAugmentedImageDatabase(null);
        config.setFocusMode(Config.FocusMode.AUTO);
        config.setPlaneFindingMode(Config.PlaneFindingMode.HORIZONTAL);
        config.setLightEstimationMode(Config.LightEstimationMode.ENVIRONMENTAL_HDR);
        config.setDepthMode(Config.DepthMode.DISABLED);
        config.setInstantPlacementMode(Config.InstantPlacementMode.DISABLED);
        config.setCloudAnchorMode(Config.CloudAnchorMode.DISABLED);
        session.configure(config);
        session.setCameraTextureName(iCreateExternalTextureId);
        this.session = session;
        this.cameraId = session.getCameraConfig().getCameraId();
        Object systemService = ContextCompat.getSystemService(activity, CameraManager.class);
        Intrinsics.checkNotNull(systemService);
        this.cameraManager = (CameraManager) systemService;
    }
```


Camera selection: keep camera id `"0"`, take the **widest** CPU image — except on one Xiaomi model where
the **narrowest** is taken instead (crash/hang workaround).


```java
    private final CameraConfig findAppropriateCameraConfig(Session currentSession) {
        List<CameraConfig> supportedCameraConfigs = currentSession.getSupportedCameraConfigs(new CameraConfigFilter(currentSession));
        Intrinsics.checkNotNullExpressionValue(supportedCameraConfigs, "getSupportedCameraConfigs(...)");
        ArrayList arrayList = new ArrayList();
        for (Object obj : supportedCameraConfigs) {
            if (Intrinsics.areEqual(((CameraConfig) obj).getCameraId(), "0")) {
                arrayList.add(obj);
            }
        }
        ArrayList arrayList2 = arrayList;
        Iterator it = arrayList2.iterator();
        if (!it.hasNext()) {
            throw new NoSuchElementException();
        }
        Object next = it.next();
        if (it.hasNext()) {
            int width = ((CameraConfig) next).getImageSize().getWidth();
            do {
                Object next2 = it.next();
                int width2 = ((CameraConfig) next2).getImageSize().getWidth();
                if (width < width2) {
                    next = next2;
                    width = width2;
                }
            } while (it.hasNext());
        }
        if (DeviceModel.INSTANCE.getLIMITED_IMAGE_SIZE_DEVICES().contains(Build.MODEL)) {
            Iterator it2 = arrayList2.iterator();
            if (!it2.hasNext()) {
                throw new NoSuchElementException();
            }
            next = it2.next();
            if (it2.hasNext()) {
                int width3 = ((CameraConfig) next).getImageSize().getWidth();
                do {
                    Object next3 = it2.next();
                    int width4 = ((CameraConfig) next3).getImageSize().getWidth();
                    if (width3 > width4) {
                        next = next3;
                        width3 = width4;
                    }
                } while (it2.hasNext());
            }
        }
        ArrayList arrayList3 = new ArrayList();
        for (Object obj2 : arrayList2) {
            CameraConfig cameraConfig = (CameraConfig) next;
            if (Intrinsics.areEqual(((CameraConfig) obj2).getImageSize(), cameraConfig != null ? cameraConfig.getImageSize() : null)) {
                arrayList3.add(obj2);
            }
        }
        Iterator it3 = arrayList3.iterator();
        if (!it3.hasNext()) {
            throw new NoSuchElementException();
        }
        Object next4 = it3.next();
        if (it3.hasNext()) {
            int width5 = ((CameraConfig) next4).getTextureSize().getWidth();
            do {
                Object next5 = it3.next();
                int width6 = ((CameraConfig) next5).getTextureSize().getWidth();
                if (width5 < width6) {
                    next4 = next5;
                    width5 = width6;
                }
            } while (it3.hasNext());
        }
        Intrinsics.checkNotNullExpressionValue(next4, "maxBy(...)");
        return (CameraConfig) next4;
    }
```


### The frame loop

`Choreographer` (vsync) → `session.update()` → listeners → Filament render. `ScanController.onFrameUpdate`
is one of those listeners, so **calibration and capture both run inside this callback**.


```java
package com.myfit.sdk.utils.arcore;

public final class FrameCallback implements Choreographer.FrameCallback {
    private final ArCore arCore;
    private final Choreographer choreographer;
    private long lastRenderedTimestamp;
    private final Function1<Frame, Unit> onFrameUpdate;

    /* JADX WARN: Multi-variable type inference failed */
    public FrameCallback(ArCore arCore, Function1<? super Frame, Unit> onFrameUpdate) {
        Intrinsics.checkNotNullParameter(arCore, "arCore");
        Intrinsics.checkNotNullParameter(onFrameUpdate, "onFrameUpdate");
        this.arCore = arCore;
        this.onFrameUpdate = onFrameUpdate;
        this.choreographer = Choreographer.getInstance();
    }

    @Override // android.view.Choreographer.FrameCallback
    public void doFrame(long frameTimeNanos) throws Exception {
        this.choreographer.postFrameCallback(this);
        Frame frameUpdate = this.arCore.getSession().update();
        Intrinsics.checkNotNullExpressionValue(frameUpdate, "update(...)");
        if (frameUpdate.getTimestamp() == 0 || frameUpdate.getTimestamp() == this.lastRenderedTimestamp) {
            return;
        }
        this.arCore.update(frameUpdate);
        this.onFrameUpdate.invoke(frameUpdate);
        if (this.arCore.getFilament().getUiHelper().isReadyToRender()) {
            Renderer renderer = this.arCore.getFilament().getRenderer();
            SwapChain swapChain = this.arCore.getFilament().getSwapChain();
            Intrinsics.checkNotNull(swapChain);
            if (renderer.beginFrame(swapChain, frameTimeNanos)) {
                this.arCore.getFilament().getRenderer().render(this.arCore.getFilament().getView());
                this.arCore.getFilament().getRenderer().endFrame();
                this.lastRenderedTimestamp = frameUpdate.getTimestamp();
            }
        }
    }

    public final void start() {
        this.choreographer.postFrameCallback(this);
    }

    public final void stop() {
        this.choreographer.removeFrameCallback(this);
    }
}
```

### Wiring (`ARFragment`)

`initArCore()` refuses to start when ARCore is missing, `createUx()` builds `Filament` + `ArCore` +
`FrameCallback`, `startUx()` resumes the session and starts the Choreographer loop.


```java
    public final boolean initArCore() {
        ScanStepList scanStepList;
        if (!ARCoreInstallHelper.INSTANCE.isAvailable(getContext())) {
            ScanController scanController = this.scanController;
            if (scanController == null) {
                return false;
            }
            scanController.goToScanStep(ScanStep.ARCORE_VIEW);
            return false;
        }
        ScanController scanController2 = this.scanController;
        if (scanController2 != null && (scanStepList = scanController2.getScanStepList()) != null) {
            scanStepList.goToStep(ScanStep.INITIALIZE_VIEW);
        }
        BuildersKt__Builders_commonKt.launch$default(this.createScope, null, null, new C01521(null), 3, null);
        return true;
    }
```

```java
    public final Object startUx(Continuation<? super Unit> continuation) throws Throwable {
        C01541 c01541;
        if (continuation instanceof C01541) {
            c01541 = (C01541) continuation;
            if ((c01541.label & Integer.MIN_VALUE) != 0) {
                c01541.label -= Integer.MIN_VALUE;
            } else {
                c01541 = new C01541(continuation);
            }
        }
        Object obj = c01541.result;
        Object coroutine_suspended = IntrinsicsKt.getCOROUTINE_SUSPENDED();
        int i = c01541.label;
        ArCore arCore = null;
        try {
            if (i == 0) {
                ResultKt.throwOnFailure(obj);
                ArCore arCore2 = this.arCore;
                if (arCore2 == null) {
                    Intrinsics.throwUninitializedPropertyAccessException("arCore");
                    arCore2 = null;
                }
                arCore2.getSession().resume();
                FrameCallback frameCallback = this.frameCallback;
                if (frameCallback == null) {
                    Intrinsics.throwUninitializedPropertyAccessException("frameCallback");
                    frameCallback = null;
                }
                frameCallback.start();
                c01541.label = 1;
                if (DelayKt.awaitCancellation(c01541) == coroutine_suspended) {
                    return coroutine_suspended;
                }
            } else {
                if (i != 1) {
                    throw new IllegalStateException("call to 'resume' before 'invoke' with coroutine");
                }
                ResultKt.throwOnFailure(obj);
            }
            throw new KotlinNothingValueException();
        } catch (Throwable th) {
            FrameCallback frameCallback2 = this.frameCallback;
            if (frameCallback2 == null) {
                Intrinsics.throwUninitializedPropertyAccessException("frameCallback");
                frameCallback2 = null;
            }
            frameCallback2.stop();
            ArCore arCore3 = this.arCore;
            if (arCore3 == null) {
                Intrinsics.throwUninitializedPropertyAccessException("arCore");
            } else {
                arCore = arCore3;
            }
            arCore.getSession().pause();
            throw th;
        }
    }
```


---
## 2. Calibration — the actual algorithm

**This is not lens/intrinsic calibration.** It is a motion-convergence gate on ARCore's pose, used to
guarantee that tracking is live before the first photo. Constants are hard-coded defaults; the sampling
source is `camera.displayOrientedPose` + `trackingState`.


```java
package com.myfit.sdk.controllers;

public final class CalibrationDetector {
    public static final long DEFAULT_CHECK_INTERVAL_NANOS = 100000000;
    public static final int DEFAULT_REQUIRED_VALID_CHECKS = 20;
    public static final long DEFAULT_SESSION_GAP_THRESHOLD_NANOS = 1000000000;
    public static final float DEFAULT_SPEED_THRESHOLD_METERS = 0.015f;
    public static final int DEFAULT_SPEED_WINDOW_SIZE = 10;
    public static final long DEFAULT_TIMEOUT_NANOS = 10000000000L;
    private long activeElapsedNanos;
    private final long checkIntervalNanos;
    private Long lastCheckTimestampNanos;
    private float lastRollingMeanMeters;
    private float[] lastTranslation;
    private boolean previousCheckValid;
    private final int requiredValidChecks;
    private final long sessionGapThresholdNanos;
    private final float speedThresholdMeters;
    private final ArrayDeque<Float> speedWindow;
    private final int speedWindowSize;
    private State state;
    private final long timeoutNanos;
    private int validCheckCount;

    public CalibrationDetector() {
        this(0L, 0.0f, 0, 0, 0L, 0L, 63, null);
    }

    public CalibrationDetector(long j, float f, int i, int i2, long j2, long j3) {
        this.checkIntervalNanos = j;
        this.speedThresholdMeters = f;
        this.speedWindowSize = i;
        this.requiredValidChecks = i2;
        this.timeoutNanos = j2;
        this.sessionGapThresholdNanos = j3;
        if (i <= 0) {
            throw new IllegalArgumentException("speedWindowSize must be > 0".toString());
        }
        this.state = State.IN_PROGRESS;
        this.speedWindow = new ArrayDeque<>();
    }

    public /* synthetic */ CalibrationDetector(long j, float f, int i, int i2, long j2, long j3, int i3, DefaultConstructorMarker defaultConstructorMarker) {
        this((i3 & 1) != 0 ? DEFAULT_CHECK_INTERVAL_NANOS : j, (i3 & 2) != 0 ? 0.015f : f, (i3 & 4) != 0 ? 10 : i, (i3 & 8) != 0 ? 20 : i2, (i3 & 16) != 0 ? 10000000000L : j2, (i3 & 32) != 0 ? 1000000000L : j3);
    }

    /* JADX WARN: Failed to restore enum class, 'enum' modifier and super class removed */
    /* JADX WARN: Unknown enum class pattern. Please report as an issue! */
    public static final class State {
        private static final /* synthetic */ EnumEntries $ENTRIES;
        private static final /* synthetic */ State[] $VALUES;
        public static final State IN_PROGRESS = new State("IN_PROGRESS", 0);
        public static final State FINISHED_STANDARD = new State("FINISHED_STANDARD", 1);
        public static final State FINISHED_FAILOVER = new State("FINISHED_FAILOVER", 2);

        private static final /* synthetic */ State[] $values() {
            return new State[]{IN_PROGRESS, FINISHED_STANDARD, FINISHED_FAILOVER};
        }

        public static EnumEntries<State> getEntries() {
            return $ENTRIES;
        }

        public static State valueOf(String str) {
            return (State) Enum.valueOf(State.class, str);
        }

        public static State[] values() {
            return (State[]) $VALUES.clone();
        }

        static {
            State[] stateArr$values = $values();
            $VALUES = stateArr$values;
            $ENTRIES = EnumEntriesKt.enumEntries(stateArr$values);
        }

        private State(String str, int i) {
        }
    }

    public static final class Sample {
        private final boolean isTracking;
        private final long timestampNanos;
        private final float[] translation;

        public Sample(long j, float[] translation, boolean z) {
            Intrinsics.checkNotNullParameter(translation, "translation");
            this.timestampNanos = j;
            this.translation = translation;
            this.isTracking = z;
        }

        public final long getTimestampNanos() {
            return this.timestampNanos;
        }

        public final float[] getTranslation() {
            return this.translation;
        }

        /* renamed from: isTracking, reason: from getter */
        public final boolean getIsTracking() {
            return this.isTracking;
        }
    }

    public final State getState() {
        return this.state;
    }

    public final float getLastRollingMeanMeters() {
        return this.lastRollingMeanMeters;
    }

    public final int getValidCheckCount() {
        return this.validCheckCount;
    }

    public final boolean shouldCheck(long timestampNanos) {
        Long l = this.lastCheckTimestampNanos;
        return l == null || timestampNanos - l.longValue() >= this.checkIntervalNanos;
    }

    public final State addSample(Sample sample) {
        Intrinsics.checkNotNullParameter(sample, "sample");
        if (this.state == State.IN_PROGRESS && shouldCheck(sample.getTimestampNanos())) {
            Long l = this.lastCheckTimestampNanos;
            Long lValueOf = l != null ? Long.valueOf(sample.getTimestampNanos() - l.longValue()) : null;
            boolean z = false;
            boolean z2 = lValueOf != null && lValueOf.longValue() >= this.sessionGapThresholdNanos;
            if (lValueOf != null && !z2) {
                this.activeElapsedNanos += lValueOf.longValue();
            }
            this.lastCheckTimestampNanos = Long.valueOf(sample.getTimestampNanos());
            float[] fArr = this.lastTranslation;
            float[] translation = sample.getTranslation();
            float[] fArrCopyOf = Arrays.copyOf(translation, translation.length);
            Intrinsics.checkNotNullExpressionValue(fArrCopyOf, "copyOf(...)");
            this.lastTranslation = fArrCopyOf;
            if (fArr == null || z2) {
                this.previousCheckValid = false;
                return this.state;
            }
            float f = sample.getTranslation()[0] - fArr[0];
            float f2 = sample.getTranslation()[1] - fArr[1];
            float f3 = sample.getTranslation()[2] - fArr[2];
            float fSqrt = (float) Math.sqrt((f * f) + (f2 * f2) + (f3 * f3));
            if (this.speedWindow.size() >= this.speedWindowSize) {
                this.speedWindow.removeFirst();
            }
            this.speedWindow.addLast(Float.valueOf(fSqrt));
            float fAverageOfFloat = (float) CollectionsKt.averageOfFloat(this.speedWindow);
            this.lastRollingMeanMeters = fAverageOfFloat;
            if (sample.getIsTracking() && fAverageOfFloat >= this.speedThresholdMeters) {
                z = true;
            }
            if (z && this.previousCheckValid) {
                int i = this.validCheckCount + 1;
                this.validCheckCount = i;
                if (i >= this.requiredValidChecks) {
                    State state = State.FINISHED_STANDARD;
                    this.state = state;
                    return state;
                }
            }
            this.previousCheckValid = z;
            if (this.activeElapsedNanos >= this.timeoutNanos) {
                this.state = State.FINISHED_FAILOVER;
            }
            return this.state;
        }
        return this.state;
    }
}
```

### Who feeds it


```java
package com.myfit.sdk.controllers;

public final class CalibrationFrameMonitor {
    private static final String TAG = "CalibrationFrameMonitor";
    private int checkCount;
    private final CalibrationDetector detector;
    private boolean finished;
    private final Function1<CalibrationDetector.State, Unit> onCalibrationFinished;

    /* JADX WARN: Multi-variable type inference failed */
    public CalibrationFrameMonitor(CalibrationDetector detector, Function1<? super CalibrationDetector.State, Unit> onCalibrationFinished) {
        Intrinsics.checkNotNullParameter(detector, "detector");
        Intrinsics.checkNotNullParameter(onCalibrationFinished, "onCalibrationFinished");
        this.detector = detector;
        this.onCalibrationFinished = onCalibrationFinished;
    }

    public /* synthetic */ CalibrationFrameMonitor(CalibrationDetector calibrationDetector, Function1 function1, int i, DefaultConstructorMarker defaultConstructorMarker) {
        this((i & 1) != 0 ? new CalibrationDetector(0L, 0.0f, 0, 0, 0L, 0L, 63, null) : calibrationDetector, function1);
    }

    public final void onFrame(Frame frame) {
        Intrinsics.checkNotNullParameter(frame, "frame");
        if (this.finished || !this.detector.shouldCheck(frame.getTimestamp())) {
            return;
        }
        long timestamp = frame.getTimestamp();
        float[] translation = frame.getCamera().getDisplayOrientedPose().getTranslation();
        Intrinsics.checkNotNullExpressionValue(translation, "getTranslation(...)");
        CalibrationDetector.Sample sample = new CalibrationDetector.Sample(timestamp, translation, frame.getCamera().getTrackingState() == TrackingState.TRACKING);
        CalibrationDetector.State stateAddSample = this.detector.addSample(sample);
        int i = this.checkCount + 1;
        this.checkCount = i;
        Log.d(TAG, "check #" + i + " tracking=" + sample.getIsTracking() + " mean=" + this.detector.getLastRollingMeanMeters() + "m validPairs=" + this.detector.getValidCheckCount() + "/20 state=" + stateAddSample);
        if (stateAddSample != CalibrationDetector.State.IN_PROGRESS) {
            Log.i(TAG, "Calibration detection finished after " + this.checkCount + " checks, outcome=" + stateAddSample);
            this.finished = true;
            this.onCalibrationFinished.invoke(stateAddSample);
        }
    }
}
```

### Who arms it, and what happens on each outcome

`ScanController` — note the fail-open branch (`FINISHED_FAILOVER` still calls `nextStepAction()`), and that
`FrameProcessService` is created here, i.e. **before** any capture, pointing at the scan-group directory.


```java
    public final void startCalibrationStep() throws JsonProcessingException {
        String id;
        String title;
        String str;
        ScanAPIModel scanApiModel = ApiModelManager.INSTANCE.getApiModel().getScanApiModel();
        if (scanApiModel == null) {
            throw new IllegalStateException("Scan API Model is null".toString());
        }
        LogService.INSTANCE.logSDKOptions();
        String str2 = "";
        LogService.INSTANCE.logScan(new LogAPIModel("Sscan", ""));
        DiskStorageHelper.INSTANCE.setCurrentScanDirectory(scanApiModel.getScanGroupId());
        UploadBackgroundViewModel.INSTANCE.setScanId(scanApiModel.getId());
        UploadBackgroundViewModel.Companion companion = UploadBackgroundViewModel.INSTANCE;
        ScanGroupAPIModel scanGroupAPIModel = ApiModelManager.INSTANCE.getApiModel().getScanGroupAPIModel();
        if (scanGroupAPIModel == null || (id = scanGroupAPIModel.getId()) == null) {
            id = "";
        }
        companion.setScanGroupId(id);
        UploadBackgroundViewModel.Companion companion2 = UploadBackgroundViewModel.INSTANCE;
        ScanGroupAPIModel scanGroupAPIModel2 = ApiModelManager.INSTANCE.getApiModel().getScanGroupAPIModel();
        if (scanGroupAPIModel2 == null || (title = scanGroupAPIModel2.getTitle()) == null) {
            title = "";
        }
        companion2.setScanTitle(title);
        UploadBackgroundViewModel.Companion companion3 = UploadBackgroundViewModel.INSTANCE;
        LocalizedText name = ApiModelManager.INSTANCE.getApiModel().getScanExperience().getName();
        if (name != null && (str = name.get()) != null) {
            str2 = str;
        }
        companion3.setScanExperience(str2);
        this.scanModel.setCurrentState(ScanState.SCAN_IN_PROGRESS);
        this.frameProcessService = new FrameProcessService(scanApiModel.getCategory(), new FSLocalFrameRepository(DiskStorageHelper.INSTANCE.getMainPath()), new RawImageProcessor(ImageRotation.CW90.INSTANCE), new FrameMetadataProcessor(new FeaturePointProcessor(ImageRotation.CW90.INSTANCE), new CameraMatrixProcessor()), new ObjectMapper(), RawFrameDumper.INSTANCE.getNOOP());
        String str3 = LOG_TAG;
        Log.d(str3, "Scan Started with id : " + scanApiModel.getId() + " / " + scanApiModel.getTitle());
        LogService.INSTANCE.logScan(new LogAPIModel(CALIBRATION_LOG_TAG, "Start"));
        this.calibrationFrameMonitor = new CalibrationFrameMonitor(null, new AnonymousClass2(this), 1, 0 == true ? 1 : 0);
        Log.d(str3, "Calibration monitor armed");
    }
```

```java
    public final void onCalibrationFinished(CalibrationDetector.State state) throws JsonProcessingException {
        stopCalibrationStep();
        int i = WhenMappings.$EnumSwitchMapping$1[state.ordinal()];
        if (i == 1) {
            Log.i(CALIBRATION_LOG_TAG, "Calibration finished by motion detection");
            LogService.INSTANCE.logScan(new LogAPIModel(CALIBRATION_LOG_TAG, "Finished"));
        } else if (i != 2) {
            if (i != 3) {
                throw new NoWhenBranchMatchedException();
            }
            return;
        } else {
            Log.w(CALIBRATION_LOG_TAG, "Calibration failover after 10s - motion detection never converged");
            LogService.INSTANCE.logScan(new LogAPIModel(CALIBRATION_LOG_TAG, "Finished (Failover)"));
        }
        nextStepAction();
    }
```

```java
    private final void stopCalibrationStep() {
        this.calibrationFrameMonitor = null;
        Log.d(LOG_TAG, "Calibration monitor stopped");
    }
```


---
## 3. When a picture is taken

No timer, no shutter button: a frame is captured when the pose delta exceeds the tolerance that the
**server** configured for the category (`freeScanTolerance`, default 0.1 m / 0.1 rad).


```java
package com.myfit.sdk.controllers;

public final class MotionBasedFrameCapture {
    private final FrameDataExtractor frameDataExtractor;
    private Pose lastRotationPose;
    private Pose lastTranslationPose;

    public MotionBasedFrameCapture(FrameDataExtractor frameDataExtractor) {
        Intrinsics.checkNotNullParameter(frameDataExtractor, "frameDataExtractor");
        this.frameDataExtractor = frameDataExtractor;
    }

    private final boolean detectTranslation(Pose lastPose, Pose currentPose) {
        float translationTolerance = ApiModelManager.INSTANCE.getApiModel().getCategoryAPIModel().getFreeScanTolerance().getTranslationTolerance();
        for (int i = 0; i < 3; i++) {
            if (Math.abs(lastPose.getTranslation()[i] - currentPose.getTranslation()[i]) > translationTolerance) {
                return true;
            }
        }
        return false;
    }

    private final boolean detectRotation(Pose lastPose, Pose currentPose) {
        float rotationTolerance = ApiModelManager.INSTANCE.getApiModel().getCategoryAPIModel().getFreeScanTolerance().getRotationTolerance();
        float[] rotationQuaternion = lastPose.getRotationQuaternion();
        Intrinsics.checkNotNullExpressionValue(rotationQuaternion, "getRotationQuaternion(...)");
        Quaternion quaternionInverted = new Quaternion(rotationQuaternion).inverted();
        float[] rotationQuaternion2 = currentPose.getRotationQuaternion();
        Intrinsics.checkNotNullExpressionValue(rotationQuaternion2, "getRotationQuaternion(...)");
        return ((float) 2) * ((float) Math.acos((double) Quaternion.INSTANCE.multiply(quaternionInverted, new Quaternion(rotationQuaternion2)).getW())) > rotationTolerance;
    }

    public final Object captureFrameIfNeeded(String str, int i, Frame frame, Continuation<? super RawFrame> continuation) {
        Pose displayOrientedPose = frame.getCamera().getDisplayOrientedPose();
        if (this.lastTranslationPose == null) {
            Intrinsics.checkNotNull(displayOrientedPose);
            this.lastTranslationPose = displayOrientedPose;
        }
        if (this.lastRotationPose == null) {
            Intrinsics.checkNotNull(displayOrientedPose);
            this.lastRotationPose = displayOrientedPose;
        }
        Pose pose = this.lastTranslationPose;
        if (pose == null) {
            Intrinsics.throwUninitializedPropertyAccessException("lastTranslationPose");
            pose = null;
        }
        Intrinsics.checkNotNull(displayOrientedPose);
        boolean zDetectTranslation = detectTranslation(pose, displayOrientedPose);
        if (zDetectTranslation) {
            this.lastTranslationPose = displayOrientedPose;
        }
        Pose pose2 = this.lastRotationPose;
        if (pose2 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("lastRotationPose");
            pose2 = null;
        }
        boolean zDetectRotation = detectRotation(pose2, displayOrientedPose);
        if (zDetectRotation) {
            this.lastRotationPose = displayOrientedPose;
        }
        if (zDetectTranslation || zDetectRotation) {
            return captureFrame(str, i, frame, continuation);
        }
        return null;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final Object captureFrame(String str, int i, Frame frame, Continuation<? super RawFrame> continuation) {
        return this.frameDataExtractor.captureFrame(str, i, frame, continuation);
    }
}
```

### The dispatcher

`ScanController.onFrameUpdate` runs on every ARCore frame (≈30 Hz). Note it passes the **scan-group id**
(not the scan id) as the frame's `scanId`, and that the whole capture runs in a coroutine so the render
thread is not blocked.


```java
    public void onFrameUpdate(Frame frame) {
        Intrinsics.checkNotNullParameter(frame, "frame");
        CalibrationFrameMonitor calibrationFrameMonitor = this.calibrationFrameMonitor;
        if (calibrationFrameMonitor != null) {
            calibrationFrameMonitor.onFrame(frame);
        }
        ScanGroupAPIModel scanGroupAPIModel = ApiModelManager.INSTANCE.getApiModel().getScanGroupAPIModel();
        String id = scanGroupAPIModel != null ? scanGroupAPIModel.getId() : null;
        if (id != null) {
            BuildersKt__Builders_commonKt.launch$default(this.coroutineScope, null, null, new AnonymousClass1(id, frame, null), 3, null);
        } else {
            Log.d(LOG_TAG, "Scan ID is null, skipping frame capture");
        }
    }
```

The decompiled body of the lambda it launches (state machine collapsed back to Kotlin):

```kotlin
// reconstructed from ScanController$onFrameUpdate$1.invokeSuspend(...)
val capture = scanModel.motionBasedFrameCapture ?: return
val rawFrame = capture.captureFrameIfNeeded(scanGroupId, sideIndex(), frame)
if (rawFrame != null) scanController.onFrameCaptured(rawFrame)
```


```java
    public final int sideIndex() {
        return this.scanControllerViewModel.getCptFrameCaptured() + 1;
    }
```

```java
    public final void onFrameCaptured(RawFrame capturedFrame) {
        Intrinsics.checkNotNullParameter(capturedFrame, "capturedFrame");
        FrameProcessService frameProcessService = this.frameProcessService;
        if ((frameProcessService != null ? frameProcessService.submit(capturedFrame) : null) == SaveSubmissionResult.QUEUED) {
            ScanControllerViewModel scanControllerViewModel = this.scanControllerViewModel;
            scanControllerViewModel.setCptFrameCaptured(scanControllerViewModel.getCptFrameCaptured() + 1);
        }
    }
```

```java
    public final void startFreeScanStep() {
        this.scanModel.setMotionBasedFrameCapture(new MotionBasedFrameCapture(new FrameDataExtractor()));
    }
```

```java
    public final void stopFreeScanStep() throws JsonProcessingException {
        this.scanModel.setMotionBasedFrameCapture(null);
        onScanComplete();
    }
```

The frame counter (`cptFrameCaptured`) is the only thing that distinguishes frames: it becomes
`sideIndex` → `frameSide` → the JSON's `sideId` and the metadata file name.

---
## 4. Pulling the data out of ARCore

`FrameDataExtractor` performs the three acquisitions. The point cloud is mandatory: if it throws, the
frame is dropped (nothing to anchor the reconstruction to).


```java
    public final Object captureFrame(String str, int i, Frame frame, Continuation<? super RawFrame> continuation) throws Throwable {
        C01501 c01501;
        if (continuation instanceof C01501) {
            c01501 = (C01501) continuation;
            if ((c01501.label & Integer.MIN_VALUE) != 0) {
                c01501.label -= Integer.MIN_VALUE;
            } else {
                c01501 = new C01501(continuation);
            }
        }
        Object objAcquireRawImage = c01501.result;
        Object coroutine_suspended = IntrinsicsKt.getCOROUTINE_SUSPENDED();
        int i2 = c01501.label;
        if (i2 == 0) {
            ResultKt.throwOnFailure(objAcquireRawImage);
            c01501.L$0 = str;
            c01501.L$1 = frame;
            c01501.I$0 = i;
            c01501.label = 1;
            objAcquireRawImage = acquireRawImage(frame, c01501);
            if (objAcquireRawImage == coroutine_suspended) {
                return coroutine_suspended;
            }
        } else {
            if (i2 != 1) {
                throw new IllegalStateException("call to 'resume' before 'invoke' with coroutine");
            }
            i = c01501.I$0;
            frame = (Frame) c01501.L$1;
            str = (String) c01501.L$0;
            ResultKt.throwOnFailure(objAcquireRawImage);
        }
        String str2 = str;
        int i3 = i;
        RawImage rawImage = (RawImage) objAcquireRawImage;
        if (rawImage == null) {
            return null;
        }
        RawPointCloud rawPointCloudAcquireRawCloudPoint = acquireRawCloudPoint(frame);
        if (rawPointCloudAcquireRawCloudPoint == null) {
            Log.w(TAG, "Failed to acquire raw point cloud for frame at timestamp " + frame.getTimestamp());
            return null;
        }
        return new RawFrame(str2, i3, frame.getTimestamp(), rawImage, rawPointCloudAcquireRawCloudPoint, acquireRawCamera(frame));
    }
```

```java
    private final RawPointCloud acquireRawCloudPoint(Frame frame) {
        try {
            PointCloud pointCloudAcquirePointCloud = frame.acquirePointCloud();
            try {
                PointCloud pointCloud = pointCloudAcquirePointCloud;
                FloatBuffer points = pointCloud.getPoints();
                float[] fArr = new float[points.remaining()];
                points.rewind();
                points.get(fArr);
                IntBuffer ids = pointCloud.getIds();
                int[] iArr = new int[ids.remaining()];
                ids.rewind();
                ids.get(iArr);
                RawPointCloud rawPointCloud = new RawPointCloud(fArr, iArr);
                AutoCloseableKt.closeFinally(pointCloudAcquirePointCloud, null);
                return rawPointCloud;
            } finally {
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to acquire point cloud: " + e);
            return null;
        }
    }
```

```java
    private final RawCamera acquireRawCamera(Frame frame) {
        float[] fArr = new float[16];
        frame.getCamera().getViewMatrix(fArr, 0);
        float[] fArr2 = new float[16];
        frame.getCamera().getProjectionMatrix(fArr2, 0, 0.001f, 100.0f);
        float[] fArr3 = new float[16];
        frame.getCamera().getPose().toMatrix(fArr3, 0);
        return new RawCamera(fArr, fArr2, fArr3);
    }
```

`acquireRawImage` is a retry wrapper around `frame.acquireCameraImage()`: up to 4 attempts with linear
backoff on `NotYetAvailableException` (the camera has not produced a buffer for that timestamp yet), then
each `Image.Plane` (buffer + `rowStride` + `pixelStride`) is copied verbatim into a `RawImage`.

---
## 5. From ARCore frame to the uploaded JPEG


```java
package com.myfit.sdk.services.application.processors;

public final class RawImageProcessor {
    private final ImageRotation rotation;

    /* JADX WARN: Multi-variable type inference failed */
    public RawImageProcessor() {
        this(null, 1, 0 == true ? 1 : 0);
    }

    public RawImageProcessor(ImageRotation rotation) {
        Intrinsics.checkNotNullParameter(rotation, "rotation");
        this.rotation = rotation;
    }

    public /* synthetic */ RawImageProcessor(ImageRotation.CW90 cw90, int i, DefaultConstructorMarker defaultConstructorMarker) {
        this((i & 1) != 0 ? ImageRotation.CW90.INSTANCE : cw90);
    }

    public final void process(RawImage rawImage, CropSize cropSize, File output) {
        Intrinsics.checkNotNullParameter(rawImage, "rawImage");
        Intrinsics.checkNotNullParameter(cropSize, "cropSize");
        Intrinsics.checkNotNullParameter(output, "output");
        YuvImage yUVImage = ImageUtil.INSTANCE.toYUVImage(rawImage);
        CropRect cropRectForRotatedOutput = CenteredCrop.INSTANCE.forRotatedOutput(rawImage.getWidth(), rawImage.getHeight(), cropSize);
        Rect rect = new Rect(cropRectForRotatedOutput.getLeft(), cropRectForRotatedOutput.getTop(), cropRectForRotatedOutput.getLeft() + cropRectForRotatedOutput.getWidth(), cropRectForRotatedOutput.getTop() + cropRectForRotatedOutput.getHeight());
        FileOutputStream fileOutputStream = new FileOutputStream(output);
        try {
            yUVImage.compressToJpeg(rect, 100, fileOutputStream);
            CloseableKt.closeFinally(fileOutputStream, null);
            Bitmap bitmapDecodeFile = BitmapFactory.decodeFile(output.getPath());
            if (bitmapDecodeFile == null) {
                return;
            }
            Bitmap bitmapRotateBitmap = this.rotation.rotateBitmap(bitmapDecodeFile);
            bitmapDecodeFile.recycle();
            fileOutputStream = new FileOutputStream(output);
            try {
                bitmapRotateBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream);
                CloseableKt.closeFinally(fileOutputStream, null);
            } finally {
            }
        } finally {
            try {
                throw th;
            } finally {
            }
        }
    }
}
```

The crop rectangle is *centred* and expressed in the **rotated (portrait) orientation**, which is why
`forRotatedOutput` swaps width/height when clamping:


```java
package com.myfit.sdk.services.application.processors;

public final class CenteredCrop {
    public static final CenteredCrop INSTANCE = new CenteredCrop();

    private CenteredCrop() {
    }

    public final CropRect forRotatedOutput(int imageWidth, int imageHeight, CropSize cropSize) {
        Intrinsics.checkNotNullParameter(cropSize, "cropSize");
        if (cropSize.getWidth() <= 0 || cropSize.getHeight() <= 0) {
            return new CropRect(0, 0, imageWidth, imageHeight);
        }
        int iCoerceAtMost = RangesKt.coerceAtMost(cropSize.getHeight(), imageWidth);
        int iCoerceAtMost2 = RangesKt.coerceAtMost(cropSize.getWidth(), imageHeight);
        return new CropRect(RangesKt.coerceAtLeast((imageWidth - iCoerceAtMost) / 2, 0), RangesKt.coerceAtLeast((imageHeight - iCoerceAtMost2) / 2, 0), iCoerceAtMost, iCoerceAtMost2);
    }
}
```

`CropRect` itself is a plain data holder (generated `equals`/`hashCode`/`copy`/`toString` stripped here):


```java
public final /* data */ class CropRect {
    private final int left;
    private final int top;
    private final int width;
    private final int height;

    public CropRect(int left, int top, int width, int height) {   // compacted from the JADX ctor
        this.left = left;
        this.top = top;
        this.width = width;
        this.height = height;
    }
}
```

`ImageUtil.toYUVImage` is the bridge from ARCore's three-plane YUV to the `YuvImage` API that Android
can JPEG-encode. The full conversion loop is ~150 lines of stride handling; signature and effect:

```java
// utils/ImageUtil.java  (loop body omitted)
public final YuvImage toYUVImage(RawImage rawImage) {
    return new YuvImage(
        convertYUV420toNV21(rawImage, createConversionBuffer(rawImage)),
        ImageFormat.NV21, rawImage.getWidth(), rawImage.getHeight(), null);
}
```

---
## 6. Pose + 3D→2D projection (what the server receives as numbers)

Only `camera.getPose()` is exported. `projection × view` is used **locally** to compute `x2d/y2d`, so
intrinsics never leave the phone.


```java
    public final Matrix buildViewProjectionMatrix(RawFrame frame) {
        Matrix matrix = new Matrix(frame.getRawCamera().getProjectionMatrix());
        Matrix matrix2 = new Matrix(frame.getRawCamera().getViewMatrix());
        Matrix matrix3 = new Matrix();
        Matrix.INSTANCE.multiply(matrix, matrix2, matrix3);
        return matrix3;
    }
```

```java
    public final Point2D projectToWebPixel(Vector3 world, Matrix mvp, Size imageSize) {
        float[] fArrPerspectiveDivide;
        float[] fArrMultiplyMVP = multiplyMVP(world, mvp);
        if (fArrMultiplyMVP == null || (fArrPerspectiveDivide = perspectiveDivide(fArrMultiplyMVP)) == null) {
            return null;
        }
        float[] fArrNdcToNormalisedUV = ndcToNormalisedUV(fArrPerspectiveDivide);
        if (isInsideFrustum(fArrNdcToNormalisedUV)) {
            return normalisedUVToWebPixel(this.rotation.rotateUV(fArrNdcToNormalisedUV), imageSize);
        }
        return null;
    }
```

```java
    private final float[] perspectiveDivide(float[] clipPoint) {
        float f = clipPoint[0];
        float f2 = clipPoint[1];
        float f3 = clipPoint[2];
        if (f3 <= 0.0f) {
            return null;
        }
        return new float[]{f / f3, f2 / f3};
    }
```

```java
    private final float[] ndcToNormalisedUV(float[] ndc) {
        return new float[]{(ndc[0] + 1.0f) / 2.0f, (1.0f - ndc[1]) / 2.0f};
    }
```

```java
    private final Point2D normalisedUVToWebPixel(float[] webUV, Size imageSize) {
        return new Point2D(webUV[0] * imageSize.getHeight(), webUV[1] * imageSize.getWidth());
    }
```

```java
    public final Point2D applyCrop(Point2D pixel, CropRect cropRect) {
        float x = pixel.getX() - cropRect.getTop();
        float y = pixel.getY() - cropRect.getLeft();
        if (x < 0.0f || x >= cropRect.getHeight() || y < 0.0f || y >= cropRect.getWidth()) {
            return null;
        }
        return new Point2D(x, y);
    }
```

> The rotation is applied **in UV space** (`this.rotation.rotateUV`, `CW90` → `(u,v) ⇒ (v, 1−u)`), not by
> swapping axes at the end. Composite result for one point:
>
> ```
> clip    = (projection × view) · world        ; drop if w ≤ 0
> u,v     = ((x+1)/2, (1−y)/2)                 ; NDC → image UV
> u',v'   = (v, 1−u)                           ; ImageRotation.CW90.rotateUV
> px,py   = (u'·H_src, v'·W_src)               ; pixels in the rotated frame
> x2d,y2d = (px − crop.top, py − crop.left)    ; crop lives in source/landscape coords
> ```
>
> Kept only if `0 ≤ x2d < crop.height` and `0 ≤ y2d < crop.width` — which after the 90° turn equals the
> uploaded JPEG's width/height.



The `cameraMatrix` that is uploaded (note: it transposes the **pose**, despite the parameter name):


```java
    public final List<List<Float>> arrayToMatrix(float[] rawProjectionMatrix) {
        ArrayList arrayList = new ArrayList(4);
        for (int i = 0; i < 4; i++) {
            ArrayList arrayList2 = new ArrayList(4);
            for (int i2 = 0; i2 < 4; i2++) {
                int i3 = (i2 * 4) + i;
                arrayList2.add(Float.valueOf(i3 < rawProjectionMatrix.length ? rawProjectionMatrix[i3] : 0.0f));
            }
            arrayList.add(arrayList2);
        }
        return arrayList;
    }
```

> Naming trap: the parameter is called `rawProjectionMatrix`, but the caller passes
> `frame.rawCamera.cameraMatrix` — i.e. the **pose**, not the projection. The function is a
> column-major → row-major transpose, nothing more.


---
## 7. Persisting image + metadata

`FrameProcessService.saveFrameData` decompiles into a four-state coroutine; reconstructed to Kotlin
(behaviour identical to the bytecode branch structure):

```kotlin
// reconstructed from FrameProcessService$saveFrameData$2.invokeSuspend(...)
private suspend fun saveFrameData(rawFrame: RawFrame): Int = withContext(Dispatchers.IO) {
    val imageFile    = frameRepository.createImageFile(rawFrame)      // …/<scanGroupId>/frames/frame_<ts>.jpg
    rawImageProcessor.process(rawFrame.image, scanCategory.imageCrop, imageFile)

    val metadataFile = frameRepository.createMetadataFile(rawFrame)   // …/<scanGroupId>/frames/frame_<side>.json
    val metadata     = frameMetadataProcessor.process(rawFrame, imageFile, scanCategory.imageCrop)
    objectMapper.writeValue(metadataFile, metadata)                    // Jackson → 5-field JSON
    ...
}
```

The worker consumes a `Channel<RawFrame>` of capacity **10**; when full, frames are dropped
(`DROPPED_QUEUE_FULL`) instead of stalling the camera.


The public methods only delegate to per-frame coroutine workers; the actual naming lives in
`FSLocalFrameRepository$createImageFile$2` / `$createMetadataFile$2`. Reconstructed:

```kotlin
// reconstructed from FSLocalFrameRepository$createImageFile$2 / $createMetadataFile$2
private suspend fun createImageFile(frame: RawFrame): File = withContext(Dispatchers.IO) {
    val dir = File(getScanDirectory(frame.scanId), "frames").apply { mkdirs() }
    File(dir, "frame_${frame.timestamp}.jpg").apply { createNewFile() }      // overwritten if it exists
}

private suspend fun createMetadataFile(frame: RawFrame): File = withContext(Dispatchers.IO) {
    val dir = File(getScanDirectory(frame.scanId), "frames").apply { mkdirs() }
    File(dir, "frame_${frame.sideIndex}.json").apply { createNewFile() }     // overwritten if it exists
}
```


Note the asymmetry: the **image** is named by ARCore's `timestamp` (ns since boot), the **metadata** by
the frame counter. `FrameMetadata.image` stores only the JPEG file name, which is how the uploader pairs them.

---
## 8. Call chain (one frame)

```
Choreographer.doFrame()
└─ FrameCallback.doFrame()                         utils/arcore/FrameCallback.java
   ├─ session.update()            → Frame
   ├─ onFrameUpdate(frame)        → ARFragment.onFrameUpdate → listeners
   │   └─ ScanController.onFrameUpdate()
   │       ├─ CalibrationFrameMonitor.onFrame()    (only while CALIBRATION_VIEW)
   │       │   └─ CalibrationDetector.addSample()
   │       └─ MotionBasedFrameCapture.captureFrameIfNeeded()
   │           └─ FrameDataExtractor.captureFrame()
   │               ├─ acquireRawImage()   (acquireCameraImage, 4× retry)
   │               ├─ acquireRawCloudPoint()
   │               └─ acquireRawCamera()  (view / projection / pose)
   │           └─ (returns RawFrame) → ScanController.onFrameCaptured()
   │               └─ FrameProcessService.submit()   Channel(10), non-blocking
   │                   └─ saveFrameData()
   │                       ├─ RawImageProcessor.process()  YUV→crop→JPEG→rotate→JPEG(q=100)
   │                       ├─ FrameMetadataProcessor.process()  featurePoints + cameraMatrix
   │                       └─ objectMapper.writeValue(...)  frame_<n>.json
   └─ renderer.beginFrame()/render()/endFrame()   (Filament preview)
```

Timing (defaults): calibration sampled every **100 ms**, 20 consecutive valid samples, failover at **10 s**;
capture gate **0.1 m / 0.1 rad**; JPEG **q=100** twice; `setUpdateMode(LATEST_CAMERA_IMAGE)` means frames the
phone is too slow to consume are simply skipped.

---
## 9. Decompiler caveats — read before trusting a line

| Symptom in this dump | Meaning |
|---|---|
| `C01521`, `AnonymousClass2`, `$1$1` | Kotlin lambda / coroutine classes; the `suspend` code is split into a state machine (`label`, `L$0`, `I$1`) |
| `Intrinsics.checkNotNullParameter(...)`, `Intrinsics.checkNotNullExpressionValue(...)` | Kotlin's null-contract calls, not logic |
| `BuildersKt__Builders_commonKt.launch$default(scope, …, 3, null)` | `scope.launch { … }` |
| `ResultKt.throwOnFailure(obj)`, `IntrinsicsKt.getCOROUTINE_SUSPENDED()` | coroutine plumbing |
| `fArr`, `fArr2`, `i21`, `str7` | local variables with erased names |
| `/* compiled from: X.kt */` | original Kotlin file name (kept even though the source is gone) |
| `@Metadata(d1=…, d2=…)` blocks | Kotlin reflection metadata — can be decoded back to signatures, stripped here |
| `// JADX WARN: …` | the decompiler could not perfectly restore that region |

Anything in this document that is *reconstructed* rather than quoted is marked
`// reconstructed from …` — those three blocks are the only non-verbatim code.

---
## 10. Reproducing / verifying

```bash
# 1. get the APK (as done for architecture.md)
apkeep -a fr.myned.my3dScanner -d apk-pure .
# 2. decompile
jadx -d out --show-bad-code fr.myned.my3dScanner.apk
# 3. jump straight to the interesting classes
${EDITOR} out/sources/com/myfit/sdk/controllers/CalibrationDetector.java
${EDITOR} out/sources/com/myfit/sdk/controllers/MotionBasedFrameCapture.java
${EDITOR} out/sources/com/myfit/sdk/services/application/processors/RawImageProcessor.java
```

Dynamic confirmation (needs a real account): proxy the device with mitmproxy, complete a scan, and check
that the `PUT` body is a JPEG whose dimensions equal the category's `imageCrop`, and that each
`POST /scan/{id}/frame` body carries `featurePoints[].x2d/y2d` inside those dimensions.
