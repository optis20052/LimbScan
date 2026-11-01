package com.spentys.limb_capture

import android.app.Activity
import android.app.Application
import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.view.WindowManager
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.StandardMessageCodec
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class ArViewFactory(private val registry: ArViewRegistry, private val activity: () -> Activity?) :
    PlatformViewFactory(StandardMessageCodec.INSTANCE) {
    override fun create(context: Context, viewId: Int, args: Any?): PlatformView {
        val act = activity() ?: throw IllegalStateException("LimbCapture needs an Activity")
        return ArCaptureView(act, registry)
    }
}

/**
 * Live AR camera preview + capture.
 *
 * ARCore is configured the way a photogrammetry capture needs it:
 *  - the back camera config with the LARGEST CPU image (the photo we keep),
 *  - continuous AUTO focus (the browser/WebXR path was fixed-focus = soft photos),
 *  - no plane finding / light estimation (saves CPU, not needed); depth only to hit-test the tap on the arm.
 * Each capture returns a JPEG plus the camera pose (camera->world, metres) and the exact intrinsics of the
 * stored image, so the server gets metric scale with no reference object.
 */
class ArCaptureView(private val activity: Activity, private val registry: ArViewRegistry) :
    PlatformView, GLSurfaceView.Renderer {

    private val main = Handler(Looper.getMainLooper())
    private val glView = GLSurfaceView(activity)
    private val background = BackgroundRenderer()
    private var session: Session? = null
    private var viewW = 0
    private var viewH = 0
    private var lastPoseEvent = 0L
    private var pendingCapture: Pair<String, MethodChannel.Result>? = null
    private var pendingHit: Pair<FloatArray, MethodChannel.Result>? = null
    private var pendingDepth: Pair<FloatArray, MethodChannel.Result>? = null
    private val encoder = Executors.newSingleThreadExecutor()
    private val inFlight = AtomicInteger(0)
    private var torch = false
    private var cpuImageSize = intArrayOf(0, 0)
    private val limb = LimbDetector(activity.applicationContext)
    private var lastLimbRequest = 0L
    private val viewGrid = FloatArray(LimbDetector.GRID_W * LimbDetector.GRID_H * 2).also { g ->
        for (j in 0 until LimbDetector.GRID_H) for (i in 0 until LimbDetector.GRID_W) {
            val k = 2 * (j * LimbDetector.GRID_W + i)
            g[k] = (i + 0.5f) / LimbDetector.GRID_W; g[k + 1] = (j + 0.5f) / LimbDetector.GRID_H
        }
    }

    private val lifecycle = object : Application.ActivityLifecycleCallbacks {
        override fun onActivityResumed(a: Activity) { if (a === activity) resume() }
        override fun onActivityPaused(a: Activity) { if (a === activity) pause() }
        override fun onActivityCreated(a: Activity, b: Bundle?) {}
        override fun onActivityStarted(a: Activity) {}
        override fun onActivityStopped(a: Activity) {}
        override fun onActivitySaveInstanceState(a: Activity, b: Bundle) {}
        override fun onActivityDestroyed(a: Activity) {}
    }

    init {
        glView.preserveEGLContextOnPause = true
        glView.setEGLContextClientVersion(2)
        glView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        glView.setRenderer(this)
        glView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        // never dim or lock while scanning: the user's hands are busy (one holds the phone, one is scanned)
        glView.keepScreenOn = true
        activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        activity.application.registerActivityLifecycleCallbacks(lifecycle)
        createSession()
        resume()
        registry.active = this
    }

    private fun createSession() {
        try {
            val s = Session(activity)
            // largest CPU image on the back camera = the sharpest, most detailed photo we can keep
            val filter = CameraConfigFilter(s).setFacingDirection(CameraConfig.FacingDirection.BACK)
            val configs = s.getSupportedCameraConfigs(filter)
            val best = configs.maxByOrNull { it.imageSize.width * it.imageSize.height }
            if (best != null) {
                s.cameraConfig = best
                cpuImageSize = intArrayOf(best.imageSize.width, best.imageSize.height)
            }
            val cfg = Config(s).apply {
                focusMode = Config.FocusMode.AUTO
                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                planeFindingMode = Config.PlaneFindingMode.DISABLED
                lightEstimationMode = Config.LightEstimationMode.DISABLED
                // depth (where supported) makes the tap on the arm hit its real surface even with few tracked points
                depthMode = if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) Config.DepthMode.AUTOMATIC
                            else Config.DepthMode.DISABLED
            }
            s.configure(cfg)
            session = s
            Log.i(TAG, "ARCore session: CPU image ${cpuImageSize[0]}x${cpuImageSize[1]}, focus AUTO, depth ${cfg.depthMode}")
        } catch (e: Throwable) {
            Log.e(TAG, "ARCore session failed", e)
            CrashLog.note(activity, "createSession", e)
            emit(mapOf("type" to "error", "message" to (e.message ?: e.javaClass.simpleName)))
        }
    }

    private fun resume() {
        try { session?.resume(); glView.onResume() } catch (e: Throwable) {
            Log.e(TAG, "resume failed", e)
            emit(mapOf("type" to "error", "message" to "Camera unavailable: ${e.message}"))
        }
    }

    private fun pause() { glView.onPause(); session?.pause() }

    /** Test-model mode: follow the object at this view point (0..1) by its colour; null = back to human skin. */
    fun pickColour(u: Float, v: Float) = limb.pick(u, v)
    fun clearColour() = limb.clearColour()
    fun findHands(on: Boolean) { limb.handsOn = on }

    fun setTorch(on: Boolean) {
        torch = on
        val s = session ?: return
        try {
            val cfg = s.config
            cfg.flashMode = if (on) Config.FlashMode.TORCH else Config.FlashMode.OFF
            s.configure(cfg)
        } catch (e: Exception) { Log.w(TAG, "torch not supported: ${e.message}") }
    }

    fun requestCapture(dir: String, result: MethodChannel.Result) {
        if (inFlight.get() >= 3) { result.error("busy", "encoder busy", null); return }
        synchronized(this) {
            if (pendingCapture != null) { result.error("busy", "capture pending", null); return }
            pendingCapture = dir to result
        }
    }

    /** The 3D point on the surface seen at view point (u, v) (0..1), on the next frame: [x, y, z] or null. */
    fun requestHit(u: Float, v: Float, result: MethodChannel.Result) {
        synchronized(this) {
            pendingHit?.second?.let { r -> main.post { r.success(null) } }
            pendingHit = floatArrayOf(u, v) to result
        }
    }

    /** Distance (m) from the camera along the viewing ray of each view point (u, v pairs, 0..1) from ARCore's depth
     *  image on the next frame: one value per point, 0 where there is no depth; null when depth is not available. */
    fun requestDepth(uv: FloatArray, result: MethodChannel.Result) {
        synchronized(this) {
            pendingDepth?.second?.let { r -> main.post { r.success(null) } }
            pendingDepth = uv to result
        }
    }

    private fun sampleDepth(frame: com.google.ar.core.Frame, uv: FloatArray): List<Double>? {
        val img = try { frame.acquireDepthImage16Bits() } catch (e: Exception) { return null }
        try {
            val camera = frame.camera
            val proj = FloatArray(16)
            camera.getProjectionMatrix(proj, 0, 0.03f, 20f)
            val rel = camera.pose.inverse().compose(camera.displayOrientedPose)  // display camera -> sensor camera
            // the depth image covers the CPU image's field of view: map rays with the CPU image intrinsics
            val intr = camera.imageIntrinsics
            val f = intr.focalLength; val c = intr.principalPoint; val dims = intr.imageDimensions
            val plane = img.planes[0]
            val buf = plane.buffer.order(java.nio.ByteOrder.nativeOrder())
            val dw = img.width; val dh = img.height
            val d = FloatArray(3); val r = FloatArray(3)
            val out = ArrayList<Double>(uv.size / 2)
            for (k in 0 until uv.size / 2) {
                val ndcX = 2f * uv[2 * k] - 1f; val ndcY = 1f - 2f * uv[2 * k + 1]
                d[0] = (ndcX + proj[8]) / proj[0]; d[1] = (ndcY + proj[9]) / proj[5]; d[2] = -1f
                rel.rotateVector(d, 0, r, 0)
                val z = -r[2]
                val iu = (f[0] * r[0] / z + c[0]) / dims[0]; val iv = (f[1] * -r[1] / z + c[1]) / dims[1]
                val x = (iu * dw).toInt(); val y = (iv * dh).toInt()
                if (z <= 1e-6f || x < 0 || y < 0 || x >= dw || y >= dh) { out.add(0.0); continue }
                val mm = buf.getShort(y * plane.rowStride + x * plane.pixelStride).toInt() and 0xffff
                // depth is along the optical axis: the distance along this ray is depth * |r| / z
                val n = Math.sqrt((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).toDouble())
                out.add(if (mm == 0) 0.0 else mm / 1000.0 * n / z)
            }
            return out
        } finally { img.close() }
    }

    override fun getView(): View = glView

    override fun dispose() {
        registry.active = null
        activity.window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        activity.application.unregisterActivityLifecycleCallbacks(lifecycle)
        pause()
        session?.close(); session = null
        encoder.shutdown()
        limb.close()
    }

    // ---- GL renderer ------------------------------------------------------------------------------------------
    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        try {
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            background.createOnGlThread()
            session?.setCameraTextureName(background.textureId)
        } catch (e: Throwable) { CrashLog.note(activity, "onSurfaceCreated", e) }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewW = width; viewH = height
        Log.d(TAG, "AR view surface ${width}x${height}")
        GLES20.glViewport(0, 0, width, height)
        @Suppress("DEPRECATION")
        val rotation = activity.windowManager.defaultDisplay.rotation
        session?.setDisplayGeometry(rotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        try { drawFrame() } catch (e: Throwable) {
            Log.e(TAG, "frame failed", e)
            CrashLog.note(activity, "onDrawFrame", e)
            emit(mapOf("type" to "error", "message" to "frame: ${e.javaClass.simpleName}: ${e.message}"))
        }
    }

    private fun drawFrame() {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        val frame = try { s.update() } catch (e: Exception) { return }
        background.draw(frame)
        val camera = frame.camera
        val tracking = camera.trackingState
        val now = System.currentTimeMillis()
        val vp = viewProjection(camera)
        if (now - lastPoseEvent > 30) {  // ~30 Hz: guidance + AR overlays that must follow the camera smoothly
            lastPoseEvent = now
            val dp = camera.displayOrientedPose  // axes aligned with the screen: x right, y up, -z forward
            val ev = HashMap<String, Any>()
            ev["type"] = "pose"
            ev["tracking"] = when (tracking) {
                TrackingState.TRACKING -> "tracking"; TrackingState.PAUSED -> "paused"; else -> "stopped"
            }
            if (tracking != TrackingState.TRACKING) ev["reason"] = camera.trackingFailureReason.name
            ev["pos"] = listOf(dp.tx().toDouble(), dp.ty().toDouble(), dp.tz().toDouble())
            val z = dp.zAxis; val x = dp.xAxis
            ev["fwd"] = listOf(-z[0].toDouble(), -z[1].toDouble(), -z[2].toDouble())
            ev["right"] = listOf(x[0].toDouble(), x[1].toDouble(), x[2].toDouble())
            ev["t"] = now
            ev["vp"] = vp.map { it.toDouble() }
            emit(ev)
        }
        if (limb.available && limb.idle && now - lastLimbRequest > 60) {  // as fast as the model runs (~10-15 Hz)
            lastLimbRequest = now
            requestLimb(frame, vp)
        }
        synchronized(this) { val h = pendingHit; pendingHit = null; h }?.let { (uv, result) ->
            val hit = if (tracking != TrackingState.TRACKING) null else try {
                // the nearest hit on the depth map only: sparse feature points near the ray can sit on the wall behind
                frame.hitTest(uv[0] * viewW, uv[1] * viewH).firstOrNull { it.trackable is com.google.ar.core.DepthPoint }
            } catch (e: Exception) { null }
            val p = hit?.hitPose
            Log.d(TAG, "hit test ${uv.joinToString()} -> ${hit?.trackable?.javaClass?.simpleName} d=${hit?.distance}")
            main.post { result.success(p?.let { listOf(it.tx().toDouble(), it.ty().toDouble(), it.tz().toDouble()) }) }
        }
        synchronized(this) { val d = pendingDepth; pendingDepth = null; d }?.let { (uv, result) ->
            val out = if (tracking != TrackingState.TRACKING) null else sampleDepth(frame, uv)
            val ok = out?.filter { it > 0 }?.sorted()
            Log.d(TAG, "depth ${uv.size / 2} points -> ${ok?.size ?: "n/a"} valid, median ${ok?.getOrNull(ok.size / 2)}")
            main.post { result.success(out) }
        }
        val req = synchronized(this) { val r = pendingCapture; pendingCapture = null; r } ?: return
        val (dir, result) = req
        if (tracking != TrackingState.TRACKING) { main.post { result.error("not_tracking", "AR tracking not ready", null) }; return }
        val image = try { frame.acquireCameraImage() } catch (e: NotYetAvailableException) {
            main.post { result.error("not_ready", "camera image not ready", null) }; return
        } catch (e: Exception) { main.post { result.error("image", e.message, null) }; return }
        // everything we need from this frame, copied before the image is released
        val pose = FloatArray(16).also { camera.pose.toMatrix(it, 0) }  // camera->world, column-major, metres
        val intr = camera.imageIntrinsics
        val f = intr.focalLength; val c = intr.principalPoint; val dims = intr.imageDimensions
        val seg = limb.colourInfo()
        val raw = FrameEncoder.copyYuv(image)
        image.close()
        inFlight.incrementAndGet()
        encoder.execute {
            try {
                val out = FrameEncoder.encodePortrait(raw, dir, now)
                // CPU image is landscape (sensor); stored JPEG is rotated 90 deg clockwise to portrait:
                // (x, y) -> (H - 1 - y, x)  =>  fx' = fy, fy' = fx, cx' = H - cy, cy' = cx
                val k = listOf(f[1].toDouble(), f[0].toDouble(), (dims[1] - c[1]).toDouble(), c[0].toDouble())
                val map = hashMapOf<String, Any>(
                    "path" to out.path, "width" to out.width, "height" to out.height,
                    "pose" to pose.map { it.toDouble() }, "K" to k, "sharpness" to out.sharpness,
                    "t" to now, "src" to "arcore", "cpuImage" to listOf(dims[0], dims[1])
                )
                if (seg != null) map["seg"] = seg
                main.post { result.success(map) }
            } catch (e: Throwable) {
                main.post { result.error("encode", e.message, null) }
            } finally { inFlight.decrementAndGet() }
        }
    }

    /** Small portrait copy of the camera image + where each on-screen grid cell lands in it, then segment. */
    private fun viewProjection(camera: com.google.ar.core.Camera): FloatArray {
        val proj = FloatArray(16); val view = FloatArray(16); val vp = FloatArray(16)
        camera.getProjectionMatrix(proj, 0, 0.03f, 20f)
        camera.getViewMatrix(view, 0)
        Matrix.multiplyMM(vp, 0, proj, 0, view, 0)
        return vp
    }

    private var limbRequests = 0

    /** Where each on-screen grid point lands in the CPU camera image (normalized, landscape), from the camera models
     *  themselves: screen point -> viewing ray (preview projection) -> physical camera frame -> CPU image intrinsics.
     *  ARCore's transformCoordinates2d assumes the CPU image covers the same field of view as the preview, which is not
     *  true for the large 4:3 image we analyse (the outline came out stretched and shifted). */
    private fun viewToImage(camera: com.google.ar.core.Camera): FloatArray {
        val proj = FloatArray(16)
        camera.getProjectionMatrix(proj, 0, 0.03f, 20f)
        val rel = camera.pose.inverse().compose(camera.displayOrientedPose)  // display camera -> sensor camera
        val intr = camera.imageIntrinsics
        val f = intr.focalLength; val c = intr.principalPoint; val dims = intr.imageDimensions
        val out = FloatArray(viewGrid.size)
        val d = FloatArray(3); val r = FloatArray(3)
        for (k in 0 until viewGrid.size / 2) {
            val ndcX = 2f * viewGrid[2 * k] - 1f; val ndcY = 1f - 2f * viewGrid[2 * k + 1]
            d[0] = (ndcX + proj[8]) / proj[0]; d[1] = (ndcY + proj[9]) / proj[5]; d[2] = -1f
            rel.rotateVector(d, 0, r, 0)
            val z = -r[2]
            if (z <= 1e-6f) { out[2 * k] = -1f; out[2 * k + 1] = -1f; continue }
            out[2 * k] = (f[0] * r[0] / z + c[0]) / dims[0]
            out[2 * k + 1] = (f[1] * -r[1] / z + c[1]) / dims[1]
        }
        return out
    }

    private fun requestLimb(frame: com.google.ar.core.Frame, vp: FloatArray) {
        val img = try { frame.acquireCameraImage() } catch (e: Exception) { return }
        val big = limb.handMode
        val sw = if (big) LimbDetector.HAND_W else LimbDetector.SMALL_W; val sh = if (big) LimbDetector.HAND_H else LimbDetector.SMALL_H
        val rgb = try { LimbDetector.sampleRgb(img, sw, sh) } finally { img.close() }
        val mapped = viewToImage(frame.camera)
        if (++limbRequests % 50 == 1) {  // diagnostics: ARCore's own view -> CPU image mapping vs ours
            val ar = FloatArray(viewGrid.size)
            frame.transformCoordinates2d(Coordinates2d.VIEW_NORMALIZED, viewGrid, Coordinates2d.IMAGE_NORMALIZED, ar)
            val c = 2 * ((LimbDetector.GRID_H / 2) * LimbDetector.GRID_W + LimbDetector.GRID_W / 2); val e = 0
            Log.d(TAG, "view->image centre ours ${"%.3f,%.3f".format(mapped[c], mapped[c + 1])} arcore ${"%.3f,%.3f".format(ar[c], ar[c + 1])}" +
                " | corner ours ${"%.3f,%.3f".format(mapped[e], mapped[e + 1])} arcore ${"%.3f,%.3f".format(ar[e], ar[e + 1])}")
        }
        val n = viewGrid.size / 2
        val gx = FloatArray(n); val gy = FloatArray(n)
        for (k in 0 until n) {  // landscape image (u, v) -> portrait sample (1 - v, u)
            gx[k] = 1f - mapped[2 * k + 1]; gy[k] = mapped[2 * k]
        }
        val points = try {
            frame.acquirePointCloud().use { pc ->
                val b = pc.points; val out = FloatArray(b.remaining() / 4 * 3)
                var o = 0
                while (b.remaining() >= 4) { out[o++] = b.get(); out[o++] = b.get(); out[o++] = b.get(); b.get() }
                out
            }
        } catch (e: Exception) { FloatArray(0) }
        val cp = frame.camera.displayOrientedPose
        limb.submit(LimbDetector.Request(rgb, sw, sh, gx, gy, vp, points,
            floatArrayOf(cp.tx(), cp.ty(), cp.tz()))) { emit(it) }
    }

    private fun emit(ev: Map<String, Any>) { main.post { registry.sink?.success(ev) } }

    companion object { const val TAG = "LimbCapture" }
}
