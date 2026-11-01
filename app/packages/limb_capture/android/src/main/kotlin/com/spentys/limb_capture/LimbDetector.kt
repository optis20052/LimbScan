package com.spentys.limb_capture

import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenterOptions
import com.google.mediapipe.tasks.vision.interactivesegmenter.Stroke
import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Cuts out the object the user tapped (arm, hand or a printed model) with MediaPipe's interactive segmenter
 * (MagicTouch), on a small portrait copy of the camera image. Only runs while the arm is being placed: the app
 * fits its 3D cylinder to this outline once, then scanning needs no live segmentation.
 *
 * Results are in VIEW coordinates (0..1 across the on-screen preview): outline grid, box, and ARCore's tracked
 * 3D points on the object (distance, a surface point).
 */
class LimbDetector(private val context: Context) {
    /** points: ARCore feature points of the frame (world x, y, z per point), cam: camera position (world). */
    class Request(val rgb: IntArray, val w: Int, val h: Int, val gridX: FloatArray, val gridY: FloatArray, val vp: FloatArray,
                  val points: FloatArray = FloatArray(0), val cam: FloatArray = FloatArray(3))

    private val worker = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var runs = 0
    val available get() = colourMode || (handsOn && hands != null)
    /** Look for hands (MediaPipe hand landmarker) on every frame while the arm is not placed yet. */
    @Volatile var handsOn = false
    val handMode get() = handsOn && !colourMode
    @Volatile private var hands: HandLandmarker? = null
    val idle get() = !busy.get()

    // ---- follow the tapped object with MediaPipe's interactive segmenter (MagicTouch) ----
    @Volatile private var touch: InteractiveSegmenter? = null
    @Volatile private var pickAt: FloatArray? = null      // view point to start from on the next frame
    @Volatile private var seed: FloatArray? = null        // a point inside the object (portrait image, 0..1): followed
    @Volatile private var modelOn = false
    @Volatile private var anchor: FloatArray? = null      // the object's position in the world (AR points on it)
    private var areaRef: Double? = null                    // the object's usual size in pixels (smoothed)
    private var spills = 0
    val colourMode get() = modelOn

    // a person's arm and hand: MediaPipe's selfie-multiclass model (class 2 = body skin; face, hair, clothes and the
    // room are other classes), whatever is behind the arm. The tapped-object cut-out is for anything else (printed model)
    @Volatile private var skin: ImageSegmenter? = null
    @Volatile private var skinMode: Boolean? = null       // decided on the first frame after a tap

    init {  // model loads off the UI thread, before the tap
        worker.execute {
            if (hands == null) hands = createHands()
            if (touch == null) touch = createTouch()
            if (skin == null) skin = createSkin()
        }
    }

    private fun createHands(): HandLandmarker? {
        for (delegate in listOf(Delegate.GPU, Delegate.CPU)) {
            try {
                val h = HandLandmarker.createFromOptions(context, HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").setDelegate(delegate).build())
                    .setRunningMode(RunningMode.IMAGE).setNumHands(2)
                    .setMinHandDetectionConfidence(0.4f).setMinHandPresenceConfidence(0.4f).build())
                h.detect(BitmapImageBuilder(Bitmap.createBitmap(HAND_W, HAND_H, Bitmap.Config.ARGB_8888)).build())
                Log.i(ArCaptureView.TAG, "hand landmarker on $delegate")
                return h
            } catch (e: Throwable) {
                Log.w(ArCaptureView.TAG, "hand landmarker $delegate unavailable: ${e.message}")
            }
        }
        return null
    }

    /** Hands in the frame: 21 landmarks each in VIEW coordinates (0..1), their metric 3D shape (metres, camera
     *  axes: x right, y down, z away; origin at the hand's centre) and the detection score. */
    private fun findHands(r: Request): Map<String, Any>? {
        val det = hands ?: return null
        val bmp = Bitmap.createBitmap(r.rgb, r.w, r.h, Bitmap.Config.ARGB_8888)
        val res = try { det.detect(BitmapImageBuilder(bmp).build()) } finally { bmp.recycle() }
        val toView = imageToView(r) ?: return null
        val list = ArrayList<Map<String, Any>>()
        for (k in res.landmarks().indices) {
            val lm = res.landmarks()[k]
            val wl = res.worldLandmarks().getOrNull(k) ?: continue
            val uv = ArrayList<Double>(42)
            for (p in lm) {
                uv.add((toView[0] * p.x() + toView[1] * p.y() + toView[2]).toDouble())
                uv.add((toView[3] * p.x() + toView[4] * p.y() + toView[5]).toDouble())
            }
            val xyz = ArrayList<Double>(63)
            for (p in wl) { xyz.add(p.x().toDouble()); xyz.add(p.y().toDouble()); xyz.add(p.z().toDouble()) }
            val score = res.handednesses().getOrNull(k)?.firstOrNull()?.score()?.toDouble() ?: 0.0
            list.add(mapOf("lm" to uv, "world" to xyz, "score" to score))
        }
        return hashMapOf("type" to "hand", "hands" to list, "vp" to r.vp.map { it.toDouble() })
    }

    /** Portrait image point (0..1) -> view point (0..1): the affine map fitted to the frame's view -> image grid
     *  (the image is the camera's view cropped and scaled, so the map is affine). [a, b, c, d, e, f]. */
    private fun imageToView(r: Request): DoubleArray? {
        // least squares view = M [ix, iy, 1] over the grid points
        val ata = Array(3) { DoubleArray(3) }; val atu = DoubleArray(3); val atv = DoubleArray(3)
        for (j in 0 until GRID_H step 4) for (i in 0 until GRID_W step 4) {
            val k = j * GRID_W + i
            val ix = r.gridX[k].toDouble(); val iy = r.gridY[k].toDouble()
            if (ix < 0 || iy < 0 || ix > 1 || iy > 1) continue
            val row = doubleArrayOf(ix, iy, 1.0)
            val u = (i + 0.5) / GRID_W; val v = (j + 0.5) / GRID_H
            for (a in 0..2) { for (b in 0..2) ata[a][b] += row[a] * row[b]; atu[a] += row[a] * u; atv[a] += row[a] * v }
        }
        val mu = solve3(ata, atu) ?: return null; val mv = solve3(ata, atv) ?: return null
        return doubleArrayOf(mu[0], mu[1], mu[2], mv[0], mv[1], mv[2])
    }

    private fun solve3(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val m = Array(3) { i -> doubleArrayOf(a[i][0], a[i][1], a[i][2], b[i]) }
        for (c in 0..2) {
            val p = (c..2).maxByOrNull { Math.abs(m[it][c]) }!!
            if (Math.abs(m[p][c]) < 1e-12) return null
            val t = m[c]; m[c] = m[p]; m[p] = t
            for (r in 0..2) if (r != c) { val f = m[r][c] / m[c][c]; for (k in c..3) m[r][k] -= f * m[c][k] }
        }
        return DoubleArray(3) { m[it][3] / m[it][it] }
    }

    private fun createSkin(): ImageSegmenter? {
        for (delegate in listOf(Delegate.GPU, Delegate.CPU)) {
            try {
                val seg = ImageSegmenter.createFromOptions(context, ImageSegmenter.ImageSegmenterOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath("selfie_multiclass.tflite").setDelegate(delegate).build())
                    .setRunningMode(RunningMode.IMAGE).setOutputConfidenceMasks(true).setOutputCategoryMask(false).build())
                // run once: a delegate can construct fine and still fail on the first inference
                seg.segment(BitmapImageBuilder(Bitmap.createBitmap(SMALL_W, SMALL_H, Bitmap.Config.ARGB_8888)).build())
                Log.i(ArCaptureView.TAG, "skin segmenter on $delegate")
                return seg
            } catch (e: Throwable) {
                Log.w(ArCaptureView.TAG, "skin segmenter $delegate unavailable: ${e.message}")
            }
        }
        return null
    }

    /** Body-skin probability per pixel of the request image, or null. */
    private fun skinProb(r: Request, bmp: Bitmap): FloatArray? {
        val seg = skin ?: return null
        val res = seg.segment(BitmapImageBuilder(bmp).build())
        val masks = res.confidenceMasks().orElse(null) ?: return null
        if (masks.size <= 2) return null
        val m = masks[2]
        val buf = ByteBufferExtractor.extract(m).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val mw = m.width; val mh = m.height
        val raw = FloatArray(mw * mh).also { buf.get(it) }
        return if (mw == r.w && mh == r.h) raw else FloatArray(r.w * r.h) { i -> raw[((i / r.w) * mh / r.h) * mw + (i % r.w) * mw / r.w] }
    }

    fun pick(u: Float, v: Float) {
        pickAt = floatArrayOf(u, v); modelOn = true
        if (touch == null) worker.execute { if (touch == null) touch = createTouch() }
    }
    fun clearColour() { modelOn = false; seed = null; pickAt = null; anchor = null; areaRef = null; skinMode = null }

    private fun createTouch(): InteractiveSegmenter? {
        for (delegate in listOf(Delegate.GPU, Delegate.CPU)) {
            try {
                val s = InteractiveSegmenter.createFromOptions(context, InteractiveSegmenterOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath("interactive_segmentation.task").setDelegate(delegate).build())
                    .build())
                Log.i(ArCaptureView.TAG, "touch segmenter on $delegate")
                return s
            } catch (e: Throwable) {
                Log.w(ArCaptureView.TAG, "touch segmenter $delegate unavailable: ${e.message}")
            }
        }
        return null
    }

    /** What the server needs to cut the same object out of the photos (null for a person). */
    fun colourInfo(): Map<String, Any>? {
        if (!modelOn || skinMode == true) return null
        val sd = seed ?: return null
        return mapOf("mode" to "touch", "seed" to sd.map { it.toDouble() })
    }

    fun submit(req: Request, onResult: (Map<String, Any>) -> Unit) {
        if (!available) return
        if (!busy.compareAndSet(false, true)) return
        worker.execute {
            try {
                if (!colourMode) { findHands(req)?.let(onResult); return@execute }
                val prob = colourProb(req)
                if (prob != null) onResult(summarise(prob, req.w, req.h, req))
            } catch (e: Throwable) {
                Log.w(ArCaptureView.TAG, "segmentation failed: ${e.message}")
            } finally { busy.set(false) }
        }
    }

    fun close() {
        worker.execute {
            try { touch?.close() } catch (_: Throwable) {}
            try { skin?.close() } catch (_: Throwable) {}
            try { hands?.close() } catch (_: Throwable) {}
        }
        worker.shutdown()
    }

    /** Object probability from the interactive segmenter at the seed point, then only the connected region there;
     *  the seed moves to the object's inner point nearest its centre, so it follows the object frame to frame. */
    private fun colourProb(r: Request): FloatArray? {
        val seg = touch ?: return null
        val w = r.w; val h = r.h; val n = w * h
        pickAt?.let { pv ->  // a new tap: start again from that point
            seed = viewToImage(r, pv[0], pv[1])
            anchor = null
            areaRef = null
            pickAt = null
            skinMode = null
        }
        // the object stays where it is in the world while the phone moves: its 3D anchor, projected into this frame,
        // says where to look (a point fixed on the screen slid onto the background when the phone moved)
        anchor?.let { a ->
            val v = projectView(r.vp, a) ?: return FloatArray(n)  // behind the camera / out of view
            if (v[0] < 0f || v[1] < 0f || v[0] >= 1f || v[1] >= 1f) return FloatArray(n)
            seed = viewToImage(r, v[0], v[1])
        }
        val sd = seed ?: return null
        val bmp = Bitmap.createBitmap(r.rgb, w, h, Bitmap.Config.ARGB_8888)
        val t0 = System.nanoTime()
        // a person: body skin at (or right next to) the tapped point -> follow the skin; else the tapped object
        var prob: FloatArray? = null
        if (skinMode != false) {
            val sp = try { skinProb(r, bmp) } catch (e: Throwable) { null }
            if (sp != null) {
                if (skinMode == null) {
                    val sx = (sd[0] * w).toInt(); val sy = (sd[1] * h).toInt()
                    var best = 0f
                    for (dy in -3..3) for (dx in -3..3) {
                        val x = sx + dx; val y = sy + dy
                        if (x in 0 until w && y in 0 until h) best = maxOf(best, sp[y * w + x])
                    }
                    skinMode = best >= 0.5f
                    Log.i(ArCaptureView.TAG, "tap on ${if (skinMode == true) "skin: following the arm as skin" else "an object: cut-out"} (skin ${"%.2f".format(best)})")
                }
                if (skinMode == true) prob = sp
            } else if (skinMode == null) skinMode = false
        }
        if (prob == null) {
            seg.setImage(BitmapImageBuilder(bmp).build())
            val stroke = Stroke.builder().setBrushMode(Stroke.BrushMode.POSITIVE)
                .setPoints(listOf(NormalizedKeypoint.create(sd[0], sd[1]))).setCompleted(true).build()
            val out = seg.segment(listOf(stroke))
            val mw = out.width; val mh = out.height
            val buf = ByteBufferExtractor.extract(out).order(ByteOrder.nativeOrder())
            val raw = FloatArray(mw * mh)
            if (buf.remaining() >= mw * mh * 4) buf.asFloatBuffer().get(raw) else for (i in 0 until mw * mh) raw[i] = (buf.get(i).toInt() and 0xff) / 255f
            // resample to the request size if the segmenter returned another size
            prob = if (mw == w && mh == h) raw else FloatArray(n) { i -> raw[((i / w) * mh / h) * mw + (i % w) * mw / w] }
        }
        if (++runs % 10 == 0) Log.d(ArCaptureView.TAG, "segment ${(System.nanoTime() - t0) / 1_000_000} ms (skin=$skinMode)")
        bmp.recycle()
        // keep the connected region at the seed
        val sx = (sd[0] * w).toInt().coerceIn(0, w - 1); val sy = (sd[1] * h).toInt().coerceIn(0, h - 1)
        val keep = BooleanArray(n)
        val queue = IntArray(n)
        var start = sy * w + sx
        if (prob[start] < 0.5f) {  // seed slipped off: nearest object pixel
            var bd = Int.MAX_VALUE
            for (i in 0 until n) if (prob[i] >= 0.5f) { val d = (i % w - sx) * (i % w - sx) + (i / w - sy) * (i / w - sy); if (d < bd) { bd = d; start = i } }
            if (bd == Int.MAX_VALUE) return FloatArray(n)
        }
        var head = 0; var tail = 0; queue[tail++] = start; keep[start] = true
        var cx = 0.0; var cy = 0.0
        while (head < tail) {
            val i = queue[head++]; val x = i % w; val y = i / w
            cx += x; cy += y
            if (x > 0 && !keep[i - 1] && prob[i - 1] >= 0.5f) { keep[i - 1] = true; queue[tail++] = i - 1 }
            if (x < w - 1 && !keep[i + 1] && prob[i + 1] >= 0.5f) { keep[i + 1] = true; queue[tail++] = i + 1 }
            if (y > 0 && !keep[i - w] && prob[i - w] >= 0.5f) { keep[i - w] = true; queue[tail++] = i - w }
            if (y < h - 1 && !keep[i + w] && prob[i + w] >= 0.5f) { keep[i + w] = true; queue[tail++] = i + w }
        }
        cx /= tail; cy /= tail
        // a cut-out far bigger than the object was, or not around the anchor, is the background: drop it
        if (tail > n * 0.6) return FloatArray(n)
        // a sudden jump in size = the cut-out spilled onto something touching it (blanket, chair): skip the frame and
        // keep following the object (a real size change, walking closer, is gradual). Give in after a while.
        val ref = areaRef
        if (ref != null && tail > 1.6 * ref && spills < 12) { spills++; return null }
        spills = 0
        areaRef = if (ref == null) tail.toDouble() else ref + 0.15 * (tail - ref)
        var bi = start; var bd = Double.MAX_VALUE
        for (q in 0 until tail) { val i = queue[q]; val d = (i % w - cx) * (i % w - cx) + (i / w - cy) * (i / w - cy); if (d < bd) { bd = d; bi = i } }
        seed = floatArrayOf((bi % w + 0.5f) / w, (bi / w + 0.5f) / h)
        for (i in 0 until n) if (!keep[i]) prob[i] = 0f
        updateAnchor(r, keep, w, h)
        return prob
    }

    /** The object's 3D anchor: median of the frame's AR points that land on the kept region (slowly averaged). */
    private fun updateAnchor(r: Request, keep: BooleanArray, w: Int, h: Int) {
        val xs = ArrayList<Float>(); val ys = ArrayList<Float>(); val zs = ArrayList<Float>()
        var q = 0
        while (q + 2 < r.points.size) {
            val x = r.points[q]; val y = r.points[q + 1]; val z = r.points[q + 2]; q += 3
            val v = projectView(r.vp, floatArrayOf(x, y, z)) ?: continue
            if (v[0] < 0f || v[1] < 0f || v[0] >= 1f || v[1] >= 1f) continue
            val im = viewToImage(r, v[0], v[1])
            val i = (im[1] * h).toInt().coerceIn(0, h - 1) * w + (im[0] * w).toInt().coerceIn(0, w - 1)
            if (keep[i]) { xs.add(x); ys.add(y); zs.add(z) }
        }
        if (xs.size < 4) return
        val m = floatArrayOf(xs.sorted()[xs.size / 2], ys.sorted()[ys.size / 2], zs.sorted()[zs.size / 2])
        val a = anchor
        anchor = if (a == null) m else floatArrayOf(a[0] + 0.2f * (m[0] - a[0]), a[1] + 0.2f * (m[1] - a[1]), a[2] + 0.2f * (m[2] - a[2]))
    }

    /** World point -> view point (0..1, y down) with the frame's view-projection, or null behind the camera. */
    private fun projectView(vp: FloatArray, p: FloatArray): FloatArray? {
        val cw = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15]
        if (cw <= 0.01f) return null
        return floatArrayOf(((vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12]) / cw + 1f) / 2f,
            (1f - (vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13]) / cw) / 2f)
    }

    /** View point (0..1) -> portrait image point (0..1) through the frame's grid map. */
    private fun viewToImage(r: Request, u: Float, v: Float): FloatArray {
        val gi = (u * GRID_W).toInt().coerceIn(0, GRID_W - 1); val gj = (v * GRID_H).toInt().coerceIn(0, GRID_H - 1)
        val k = gj * GRID_W + gi
        return floatArrayOf(r.gridX[k].coerceIn(0f, 1f), r.gridY[k].coerceIn(0f, 1f))
    }

    private fun summarise(prob: FloatArray, mw: Int, mh: Int, r: Request): Map<String, Any> {
        // limb share of the camera image + luminance of limb vs background (backlight check)
        var nLimb = 0; var lumLimb = 0.0; var lumBg = 0.0
        val sx = r.w.toFloat() / mw; val sy = r.h.toFloat() / mh
        for (y in 0 until mh) for (x in 0 until mw) {
            val px = r.rgb[(y * sy).toInt().coerceAtMost(r.h - 1) * r.w + (x * sx).toInt().coerceAtMost(r.w - 1)]
            val lum = 0.3 * ((px shr 16) and 0xff) + 0.59 * ((px shr 8) and 0xff) + 0.11 * (px and 0xff)
            if (prob[y * mw + x] > 0.5f) { nLimb++; lumLimb += lum } else lumBg += lum
        }
        val nBg = mw * mh - nLimb
        // mask on the on-screen grid (precomputed view -> image mapping)
        val gw = GRID_W; val gh = GRID_H
        val grid = ByteArray(gw * gh)
        var n = 0; var sxv = 0.0; var syv = 0.0; var edge = 0
        var x0 = 1.0; var y0 = 1.0; var x1 = 0.0; var y1 = 0.0
        for (j in 0 until gh) for (i in 0 until gw) {
            val k = j * gw + i
            val u = r.gridX[k]; val v = r.gridY[k]
            if (u < 0f || v < 0f || u >= 1f || v >= 1f) continue
            val p = bilinear(prob, mw, mh, u * mw - 0.5f, v * mh - 0.5f)
            grid[k] = (p * 255).toInt().coerceIn(0, 255).toByte()  // full confidence: the app draws a smooth contour
            if (p > 0.5f) {
                val vx = (i + 0.5) / gw; val vy = (j + 0.5) / gh
                n++; sxv += vx; syv += vy
                x0 = minOf(x0, vx); y0 = minOf(y0, vy); x1 = maxOf(x1, vx); y1 = maxOf(y1, vy)
                if (i == 0 || j == 0 || i == gw - 1 || j == gh - 1) edge++
            }
        }
        // distance to the arm: median camera distance of ARCore's tracked 3D points that fall on the arm
        val dists = ArrayList<Float>()
        val ax = ArrayList<Float>(); val ay = ArrayList<Float>(); val az = ArrayList<Float>()
        val pts = r.points
        var q = 0
        while (q + 2 < pts.size) {
            val x = pts[q]; val y = pts[q + 1]; val z = pts[q + 2]; q += 3
            val vpm = r.vp
            val cw = vpm[3] * x + vpm[7] * y + vpm[11] * z + vpm[15]
            if (cw <= 0.01f) continue
            val sx2 = ((vpm[0] * x + vpm[4] * y + vpm[8] * z + vpm[12]) / cw + 1f) / 2f
            val sy2 = (1f - (vpm[1] * x + vpm[5] * y + vpm[9] * z + vpm[13]) / cw) / 2f
            if (sx2 < 0f || sy2 < 0f || sx2 >= 1f || sy2 >= 1f) continue
            val gi = (sx2 * gw).toInt(); val gj = (sy2 * gh).toInt()
            if ((grid[gj * gw + gi].toInt() and 0xff) < 160) continue
            val dx = x - r.cam[0]; val dy = y - r.cam[1]; val dz = z - r.cam[2]
            dists.add(Math.sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat())
            ax.add(x); ay.add(y); az.add(z)
        }
        val dist = if (dists.size >= 4) dists.sorted()[dists.size / 2].toDouble() else -1.0
        // a point on the arm's visible surface (per-axis median of its tracked points), world frame
        val centre = if (ax.size >= 4) listOf(ax.sorted()[ax.size / 2].toDouble(), ay.sorted()[ay.size / 2].toDouble(),
            az.sorted()[az.size / 2].toDouble()) else null
        return hashMapOf(
            "type" to "limb",
            "dist" to dist, "distPoints" to dists.size,
            "centre" to (centre ?: emptyList<Double>()),
            "frac" to nLimb.toDouble() / (mw * mh),
            "cx" to (if (n > 0) sxv / n else -1.0), "cy" to (if (n > 0) syv / n else -1.0),
            "box" to listOf(x0, y0, x1, y1),
            "edge" to (if (n > 0) edge.toDouble() / n else 0.0),
            "limbLum" to (if (nLimb > 0) lumLimb / nLimb else 0.0),
            "bgLum" to (if (nBg > 0) lumBg / nBg else 0.0),
            "gw" to gw, "gh" to gh, "mask" to grid, "vp" to r.vp.map { it.toDouble() },
            "t" to System.currentTimeMillis(),
        )
    }

    private fun bilinear(a: FloatArray, w: Int, h: Int, x: Float, y: Float): Float {
        val x0 = x.toInt().coerceIn(0, w - 1); val y0 = y.toInt().coerceIn(0, h - 1)
        val x1 = (x0 + 1).coerceAtMost(w - 1); val y1 = (y0 + 1).coerceAtMost(h - 1)
        val fx = (x - x0).coerceIn(0f, 1f); val fy = (y - y0).coerceIn(0f, 1f)
        val top = a[y0 * w + x0] * (1 - fx) + a[y0 * w + x1] * fx
        val bot = a[y1 * w + x0] * (1 - fx) + a[y1 * w + x1] * fx
        return top * (1 - fy) + bot * fy
    }

    companion object {
        const val GRID_W = 72     // on-screen mask grid (the app contours it with sub-cell interpolation)
        const val GRID_H = 128
        const val SMALL_W = 180   // portrait sample size fed to the model
        const val SMALL_H = 320
        const val HAND_W = 360    // hands: a larger sample (the landmarks need the fingers' detail)
        const val HAND_H = 640

        /** Small portrait RGB copy of a landscape YUV_420_888 camera image (rotated 90 deg clockwise). */
        fun sampleRgb(image: Image, SMALL_W: Int = LimbDetector.SMALL_W, SMALL_H: Int = LimbDetector.SMALL_H): IntArray {
            val w = image.width; val h = image.height
            val yP = image.planes[0]; val uP = image.planes[1]; val vP = image.planes[2]
            val yb = yP.buffer; val ub = uP.buffer; val vb = vP.buffer
            val out = IntArray(SMALL_W * SMALL_H)
            for (py in 0 until SMALL_H) {
                val lx = (py * w / SMALL_H).coerceAtMost(w - 1)            // portrait y  <- landscape x
                for (px in 0 until SMALL_W) {
                    val ly = (h - 1 - px * h / SMALL_W).coerceIn(0, h - 1)  // portrait x  <- landscape (H-1-y)
                    val yy = (yb.get(ly * yP.rowStride + lx * yP.pixelStride).toInt() and 0xff)
                    val ci = (ly / 2) * uP.rowStride + (lx / 2) * uP.pixelStride
                    val cv = (ly / 2) * vP.rowStride + (lx / 2) * vP.pixelStride
                    val u = (ub.get(ci).toInt() and 0xff) - 128
                    val v = (vb.get(cv).toInt() and 0xff) - 128
                    val r = (yy + 1.402 * v).toInt().coerceIn(0, 255)
                    val g = (yy - 0.344 * u - 0.714 * v).toInt().coerceIn(0, 255)
                    val b = (yy + 1.772 * u).toInt().coerceIn(0, 255)
                    out[py * SMALL_W + px] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            return out
        }
    }
}
