package com.spentys.limb_capture

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
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * On-device limb detection with MediaPipe's selfie-multiclass model (class 2 = body skin), the same model the
 * server uses to mask the limb. Runs on a small portrait copy of the camera image a few times per second.
 *
 * Results are reported in VIEW coordinates (0..1 across the on-screen preview), so the app can highlight the
 * arm exactly where the user sees it, plus: how much of the frame is arm, whether it is cut off at the edge,
 * and whether it is backlit (dark arm in front of a bright window).
 */
class LimbDetector(context: Context) {
    class Request(val rgb: IntArray, val w: Int, val h: Int, val gridX: FloatArray, val gridY: FloatArray, val vp: FloatArray)

    private val worker = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var runs = 0
    @Volatile private var segmenter: ImageSegmenter? = null

    init { worker.execute { segmenter = create(context) } }  // model load off the UI thread

    /** GPU first (~3x faster, so the highlight keeps up with the camera), CPU if the GPU path is unavailable. */
    private fun create(context: Context): ImageSegmenter? {
        for (delegate in listOf(Delegate.GPU, Delegate.CPU)) {
            try {
                val seg = ImageSegmenter.createFromOptions(
                    context,
                    ImageSegmenter.ImageSegmenterOptions.builder()
                        .setBaseOptions(BaseOptions.builder().setModelAssetPath("selfie_multiclass.tflite").setDelegate(delegate).build())
                        .setRunningMode(RunningMode.IMAGE)
                        .setOutputConfidenceMasks(true)
                        .setOutputCategoryMask(false)
                        .build()
                )
                // run once: a delegate can construct fine and still fail on the first inference
                seg.segment(BitmapImageBuilder(Bitmap.createBitmap(SMALL_W, SMALL_H, Bitmap.Config.ARGB_8888)).build())
                Log.i(ArCaptureView.TAG, "limb segmenter on $delegate")
                return seg
            } catch (e: Throwable) {
                Log.w(ArCaptureView.TAG, "limb segmenter $delegate unavailable: ${e.message}")
            }
        }
        return null
    }

    val available get() = segmenter != null
    val idle get() = !busy.get()

    fun submit(req: Request, onResult: (Map<String, Any>) -> Unit) {
        val seg = segmenter ?: return
        if (!busy.compareAndSet(false, true)) return
        worker.execute {
            try {
                val bmp = Bitmap.createBitmap(req.rgb, req.w, req.h, Bitmap.Config.ARGB_8888)
                val t0 = System.nanoTime()
                val res = seg.segment(BitmapImageBuilder(bmp).build())
                if (++runs % 20 == 0) Log.d(ArCaptureView.TAG, "segment ${(System.nanoTime() - t0) / 1_000_000} ms")
                val masks = res.confidenceMasks().orElse(null)
                if (masks != null && masks.size > 2) {
                    val m = masks[2]
                    val buf = ByteBufferExtractor.extract(m).order(ByteOrder.nativeOrder()).asFloatBuffer()
                    val mw = m.width; val mh = m.height
                    val prob = FloatArray(mw * mh).also { buf.get(it) }
                    onResult(summarise(prob, mw, mh, req))
                }
                bmp.recycle()
            } catch (e: Throwable) {
                Log.w(ArCaptureView.TAG, "segmentation failed: ${e.message}")
            } finally { busy.set(false) }
        }
    }

    fun close() { worker.execute { try { segmenter?.close() } catch (_: Throwable) {} }; worker.shutdown() }

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
        return hashMapOf(
            "type" to "limb",
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

        /** Small portrait RGB copy of a landscape YUV_420_888 camera image (rotated 90 deg clockwise). */
        fun sampleRgb(image: Image): IntArray {
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
