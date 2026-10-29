package com.spentys.limb_capture

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream

/** YUV_420_888 camera image -> portrait JPEG on disk (+ a sharpness score for blur rejection). */
object FrameEncoder {
    class Raw(val nv21: ByteArray, val width: Int, val height: Int)
    class Out(val path: String, val width: Int, val height: Int, val sharpness: Double)

    /** Copy the three planes into NV21 (Y + interleaved VU), honouring row and pixel strides. */
    fun copyYuv(image: Image): Raw {
        val w = image.width; val h = image.height
        val out = ByteArray(w * h * 3 / 2)
        val y = image.planes[0]; val u = image.planes[1]; val v = image.planes[2]
        val yb = y.buffer; val ub = u.buffer; val vb = v.buffer
        var o = 0
        for (row in 0 until h) {
            yb.position(row * y.rowStride)
            if (y.pixelStride == 1) { yb.get(out, o, w); o += w }
            else for (col in 0 until w) out[o++] = yb.get(row * y.rowStride + col * y.pixelStride)
        }
        for (row in 0 until h / 2) for (col in 0 until w / 2) {
            out[o++] = vb.get(row * v.rowStride + col * v.pixelStride)
            out[o++] = ub.get(row * u.rowStride + col * u.pixelStride)
        }
        return Raw(out, w, h)
    }

    fun encodePortrait(raw: Raw, dir: String, t: Long, quality: Int = 95): Out {
        val sharp = sharpness(raw)
        val tmp = ByteArrayOutputStream()
        YuvImage(raw.nv21, ImageFormat.NV21, raw.width, raw.height, null)
            .compressToJpeg(Rect(0, 0, raw.width, raw.height), 100, tmp)
        val land = BitmapFactory.decodeByteArray(tmp.toByteArray(), 0, tmp.size())
        val portrait = Bitmap.createBitmap(land, 0, 0, land.width, land.height, Matrix().apply { postRotate(90f) }, true)
        land.recycle()
        File(dir).mkdirs()
        val f = File(dir, "frame_$t.jpg")
        FileOutputStream(f).use { portrait.compress(Bitmap.CompressFormat.JPEG, quality, it) }
        val res = Out(f.absolutePath, portrait.width, portrait.height, sharp)
        portrait.recycle()
        return res
    }

    /** Variance of a 4-neighbour Laplacian on the Y plane, subsampled by 4 (fast, rotation-independent). */
    private fun sharpness(raw: Raw): Double {
        val w = raw.width; val h = raw.height; val y = raw.nv21
        var s = 0.0; var s2 = 0.0; var n = 0
        var r = 4
        while (r < h - 4) {
            var c = 4
            while (c < w - 4) {
                val i = r * w + c
                val l = 4 * (y[i].toInt() and 0xff) - (y[i - 1].toInt() and 0xff) - (y[i + 1].toInt() and 0xff) -
                    (y[i - w].toInt() and 0xff) - (y[i + w].toInt() and 0xff)
                s += l; s2 += l.toDouble() * l; n++
                c += 4
            }
            r += 4
        }
        val m = s / n
        return s2 / n - m * m
    }
}
