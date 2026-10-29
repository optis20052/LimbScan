package com.spentys.limb_capture

import android.opengl.GLES11Ext
import android.opengl.GLES20
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** Draws the ARCore camera image (external OES texture) full-screen, correctly cropped and rotated. */
class BackgroundRenderer {
    var textureId = -1
        private set
    private var program = 0
    private var posAttr = 0
    private var uvAttr = 0
    private val quad: FloatBuffer = floatBuffer(floatArrayOf(-1f, -1f, +1f, -1f, -1f, +1f, +1f, +1f))
    private val uv: FloatBuffer = floatBuffer(FloatArray(8))

    fun createOnGlThread() {
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        textureId = tex[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        program = link(VERTEX, FRAGMENT)
        posAttr = GLES20.glGetAttribLocation(program, "a_Position")
        uvAttr = GLES20.glGetAttribLocation(program, "a_TexCoord")
    }

    fun draw(frame: Frame) {
        if (frame.hasDisplayGeometryChanged()) {
            quad.position(0); uv.position(0)
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quad,
                Coordinates2d.TEXTURE_NORMALIZED, uv)
        }
        if (frame.timestamp == 0L) return
        quad.position(0); uv.position(0)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glVertexAttribPointer(posAttr, 2, GLES20.GL_FLOAT, false, 0, quad)
        GLES20.glVertexAttribPointer(uvAttr, 2, GLES20.GL_FLOAT, false, 0, uv)
        GLES20.glEnableVertexAttribArray(posAttr)
        GLES20.glEnableVertexAttribArray(uvAttr)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(posAttr)
        GLES20.glDisableVertexAttribArray(uvAttr)
        GLES20.glDepthMask(true)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    private fun floatBuffer(a: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(a); position(0) }

    private fun compile(type: Int, src: String): Int {
        val s = GLES20.glCreateShader(type)
        GLES20.glShaderSource(s, src); GLES20.glCompileShader(s)
        return s
    }

    private fun link(vs: String, fs: String): Int {
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, compile(GLES20.GL_VERTEX_SHADER, vs))
        GLES20.glAttachShader(p, compile(GLES20.GL_FRAGMENT_SHADER, fs))
        GLES20.glLinkProgram(p)
        return p
    }

    companion object {
        private const val VERTEX = """
            attribute vec4 a_Position; attribute vec2 a_TexCoord; varying vec2 v_TexCoord;
            void main() { gl_Position = a_Position; v_TexCoord = a_TexCoord; }"""
        private const val FRAGMENT = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float; varying vec2 v_TexCoord; uniform samplerExternalOES s_Texture;
            void main() { gl_FragColor = texture2D(s_Texture, v_TexCoord); }"""
    }
}
