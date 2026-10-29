package com.spentys.limb_capture

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import java.io.File

/**
 * Field crash reports without a USB cable: Java crashes are written to a file by an uncaught-exception handler,
 * native crashes are read back from Android's exit records on the next launch. The app uploads them to the server.
 */
object CrashLog {
    private const val FILE = "limbscan_crash.txt"
    private const val PREFS = "limbscan_crash"

    fun install(ctx: Context) {
        val app = ctx.applicationContext
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            try { write(app, "UNCAUGHT on thread ${t.name}\n${e.stackTraceToString()}") } catch (_: Throwable) {}
            previous?.uncaughtException(t, e)
        }
    }

    /** Caught but unexpected: keep it for the report too. */
    fun note(ctx: Context, where: String, e: Throwable) {
        try { write(ctx.applicationContext, "CAUGHT in $where\n${e.stackTraceToString()}") } catch (_: Throwable) {}
    }

    private fun write(ctx: Context, text: String) {
        File(ctx.filesDir, FILE).appendText("---- ${System.currentTimeMillis()} ${Build.MANUFACTURER} ${Build.MODEL} " +
            "Android ${Build.VERSION.RELEASE}\n$text\n")
    }

    /** Everything recorded since the last call (then cleared), or null. */
    fun collect(ctx: Context): String? {
        val out = StringBuilder()
        val f = File(ctx.filesDir, FILE)
        if (f.exists()) { out.append(f.readText()); f.delete() }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val seen = prefs.getLong("seen", 0L)
            val am = ctx.getSystemService(ActivityManager::class.java)
            var newest = seen
            for (x in am.getHistoricalProcessExitReasons(ctx.packageName, 0, 5)) {
                if (x.timestamp <= seen) continue
                newest = maxOf(newest, x.timestamp)
                if (x.reason != ApplicationExitInfo.REASON_CRASH && x.reason != ApplicationExitInfo.REASON_CRASH_NATIVE &&
                    x.reason != ApplicationExitInfo.REASON_ANR) continue
                out.append("---- exit ${x.timestamp} reason=${x.reason} status=${x.status} ${Build.MODEL} " +
                    "Android ${Build.VERSION.RELEASE}\n${x.description}\n")
                if (x.reason == ApplicationExitInfo.REASON_ANR) {
                    try { x.traceInputStream?.bufferedReader()?.use { out.append(it.readText().take(20000)) } } catch (_: Throwable) {}
                }
                if (x.reason == ApplicationExitInfo.REASON_CRASH_NATIVE) {
                    // tombstone is a protobuf; the printable strings (abort message, backtrace symbols) are what we need
                    try {
                        x.traceInputStream?.use { s ->
                            val b = s.readBytes()
                            out.append(printable(b).take(30000))
                        }
                    } catch (_: Throwable) {}
                }
            }
            prefs.edit().putLong("seen", newest).apply()
        }
        return out.toString().ifBlank { null }
    }

    private fun printable(b: ByteArray): String {
        val sb = StringBuilder(); val run = StringBuilder()
        for (x in b) {
            val c = x.toInt() and 0xff
            if (c in 32..126) run.append(c.toChar()) else {
                if (run.length >= 6) sb.append(run).append('\n')
                run.setLength(0)
            }
        }
        if (run.length >= 6) sb.append(run)
        return sb.toString()
    }
}
