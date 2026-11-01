package com.spentys.limb_capture

import android.app.Activity
import com.google.ar.core.ArCoreApk
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/**
 * LimbCapture: native AR capture for LimbScan.
 *
 *  - MethodChannel `limb_capture`        availability / ARCore install / capture a frame
 *  - EventChannel  `limb_capture/pose`   live camera pose + tracking state (~15 Hz) for on-screen guidance
 *  - PlatformView  `limb_capture/arview` the live AR camera preview (owns the ARCore session)
 */
class LimbCapturePlugin : FlutterPlugin, MethodChannel.MethodCallHandler, ActivityAware {
    private lateinit var channel: MethodChannel
    private lateinit var poseChannel: EventChannel
    private var activity: Activity? = null
    private val views = ArViewRegistry()

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        CrashLog.install(binding.applicationContext)
        channel = MethodChannel(binding.binaryMessenger, "limb_capture")
        channel.setMethodCallHandler(this)
        poseChannel = EventChannel(binding.binaryMessenger, "limb_capture/pose")
        poseChannel.setStreamHandler(views.poseStream)
        binding.platformViewRegistry.registerViewFactory("limb_capture/arview", ArViewFactory(views) { activity })
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        poseChannel.setStreamHandler(null)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "checkAvailability" -> {
                val act = activity ?: return result.success("unknown")
                val a = ArCoreApk.getInstance().checkAvailability(act)
                result.success(
                    when {
                        a == ArCoreApk.Availability.SUPPORTED_INSTALLED -> "supported"
                        a == ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD ||
                            a == ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED -> "install_required"
                        a.isTransient -> "unknown"
                        else -> "unsupported"
                    }
                )
            }
            "requestInstall" -> {
                val act = activity ?: return result.error("no_activity", "No activity", null)
                try {
                    val status = ArCoreApk.getInstance().requestInstall(act, true)
                    result.success(status == ArCoreApk.InstallStatus.INSTALLED)
                } catch (e: Exception) {
                    result.error("install_failed", e.message, null)
                }
            }
            "capture" -> {
                val dir = call.argument<String>("dir") ?: return result.error("args", "dir missing", null)
                val view = views.active ?: return result.error("no_view", "AR view not running", null)
                view.requestCapture(dir, result)
            }
            "hitTest" -> {
                val view = views.active ?: return result.success(null)
                view.requestHit((call.argument<Double>("u") ?: 0.5).toFloat(), (call.argument<Double>("v") ?: 0.5).toFloat(), result)
            }
            "depth" -> {
                val view = views.active ?: return result.success(null)
                val uv = (call.argument<List<Double>>("uv") ?: emptyList()).map { it.toFloat() }.toFloatArray()
                view.requestDepth(uv, result)
            }
            "readLog" -> {
                // this app's own log (an app may read its own lines): sent to the server with each scan, so a test
                // needs no USB cable
                val lines = call.argument<Int>("lines") ?: 6000
                Thread {
                    val text = try {
                        // the whole buffer of this process, minus system chatter that would push our lines out
                        // (ARCore's depth module can log ~10 errors a second)
                        val noise = listOf("spherical_rectifier", "AF_DEBUG", "gralloc", "SV[", "VRI[", "BLASTBuffer",
                            "InsetsController", "InputMethod", "ViewRootImpl", "SurfaceView", "BufferQueue")
                        val p = ProcessBuilder("logcat", "-d", "-v", "time", "--pid=${android.os.Process.myPid()}")
                            .redirectErrorStream(true).start()
                        val kept = p.inputStream.bufferedReader().lineSequence().filter { l -> noise.none { l.contains(it) } }.toList()
                        p.waitFor()
                        kept.takeLast(lines).joinToString("\n")
                    } catch (e: Exception) { "log unavailable: ${e.message}" }
                    android.os.Handler(android.os.Looper.getMainLooper()).post { result.success(text) }
                }.start()
            }
            "lastCrash" -> result.success(activity?.let { CrashLog.collect(it) })
            "pickColour" -> {
                views.active?.pickColour((call.argument<Double>("u") ?: 0.5).toFloat(), (call.argument<Double>("v") ?: 0.5).toFloat())
                result.success(null)
            }
            "findHands" -> {
                views.active?.findHands(call.argument<Boolean>("on") ?: true)
                result.success(null)
            }
            "clearColour" -> {
                views.active?.clearColour()
                result.success(null)
            }
            "setTorch" -> {
                views.active?.setTorch(call.argument<Boolean>("on") ?: false)
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) { activity = binding.activity }
    override fun onDetachedFromActivityForConfigChanges() { activity = null }
    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) { activity = binding.activity }
    override fun onDetachedFromActivity() { activity = null }
}

/** Keeps track of the running AR view and forwards its pose events to Flutter. */
class ArViewRegistry {
    @Volatile var active: ArCaptureView? = null
    @Volatile var sink: EventChannel.EventSink? = null
    val poseStream = object : EventChannel.StreamHandler {
        override fun onListen(arguments: Any?, events: EventChannel.EventSink?) { sink = events }
        override fun onCancel(arguments: Any?) { sink = null }
    }
}
