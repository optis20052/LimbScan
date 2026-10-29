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
            "lastCrash" -> result.success(activity?.let { CrashLog.collect(it) })
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
