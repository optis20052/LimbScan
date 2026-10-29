import Flutter
import UIKit

/// iOS side of LimbCapture. The ARKit capture (autofocus, high-resolution frames, metric pose + intrinsics,
/// optional LiDAR depth) is the next milestone; until then the plugin reports AR as unavailable.
public class LimbCapturePlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "limb_capture", binaryMessenger: registrar.messenger())
    registrar.addMethodCallDelegate(LimbCapturePlugin(), channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "checkAvailability": result("unsupported")
    case "requestInstall": result(false)
    default: result(FlutterMethodNotImplemented)
    }
  }
}
