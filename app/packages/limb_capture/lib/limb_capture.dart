/// LimbCapture: native AR capture for LimbScan.
///
/// Android: ARCore (autofocus, largest CPU camera image, metric pose + intrinsics per photo).
/// iOS: ARKit (coming next).
library;

import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

/// Live camera pose for on-screen guidance (screen-aligned axes, metres, AR world frame).
class ArPose {
  ArPose(this.tracking, this.reason, this.pos, this.fwd, this.right, this.t, {this.vp});

  factory ArPose.fromMap(Map m) => ArPose(
        m['tracking'] as String,
        m['reason'] as String?,
        _vec(m['pos']),
        _vec(m['fwd']),
        _vec(m['right']),
        m['t'] as int,
        vp: m['vp'] == null ? null : _vec(m['vp']),
      );

  /// World -> clip matrix of the on-screen preview (projection x view, 4x4 column-major), for AR overlays.
  final List<double>? vp;

  final String tracking; // tracking | paused | stopped
  final String? reason; // why tracking is not available (e.g. INSUFFICIENT_FEATURES, EXCESSIVE_MOTION)
  final List<double> pos;
  final List<double> fwd;
  final List<double> right;
  final int t;

  bool get isTracking => tracking == 'tracking';
}

/// Live on-device limb detection (MediaPipe body-skin), in on-screen coordinates (0..1 of the preview).
class LimbInfo {
  LimbInfo.fromMap(Map m)
      : frac = (m['frac'] as num).toDouble(),
        cx = (m['cx'] as num).toDouble(),
        cy = (m['cy'] as num).toDouble(),
        box = _vec(m['box']),
        edge = (m['edge'] as num).toDouble(),
        limbLum = (m['limbLum'] as num).toDouble(),
        bgLum = (m['bgLum'] as num).toDouble(),
        gridW = m['gw'] as int,
        gridH = m['gh'] as int,
        mask = m['mask'] as Uint8List,
        vp = m['vp'] == null ? null : _vec(m['vp']),
        t = m['t'] as int;

  /// Share of the camera image that is limb skin.
  final double frac;
  /// Limb centre on screen (0..1), -1 when no limb.
  final double cx, cy;
  /// Limb bounding box on screen: x0, y0, x1, y1 (0..1).
  final List<double> box;
  /// Share of limb cells touching the screen border (limb cut off).
  final double edge;
  final double limbLum, bgLum;
  /// Limb confidence on a gridW x gridH grid over the preview (0..255), for the live highlight.
  final int gridW, gridH;
  final Uint8List mask;
  /// World -> clip matrix of the preview at the moment of the analysed camera frame (to warp the mask to now).
  final List<double>? vp;
  final int t;

  bool get visible => frac > 0.04 && cx >= 0;
  bool get backlit => visible && (limbLum < 55 || bgLum > 1.7 * limbLum);
  bool get cutOff => visible && edge > 0.25 && frac > 0.35;
  bool get tooSmall => visible && frac < 0.07;
}

/// One captured photo + everything the server needs for metric reconstruction.
class CapturedFrame {
  CapturedFrame.fromMap(Map m)
      : path = m['path'] as String,
        width = m['width'] as int,
        height = m['height'] as int,
        pose = _vec(m['pose']),
        k = _vec(m['K']),
        sharpness = (m['sharpness'] as num).toDouble(),
        t = m['t'] as int,
        source = m['src'] as String;

  final String path;
  final int width;
  final int height;
  final List<double> pose; // camera -> world, 4x4 column-major, metres
  final List<double> k; // fx, fy, cx, cy in pixels of the stored (portrait) JPEG
  final double sharpness;
  final int t;
  final String source; // arcore | arkit

  /// JSON sent with the upload (same fields the LimbScan server reads from WebXR captures).
  Map<String, Object> toMeta() => {'pose': pose, 'K': k, 'w': width, 'h': height, 't': t, 'src': source};
}

List<double> _vec(Object? v) => (v as List).map((e) => (e as num).toDouble()).toList();

class LimbCapture {
  static const _ch = MethodChannel('limb_capture');
  static const _ev = EventChannel('limb_capture/pose');
  // One native event stream (an EventChannel supports a single listener); typed views are filtered from it.
  static final Stream<dynamic> _events = _ev.receiveBroadcastStream().asBroadcastStream();
  static Stream<ArPose>? _poses;

  /// `supported` | `install_required` | `unsupported` | `unknown`
  static Future<String> checkAvailability() async => (await _ch.invokeMethod<String>('checkAvailability')) ?? 'unknown';

  /// Asks the user to install / update Google Play Services for AR. Returns true when installed.
  static Future<bool> requestInstall() async => (await _ch.invokeMethod<bool>('requestInstall')) ?? false;

  /// Grab the current camera frame into [dir]. Throws [PlatformException] (e.g. `not_tracking`, `busy`).
  static Future<CapturedFrame> capture(String dir) async =>
      CapturedFrame.fromMap((await _ch.invokeMethod<Map>('capture', {'dir': dir}))!);

  static Future<void> setTorch(bool on) => _ch.invokeMethod('setTorch', {'on': on});

  /// Crash / error report recorded since the last call (Java + native exits), or null.
  static Future<String?> lastCrash() => _ch.invokeMethod<String>('lastCrash');

  /// Pose events (~15 Hz) while an [ArCameraView] is on screen.
  static Stream<ArPose> get poses => _poses ??= _events
      .where((e) => e is Map && e['type'] == 'pose')
      .map((e) => ArPose.fromMap(e as Map));

  /// Live limb detection results (~5 Hz) while an [ArCameraView] is on screen.
  static Stream<LimbInfo> get limbs => _limbs ??= _events
      .where((e) => e is Map && e['type'] == 'limb')
      .map((e) => LimbInfo.fromMap(e as Map));
  static Stream<LimbInfo>? _limbs;

  /// Errors from the native AR session (camera unavailable, ARCore problems).
  static Stream<String> get errors => _events
      .where((e) => e is Map && e['type'] == 'error')
      .map((e) => (e as Map)['message'] as String);
}

/// The live AR camera preview. It owns the native AR session: put it on screen to start tracking.
class ArCameraView extends StatelessWidget {
  const ArCameraView({super.key});

  static const _viewType = 'limb_capture/arview';

  @override
  Widget build(BuildContext context) {
    if (defaultTargetPlatform == TargetPlatform.android) {
      // Hybrid composition: the ARCore preview is a GLSurfaceView, which needs a real Android view.
      return PlatformViewLink(
        viewType: _viewType,
        surfaceFactory: (context, controller) => AndroidViewSurface(
          controller: controller as AndroidViewController,
          gestureRecognizers: const <Factory<OneSequenceGestureRecognizer>>{},
          hitTestBehavior: PlatformViewHitTestBehavior.transparent,
        ),
        onCreatePlatformView: (params) => PlatformViewsService.initExpensiveAndroidView(
          id: params.id,
          viewType: _viewType,
          layoutDirection: TextDirection.ltr,
          creationParamsCodec: const StandardMessageCodec(),
          onFocus: () => params.onFocusChanged(true),
        )
          ..addOnPlatformViewCreatedListener(params.onPlatformViewCreated)
          ..create(),
      );
    }
    if (defaultTargetPlatform == TargetPlatform.iOS) {
      return const UiKitView(viewType: _viewType, creationParamsCodec: StandardMessageCodec());
    }
    return const Center(child: Text('AR capture needs Android or iOS'));
  }
}
