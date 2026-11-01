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

/// Live cut-out of the tapped limb (MediaPipe interactive segmenter), in on-screen coordinates (0..1 of the preview).
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
        dist = m['dist'] == null ? -1 : (m['dist'] as num).toDouble(),
        surface = (m['centre'] as List?)?.isNotEmpty == true ? _vec(m['centre']) : null,
        t = m['t'] as int;

  /// The same detection with another confidence grid (e.g. smoothed over time by the app).
  LimbInfo withMask(Uint8List m) =>
      LimbInfo._(frac, cx, cy, box, edge, limbLum, bgLum, gridW, gridH, m, vp, dist, surface, t);

  LimbInfo._(this.frac, this.cx, this.cy, this.box, this.edge, this.limbLum, this.bgLum, this.gridW, this.gridH, this.mask,
      this.vp, this.dist, this.surface, this.t);

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
  /// Camera -> arm distance in metres from ARCore's tracked points on the arm, or -1 when too few points.
  final double dist;
  /// A point on the arm's visible surface (median of ARCore's tracked points on it), world frame, or null.
  final List<double>? surface;
  final int t;

  bool get visible => frac > 0.04 && cx >= 0;
  /// The arm itself is too dark to photograph. A bright window behind a well-lit arm is fine (the background is cut
  /// away; walking round the arm always passes the window): only a dark arm against bright light counts.
  bool get backlit => visible && (limbLum < 55 || (limbLum < 85 && bgLum > 2.0 * limbLum));
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
        source = m['src'] as String,
        seg = m['seg'] == null ? null : Map<String, Object>.from(m['seg'] as Map);

  final String path;
  final int width;
  final int height;
  final List<double> pose; // camera -> world, 4x4 column-major, metres
  final List<double> k; // fx, fy, cx, cy in pixels of the stored (portrait) JPEG
  final double sharpness;
  final int t;
  final String source; // arcore | arkit
  /// How the server should cut the limb out of this photo ({mode: auto, seed: [x, y]}: a point on the limb, 0..1 of
  /// the stored image). Set by the app from the placed arm cylinder; null = server default (skin).
  Map<String, Object>? seg;

  /// JSON sent with the upload (same fields the LimbScan server reads from WebXR captures).
  Map<String, Object> toMeta() =>
      {'pose': pose, 'K': k, 'w': width, 'h': height, 't': t, 'src': source, 'seg': ?seg};
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

  /// Start following (cutting out) the object at this screen point (0..1 of the view): results on [limbs].
  static Future<void> pickColour(double u, double v) => _ch.invokeMethod('pickColour', {'u': u, 'v': v});

  /// Look for hands on every frame (results on [hands]) until switched off or an object is tapped ([pickColour]).
  static Future<void> findHands(bool on) => _ch.invokeMethod('findHands', {'on': on});

  /// Hands seen in the camera (~10 Hz) while [findHands] is on.
  static Stream<HandsInfo> get hands => _hands ??= _events
      .where((e) => e is Map && e['type'] == 'hand')
      .map((e) => HandsInfo.fromMap(e as Map));
  static Stream<HandsInfo>? _hands;

  /// Stop the live cut-out (no more [limbs] events).
  static Future<void> clearColour() => _ch.invokeMethod('clearColour');

  /// The 3D point (AR world, metres) on the real surface seen at screen point (u, v) (0..1 of the view), or null
  /// when nothing was hit (ARCore depth / tracked points / planes).
  static Future<List<double>?> hitTest(double u, double v) async {
    final r = await _ch.invokeMethod<List>('hitTest', {'u': u, 'v': v});
    return r == null ? null : _vec(r);
  }

  /// Distance (m) from the camera along the viewing ray of each view point (u, v pairs, 0..1) from ARCore's depth
  /// map: one value per point, 0 where unknown; null when depth is not available.
  static Future<List<double>?> depth(List<double> uv) async {
    final r = await _ch.invokeMethod<List>('depth', {'uv': uv});
    return r?.map((e) => (e as num).toDouble()).toList();
  }

  /// The app's own recent log lines (Android logcat of this process).
  static Future<String?> readLog({int lines = 6000}) => _ch.invokeMethod<String>('readLog', {'lines': lines});

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


/// One hand from MediaPipe's hand landmarker: 21 landmarks (wrist 0, thumb 1-4, index 5-8, middle 9-12, ring 13-16,
/// little 17-20).
class HandInfo {
  HandInfo(this.lm, this.world, this.score);
  /// Landmarks in view coordinates (0..1): u0, v0, u1, v1, ...
  final List<double> lm;
  /// The hand's metric 3D shape (metres; camera axes x right, y down, z away; origin at the hand's centre).
  final List<double> world;
  final double score;
}

/// The hands seen in one camera frame, with that frame's view-projection matrix.
class HandsInfo {
  HandsInfo(this.hands, this.vp);
  factory HandsInfo.fromMap(Map m) => HandsInfo([
        for (final h in (m['hands'] as List? ?? const []))
          HandInfo(
            [for (final v in (h as Map)['lm'] as List) (v as num).toDouble()],
            [for (final v in h['world'] as List) (v as num).toDouble()],
            (h['score'] as num?)?.toDouble() ?? 0,
          ),
      ], [for (final v in (m['vp'] as List? ?? const [])) (v as num).toDouble()]);
  final List<HandInfo> hands;
  final List<double> vp;
}
