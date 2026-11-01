import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:limb_capture/limb_capture.dart';

import '../capture/ar_math.dart';
import '../theme.dart';

/// Live arm highlight over the camera preview.
///
/// The on-device segmentation (a 72x128 confidence grid over the preview) is drawn as a soft fill with a crisp,
/// sub-cell contour (marching squares) and a light sweep. The mask is lightly averaged over the last detections (no
/// flicker) and warped from its camera frame into the current one (perspective transform at the arm's measured
/// distance), so the outline stays on the arm while the phone moves.
class LimbOverlay extends StatefulWidget {
  const LimbOverlay({
    super.key,
    required this.limb,
    required this.pose,
    required this.locked,
    required this.armDistance,
    required this.sweep,
    this.shade,
  });
  final ValueListenable<LimbInfo?> limb;
  final ValueListenable<ArPose?> pose;
  final ValueListenable<bool> locked;
  final double Function() armDistance; // metres, camera -> arm (best current estimate)
  final Animation<double> sweep;
  /// Per-pixel colours for the mask (premultiplied RGBA on its grid), e.g. scanned skin in green; null = plain.
  final Uint8List? Function(LimbInfo l)? shade;

  @override
  State<LimbOverlay> createState() => _LimbOverlayState();
}

class _LimbOverlayState extends State<LimbOverlay> {
  ui.Image? _mask;
  List<Offset> _contour = const []; // segment endpoints, view-normalized, pairs
  LimbInfo? _shown;
  bool _painted = false; // the mask carries its own colours (scan progress)
  Float32List? _smooth; // confidence grid averaged over the last detections: a calm, steady outline
  int _smoothAt = 0;

  @override
  void initState() {
    super.initState();
    widget.limb.addListener(_onLimb);
  }

  @override
  void dispose() {
    widget.limb.removeListener(_onLimb);
    _mask?.dispose();
    super.dispose();
  }

  /// Each new detection counts half, the previous average the other half: the outline glides instead of jumping
  /// (the detector's edge flickers by a few pixels from frame to frame).
  Uint8List _temporal(LimbInfo l) {
    final n = l.mask.length;
    final now = DateTime.now().millisecondsSinceEpoch;
    final prev = _smooth;
    final fresh = prev == null || prev.length != n || now - _smoothAt > 500;
    final acc = fresh ? Float32List(n) : prev;
    for (var i = 0; i < n; i++) {
      acc[i] = fresh ? l.mask[i].toDouble() : 0.35 * acc[i] + 0.65 * l.mask[i];
    }
    _smooth = acc;
    _smoothAt = now;
    final out = Uint8List(n);
    for (var i = 0; i < n; i++) {
      out[i] = acc[i].round().clamp(0, 255);
    }
    return out;
  }

  void _onLimb() {
    final l0 = widget.limb.value;
    if (l0 == null || identical(l0, _shown)) return;
    final l = l0.withMask(_temporal(l0));
    final contour = marchingSquares(l.mask, l.gridW, l.gridH, 128);
    final shaded = l.visible ? widget.shade?.call(l) : null;
    final px = shaded ?? _alphaPixels(l.mask);
    ui.decodeImageFromPixels(px, l.gridW, l.gridH, ui.PixelFormat.rgba8888, (img) {
      if (!mounted) return img.dispose();
      _mask?.dispose();
      _mask = img;
      _shown = l;
      _contour = contour;
      _painted = shaded != null;
      setState(() {});
    });
  }

  @override
  Widget build(BuildContext context) => IgnorePointer(
    child: CustomPaint(
      size: Size.infinite,
      painter: _OverlayPainter(this, Listenable.merge([widget.pose, widget.sweep, widget.locked])),
    ),
  );
}

/// Confidence grid -> premultiplied white with alpha = confidence (thresholded later by a colour matrix).
Uint8List _alphaPixels(Uint8List mask) {
  final out = Uint8List(mask.length * 4);
  for (var i = 0; i < mask.length; i++) {
    final a = mask[i];
    out[4 * i] = a;
    out[4 * i + 1] = a;
    out[4 * i + 2] = a;
    out[4 * i + 3] = a;
  }
  return out;
}

/// 3x3 homography (row-major, h[8] = 1) mapping 4 points [src] onto [dst], or null if degenerate.
List<double>? homography(List<Offset> src, List<Offset> dst) {
  final a = List.generate(8, (_) => List.filled(9, 0.0));
  for (var i = 0; i < 4; i++) {
    final x = src[i].dx, y = src[i].dy, X = dst[i].dx, Y = dst[i].dy;
    a[2 * i] = [x, y, 1, 0, 0, 0, -x * X, -y * X, X];
    a[2 * i + 1] = [0, 0, 0, x, y, 1, -x * Y, -y * Y, Y];
  }
  for (var c = 0; c < 8; c++) {
    var piv = c;
    for (var r = c + 1; r < 8; r++) {
      if (a[r][c].abs() > a[piv][c].abs()) piv = r;
    }
    if (a[piv][c].abs() < 1e-9) return null;
    final t = a[c];
    a[c] = a[piv];
    a[piv] = t;
    for (var r = 0; r < 8; r++) {
      if (r == c) continue;
      final f = a[r][c] / a[c][c];
      for (var k = c; k < 9; k++) {
        a[r][k] -= f * a[c][k];
      }
    }
  }
  return [for (var r = 0; r < 8; r++) a[r][8] / a[r][r], 1.0];
}

/// Iso-contour of a grid at [level] as line segments (pairs of points), in 0..1 coordinates of the grid area.
List<Offset> marchingSquares(Uint8List g, int w, int h, int level) {
  final out = <Offset>[];
  Offset at(double i, double j) => Offset((i + 0.5) / w, (j + 0.5) / h);
  double t(int a, int b) => a == b ? 0.5 : ((level - a) / (b - a)).clamp(0.0, 1.0);
  for (var j = 0; j < h - 1; j++) {
    for (var i = 0; i < w - 1; i++) {
      final a = g[j * w + i], b = g[j * w + i + 1], c = g[(j + 1) * w + i + 1], d = g[(j + 1) * w + i];
      final idx = (a >= level ? 8 : 0) | (b >= level ? 4 : 0) | (c >= level ? 2 : 0) | (d >= level ? 1 : 0);
      if (idx == 0 || idx == 15) continue;
      final top = at(i + t(a, b), j.toDouble());
      final right = at(i + 1.0, j + t(b, c));
      final bottom = at(i + t(d, c), j + 1.0);
      final left = at(i.toDouble(), j + t(a, d));
      switch (idx) {
        case 1 || 14:
          out
            ..add(left)
            ..add(bottom);
        case 2 || 13:
          out
            ..add(bottom)
            ..add(right);
        case 3 || 12:
          out
            ..add(left)
            ..add(right);
        case 4 || 11:
          out
            ..add(top)
            ..add(right);
        case 5:
          out
            ..add(left)
            ..add(top)
            ..add(bottom)
            ..add(right);
        case 6 || 9:
          out
            ..add(top)
            ..add(bottom);
        case 7 || 8:
          out
            ..add(left)
            ..add(top);
        case 10:
          out
            ..add(top)
            ..add(right)
            ..add(left)
            ..add(bottom);
      }
    }
  }
  return out;
}


class _OverlayPainter extends CustomPainter {
  _OverlayPainter(this.s, Listenable repaint) : super(repaint: repaint);
  final _LimbOverlayState s;

  @override
  void paint(Canvas canvas, Size size) {
    final l = s._shown;
    final fresh = l != null && DateTime.now().millisecondsSinceEpoch - l.t < 900 && l.visible;
    final locked = s.widget.locked.value;
    final pose = s.widget.pose.value;
    if (!fresh) {
      _reticle(canvas, size, s.widget.sweep.value);
      return;
    }
    // the outline comes from a slightly older camera frame: map it into the current view (perspective warp for the
    // arm's measured distance), so it stays on the arm while the phone moves
    canvas.save();
    if (pose != null && pose.isTracking && pose.vp != null && l.vp != null) {
      final w = _motionWarp(l, pose.vp!, size, l.dist > 0 ? l.dist : s.widget.armDistance());
      if (w != null) canvas.transform(w.storage);
    }
    final rect = Offset.zero & size;
    final painted = s._painted;
    final colour = painted ? Colors.white : (locked ? const Color(0xFF3DDC84) : Colors.white);
    final m = s._mask;
    if (m != null) {
      final src = Rect.fromLTWH(0, 0, m.width.toDouble(), m.height.toDouble());
      // soft fill: smooth upsampled confidence, thresholded to a clean edge by a colour matrix
      canvas.saveLayer(rect, Paint()..color = Color.fromRGBO(0, 0, 0, painted ? 0.42 : 0.14)); // fill opacity
      canvas.saveLayer(rect, Paint()..colorFilter = painted ? _thresholdAlpha() : _threshold(colour));
      canvas.drawImageRect(m, src, rect, Paint()..filterQuality = FilterQuality.low);
      canvas.restore();
      canvas.restore();
      // light sweep travelling along the arm, clipped to the arm
      canvas.saveLayer(rect, Paint());
      canvas.saveLayer(rect, Paint()..colorFilter = _threshold(Colors.white));
      canvas.drawImageRect(m, src, rect, Paint()..filterQuality = FilterQuality.low);
      canvas.restore();
      final y = (s.widget.sweep.value * 1.6 - 0.3) * size.height;
      canvas.drawRect(
        rect,
        Paint()
          ..blendMode = BlendMode.srcIn
          ..shader = ui.Gradient.linear(
            Offset(0, y - 90),
            Offset(0, y + 90),
            [
              Colors.white.withValues(alpha: 0),
              Colors.white.withValues(alpha: 0.28),
              Colors.white.withValues(alpha: 0),
            ],
            [0, 0.5, 1],
          ),
      );
      canvas.restore();
    }
    // crisp contour + glow
    final pts = s._contour;
    if (pts.isNotEmpty) {
      final path = Path();
      for (var k = 0; k + 1 < pts.length; k += 2) {
        path
          ..moveTo(pts[k].dx * size.width, pts[k].dy * size.height)
          ..lineTo(pts[k + 1].dx * size.width, pts[k + 1].dy * size.height);
      }
      canvas.drawPath(
        path,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 9
          ..color = colour.withValues(alpha: painted ? 0.25 : 0.35)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6),
      );
      canvas.drawPath(
        path,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.6
          ..strokeCap = StrokeCap.round
          ..color = colour,
      );
    }
    canvas.restore();
  }

  /// Homography taking screen positions in the mask's frame to the current frame, for points at [dist] metres
  /// (fitted through the four corners of the arm's box); null when the motion is implausible.
  Matrix4? _motionWarp(LimbInfo l, List<double> vpNow, Size size, double dist) {
    final b = l.box;
    final src = <Offset>[], dst = <Offset>[];
    for (final (u, v) in [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]) {
      final world = unproject(l.vp!, u, v, dist);
      final q = world == null ? null : project(vpNow, world, size);
      if (q == null) return null;
      final p = Offset(u * size.width, v * size.height);
      if ((q - p).distance > size.shortestSide * 0.6) return null;
      src.add(p);
      dst.add(q);
    }
    if ((src[1] - src[0]).distance < 4 || (src[3] - src[0]).distance < 4) return null;
    final h = homography(src, dst);
    if (h == null) return null;
    return Matrix4(h[0], h[3], 0, h[6], h[1], h[4], 0, h[7], 0, 0, 1, 0, h[2], h[5], 0, 1);
  }

  /// Constant colour, alpha' = k * (alpha - 0.5) + 0.5 (clamped by the layer): a smooth but clean edge.
  ColorFilter _threshold(Color c) {
    const k = 7.0;
    return ColorFilter.matrix([
      0, 0, 0, 0, c.r * 255, //
      0, 0, 0, 0, c.g * 255,
      0, 0, 0, 0, c.b * 255,
      0, 0, 0, k, (0.5 - 0.5 * k) * 255,
    ]);
  }

  /// Same clean edge, keeping the per-pixel colours.
  ColorFilter _thresholdAlpha() {
    const k = 7.0;
    return const ColorFilter.matrix([
      1, 0, 0, 0, 0, //
      0, 1, 0, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, k, (0.5 - 0.5 * k) * 255,
    ]);
  }

  void _reticle(Canvas c, Size size, double t) {
    final centre = size.center(Offset.zero);
    final pulse = (math.sin(t * 2 * math.pi) + 1) / 2;
    final r = size.shortestSide * (0.22 + 0.02 * pulse);
    final p = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round
      ..color = Colors.white.withValues(alpha: 0.5 + 0.4 * pulse);
    for (var k = 0; k < 4; k++) {
      final a0 = k * math.pi / 2 + math.pi / 4 - 0.35 + t * 2 * math.pi * 0.25;
      c.drawArc(Rect.fromCircle(center: centre, radius: r), a0, 0.7, false, p);
    }
    c.drawCircle(centre, 4 + 2 * pulse, Paint()..color = Colors.white.withValues(alpha: 0.8));
  }

  @override
  bool shouldRepaint(covariant _OverlayPainter o) => true;
}

/// Big pulsing chevron on the screen edge: "go this way around the arm".
class EdgeArrow extends StatelessWidget {
  const EdgeArrow({super.key, required this.right, required this.pulse});
  final bool right;
  final Animation<double> pulse;

  @override
  Widget build(BuildContext context) => IgnorePointer(
    child: AnimatedBuilder(
      animation: pulse,
      builder: (_, _) => Transform.translate(
        offset: Offset((right ? 1 : -1) * 10 * pulse.value, 0),
        child: Container(
          width: 64,
          height: 120,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: right ? Alignment.centerRight : Alignment.centerLeft,
              end: right ? Alignment.centerLeft : Alignment.centerRight,
              colors: [C.primary.withValues(alpha: 0.55), C.primary.withValues(alpha: 0)],
            ),
          ),
          child: Icon(
            right ? Icons.chevron_right_rounded : Icons.chevron_left_rounded,
            size: 64,
            color: Colors.white.withValues(alpha: 0.7 + 0.3 * pulse.value),
          ),
        ),
      ),
    ),
  );
}

/// Vertical distance gauge: green zone 20–50 cm, marker at the current distance.
class DistanceGauge extends StatelessWidget {
  const DistanceGauge({super.key, required this.metres});
  final double? metres;

  @override
  Widget build(BuildContext context) {
    final m = metres;
    final ok = m != null && m >= 0.2 && m <= 0.5;
    return SizedBox(
      width: 46,
      height: 170,
      child: Column(
        children: [
          Text(
            m == null ? '– cm' : '${(m * 100).round()} cm',
            style: TextStyle(
              color: ok || m == null ? Colors.white : C.amber,
              fontSize: 12,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 6),
          Expanded(
            child: CustomPaint(size: Size.infinite, painter: _GaugePainter(m)),
          ),
        ],
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  _GaugePainter(this.m);
  final double? m;
  static const lo = 0.1, hi = 0.8; // gauge range, metres (top = far)

  double _y(double v, Size s) => s.height * (1 - ((v - lo) / (hi - lo)).clamp(0.0, 1.0));

  @override
  void paint(Canvas canvas, Size size) {
    final x = size.width / 2;
    final track = RRect.fromRectAndRadius(Rect.fromLTWH(x - 4, 0, 8, size.height), const Radius.circular(4));
    canvas.drawRRect(track, Paint()..color = const Color(0x40FFFFFF));
    final zone = Rect.fromLTRB(x - 4, _y(0.5, size), x + 4, _y(0.2, size));
    canvas.drawRRect(RRect.fromRectAndRadius(zone, const Radius.circular(4)), Paint()..color = C.teal);
    if (m != null) {
      final y = _y(m!, size);
      final ok = m! >= 0.2 && m! <= 0.5;
      canvas.drawCircle(Offset(x, y), 9, Paint()..color = ok ? Colors.white : C.amber);
      canvas.drawCircle(
        Offset(x, y),
        9,
        Paint()
          ..color = C.bg
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _GaugePainter o) => o.m != m;
}

/// Brief white frame flash each time a photo is kept.
class ShutterFlash extends StatelessWidget {
  const ShutterFlash({super.key, required this.count});
  final int count;

  @override
  Widget build(BuildContext context) => IgnorePointer(
    child: TweenAnimationBuilder<double>(
      key: ValueKey(count),
      tween: Tween(begin: count == 0 ? 0 : 1, end: 0),
      duration: const Duration(milliseconds: 350),
      builder: (_, v, _) => DecoratedBox(
        decoration: BoxDecoration(
          border: Border.all(color: Colors.white.withValues(alpha: 0.8 * v), width: 6),
        ),
        child: const SizedBox.expand(),
      ),
    ),
  );
}
