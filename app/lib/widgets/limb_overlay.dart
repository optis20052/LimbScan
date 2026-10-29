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
/// sub-cell contour (marching squares) and a light sweep. Segmentation lags the camera by ~100 ms, so the mask
/// is shifted by how the camera moved since that frame (AR poses), which keeps the glow glued to the arm.
class LimbOverlay extends StatefulWidget {
  const LimbOverlay({
    super.key,
    required this.limb,
    required this.pose,
    required this.locked,
    required this.armDistance,
    required this.sweep,
  });
  final ValueListenable<LimbInfo?> limb;
  final ValueListenable<ArPose?> pose;
  final ValueListenable<bool> locked;
  final double Function() armDistance; // metres, camera -> arm (best current estimate)
  final Animation<double> sweep;

  @override
  State<LimbOverlay> createState() => _LimbOverlayState();
}

class _LimbOverlayState extends State<LimbOverlay> {
  ui.Image? _mask;
  List<Offset> _contour = const []; // segment endpoints, view-normalized, pairs
  LimbInfo? _shown;
  List<double>? _anchor; // world point at the arm centre for the shown mask

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

  void _onLimb() {
    final l = widget.limb.value;
    if (l == null || identical(l, _shown)) return;
    final contour = marchingSquares(l.mask, l.gridW, l.gridH, 128);
    final px = _alphaPixels(l.mask);
    final anchor = (l.vp != null && l.visible) ? unproject(l.vp!, l.cx, l.cy, widget.armDistance()) : null;
    ui.decodeImageFromPixels(px, l.gridW, l.gridH, ui.PixelFormat.rgba8888, (img) {
      if (!mounted) return img.dispose();
      _mask?.dispose();
      _mask = img;
      _shown = l;
      _contour = contour;
      _anchor = anchor;
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

const debugShift = false;

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
    // follow the camera: where the arm centre of that (older) mask appears now
    var shift = Offset.zero;
    final a = s._anchor;
    if (a != null && pose?.vp != null && pose!.isTracking) {
      final now = project(pose.vp!, a, size);
      if (now != null) {
        shift = now - Offset(l.cx * size.width, l.cy * size.height);
        if (shift.distance > size.shortestSide * 0.3) shift = Offset.zero; // implausible: don't jump
      }
      if (debugShift && DateTime.now().millisecondsSinceEpoch % 1000 < 40) {
        // ignore: avoid_print
        print('LIMBDBG shift=${shift.dx.toStringAsFixed(1)},${shift.dy.toStringAsFixed(1)} c=${l.cx.toStringAsFixed(3)},${l.cy.toStringAsFixed(3)} track=${pose.isTracking}');
      }
    }
    canvas.save();
    canvas.translate(shift.dx, shift.dy);
    final rect = Offset.zero & size;
    final colour = locked ? C.teal : C.primary;
    final m = s._mask;
    if (m != null) {
      final src = Rect.fromLTWH(0, 0, m.width.toDouble(), m.height.toDouble());
      // soft fill: smooth upsampled confidence, thresholded to a clean edge by a colour matrix
      canvas.saveLayer(rect, Paint()..color = const Color.fromRGBO(0, 0, 0, 0.24)); // fill opacity
      canvas.saveLayer(rect, Paint()..colorFilter = _threshold(colour));
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
          ..color = colour.withValues(alpha: 0.35)
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
