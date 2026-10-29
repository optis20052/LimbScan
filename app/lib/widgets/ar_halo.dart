import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:limb_capture/limb_capture.dart';

import '../capture/ar_math.dart';
import '../capture/guidance.dart';
import '../theme.dart';

/// A ring floating in 3D around the arm (AR, anchored in the world): one segment per side of the arm. Sides
/// already photographed glow teal, the side the phone is on now is white, the next side to go to pulses amber.
/// Small dots mark where photos were taken. It is the coverage ring, but where the user is looking.
class ArHalo extends StatelessWidget {
  const ArHalo({
    super.key,
    required this.pose,
    required this.coverage,
    required this.pass,
    required this.pops,
    required this.tick,
    this.target,
  });
  final ValueListenable<ArPose?> pose;
  final Coverage coverage;
  final int pass;
  final Map<int, int> pops; // segment -> time (ms) a photo filled it, for the "pop" animation
  final Animation<double> tick;
  final int? target; // segment to go to next

  @override
  Widget build(BuildContext context) => IgnorePointer(
    child: CustomPaint(size: Size.infinite, painter: _HaloPainter(this, Listenable.merge([pose, tick]))),
  );
}

class _HaloPainter extends CustomPainter {
  _HaloPainter(this.h, Listenable repaint) : super(repaint: repaint);
  final ArHalo h;
  static const radius = 0.085; // metres: a bit wider than a forearm
  static const n = 24;

  @override
  void paint(Canvas canvas, Size size) {
    final p = h.pose.value;
    final c = h.coverage.centre;
    final vp = p?.vp;
    if (p == null || c == null || vp == null || !p.isTracking || h.coverage.shots.length < 6) return;
    final now = DateTime.now().millisecondsSinceEpoch;
    final bins = h.coverage.bins[h.pass];
    final az = azimuthAround(c, p.pos);
    final cur = (az / (360 / n)).floor() % n;
    final dc = depth(vp, c);
    final pulse = (math.sin(h.tick.value * 2 * math.pi) + 1) / 2;
    List<double> ringPoint(double deg) {
      final a = deg * math.pi / 180;
      return [c[0] + radius * math.sin(a), c[1], c[2] + radius * math.cos(a)];
    }

    // far half first, then near half on top
    final order = List.generate(n, (i) => i)
      ..sort((a, b) => depth(vp, ringPoint((b + 0.5) * 15)).compareTo(depth(vp, ringPoint((a + 0.5) * 15))));
    for (final i in order) {
      final path = Path();
      var ok = true;
      for (var k = 0; k <= 4; k++) {
        final q = project(vp, ringPoint(i * 15 + 1.5 + k * 3), size);
        if (q == null) {
          ok = false;
          break;
        }
        k == 0 ? path.moveTo(q.dx, q.dy) : path.lineTo(q.dx, q.dy);
      }
      if (!ok) continue;
      final far = depth(vp, ringPoint((i + 0.5) * 15)) > dc;
      final fade = far ? 0.45 : 1.0;
      final popT = h.pops[i] == null ? 1.0 : ((now - h.pops[i]!) / 500).clamp(0.0, 1.0);
      final covered = bins[i] > 0;
      Color colour;
      double width;
      if (i == cur) {
        colour = Colors.white;
        width = 7;
      } else if (covered) {
        colour = C.teal;
        width = 6;
      } else if (i == h.target) {
        colour = C.amber.withValues(alpha: 0.5 + 0.5 * pulse);
        width = 6 + 2 * pulse;
      } else {
        colour = Colors.white.withValues(alpha: 0.3);
        width = 3.5;
      }
      width += 8 * (1 - popT);
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeWidth = width
        ..color = colour.withValues(alpha: colour.a * fade);
      if (covered || i == cur || popT < 1) {
        canvas.drawPath(
          path,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = width + 8
            ..color = (popT < 1 ? Colors.white : colour).withValues(alpha: 0.3 * fade)
            ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 7),
        );
      }
      canvas.drawPath(path, paint);
    }
    // breadcrumbs: where photos were taken (in view only)
    final dot = Paint()..color = C.teal.withValues(alpha: 0.85);
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = Colors.white.withValues(alpha: 0.6);
    for (final s in h.coverage.shots) {
      final q = project(vp, s.pos, size);
      if (q == null || !size.contains(q)) continue;
      final d = depth(vp, s.pos);
      final r = (0.9 / d).clamp(2.0, 7.0);
      canvas.drawCircle(q, r, dot);
      canvas.drawCircle(q, r + 2, ring);
    }
  }

  @override
  bool shouldRepaint(covariant _HaloPainter o) => true;
}
