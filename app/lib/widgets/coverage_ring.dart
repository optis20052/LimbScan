import 'dart:math' as math;

import 'package:flutter/material.dart';

/// Sides of the arm already photographed (green) and where the phone is now (white dot).
class CoverageRing extends StatelessWidget {
  const CoverageRing({super.key, required this.bins, required this.azimuth, required this.progress, this.size = 96});

  final List<int> bins;
  final double? azimuth;
  final double progress;
  final double size;

  @override
  Widget build(BuildContext context) => SizedBox(
        width: size,
        height: size,
        child: CustomPaint(painter: _RingPainter(bins, azimuth, progress)),
      );
}

class _RingPainter extends CustomPainter {
  _RingPainter(this.bins, this.azimuth, this.progress);

  final List<int> bins;
  final double? azimuth;
  final double progress;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    final r = size.width * 0.4;
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = size.width * 0.11;
    final n = bins.length;
    for (var i = 0; i < n; i++) {
      final a0 = i / n * 2 * math.pi - math.pi / 2;
      stroke.color = bins[i] > 0 ? const Color(0xF222C3A6) : const Color(0x33FFFFFF);
      canvas.drawArc(Rect.fromCircle(center: c, radius: r), a0 + 0.03, 2 * math.pi / n - 0.06, false, stroke);
    }
    if (azimuth != null) {
      final a = azimuth! / 360 * 2 * math.pi - math.pi / 2;
      canvas.drawCircle(c + Offset(math.cos(a), math.sin(a)) * r, size.width * 0.055, Paint()..color = Colors.white);
    }
    final tp = TextPainter(
      text: TextSpan(
          text: '${(progress * 100).round()}%',
          style: TextStyle(color: Colors.white, fontSize: size.width * 0.2, fontWeight: FontWeight.w800)),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, c - Offset(tp.width / 2, tp.height / 2));
  }

  @override
  bool shouldRepaint(covariant _RingPainter old) => true;
}
