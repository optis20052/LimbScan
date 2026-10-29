import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme.dart';

/// Animated, code-drawn illustrations (sharp at any size, no image assets).
enum Scene { upright, resting, light, still, passHand, passForearm, passAbove, passSides }

class Illustration extends StatefulWidget {
  const Illustration(this.scene, {super.key, this.height = 200});
  final Scene scene;
  final double height;

  @override
  State<Illustration> createState() => _IllustrationState();
}

class _IllustrationState extends State<Illustration> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this, duration: const Duration(seconds: 4))..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => SizedBox(
        height: widget.height,
        child: AnimatedBuilder(
          animation: _c,
          builder: (_, _) => CustomPaint(painter: _ScenePainter(widget.scene, _c.value), size: Size.infinite),
        ),
      );
}

class _ScenePainter extends CustomPainter {
  _ScenePainter(this.scene, this.t);
  final Scene scene;
  final double t; // 0..1 animation phase

  final skin = Paint()..color = C.skin;
  final skinEdge = Paint()
    ..color = C.skinLine
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2;
  final table = Paint()..color = const Color(0xFF4A3A2E);

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width, h = size.height;
    canvas.save();
    // design canvas 320 x 200, scaled to fit and centred
    final s = math.min(w / 320, h / 200);
    canvas.translate((w - 320 * s) / 2, (h - 200 * s) / 2);
    canvas.scale(s);
    switch (scene) {
      case Scene.upright:
        _table(canvas);
        _uprightArm(canvas);
        _orbit(canvas, const Offset(160, 70), 74, 16);
      case Scene.resting:
        _table(canvas);
        _pillow(canvas);
        _restingArm(canvas);
        _orbit(canvas, const Offset(170, 118), 110, 26);
      case Scene.light:
        _light(canvas);
      case Scene.still:
        _table(canvas);
        _uprightArm(canvas);
        _still(canvas);
      case Scene.passHand:
        _table(canvas);
        _uprightArm(canvas);
        _orbit(canvas, const Offset(160, 58), 78, 16, highlight: true);
      case Scene.passForearm:
        _table(canvas);
        _uprightArm(canvas);
        _orbit(canvas, const Offset(160, 120), 78, 16, highlight: true, tilt: true);
      case Scene.passAbove:
        _table(canvas);
        _pillow(canvas);
        _restingArm(canvas);
        _orbit(canvas, const Offset(170, 82), 110, 22, highlight: true, tilt: true);
      case Scene.passSides:
        _table(canvas);
        _pillow(canvas);
        _restingArm(canvas);
        _orbit(canvas, const Offset(170, 122), 120, 18, highlight: true);
    }
    canvas.restore();
  }

  void _table(Canvas c) {
    c.drawRRect(RRect.fromRectAndRadius(const Rect.fromLTWH(20, 178, 280, 8), const Radius.circular(4)), table);
  }

  void _pillow(Canvas c) => c.drawOval(const Rect.fromLTWH(60, 150, 150, 28), Paint()..color = const Color(0xFF5B6E99));

  void _uprightArm(Canvas c) {
    c.drawOval(const Rect.fromLTWH(128, 168, 64, 12), Paint()..color = const Color(0xFF5B6E99)); // folded towel
    final fore = Path()
      ..moveTo(146, 172)
      ..cubicTo(144, 140, 142, 110, 145, 82)
      ..lineTo(175, 82)
      ..cubicTo(177, 110, 177, 140, 174, 172)
      ..close();
    final hand = Path()
      ..moveTo(143, 84)
      ..cubicTo(138, 68, 134, 54, 139, 42)
      ..lineTo(143, 18)
      ..cubicTo(144, 12, 150, 12, 150, 18)
      ..lineTo(152, 38)
      ..lineTo(154, 10)
      ..cubicTo(155, 4, 161, 4, 161, 10)
      ..lineTo(162, 38)
      ..lineTo(166, 12)
      ..cubicTo(167, 6, 173, 6, 173, 12)
      ..lineTo(172, 42)
      ..lineTo(178, 26)
      ..cubicTo(180, 20, 186, 22, 185, 28)
      ..lineTo(180, 58)
      ..cubicTo(179, 70, 177, 78, 175, 84)
      ..close();
    final thumb = Path()
      ..moveTo(143, 70)
      ..cubicTo(131, 62, 122, 56, 118, 50)
      ..cubicTo(115, 46, 119, 41, 124, 44)
      ..lineTo(142, 56);
    for (final p in [fore, hand, thumb]) {
      c.drawPath(p, skin);
      c.drawPath(p, skinEdge);
    }
  }

  void _restingArm(Canvas c) {
    final arm = Path()
      ..moveTo(40, 150)
      ..cubicTo(90, 144, 160, 142, 220, 146)
      ..cubicTo(240, 147, 262, 140, 286, 144)
      ..cubicTo(298, 146, 298, 158, 286, 160)
      ..cubicTo(262, 164, 240, 160, 220, 162)
      ..cubicTo(160, 166, 90, 168, 40, 166)
      ..close();
    c.drawPath(arm, skin);
    c.drawPath(arm, skinEdge);
  }

  /// Dashed orbit ellipse + a phone travelling along it, pointing at the centre.
  void _orbit(Canvas c, Offset centre, double rx, double ry, {bool highlight = false, bool tilt = false}) {
    final rect = Rect.fromCenter(center: centre, width: rx * 2, height: ry * 2);
    final dash = Paint()
      ..color = (highlight ? C.primary : C.primary.withValues(alpha: 0.7))
      ..style = PaintingStyle.stroke
      ..strokeWidth = highlight ? 4 : 3;
    const n = 36;
    for (var i = 0; i < n; i += 2) {
      final a0 = (i / n + t * 0.25) * 2 * math.pi, a1 = ((i + 1) / n + t * 0.25) * 2 * math.pi;
      c.drawArc(rect, a0, a1 - a0, false, dash);
    }
    final a = t * 2 * math.pi;
    final p = Offset(centre.dx + rx * math.cos(a), centre.dy + ry * math.sin(a));
    // draw the phone in front of / behind the arm depending on where it is on the orbit
    _phone(c, p, math.sin(a) > 0 ? 1.0 : 0.75, tilt ? 0.35 : 0);
  }

  void _phone(Canvas c, Offset p, double scale, double tiltDown) {
    c.save();
    c.translate(p.dx, p.dy);
    c.scale(scale);
    c.rotate(tiltDown);
    final body = RRect.fromRectAndRadius(Rect.fromCenter(center: Offset.zero, width: 20, height: 34), const Radius.circular(5));
    c.drawRRect(body, Paint()..color = C.primary);
    c.drawRRect(body, Paint()
      ..color = const Color(0xFFB9D3FF)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6);
    c.drawCircle(const Offset(0, -11), 2.6, Paint()..color = C.bg);
    c.restore();
  }

  void _still(Canvas c) {
    // "hold still" rings around the hand + moving phone left-right
    final pulse = (math.sin(t * 2 * math.pi) + 1) / 2;
    c.drawCircle(const Offset(160, 60), 52 + 6 * pulse,
        Paint()
          ..color = C.teal.withValues(alpha: 0.35 + 0.3 * pulse)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3);
    _phone(c, Offset(60 + 200 * ((math.sin(t * 2 * math.pi) + 1) / 2), 100), 1, 0);
  }

  void _light(Canvas c) {
    // good: lamp behind the phone holder lights the arm
    final lamp = Paint()..color = C.amber;
    c.drawPath(Path()
      ..moveTo(40, 30)
      ..lineTo(80, 30)
      ..lineTo(72, 52)
      ..lineTo(48, 52)
      ..close(), lamp);
    for (var i = 0; i < 6; i++) {
      c.drawLine(Offset(52 + i * 4.0, 54), Offset(100 + i * 12.0, 150),
          Paint()
            ..color = C.amber.withValues(alpha: 0.25 + 0.15 * math.sin(t * 2 * math.pi + i))
            ..strokeWidth = 2);
    }
    final arm = Path()
      ..moveTo(110, 150)
      ..cubicTo(130, 120, 140, 100, 150, 80)
      ..lineTo(166, 86)
      ..cubicTo(156, 106, 148, 126, 130, 156)
      ..close();
    c.drawPath(arm, skin);
    c.drawPath(arm, skinEdge);
    _check(c, const Offset(120, 180), true);
    // bad: window behind the arm -> silhouette
    c.drawRRect(RRect.fromRectAndRadius(const Rect.fromLTWH(212, 24, 84, 110), const Radius.circular(6)),
        Paint()..color = const Color(0xFFEAF2FF));
    final sil = Path()
      ..moveTo(236, 150)
      ..cubicTo(248, 120, 254, 100, 262, 78)
      ..lineTo(278, 84)
      ..cubicTo(270, 106, 262, 126, 252, 156)
      ..close();
    c.drawPath(sil, Paint()..color = const Color(0xFF3B2F2A));
    _check(c, const Offset(254, 180), false);
  }

  void _check(Canvas c, Offset p, bool ok) {
    c.drawCircle(p, 12, Paint()..color = ok ? C.teal : C.red);
    final ic = Paint()
      ..color = Colors.white
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    if (ok) {
      c.drawPath(Path()
        ..moveTo(p.dx - 5, p.dy)
        ..lineTo(p.dx - 1, p.dy + 4)
        ..lineTo(p.dx + 6, p.dy - 4), ic);
    } else {
      c.drawLine(p + const Offset(-5, -5), p + const Offset(5, 5), ic);
      c.drawLine(p + const Offset(5, -5), p + const Offset(-5, 5), ic);
    }
  }

  @override
  bool shouldRepaint(covariant _ScenePainter old) => old.t != t || old.scene != scene;
}
