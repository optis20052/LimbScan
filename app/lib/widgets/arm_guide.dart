import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:limb_capture/limb_capture.dart';

import '../capture/ar_math.dart';
import '../capture/arm_model.dart';
import '../capture/guidance.dart' show azimuthAround;
import '../theme.dart';

const kGreen = Color(0xFF3DDC84);

/// The placed arm in 3D: a see-through cylinder fixed on the arm. Its side is split into the sectors of the level
/// ring and its top into the sectors of the ring from above; each turns green once a photo from that side exists.
/// When the arm is off screen an arrow on the screen edge points to it.
class ArmGuide extends StatelessWidget {
  const ArmGuide({
    super.key,
    required this.pose,
    required this.arm,
    required this.rings,
    required this.tick,
    this.target,
    this.pops = const {},
    this.placing = false,
  });

  final ValueListenable<ArPose?> pose;
  final ArmCylinder? Function() arm;
  final ScanRings rings;
  final Animation<double> tick;
  final (int, int)? Function()? target; // (ring, sector) to go to next
  final Map<(int, int), int> pops; // (ring, sector) -> time (ms) it was filled
  final bool placing; // just placed: show the cylinder as a blue "is this right?" outline

  @override
  Widget build(BuildContext context) => IgnorePointer(
    child: CustomPaint(size: Size.infinite, painter: _ArmPainter(this, Listenable.merge([pose, tick]))),
  );
}

class _ArmPainter extends CustomPainter {
  _ArmPainter(this.g, Listenable repaint) : super(repaint: repaint);
  final ArmGuide g;

  @override
  void paint(Canvas canvas, Size size) {
    final p = g.pose.value, arm = g.arm(), vp = p?.vp;
    if (p == null || arm == null || vp == null || !p.isTracking) return;
    final now = DateTime.now().millisecondsSinceEpoch;
    final t = g.tick.value;
    final pulse = (math.sin(t * 2 * math.pi) + 1) / 2;
    final mid = arm.mid;
    final q = project(vp, mid, size);
    final onScreen = q != null && q.dx > 0 && q.dy > 0 && q.dx < size.width && q.dy < size.height;
    if (!onScreen) {
      _edgeArrow(canvas, size, vp, mid, pulse);
      return;
    }
    // a vertical band around the arm: for an upright arm its full length, for a resting arm a band at its height
    final upright = (arm.b[1] - arm.a[1]).abs() > 0.5 * arm.length;
    final y0 = upright ? math.min(arm.a[1], arm.b[1]) : mid[1] - arm.r;
    final y1 = upright ? math.max(arm.a[1], arm.b[1]) : mid[1] + arm.r;
    final r = arm.r + 0.01;
    List<double> at(double deg, double y, [double rr = -1]) {
      final a = deg * math.pi / 180, k = rr < 0 ? r : rr;
      return [mid[0] + k * math.sin(a), y, mid[2] + k * math.cos(a)];
    }

    final dc = depth(vp, mid);
    final camAz = azimuthAround(mid, p.pos);
    final tg = g.target?.call();
    final blue = C.primary;

    // Colour per side lives on the rims (thick arcs top and bottom, per sector); the surface only gets a light
    // tint on the near side, so the arm itself stays visible through the cylinder.
    final n0 = ScanRings.sectors[ScanRings.level], step0 = 360 / n0;
    Color? stateColour(int ring, int i, {required bool rimLine}) {
      final c = g.rings.counts[ring][i];
      final pop = g.pops[(ring, i)];
      final popT = pop == null ? 1.0 : ((now - pop) / 500).clamp(0.0, 1.0);
      final isTarget = !g.placing && tg != null && tg.$1 == ring && tg.$2 == i;
      if (g.placing) return blue.withValues(alpha: rimLine ? 0.95 : 0.12);
      if (c > 0) return kGreen.withValues(alpha: rimLine ? 1.0 : 0.16 + 0.4 * (1 - popT));
      if (isTarget) return C.amber.withValues(alpha: rimLine ? 0.6 + 0.4 * pulse : 0.08 + 0.14 * pulse);
      return rimLine ? Colors.white.withValues(alpha: 0.75) : null;
    }

    for (var i = 0; i < n0; i++) {
      if (depth(vp, at((i + 0.5) * step0, mid[1])) > dc) continue; // the far side: rims only
      final fill = stateColour(ScanRings.level, i, rimLine: false);
      if (fill == null) continue;
      final pts = [
        for (var k = 0; k <= 3; k++) project(vp, at(i * step0 + k * step0 / 3, y1), size),
        for (var k = 3; k >= 0; k--) project(vp, at(i * step0 + k * step0 / 3, y0), size),
      ];
      if (pts.contains(null)) continue;
      canvas.drawPath(Path()..addPolygon(pts.cast<Offset>(), true), Paint()..color = fill);
    }
    // top cap (ring from above): light wedges
    final n1 = ScanRings.sectors[ScanRings.high], step1 = 360 / n1;
    for (var i = 0; i < n1; i++) {
      final fill = stateColour(ScanRings.high, i, rimLine: false);
      if (fill == null) continue;
      final pts = [
        project(vp, [mid[0], y1, mid[2]], size),
        for (var k = 0; k <= 4; k++) project(vp, at(i * step1 + k * step1 / 4, y1), size),
      ];
      if (pts.contains(null)) continue;
      canvas.drawPath(Path()..addPolygon(pts.cast<Offset>(), true), Paint()..color = fill.withValues(alpha: fill.a * 1.6));
    }
    // rims: the bottom one shows the ring around the arm, the top one the ring from above
    void rimArc(double y, double deg0, double deg1, Color col, bool far) {
      final path = Path();
      for (var k = 0; k <= 4; k++) {
        final o = project(vp, at(deg0 + (deg1 - deg0) * k / 4, y), size);
        if (o == null) return;
        k == 0 ? path.moveTo(o.dx, o.dy) : path.lineTo(o.dx, o.dy);
      }
      canvas.drawPath(
        path,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeCap = StrokeCap.butt
          ..strokeWidth = far ? 2.5 : 7
          ..color = col.withValues(alpha: col.a * (far ? 0.45 : 1)),
      );
    }

    for (final (ring, y) in [(ScanRings.level, y0), (ScanRings.high, y1)]) {
      final n = ScanRings.sectors[ring], step = 360 / n;
      for (var i = 0; i < n; i++) {
        final far = depth(vp, at((i + 0.5) * step, y)) > dc;
        rimArc(y, i * step + 0.8, (i + 1) * step - 0.8, stateColour(ring, i, rimLine: true)!, far);
      }
    }
    final rim = g.placing ? blue : Colors.white;
    // silhouette edges
    final camDir = (camAz) * math.pi / 180;
    for (final s in [-1.0, 1.0]) {
      final side = [math.cos(camDir) * s, math.sin(camDir) * -s]; // perpendicular to the view in the ground plane
      final top = project(vp, [mid[0] + r * side[0], y1, mid[2] + r * side[1]], size);
      final bot = project(vp, [mid[0] + r * side[0], y0, mid[2] + r * side[1]], size);
      if (top != null && bot != null) {
        canvas.drawLine(
          top,
          bot,
          Paint()
            ..strokeWidth = 1.5
            ..color = rim.withValues(alpha: 0.55),
        );
      }
    }
    if (!upright) _restingArm(canvas, size, vp, arm, rim);
    // you are here: a dot on the band facing the phone
    if (!g.placing) {
      final me = project(vp, at(camAz, (y0 + y1) / 2, r + 0.004), size);
      if (me != null) {
        canvas.drawCircle(me, 11, Paint()..color = Colors.black.withValues(alpha: 0.25));
        canvas.drawCircle(me, 7, Paint()..color = Colors.white);
      }
    }
  }

  /// A resting arm: its own cylinder outline along the table, under the coverage band.
  void _restingArm(Canvas canvas, Size size, List<double> vp, ArmCylinder arm, Color rim) {
    final a = project(vp, arm.a, size), b = project(vp, arm.b, size);
    if (a == null || b == null) return;
    canvas.drawLine(
      a,
      b,
      Paint()
        ..strokeWidth = 2
        ..color = rim.withValues(alpha: 0.6),
    );
    for (final e in [a, b]) {
      canvas.drawCircle(
        e,
        6,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = rim,
      );
    }
  }

  void _edgeArrow(Canvas canvas, Size size, List<double> vp, List<double> mid, double pulse) {
    final x = vp[0] * mid[0] + vp[4] * mid[1] + vp[8] * mid[2] + vp[12];
    final y = vp[1] * mid[0] + vp[5] * mid[1] + vp[9] * mid[2] + vp[13];
    final w = vp[3] * mid[0] + vp[7] * mid[1] + vp[11] * mid[2] + vp[15];
    var dx = x, dy = -y; // direction on screen (y down); behind the camera: flip
    if (w < 0) {
      dx = -dx;
      dy = -dy;
    }
    final n = math.sqrt(dx * dx + dy * dy);
    if (n < 1e-6) return;
    dx /= n;
    dy /= n;
    final c = Offset(size.width / 2, size.height / 2);
    final k = math.min((size.width / 2 - 44) / dx.abs().clamp(1e-6, 1), (size.height / 2 - 120) / dy.abs().clamp(1e-6, 1));
    final tip = c + Offset(dx, dy) * k;
    final u = Offset(dx, dy), nrm = Offset(-dy, dx);
    final s = 26.0 + 6 * pulse;
    final path = Path()
      ..moveTo((tip + u * s * 0.4).dx, (tip + u * s * 0.4).dy)
      ..lineTo((tip - u * s * 0.6 + nrm * s * 0.7).dx, (tip - u * s * 0.6 + nrm * s * 0.7).dy)
      ..lineTo((tip - u * s * 0.6 - nrm * s * 0.7).dx, (tip - u * s * 0.6 - nrm * s * 0.7).dy)
      ..close();
    canvas.drawPath(path, Paint()..color = C.amber);
  }

  @override
  bool shouldRepaint(covariant _ArmPainter o) => true;
}

/// A flat map of both rings seen from above, turned so the phone is always at the bottom: the outer ring is
/// "around the arm", the inner ring "from above". Green = done, amber = where to go, white dot = you.
class ScanDial extends StatelessWidget {
  const ScanDial({
    super.key,
    required this.pose,
    required this.arm,
    required this.rings,
    required this.tick,
    required this.active,
    this.target,
    this.size = 112,
  });

  final ValueListenable<ArPose?> pose;
  final ArmCylinder? Function() arm;
  final ScanRings rings;
  final Animation<double> tick;
  final int? Function() active; // ring the user should be doing now
  final (int, int)? Function()? target;
  final double size;

  @override
  Widget build(BuildContext context) => SizedBox(
    width: size,
    height: size,
    child: CustomPaint(painter: _DialPainter(this, Listenable.merge([pose, tick]))),
  );
}

class _DialPainter extends CustomPainter {
  _DialPainter(this.d, Listenable repaint) : super(repaint: repaint);
  final ScanDial d;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    final p = d.pose.value, arm = d.arm();
    final camAz = p == null || arm == null ? 0.0 : azimuthAround(arm.mid, p.pos);
    final sign = p == null || arm == null ? 1.0 : ScanRings.rightSign(arm.mid, p.pos, p.right);
    final elev = p == null || arm == null ? 0.0 : ScanRings.elevation(arm.mid, p.pos);
    final here = ScanRings.ringOf(elev);
    final act = d.active();
    final tg = d.target?.call();
    final pulse = (math.sin(d.tick.value * 2 * math.pi) + 1) / 2;
    // screen angle of a direction around the arm: the phone at the bottom, moving right = moving right on the dial
    double ang(double az) => (90 - sign * (az - camAz)) * math.pi / 180;
    final radii = [size.width / 2 - 8, size.width / 2 - 26];
    final widths = [11.0, 9.0];
    for (final ring in [ScanRings.level, ScanRings.high]) {
      final n = ScanRings.sectors[ring], step = 360 / n;
      final dim = act != null && act != ring;
      for (var i = 0; i < n; i++) {
        final cnt = d.rings.counts[ring][i];
        final isT = tg != null && tg.$1 == ring && tg.$2 == i;
        final col = cnt > 0
            ? kGreen
            : isT
                ? C.amber.withValues(alpha: 0.5 + 0.5 * pulse)
                : Colors.white.withValues(alpha: 0.22);
        final a0 = ang(i * step + 1.5), a1 = ang((i + 1) * step - 1.5);
        canvas.drawArc(
          Rect.fromCircle(center: c, radius: radii[ring]),
          math.min(a0, a1),
          (a1 - a0).abs(),
          false,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = widths[ring]
            ..color = col.withValues(alpha: col.a * (dim ? 0.45 : 1)),
        );
      }
    }
    // you: on the ring you are in (between the rings when too low / too high)
    final rr = here == null ? (radii[0] + radii[1]) / 2 : radii[here];
    final me = c + Offset(0, rr);
    canvas.drawCircle(me, 8, Paint()..color = Colors.black.withValues(alpha: 0.35));
    canvas.drawCircle(me, 6, Paint()..color = here == null ? C.amber : Colors.white);
    // the arm in the middle
    canvas.drawCircle(c, 7, Paint()..color = Colors.white.withValues(alpha: 0.85));
  }

  @override
  bool shouldRepaint(covariant _DialPainter o) => true;
}

/// What the eyes need while walking, drawn on the camera image around the hand so they never have to leave it: the walk
/// as a ring along the frame, around the hand (green: done, amber: next, white circle: where the phone is now), an arrow beside the hand
/// when a way to go is known, one short instruction below it, and the frame in the instruction's colour, which is
/// seen without looking at it (green: keep going, amber: read this).
class WalkHud extends StatelessWidget {
  const WalkHud({
    super.key,
    required this.arc,
    required this.tick,
    required this.pulse,
    required this.text,
    required this.icon,
    this.colour = Colors.white,
    this.arrow = 0,
    this.mirror = false,
  });

  final ScanArc arc;
  final Listenable tick;
  final Animation<double> pulse;
  final String text;
  final IconData icon;
  final Color colour;
  final int arrow; // -1 left, +1 right, 2 either way, 0 none
  final bool mirror; // walking left: the ring runs right to left, the way the phone really moves

  Widget _chevron(int side) => Align(
    alignment: Alignment(0.6 * side, 0),
    child: AnimatedBuilder(
      animation: pulse,
      builder: (_, child) => Transform.translate(offset: Offset(side * 8 * pulse.value, 0), child: child),
      child: Icon(
        side < 0 ? Icons.keyboard_double_arrow_left_rounded : Icons.keyboard_double_arrow_right_rounded,
        size: 56,
        color: colour,
        shadows: const [Shadow(blurRadius: 8, color: Colors.black87)],
      ),
    ),
  );

  @override
  Widget build(BuildContext context) => Stack(
    fit: StackFit.expand,
    children: [
      CustomPaint(painter: _ArcPainter(arc, pulse, mirror, Listenable.merge([tick, pulse]))),
      if (arrow == -1 || arrow == 2) _chevron(-1),
      if (arrow == 1 || arrow == 2) _chevron(1),
      Align(
        alignment: const Alignment(0, 0.66),
        child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 200),
          layoutBuilder: (current, previous) => current ?? const SizedBox.shrink(), // never two texts on top
          child: Container(
            key: ValueKey(text),
            margin: const EdgeInsets.symmetric(horizontal: 14),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
            decoration: BoxDecoration(color: Colors.black.withValues(alpha: 0.6), borderRadius: BorderRadius.circular(16)),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, color: colour, size: 24),
                const SizedBox(width: 8),
                Flexible(
                  child: Text(
                    text,
                    maxLines: 2,
                                        style: TextStyle(color: colour, fontSize: 18, fontWeight: FontWeight.w800, height: 1.15),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: colour, width: 4),
        ),
      ),
    ],
  );
}

class _ArcPainter extends CustomPainter {
  _ArcPainter(this.a, this.pulse, this.mirror, Listenable repaint) : super(repaint: repaint);
  final ScanArc a;
  final Animation<double> pulse;
  final bool mirror;
  static const _from = 150 * math.pi / 180, _sweep = 240 * math.pi / 180; // open at the bottom

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    // an oval along the frame: all around the hand, never over it (photos need the whole hand on screen)
    final rx = size.width / 2 - 18, ry = size.height / 2 - 18;
    Offset at(double f) {
      final t = _from + _sweep * (mirror ? 1 - f.clamp(0.0, 1.0) : f.clamp(0.0, 1.0));
      return c + Offset(rx * math.cos(t), ry * math.sin(t));
    }

    // over the camera image: everything has a dark edge, so it reads on skin, a white wall or a dark room alike
    final edge = Paint()..color = Colors.black.withValues(alpha: 0.45);
    final rect = Rect.fromCenter(center: c, width: 2 * rx, height: 2 * ry);
    canvas.drawArc(rect, _from, _sweep, false, Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..color = Colors.black.withValues(alpha: 0.3));
    canvas.drawArc(rect, _from, _sweep, false, Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5
      ..color = Colors.white.withValues(alpha: 0.45));
    final counts = a.dotCounts, next = a.nextDot;
    final free = a.walkDir == null; // no way chosen yet: the walk can start from either end of the ring
    for (var k = 0; k < ScanArc.dots; k++) {
      final p = at(k / (ScanArc.dots - 1));
      final end = k == 0 || k == ScanArc.dots - 1;
      final done = counts[k] >= ScanArc.need(k) && !free;
      canvas.drawCircle(p, 9.5, edge);
      if (done) {
        canvas.drawCircle(p, 8, Paint()..color = kGreen);
      } else if (free ? end : k == next) {
        canvas.drawCircle(p, 9 + 5 * pulse.value, Paint()..color = C.amber.withValues(alpha: 0.4));
        canvas.drawCircle(p, 8, Paint()..color = C.amber);
      } else {
        canvas.drawCircle(p, 6, Paint()..color = Colors.white.withValues(alpha: 0.6));
      }
      // photos so far at an open dot: it fills up green while the phone is there
      if (!done && !free && counts[k] > 0) {
        final slice = Rect.fromCircle(center: p, radius: 8);
        canvas.drawArc(slice, -math.pi / 2, 2 * math.pi * counts[k] / ScanArc.need(k), true, Paint()..color = kGreen);
      }
    }
    if (free) return;
    // you: where the phone is on the walk now
    final me = at(a.position / ScanArc.target);
    canvas.drawCircle(me, 12, Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..color = Colors.black.withValues(alpha: 0.5));
    canvas.drawCircle(me, 12, Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.white);
  }

  @override
  bool shouldRepaint(covariant _ArcPainter o) => true;
}
