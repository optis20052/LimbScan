import 'dart:math' as math;
import 'dart:typed_data';

import 'dart:ui' show Size;

import 'package:flutter_test/flutter_test.dart';
import 'package:limbscan/capture/ar_math.dart';
import 'package:limbscan/capture/arm_model.dart';

/// Perspective camera looking down -Z (OpenGL convention), column-major.
List<double> perspective(double f, double aspect, double n, double fa) => [
      f / aspect, 0, 0, 0, //
      0, f, 0, 0,
      0, 0, (fa + n) / (n - fa), -1,
      0, 0, 2 * fa * n / (n - fa), 0,
    ];

List<double> mul(List<double> a, List<double> b) => [
      for (var c = 0; c < 4; c++)
        for (var r = 0; r < 4; r++) a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3],
    ];

/// View matrix of a camera at the origin pitched down by [deg] (column-major).
List<double> pitchDown(double deg) {
  final a = deg * math.pi / 180, c = math.cos(a), s = math.sin(a);
  // rows = camera axes in world: x (1,0,0), y (0,c,-s), z = -forward (0,s,c)
  return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
}

const gw = 72, gh = 128;

/// Outline of a cylinder seen by the camera at the origin: (mask, front-surface point under the screen centre).
(Uint8List, List<double>?) render(List<double> vp, List<double> a, List<double> b, double r) {
  final m = Uint8List(gw * gh);
  final ax = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  final len = math.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
  final u = [ax[0] / len, ax[1] / len, ax[2] / len];
  double? hitT(List<double> d) {
    // |(t d - a) - ((t d - a).u) u| = r, 0 <= (t d - a).u <= len
    double dot(List<double> x, List<double> y) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    final du = dot(d, u), au = dot(a, u);
    final dp = [d[0] - du * u[0], d[1] - du * u[1], d[2] - du * u[2]];
    final ap = [-a[0] + au * u[0], -a[1] + au * u[1], -a[2] + au * u[2]];
    final qa = dot(dp, dp), qb = 2 * dot(dp, ap), qc = dot(ap, ap) - r * r;
    final disc = qb * qb - 4 * qa * qc;
    if (disc < 0) return null;
    final t = (-qb - math.sqrt(disc)) / (2 * qa);
    final along = t * du - au;
    return t > 0 && along >= 0 && along <= len ? t : null;
  }

  List<double> dir(double x, double y) {
    final p = unproject(vp, x, y, 1)!, o = unproject(vp, x, y, 0)!;
    final d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    final n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    return [d[0] / n, d[1] / n, d[2] / n];
  }

  for (var j = 0; j < gh; j++) {
    for (var i = 0; i < gw; i++) {
      if (hitT(dir((i + 0.5) / gw, (j + 0.5) / gh)) != null) m[j * gw + i] = 255;
    }
  }
  final d = dir(0.5, 0.5), t = hitT(d);
  return (m, t == null ? null : [d[0] * t, d[1] * t, d[2] * t]);
}

void main() {
  test('an upright arm: vertical axis, its height and radius come back from one outline', () {
    final vp = perspective(1.5, gw / gh, 0.03, 20);
    final a = [0.0, -0.12, -0.40], b = [0.0, 0.14, -0.40];
    final (mask, hit) = render(vp, a, b, 0.04);
    final c = ArmCylinder.fit(mask: mask, gw: gw, gh: gh, vp: vp, fwd: [0.0, 0, -1], hit: hit!, upright: true)!;
    expect(c.a[1], closeTo(-0.12, 0.015));
    expect(c.b[1], closeTo(0.14, 0.015));
    expect(c.mid[0], closeTo(0.0, 0.01));
    expect(c.mid[2], closeTo(-0.40, 0.015));
    expect(c.r, closeTo(0.04, 0.01));
  });

  test('an upright arm running off the bottom of the screen gets a typical hand + forearm length', () {
    final vp = perspective(1.5, gw / gh, 0.03, 20);
    final (mask, hit) = render(vp, [0.0, -0.6, -0.30], [0.0, 0.10, -0.30], 0.04);
    final c = ArmCylinder.fit(mask: mask, gw: gw, gh: gh, vp: vp, fwd: [0.0, 0, -1], hit: hit!, upright: true)!;
    expect(c.b[1], closeTo(0.10, 0.015));
    expect(c.length, closeTo(0.40, 0.03));
  });

  test('a resting arm: horizontal axis along the arm, a radius below the skin', () {
    final vp = mul(perspective(1.5, gw / gh, 0.03, 20), pitchDown(45));
    final a = [-0.12, -0.30, -0.30], b = [0.12, -0.30, -0.30];
    final (mask, hit) = render(vp, a, b, 0.04);
    final fwd = [0.0, -math.sin(math.pi / 4), -math.cos(math.pi / 4)];
    final c = ArmCylinder.fit(mask: mask, gw: gw, gh: gh, vp: vp, fwd: fwd, hit: hit!, upright: false)!;
    final dx = (c.b[0] - c.a[0]).abs(), dz = (c.b[2] - c.a[2]).abs();
    expect(dz / dx, lessThan(0.15)); // along x
    expect(c.length, closeTo(0.24, 0.05));
    expect(c.mid[1], closeTo(-0.30, 0.02));
    expect(c.mid[2], closeTo(-0.30, 0.03));
  });

  test('apparent half-width: an arm 4 cm wide at 40 cm reads as tan 0.1, upright or lying across the view', () {
    final vp = perspective(1.5, gw / gh, 0.03, 20);
    final (up, _) = render(vp, [0.0, -0.12, -0.40], [0.0, 0.14, -0.40], 0.04);
    expect(apparentHalfWidth(up, gw, gh, vp), closeTo(0.1, 0.012));
    final (across, _) = render(vp, [-0.2, 0.0, -0.40], [0.2, 0.0, -0.40], 0.04);
    expect(apparentHalfWidth(across, gw, gh, vp), closeTo(0.1, 0.012));
    final (far, _) = render(vp, [0.0, -0.3, -1.2], [0.0, 0.3, -1.2], 0.04);
    expect(apparentHalfWidth(far, gw, gh, vp)! * 0.4, lessThan(armHalfWidthMin)); // 40 cm is not where that arm is
  });

  test('depth gate: a cut-out that ran from the hand onto the body behind keeps only the hand', () {
    // hand: columns 30..41 at 0.6 m; touching it, a body: columns 10..60, rows 80..127 at 1.0 m
    final mask = Uint8List(gw * gh);
    final cells = <int>[], depth = <double>[];
    for (var j = 0; j < gh; j++) {
      for (var i = 0; i < gw; i++) {
        final hand = i >= 30 && i < 42 && j >= 20 && j < 90, body = i >= 10 && i < 61 && j >= 80;
        if (!hand && !body) continue;
        mask[j * gw + i] = 255;
        cells.add(j * gw + i);
        depth.add(hand ? 0.6 : 1.0);
      }
    }
    final g = depthGate(mask, gw, gh, cells, depth, 50 * gw + 35)!;
    expect(g.near, closeTo(0.6, 1e-9));
    var n = 0, outside = 0;
    for (var k = 0; k < g.mask.length; k++) {
      if (g.mask[k] == 0) continue;
      n++;
      final i = k % gw;
      if (i < 30 || i >= 42) outside++;
    }
    expect(n, 12 * 70);
    expect(outside, 0);
    expect(depthGate(mask, gw, gh, cells, List.filled(cells.length, 0.0), 50 * gw + 35), isNull); // no depth
  });

  test('hand: distance from its metric shape, cylinder down the forearm from the wrist', () {
    // an upright hand facing the camera, 40 cm away (camera at the origin looking down -Z, portrait view)
    final vp = perspective(1.5, 1080 / 1920, 0.03, 20);
    // landmarker metric shape (x right, y down, z away; metres): wrist at the bottom, fingers up
    final shape = <List<double>>[
      [0, 0.06, 0], // 0 wrist
      for (var k = 0; k < 4; k++) [-0.04 + 0.01 * k, 0.04 - 0.02 * k, 0.0], // 1-4 thumb
      for (final x in [-0.025, 0.0, 0.02, 0.035])
        for (var k = 0; k < 4; k++) [x, -0.02 - 0.025 * k, 0.005 * k], // 5-20 fingers
    ];
    final world = [for (final q in shape) ...q];
    final lm = <double>[];
    for (final q in shape) {
      final o = project(vp, [q[0], -q[1], -(0.40 + q[2])], const Size(1, 1))!;
      lm..add(o.dx)..add(o.dy);
    }
    expect(handDistance(lm, world, vp), closeTo(0.40, 0.01));
    final pts = handPoints(lm, world, vp, [0.0, 0, 0], 0.40)!;
    expect(pts[0][1], closeTo(-0.06, 0.005));
    final c = armFromHand(pts, world)!;
    expect(c.b[1], greaterThan(0.07)); // fingertips at the top
    expect(c.a[1], closeTo(-0.06 - 0.22, 0.01)); // elbow end below the wrist
    expect((c.a[0] - c.b[0]).abs(), lessThan(0.01));
    expect(c.r, closeTo(0.039, 0.003)); // 0.65 x the 6 cm knuckle span
  });

  test('hand track: joints triangulated from views around the hand recover a tilted arm', () {
    // a hand tilted 25 degrees towards the first camera (invisible from that view), wrist at (0, 0, -0.4)
    final tilt = 25 * math.pi / 180;
    final up = [0.0, math.cos(tilt), math.sin(tilt)]; // wrist -> fingers
    final side = [1.0, 0.0, 0.0];
    List<double> at(double s, double f) => [0 + side[0] * s + up[0] * f, side[1] * s + up[1] * f, -0.4 + side[2] * s + up[2] * f];
    final joints = <List<double>>[
      at(0, 0), // 0 wrist
      for (var k = 1; k <= 4; k++) at(-0.03 - 0.008 * k, 0.02 * k), // thumb
      for (final x in [-0.025, 0.0, 0.02, 0.035])
        for (var k = 0; k < 4; k++) at(x, 0.09 + 0.025 * k),
    ];
    List<double> lookAt(List<double> eye, List<double> target) {
      List<double> n(List<double> v) {
        final l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return [v[0] / l, v[1] / l, v[2] / l];
      }
      List<double> x(List<double> a, List<double> b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      final z = n([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
      final xx = n(x([0, 1, 0], z)), yy = x(z, xx);
      double d(List<double> a) => -(a[0] * eye[0] + a[1] * eye[1] + a[2] * eye[2]);
      return [xx[0], yy[0], z[0], 0, xx[1], yy[1], z[1], 0, xx[2], yy[2], z[2], 0, d(xx), d(yy), d(z), 1];
    }

    final proj = perspective(1.5, 1080 / 1920, 0.03, 20);
    final track = HandTrack();
    for (var deg = -60; deg <= 60; deg += 10) {
      final a = deg * math.pi / 180;
      final eye = [0.35 * math.sin(a), 0.05, -0.4 + 0.35 * math.cos(a)];
      final vp = mul(proj, lookAt(eye, [0.0, 0.05, -0.4]));
      final lm = <double>[];
      for (final j in joints) {
        final o = project(vp, j, const Size(1, 1))!;
        lm..add(o.dx)..add(o.dy);
      }
      track.add(lm, vp);
    }
    final arm = track.arm()!;
    final w = track.point(0)!;
    expect(w[0], closeTo(0, 0.002));
    expect(w[1], closeTo(0, 0.002));
    expect(w[2], closeTo(-0.4, 0.002));
    final ax = [arm.b[0] - arm.a[0], arm.b[1] - arm.a[1], arm.b[2] - arm.a[2]];
    final l = math.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
    expect(ax[1] / l, closeTo(math.cos(tilt), 0.01)); // the tilt one view could not see
    expect(ax[2] / l, closeTo(math.sin(tilt), 0.01));
  });

  test('rings: elevation picks the ring, sectors fill, a ring completes', () {
    expect(ScanRings.ringOf(0), ScanRings.level);
    expect(ScanRings.ringOf(45), ScanRings.high);
    expect(ScanRings.ringOf(-40), isNull);
    expect(ScanRings.ringOf(85), isNull);
    final r = ScanRings();
    for (var k = 0; k < 23; k++) {
      final az = k * 15.0 + 7;
      final a = az * math.pi / 180;
      expect(r.add(ScanRings.level, az, [0.35 * math.sin(a), 0.0, 0.35 * math.cos(a)], [0.0, 0, -1]), isTrue);
    }
    expect(r.complete(ScanRings.level), isFalse); // one grey sector left: not done
    r.add(ScanRings.level, 23 * 15.0 + 7, [0.0, 0, 0], [0.0, 0, -1]);
    expect(r.complete(ScanRings.level), isTrue);
    expect(r.full(ScanRings.level, 7), isFalse);
    r.add(ScanRings.level, 8, [0.0, 0, 0.5], [0.0, 0, -1]);
    expect(r.full(ScanRings.level, 7), isTrue);
  });

  test('a ring with sectors left never completes by itself (the user taps Finish)', () {
    final r = ScanRings();
    for (var k = 0; k < 15; k++) {
      r.add(ScanRings.level, k * 15.0 + 1, [0.0, 0, 0], [0.0, 0, -1]);
    }
    for (var k = 0; k < 20; k++) {
      r.add(ScanRings.level, 1, [0.0, 0, 0], [0.0, 0, -1]);
    }
    expect(r.complete(ScanRings.level), isFalse);
  });

  test('image point: sensor up is portrait right, sensor right is portrait down', () {
    // camera at the origin, sensor axes = world axes (looking down -Z), portrait K
    final pose = <double>[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    final k = [1000.0, 1000.0, 540.0, 960.0];
    final c = imagePoint(pose, k, 1080, 1920, [0.0, 0, -1])!;
    expect(c[0], closeTo(0.5, 1e-9));
    expect(c[1], closeTo(0.5, 1e-9));
    final up = imagePoint(pose, k, 1080, 1920, [0.0, 0.1, -1])!;
    expect(up[0], closeTo((540 + 100) / 1080, 1e-9));
    final right = imagePoint(pose, k, 1080, 1920, [0.1, 0.0, -1])!;
    expect(right[1], closeTo((960 + 100) / 1920, 1e-9));
    expect(imagePoint(pose, k, 1080, 1920, [0.0, 0, 1]), isNull);
  });

  group('ScanArc (walk around the hand, target dots)', () {
    List<double> at(double az) => [0.3 * math.sin(az * math.pi / 180), 0, 0.3 * math.cos(az * math.pi / 180)];
    List<double> look(double az) => [-math.sin(az * math.pi / 180), 0, -math.cos(az * math.pi / 180)];

    // walk from [a0] to [a1] (degrees, either way) in 1-degree samples 30 ms apart; a photo whenever one is wanted,
    // except where [blocked] says the photo was bad (blurry, hand cut off)
    int walk(ScanArc a, double a0, double a1, {int t0 = 0, bool Function(double)? blocked}) {
      final n = (a1 - a0).abs().round(), k = a1 > a0 ? 1 : -1;
      var t = t0;
      for (var i = 0; i <= n; i++) {
        final az = a0 + k * i;
        a.track(az % 360, t += 30);
        if (a.wants(0) && !(blocked?.call(az) ?? false)) a.add(0, at(az), look(az));
      }
      return t;
    }

    test('one good walk turns every dot green and completes, past 180 degrees without wrapping', () {
      final a = ScanArc();
      walk(a, 300, 300 + ScanArc.target + 5); // across 0 degrees
      expect(a.walkDir, 1);
      expect(a.furthest, closeTo(ScanArc.target + 5, 1));
      expect(a.dotsDone, ScanArc.dots);
      expect(a.complete, isTrue);
      expect(a.progress, 1);
    });

    test('a dot passed without good photos stays open and sends the user back to it', () {
      final a = ScanArc();
      // no usable photo between 50 and 75 degrees: the dot at 60 is missed
      var t = walk(a, 0, -120, blocked: (az) => az <= -50 && az >= -75);
      expect(a.walkDir, -1);
      expect(a.dotDone(3), isFalse);
      expect(a.nextDot, 3);
      expect(a.mustGoBack, isTrue);
      expect(a.complete, isFalse);
      t = walk(a, -120, -60, t0: t); // back to the amber dot
      expect(a.dotDone(3), isTrue);
      expect(a.mustGoBack, isFalse);
      expect(a.furthest, closeTo(120, 1)); // going back never lowers how far the walk got
    });

    test('walking either way from the start works the same', () {
      final a = ScanArc();
      walk(a, 90, 90 - ScanArc.target - 5);
      expect(a.walkDir, -1);
      expect(a.complete, isTrue);
    });
  });

  group('hand pose', () {
    // a flat hand: wrist at 0, fingers straight along y, spread in x
    List<double> hand({double bend = 0, double turn = 0}) {
      final pts = <List<double>>[[0, 0, 0]];
      for (var f = 0; f < 5; f++) {
        final x = (f - 2) * 0.02;
        for (var j = 1; j <= 4; j++) {
          final a = j >= 2 ? bend * (j - 1) : 0.0; // bend the finger joints by `bend` radians each
          pts.add([x, 0.03 + 0.02 * j * math.cos(a), 0.02 * j * math.sin(a)]);
        }
      }
      // the same hand seen from elsewhere: rotate about z
      final c = math.cos(turn), s_ = math.sin(turn);
      return [for (final p in pts) ...[c * p[0] - s_ * p[1], s_ * p[0] + c * p[1], p[2]]];
    }

    test('the same hand from another angle is the same pose', () {
      expect(poseChange(handPose(hand())!, handPose(hand(turn: 1.2))!), lessThan(1));
    });

    test('bent fingers are a different pose', () {
      expect(poseChange(handPose(hand())!, handPose(hand(bend: 0.6))!), greaterThan(20));
    });
  });

  group('walk dots', () {
    // the phone walks a circle of [radius] m round the hand at [speed] degrees a second, photos at most every [gap] s
    ({ScanArc arc, List<double?> greenAt, bool back}) walk(double radius, double speed, {double gap = 0.33}) {
      final arc = ScanArc();
      final greenAt = List<double?>.filled(ScanArc.dots, null);
      var last = -10.0, back = false;
      for (var t = 0.0; t < 40 && !arc.complete; t += 0.02) {
        final az = speed * t, a = az * math.pi / 180;
        final pos = [radius * math.sin(a), 0.0, radius * math.cos(a)];
        final fwd = [-math.sin(a), 0.0, -math.cos(a)];
        arc.track(az, (t * 1000).round());
        back = back || arc.mustGoBack;
        if (t - last >= gap && arc.wants(0) && !arc.duplicate(pos, fwd)) {
          arc.add(0, pos, fwd);
          last = t;
          for (var k = 0; k < ScanArc.dots; k++) {
            if (greenAt[k] == null && arc.dotDone(k)) greenAt[k] = arc.position - k * ScanArc.dotStep;
          }
        }
      }
      return (arc: arc, greenAt: greenAt, back: back);
    }

    for (final radius in [0.30, 0.38, 0.45]) {
      test('one walk at $radius m fills every dot, each about when the phone reaches it', () {
        final r = walk(radius, 11);
        expect(r.arc.complete, isTrue);
        expect(r.back, isFalse);
        // degrees past the dot when it turned green: 8 with photos 3.5 cm apart at every dot (scan 0dc8)
        expect(r.greenAt.skip(1).map((g) => g!).reduce(math.max), lessThan(6));
        expect(r.arc.photos, inInclusiveRange(43, 62));
      });
    }

    test('a slower walk turns a dot green as the phone reaches it', () {
      expect(walk(0.38, 6).greenAt.skip(1).map((g) => g!).reduce(math.max), lessThan(3));
    });
  });
}
