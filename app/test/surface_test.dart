import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:limb_capture/limb_capture.dart';
import 'package:limbscan/capture/surface.dart';

/// Perspective camera looking down -Z (OpenGL convention), column-major.
List<double> perspective(double f, double aspect, double n, double fa) => [
      f / aspect, 0, 0, 0, //
      0, f, 0, 0,
      0, 0, (fa + n) / (n - fa), -1,
      0, 0, 2 * fa * n / (n - fa), 0,
    ];

/// Column-major a x b.
List<double> mul(List<double> a, List<double> b) => [
      for (var c = 0; c < 4; c++)
        for (var r = 0; r < 4; r++) [for (var k = 0; k < 4; k++) a[k * 4 + r] * b[c * 4 + k]].reduce((x, y) => x + y),
    ];

List<double> lookAt(List<double> eye, List<double> target) {
  List<double> n(List<double> v) {
    final l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / l, v[1] / l, v[2] / l];
  }

  List<double> cross(List<double> a, List<double> b) =>
      [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  double dot(List<double> a, List<double> b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  final f = n([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  final s = n(cross(f, [0, 1, 0]));
  final u = cross(s, f);
  return [
    s[0], u[0], -f[0], 0, //
    s[1], u[1], -f[1], 0,
    s[2], u[2], -f[2], 0,
    -dot(s, eye), -dot(u, eye), dot(f, eye), 1,
  ];
}

/// A phone 35 cm from a vertical arm at the origin, at azimuth [deg] around it.
(List<double>, List<double>) camera(double deg) {
  final a = deg * math.pi / 180;
  final eye = [0.35 * math.sin(a), 0.0, 0.35 * math.cos(a)];
  return (eye, mul(perspective(1.5, 0.5, 0.03, 20), lookAt(eye, [0, 0, 0])));
}

/// The arm mask as seen from that camera: a vertical band in the middle of the screen.
LimbInfo mask(List<double> vp) {
  const w = 72, h = 128;
  final m = Uint8List(w * h);
  for (var j = 20; j < 108; j++) {
    for (var i = 24; i < 48; i++) {
      m[j * w + i] = 255;
    }
  }
  return LimbInfo.fromMap({
    'frac': 0.2, 'cx': 0.5, 'cy': 0.5, 'box': [0.33, 0.15, 0.67, 0.85], 'edge': 0.0, 'limbLum': 120.0, //
    'bgLum': 90.0, 'gw': w, 'gh': h, 'mask': m, 'vp': vp, 't': 0,
  });
}

void main() {
  /// Show the arm from every side (so all of it counts as "seen"), with photos from [photoDegs].
  SurfaceCoverage scan(Iterable<double> photoDegs) {
    final s = SurfaceCoverage(1)..setFrame([0, 0, 0]);
    for (final d in photoDegs) {
      final (eye, vp) = camera(d);
      s.addShot(eye, vp, 0);
    }
    for (var k = 0; k < 4; k++) {
      for (var d = 0.0; d < 360; d += 30) {
        s.shade(mask(camera(d).$2), 0, 0, record: true);
      }
    }
    return s;
  }

  test('photos all around the arm paint it all green', () {
    final s = scan([for (var d = 0.0; d < 360; d += 15) d]);
    expect(s.progress(0)!, greaterThan(SurfaceCoverage.done));
    expect(s.sideBins(0)!.every((b) => b == 1), isTrue);
  });

  test('photos from one side only leave the other side white', () {
    final s = scan([for (var d = 0.0; d <= 120; d += 15) d]);
    expect(s.progress(0)!, lessThan(0.6));
    final bins = s.sideBins(0)!;
    expect(bins[2], 1); // 30-45°: photographed
    expect(bins[14], 0); // 210-225°: never photographed
  });

  test('a single photo is not enough to call skin scanned', () {
    final s = scan([0]);
    expect(s.progress(0)!, 0);
  });

  test('the live mask is coloured green where scanned and white where not', () {
    // (colours are also smoothed over time, so each view gets its own coverage model here)
    final front = scan([for (var d = -45.0; d <= 45; d += 15) d]).shade(mask(camera(0).$2), 0, 0, record: false)!;
    final sb = scan([for (var d = -45.0; d <= 45; d += 15) d]);
    late Uint8List back;
    for (var k = 0; k < 10; k++) {
      back = sb.shade(mask(camera(180).$2), 0, 0, record: false)!; // the colour settles over a few frames
    }
    final i = (64 * 72 + 36) * 4; // centre pixel
    expect(front[i + 1] > front[i] + 60, isTrue, reason: 'green: G well above R'); // green
    expect((back[i + 1] - back[i]).abs(), lessThan(8), reason: 'white: R == G');
  });
}
