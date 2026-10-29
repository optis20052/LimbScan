import 'dart:math' as math;
import 'dart:ui';

/// World <-> screen for AR overlays. `vp` = projection x view of the on-screen preview, 4x4 column-major
/// (as ARCore returns it), so screen positions match the camera image exactly.

/// World point -> screen pixels, or null when it is behind the camera.
Offset? project(List<double> vp, List<double> p, Size s) {
  final x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
  final y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
  final w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
  if (w <= 1e-4) return null;
  return Offset((x / w + 1) / 2 * s.width, (1 - y / w) / 2 * s.height);
}

/// Depth along the camera axis (clip w), used to fade the far side of 3D overlays.
double depth(List<double> vp, List<double> p) => vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];

/// Inverse of a 4x4 column-major matrix (Gauss-Jordan); null if singular.
List<double>? invert(List<double> m) {
  final a = List.generate(4, (r) => List.generate(8, (c) => c < 4 ? m[c * 4 + r] : (c - 4 == r ? 1.0 : 0.0)));
  for (var c = 0; c < 4; c++) {
    var piv = c;
    for (var r = c + 1; r < 4; r++) {
      if (a[r][c].abs() > a[piv][c].abs()) piv = r;
    }
    if (a[piv][c].abs() < 1e-12) return null;
    final t = a[c];
    a[c] = a[piv];
    a[piv] = t;
    final d = a[c][c];
    for (var k = 0; k < 8; k++) {
      a[c][k] /= d;
    }
    for (var r = 0; r < 4; r++) {
      if (r == c) continue;
      final f = a[r][c];
      if (f == 0) continue;
      for (var k = 0; k < 8; k++) {
        a[r][k] -= f * a[c][k];
      }
    }
  }
  return [for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) a[r][c + 4]];
}

List<double> _apply(List<double> m, double x, double y, double z) {
  final w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

/// The world point seen at view-normalized screen position (u, v), [dist] metres in front of the camera.
List<double>? unproject(List<double> vp, double u, double v, double dist) {
  final inv = invert(vp);
  if (inv == null) return null;
  final nx = 2 * u - 1, ny = 1 - 2 * v;
  final far = _apply(inv, nx, ny, 1);
  final near = _apply(inv, nx, ny, -1); // the near plane is ~3 cm in front of the camera
  final d = [far[0] - near[0], far[1] - near[1], far[2] - near[2]];
  final n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  if (n < 1e-9) return null;
  return [near[0] + d[0] / n * dist, near[1] + d[1] / n * dist, near[2] + d[2] / n * dist];
}
