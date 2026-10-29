import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:limbscan/capture/ar_math.dart';
import 'package:limbscan/widgets/limb_overlay.dart';

/// Simple perspective camera at the origin looking down -Z (OpenGL convention), column-major.
List<double> perspective(double f, double aspect, double n, double fa) => [
      f / aspect, 0, 0, 0, //
      0, f, 0, 0,
      0, 0, (fa + n) / (n - fa), -1,
      0, 0, 2 * fa * n / (n - fa), 0,
    ];

void main() {
  final vp = perspective(1.5, 0.5, 0.03, 20);
  const size = Size(1000, 2000);

  test('a point straight ahead projects to the screen centre; behind the camera is null', () {
    final c = project(vp, [0, 0, -0.4], size)!;
    expect(c.dx, closeTo(500, 1e-6));
    expect(c.dy, closeTo(1000, 1e-6));
    expect(project(vp, [0, 0, 0.4], size), isNull);
  });

  test('unproject inverts project at the given distance', () {
    final p = unproject(vp, 0.3, 0.7, 0.35)!;
    final q = project(vp, p, size)!;
    expect(q.dx / size.width, closeTo(0.3, 1e-6));
    expect(q.dy / size.height, closeTo(0.7, 1e-6));
  });

  test('matrix inverse', () {
    final inv = invert(vp)!;
    // vp * inv = identity (check one column)
    double at(List<double> m, int r, int c) => m[c * 4 + r];
    for (var r = 0; r < 4; r++) {
      for (var c = 0; c < 4; c++) {
        var s = 0.0;
        for (var k = 0; k < 4; k++) {
          s += at(vp, r, k) * at(inv, k, c);
        }
        expect(s, closeTo(r == c ? 1 : 0, 1e-9));
      }
    }
  });

  test('marching squares outlines a filled block with sub-cell edges', () {
    const w = 10, h = 10;
    final g = Uint8List(w * h);
    for (var j = 3; j <= 6; j++) {
      for (var i = 3; i <= 6; i++) {
        g[j * w + i] = 255;
      }
    }
    final seg = marchingSquares(g, w, h, 128);
    expect(seg.length, greaterThan(8));
    expect(seg.length.isEven, isTrue);
    // edges sit halfway between filled (255) and empty (0) cells: x = (2.5 + 0.5) / 10 on the left side
    final xs = seg.map((p) => p.dx);
    expect(xs.reduce((a, b) => a < b ? a : b), closeTo(0.3, 0.02));
  });
}
