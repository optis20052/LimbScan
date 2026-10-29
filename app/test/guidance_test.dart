import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:limbscan/capture/guidance.dart';

/// Phone positions on a circle of radius r around an arm at c, all looking at it.
List<Shot> orbit(List<double> c, double r, Iterable<double> degs) => [
      for (final d in degs)
        () {
          final a = d * math.pi / 180;
          final p = [c[0] + r * math.sin(a), c[1], c[2] + r * math.cos(a)];
          final f = [c[0] - p[0], 0.0, c[2] - p[2]];
          final n = math.sqrt(f[0] * f[0] + f[2] * f[2]);
          return Shot(p, [f[0] / n, 0, f[2] / n], 0);
        }()
    ];

void main() {
  test('arm centre is recovered from the viewing rays', () {
    final shots = orbit([0.1, 1.2, -0.4], 0.35, [0, 20, 40, 60, 80, 100]);
    final c = rayCentre(shots)!;
    expect(c[0], closeTo(0.1, 1e-6));
    expect(c[2], closeTo(-0.4, 1e-6));
  });

  test('same spot is a duplicate, 10 cm away is not', () {
    final shots = orbit([0, 1, 0], 0.35, [0, 30]);
    expect(isDuplicate(shots, shots[1].pos, shots[1].fwd), isTrue);
    expect(isDuplicate(shots, [shots[1].pos[0] + 0.1, 1, shots[1].pos[2]], shots[1].fwd), isFalse);
  });

  test('coverage grows around the arm and a pass completes', () {
    final cov = Coverage(uprightPasses);
    for (final s in orbit([0, 1, 0], 0.35, [for (var d = 0; d < 360; d += 12) d.toDouble()])) {
      cov.add(s);
    }
    expect(cov.covered(0), greaterThan(0.9));
    expect(cov.progress(0), 1.0);
  });

  test('a pass does not end while the user is still going round', () {
    final cov = Coverage(uprightPasses);
    // 240 degrees, dense: enough photos and above the 70 % minimum, but every photo still adds new sides
    for (final s in orbit([0, 1, 0], 0.35, [for (var d = 0; d < 240; d += 8) d.toDouble()])) {
      cov.add(s);
    }
    expect(cov.covered(0), greaterThan(0.65));
    expect(cov.progress(0), lessThan(1.0));
  });

  test('covered what you can reach: completes after no new side for a while', () {
    final cov = Coverage(uprightPasses);
    final shots = orbit([0, 1, 0], 0.35, [for (var d = 0; d < 260; d += 8) d.toDouble()]);
    for (final s in shots) {
      cov.add(s);
    }
    // keep shooting the same sides from other heights: no new side
    for (var k = 0; k < 12; k++) {
      final s = shots[k * 2];
      cov.add(Shot([s.pos[0], s.pos[1] + 0.05 * (k % 3 + 1), s.pos[2]], s.fwd, 0));
    }
    expect(cov.progress(0), 1.0);
  });

  test('direction hint points to the uncovered side', () {
    final shots = orbit([0, 1, 0], 0.35, [0, 20, 40, 60, 80, 100, 120, 140]);
    final bins = List.filled(24, 0);
    for (final s in shots) {
      bins[(azimuthAround([0, 1, 0], s.pos) / 15).floor() % 24]++;
    }
    final last = shots.last;
    final right = [-last.fwd[2], 0.0, last.fwd[0]]; // camera right for a level phone
    expect(nextDirection(bins, azimuthAround([0, 1, 0], last.pos), right, [0, 1, 0], last.pos), isNotNull);
  });
}
