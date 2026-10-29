import 'dart:math' as math;

/// Capture guidance geometry, all in the AR world frame (metres, +Y up).
///
/// Ported from the web capture that was validated on real scans: the arm's position is estimated as the point
/// closest to all camera viewing rays (the phone always looks at the arm), coverage is counted as sides
/// AROUND THE ARM (not phone heading), near-duplicate views are skipped, and the user is told which way to go.

typedef Vec3 = List<double>;

double _dot(Vec3 a, Vec3 b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
double _dist(Vec3 a, Vec3 b) => math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2) + math.pow(a[2] - b[2], 2));

class Shot {
  Shot(this.pos, this.fwd, this.pass);
  final Vec3 pos;
  final Vec3 fwd;
  final int pass;
}

/// Least-squares point closest to all viewing rays.
Vec3? rayCentre(List<Shot> shots) {
  final a = List.generate(3, (_) => List.filled(3, 0.0));
  final b = List.filled(3, 0.0);
  for (final s in shots) {
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        final m = (i == j ? 1.0 : 0.0) - s.fwd[i] * s.fwd[j];
        a[i][j] += m;
        b[i] += m * s.pos[j];
      }
    }
  }
  double det(List<List<double>> m) =>
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  final d = det(a);
  if (d.abs() < 1e-9) return null;
  return List.generate(3, (k) => det(List.generate(3, (i) => List.generate(3, (j) => j == k ? b[i] : a[i][j]))) / d);
}

/// Azimuth (degrees, 0..360) of the phone around the arm, in the horizontal plane.
double azimuthAround(Vec3 c, Vec3 p) => (math.atan2(p[0] - c[0], p[2] - c[2]) * 180 / math.pi + 360) % 360;

/// Camera elevation in degrees (+ looking up, - looking down).
double elevation(Vec3 fwd) => math.asin(fwd[1].clamp(-1.0, 1.0)) * 180 / math.pi;

double angleDeg(Vec3 a, Vec3 b) => math.acos(_dot(a, b).clamp(-1.0, 1.0)) * 180 / math.pi;

/// A photo from (almost) the same place and direction adds nothing: skip it.
bool isDuplicate(List<Shot> shots, Vec3 pos, Vec3 fwd, {double minMove = 0.035, double minTurnDeg = 7}) {
  for (final s in shots) {
    if (_dist(pos, s.pos) < minMove && angleDeg(fwd, s.fwd) < minTurnDeg) return true;
  }
  return false;
}

/// Which way (left/right on screen) is the nearest side of the arm that has no photo yet?
String? nextDirection(List<int> bins, double azNow, Vec3 right, Vec3 c, Vec3 pos) {
  final n = bins.length;
  final ci = (azNow / (360 / n)).floor() % n;
  int? step;
  for (var k = 1; k <= n ~/ 2; k++) {
    if (bins[(ci + k) % n] == 0) { step = 1; break; }
    if (bins[(ci - k + n) % n] == 0) { step = -1; break; }
  }
  if (step == null) return null;
  final azR = azimuthAround(c, [pos[0] + 0.05 * right[0], pos[1] + 0.05 * right[1], pos[2] + 0.05 * right[2]]);
  final dR = ((azR - azNow + 540) % 360) - 180; // > 0: moving right increases azimuth
  return step * (dR == 0 ? 1 : dR.sign) > 0 ? 'right' : 'left';
}

/// One guided circle around the limb.
class Pass {
  const Pass(this.title, this.instruction, this.minElev, this.maxElev, this.minPhotos, this.coverage);
  final String title;
  final String instruction;
  final double minElev;
  final double maxElev;
  final int minPhotos;
  final double coverage; // share of the 24 sides around the arm needed
}

const uprightPasses = [
  Pass('Around the hand & wrist',
      'Phone at hand height, 30–40 cm away. Go slowly all the way around the hand and wrist.', -40, 30, 24, 0.7),
  Pass('Around the forearm',
      'Lower the phone to the forearm, tilt slightly down, and circle once more down to the elbow.', -60, 10, 20, 0.6),
];

const restingPasses = [
  Pass('From above', 'Phone 30–50 cm above the arm, looking down. Move all the way around it.', -90, -30, 26, 0.6),
  Pass('From the sides, lower', 'Lower the phone to table height and look at the sides of the arm.', -50, 15, 22, 0.5),
];

/// Coverage bookkeeping for one capture session.
class Coverage {
  Coverage(this.passes) : bins = List.generate(passes.length, (_) => List.filled(24, 0));

  final List<Pass> passes;
  final List<List<int>> bins;
  final List<Shot> shots = [];
  final Map<int, int> _photosAtLastNewSide = {};
  Vec3? centre;

  int photos(int pass) => shots.where((s) => s.pass == pass).length;

  void add(Shot s) {
    shots.add(s);
    if (shots.length >= 4) centre = rayCentre(shots) ?? centre;
    _recount(s.pass);
  }

  /// Recount sides with the current (improving) arm-centre estimate.
  void _recount(int pass) {
    final c = centre;
    if (c == null || shots.length < 6) return;
    final before = bins[pass].where((v) => v > 0).length;
    final b = List.filled(24, 0);
    for (final s in shots.where((s) => s.pass == pass)) {
      b[(azimuthAround(c, s.pos) / 15).floor() % 24]++;
    }
    bins[pass] = b;
    if (b.where((v) => v > 0).length > before) _photosAtLastNewSide[pass] = photos(pass);
  }

  double covered(int pass) => bins[pass].where((v) => v > 0).length / 24;

  /// 0..1 progress. A pass is complete when (nearly) the whole circle is covered, or - scanning alone - when
  /// at least the pass's minimum share is covered and ~10 photos in a row added no new side ("covered what you
  /// can reach"). Never just because a minimum share was reached: the user may still be going round.
  double progress(int pass) {
    final p = passes[pass];
    final n = photos(pass);
    if (centre == null || shots.length < 6) return math.min(0.15, n / p.minPhotos);
    final cov = covered(pass);
    if (n >= p.minPhotos && cov >= fullCircle) return 1;
    if (n >= p.minPhotos && cov >= p.coverage && n - (_photosAtLastNewSide[pass] ?? 0) >= 10) return 1;
    return math.min(0.97, math.min(cov / fullCircle, n / p.minPhotos));
  }

  static const fullCircle = 0.9; // 22 of 24 sides
}
