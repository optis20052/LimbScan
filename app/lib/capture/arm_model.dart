import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'ar_math.dart';
import 'guidance.dart' show Vec3, azimuthAround, nextDirection, isDuplicate, Shot;

/// The arm while scanning: a cylinder placed ONCE in the AR world when the user taps the arm, then fixed. All
/// guidance (where to go, what is done, when to take a photo) is geometry against this cylinder and the phone's
/// pose: nothing depends on finding the arm in every camera frame again.

Vec3 _add(Vec3 a, Vec3 b, [double k = 1]) => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
Vec3 _sub(Vec3 a, Vec3 b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
double _dot(Vec3 a, Vec3 b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
double _len(Vec3 a) => math.sqrt(_dot(a, a));
Vec3 _unit(Vec3 a) {
  final n = _len(a);
  return n < 1e-9 ? [0, 0, 0] : [a[0] / n, a[1] / n, a[2] / n];
}

double _median(List<double> v) => ([...v]..sort())[v.length ~/ 2];

double _pct(List<double> v, double q) {
  final s = [...v]..sort();
  return s[((s.length - 1) * q).round()];
}

class ArmCylinder {
  const ArmCylinder(this.a, this.b, this.r);

  /// Axis end points (world, metres): upright arm a = elbow end (bottom), b = fingertips (top).
  final Vec3 a, b;
  final double r;

  Vec3 get mid => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  double get length => _len(_sub(b, a));
  Vec3 at(double t) => _add(a, _sub(b, a), t);

  /// Distance from a point to the axis segment.
  double distanceTo(Vec3 p) {
    final d = _sub(b, a);
    final l2 = _dot(d, d);
    final t = l2 < 1e-12 ? 0.0 : (_dot(_sub(p, a), d) / l2).clamp(0.0, 1.0);
    return _len(_sub(p, at(t)));
  }

  /// Whether the whole hand (the last [hand] metres of the axis, up to the fingertips) is on screen.
  bool handInView(List<double> vp, {double hand = 0.2, double margin = 0.04}) {
    final l = _len(_sub(b, a));
    final t0 = l < 1e-6 ? 0.0 : (1 - hand / l).clamp(0.0, 1.0);
    for (var k = 0; k <= 8; k++) {
      final q = project(vp, at(t0 + (1 - t0) * k / 8), const Size(1, 1));
      if (q == null || q.dx < margin || q.dx > 1 - margin || q.dy < margin || q.dy > 1 - margin) return false;
    }
    return true;
  }

  /// Share of the axis that is on screen (vp: the preview's world -> clip matrix).
  double visibleShare(List<double> vp, {double margin = 0.04}) {
    var n = 0;
    for (var k = 0; k <= 10; k++) {
      final q = project(vp, at(k / 10), const Size(1, 1));
      if (q != null && q.dx > margin && q.dx < 1 - margin && q.dy > margin && q.dy < 1 - margin) n++;
    }
    return n / 11;
  }

  static ArmCylinder median(List<ArmCylinder> fits) {
    double m(double Function(ArmCylinder c) f) => _median([for (final c in fits) f(c)]);
    return ArmCylinder(
      [m((c) => c.a[0]), m((c) => c.a[1]), m((c) => c.a[2])],
      [m((c) => c.b[0]), m((c) => c.b[1]), m((c) => c.b[2])],
      m((c) => c.r),
    );
  }

  /// Fit the cylinder to the arm's outline in one camera frame.
  ///
  /// [mask]: the tapped object's outline on a gw x gh grid over the screen (>= 128 = object), [vp]: world -> clip
  /// matrix of that frame, [fwd]: the camera's view direction, [hit]: the 3D point on the arm's surface
  /// under the tap. Upright arm: vertical axis through the arm's middle, from the outline's top to its bottom (or a
  /// typical hand + forearm length when the outline runs off the screen). Resting arm: horizontal axis along the
  /// outline, on the table.
  static ArmCylinder? fit({
    required Uint8List mask,
    required int gw,
    required int gh,
    required List<double> vp,
    required Vec3 fwd,
    required Vec3 hit,
    required bool upright,
  }) {
    // where the viewing ray through screen point (u, v) meets the plane (point o, normal n)
    Vec3? onPlane(double u, double v, Vec3 o, Vec3 n) {
      final p0 = unproject(vp, u, v, 0.0), p1 = unproject(vp, u, v, 1.0);
      if (p0 == null || p1 == null) return null;
      final d = _unit(_sub(p1, p0));
      final den = _dot(d, n);
      if (den.abs() < 1e-6) return null;
      final t = _dot(_sub(o, p0), n) / den;
      if (t <= 0 || t > 3) return null;
      return _add(p0, d, t);
    }

    bool on(int i, int j) => i >= 0 && j >= 0 && i < gw && j < gh && mask[j * gw + i] >= 128;
    if (upright) {
      final fh = _unit([fwd[0], 0, fwd[2]]);
      if (_len(fh) < 0.5) return null; // looking straight down: no side view
      final right = [fh[2], 0.0, -fh[0]];
      final rp = _unit(right);
      final o = _add(hit, fh, 0.03); // the axis is ~3 cm behind the skin facing the phone
      final ys = <double>[], mids = <double>[], halfW = <double>[];
      var bottomCut = false;
      for (var j = 0; j < gh; j++) {
        var i0 = -1, i1 = -1;
        for (var i = 0; i < gw; i++) {
          if (on(i, j)) {
            if (i0 < 0) i0 = i;
            i1 = i;
          }
        }
        if (i0 < 0 || i1 - i0 < 1) continue;
        if (j >= gh - 2) bottomCut = true;
        final l = onPlane((i0 + 0.5) / gw, (j + 0.5) / gh, o, fh);
        final r = onPlane((i1 + 0.5) / gw, (j + 0.5) / gh, o, fh);
        if (l == null || r == null) continue;
        ys.add((l[1] + r[1]) / 2);
        final xl = _dot(_sub(l, o), rp), xr = _dot(_sub(r, o), rp);
        mids.add((xl + xr) / 2);
        halfW.add((xr - xl).abs() / 2);
      }
      if (ys.length < 4) return null;
      final top = _pct(ys, 0.98);
      var bottom = _pct(ys, 0.02);
      if (bottomCut) bottom = math.min(bottom, top - 0.40); // the forearm goes on below the screen
      final len = (top - bottom).clamp(0.12, 0.50);
      bottom = top - len;
      final centre = _add(o, rp, _median(mids));
      final r = _median(halfW).clamp(0.03, 0.07);
      return ArmCylinder([centre[0], bottom, centre[2]], [centre[0], top, centre[2]], r);
    }
    // resting: the outline on the horizontal plane through the tapped point, its long direction = the arm
    final up = <double>[0, 1, 0];
    final pts = <Vec3>[];
    for (var j = 0; j < gh; j += 2) {
      for (var i = 0; i < gw; i += 2) {
        if (!on(i, j)) continue;
        final p = onPlane((i + 0.5) / gw, (j + 0.5) / gh, hit, up);
        if (p != null) pts.add(p);
      }
    }
    if (pts.length < 8) return null;
    final mx = _median([for (final p in pts) p[0]]), mz = _median([for (final p in pts) p[2]]);
    var sxx = 0.0, sxz = 0.0, szz = 0.0;
    for (final p in pts) {
      sxx += (p[0] - mx) * (p[0] - mx);
      sxz += (p[0] - mx) * (p[2] - mz);
      szz += (p[2] - mz) * (p[2] - mz);
    }
    final ang = 0.5 * math.atan2(2 * sxz, sxx - szz);
    final d = [math.cos(ang), 0.0, math.sin(ang)], nrm = [-math.sin(ang), 0.0, math.cos(ang)];
    final across = [for (final p in pts) (p[0] - mx) * nrm[0] + (p[2] - mz) * nrm[2]];
    final r = ((_pct(across, 0.9) - _pct(across, 0.1)) / 2).clamp(0.03, 0.07);
    // the axis: a radius behind the tapped skin along the view (the outline on a flat plane is biased when seen
    // at an angle); the outline gives only the arm's direction and its two ends
    final o = _add(hit, _unit(fwd), r);
    final along = [for (final p in pts) (p[0] - o[0]) * d[0] + (p[2] - o[2]) * d[2]];
    final t0 = _pct(along, 0.02), t1 = _pct(along, 0.98);
    final y = o[1];
    return ArmCylinder([o[0] + t0 * d[0], y, o[2] + t0 * d[2]], [o[0] + t1 * d[0], y, o[2] + t1 * d[2]], r);
  }
}

/// The arm's apparent half-width as the tangent of an angle, from its outline: the spread of the outline's viewing
/// rays across the outline's long direction. Its real half-width is ~4 cm, so this bounds how far away it can be.
double? apparentHalfWidth(Uint8List mask, int gw, int gh, List<double> vp) {
  final dirs = <Vec3>[];
  for (var j = 0; j < gh; j++) {
    for (var i = 0; i < gw; i++) {
      if (mask[j * gw + i] < 128) continue;
      final p0 = unproject(vp, (i + 0.5) / gw, (j + 0.5) / gh, 0.0), p1 = unproject(vp, (i + 0.5) / gw, (j + 0.5) / gh, 1.0);
      if (p0 != null && p1 != null) dirs.add(_unit(_sub(p1, p0)));
    }
  }
  if (dirs.length < 20) return null;
  final m = _unit(dirs.fold<Vec3>([0, 0, 0], (s, d) => _add(s, d)));
  var e1 = [m[2], 0.0, -m[0]];
  if (_len(e1) < 0.1) e1 = [1.0, 0, 0];
  e1 = _unit(e1);
  final e2 = [m[1] * e1[2] - m[2] * e1[1], m[2] * e1[0] - m[0] * e1[2], m[0] * e1[1] - m[1] * e1[0]];
  final xs = [for (final d in dirs) _dot(d, e1) / _dot(d, m)], ys = [for (final d in dirs) _dot(d, e2) / _dot(d, m)];
  final mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
  var sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (var k = 0; k < xs.length; k++) {
    sxx += (xs[k] - mx) * (xs[k] - mx);
    sxy += (xs[k] - mx) * (ys[k] - my);
    syy += (ys[k] - my) * (ys[k] - my);
  }
  final ang = 0.5 * math.atan2(2 * sxy, sxx - syy) + math.pi / 2; // the short direction
  final across = [for (var k = 0; k < xs.length; k++) xs[k] * math.cos(ang) + ys[k] * math.sin(ang)];
  return (_pct(across, 0.9) - _pct(across, 0.1)) / 1.6; // 10-90 % of a uniform width = 1.6 half-widths
}

/// The tapped outline cut down to what is at the arm's distance. A colour cut-out can run from the hand onto a body,
/// a face or a chair behind it (similar colour, touching on screen): those are much further away. [depth] holds the
/// distance (m, 0 = unknown) at each of the outline cells [cells]; the kept part is the connected piece through the
/// tapped cell within [thick] m of the tapped distance. Null when there is too little depth to tell.
({Uint8List mask, double near})? depthGate(Uint8List mask, int gw, int gh, List<int> cells, List<double> depth, int tap,
    {double thick = 0.12}) {
  final dAt = <int, double>{};
  for (var k = 0; k < cells.length && k < depth.length; k++) {
    if (depth[k] > 0.08 && depth[k] < 3) dAt[cells[k]] = depth[k];
  }
  if (dAt.length < math.max(20, cells.length * 0.3)) return null;
  final tx = tap % gw, ty = tap ~/ gw;
  // the tapped distance: depths in a small window around the tap (nearest ring that has any)
  double? near;
  for (var r = 2; r <= 12 && near == null; r += 2) {
    final w = <double>[
      for (var y = ty - r; y <= ty + r; y++)
        for (var x = tx - r; x <= tx + r; x++)
          if (x >= 0 && y >= 0 && x < gw && y < gh && dAt.containsKey(y * gw + x)) dAt[y * gw + x]!,
    ];
    if (w.length >= 3) near = _median(w);
  }
  near ??= _pct(dAt.values.toList(), 0.2);
  bool ok(int c) {
    final d = dAt[c];
    return d == null ? false : d > near! - 0.10 && d < near + thick;
  }

  // cells without a depth sample take their neighbours' verdict
  final keep = Uint8List(gw * gh);
  for (final c in cells) {
    if (dAt.containsKey(c)) {
      if (ok(c)) keep[c] = 1;
    } else {
      final x = c % gw, y = c ~/ gw;
      var yes = 0, no = 0;
      for (final (dx, dy) in const [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]) {
        final nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= gw || ny >= gh || !dAt.containsKey(ny * gw + nx)) continue;
        ok(ny * gw + nx) ? yes++ : no++;
      }
      if (yes > no) keep[c] = 1;
    }
  }
  // the connected piece through the tap (or the kept cell nearest to it)
  var seed = -1;
  var best = 1 << 30;
  for (var c = 0; c < keep.length; c++) {
    if (keep[c] == 0) continue;
    final dd = (c % gw - tx) * (c % gw - tx) + (c ~/ gw - ty) * (c ~/ gw - ty);
    if (dd < best) {
      best = dd;
      seed = c;
    }
  }
  if (seed < 0) return null;
  final out = Uint8List(gw * gh);
  final stack = [seed];
  out[seed] = 255;
  while (stack.isNotEmpty) {
    final c = stack.removeLast();
    final x = c % gw, y = c ~/ gw;
    for (final (dx, dy) in const [(1, 0), (-1, 0), (0, 1), (0, -1)]) {
      final nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
      final n = ny * gw + nx;
      if (keep[n] == 1 && out[n] == 0) {
        out[n] = 255;
        stack.add(n);
      }
    }
  }
  return (mask: out, near: near);
}

/// The arm's typical half-width (hand, wrist, forearm average) and the range a measured distance must agree with.
const armHalfWidth = 0.04, armHalfWidthMin = 0.028, armHalfWidthMax = 0.058;

/// A point on the limb in a stored photo (0..1 of the portrait JPEG), from the photo's camera pose (camera ->
/// world, column-major) and portrait intrinsics K = fx, fy, cx, cy: tells the server where the limb is.
List<double>? imagePoint(List<double> pose, List<double> k, int w, int h, Vec3 p) {
  final d = _sub(p, [pose[12], pose[13], pose[14]]);
  final x = d[0] * pose[0] + d[1] * pose[1] + d[2] * pose[2]; // sensor camera axes: x right, y up (landscape)
  final y = d[0] * pose[4] + d[1] * pose[5] + d[2] * pose[6];
  final z = -(d[0] * pose[8] + d[1] * pose[9] + d[2] * pose[10]);
  if (z <= 0.01) return null;
  // the stored JPEG is the landscape sensor image turned 90 degrees clockwise: sensor up -> portrait right
  final u = (k[2] + k[0] * y / z) / w, v = (k[3] + k[1] * x / z) / h;
  if (u < 0 || v < 0 || u > 1 || v > 1) return null;
  return [u, v];
}

/// Two rings of viewpoints around the arm: at its height (level) and from above. Each ring is split into sectors
/// by the phone's direction from the arm; a sector is done with one photo and full at [maxPerSector].
class ScanRings {
  static const level = 0, high = 1;
  static const sectors = [24, 12];
  static const need = [24, 12]; // every sector: done means the whole dial is green (unreachable ones: tap Finish)
  static const maxPerSector = 2;
  static const labels = ['around the arm', 'from above'];

  final counts = [List.filled(24, 0), List.filled(12, 0)];
  final List<Shot> shots = [];
  final _lastNew = [0, 0];
  final _photos = [0, 0];

  /// Which ring the phone is in from its elevation above the arm's middle (degrees), or null: too low / too high.
  static int? ringOf(double elev) => elev >= -20 && elev < 32
      ? level
      : elev >= 32 && elev <= 78
          ? high
          : null;

  static double elevation(Vec3 mid, Vec3 cam) =>
      math.atan2(cam[1] - mid[1], math.sqrt(math.pow(cam[0] - mid[0], 2) + math.pow(cam[2] - mid[2], 2))) *
      180 /
      math.pi;

  static int sectorOf(int ring, double az) => (az / (360 / sectors[ring])).floor() % sectors[ring];

  int done(int ring) => counts[ring].where((c) => c > 0).length;

  bool full(int ring, double az) => counts[ring][sectorOf(ring, az)] >= maxPerSector;

  bool duplicate(Vec3 pos, Vec3 fwd) => isDuplicate(shots, pos, fwd);

  /// Record a photo; returns true when it filled a new sector.
  bool add(int ring, double az, Vec3 pos, Vec3 fwd) {
    shots.add(Shot(pos, fwd, ring));
    _photos[ring]++;
    final s = sectorOf(ring, az);
    final isNew = counts[ring][s] == 0;
    counts[ring][s]++;
    if (isNew) _lastNew[ring] = _photos[ring];
    return isNew;
  }

  /// A ring is complete when every sector has a photo. The scan never ends by itself with grey sectors left: when
  /// some cannot be reached (wall, table), the user taps Finish.
  bool complete(int ring) => done(ring) >= need[ring];

  double share(int ring) => complete(ring) ? 1 : math.min(1, done(ring) / need[ring]);

  /// Overall progress 0..1 (the level ring carries most of the shape).
  double get progress => 0.65 * share(level) + 0.35 * share(high);

  bool get finished => complete(level) && complete(high);

  int get photos => shots.length;

  /// Nearest sector without a photo in [ring], and which way (left / right on screen) to move to get there.
  (int, String)? next(int ring, double az, Vec3 right, Vec3 mid, Vec3 pos) {
    final c = counts[ring], n = c.length, cur = sectorOf(ring, az);
    for (var k = 1; k <= n ~/ 2; k++) {
      for (final s in [(cur + k) % n, (cur - k + n) % n]) {
        if (c[s] == 0) {
          final dir = nextDirection(c, az, right, mid, pos);
          return dir == null ? null : (s, dir);
        }
      }
    }
    return null;
  }

  /// Screen-direction sign: +1 when moving the phone to the right increases its azimuth around the arm.
  static double rightSign(Vec3 mid, Vec3 pos, Vec3 right) {
    final az = azimuthAround(mid, pos);
    final azR = azimuthAround(mid, _add(pos, right, 0.05));
    final d = ((azR - az + 540) % 360) - 180;
    return d >= 0 ? 1 : -1;
  }
}

/// The scan as one walk around the hand (person seated, elbow on the table, hand up), from where the scan starts
/// (the first dot) round the front to the other side: [dots] target points every [dotStep] degrees. A dot is done
/// (green) only when [perDot] good photos were taken from within [dotStep]/2 of it; a dot passed without them stays
/// open and the user is sent back to it. Done when every dot is done. Photos are taken every [spacing] degrees the
/// phone moves (or [lift] degrees of height change), only when the whole hand is on screen and sharp.
class ScanArc {
  static const spacing = 2.5, lift = 12.0;
  static const dotStep = 20.0, dots = 11, perDot = 4; // 0..200 degrees; 44+ photos (22 were too few for the model)
  static const target = dotStep * (dots - 1), maxPhotos = 100;

  final List<Shot> shots = [];
  final List<double> _shotPositions = [];

  double? _last; // last azimuth seen (degrees)
  int _lastT = 0;
  double _u = 0; // unwrapped azimuth change since the start (degrees, signed)
  int? walkDir; // +1 / -1 once the phone has gone 15 degrees one way; fixed after that
  double furthest = 0; // furthest point of the walk so far (degrees)
  double? _shotPos, _shotEl; // where the last photo was taken

  static double diff(double a, double b) => (((a - b) % 360) + 540) % 360 - 180;

  /// Where the phone is on the walk now (degrees from the start, in the walking direction).
  double get position => walkDir == null ? _u.abs() : walkDir! * _u;

  /// Follow the phone round the hand: call with its azimuth (looking at the hand) whenever the hand is in view.
  /// [t] in ms. Small steps between samples are added up (no wrap at 180 degrees); after the hand was out of view the
  /// whole change counts.
  void track(double az, int t) {
    final last = _last;
    if (last != null) {
      var d = diff(az, last);
      if (t - _lastT < 500) d = d.clamp(-15.0, 15.0);
      _u += d;
    }
    _last = az;
    _lastT = t;
    if (walkDir == null && _u.abs() >= 15) walkDir = _u > 0 ? 1 : -1;
    furthest = math.max(furthest, position);
  }

  /// The dot a position belongs to, or null outside the walk.
  static int? dotAt(double pos) {
    final k = (pos / dotStep).round();
    return k < 0 || k >= dots ? null : k;
  }

  /// Photos a dot needs. The walk starts at the first dot, so only the half of it ahead of the phone can get any.
  static int need(int k) => k == 0 ? perDot - 1 : perDot;

  /// Good photos taken at each dot.
  List<int> get dotCounts {
    final c = List.filled(dots, 0);
    for (final p in _shotPositions) {
      final k = dotAt(p);
      if (k != null) c[k]++;
    }
    return c;
  }

  bool dotDone(int k) => dotCounts[k] >= need(k);

  /// The first dot still open (the one to go to), or null when all are done.
  int? get nextDot {
    final c = dotCounts;
    for (var k = 0; k < dots; k++) {
      if (c[k] < need(k)) return k;
    }
    return null;
  }

  /// The phone has walked past an open dot: it must go back.
  bool get mustGoBack {
    final k = nextDot;
    return k != null && position > k * dotStep + 1.5 * dotStep; // well past it, not just stepping beyond it
  }

  /// A photo here adds a view: the phone moved [spacing] degrees round the hand or [lift] degrees up or down since
  /// the last one; at an open dot, a photo whenever it still needs one.
  bool wants(double el) {
    if (photos >= maxPhotos) return false;
    final k = dotAt(position);
    if (k != null && !dotDone(k) && (_shotPos == null || (position - _shotPos!).abs() >= 1.0)) return true;
    return _shotPos == null || (position - _shotPos!).abs() >= spacing || (el - _shotEl!).abs() >= lift;
  }

  /// A photo here repeats one already taken. Closer photos are allowed at an open dot: 3.5 cm apart, under 4 fit in a
  /// dot at 38 cm (scan 0dc8), so each turned green 8 degrees past it and any dot left one short meant walking back.
  bool duplicate(Vec3 pos, Vec3 fwd) {
    final k = dotAt(position);
    return isDuplicate(shots, pos, fwd, minMove: k != null && !dotDone(k) ? 0.02 : 0.035);
  }

  void add(double el, Vec3 pos, Vec3 fwd) {
    shots.add(Shot(pos, fwd, 0));
    _shotPositions.add(position);
    _shotPos = position;
    _shotEl = el;
  }

  int get photos => shots.length;
  int get dotsDone => [for (var k = 0; k < dots; k++) if (dotDone(k)) k].length;
  double get progress => dotsDone / dots;
  bool get complete => nextDot == null;
  bool get canFinish => dotsDone >= dots - 3 && photos >= 20;
}

// ---- the arm from a hand found by MediaPipe's hand landmarker ---------------------------------------------------

Vec3? _ray(List<double> vp, double u, double v) {
  final p0 = unproject(vp, u, v, 0.0), p1 = unproject(vp, u, v, 1.0);
  return p0 == null || p1 == null ? null : _unit(_sub(p1, p0));
}

/// Pairs of landmarks across the palm and along it (wrist, knuckles): stiff, so the hand's shape there is reliable.
const _palmPairs = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17), (5, 17), (5, 13), (9, 17)];

/// Distance (m) from the camera to the hand: the landmarker's metric shape of the hand (camera axes) against the
/// angles between the viewing rays of the same landmarks. Both are the hand as seen from the camera, so a tilted
/// hand (foreshortened on screen) is foreshortened in both and the ratio still holds.
double? handDistance(List<double> lm, List<double> world, List<double> vp) {
  final ds = <double>[];
  for (final (i, j) in _palmPairs) {
    final a = _ray(vp, lm[2 * i], lm[2 * i + 1]), b = _ray(vp, lm[2 * j], lm[2 * j + 1]);
    if (a == null || b == null) continue;
    final ang = math.acos(_dot(a, b).clamp(-1.0, 1.0));
    final lateral = math.sqrt(math.pow(world[3 * i] - world[3 * j], 2) + math.pow(world[3 * i + 1] - world[3 * j + 1], 2));
    if (ang < 1e-4 || lateral < 0.01) continue;
    ds.add(lateral / (2 * math.tan(ang / 2)));
  }
  return ds.length < 4 ? null : _median(ds);
}

/// The hand's landmarks in the AR world, [dist] metres from the camera at [cam] (the hand's centre), each moved by
/// its own depth in the landmarker's metric shape.
List<Vec3>? handPoints(List<double> lm, List<double> world, List<double> vp, Vec3 cam, double dist) {
  final out = <Vec3>[];
  for (var i = 0; i < 21; i++) {
    final d = _ray(vp, lm[2 * i], lm[2 * i + 1]);
    if (d == null) return null;
    out.add(_add(cam, d, dist + world[3 * i + 2]));
  }
  return out;
}

/// The arm's cylinder from the hand: its axis runs through the wrist along the middle finger's line (the forearm
/// continues it), from the fingertips to [forearm] m below the wrist; its radius from the palm's width.
ArmCylinder? armFromHand(List<Vec3> p, List<double> world, {double forearm = 0.22}) {
  final u = _unit(_sub(p[0], p[9])); // knuckles -> wrist: down the forearm
  if (_len(_sub(p[0], p[9])) < 0.02) return null;
  var tip = 0.0;
  for (final q in p) {
    tip = math.max(tip, -_dot(_sub(q, p[0]), u));
  }
  final palm = math.sqrt(math.pow(world[15] - world[51], 2) + math.pow(world[16] - world[52], 2) + math.pow(world[17] - world[53], 2));
  final r = (0.65 * palm).clamp(0.03, 0.06);
  return ArmCylinder(_add(p[0], u, forearm), _add(p[0], u, -(tip + 0.01)), r);
}


/// The arm's outline in a stored photo: points around the cylinder (8 rings of 12, 1.5 cm wider than the arm)
/// projected into the photo (0..1). The server keeps only the cut-out inside their convex hull, so a body, a chair or
/// a table the cut-out ran onto is dropped. Points behind the camera are left out.
List<List<double>> imageOutline(List<double> pose, List<double> k, int w, int h, ArmCylinder arm) {
  final ax = _unit(_sub(arm.b, arm.a));
  var e1 = [ax[1], -ax[0], 0.0];
  if (_len(e1) < 0.1) e1 = [0.0, ax[2], -ax[1]];
  e1 = _unit(e1);
  final e2 = [ax[1] * e1[2] - ax[2] * e1[1], ax[2] * e1[0] - ax[0] * e1[2], ax[0] * e1[1] - ax[1] * e1[0]];
  final r = arm.r + 0.015;
  final out = <List<double>>[];
  for (var i = 0; i <= 7; i++) {
    final c = arm.at(i / 7);
    for (var j = 0; j < 12; j++) {
      final a = j * math.pi / 6;
      final q = _add(_add(c, e1, r * math.cos(a)), e2, r * math.sin(a));
      final uv = imagePoint(pose, k, w, h, q);
      if (uv != null) out.add([double.parse(uv[0].toStringAsFixed(4)), double.parse(uv[1].toStringAsFixed(4))]);
    }
  }
  return out;
}

/// A point inside the hand (about the palm's middle), for the server's cut-out to start from.
Vec3 palmPoint(ArmCylinder arm) => _add(arm.b, _unit(_sub(arm.a, arm.b)), math.min(0.09, arm.length / 2));

/// The hand's landmarks seen from many phone positions while scanning: each sighting is a viewing ray per landmark.
/// Rays from different sides meet at the real joint, so the 3D hand (and the arm's cylinder) is no longer a guess
/// from one view, whose tilt towards or away from the camera cannot be seen.
class HandTrack {
  final List<List<(Vec3, Vec3)>> _views = []; // per sighting: 21 rays (origin, unit direction)
  Vec3? _lastDir;

  int get views => _views.length;

  /// Adds a sighting (landmarks in view coordinates, the frame's view-projection) when the phone has moved on.
  bool add(List<double> lm, List<double> vp) {
    final rays = <(Vec3, Vec3)>[];
    for (var i = 0; i < 21; i++) {
      final p0 = unproject(vp, lm[2 * i], lm[2 * i + 1], 0.0), p1 = unproject(vp, lm[2 * i], lm[2 * i + 1], 1.0);
      if (p0 == null || p1 == null) return false;
      rays.add((p0, _unit(_sub(p1, p0))));
    }
    final d = rays[0].$2;
    final last = _lastDir;
    if (last != null && _dot(last, d) > math.cos(4 * math.pi / 180)) return false; // same view: no new information
    _lastDir = d;
    _views.add(rays);
    if (_views.length > 80) _views.removeAt(0);
    return true;
  }

  /// The point nearest to all rays of landmark [i] (least squares), with outlier rays (> 2.5 cm off) dropped once.
  /// Null while the rays come from too narrow a spread of directions to pin the depth.
  Vec3? point(int i) {
    if (_views.length < 4) return null;
    var rays = [for (final v in _views) v[i]];
    var spread = 0.0;
    for (final r in rays) {
      spread = math.max(spread, math.acos(_dot(r.$2, rays.first.$2).clamp(-1.0, 1.0)));
    }
    if (spread < 20 * math.pi / 180) return null;
    Vec3? solve(List<(Vec3, Vec3)> rs) {
      final a = List.generate(3, (_) => List.filled(3, 0.0)), b = List.filled(3, 0.0);
      for (final (o, d) in rs) {
        for (var r = 0; r < 3; r++) {
          for (var c = 0; c < 3; c++) {
            final m = (r == c ? 1.0 : 0.0) - d[r] * d[c];
            a[r][c] += m;
            b[r] += m * o[c];
          }
        }
      }
      return _solve3(a, b);
    }

    double off(Vec3 x, (Vec3, Vec3) r) {
      final q = _sub(x, r.$1);
      return _len(_sub(q, [r.$2[0] * _dot(q, r.$2), r.$2[1] * _dot(q, r.$2), r.$2[2] * _dot(q, r.$2)]));
    }

    var x = solve(rays);
    if (x == null) return null;
    final kept = rays.where((r) => off(x!, r) < 0.025).toList();
    if (kept.length >= 4 && kept.length < rays.length) {
      rays = kept;
      x = solve(rays);
    }
    return x;
  }

  /// The arm from the triangulated hand, or null while the hand is not pinned down yet (or looks implausible).
  ArmCylinder? arm({double forearm = 0.22}) {
    final w = point(0), m = point(9), i5 = point(5), p17 = point(17);
    if (w == null || m == null || i5 == null || p17 == null) return null;
    final len = _len(_sub(w, m)), palm = _len(_sub(i5, p17));
    if (len < 0.05 || len > 0.14 || palm < 0.04 || palm > 0.11) return null;
    final u = _unit(_sub(w, m));
    var tip = len;
    for (final k in const [8, 12, 16]) {
      final t = point(k);
      if (t != null) tip = math.max(tip, -_dot(_sub(t, w), u));
    }
    tip = math.min(tip, 0.24);
    return ArmCylinder(_add(w, u, forearm), _add(w, u, -(tip + 0.01)), (0.65 * palm).clamp(0.03, 0.06));
  }
}

Vec3? _solve3(List<List<double>> a, List<double> b) {
  final m = [for (var r = 0; r < 3; r++) [...a[r], b[r]]];
  for (var c = 0; c < 3; c++) {
    var p = c;
    for (var r = c + 1; r < 3; r++) {
      if (m[r][c].abs() > m[p][c].abs()) p = r;
    }
    if (m[p][c].abs() < 1e-9) return null;
    final t = m[c];
    m[c] = m[p];
    m[p] = t;
    for (var r = 0; r < 3; r++) {
      if (r == c) continue;
      final f = m[r][c] / m[c][c];
      for (var k = c; k < 4; k++) {
        m[r][k] -= f * m[c][k];
      }
    }
  }
  return [for (var r = 0; r < 3; r++) m[r][3] / m[r][r]];
}

// ---- hand pose: are the fingers still the way they were? -------------------------------------------------------

const _fingers = [
  [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], //
];

double _angleDeg(List<double> w, int a, int b, int c) {
  final u = [for (var k = 0; k < 3; k++) w[3 * a + k] - w[3 * b + k]];
  final v = [for (var k = 0; k < 3; k++) w[3 * c + k] - w[3 * b + k]];
  final nu = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), nv = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (nu < 1e-6 || nv < 1e-6) return 0;
  final c_ = ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)).clamp(-1.0, 1.0);
  return math.acos(c_) * 180 / math.pi;
}

/// The hand's shape as angles (degrees) from its metric 3D landmarks: the bend at every finger joint and the spread
/// between neighbouring fingers. Independent of where the camera is (the landmarker's metric hand), so it changes
/// only when the fingers move.
List<double>? handPose(List<double> world) {
  if (world.length < 63) return null;
  final out = <double>[];
  for (final f in _fingers) {
    out.add(_angleDeg(world, 0, f[0], f[1]));
    out.add(_angleDeg(world, f[0], f[1], f[2]));
    out.add(_angleDeg(world, f[1], f[2], f[3]));
  }
  for (var i = 1; i < 4; i++) {
    out.add(_angleDeg(world, _fingers[i][1], _fingers[i][0], _fingers[i + 1][1]));
  }
  return out;
}

/// How far a hand's shape is from a reference shape: the median change over its angles (degrees). The median, so a
/// few joints the landmarker misjudges when the hand is seen edge-on do not count as a movement.
double poseChange(List<double> a, List<double> b) {
  final d = [for (var i = 0; i < math.min(a.length, b.length); i++) (a[i] - b[i]).abs()]..sort();
  return d.isEmpty ? 0 : d[d.length ~/ 2];
}
