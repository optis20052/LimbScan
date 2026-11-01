import 'dart:math' as math;
import 'dart:typed_data';

import 'package:limb_capture/limb_capture.dart';

import 'ar_math.dart';
import 'guidance.dart';

/// Which parts of the arm SKIN have been photographed well, for painting them green live.
///
/// The arm is modelled as a cylinder (axis through the estimated arm centre: vertical for an upright arm,
/// along the table for a resting one). Its surface is a grid of cells (angle around the axis x height along
/// it). A photo counts for a cell when the cell is inside the photo and faces the camera (within 60°); a cell
/// is "scanned" with [goodViews] such photos, which is what the 3D reconstruction needs. Each pixel of the
/// live arm mask is mapped to its cell by intersecting its camera ray with the cylinder, so the colours follow
/// the real arm outline.
class SurfaceCoverage {
  SurfaceCoverage(this.passes, {this.resting = false});

  final int passes;
  final bool resting;

  static const nTheta = 48; // 7.5° around the arm
  static const nS = 80; // 1 cm along it
  static const sMin = -0.40; // metres from the centre along the axis
  static const ds = 0.01;
  static const radius = 0.04; // a forearm is ~25 cm round; only shapes the mapping near the silhouette
  static const goodViews = 2;
  static const presentHits = 3; // a cell is "arm" once the mask has landed on it this many times
  static const done = 0.92; // share of the seen arm that must be green
  static const cells = nTheta * nS;

  Vec3? centre;
  Vec3 axis = const [0, 1, 0];
  final List<_View> _shots = [];
  late final List<Uint8List> _views = List.generate(passes, (_) => Uint8List(cells));
  late final List<Uint16List> _hits = List.generate(passes, (_) => Uint16List(cells));
  late final List<Int32List> _greenAt = List.generate(passes, (_) => Int32List(cells)); // ms since start
  late final List<int> _photos = List.filled(passes, 0);
  late final List<int> _photosAtLastGreen = List.filled(passes, 0);
  final int _t0 = DateTime.now().millisecondsSinceEpoch;

  int _ms(int now) => now - _t0;

  // ---- frame --------------------------------------------------------------------------------------------------
  (Vec3, Vec3) _basis() {
    if (!resting) return (const [0, 0, 1], const [1, 0, 0]); // theta = azimuth around the arm (as in guidance)
    final a = axis;
    final up = [-a[1] * a[0], 1 - a[1] * a[1], -a[1] * a[2]]; // world up, minus its axis component
    final e1 = _norm(up);
    return (e1, _cross(a, e1));
  }

  /// Move the cylinder (better centre / axis estimate). Recounts photos when it moved noticeably.
  void setFrame(Vec3 c, {Vec3? axis}) {
    final a = axis == null ? this.axis : _norm(axis);
    final old = centre;
    final moved = old == null || _len(_sub(c, old)) > 0.015 || _dot(a, this.axis).abs() < math.cos(6 * math.pi / 180);
    centre = c;
    this.axis = a;
    if (moved && old != null) _rebuild();
  }

  void _rebuild() {
    for (var p = 0; p < passes; p++) {
      _views[p].fillRange(0, cells, 0);
      _hits[p].fillRange(0, cells, 0);
      _greenAt[p].fillRange(0, cells, 0);
    }
    for (final s in _shots) {
      _count(s, glow: false);
    }
  }

  // ---- photos -------------------------------------------------------------------------------------------------
  void addShot(Vec3 pos, List<double> vp, int pass) {
    final s = _View(pos, vp, pass);
    _shots.add(s);
    _photos[pass]++;
    if (centre != null && _count(s, glow: true) > 0) _photosAtLastGreen[pass] = _photos[pass];
  }

  /// Adds one photo to the cells it sees well; returns how many cells turned green.
  int _count(_View s, {required bool glow}) {
    final c = centre!;
    final (e1, e2) = _basis();
    final v = _views[s.pass];
    final now = _ms(DateTime.now().millisecondsSinceEpoch);
    var newGreen = 0;
    for (var it = 0; it < nTheta; it++) {
      final th = (it + 0.5) * 2 * math.pi / nTheta;
      final n = [
        math.cos(th) * e1[0] + math.sin(th) * e2[0],
        math.cos(th) * e1[1] + math.sin(th) * e2[1],
        math.cos(th) * e1[2] + math.sin(th) * e2[2],
      ];
      for (var js = 0; js < nS; js++) {
        final sv = sMin + (js + 0.5) * ds;
        final p = [
          c[0] + sv * axis[0] + radius * n[0],
          c[1] + sv * axis[1] + radius * n[1],
          c[2] + sv * axis[2] + radius * n[2],
        ];
        final toCam = _sub(s.pos, p);
        final d = _len(toCam);
        if (d < 0.1 || d > 0.8 || _dot(n, toCam) / d < 0.5) continue; // too near/far, or not facing the camera
        if (!_inFrame(s.vp, p)) continue;
        final k = js * nTheta + it;
        if (v[k] < 255) v[k]++;
        if (v[k] == goodViews) {
          newGreen++;
          _greenAt[s.pass][k] = glow ? now : 0;
        }
      }
    }
    return newGreen;
  }

  static bool _inFrame(List<double> vp, Vec3 p) {
    final w = depth(vp, p);
    if (w < 0.05) return false;
    final x = (vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12]) / w;
    final y = (vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13]) / w;
    return x.abs() < 0.92 && y.abs() < 0.92;
  }

  // ---- live mask -> cells -------------------------------------------------------------------------------------
  /// The cylinder cell seen through each grid pixel of [l] (-1: not on the cylinder's length).
  Int32List? _cellsFor(LimbInfo l) {
    final c = centre, vp = l.vp;
    if (c == null || vp == null) return null;
    final inv = invert(vp);
    if (inv == null) return null;
    final (e1, e2) = _basis();
    final a = axis;
    final out = Int32List(l.gridW * l.gridH)..fillRange(0, l.gridW * l.gridH, -1);
    for (var j = 0; j < l.gridH; j++) {
      for (var i = 0; i < l.gridW; i++) {
        final idx = j * l.gridW + i;
        if (l.mask[idx] < 40) continue;
        final nx = 2 * (i + 0.5) / l.gridW - 1, ny = 1 - 2 * (j + 0.5) / l.gridH;
        final o = _apply(inv, nx, ny, -1), f = _apply(inv, nx, ny, 1);
        final d = _norm(_sub(f, o));
        // ray vs. infinite cylinder: first hit on the near side, or the point closest to the axis (silhouette)
        final oc = _sub(o, c);
        final dp = _sub(d, _scale(a, _dot(d, a)));
        final op = _sub(oc, _scale(a, _dot(oc, a)));
        final qa = _dot(dp, dp), qb = 2 * _dot(op, dp), qc = _dot(op, op) - radius * radius;
        if (qa < 1e-9) continue; // looking straight along the axis
        final disc = qb * qb - 4 * qa * qc;
        final t = disc >= 0 ? (-qb - math.sqrt(disc)) / (2 * qa) : -qb / (2 * qa);
        if (t <= 0) continue;
        final rel = _add(oc, _scale(d, t));
        final sv = _dot(rel, a);
        final js = ((sv - sMin) / ds).floor();
        if (js < 0 || js >= nS) continue;
        final rad = _sub(rel, _scale(a, sv));
        var th = math.atan2(_dot(rad, e2), _dot(rad, e1));
        if (th < 0) th += 2 * math.pi;
        out[idx] = js * nTheta + (th / (2 * math.pi) * nTheta).floor() % nTheta;
      }
    }
    return out;
  }

  /// Live colours for the arm mask (premultiplied RGBA on the mask grid): white = not scanned yet, light green =
  /// one good photo, green = scanned (with a short flash when it turns green). Also records which cells are arm
  /// when [record] (the phone is tracking), for the completion share.
  Uint8List? shade(LimbInfo l, int pass, int now, {required bool record}) {
    final map = _cellsFor(l);
    if (map == null) return null;
    final v = _views[pass], hits = _hits[pass], at = _greenAt[pass];
    final t = _ms(now);
    final w = l.gridW, h = l.gridH, n = map.length;
    // coverage 0 (not scanned) .. 1 (scanned) per pixel, and the "just turned green" flash
    var g = Float32List(n);
    final flash = Float32List(n);
    final seen = <int>{};
    for (var i = 0; i < n; i++) {
      final k = map[i];
      if (l.mask[i] == 0 || k < 0) continue;
      if (record && l.mask[i] >= 128 && seen.add(k) && hits[k] < 65535) hits[k]++;
      final c = v[k];
      g[i] = c >= goodViews ? 1.0 : (c > 0 ? 0.45 : 0.0);
      if (c >= goodViews && at[k] > 0 && t - at[k] < 700) flash[i] = 0.6 * (1 - (t - at[k]) / 700);
    }
    // soft, skin-like colour: blurred across ~5 grid cells (the cylinder cells would show as stripes) and over time
    g = _blur(_blur(g, w, h), w, h);
    final prev = _gPrev[pass];
    if (prev != null && prev.length == n) {
      for (var i = 0; i < n; i++) {
        g[i] = 0.45 * g[i] + 0.55 * prev[i];
      }
    }
    _gPrev[pass] = g;
    final out = Uint8List(n * 4);
    for (var i = 0; i < n; i++) {
      final a = l.mask[i];
      if (a == 0) continue;
      final x = g[i].clamp(0.0, 1.0);
      var r = 1 + (_green.$1 - 1) * x, gg = 1 + (_green.$2 - 1) * x, b = 1 + (_green.$3 - 1) * x;
      final f = _blur3(flash, w, h, i);
      r += (1 - r) * f;
      gg += (1 - gg) * f;
      b += (1 - b) * f;
      out[4 * i] = (r * a).round();
      out[4 * i + 1] = (gg * a).round();
      out[4 * i + 2] = (b * a).round();
      out[4 * i + 3] = a;
    }
    return out;
  }

  late final List<Float32List?> _gPrev = List.filled(passes, null);

  /// 5x5 box blur (two separable passes of radius 2).
  static Float32List _blur(Float32List a, int w, int h) {
    final tmp = Float32List(a.length), out = Float32List(a.length);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var s = 0.0;
        var c = 0;
        for (var d = -2; d <= 2; d++) {
          final xx = x + d;
          if (xx < 0 || xx >= w) continue;
          s += a[y * w + xx];
          c++;
        }
        tmp[y * w + x] = s / c;
      }
    }
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var s = 0.0;
        var c = 0;
        for (var d = -2; d <= 2; d++) {
          final yy = y + d;
          if (yy < 0 || yy >= h) continue;
          s += tmp[yy * w + x];
          c++;
        }
        out[y * w + x] = s / c;
      }
    }
    return out;
  }

  static double _blur3(Float32List a, int w, int h, int i) {
    final x = i % w, y = i ~/ w;
    var s = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        final xx = (x + dx).clamp(0, w - 1), yy = (y + dy).clamp(0, h - 1);
        s += a[yy * w + xx];
      }
    }
    return s / 9;
  }

  static const _green = (0.20, 0.86, 0.45);

  /// Resting arm: its direction on the table, from the long axis of the mask (rays to the table-height plane).
  void estimateAxis(LimbInfo l) {
    final c = centre, vp = l.vp;
    if (!resting || c == null || vp == null || _shots.length > 12) return;
    double sw = 0, mx = 0, my = 0;
    for (var j = 0; j < l.gridH; j++) {
      for (var i = 0; i < l.gridW; i++) {
        final w = l.mask[j * l.gridW + i] >= 128 ? 1.0 : 0.0;
        sw += w;
        mx += w * i;
        my += w * j;
      }
    }
    if (sw < 50) return;
    mx /= sw;
    my /= sw;
    double sxx = 0, syy = 0, sxy = 0;
    for (var j = 0; j < l.gridH; j++) {
      for (var i = 0; i < l.gridW; i++) {
        if (l.mask[j * l.gridW + i] < 128) continue;
        sxx += (i - mx) * (i - mx);
        syy += (j - my) * (j - my);
        sxy += (i - mx) * (j - my);
      }
    }
    final ang = 0.5 * math.atan2(2 * sxy, sxx - syy);
    final du = math.cos(ang) * 0.15, dv = math.sin(ang) * 0.15 * l.gridW / l.gridH;
    final u = (mx + 0.5) / l.gridW, v = (my + 0.5) / l.gridH;
    final p1 = _onPlane(vp, u - du, v - dv, c[1]), p2 = _onPlane(vp, u + du, v + dv, c[1]);
    if (p1 == null || p2 == null) return;
    var a = _norm([p2[0] - p1[0], 0, p2[2] - p1[2]]);
    if (_dot(a, axis) < 0) a = _scale(a, -1);
    setFrame(c, axis: _norm(_add(_scale(axis, 0.8), _scale(a, 0.2))));
  }

  static Vec3? _onPlane(List<double> vp, double u, double v, double y) {
    final inv = invert(vp);
    if (inv == null) return null;
    final o = _apply(inv, 2 * u - 1, 1 - 2 * v, -1), f = _apply(inv, 2 * u - 1, 1 - 2 * v, 1);
    final d = _sub(f, o);
    if (d[1].abs() < 1e-6) return null;
    final t = (y - o[1]) / d[1];
    return t > 0 ? _add(o, _scale(d, t)) : null;
  }

  // ---- progress -----------------------------------------------------------------------------------------------
  /// Share of the arm seen in this pass that is scanned, or null while too little of it has been seen.
  double? progress(int pass) {
    var present = 0, green = 0;
    for (var k = 0; k < cells; k++) {
      if (_hits[pass][k] < presentHits) continue;
      present++;
      if (_views[pass][k] >= goodViews) green++;
    }
    return present < 60 ? null : green / present;
  }

  int photosSinceLastGreen(int pass) => _photos[pass] - _photosAtLastGreen[pass];

  /// Upright arm: 24 sides around it (like the coverage ring), 1 where the seen skin on that side is scanned.
  List<int>? sideBins(int pass) {
    if (resting || progress(pass) == null) return null;
    final present = List.filled(24, 0), green = List.filled(24, 0);
    for (var k = 0; k < cells; k++) {
      if (_hits[pass][k] < presentHits) continue;
      final b = (k % nTheta) * 24 ~/ nTheta;
      present[b]++;
      if (_views[pass][k] >= goodViews) green[b]++;
    }
    return [for (var b = 0; b < 24; b++) present[b] > 0 && green[b] >= 0.9 * present[b] ? 1 : 0];
  }
}

class _View {
  _View(this.pos, this.vp, this.pass);
  final Vec3 pos;
  final List<double> vp;
  final int pass;
}

Vec3 _apply(List<double> m, double x, double y, double z) {
  final w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

Vec3 _sub(Vec3 a, Vec3 b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
Vec3 _add(Vec3 a, Vec3 b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
Vec3 _scale(Vec3 a, double k) => [a[0] * k, a[1] * k, a[2] * k];
double _dot(Vec3 a, Vec3 b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
double _len(Vec3 a) => math.sqrt(_dot(a, a));
Vec3 _norm(Vec3 a) => _scale(a, 1 / math.max(1e-12, _len(a)));
Vec3 _cross(Vec3 a, Vec3 b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
