import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:limb_capture/limb_capture.dart';
import 'package:path_provider/path_provider.dart';

import '../api/limbscan_api.dart';
import '../capture/ar_math.dart';
import '../capture/arm_model.dart';
import '../capture/guidance.dart';
import '../capture/upload_queue.dart';
import '../config.dart';
import '../theme.dart';
import '../widgets/arm_guide.dart';
import '../widgets/limb_overlay.dart';
import 'result_screen.dart';

enum Tone { good, info, warn }

class Hint {
  const Hint(this.text, this.tone, this.icon);
  final String text;
  final Tone tone;
  final IconData icon;
}

/// place: waiting for the tap on the arm; fitting: cutting it out and fitting the cylinder; confirm: cylinder
/// shown, starts by itself; scan: circling; done: all sides captured.
enum Stage { place, fitting, confirm, calibrate, scan, done }

/// Guided AR capture in two moves (person seated, elbow resting on a table, hand up).
///
/// 1. Place: the user taps the arm once. The app cuts the tapped object out of that frame, hit-tests the tap in 3D
///    and fits a cylinder to the arm. The cylinder is then FIXED in the AR world: nothing afterwards depends on
///    finding the arm in the camera image again (that was the jittery, lost-arm part).
/// 2. Walk: the user walks once around the hand, from one side of the person round the front to the other, keeping
///    the hand on screen. Photos are taken automatically every few degrees when the whole hand is on screen, close
///    and steady. A ring around the hand on the camera image shows the walk (green: done, amber: still to go), and an
///    arrow and one instruction beside the hand say which way. Done at 200 degrees: the body and the head block the
///    rest, so nothing more is asked.
///
/// Scale comes from the phone's AR tracking (no card). Each photo carries a point on the arm (the cylinder's middle
/// projected into it) so the server cuts out the right object: a real arm or a printed model, same flow.
class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key, required this.config});
  final AppConfig config;

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> with TickerProviderStateMixin {
  late final LimbScanApi _api = LimbScanApi(widget.config.serverUrl);
  /// Upright (forearm standing, hand up) or lying (on a table), from the placed arm itself: within 45 degrees of
  /// vertical is upright. Before placement, the posture chosen in setup.
  bool get _upright {
    final a = _arm;
    if (a == null) return widget.config.pose != 'resting';
    final d = [a.b[0] - a.a[0], a.b[1] - a.a[1], a.b[2] - a.a[2]];
    final l = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    return l > 1e-6 && d[1].abs() / l > math.cos(math.pi / 4);
  }

  final _arc = ScanArc();
  late final AnimationController _pulse = AnimationController(vsync: this, duration: const Duration(milliseconds: 900))
    ..repeat(reverse: true);
  late final AnimationController _sweep = AnimationController(vsync: this, duration: const Duration(milliseconds: 2400))
    ..repeat();
  // high-rate state for the AR overlays: they repaint on their own, without rebuilding the whole screen
  final _poseN = ValueNotifier<ArPose?>(null);
  final _limbN = ValueNotifier<LimbInfo?>(null);
  final _lockedN = ValueNotifier<bool>(true);
  final _tts = FlutterTts();
  StreamSubscription<ArPose>? _poseSub;
  StreamSubscription<LimbInfo>? _limbSub;
  StreamSubscription<HandsInfo>? _handSub;
  final _handN = ValueNotifier<HandInfo?>(null); // the hand being found, drawn while placing
  final List<(int, double, double)> _handSeen = []; // (time, wrist u, v) of recent sightings: placed once steady
  bool _handPlacing = false;
  HandTrack _track = HandTrack(); // the hand from every view while scanning: corrects the cylinder as the phone moves
  int _refined = 0;
  Timer? _logTimer;
  late final String _session = DateTime.now().toIso8601String().substring(0, 19).replaceAll(':', '-');
  int _lastHandAt = 0;
  StreamSubscription<String>? _errSub;
  UploadQueue? _uploads;
  String? _scanId;
  String? _dir;

  Stage _stage = Stage.place;
  ArmCylinder? _arm;
  final List<ArmCylinder> _fits = [];
  int _fitStart = 0;
  List<double>? _hit; // the tapped point on the arm, once its distance is known
  List<double>? _tapHit; // ARCore's own hit test at the tap (depth map only), a candidate
  double _tapU = 0.5, _tapV = 0.5;
  bool _measuring = false;
  Uint8List? _gate; // the tapped outline cells at the arm's distance (a cut-out spilling onto a body behind is cut off)
  Timer? _confirmTimer;
  String? _go; // which way to keep walking: left / right
  String? _banner; // short message at the top
  int _bannerAt = 0;

  bool _capturing = false;
  bool _finishing = false;
  bool _torch = false;
  bool _torchSet = false; // the torch goes on by default once the AR camera runs (even, moving-with-the-phone light)
  int _lastShotMs = 0;
  int? _trackingSince;
  double _pathSinceTracking = 0; // metres moved since tracking (re)started: AR scale settles with motion
  ArPose? _pose;
  double _rotSpeed = 0;
  double _linSpeed = 0; // m/s, smoothed
  Hint _hint = const Hint('Move the phone slowly so it can start tracking', Tone.info, Icons.screen_rotation_alt_rounded);
  int _hintAt = 0;
  int _spokeAt = 0;
  int _builtAt = 0;
  final List<double> _sharpHist = [];

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    _tts.setSpeechRate(0.5);
    _start();
  }

  Future<void> _start() async {
    _dir = '${(await getTemporaryDirectory()).path}/limbscan_${DateTime.now().millisecondsSinceEpoch}';
    try {
      _scanId = await _api.createScan('AR scan ${DateTime.now().toString().substring(0, 16)}');
      _uploads = UploadQueue(_api, _scanId!, onChange: () => mounted ? setState(() {}) : null);
    } catch (e) {
      _setHint(
        const Hint('Server not reachable: check the address in Settings', Tone.warn, Icons.cloud_off_rounded),
        force: true,
      );
    }
    LimbCapture.clearColour().ignore();
    _poseSub = LimbCapture.poses.listen(_onPose);
    _limbSub = LimbCapture.limbs.listen(_onLimb);
    _handSub = LimbCapture.hands.listen(_onHands);
    LimbCapture.findHands(true).ignore();
    _errSub = LimbCapture.errors.listen(
      (m) => _setHint(Hint('AR error: $m', Tone.warn, Icons.error_outline_rounded), force: true),
    );
    _speak('Point the phone at the hand.', urgent: true);
    _logTimer = Timer.periodic(const Duration(seconds: 20), (_) => _sendLog());
  }

  /// The phone's log goes to the server (under the scan id once there is one): tests need no USB cable.
  Future<void> _sendLog() async {
    try {
      final text = await LimbCapture.readLog();
      if (text != null && text.isNotEmpty) await _api.sendLog(_scanId ?? _session, text);
    } catch (_) {}
  }

  @override
  void dispose() {
    _logTimer?.cancel();
    _sendLog();
    _poseSub?.cancel();
    _limbSub?.cancel();
    _errSub?.cancel();
    _confirmTimer?.cancel();
    _pulse.dispose();
    _sweep.dispose();
    _poseN.dispose();
    _limbN.dispose();
    _handN.dispose();
    _handSub?.cancel();
    _lockedN.dispose();
    _tts.stop();
    if (_torch) LimbCapture.setTorch(false).ignore();
    LimbCapture.clearColour().ignore();
    SystemChrome.setPreferredOrientations([]);
    super.dispose();
  }

  int get _now => DateTime.now().millisecondsSinceEpoch;

  /// Rebuild the HUD at most ~8 times a second (the AR overlays repaint by themselves at camera rate).
  void _refresh({bool now = false}) {
    if (!mounted) return;
    if (now || _now - _builtAt > 120) {
      _builtAt = _now;
      setState(() {});
    }
  }

  void _speak(String s, {bool urgent = false}) {
    if (!widget.config.voice) return;
    if (!urgent && _now - _spokeAt < 3500) return;
    _spokeAt = _now;
    _tts.speak(s);
  }

  void _setHint(Hint h, {bool force = false}) {
    if (h.text == _hint.text || (!force && _now - _hintAt < 900)) return;
    debugPrint('LimbScan hint: ${h.text}');
    _hint = h;
    _hintAt = _now;
    if (h.tone == Tone.warn) _speak(h.text.replaceAll(':', ','));
  }

  bool get _warm =>
      _pose != null &&
      _pose!.isTracking &&
      _trackingSince != null &&
      _pose!.t - _trackingSince! > 1000 &&
      _pathSinceTracking > 0.05;

  // ---- 1. place the arm --------------------------------------------------------------------------------------
  Future<void> _onTap(Offset at, Size box) async {
    if (_stage == Stage.scan || _stage == Stage.done || _finishing) return;
    if (!_warm) {
      _setHint(
        const Hint('Move the phone slowly for a second first', Tone.warn, Icons.screen_rotation_alt_rounded),
        force: true,
      );
      _refresh(now: true);
      return;
    }
    final u = at.dx / box.width, v = at.dy / box.height;
    HapticFeedback.selectionClick();
    _confirmTimer?.cancel();
    final started = _now;
    setState(() {
      _stage = Stage.fitting;
      _arm = null;
      _fits.clear();
      _fitStart = started;
      _hit = null;
      _tapHit = null;
      _measuring = false;
      _gate = null;
      _tapU = u;
      _tapV = v;
    });
    await LimbCapture.pickColour(u, v);
    final h = await LimbCapture.hitTest(u, v);
    if (_fitStart == started) _tapHit = h;
    Future<void>.delayed(const Duration(milliseconds: 2500), () {
      if (!mounted || _stage != Stage.fitting || _fitStart != started) return;
      if (_fits.isNotEmpty) return _placed();
      LimbCapture.clearColour().ignore();
      setState(() => _stage = Stage.place);
      _setHint(
        const Hint('Could not find the arm there: tap the middle of the arm', Tone.warn, Icons.touch_app_rounded),
        force: true,
      );
    });
  }

  void _onLimb(LimbInfo l) {
    _limbN.value = l;
    if (_stage != Stage.fitting || !l.visible || l.vp == null) return;
    final p = _pose;
    if (p == null) return;
    final hit = _hit;
    if (hit == null) {
      if (!_measuring) _measure(l, p);
      return;
    }
    final g = _gate;
    final mask = g == null || g.length != l.mask.length
        ? l.mask
        : Uint8List.fromList([for (var k = 0; k < g.length; k++) _near(g, k, l.gridW, l.gridH) ? l.mask[k] : 0]);
    final f = ArmCylinder.fit(
      mask: mask,
      gw: l.gridW,
      gh: l.gridH,
      vp: l.vp!,
      fwd: p.fwd,
      hit: hit,
      upright: _upright,
    );
    if (f != null) _fits.add(f);
    if (_fits.length >= 5 || (_fits.length >= 2 && _now - _fitStart > 1200)) _placed();
  }

  /// The hand is found by MediaPipe's hand landmarker (any background, no tap): once it has stayed put for a moment
  /// the cylinder is placed from its landmarks. Tapping is the fallback (an object that is not a hand).
  void _onHands(HandsInfo h) {
    if ((_stage == Stage.confirm || _stage == Stage.calibrate || _stage == Stage.scan) && h.vp.length == 16) return _refine(h);
    if (_stage != Stage.place || !_warm || _handPlacing || h.vp.length != 16) return;
    final p = _pose;
    if (p == null) return;
    // the hand being scanned: big on screen and near the middle (a second hand, or a person behind, is not)
    HandInfo? best;
    var bestScore = 0.0;
    for (final x in h.hands) {
      final span = math.sqrt(math.pow(x.lm[0] - x.lm[18], 2) + math.pow(x.lm[1] - x.lm[19], 2));
      final cu = (x.lm[0] + x.lm[18]) / 2 - 0.5, cv = (x.lm[1] + x.lm[19]) / 2 - 0.5;
      final sc = span * (1 - math.min(0.8, math.sqrt(cu * cu + cv * cv)));
      if (sc > bestScore) {
        bestScore = sc;
        best = x;
      }
    }
    _handN.value = best;
    final now = _now;
    if (best == null) {
      _handSeen.clear();
      return;
    }
    _lastHandAt = now;
    _handSeen
      ..add((now, best.lm[0], best.lm[1]))
      ..removeWhere((e) => now - e.$1 > 700);
    final steady = _handSeen.length >= 4 &&
        _handSeen.every((e) => (e.$2 - best!.lm[0]).abs() < 0.04 && (e.$3 - best.lm[1]).abs() < 0.04);
    if (steady) _placeFromHand(best, h.vp, p);
  }

  Future<void> _placeFromHand(HandInfo h, List<double> vp, ArPose p) async {
    _handPlacing = true;
    try {
      final size = handDistance(h.lm, h.world, vp);
      if (size == null) return;
      // ARCore depth at the palm (wrist, knuckles, between them) confirms or corrects the size estimate
      final palm = [0, 1, 5, 9, 13, 17];
      final uv = <double>[for (final i in palm) ...[h.lm[2 * i], h.lm[2 * i + 1]]];
      for (final (i, j) in const [(0, 9), (5, 13), (9, 17), (0, 13)]) {
        uv
          ..add((h.lm[2 * i] + h.lm[2 * j]) / 2)
          ..add((h.lm[2 * i + 1] + h.lm[2 * j + 1]) / 2);
      }
      final raw = await LimbCapture.depth(uv);
      if (!mounted || _stage != Stage.place) return;
      final ds = (raw ?? const <double>[]).where((d) => d > 0.08 && d < 2).toList()..sort();
      final measured = ds.length >= 4 ? ds[ds.length ~/ 2] : null;
      // the depth map sees the skin, the landmarks are the joints ~1 cm inside it
      final agree = measured != null && measured + 0.01 > 0.75 * size && measured + 0.01 < 1.33 * size;
      final dist = agree ? measured + 0.01 : size;
      debugPrint('LimbScan place: hand at ${size.toStringAsFixed(3)} m by its size, depth '
          '${measured?.toStringAsFixed(3) ?? 'n/a'} (${ds.length}/${uv.length ~/ 2}) -> ${dist.toStringAsFixed(3)} m');
      if (dist < 0.12 || dist > 0.9) {
        _setHint(const Hint('Hold the phone 30–40 cm from the hand', Tone.warn, Icons.straighten_rounded), force: true);
        return;
      }
      final pts = handPoints(h.lm, h.world, vp, p.pos, dist);
      final arm = pts == null ? null : armFromHand(pts, h.world);
      if (arm == null) return;
      HapticFeedback.selectionClick();
      _track = HandTrack()..add(h.lm, vp);
      _refined = 0;
      setState(() {
        _stage = Stage.fitting;
        _fits
          ..clear()
          ..add(arm);
        _fitStart = _now;
      });
      _handN.value = null;
      _placed();
    } finally {
      _handPlacing = false;
      _handSeen.clear();
    }
  }

  /// While scanning: the scanned hand (the one where the cylinder is) seen from this new position joins the track; once
  /// it is triangulated from enough directions the cylinder follows the triangulated hand.
  /// The hand as the camera sees it right now (scan stage): framing and distance for the photo rules come from it,
  /// not from the placed cylinder (a cylinder off the hand, or a hand held free that moves, blocked every photo).
  ({int t, List<double> lm, double? d, Vec3? dir})? _live;

  /// Unit direction from the camera to the palm's centre as seen at view-normalized landmarks [lm].
  static Vec3? _palmRay(List<double> vp, List<double> lm) {
    var u = 0.0, v = 0.0;
    for (final i in const [0, 5, 9, 13, 17]) {
      u += lm[2 * i] / 5;
      v += lm[2 * i + 1] / 5;
    }
    final a = unproject(vp, u, v, 0), b = unproject(vp, u, v, 1);
    if (a == null || b == null) return null;
    final d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    final n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    return n < 1e-9 ? null : [d[0] / n, d[1] / n, d[2] / n];
  }

  void _refine(HandsInfo h) {
    HandInfo? big;
    var bigScore = 0.0;
    for (final x in h.hands) {
      final span = math.sqrt(math.pow(x.lm[0] - x.lm[18], 2) + math.pow(x.lm[1] - x.lm[19], 2));
      final cu = (x.lm[0] + x.lm[18]) / 2 - 0.5, cv = (x.lm[1] + x.lm[19]) / 2 - 0.5;
      final sc = span * (1 - math.min(0.8, math.sqrt(cu * cu + cv * cv)));
      if (sc > bigScore) {
        bigScore = sc;
        big = x;
      }
    }
    if (big != null) {
      _live = (t: _now, lm: big.lm, d: handDistance(big.lm, big.world, h.vp), dir: _palmRay(h.vp, big.lm));
      final pose = handPose(big.world);
      if (pose != null) {
        _poses.add(pose);
        if (_poses.length > 5) _poses.removeAt(0);
      }
    }
    final arm = _arm;
    if (arm == null) return;
    final c = project(h.vp, palmPoint(arm), const Size(1, 1));
    if (c == null) return;
    HandInfo? best;
    var bd = 0.25;
    for (final x in h.hands) {
      final mu = (x.lm[0] + x.lm[18]) / 2, mv = (x.lm[1] + x.lm[19]) / 2;
      final d = math.sqrt(math.pow(mu - c.dx, 2) + math.pow(mv - c.dy, 2));
      if (d < bd) {
        bd = d;
        best = x;
      }
    }
    if (best == null || !_track.add(best.lm, h.vp)) return;
    final next = _track.arm();
    if (next == null) return;
    final moved = math.sqrt(math.pow(next.mid[0] - arm.mid[0], 2) + math.pow(next.mid[1] - arm.mid[1], 2) + math.pow(next.mid[2] - arm.mid[2], 2));
    if (moved > 0.2) return; // not the same hand
    if (++_refined % 5 == 1) {
      debugPrint('LimbScan refine: ${_track.views} views, cylinder moved ${(moved * 1000).round()} mm, '
          'r ${(next.r * 1000).round()} mm, length ${(next.length * 1000).round()} mm');
    }
    setState(() => _arm = next);
  }

  /// How far the tapped arm is, along the tapped ray: the first measurement (ARCore depth over the outline, its hit
  /// test, tracked points on the outline) that agrees with the outline's size for a real arm's width, else that size.
  /// A wrong distance (e.g. the wall behind) puts the cylinder off the arm and makes it slide as the phone moves.
  Future<void> _measure(LimbInfo l, ArPose p) async {
    _measuring = true;
    final started = _fitStart, vp = l.vp!;
    final cells = [
      for (var k = 0; k < l.mask.length; k++)
        if (l.mask[k] >= 128) k,
    ];
    final step = math.max(1, cells.length ~/ 3000);
    final sampled = [for (var n = 0; n < cells.length; n += step) cells[n]];
    final uv = <double>[
      for (final c in sampled) ...[(c % l.gridW + 0.5) / l.gridW, (c ~/ l.gridW + 0.5) / l.gridH],
    ];
    final raw = await LimbCapture.depth(uv);
    if (!mounted || _fitStart != started || _stage != Stage.fitting) return;
    // cut the outline down to the arm's distance before measuring its width (see depthGate)
    final tap = (_tapV * l.gridH).floor().clamp(0, l.gridH - 1) * l.gridW + (_tapU * l.gridW).floor().clamp(0, l.gridW - 1);
    final gate = raw == null ? null : depthGate(l.mask, l.gridW, l.gridH, sampled, raw, tap);
    if (gate != null && gate.near > 0.9) {
      // nobody scans an arm from 90 cm: the tap landed on what is behind the arm (bed, wall, a body)
      debugPrint('LimbScan place: tapped ${gate.near.toStringAsFixed(2)} m away: not the arm');
      LimbCapture.clearColour().ignore();
      setState(() => _stage = Stage.place);
      _setHint(const Hint('That was behind the arm: tap on the arm itself', Tone.warn, Icons.touch_app_rounded), force: true);
      return;
    }
    var mask = l.mask;
    if (gate != null) {
      final before = cells.length, after = gate.mask.where((v) => v > 0).length;
      debugPrint('LimbScan place: depth gate keeps $after of $before outline cells (arm at ${gate.near.toStringAsFixed(2)} m)');
      mask = gate.mask;
      _gate = gate.mask;
      _limbN.value = l.withMask(mask);
    }
    final hw = apparentHalfWidth(mask, l.gridW, l.gridH, vp);
    final ds = gate == null
        ? raw?.where((d) => d > 0.08 && d < 2).toList()
        : [
            for (var k = 0; k < sampled.length; k++)
              if (gate.mask[sampled[k]] > 0 && raw![k] > 0.08) raw[k],
          ];
    double dist(List<double>? q) => q == null
        ? -1
        : math.sqrt(math.pow(q[0] - p.pos[0], 2) + math.pow(q[1] - p.pos[1], 2) + math.pow(q[2] - p.pos[2], 2)).toDouble();
    final candidates = <(String, double)>[
      if (ds != null && ds.length >= 10) ('depth', (ds..sort())[(ds.length * 0.3).floor()]),
      if (_tapHit != null) ('hit', dist(_tapHit)),
      if (l.surface != null) ('points', dist(l.surface)),
    ];
    bool fits(double d) => hw == null || (d * hw >= armHalfWidthMin && d * hw <= armHalfWidthMax);
    final pick = candidates.where((c) => c.$2 > 0.1 && fits(c.$2)).firstOrNull ??
        (hw != null && hw > 0 ? ('size', (armHalfWidth / hw - armHalfWidth).clamp(0.15, 0.9)) : ('default', 0.35));
    debugPrint('LimbScan place: halfWidth ${hw?.toStringAsFixed(3)} '
        '${candidates.map((c) => '${c.$1}=${c.$2.toStringAsFixed(3)}${fits(c.$2) ? '' : '(rejected)'}').join(' ')} '
        '(${ds?.length ?? 'no'} depth samples) -> ${pick.$1} ${pick.$2.toStringAsFixed(3)} m');
    final p0 = unproject(vp, _tapU, _tapV, 0.0), p1 = unproject(vp, _tapU, _tapV, 1.0);
    if (p0 == null || p1 == null) return;
    final d = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    final n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    _hit = [for (var k = 0; k < 3; k++) p.pos[k] + d[k] / n * pick.$2];
  }

  static bool _near(Uint8List g, int k, int gw, int gh) {
    final x = k % gw, y = k ~/ gw;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        final nx = x + dx, ny = y + dy;
        if (nx >= 0 && ny >= 0 && nx < gw && ny < gh && g[ny * gw + nx] > 0) return true;
      }
    }
    return false;
  }

  /// The cylinder is fixed now: show it and start scanning after a short look ("tap again if it is not right").
  void _placed() {
    if (_fits.isEmpty || _stage != Stage.fitting) return;
    LimbCapture.clearColour().ignore();
    HapticFeedback.mediumImpact();
    setState(() {
      _arm = ArmCylinder.median(_fits);
      _stage = Stage.confirm;
      _limbN.value = null;
    });
    // no countdown or check to confirm (the server finds the limb in every photo itself): first the calibration
    // move, then the walk starts by itself
    _beginCalibration();
  }

  // ---- calibration: the phone toward the hand and back ----------------------------------------------------------
  // The phone's motion tracking settles its sense of scale with movement; the size of the model comes from it.
  Vec3? _calP0, _calDir;
  double _calMin = 0, _calMax = 0, _calPath = 0;
  Vec3? _calLast;

  void _beginCalibration() {
    final p = _pose;
    if (p == null) return _beginScan();
    _calP0 = List<double>.from(p.pos);
    _calDir = List<double>.from(p.fwd);
    _calMin = _calMax = _calPath = 0;
    _calLast = List<double>.from(p.pos);
    setState(() => _stage = Stage.calibrate);
    _speak('Move the phone slowly toward the hand and back.', urgent: true);
  }

  /// 0..1: the phone has moved 15 cm toward / away from the hand and travelled 30 cm in all.
  double get _calProgress => math.min(1, math.min((_calMax - _calMin) / 0.15, _calPath / 0.30));

  void _calibrate(ArPose p) {
    final d = [for (var k = 0; k < 3; k++) p.pos[k] - _calP0![k]];
    final along = d[0] * _calDir![0] + d[1] * _calDir![1] + d[2] * _calDir![2];
    _calMin = math.min(_calMin, along);
    _calMax = math.max(_calMax, along);
    _calPath += _dist(p.pos, _calLast!);
    _calLast = List<double>.from(p.pos);
    _setHint(const Hint('Move the phone slowly toward the hand and back', Tone.info, Icons.swap_vert_rounded));
    if (_calProgress >= 1) {
      HapticFeedback.mediumImpact();
      setState(() => _stage = Stage.confirm);
      _beginScan();
    }
  }

  void _beginScan() {
    _confirmTimer?.cancel();
    if (_arm == null || _stage != Stage.confirm) return;
    // the walk's left/right sense is fixed for the whole walk (the dial must not jump)
    final p = _pose;
    // (a point 40 cm ahead stands in for the hand: only the sense of the walk is needed)
    _walkSign = p == null ? 1 : ScanRings.rightSign([for (var k = 0; k < 3; k++) p.pos[k] + 0.4 * p.fwd[k]], p.pos, p.right);
    setState(() => _stage = Stage.scan);
    _speak('Now walk slowly around the hand, keeping it on the screen.', urgent: true);
  }

  // ---- 2. walk around the hand --------------------------------------------------------------------------------
  double _walkSign = 1; // +1: walking to the screen's right increases the azimuth
  int _blurry = 0; // photos rejected as blurry in a row
  final List<List<double>> _poses = []; // the hand's shape in the last few frames
  final List<(double, List<double>)> _posesAt = []; // the hand's shape at each photo, with where it was on the walk
  int? _movedSince; // when the fingers first looked moved (it must last before it counts)
  /// The hand's shape now: the per-angle median of the last few frames (one noisy frame does not count).
  List<double>? get _poseNow {
    if (_poses.length < 3) return null;
    return [
      for (var i = 0; i < _poses.first.length; i++) (_poses.map((p) => p[i]).toList()..sort())[_poses.length ~/ 2],
    ];
  }

  void _onPose(ArPose p) {
    if (!_torchSet) {
      _torchSet = true;
      setState(() => _torch = true);
      LimbCapture.setTorch(true).ignore();
    }
    final prev = _pose;
    _pose = p;
    _poseN.value = p;
    if (prev != null && p.isTracking && prev.isTracking) _pathSinceTracking += _dist(p.pos, prev.pos);
    if (prev != null && p.t > prev.t) {
      final dt = (p.t - prev.t) / 1000.0;
      _rotSpeed = 0.7 * _rotSpeed + 0.3 * angleDeg(p.fwd, prev.fwd) / dt;
      _linSpeed = 0.7 * _linSpeed + 0.3 * _dist(p.pos, prev.pos) / dt;
    }
    if (!p.isTracking) {
      _trackingSince = null;
      _pathSinceTracking = 0;
      _setHint(_trackingHint(p.reason));
      _refresh();
      return;
    }
    _trackingSince ??= p.t;
    if (_stage == Stage.place) {
      _setHint(
        _warm
            ? (_now - _lastHandAt < 600
                ? const Hint('Hand found: hold still', Tone.good, Icons.back_hand_rounded)
                : const Hint('Point the phone at the hand', Tone.good, Icons.back_hand_rounded))
            : const Hint('Move the phone slowly so it can start tracking', Tone.info, Icons.screen_rotation_alt_rounded),
      );
    }
    if (_stage == Stage.calibrate) {
      _calibrate(p);
      _refresh();
      return;
    }
    final arm = _arm;
    if (_stage != Stage.scan || arm == null || _finishing) {
      _refresh();
      return;
    }
    final mid = arm.mid;
    final live = _live != null && _now - _live!.t < 600 ? _live : null;
    // where the phone is on the walk: the direction it looks at the hand from (the ray to the palm the landmarker
    // found; the camera's own forward when the hand is not found), so no estimate of the hand's position or distance
    // is involved (a wrong distance made a half walk count as a full circle)
    final look = live?.dir ?? p.fwd;
    final az = (math.atan2(-look[0], -look[2]) * 180 / math.pi + 360) % 360;
    final el = math.asin((-look[1]).clamp(-1.0, 1.0)) * 180 / math.pi;
    final vp = p.vp;
    final q = vp == null ? null : project(vp, mid, const Size(1, 1));
    final centred = q != null && q.dx > 0.1 && q.dx < 0.9 && q.dy > 0.08 && q.dy < 0.85;
    final inView = live != null || (centred && arm.visibleShare(vp!) >= 0.5);
    // photos only when they are good: the whole hand on screen and big enough on it (the fingers get enough pixels),
    // judged from the picture itself, and the phone nearly still (sharp photos)
    final handIn = live != null ? _allInFrame(live.lm) : (vp != null && arm.handInView(vp));
    final big = live == null || _handSize(live.lm) >= 0.15;
    // walking pace is fine: the blur check on each photo rejects the shaky ones
    final steady = _rotSpeed < 90 && _linSpeed < 0.6; // only a real swing: the blur check judges each photo
    // the fingers must stay as they were at the first photo: photos of a hand that changed shape cannot make one model
    // compared with the photos taken from nearby (within 25 degrees on the walk): the shape the landmarker sees
    // changes a little with the viewing angle (fingers hiding each other from the side), a real movement much more
    final pn = _poseNow;
    final refs = [for (final r in _posesAt) if ((r.$1 - _arc.position).abs() <= 25) r.$2];
    var change = 0.0;
    if (pn != null && live != null && refs.isNotEmpty) {
      final d = [for (final r in refs) poseChange(pn, r)]..sort();
      change = d[d.length ~/ 2];
    }
    if (change > 25) {
      _movedSince ??= _now;
    } else {
      _movedSince = null;
    }
    final moved = _movedSince != null && _now - _movedSince! > 700;
    if (moved && _now - _hintAt > 900) {
      debugPrint('LimbScan fingers: shape change ${change.toStringAsFixed(1)} deg vs ${refs.length} nearby photos');
    }
    if (inView) _arc.track(az, _now);
    final wants = _arc.wants(el);
    final dup = _arc.duplicate(p.pos, p.fwd);
    final dir = _arc.walkDir;
    _go = dir == null ? null : (dir * _walkSign > 0 ? 'right' : 'left');
    final go = _go == null ? '' : ' to the $_go';
    // one instruction at a time, most important first
    if (!inView) {
      _setHint(const Hint('Point the phone at the hand', Tone.warn, Icons.center_focus_strong_rounded));
    } else if (!handIn) {
      _setHint(const Hint('Keep the whole hand on screen', Tone.warn, Icons.fit_screen_rounded));
    } else if (!big) {
      _setHint(const Hint('A bit closer to the hand', Tone.warn, Icons.zoom_in_rounded));
    } else if (moved) {
      _setHint(const Hint('Keep the fingers still, as they were', Tone.warn, Icons.back_hand_rounded));
    } else if (!steady) {
      _setHint(const Hint('Slow down a little', Tone.warn, Icons.slow_motion_video_rounded));
    } else if (_arc.mustGoBack) {
      // an open dot behind: back to it (the opposite way to the walk)
      final back = _go == 'left' ? 'right' : 'left';
      _setHint(Hint('Go back to the $back', Tone.warn, back == 'left' ? Icons.west_rounded : Icons.east_rounded));
    } else if (_arc.walkDir == null) {
      _setHint(const Hint('Walk slowly left or right', Tone.good, Icons.swap_horiz_rounded));
    } else {
      _setHint(Hint('Walk slowly$go', Tone.good, _go == 'left' ? Icons.west_rounded : Icons.east_rounded));
    }
    if (_warm &&
        inView &&
        handIn &&
        big &&
        !moved &&
        steady &&
        wants &&
        !dup &&
        !_capturing &&
        p.t - _lastShotMs > 300 &&
        _scanId != null) {
      _capture(p, az, el);
    }
    _refresh();
  }

  /// The hand's size on screen: the larger side of its landmarks' box, as a share of the frame's height (portrait
  /// 9:16, so a width counts 9/16).
  static double _handSize(List<double> lm) {
    var x0 = 1.0, x1 = 0.0, y0 = 1.0, y1 = 0.0;
    for (var i = 0; i + 1 < lm.length; i += 2) {
      x0 = math.min(x0, lm[i]);
      x1 = math.max(x1, lm[i]);
      y0 = math.min(y0, lm[i + 1]);
      y1 = math.max(y1, lm[i + 1]);
    }
    return math.max((x1 - x0) * 9 / 16, y1 - y0);
  }

  /// Every hand landmark (fingertips included) inside the camera frame, with a small margin.
  static bool _allInFrame(List<double> lm, {double m = 0.01}) {
    for (var i = 0; i + 1 < lm.length; i += 2) {
      if (lm[i] < m || lm[i] > 1 - m || lm[i + 1] < m || lm[i + 1] > 1 - m) return false;
    }
    return true;
  }

  double _dist(Vec3 a, Vec3 b) =>
      math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));

  Hint _trackingHint(String? reason) => switch (reason) {
    'INSUFFICIENT_LIGHT' => Hint(
      _arc.photos == 0 ? 'Too dark: add light or turn on the torch' : 'Too dark: turn on a room light',
      Tone.warn,
      Icons.flashlight_on_rounded,
    ),
    'EXCESSIVE_MOTION' => const Hint('Moving too fast: slow down', Tone.warn, Icons.slow_motion_video_rounded),
    'INSUFFICIENT_FEATURES' => const Hint(
      'Point at the arm and the table around it',
      Tone.warn,
      Icons.center_focus_strong_rounded,
    ),
    'CAMERA_UNAVAILABLE' => const Hint('Camera unavailable', Tone.warn, Icons.no_photography_rounded),
    _ => const Hint('Move the phone slowly so it can start tracking', Tone.info, Icons.screen_rotation_alt_rounded),
  };

  Future<void> _capture(ArPose p, double az, double el) async {
    _capturing = true;
    _lastShotMs = p.t;
    try {
      final f = await LimbCapture.capture(_dir!);
      // blur check against the TYPICAL recent frame (not the best one ever seen)
      _sharpHist.add(f.sharpness);
      if (_sharpHist.length > 25) _sharpHist.removeAt(0);
      final sorted = [..._sharpHist]..sort();
      final median = sorted[sorted.length ~/ 2];
      if (_sharpHist.length >= 5 && f.sharpness < 0.8 * median) {
        File(f.path).delete().ignore();
        // one blurry photo is just retried; say something only when it keeps happening
        if (++_blurry >= 3) _setHint(const Hint('Slow down a little: photos are blurry', Tone.info, Icons.blur_on_rounded), force: true);
        return;
      }
      _blurry = 0;
      // where the arm is in this photo, for the server's cut-out
      final seed = imagePoint(f.pose, f.k, f.width, f.height, palmPoint(_arm!)) ??
          imagePoint(f.pose, f.k, f.width, f.height, _arm!.mid);
      if (seed != null) f.seg = {'mode': 'auto', 'seed': seed, 'roi': imageOutline(f.pose, f.k, f.width, f.height, _arm!)};
      _arc.add(el, p.pos, p.fwd);
      final pz = _poseNow;
      if (pz != null) _posesAt.add((_arc.position, pz));
      HapticFeedback.selectionClick();
      _uploads?.add(f);
      if (_arc.complete) {
        HapticFeedback.heavyImpact();
        _done();
      }
    } on PlatformException catch (_) {
      // not tracking / busy: just try again on a later pose
    } finally {
      _capturing = false;
    }
  }

  void _done() {
    setState(() => _stage = Stage.done);
    _speak('All done. Sending the photos.', urgent: true);
    Future<void>.delayed(const Duration(milliseconds: 1600), () {
      if (mounted && !_finishing) _finish(auto: true);
    });
  }

  Future<void> _finish({bool auto = false}) async {
    if (_scanId == null || _uploads == null) return;
    final n = _arc.photos;
    if (!auto && !_arc.canFinish) {
      final ok = await showDialog<bool>(
        context: context,
        builder: (c) => AlertDialog(
          title: const Text('Dots still open'),
          content: Text('${_arc.dotsDone} of ${ScanArc.dots} dots done ($n photos): parts of the hand were not '
              'photographed and the scan may fail. Go to the amber dots, or process anyway?'),
          actions: [
            TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Keep scanning')),
            FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Process')),
          ],
        ),
      );
      if (ok != true) return;
    }
    setState(() => _finishing = true);
    final up = _uploads!;
    String? problem;
    try {
      await up.drain();
      if (up.failed > 0) {
        up.retryFailed(); // one more round for photos that failed (Wi-Fi drop, tunnel hiccup)
        await up.drain();
      }
      await _sendLog();
      await _startWithRetry();
    } catch (e) {
      problem = '$e';
    }
    if (!mounted) return;
    if (problem != null) {
      setState(() => _finishing = false);
      await showDialog<void>(
        context: context,
        builder: (c) => AlertDialog(
          title: const Text('Could not reach the server'),
          content: const Text(
            'The photos are kept on the server. Tap Finish again, or open the scan from the '
            'home screen later and tap “Process now”.',
          ),
          actions: [FilledButton(onPressed: () => Navigator.pop(c), child: const Text('OK'))],
        ),
      );
      return;
    }
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (_) => ResultScreen(api: _api, scanId: _scanId!),
      ),
    );
  }

  Future<void> _startWithRetry() async {
    for (var attempt = 0; ; attempt++) {
      try {
        final a = _arm;
        return await _api.start(_scanId!, arm: a == null ? null : {'a': a.a, 'b': a.b, 'r': a.r});
      } catch (_) {
        if (attempt >= 4) rethrow;
        await Future<void>.delayed(Duration(seconds: 1 + attempt));
      }
    }
  }

  Future<void> _cancel() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Discard this scan?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('No')),
          FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Discard')),
        ],
      ),
    );
    if (ok != true) return;
    await _sendLog(); // kept in the server's logs folder even though the scan is deleted
    if (_scanId != null) _api.delete(_scanId!).ignore();
    if (mounted) Navigator.pop(context);
  }

  /// The light must not change once photos exist: photos before and after look different to the 3D
  /// reconstruction (it splits the scan in two) and switching the torch reconfigures the AR camera.
  bool get _torchLocked => _arc.photos > 0;

  Future<void> _toggleTorch() async {
    if (_torchLocked) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'The light must stay the same during a scan. Set the torch before the first photo, '
            'or start a new scan.',
          ),
        ),
      );
      return;
    }
    setState(() => _torch = !_torch);
    await LimbCapture.setTorch(_torch);
  }

  // ---- UI -----------------------------------------------------------------------------------------------------
  // The camera, a slim top bar, and one card at the bottom. Before the scan the card says what to do; while scanning
  // the guidance is on the camera image around the hand (WalkHud), so the eyes stay on the hand, and the card only
  // holds the counts and Complete.
  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.paddingOf(context);
    final scanning = _stage == Stage.scan;
    final banner = _banner != null && _now - _bannerAt < 2800 ? _banner : null;
    // The camera in a FIXED frame with the camera image's own 9:16 proportions (nothing cropped), sized once from
    // the screen: the instruction bar above it and the guide card below never change
    // its size (a frame that followed the card's height reset ARCore's display geometry mid-scan and broke tracking).
    // No 3D guide drawn on the arm: the ring around the hand shows the progress.
    final size = MediaQuery.sizeOf(context);
    const topH = 60.0, bottomH = 72.0, aspect = 9 / 16; // the camera image's own proportions (portrait)
    final camH = size.height - pad.top - 8 - topH - 8 - bottomH - pad.bottom - 20;
    final camW = math.min(size.width - 16, camH * aspect);
    final camHf = camW / aspect;
    final camTop = pad.top + 8 + topH + 8;
    final camera = Stack(
      children: [
        const Positioned.fill(child: ArCameraView()),
        Positioned.fill(
          child: LayoutBuilder(
            builder: (context, box) => GestureDetector(
              behavior: HitTestBehavior.translucent,
              onTapUp: (d) => _onTap(d.localPosition, box.biggest),
            ),
          ),
        ),
        if (_stage == Stage.fitting)
          Positioned.fill(
            child: LimbOverlay(
              limb: _limbN,
              pose: _poseN,
              locked: _lockedN,
              armDistance: () => 0.35,
              sweep: _sweep,
            ),
          ),
        if (_stage == Stage.place || _stage == Stage.confirm)
          Positioned.fill(child: IgnorePointer(child: CustomPaint(painter: _HandPainter(_handN)))),
        if (scanning) const Positioned.fill(child: IgnorePointer(child: Center(child: _Crosshair()))),
        if (scanning)
          Positioned.fill(
            child: IgnorePointer(
              child: WalkHud(
                arc: _arc,
                tick: _poseN,
                pulse: _pulse,
                text: _hint.text,
                icon: _hint.icon,
                colour: _toneColour,
                // the hints that name a way to go carry these icons
                arrow: switch (_hint.icon) {
                  Icons.west_rounded => -1,
                  Icons.east_rounded => 1,
                  Icons.swap_horiz_rounded => 2,
                  _ => 0,
                },
                mirror: _go == 'left',
              ),
            ),
          ),
        if (banner != null) Positioned(left: 12, right: 12, top: 12, child: _Banner(text: banner)),
      ],
    );
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          Positioned(
            left: (size.width - camW) / 2,
            top: camTop,
            width: camW,
            height: camHf,
            child: ClipRRect(borderRadius: BorderRadius.circular(18), child: camera),
          ),
          Positioned(left: 12, right: 12, top: pad.top + 8, child: _topBar()),
          if (_stage != Stage.done && !_finishing)
            Positioned(
              left: 12,
              right: 12,
              bottom: pad.bottom + 12,
              child: scanning ? _scanCard() : _placeCard(),
            ),
          if (_stage == Stage.done && !_finishing) _doneOverlay(),
          if (_finishing) _uploadingOverlay(),
        ],
      ),
    );
  }

  Widget _topBar() {
    final scanning = _stage == Stage.scan || _stage == Stage.done;
    return Glass(
      radius: 22,
      padding: const EdgeInsets.fromLTRB(4, 4, 4, 10),
      child: Column(
        children: [
          Row(
            children: [
              IconButton(onPressed: _cancel, icon: const Icon(Icons.close_rounded, color: Colors.white)),
              Expanded(
                child: Text(
                  scanning ? 'Scanning' : 'Point at the hand',
                  style: const TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w700),
                ),
              ),
              if (scanning)
                Padding(
                  padding: const EdgeInsets.only(right: 4),
                  child: Text(
                    '${(100 * _arc.progress).round()}%',
                    style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w800),
                  ),
                ),
              IconButton(
                onPressed: _toggleTorch,
                icon: Icon(
                  _torch ? Icons.flashlight_on_rounded : Icons.flashlight_off_rounded,
                  color: _torchLocked ? Colors.white38 : (_torch ? C.amber : Colors.white),
                ),
              ),
              IconButton(
                onPressed: () async {
                  await widget.config.setVoice(!widget.config.voice);
                  setState(() {});
                },
                icon: Icon(
                  widget.config.voice ? Icons.volume_up_rounded : Icons.volume_off_rounded,
                  color: Colors.white,
                ),
              ),
            ],
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 14),
            child: LinearProgressIndicator(
              value: scanning ? _arc.progress : 0,
              minHeight: 6,
              borderRadius: BorderRadius.circular(4),
              color: kGreen,
              backgroundColor: Colors.white24,
            ),
          ),
        ],
      ),
    );
  }

  /// Before scanning: tap the arm, then a short look at the placed cylinder.
  Widget _placeCard() {
    final Widget lead;
    final String title, sub;
    var titleColour = Colors.white;
    switch (_stage) {
      case Stage.fitting:
        lead = const SizedBox(
          width: 30,
          height: 30,
          child: CircularProgressIndicator(strokeWidth: 3, color: Colors.white),
        );
        title = 'Hand found…';
        sub = 'Keep the phone still for a moment: the scan starts by itself.';
      case Stage.calibrate:
        lead = const Icon(Icons.swap_vert_rounded, color: Colors.white, size: 32);
        title = 'Move the phone slowly toward the hand and back';
        sub = 'About a hand\'s length, once or twice. This sets the size of the scan; the walk starts by itself.';
      default:
        final warn = _hint.tone == Tone.warn;
        lead = Icon(_hint.icon, color: warn ? C.amber : Colors.white, size: 32);
        title = _hint.text;
        if (warn) titleColour = C.amber;
        final how = _upright
            ? 'Elbow resting on a table, hand up, fingers spread wide apart, completely still. Stand at the side, '
                'the whole hand on screen.'
            : 'Arm resting on the table, hand still, fingers spread wide apart, the whole hand on screen.';
        sub = '$how The scan starts by itself when the hand is found.';
    }
    return Glass(
      padding: const EdgeInsets.fromLTRB(18, 16, 14, 12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              lead,
              const SizedBox(width: 14),
              Expanded(
                child: Text(
                  title,
                  style: TextStyle(color: titleColour, fontSize: 21, fontWeight: FontWeight.w800, height: 1.15),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(sub, style: TextStyle(color: Colors.white.withValues(alpha: 0.75), fontSize: 14, height: 1.3)),
          if (_stage == Stage.calibrate) ...[
            const SizedBox(height: 12),
            LinearProgressIndicator(
              value: _calProgress,
              minHeight: 8,
              borderRadius: BorderRadius.circular(4),
              color: kGreen,
              backgroundColor: Colors.white24,
            ),
          ],
        ],
      ),
    );
  }

  Color get _toneColour => switch (_hint.tone) {
    Tone.good => kGreen,
    Tone.info => Colors.white,
    Tone.warn => C.amber,
  };

  /// While scanning: only the counts and Complete, nothing that needs reading while walking.
  Widget _scanCard() {
    final up = _uploads;
    final failed = (up?.failed ?? 0) > 0;
    final n = _arc.photos;
    final muted = TextStyle(color: Colors.white.withValues(alpha: 0.7), fontSize: 14);
    return Glass(
      padding: const EdgeInsets.fromLTRB(16, 4, 10, 4),
      child: SizedBox(
        height: 44,
        child: Row(
          children: [
            Text('${_arc.dotsDone}/${ScanArc.dots} dots · $n photos', style: muted),
            const SizedBox(width: 6),
            if (failed)
              const Icon(Icons.cloud_off_rounded, size: 16, color: C.red)
            else if ((up?.uploaded ?? 0) < n)
              Icon(Icons.cloud_upload_rounded, size: 16, color: Colors.white.withValues(alpha: 0.6))
            else if (n > 0)
              const Icon(Icons.cloud_done_rounded, size: 16, color: kGreen),
            const Spacer(),
            if (n >= 10)
              FilledButton(
                onPressed: _finishing ? null : _finish,
                style: FilledButton.styleFrom(
                  backgroundColor: _arc.canFinish ? kGreen : Colors.white24,
                  visualDensity: VisualDensity.compact,
                ),
                child: const Text('Complete'),
              ),
          ],
        ),
      ),
    );
  }

  Widget _doneOverlay() {
    final t = Theme.of(context).textTheme;
    return Positioned.fill(
      child: Container(
        color: const Color(0xE60A0F1C),
        alignment: Alignment.center,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TweenAnimationBuilder<double>(
              tween: Tween(begin: 0.4, end: 1),
              duration: const Duration(milliseconds: 450),
              curve: Curves.elasticOut,
              builder: (_, v, child) => Transform.scale(scale: v, child: child),
              child: Container(
                width: 92,
                height: 92,
                decoration: const BoxDecoration(color: kGreen, shape: BoxShape.circle),
                child: const Icon(Icons.check_rounded, color: Colors.white, size: 60),
              ),
            ),
            const SizedBox(height: 18),
            Text('Walk complete', style: t.headlineSmall),
            const SizedBox(height: 8),
            Text('Sending the photos…', style: TextStyle(color: Colors.white.withValues(alpha: 0.7))),
          ],
        ),
      ),
    );
  }

  Widget _uploadingOverlay() {
    final up = _uploads;
    final total = math.max(1, up?.queued ?? 1);
    return Positioned.fill(
      child: Container(
        color: const Color(0xE60A0F1C),
        alignment: Alignment.center,
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 120,
              height: 120,
              child: CircularProgressIndicator(
                value: (up?.uploaded ?? 0) / total,
                strokeWidth: 8,
                color: kGreen,
                backgroundColor: C.line,
              ),
            ),
            const SizedBox(height: 20),
            Text('Sending photos', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 6),
            Text('${up?.uploaded ?? 0} of $total', style: Theme.of(context).textTheme.bodyMedium),
          ],
        ),
      ),
    );
  }
}

/// "Tap here": a soft pulsing ring in the middle of the screen while waiting for the tap on the arm.
/// The hand the landmarker found, as a skeleton over the camera view while placing.
class _HandPainter extends CustomPainter {
  _HandPainter(this.hand) : super(repaint: hand);
  final ValueNotifier<HandInfo?> hand;
  static const _bones = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), //
    (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final h = hand.value;
    if (h == null) return;
    Offset at(int i) => Offset(h.lm[2 * i] * size.width, h.lm[2 * i + 1] * size.height);
    final line = Paint()
      ..color = kGreen.withValues(alpha: 0.9)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    for (final (a, b) in _bones) {
      canvas.drawLine(at(a), at(b), line);
    }
    final dot = Paint()..color = Colors.white;
    for (var i = 0; i < 21; i++) {
      canvas.drawCircle(at(i), i == 0 ? 7 : 4, dot);
    }
  }

  @override
  bool shouldRepaint(_HandPainter old) => old.hand != hand;
}

class _TapTarget extends StatefulWidget {
  const _TapTarget();

  @override
  State<_TapTarget> createState() => _TapTargetState();
}

class _TapTargetState extends State<_TapTarget> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this, duration: const Duration(milliseconds: 1400))
    ..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => IgnorePointer(
    child: AnimatedBuilder(
      animation: _c,
      builder: (_, _) => SizedBox(
        width: 120,
        height: 120,
        child: Stack(
          alignment: Alignment.center,
          children: [
            Container(
              width: 40 + 70 * _c.value,
              height: 40 + 70 * _c.value,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white.withValues(alpha: 0.8 * (1 - _c.value)), width: 3),
              ),
            ),
            const Icon(Icons.touch_app_rounded, color: Colors.white, size: 40),
          ],
        ),
      ),
    ),
  );
}

/// A ring finished: a short green message at the top.
class _Banner extends StatelessWidget {
  const _Banner({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) => TweenAnimationBuilder<double>(
    tween: Tween(begin: 0, end: 1),
    duration: const Duration(milliseconds: 300),
    builder: (_, v, child) =>
        Opacity(opacity: v, child: Transform.translate(offset: Offset(0, -12 * (1 - v)), child: child)),
    child: Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(color: kGreen, borderRadius: BorderRadius.circular(16)),
      child: Row(
        children: [
          const Icon(Icons.check_circle_rounded, color: Colors.white),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w800)),
          ),
        ],
      ),
    ),
  );
}

/// The aim mark in the middle of the camera while walking: keep it on the hand.
class _Crosshair extends StatelessWidget {
  const _Crosshair();

  @override
  Widget build(BuildContext context) => CustomPaint(size: const Size(44, 44), painter: _CrossPainter());
}

class _CrossPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = Colors.white.withValues(alpha: 0.85)
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    final w = size.width, h = size.height, k = w * 0.28;
    for (final (x, y, dx, dy) in [(0.0, 0.0, 1.0, 1.0), (w, 0.0, -1.0, 1.0), (0.0, h, 1.0, -1.0), (w, h, -1.0, -1.0)]) {
      canvas.drawLine(Offset(x, y), Offset(x + dx * k, y), p);
      canvas.drawLine(Offset(x, y), Offset(x, y + dy * k), p);
    }
    canvas.drawCircle(size.center(Offset.zero), 2.5, p..style = PaintingStyle.fill);
  }

  @override
  bool shouldRepaint(_CrossPainter old) => false;
}
