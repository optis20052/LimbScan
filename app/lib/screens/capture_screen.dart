import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:limb_capture/limb_capture.dart';
import 'package:path_provider/path_provider.dart';

import '../api/limbscan_api.dart';
import '../capture/guidance.dart';
import '../capture/upload_queue.dart';
import '../config.dart';
import '../theme.dart';
import '../widgets/ar_halo.dart';
import '../widgets/coverage_ring.dart';
import '../widgets/illustrations.dart';
import '../widgets/limb_overlay.dart';
import 'result_screen.dart';

enum Tone { good, info, warn }

class Hint {
  const Hint(this.text, this.tone, this.icon);
  final String text;
  final Tone tone;
  final IconData icon;
}

/// Guided AR capture. Photos are taken automatically only when the ARM is in view (on-device segmentation),
/// the view is new, sharp and steady. Coverage is counted around the arm; arrows, a distance gauge and voice
/// say where to go next. Scale comes from the phone's AR tracking (no card).
class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key, required this.config});
  final AppConfig config;

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> with TickerProviderStateMixin {
  late final LimbScanApi _api = LimbScanApi(widget.config.serverUrl);
  late final List<Pass> _passes = widget.config.pose == 'resting' ? restingPasses : uprightPasses;
  late final Coverage _cov = Coverage(_passes);
  late final AnimationController _pulse = AnimationController(vsync: this, duration: const Duration(milliseconds: 900))
    ..repeat(reverse: true);
  late final AnimationController _sweep = AnimationController(vsync: this, duration: const Duration(milliseconds: 2400))
    ..repeat();
  // high-rate state for the AR overlays: they repaint on their own, without rebuilding the whole screen
  final _poseN = ValueNotifier<ArPose?>(null);
  final _limbN = ValueNotifier<LimbInfo?>(null);
  final _lockedN = ValueNotifier<bool>(false);
  final Map<int, int> _pops = {};
  int? _targetSeg;
  int _builtAt = 0;
  final _tts = FlutterTts();
  StreamSubscription<ArPose>? _poseSub;
  StreamSubscription<LimbInfo>? _limbSub;
  StreamSubscription<String>? _errSub;
  UploadQueue? _uploads;
  String? _scanId;
  String? _dir;

  int _pass = 0;
  bool _running = false; // a pass is active (not showing the intro card)
  bool _capturing = false;
  bool _finishing = false;
  bool _torch = false;
  int _lastShotMs = 0;
  int? _trackingSince;
  double _pathSinceTracking = 0; // metres moved since tracking (re)started: AR scale settles with motion
  final List<double> _lumBase = []; // arm brightness at the start of the scan
  final List<double> _lumRecent = [];
  ArPose? _pose;
  double _rotSpeed = 0;
  LimbInfo? _limb;
  int? _armGoodSince;
  int? _firstPoseMs;
  Hint _hint = const Hint(
    'Move the phone slowly so it can start tracking',
    Tone.info,
    Icons.screen_rotation_alt_rounded,
  );
  int _hintAt = 0;
  int _spokeAt = 0;
  String? _dirHint;
  double? _distance;
  final List<double> _sharpHist = [];
  int _countdown = 0;
  Timer? _countdownTimer;
  int _photos = 0;

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
    _poseSub = LimbCapture.poses.listen(_onPose);
    _limbSub = LimbCapture.limbs.listen(_onLimb);
    _errSub = LimbCapture.errors.listen(
      (m) => _setHint(Hint('AR error: $m', Tone.warn, Icons.error_outline_rounded), force: true),
    );
    _showIntro(0);
  }

  @override
  void dispose() {
    _poseSub?.cancel();
    _limbSub?.cancel();
    _errSub?.cancel();
    _countdownTimer?.cancel();
    _pulse.dispose();
    _sweep.dispose();
    _poseN.dispose();
    _limbN.dispose();
    _lockedN.dispose();
    _tts.stop();
    if (_torch) LimbCapture.setTorch(false).ignore();
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
    if (h.text == _hint.text || (!force && _now - _hintAt < 1200)) return;
    _hint = h;
    _hintAt = _now;
    if (h.tone == Tone.warn) _speak(h.text.replaceAll(':', ','));
  }

  // ---- passes ------------------------------------------------------------------------------------------------
  void _showIntro(int p) {
    setState(() {
      _pass = p;
      _running = false;
      _countdown = p > 0 ? 5 : 0;
    });
    _speak('${_passes[p].title}. ${_passes[p].instruction}', urgent: true);
    _countdownTimer?.cancel();
    if (p > 0) {
      // later passes start by themselves: no tap with the scanned hand, so it stays still
      _countdownTimer = Timer.periodic(const Duration(seconds: 1), (t) {
        if (!mounted) return t.cancel();
        setState(() => _countdown--);
        if (_countdown <= 0) {
          t.cancel();
          _beginPass();
        }
      });
    }
  }

  void _beginPass() {
    _countdownTimer?.cancel();
    setState(() => _running = true);
  }

  void _passDone() {
    _running = false;
    HapticFeedback.heavyImpact();
    if (_pass < _passes.length - 1) {
      _speak('Pass complete', urgent: true);
      _showIntro(_pass + 1);
    } else {
      _speak('All done. Tap finish.', urgent: true);
      setState(() {});
    }
  }

  // ---- on-device arm detection --------------------------------------------------------------------------------
  /// No limb events at all after a few seconds = detector unavailable on this phone: don't block capture.
  bool get _detectorOff => _limb == null && _firstPoseMs != null && _now - _firstPoseMs! > 4000;

  bool get _limbFresh => _limb != null && _now - _limb!.t < 700;

  bool _good(LimbInfo l) => l.visible && !l.backlit && !l.tooSmall && !l.cutOff;

  bool get _armOk => _detectorOff || (_limbFresh && _good(_limb!));

  bool get _locked => _armGoodSince != null && _now - _armGoodSince! > 400;

  /// Light on the arm, judged before the first photo (it must then stay the same for the whole scan).
  String get _light {
    final l = _limbFresh ? _limb : null;
    if (l == null || !l.visible) return 'unknown';
    if (l.backlit) return 'backlit';
    if (l.limbLum < 60) return 'dark';
    return 'ok';
  }

  static double _median(List<double> v) => ([...v]..sort())[v.length ~/ 2];

  /// The arm got clearly brighter or darker than at the start: lights switched, torch, blinds, clouds.
  bool get _lightChanged =>
      _lumBase.length >= 8 &&
      _lumRecent.length >= 5 &&
      ((_median(_lumRecent) / _median(_lumBase)) > 1.5 || (_median(_lumRecent) / _median(_lumBase)) < 0.65);

  void _onLimb(LimbInfo l) {
    final wasLocked = _locked;
    _limb = l;
    _limbN.value = l;
    if (_good(l)) {
      _armGoodSince ??= _now;
      if (_running) {
        if (_lumBase.length < 8) {
          _lumBase.add(l.limbLum);
        } else {
          _lumRecent.add(l.limbLum);
          if (_lumRecent.length > 5) _lumRecent.removeAt(0);
        }
      }
    } else {
      _armGoodSince = null;
    }
    _lockedN.value = _locked;
    if (!wasLocked && _locked) HapticFeedback.lightImpact();
    // hands-free start: pointing at the arm for a moment starts the first pass
    final p = _pose;
    if (!_running &&
        !_finishing &&
        _pass == 0 &&
        _photos == 0 &&
        p != null &&
        p.isTracking &&
        _light == 'ok' &&
        _armGoodSince != null &&
        _now - _armGoodSince! > 1200) {
      _speak('Found your arm. Now move slowly around it.', urgent: true);
      HapticFeedback.mediumImpact();
      _beginPass();
    }
    _refresh(now: wasLocked != _locked);
  }

  // ---- per pose event: guidance + capture decision ------------------------------------------------------------
  void _onPose(ArPose p) {
    _firstPoseMs ??= _now;
    final prev = _pose;
    _pose = p;
    _poseN.value = p;
    if (prev != null && p.isTracking && prev.isTracking) _pathSinceTracking += _dist(p.pos, prev.pos);
    if (prev != null && p.t > prev.t) {
      final dt = (p.t - prev.t) / 1000.0;
      _rotSpeed = 0.7 * _rotSpeed + 0.3 * angleDeg(p.fwd, prev.fwd) / dt;
    }
    if (!p.isTracking) {
      _trackingSince = null;
      _pathSinceTracking = 0;
      _setHint(_trackingHint(p.reason));
      _refresh();
      return;
    }
    _trackingSince ??= p.t;
    final c = _cov.centre;
    _distance = c == null ? null : _dist(p.pos, c);
    if (!_running || _finishing) {
      _refresh();
      return;
    }
    // ARCore warm-up (what other apps run as a "calibration" step): tracking for 1.5 s AND ~8 cm of motion, so
    // the AR scale has settled before the first photo. Users are moving anyway, so it is invisible.
    final warm = p.t - _trackingSince! > 1500 && _pathSinceTracking > 0.08;
    final ph = _passes[_pass];
    final dist = _distance;
    final el = elevation(p.fwd);
    final steady = _rotSpeed < 90;
    final dup = isDuplicate(_cov.shots, p.pos, p.fwd);
    final az = c == null ? null : azimuthAround(c, p.pos);
    _dirHint = (c == null || az == null) ? null : nextDirection(_cov.bins[_pass], az, p.right, c, p.pos);
    final dir = _dirHint;
    final l = _limbFresh ? _limb : null;
    _targetSeg = az == null ? null : _nearestMissing(_cov.bins[_pass], az);

    if (!warm) {
      _setHint(const Hint('Good, keep moving slowly…', Tone.info, Icons.screen_rotation_alt_rounded));
    } else if (!_detectorOff && (l == null || !l.visible)) {
      _setHint(const Hint('Point the camera at the arm', Tone.warn, Icons.center_focus_strong_rounded));
    } else if (l != null && l.backlit) {
      _setHint(
        const Hint(
          'Bright light behind the arm: step around so the window is behind you',
          Tone.warn,
          Icons.wb_sunny_rounded,
        ),
      );
    } else if (_lightChanged) {
      _setHint(
        const Hint('The light changed: keep lamps, blinds and torch as they were', Tone.warn, Icons.light_mode_rounded),
      );
    } else if (l != null && l.tooSmall) {
      _setHint(const Hint('Move closer to the arm', Tone.warn, Icons.zoom_in_rounded));
    } else if (l != null && l.cutOff) {
      _setHint(const Hint('Move back a little: the arm is cut off', Tone.warn, Icons.zoom_out_map_rounded));
    } else if (!steady) {
      _setHint(const Hint('A bit slower', Tone.warn, Icons.slow_motion_video_rounded));
    } else if (dist != null && dist > 0.6) {
      _setHint(const Hint('Move closer (30–40 cm)', Tone.warn, Icons.zoom_in_rounded));
    } else if (dist != null && dist < 0.18) {
      _setHint(const Hint('Too close: move back a little', Tone.warn, Icons.zoom_out_rounded));
    } else if (dup) {
      _setHint(
        Hint(
          dir == null ? 'Got this view ✓ Move to a new spot' : 'Got this view ✓ Keep going $dir',
          Tone.good,
          Icons.check_circle_rounded,
        ),
      );
    } else if (el < ph.minElev - 10 || el > ph.maxElev + 10) {
      _setHint(
        Hint(
          el < ph.minElev ? 'Tilt the phone up a little' : 'Tilt the phone down a little',
          Tone.info,
          el < ph.minElev ? Icons.arrow_upward_rounded : Icons.arrow_downward_rounded,
        ),
      );
    } else if (dir != null && _cov.photos(_pass) > 4) {
      _setHint(
        Hint('Keep going $dir around the arm', Tone.good, dir == 'left' ? Icons.west_rounded : Icons.east_rounded),
      );
    } else {
      _setHint(
        Hint(
          _cov.photos(_pass) < 3 ? 'Start moving around the arm slowly' : 'Smooth and steady',
          Tone.good,
          Icons.motion_photos_on_rounded,
        ),
      );
    }

    final okDist = dist == null || (dist > 0.15 && dist < 0.65);
    if (warm && steady && !dup && okDist && _armOk && !_capturing && p.t - _lastShotMs > 300 && _scanId != null) {
      _capture(p);
    }
    if (_cov.progress(_pass) >= 1) {
      _passDone();
      _refresh(now: true);
    } else {
      _refresh();
    }
  }

  int? _nearestMissing(List<int> bins, double az) {
    final n = bins.length, cur = (az / (360 / n)).floor() % n;
    for (var k = 1; k <= n ~/ 2; k++) {
      if (bins[(cur + k) % n] == 0) return (cur + k) % n;
      if (bins[(cur - k + n) % n] == 0) return (cur - k + n) % n;
    }
    return null;
  }

  double _dist(Vec3 a, Vec3 b) =>
      math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));

  Hint _trackingHint(String? reason) => switch (reason) {
    'INSUFFICIENT_LIGHT' => Hint(
      _photos == 0 ? 'Too dark: add light or turn on the torch' : 'Too dark: turn on a room light',
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

  Future<void> _capture(ArPose p) async {
    _capturing = true;
    _lastShotMs = p.t;
    try {
      final f = await LimbCapture.capture(_dir!);
      // blur check against the TYPICAL recent frame (not the best one ever seen)
      _sharpHist.add(f.sharpness);
      if (_sharpHist.length > 25) _sharpHist.removeAt(0);
      final sorted = [..._sharpHist]..sort();
      final median = sorted[sorted.length ~/ 2];
      if (_sharpHist.length >= 5 && f.sharpness < 0.7 * median) {
        File(f.path).delete().ignore();
        _setHint(const Hint('Hold still a moment: blurry', Tone.info, Icons.blur_on_rounded));
        return;
      }
      _cov.add(Shot(p.pos, p.fwd, _pass));
      _photos++;
      final c = _cov.centre;
      if (c != null) _pops[(azimuthAround(c, p.pos) / 15).floor() % 24] = _now;
      HapticFeedback.selectionClick();
      _uploads?.add(f);
    } on PlatformException catch (_) {
      // not tracking / busy: just try again on a later pose
    } finally {
      _capturing = false;
    }
  }

  Future<void> _finish() async {
    if (_scanId == null || _uploads == null) return;
    if (_photos < 20) {
      final ok = await showDialog<bool>(
        context: context,
        builder: (c) => AlertDialog(
          title: const Text('Only a few photos'),
          content: Text('$_photos photos: the scan may fail. Process anyway?'),
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
        return await _api.start(_scanId!);
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
    if (_scanId != null) _api.delete(_scanId!).ignore();
    if (mounted) Navigator.pop(context);
  }

  /// The light must not change once photos exist: photos before and after look different to the 3D
  /// reconstruction (it splits the scan in two) and switching the torch reconfigures the AR camera.
  bool get _torchLocked => _photos > 0;

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
  @override
  Widget build(BuildContext context) {
    final c = _cov.centre;
    final az = (c != null && _pose != null) ? azimuthAround(c, _pose!.pos) : null;
    final allDone = !_running && _pass == _passes.length - 1 && _cov.progress(_pass) >= 1;
    final pad = MediaQuery.paddingOf(context);
    final showArrow = _running && _dirHint != null && _cov.photos(_pass) > 4 && _armOk;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          const Positioned.fill(child: ArCameraView()),
          if (!_detectorOff)
            Positioned.fill(
              child: LimbOverlay(
                limb: _limbN,
                pose: _poseN,
                locked: _lockedN,
                armDistance: () => _distance ?? 0.35,
                sweep: _sweep,
              ),
            ),
          Positioned.fill(
            child: ArHalo(
              pose: _poseN,
              coverage: _cov,
              pass: _pass,
              pops: _pops,
              tick: _sweep,
              target: _running ? _targetSeg : null,
            ),
          ),
          if (showArrow)
            Positioned(
              top: 0,
              bottom: 0,
              left: _dirHint == 'left' ? 0 : null,
              right: _dirHint == 'right' ? 0 : null,
              child: Center(
                child: EdgeArrow(right: _dirHint == 'right', pulse: _pulse),
              ),
            ),
          Positioned(left: 12, right: 12, top: pad.top + 8, child: _topBar()),
          Positioned(
            left: 12,
            top: pad.top + 112,
            child: Glass(
              radius: 20,
              padding: const EdgeInsets.all(8),
              child: CoverageRing(bins: _cov.bins[_pass], azimuth: az, progress: _cov.progress(_pass), size: 84),
            ),
          ),
          Positioned(
            right: 12,
            top: pad.top + 112,
            child: Glass(
              radius: 20,
              padding: const EdgeInsets.fromLTRB(4, 10, 4, 10),
              child: DistanceGauge(metres: _distance),
            ),
          ),
          if (_running || allDone || _finishing)
            Positioned(left: 12, right: 12, bottom: pad.bottom + 12, child: _bottomPanel(allDone)),
          if (!_running && !allDone && !_finishing) _introCard(_passes[_pass]),
          if (_finishing) _uploadingOverlay(),
        ],
      ),
    );
  }

  Widget _topBar() => Glass(
    radius: 22,
    padding: const EdgeInsets.fromLTRB(4, 6, 4, 10),
    child: Column(
      children: [
        Row(
          children: [
            IconButton(
              onPressed: _cancel,
              icon: const Icon(Icons.close_rounded, color: Colors.white),
            ),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('PASS ${_pass + 1} OF ${_passes.length}', style: Theme.of(context).textTheme.labelSmall),
                  Text(
                    _passes[_pass].title,
                    style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700),
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
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
              icon: Icon(widget.config.voice ? Icons.volume_up_rounded : Icons.volume_off_rounded, color: Colors.white),
            ),
          ],
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Row(
            children: [
              for (var k = 0; k < _passes.length; k++)
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 3),
                    child: LinearProgressIndicator(
                      value: k < _pass ? 1 : (k == _pass ? _cov.progress(k) : 0),
                      minHeight: 6,
                      borderRadius: BorderRadius.circular(4),
                      color: k < _pass ? C.teal : C.primary,
                      backgroundColor: Colors.white24,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ],
    ),
  );

  Widget _bottomPanel(bool allDone) {
    final up = _uploads;
    final tone = switch (_hint.tone) {
      Tone.good => C.teal,
      Tone.info => Colors.white,
      Tone.warn => C.amber,
    };
    return Glass(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _StatusDot('AR', _pose?.isTracking ?? false),
              const SizedBox(width: 6),
              _StatusDot(_detectorOff ? 'Arm check off' : (_locked ? 'Arm locked' : 'Arm'), _detectorOff || _locked),
              const SizedBox(width: 6),
              _StatusDot('Steady', _rotSpeed < 90),
              const Spacer(),
              if ((up?.failed ?? 0) > 0) const Pill('Upload issue', color: C.red, icon: Icons.cloud_off_rounded),
            ],
          ),
          const SizedBox(height: 12),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: Row(
              key: ValueKey(_hint.text),
              children: [
                Icon(_hint.icon, color: tone, size: 28),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    _hint.text,
                    style: TextStyle(color: tone, fontSize: 19, fontWeight: FontWeight.w800, height: 1.2),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Text(
                '$_photos',
                style: const TextStyle(color: Colors.white, fontSize: 30, fontWeight: FontWeight.w800),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'photos · ${up?.uploaded ?? 0} sent to server',
                      style: const TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                    const SizedBox(height: 5),
                    LinearProgressIndicator(
                      value: _photos == 0 ? 0 : (up?.uploaded ?? 0) / _photos,
                      minHeight: 4,
                      borderRadius: BorderRadius.circular(4),
                      color: C.teal,
                      backgroundColor: Colors.white24,
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              if (_pass < _passes.length - 1) ...[
                Expanded(
                  child: OutlinedButton(
                    onPressed: _running ? () => _showIntro(_pass + 1) : null,
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.white,
                      minimumSize: const Size.fromHeight(50),
                    ),
                    child: const Text('Next pass'),
                  ),
                ),
                const SizedBox(width: 10),
              ],
              Expanded(
                child: FilledButton(
                  onPressed: _finishing ? null : _finish,
                  style: FilledButton.styleFrom(
                    backgroundColor: allDone ? C.teal : C.primary,
                    minimumSize: const Size.fromHeight(50),
                  ),
                  child: Text(allDone ? 'Finish & measure' : 'Finish'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _introCard(Pass ph) {
    final t = Theme.of(context).textTheme;
    final resting = widget.config.pose == 'resting';
    final scene = resting
        ? (_pass == 0 ? Scene.passAbove : Scene.passSides)
        : (_pass == 0 ? Scene.passHand : Scene.passForearm);
    final first = _pass == 0;
    // first pass: compact card, the camera stays visible; pointing at the arm starts it (no tap needed)
    return Positioned.fill(
      child: Container(
        color: first ? const Color(0x33000000) : const Color(0x99000000),
        alignment: Alignment.bottomCenter,
        padding: EdgeInsets.fromLTRB(12, 0, 12, MediaQuery.paddingOf(context).bottom + 12),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: C.surface.withValues(alpha: first ? 0.92 : 1),
            borderRadius: BorderRadius.circular(28),
            border: Border.all(color: C.line),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('PASS ${_pass + 1} OF ${_passes.length}', style: t.labelSmall),
              const SizedBox(height: 4),
              Text(ph.title, style: t.headlineSmall),
              Illustration(scene, height: first ? 110 : 150),
              Text(ph.instruction, style: t.bodyLarge),
              const SizedBox(height: 16),
              if (first) ...[
                _armSearch(t),
                const SizedBox(height: 12),
                TextButton(onPressed: _beginPass, child: const Text('Start without waiting')),
              ] else
                FilledButton(
                  onPressed: _beginPass,
                  child: Text(
                    _countdown > 0 ? 'Starting in $_countdown  ·  tap to start now' : 'Start pass ${_pass + 1}',
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  /// Live "looking for the arm" + light check; the pass starts by itself once the arm is locked in good light.
  Widget _armSearch(TextTheme t) {
    final found = _locked;
    final light = _light;
    final (lightText, lightColour, lightIcon) = switch (light) {
      'ok' => ('Light is good', C.teal, Icons.wb_sunny_rounded),
      'dark' => (
        'Too dark: switch on a room light (torch only if there is no other light)',
        C.amber,
        Icons.nightlight_round,
      ),
      'backlit' => (
        'Window or lamp behind the arm: turn so the light is behind you',
        C.amber,
        Icons.wb_twilight_rounded,
      ),
      _ => ('Checking the light…', C.muted, Icons.wb_sunny_outlined),
    };
    final starting = found && light == 'ok';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _statusRow(
          starting ? Icons.check_circle_rounded : null,
          starting
              ? 'Arm found. Starting…'
              : (found ? 'Arm found' : 'Point the camera at the arm. The scan starts by itself.'),
          found ? C.teal : C.primary,
        ),
        const SizedBox(height: 8),
        _statusRow(
          lightIcon,
          lightText,
          lightColour,
          trailing: light == 'dark' && !_torchLocked
              ? TextButton(onPressed: _toggleTorch, child: Text(_torch ? 'Torch off' : 'Use torch'))
              : null,
        ),
      ],
    );
  }

  Widget _statusRow(IconData? icon, String text, Color colour, {Widget? trailing}) => AnimatedContainer(
    duration: const Duration(milliseconds: 250),
    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
    decoration: BoxDecoration(
      color: colour.withValues(alpha: 0.12),
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: colour.withValues(alpha: 0.45)),
    ),
    child: Row(
      children: [
        SizedBox(
          width: 24,
          height: 24,
          child: icon != null ? Icon(icon, color: colour) : CircularProgressIndicator(strokeWidth: 3, color: colour),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            text,
            style: TextStyle(color: C.text, fontWeight: FontWeight.w700, fontSize: 14),
          ),
        ),
        ?trailing,
      ],
    ),
  );

  Widget _uploadingOverlay() {
    final up = _uploads;
    final total = math.max(1, up?.queued ?? 1);
    return Positioned.fill(
      child: Container(
        color: const Color(0xCC0A0F1C),
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
                color: C.teal,
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

class _StatusDot extends StatelessWidget {
  const _StatusDot(this.label, this.ok);
  final String label;
  final bool ok;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
    decoration: BoxDecoration(
      color: (ok ? C.teal : C.red).withValues(alpha: 0.16),
      borderRadius: BorderRadius.circular(99),
    ),
    child: Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.circle, size: 8, color: ok ? C.teal : C.red),
        const SizedBox(width: 5),
        Text(
          label,
          style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w700),
        ),
      ],
    ),
  );
}
