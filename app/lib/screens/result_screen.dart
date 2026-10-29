import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

import '../api/limbscan_api.dart';
import '../theme.dart';

/// Processing progress (stage by stage), then the measurements natively, with the 3D model a tap away.
class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.api, required this.scanId, this.name});
  final LimbScanApi api;
  final String scanId;
  final String? name;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

/// Server pipeline stages and the progress value at which each one starts.
const _stages = [
  (0.0, 'Receiving photos', Icons.cloud_upload_rounded),
  (0.05, 'Finding the arm in each photo', Icons.back_hand_rounded),
  (0.12, 'Matching photos', Icons.join_inner_rounded),
  (0.25, 'Camera positions', Icons.photo_camera_rounded),
  (0.35, 'Real-world scale', Icons.straighten_rounded),
  (0.45, 'Dense 3D reconstruction', Icons.grain_rounded),
  (0.9, 'Model & measurements', Icons.view_in_ar_rounded),
];

class _ResultScreenState extends State<ResultScreen> {
  ScanStatus? _status;
  Timer? _timer;
  final _started = DateTime.now();

  @override
  void initState() {
    super.initState();
    _poll();
    _timer = Timer.periodic(const Duration(seconds: 2), (_) => _poll());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _poll() async {
    try {
      final s = await widget.api.status(widget.scanId);
      if (!mounted) return;
      setState(() => _status = s);
      if (s.done || s.failed) _timer?.cancel();
    } catch (_) {/* keep polling */}
  }

  void _open3d() => Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => _ModelPage(url: widget.api.resultPageUrl(widget.scanId))));

  @override
  Widget build(BuildContext context) {
    final s = _status;
    return Scaffold(
      appBar: AppBar(title: Text(widget.name ?? 'Scan result')),
      body: SafeArea(
        child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 300),
          child: s == null
              ? const Center(child: CircularProgressIndicator())
              : s.stage == 'capturing'
                  ? _notSent(s)
                  : s.done
                  ? _Results(s, on3d: _open3d)
                  : s.failed
                      ? _failed(s)
                      : _processing(s),
        ),
      ),
    );
  }

  Widget _processing(ScanStatus s) {
    final t = Theme.of(context).textTheme;
    final current = _stages.lastIndexWhere((st) => s.progress >= st.$1);
    final secs = DateTime.now().difference(_started).inSeconds;
    return ListView(key: const ValueKey('proc'), padding: const EdgeInsets.all(20), children: [
      Center(
        child: SizedBox(
          width: 150,
          height: 150,
          child: Stack(fit: StackFit.expand, children: [
            CircularProgressIndicator(value: s.progress.clamp(0.02, 1.0), strokeWidth: 10, color: C.primary, backgroundColor: C.line),
            Center(child: Text('${(s.progress * 100).round()}%', style: t.headlineLarge)),
          ]),
        ),
      ),
      const SizedBox(height: 20),
      Center(child: Text('Building the 3D model', style: t.headlineSmall)),
      const SizedBox(height: 4),
      Center(child: Text('${s.photos} photos · ${secs}s · usually about 90 s', style: t.bodyMedium)),
      const SizedBox(height: 24),
      Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Column(children: [
            for (var i = 0; i < _stages.length; i++)
              ListTile(
                leading: i < current
                    ? const Icon(Icons.check_circle_rounded, color: C.teal)
                    : i == current
                        ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3))
                        : Icon(_stages[i].$3, color: C.line),
                title: Text(_stages[i].$2,
                    style: TextStyle(
                        color: i <= current ? C.text : C.muted, fontWeight: i == current ? FontWeight.w700 : FontWeight.w500)),
              ),
          ]),
        ),
      ),
      const SizedBox(height: 12),
      Text('You can leave this screen: the scan keeps processing on the server.', style: t.bodyMedium, textAlign: TextAlign.center),
    ]);
  }

  /// Photos are on the server but processing was never started (app closed, network drop at the end).
  Widget _notSent(ScanStatus s) {
    final t = Theme.of(context).textTheme;
    return Padding(
      key: const ValueKey('notsent'),
      padding: const EdgeInsets.all(24),
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        const Icon(Icons.cloud_done_rounded, size: 56, color: C.primary),
        const SizedBox(height: 16),
        Text('${s.photos} photos on the server', style: t.headlineSmall, textAlign: TextAlign.center),
        const SizedBox(height: 8),
        Text('This scan was not processed yet.', style: t.bodyMedium, textAlign: TextAlign.center),
        const SizedBox(height: 24),
        FilledButton.icon(
          onPressed: s.photos < 10
              ? null
              : () async {
                  try {
                    await widget.api.start(widget.scanId);
                    _timer?.cancel();
                    _timer = Timer.periodic(const Duration(seconds: 2), (_) => _poll());
                    _poll();
                  } catch (_) {
                    if (mounted) {
                      ScaffoldMessenger.of(context)
                          .showSnackBar(const SnackBar(content: Text('Server not reachable. Try again.')));
                    }
                  }
                },
          icon: const Icon(Icons.play_arrow_rounded),
          label: const Text('Process now'),
        ),
      ]),
    );
  }

  Widget _failed(ScanStatus s) {
    final t = Theme.of(context).textTheme;
    return Padding(
      key: const ValueKey('fail'),
      padding: const EdgeInsets.all(24),
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Container(
          width: 84,
          height: 84,
          decoration: BoxDecoration(color: C.red.withValues(alpha: 0.12), shape: BoxShape.circle),
          child: const Icon(Icons.error_outline_rounded, size: 44, color: C.red),
        ),
        const SizedBox(height: 18),
        Text('The scan could not be measured', style: t.headlineSmall, textAlign: TextAlign.center),
        const SizedBox(height: 10),
        Text(s.error ?? 'Processing failed', style: t.bodyMedium, textAlign: TextAlign.center),
        const SizedBox(height: 10),
        Text('Usually: the arm moved, or one side was missed. Try again with slower, fuller circles.',
            style: t.bodyMedium, textAlign: TextAlign.center),
        const SizedBox(height: 24),
        FilledButton(onPressed: () => Navigator.pop(context), child: const Text('Back')),
      ]),
    );
  }
}

class _Results extends StatelessWidget {
  const _Results(this.s, {required this.on3d});
  final ScanStatus s;
  final VoidCallback on3d;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final wrist = s.value('wrist_girth_mm');
    final scale = s.value('scale_check_pct');
    final cov = s.value('data_coverage_pct');
    final profile = ((s.measurements?['profile'] as List?) ?? const [])
        .map((e) => (((e as Map)['s_mm'] as num).toDouble(), (e['girth_mm'] as num).toDouble()))
        .toList();
    final lm = s.measurements?['landmarks'] as Map<String, dynamic>?;
    final marks = <double>[
      for (final k in ['wrist', 'wrist_plus_50', 'wrist_plus_100', 'wrist_plus_150'])
        if (lm?[k]?['s_mm'] is num) (lm![k]['s_mm'] as num).toDouble(),
    ];
    return ListView(key: const ValueKey('res'), padding: const EdgeInsets.fromLTRB(20, 8, 20, 24), children: [
      Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(24),
          gradient: const LinearGradient(colors: [Color(0xFF15264A), Color(0xFF0F1A30)]),
          border: Border.all(color: C.line),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('WRIST GIRTH', style: t.labelSmall),
          const SizedBox(height: 4),
          Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
            CountUp(wrist, decimals: 1, style: const TextStyle(color: C.text, fontSize: 48, fontWeight: FontWeight.w800, height: 1)),
            const Padding(padding: EdgeInsets.only(left: 6, bottom: 6), child: Text('mm', style: TextStyle(color: C.muted, fontSize: 18))),
          ]),
          const SizedBox(height: 14),
          Wrap(spacing: 6, runSpacing: 6, children: [
            if (scale != null)
              Pill('Scale check ${scale.abs() < 1 ? '✓' : '!'} ${scale.toStringAsFixed(1)}%',
                  color: scale.abs() < 1 ? C.teal : C.amber, icon: Icons.straighten_rounded),
            if (cov != null)
              Pill('Coverage ${cov.round()}%', color: cov >= 90 ? C.teal : C.amber, icon: Icons.donut_large_rounded),
          ]),
        ]),
      ),
      const SizedBox(height: 14),
      Row(children: [
        Expanded(child: _Metric('Wrist +50', s.value('wrist_plus_50_girth_mm'), 'mm', delay: 1)),
        const SizedBox(width: 10),
        Expanded(child: _Metric('Wrist +100', s.value('wrist_plus_100_girth_mm'), 'mm', delay: 2)),
        const SizedBox(width: 10),
        Expanded(child: _Metric('Wrist +150', s.value('wrist_plus_150_girth_mm'), 'mm', delay: 3)),
      ]),
      const SizedBox(height: 10),
      Row(children: [
        Expanded(child: _Metric('Forearm volume', s.value('volume_ml'), 'ml', decimals: 0, delay: 4)),
        const SizedBox(width: 10),
        Expanded(child: _Metric('Hand max', s.value('hand_max_girth_mm'), 'mm', delay: 5)),
        const SizedBox(width: 10),
        Expanded(child: _Metric('Length', s.value('length_mm'), 'mm', decimals: 0, delay: 6)),
      ]),
      if (profile.length > 3) ...[
        const SizedBox(height: 18),
        Text('GIRTH ALONG THE ARM', style: t.labelSmall),
        const SizedBox(height: 8),
        Card(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 16, 16, 12),
            child: SizedBox(height: 170, child: CustomPaint(size: Size.infinite, painter: _ProfilePainter(profile, marks))),
          ),
        ),
      ],
      const SizedBox(height: 18),
      FilledButton.icon(onPressed: on3d, icon: const Icon(Icons.view_in_ar_rounded), label: const Text('View 3D model')),
    ]);
  }
}

class _Metric extends StatelessWidget {
  const _Metric(this.label, this.value, this.unit, {this.decimals = 1, this.delay = 0});
  final String label;
  final double? value;
  final String unit;
  final int decimals;
  final int delay; // stagger: cards fill in one after another

  @override
  Widget build(BuildContext context) => TweenAnimationBuilder<double>(
        tween: Tween(begin: 0, end: 1),
        duration: Duration(milliseconds: 400 + 90 * delay),
        curve: Curves.easeOut,
        builder: (_, v, child) => Opacity(
          opacity: ((v * (4 + delay) - delay) / 4).clamp(0.0, 1.0),
          child: Transform.translate(offset: Offset(0, 12 * (1 - v)), child: child),
        ),
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(label, style: Theme.of(context).textTheme.bodyMedium, maxLines: 1, overflow: TextOverflow.ellipsis),
              const SizedBox(height: 6),
              FittedBox(
                fit: BoxFit.scaleDown,
                alignment: Alignment.centerLeft,
                child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
                  CountUp(value,
                      decimals: decimals,
                      style: const TextStyle(color: C.text, fontSize: 22, fontWeight: FontWeight.w800)),
                  Text(' $unit', style: const TextStyle(color: C.muted, fontSize: 12)),
                ]),
              ),
            ]),
          ),
        ),
      );
}

/// A number that counts up to its value when it appears.
class CountUp extends StatelessWidget {
  const CountUp(this.value, {super.key, this.decimals = 1, required this.style});
  final double? value;
  final int decimals;
  final TextStyle style;

  @override
  Widget build(BuildContext context) => value == null
      ? Text('–', style: style)
      : TweenAnimationBuilder<double>(
          tween: Tween(begin: 0, end: value!),
          duration: const Duration(milliseconds: 1100),
          curve: Curves.easeOutCubic,
          builder: (_, v, _) => Text(v.toStringAsFixed(decimals), style: style),
        );
}

/// Girth (mm) vs distance from the hand end (mm), with the wrist / +50 / +100 / +150 stations marked.
class _ProfilePainter extends CustomPainter {
  _ProfilePainter(this.pts, this.marks);
  final List<(double, double)> pts;
  final List<double> marks;

  @override
  void paint(Canvas canvas, Size size) {
    final xs = pts.map((p) => p.$1), ys = pts.map((p) => p.$2);
    final x0 = xs.reduce(math.min), x1 = xs.reduce(math.max);
    final y0 = ys.reduce(math.min) - 10, y1 = ys.reduce(math.max) + 10;
    const left = 34.0, bottom = 18.0;
    Offset map(double x, double y) => Offset(
        left + (x - x0) / (x1 - x0) * (size.width - left), (size.height - bottom) * (1 - (y - y0) / (y1 - y0)));
    final grid = Paint()..color = C.line;
    final label = TextStyle(color: C.muted, fontSize: 10);
    for (var k = 0; k <= 3; k++) {
      final y = y0 + (y1 - y0) * k / 3;
      final p = map(x0, y);
      canvas.drawLine(p, Offset(size.width, p.dy), grid);
      _text(canvas, '${y.round()}', Offset(0, p.dy - 6), label);
    }
    for (final m in marks) {
      final p = map(m, y0);
      canvas.drawLine(Offset(p.dx, 0), Offset(p.dx, size.height - bottom), Paint()..color = C.primary.withValues(alpha: 0.35));
    }
    final path = Path();
    for (var i = 0; i < pts.length; i++) {
      final p = map(pts[i].$1, pts[i].$2);
      i == 0 ? path.moveTo(p.dx, p.dy) : path.lineTo(p.dx, p.dy);
    }
    canvas.drawPath(
        path,
        Paint()
          ..color = C.teal
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5
          ..strokeJoin = StrokeJoin.round);
    _text(canvas, 'hand', Offset(left, size.height - 12), label);
    _text(canvas, 'elbow →', Offset(size.width - 44, size.height - 12), label);
  }

  void _text(Canvas c, String s, Offset o, TextStyle st) =>
      (TextPainter(text: TextSpan(text: s, style: st), textDirection: TextDirection.ltr)..layout()).paint(c, o);

  @override
  bool shouldRepaint(covariant _ProfilePainter o) => o.pts != pts;
}

class _ModelPage extends StatefulWidget {
  const _ModelPage({required this.url});
  final String url;

  @override
  State<_ModelPage> createState() => _ModelPageState();
}

class _ModelPageState extends State<_ModelPage> {
  late final _web = WebViewController()
    ..setJavaScriptMode(JavaScriptMode.unrestricted)
    ..setBackgroundColor(C.bg)
    ..loadRequest(Uri.parse(widget.url));

  @override
  Widget build(BuildContext context) =>
      Scaffold(appBar: AppBar(title: const Text('3D model')), body: WebViewWidget(controller: _web));
}
