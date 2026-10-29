import 'package:flutter/material.dart';
import 'package:limb_capture/limb_capture.dart';
import 'package:permission_handler/permission_handler.dart';

import '../api/limbscan_api.dart';
import '../config.dart';
import '../theme.dart';
import '../widgets/illustrations.dart';
import 'result_screen.dart';
import 'settings_screen.dart';
import 'setup_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.config});
  final AppConfig config;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late LimbScanApi _api = LimbScanApi(widget.config.serverUrl);
  List<ScanSummary>? _scans;
  bool? _online;

  @override
  void initState() {
    super.initState();
    _load();
    _sendCrashReport();
  }

  /// If the last session crashed, send the native report to the server so it can be diagnosed.
  Future<void> _sendCrashReport() async {
    try {
      final r = await LimbCapture.lastCrash();
      if (r != null) await _api.reportCrash(r);
    } catch (_) {}
  }

  Future<void> _load() async {
    try {
      final s = await _api.listScans();
      if (mounted) setState(() { _scans = s; _online = true; });
    } catch (e) {
      if (mounted) setState(() => _online = false);
    }
  }

  Future<void> _newScan() async {
    var ar = await LimbCapture.checkAvailability();
    if (ar == 'install_required') {
      await LimbCapture.requestInstall();
      ar = await LimbCapture.checkAvailability();
    }
    if (ar != 'supported') {
      _snack(ar == 'unsupported' ? 'This phone does not support ARCore.' : 'AR is not ready yet ($ar). Try again.');
      return;
    }
    if (!(await Permission.camera.request()).isGranted) {
      _snack('Camera permission is needed to scan.');
      return;
    }
    if (!mounted) return;
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => SetupScreen(config: widget.config)));
    _load();
  }

  Future<void> _settings() async {
    final changed =
        await Navigator.of(context).push<bool>(MaterialPageRoute(builder: (_) => SettingsScreen(config: widget.config)));
    if (changed == true) {
      _api = LimbScanApi(widget.config.serverUrl);
      _load();
    }
  }

  void _snack(String m) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(m)));

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final scans = _scans ?? const <ScanSummary>[];
    return Scaffold(
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _load,
          child: ListView(padding: const EdgeInsets.fromLTRB(20, 12, 20, 32), children: [
            Row(children: [
              const _Logo(),
              const SizedBox(width: 10),
              Text('LimbScan', style: t.titleLarge),
              const Spacer(),
              _ServerPill(online: _online, onTap: _settings),
              IconButton(onPressed: _settings, icon: const Icon(Icons.tune_rounded, color: C.muted)),
            ]),
            const SizedBox(height: 18),
            _hero(t),
            const SizedBox(height: 28),
            Row(children: [
              Text('RECENT SCANS', style: t.labelSmall),
              const Spacer(),
              if (_scans != null) Text('${scans.length}', style: t.bodyMedium),
            ]),
            const SizedBox(height: 10),
            if (_online == false)
              _infoCard(Icons.cloud_off_rounded, 'Server not reachable',
                  '${widget.config.serverUrl}\nThe phone must be on the same Wi-Fi as the scan PC.', C.amber),
            if (_scans == null && _online != false)
              const Padding(padding: EdgeInsets.all(32), child: Center(child: CircularProgressIndicator())),
            if (_scans != null && scans.isEmpty)
              _infoCard(Icons.view_in_ar_rounded, 'No scans yet', 'Your scans and measurements will appear here.', C.muted),
            for (final s in scans) ...[_ScanTile(s, onTap: () => _open(s)), const SizedBox(height: 10)],
          ]),
        ),
      ),
    );
  }

  Future<void> _open(ScanSummary s) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => ResultScreen(api: _api, scanId: s.id, name: s.name)));
    _load();
  }

  Widget _hero(TextTheme t) => Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(28),
          gradient: const LinearGradient(
              begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [Color(0xFF15264A), Color(0xFF0F1A30)]),
          border: Border.all(color: C.line),
        ),
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('HAND & FOREARM', style: t.labelSmall),
          const SizedBox(height: 6),
          Text('Measure a limb in under a minute', style: t.headlineSmall),
          const SizedBox(height: 6),
          Text('Walk the phone around the arm. Photos are taken for you; '
              'the 3D model and girths come back in about 90 seconds.', style: t.bodyMedium),
          const Illustration(Scene.upright, height: 170),
          const Row(children: [
            Pill('No card needed', color: C.teal, icon: Icons.check_rounded),
            SizedBox(width: 6),
            Pill('AR millimetre scale', color: C.primary, icon: Icons.straighten_rounded),
          ]),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _newScan,
            icon: const Icon(Icons.view_in_ar_rounded),
            label: const Text('New scan'),
          ),
        ]),
      );

  Widget _infoCard(IconData icon, String title, String text, Color color) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(children: [
            Icon(icon, color: color, size: 28),
            const SizedBox(width: 14),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(title, style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 2),
                Text(text, style: Theme.of(context).textTheme.bodyMedium),
              ]),
            ),
          ]),
        ),
      );
}

class _Logo extends StatelessWidget {
  const _Logo();

  @override
  Widget build(BuildContext context) => Container(
        width: 34,
        height: 34,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10),
          gradient: const LinearGradient(colors: [C.primary, C.teal]),
        ),
        child: const Icon(Icons.back_hand_rounded, color: Colors.white, size: 20),
      );
}

class _ServerPill extends StatelessWidget {
  const _ServerPill({required this.online, required this.onTap});
  final bool? online;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: switch (online) {
          true => const Pill('Server', color: C.teal, icon: Icons.circle),
          false => const Pill('Offline', color: C.amber, icon: Icons.circle),
          null => const Pill('…'),
        },
      );
}

class _ScanTile extends StatelessWidget {
  const _ScanTile(this.s, {required this.onTap});
  final ScanSummary s;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final (color, label, icon) = switch (s.stage) {
      'done' => (C.teal, 'Ready', Icons.check_circle_rounded),
      'failed' => (C.red, 'Failed', Icons.error_rounded),
      'capturing' => (C.muted, 'Not sent', Icons.pending_rounded),
      _ => (C.amber, 'Processing', Icons.autorenew_rounded),
    };
    final vol = (s.summary?['volume_ml'] as num?)?.toDouble();
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(children: [
            Container(
              width: 46,
              height: 46,
              decoration: BoxDecoration(color: color.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(14)),
              child: Icon(icon, color: color),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(s.name, style: t.titleMedium, maxLines: 1, overflow: TextOverflow.ellipsis),
                const SizedBox(height: 3),
                Text(_date(s.created), style: t.bodyMedium),
              ]),
            ),
            if (s.wristGirth != null)
              Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                Text('${s.wristGirth!.toStringAsFixed(1)} mm', style: t.titleMedium),
                Text(vol != null ? 'wrist · ${vol.round()} ml' : 'wrist', style: t.bodyMedium),
              ])
            else
              Pill(label, color: color),
          ]),
        ),
      ),
    );
  }

  static String _date(DateTime d) {
    const m = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return '${d.day} ${m[d.month - 1]} · ${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}';
  }
}
