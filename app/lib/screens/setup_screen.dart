import 'package:flutter/material.dart';

import '../config.dart';
import '../theme.dart';
import '../widgets/illustrations.dart';
import 'capture_screen.dart';

/// Three short steps before scanning: arm position, light, stillness. Then the AR capture.
class SetupScreen extends StatefulWidget {
  const SetupScreen({super.key, required this.config});
  final AppConfig config;

  @override
  State<SetupScreen> createState() => _SetupScreenState();
}

class _SetupScreenState extends State<SetupScreen> {
  late String _pose = widget.config.pose;
  final _pager = PageController();
  int _page = 0;
  static const _pages = 3;

  @override
  void dispose() {
    _pager.dispose();
    super.dispose();
  }

  Future<void> _next() async {
    if (_page < _pages - 1) {
      _pager.nextPage(duration: const Duration(milliseconds: 320), curve: Curves.easeOutCubic);
      return;
    }
    await widget.config.setPose(_pose);
    if (!mounted) return;
    Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => CaptureScreen(config: widget.config)));
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(
        title: Text('Step ${_page + 1} of $_pages', style: t.bodyMedium),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(6),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(children: [
              for (var i = 0; i < _pages; i++)
                Expanded(
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 250),
                    height: 4,
                    margin: const EdgeInsets.symmetric(horizontal: 3),
                    decoration: BoxDecoration(
                        color: i <= _page ? C.primary : C.line, borderRadius: BorderRadius.circular(4)),
                  ),
                ),
            ]),
          ),
        ),
      ),
      body: SafeArea(
        child: Column(children: [
          Expanded(
            child: PageView(
              controller: _pager,
              onPageChanged: (p) => setState(() => _page = p),
              children: [_poseStep(t), _lightStep(t), _stillStep(t)],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
            child: FilledButton.icon(
              onPressed: _next,
              icon: Icon(_page == _pages - 1 ? Icons.view_in_ar_rounded : Icons.arrow_forward_rounded),
              label: Text(_page == _pages - 1 ? 'Start scanning' : 'Next'),
            ),
          ),
        ]),
      ),
    );
  }

  Widget _step(TextTheme t, String kicker, String title, String body, List<Widget> children) => ListView(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
        children: [
          Text(kicker, style: t.labelSmall),
          const SizedBox(height: 6),
          Text(title, style: t.headlineSmall),
          const SizedBox(height: 8),
          Text(body, style: t.bodyMedium),
          const SizedBox(height: 18),
          ...children,
        ],
      );

  Widget _poseStep(TextTheme t) => _step(t, 'ARM POSITION', 'How is the arm placed?',
          'Pick the position the patient can hold still for a minute.', [
        _PoseCard(
          selected: _pose == 'upright',
          scene: Scene.upright,
          title: 'Upright',
          text: 'Elbow RESTING on a table, forearm up, fingers spread wide apart. Never held up in the air: '
              'a hand without support moves a little and the scan comes out wrong.',
          onTap: () => setState(() => _pose = 'upright'),
        ),
        const SizedBox(height: 12),
        _PoseCard(
          selected: _pose == 'resting',
          scene: Scene.resting,
          title: 'Resting',
          text: 'Forearm lying on a pillow or the table, hand free past the edge if possible.',
          onTap: () => setState(() => _pose = 'resting'),
        ),
      ]);

  Widget _lightStep(TextTheme t) => _step(t, 'LIGHT', 'Light on the arm, not behind it',
          'Stand with the light behind you. A bright window behind the arm turns the hand into a dark silhouette.', [
        const Card(child: Padding(padding: EdgeInsets.all(12), child: Illustration(Scene.light, height: 200))),
        const SizedBox(height: 14),
        const _Tip(Icons.checkroom_rounded, 'Bare skin', 'Sleeve above the elbow; remove watch and rings if allowed.'),
        const _Tip(Icons.flashlight_on_rounded, 'Torch on', 'The phone\'s light is on during the scan: even light that moves with the phone. Switch it off on the scan screen before the first photo if it glares.'),
      ]);

  Widget _stillStep(TextTheme t) => _step(t, 'HOW IT WORKS', 'Only the phone moves',
          'The arm and fingers stay completely still. You walk once around the hand, from one side of the patient to '
          'the other, with the whole hand on screen.', [
        const Card(child: Padding(padding: EdgeInsets.all(12), child: Illustration(Scene.still, height: 170))),
        const SizedBox(height: 14),
        const _Tip(Icons.back_hand_rounded, '1. Point at the hand', 'The app finds the hand by itself.'),
        const _Tip(Icons.swap_vert_rounded, '2. Toward the hand and back',
            'Move the phone slowly toward the hand and back until the bar is full: this sets the size of the scan.'),
        const _Tip(Icons.radio_button_checked_rounded, '3. Walk from dot to dot',
            'Photos are automatic. A dot turns green when it has good photos; if one stays amber, the app sends you '
            'back to it. It finishes by itself when all dots are green.'),
        const _Tip(Icons.straighten_rounded, 'No card or markers', 'The phone measures its own movement in millimetres.'),
      ]);
}

class _PoseCard extends StatelessWidget {
  const _PoseCard(
      {required this.selected, required this.scene, required this.title, required this.text, required this.onTap});
  final bool selected;
  final Scene scene;
  final String title;
  final String text;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: selected ? const Color(0xFF14254A) : C.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: selected ? C.primary : C.line, width: selected ? 2 : 1),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Illustration(scene, height: 120),
          Row(children: [
            Text(title, style: t.titleLarge),
            const Spacer(),
            Icon(selected ? Icons.radio_button_checked_rounded : Icons.radio_button_off_rounded,
                color: selected ? C.primary : C.muted),
          ]),
          const SizedBox(height: 4),
          Text(text, style: t.bodyMedium),
        ]),
      ),
    );
  }
}

class _Tip extends StatelessWidget {
  const _Tip(this.icon, this.title, this.text);
  final IconData icon;
  final String title;
  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 14),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(color: C.primary.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(12)),
            child: Icon(icon, color: C.primary, size: 20),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(title, style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 2),
              Text(text, style: Theme.of(context).textTheme.bodyMedium),
            ]),
          ),
        ]),
      );
}
