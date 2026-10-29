import 'package:flutter/material.dart';

import '../api/limbscan_api.dart';
import '../config.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.config});
  final AppConfig config;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final _server = TextEditingController(text: widget.config.serverUrl);
  String? _test;

  Future<void> _check() async {
    setState(() => _test = 'Checking…');
    final ok = await LimbScanApi(_server.text.trim()).ping();
    setState(() => _test = ok ? '✓ Server reachable' : '✗ Not reachable');
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Settings')),
        body: ListView(padding: const EdgeInsets.all(16), children: [
          TextField(
            controller: _server,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
                labelText: 'Server address',
                helperText: 'LimbScan server, e.g. http://192.168.50.223:8080 or the https tunnel link'),
          ),
          const SizedBox(height: 12),
          Row(children: [
            OutlinedButton(onPressed: _check, child: const Text('Test connection')),
            const SizedBox(width: 12),
            if (_test != null) Text(_test!),
          ]),
          const SizedBox(height: 24),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Voice guidance'),
            subtitle: const Text('Spoken hints while scanning'),
            value: widget.config.voice,
            onChanged: (v) async {
              await widget.config.setVoice(v);
              setState(() {});
            },
          ),
          const SizedBox(height: 24),
          FilledButton(
            onPressed: () async {
              await widget.config.setServerUrl(_server.text);
              if (context.mounted) Navigator.pop(context, true);
            },
            child: const Text('Save'),
          ),
        ]),
      );
}
