import 'package:shared_preferences/shared_preferences.dart';

/// App settings (persisted). The server is the LimbScan reconstruction server (fixed Cloudflare tunnel address).
class AppConfig {
  AppConfig._(this._prefs);

  static const defaultServer = 'https://limbscan.yori.to';
  final SharedPreferences _prefs;

  static Future<AppConfig> load() async => AppConfig._(await SharedPreferences.getInstance());

  String get serverUrl {
    final saved = _prefs.getString('server');
    // temporary quick-tunnel links (and the old PC address) are replaced by the fixed address
    final stale = saved == null || saved.contains('trycloudflare.com') || saved.contains('192.168.50.223');
    return (stale ? defaultServer : saved).replaceAll(RegExp(r'/+$'), '');
  }

  Future<void> setServerUrl(String v) => _prefs.setString('server', v.trim());

  String get pose => _prefs.getString('pose') ?? 'upright';
  Future<void> setPose(String v) => _prefs.setString('pose', v);

  /// Scan a test object (e.g. a 3D-printed hand) instead of a person: the app follows the colour you tap.

  bool get voice => _prefs.getBool('voice') ?? true;
  Future<void> setVoice(bool v) => _prefs.setBool('voice', v);
}
