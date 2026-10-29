import 'package:shared_preferences/shared_preferences.dart';

/// App settings (persisted). The server is the LimbScan reconstruction server on the GPU PC.
class AppConfig {
  AppConfig._(this._prefs);

  static const defaultServer = 'http://192.168.50.223:8080';
  final SharedPreferences _prefs;

  static Future<AppConfig> load() async => AppConfig._(await SharedPreferences.getInstance());

  String get serverUrl => (_prefs.getString('server') ?? defaultServer).replaceAll(RegExp(r'/+$'), '');
  Future<void> setServerUrl(String v) => _prefs.setString('server', v.trim());

  String get pose => _prefs.getString('pose') ?? 'upright';
  Future<void> setPose(String v) => _prefs.setString('pose', v);

  bool get voice => _prefs.getBool('voice') ?? true;
  Future<void> setVoice(bool v) => _prefs.setBool('voice', v);
}
