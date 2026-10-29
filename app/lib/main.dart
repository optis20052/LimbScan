import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'config.dart';
import 'screens/home_screen.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: C.bg,
  ));
  final config = await AppConfig.load();
  runApp(LimbScanApp(config: config));
}

class LimbScanApp extends StatelessWidget {
  const LimbScanApp({super.key, required this.config});
  final AppConfig config;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'LimbScan',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(),
        home: HomeScreen(config: config),
      );
}
