import 'dart:ui';

import 'package:flutter/material.dart';

/// LimbScan design system: dark clinical theme, one blue primary, teal = "arm / good", amber = "attention".
class C {
  static const bg = Color(0xFF0A0F1C);
  static const surface = Color(0xFF121A2B);
  static const surface2 = Color(0xFF1A2438);
  static const line = Color(0xFF26324A);
  static const text = Color(0xFFE8EEF8);
  static const muted = Color(0xFF8A96AD);
  static const primary = Color(0xFF3D8BFD);
  static const teal = Color(0xFF22C3A6);
  static const amber = Color(0xFFF5B83D);
  static const red = Color(0xFFF2545B);
  static const skin = Color(0xFFE9B99A);
  static const skinLine = Color(0xFFC98D6E);
}

ThemeData buildTheme() {
  final scheme = ColorScheme.fromSeed(
    seedColor: C.primary,
    brightness: Brightness.dark,
    primary: C.primary,
    secondary: C.teal,
    surface: C.surface,
    error: C.red,
  );
  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: C.bg,
    appBarTheme: const AppBarTheme(backgroundColor: C.bg, foregroundColor: C.text, elevation: 0, centerTitle: false),
    cardTheme: CardThemeData(
      color: C.surface,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20), side: const BorderSide(color: C.line)),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size.fromHeight(56),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        textStyle: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size.fromHeight(52),
        foregroundColor: C.text,
        side: const BorderSide(color: C.line),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
      ),
    ),
    textTheme: const TextTheme(
      headlineLarge: TextStyle(fontSize: 30, fontWeight: FontWeight.w800, color: C.text, letterSpacing: -0.5),
      headlineSmall: TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: C.text),
      titleLarge: TextStyle(fontSize: 19, fontWeight: FontWeight.w700, color: C.text),
      titleMedium: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: C.text),
      bodyLarge: TextStyle(fontSize: 16, height: 1.45, color: C.text),
      bodyMedium: TextStyle(fontSize: 14, height: 1.4, color: C.muted),
      labelSmall: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, letterSpacing: 1.1, color: C.primary),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: C.surface2,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: const BorderSide(color: C.line)),
      enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: const BorderSide(color: C.line)),
    ),
  );
}

/// Frosted panel used over the camera preview.
class Glass extends StatelessWidget {
  const Glass({super.key, required this.child, this.radius = 24, this.padding = const EdgeInsets.all(16)});
  final Widget child;
  final double radius;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) => ClipRRect(
        borderRadius: BorderRadius.circular(radius),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 18, sigmaY: 18),
          child: Container(
            padding: padding,
            decoration: BoxDecoration(
              color: const Color(0xB30A0F1C),
              borderRadius: BorderRadius.circular(radius),
              border: Border.all(color: const Color(0x33FFFFFF)),
            ),
            child: child,
          ),
        ),
      );
}

class Pill extends StatelessWidget {
  const Pill(this.label, {super.key, this.color = C.muted, this.icon});
  final String label;
  final Color color;
  final IconData? icon;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.14),
          borderRadius: BorderRadius.circular(99),
          border: Border.all(color: color.withValues(alpha: 0.45)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          if (icon != null) ...[Icon(icon, size: 14, color: color), const SizedBox(width: 5)],
          Text(label, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w700)),
        ]),
      );
}
