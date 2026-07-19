import 'package:flutter/material.dart';

abstract final class AppColors {
  static const surface = Color(0xFFF7FAFD);
  static const surfaceLow = Color(0xFFF1F4F7);
  static const surfaceContainer = Color(0xFFEBEEF1);
  static const white = Color(0xFFFFFFFF);
  static const navy = Color(0xFF0B2233);
  static const text = Color(0xFF181C1F);
  static const muted = Color(0xFF3F484E);
  static const outline = Color(0xFF6F787E);
  static const outlineVariant = Color(0xFFBEC8CE);
  static const primary = Color(0xFF006483);
  static const primaryContainer = Color(0xFF087EA4);
  static const action = Color(0xFF0369A1);
  static const pale = Color(0xFFEAF7FC);
  static const selected = Color(0xFF7BC2FF);
  static const track = Color(0xFFD9F0FA);

  // Compatibility aliases for the older lesson flow.
  static const blue = primaryContainer;
  static const blueDark = primary;
  static const violet = primary;
  static const mint = primaryContainer;
  static const amber = primaryContainer;
  static const coral = primaryContainer;
  static const canvas = surface;
  static const stroke = Color(0xFFD9EAF1);
}

abstract final class AppTheme {
  static ThemeData get light {
    const scheme = ColorScheme.light(
      primary: AppColors.primaryContainer,
      onPrimary: Colors.white,
      secondary: AppColors.primary,
      onSecondary: Colors.white,
      surface: AppColors.white,
      onSurface: AppColors.text,
      outline: AppColors.outline,
      error: Color(0xFFBA1A1A),
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: AppColors.surface,
      fontFamily: 'BeVietnamPro',
      splashFactory: InkRipple.splashFactory,
      textTheme: const TextTheme(
        headlineLarge: TextStyle(
          color: AppColors.navy,
          fontSize: 28,
          height: 1.28,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.4,
        ),
        headlineMedium: TextStyle(
          color: AppColors.navy,
          fontSize: 24,
          height: 1.33,
          fontWeight: FontWeight.w700,
        ),
        titleLarge: TextStyle(
          color: AppColors.navy,
          fontSize: 20,
          height: 1.4,
          fontWeight: FontWeight.w600,
        ),
        titleMedium: TextStyle(
          color: AppColors.navy,
          fontSize: 16,
          height: 1.5,
          fontWeight: FontWeight.w600,
        ),
        bodyLarge: TextStyle(color: AppColors.text, fontSize: 16, height: 1.5),
        bodyMedium: TextStyle(
          color: AppColors.muted,
          fontSize: 14,
          height: 1.45,
        ),
        labelLarge: TextStyle(
          fontSize: 14,
          height: 1.4,
          fontWeight: FontWeight.w600,
        ),
      ),
      cardTheme: const CardThemeData(
        color: AppColors.white,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(8)),
          side: BorderSide(color: Color(0xFFD8EDF6)),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: AppColors.primaryContainer,
          foregroundColor: Colors.white,
          minimumSize: const Size(0, 48),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.primaryContainer,
          minimumSize: const Size(0, 48),
          side: const BorderSide(color: AppColors.primaryContainer),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.white,
        hintStyle: const TextStyle(color: AppColors.outline),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 14,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: AppColors.outlineVariant),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: AppColors.outlineVariant),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(
            color: AppColors.primaryContainer,
            width: 2,
          ),
        ),
      ),
      dividerColor: const Color(0xFFD8EDF6),
    );
  }
}
