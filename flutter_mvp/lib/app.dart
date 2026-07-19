import 'package:flutter/material.dart';

import 'screens/home_shell.dart';
import 'services/recognition_gateway.dart';
import 'theme/app_theme.dart';

class LightSoundApp extends StatelessWidget {
  const LightSoundApp({super.key, this.enableMedia = true, this.gateway});

  final bool enableMedia;
  final RecognitionGateway? gateway;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Light & Sound',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      home: HomeShell(enableMedia: enableMedia, gateway: gateway),
    );
  }
}
