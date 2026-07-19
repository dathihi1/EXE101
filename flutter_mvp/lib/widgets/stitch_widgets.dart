import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

class StitchHeader extends StatelessWidget {
  const StitchHeader({
    super.key,
    required this.title,
    required this.onMenu,
    required this.onNotifications,
  });

  final String title;
  final VoidCallback onMenu;
  final VoidCallback onNotifications;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 64,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: const BoxDecoration(
        color: AppColors.white,
        border: Border(bottom: BorderSide(color: Color(0xFFDDE7EC))),
      ),
      child: Row(
        children: [
          IconButton(
            key: const Key('header-menu'),
            tooltip: 'Menu',
            onPressed: onMenu,
            icon: const Icon(Icons.menu_rounded),
          ),
          Expanded(
            child: Text(
              title,
              textAlign: TextAlign.center,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: AppColors.primary,
                fontSize: 22,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          IconButton(
            key: const Key('header-notifications'),
            tooltip: 'Thông báo',
            onPressed: onNotifications,
            icon: const Icon(Icons.notifications_none_rounded),
          ),
        ],
      ),
    );
  }
}

class OceanCard extends StatelessWidget {
  const OceanCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.color,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: color ?? Colors.white,
        border: Border.all(color: const Color(0xFFD8EDF6)),
        borderRadius: BorderRadius.circular(8),
        boxShadow: const [
          BoxShadow(
            color: Color(0x0D087EA4),
            blurRadius: 14,
            offset: Offset(0, 5),
          ),
        ],
      ),
      child: Material(type: MaterialType.transparency, child: child),
    );
  }
}

class OceanProgress extends StatelessWidget {
  const OceanProgress({super.key, required this.value, this.height = 8});

  final double value;
  final double height;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(height),
      child: LinearProgressIndicator(
        value: value.clamp(0, 1),
        minHeight: height,
        backgroundColor: AppColors.track,
        valueColor: const AlwaysStoppedAnimation(AppColors.primaryContainer),
      ),
    );
  }
}

class OceanIconBox extends StatelessWidget {
  const OceanIconBox({
    super.key,
    required this.icon,
    this.size = 48,
    this.filled = false,
  });

  final IconData icon;
  final double size;
  final bool filled;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: filled ? AppColors.selected : AppColors.pale,
        borderRadius: BorderRadius.circular(size / 2),
      ),
      child: Icon(icon, color: AppColors.primaryContainer, size: size * 0.5),
    );
  }
}

class StitchSectionTitle extends StatelessWidget {
  const StitchSectionTitle({
    super.key,
    required this.title,
    this.action,
    this.onAction,
  });

  final String title;
  final String? action;
  final VoidCallback? onAction;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(title, style: Theme.of(context).textTheme.titleLarge),
        ),
        if (action != null)
          TextButton(
            onPressed: onAction,
            child: Text(
              action!,
              style: const TextStyle(
                color: AppColors.primaryContainer,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
      ],
    );
  }
}
