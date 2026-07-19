import 'package:flutter/material.dart';

import '../data/training_catalog.dart';
import '../models/learning_models.dart';
import '../theme/app_theme.dart';
import '../widgets/stitch_widgets.dart';

class StitchHomeScreen extends StatelessWidget {
  const StitchHomeScreen({
    super.key,
    required this.session,
    required this.onNavigate,
    required this.onMenu,
    required this.onNotifications,
  });
  final LearningSession session;
  final ValueChanged<int> onNavigate;
  final VoidCallback onMenu;
  final VoidCallback onNotifications;

  @override
  Widget build(BuildContext context) {
    final progress = session.learnedCount / allTrainingSigns.length;
    return Column(
      children: [
        StitchHeader(
          title: 'Light & Sound VSL',
          onMenu: onMenu,
          onNotifications: onNotifications,
        ),
        Expanded(
          child: ListView(
            key: const Key('stitch-home-scroll'),
            padding: const EdgeInsets.all(16),
            children: [
              Text(
                'Học ngôn ngữ ký hiệu Việt Nam',
                style: Theme.of(context).textTheme.headlineLarge,
              ),
              const SizedBox(height: 6),
              const Text(
                'Bộ học gồm đúng 30 từ đã huấn luyện. Kết quả được cập nhật sau khi AI xác minh.',
              ),
              const SizedBox(height: 20),
              OceanCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${session.learnedCount}/30 từ đã luyện',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 12),
                    OceanProgress(value: progress),
                    const SizedBox(height: 8),
                    Text(
                      '${session.masteredCount} từ đã được AI xác minh · ${session.attempts} lượt thử',
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              OceanCard(
                color: AppColors.pale,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const OceanIconBox(icon: Icons.videocam_outlined),
                    const SizedBox(height: 14),
                    Text(
                      'Luyện tập theo video mẫu',
                      style: Theme.of(context).textTheme.headlineMedium,
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      'Xem động tác ở chế độ gương, ghi 12 khung hình và nhận kết quả AI.',
                    ),
                    const SizedBox(height: 16),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton(
                        key: const Key('start-practice'),
                        onPressed: () => onNavigate(2),
                        child: const Text('Bắt đầu luyện tập'),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                key: const Key('open-lessons'),
                onPressed: () => onNavigate(1),
                icon: const Icon(Icons.school_outlined),
                label: const Text('Xem 30 từ đã huấn luyện'),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
