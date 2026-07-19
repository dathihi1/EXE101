import 'package:flutter/material.dart';

import '../data/training_catalog.dart';
import '../models/learning_models.dart';
import '../widgets/stitch_widgets.dart';

class StitchProgressScreen extends StatelessWidget {
  const StitchProgressScreen({
    super.key,
    required this.session,
    required this.onMenu,
    required this.onNotifications,
  });
  final LearningSession session;
  final VoidCallback onMenu;
  final VoidCallback onNotifications;

  @override
  Widget build(BuildContext context) {
    final accuracy = session.attempts == 0
        ? 0.0
        : session.masteredCount / session.attempts;
    return Column(
      children: [
        StitchHeader(
          title: 'Tiến độ học tập',
          onMenu: onMenu,
          onNotifications: onNotifications,
        ),
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              OceanCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Từ đã luyện',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '${session.learnedCount}/${allTrainingSigns.length}',
                      style: Theme.of(context).textTheme.headlineLarge,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              OceanCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Kết quả AI',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '${session.masteredCount} đạt · ${session.attempts} lượt thử',
                    ),
                    const SizedBox(height: 6),
                    Text('Tỷ lệ đạt ${(accuracy * 100).toStringAsFixed(0)}%'),
                  ],
                ),
              ),
              if (session.attempts == 0)
                const Padding(
                  padding: EdgeInsets.only(top: 24),
                  child: Center(
                    child: Text(
                      'Chưa có dữ liệu luyện tập. Hãy ghi video và gửi AI để bắt đầu.',
                    ),
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }
}
