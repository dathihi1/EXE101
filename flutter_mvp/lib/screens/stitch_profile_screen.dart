import 'package:flutter/material.dart';

import '../models/learning_models.dart';
import '../widgets/stitch_widgets.dart';

class StitchProfileScreen extends StatefulWidget {
  const StitchProfileScreen({
    super.key,
    required this.session,
    required this.onMenu,
    required this.onNotifications,
  });
  final LearningSession session;
  final VoidCallback onMenu;
  final VoidCallback onNotifications;

  @override
  State<StitchProfileScreen> createState() => _StitchProfileScreenState();
}

class _StitchProfileScreenState extends State<StitchProfileScreen> {
  bool captions = true;
  bool mirror = true;

  @override
  Widget build(BuildContext context) => Column(
    children: [
      StitchHeader(
        title: 'Hồ sơ',
        onMenu: widget.onMenu,
        onNotifications: widget.onNotifications,
      ),
      Expanded(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            const CircleAvatar(
              radius: 42,
              child: Icon(Icons.person_outline, size: 42),
            ),
            const SizedBox(height: 12),
            Text(
              'Người học VSL',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const Text(
              'Phiên khách · không có dữ liệu cá nhân mẫu',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            OceanCard(
              child: Column(
                children: [
                  ListTile(
                    leading: const Icon(Icons.task_alt),
                    title: const Text('Từ đã luyện'),
                    trailing: Text('${widget.session.learnedCount}/30'),
                  ),
                  ListTile(
                    leading: const Icon(Icons.verified_outlined),
                    title: const Text('Được AI xác minh'),
                    trailing: Text('${widget.session.masteredCount}'),
                  ),
                  ListTile(
                    leading: const Icon(Icons.repeat),
                    title: const Text('Lượt thử'),
                    trailing: Text('${widget.session.attempts}'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            OceanCard(
              child: Column(
                children: [
                  SwitchListTile(
                    title: const Text('Phụ đề'),
                    value: captions,
                    onChanged: (value) => setState(() => captions = value),
                  ),
                  SwitchListTile(
                    title: const Text('Chế độ gương mặc định'),
                    value: mirror,
                    onChanged: (value) => setState(() => mirror = value),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    ],
  );
}
