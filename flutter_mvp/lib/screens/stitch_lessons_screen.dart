import 'package:flutter/material.dart';

import '../data/training_catalog.dart';
import '../models/learning_models.dart';
import '../theme/app_theme.dart';
import '../widgets/stitch_widgets.dart';

class StitchLessonsScreen extends StatelessWidget {
  const StitchLessonsScreen({
    super.key,
    required this.session,
    required this.onPractice,
    required this.onMenu,
    required this.onNotifications,
  });

  final LearningSession session;
  final ValueChanged<String> onPractice;
  final VoidCallback onMenu;
  final VoidCallback onNotifications;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        StitchHeader(
          title: 'Bài học',
          onMenu: onMenu,
          onNotifications: onNotifications,
        ),
        Expanded(
          child: ListView(
            key: const Key('course-list'),
            padding: const EdgeInsets.fromLTRB(16, 20, 16, 28),
            children: [
              Text(
                'Chọn một bài để xem các từ',
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              const SizedBox(height: 6),
              const Text('30 từ đã huấn luyện được sắp xếp thành 4 bài học.'),
              const SizedBox(height: 18),
              for (final course in trainingCourses) ...[
                _CourseCard(
                  course: course,
                  session: session,
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => _CourseDetailPage(
                        course: course,
                        session: session,
                        onPractice: onPractice,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

class _CourseCard extends StatelessWidget {
  const _CourseCard({
    required this.course,
    required this.session,
    required this.onTap,
  });

  final Course course;
  final LearningSession session;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final learned = course.lessons
        .where((item) => session.isLearned(item.id))
        .length;
    return OceanCard(
      padding: EdgeInsets.zero,
      child: ListTile(
        key: Key('course-${course.id}'),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 12,
        ),
        leading: CircleAvatar(
          backgroundColor: course.color.withValues(alpha: 0.22),
          child: Icon(course.icon, color: AppColors.text),
        ),
        title: Text(
          course.title,
          style: Theme.of(context).textTheme.titleMedium,
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 5),
          child: Text(
            '${course.lessons.length} từ · Đã luyện $learned/${course.lessons.length}',
          ),
        ),
        trailing: const Icon(Icons.chevron_right_rounded),
        onTap: onTap,
      ),
    );
  }
}

class _CourseDetailPage extends StatelessWidget {
  const _CourseDetailPage({
    required this.course,
    required this.session,
    required this.onPractice,
  });

  final Course course;
  final LearningSession session;
  final ValueChanged<String> onPractice;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(course.title)),
      body: ListView.separated(
        key: Key('course-words-${course.id}'),
        padding: const EdgeInsets.all(16),
        itemCount: course.lessons.length,
        separatorBuilder: (_, _) => const SizedBox(height: 8),
        itemBuilder: (context, index) {
          final sign = course.lessons[index];
          final status = session.isVerified(sign.id)
              ? 'Đã được AI xác minh'
              : session.isLearned(sign.id)
              ? 'Đã luyện, chưa xác minh'
              : 'Chưa luyện';
          return OceanCard(
            padding: EdgeInsets.zero,
            child: ListTile(
              key: Key('sign-${sign.id}'),
              leading: Icon(sign.icon),
              title: Text(sign.name),
              subtitle: Text(status),
              trailing: const Icon(Icons.play_circle_outline_rounded),
              onTap: () {
                Navigator.of(context).pop();
                onPractice(sign.id);
              },
            ),
          );
        },
      ),
    );
  }
}
