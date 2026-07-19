import 'package:flutter/material.dart';

class SignLesson {
  const SignLesson({
    required this.id,
    required this.name,
    required this.tip,
    required this.icon,
  });

  final String id;
  final String name;
  final String tip;
  final IconData icon;
}

class Course {
  const Course({
    required this.id,
    required this.title,
    required this.description,
    required this.color,
    required this.icon,
    required this.lessons,
  });

  final String id;
  final String title;
  final String description;
  final Color color;
  final IconData icon;
  final List<SignLesson> lessons;
}

class LearningSession {
  final Set<String> learnedIds = {};
  final Set<String> verifiedIds = {};
  int attempts = 0;

  int get learnedCount => learnedIds.length;
  int get masteredCount => learnedIds.intersection(verifiedIds).length;
  int get xp => learnedCount * 20 + verifiedIds.length * 30;

  bool isLearned(String id) => learnedIds.contains(id);
  bool isVerified(String id) => verifiedIds.contains(id);

  void markLearned(String id) => learnedIds.add(id);
  void recordAttempt() => attempts++;
  void markVerified(String id) {
    learnedIds.add(id);
    verifiedIds.add(id);
  }
}
