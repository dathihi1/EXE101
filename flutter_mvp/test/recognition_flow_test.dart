import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:light_sound_mvp/models/learning_models.dart';
import 'package:light_sound_mvp/screens/stitch_practice_screen.dart';
import 'package:light_sound_mvp/services/recognition_gateway.dart';

class _SuccessfulGateway implements RecognitionGateway {
  String? expectedId;
  int frameCount = 0;

  @override
  Future<RecognitionResult> recognize({
    required String expectedSignId,
    required List<Uint8List> jpegFrames,
  }) async {
    expectedId = expectedSignId;
    frameCount = jpegFrames.length;
    return const RecognitionResult(
      status: 'success',
      verified: true,
      predictedSignId: 'anh',
      predictedLabel: 'Anh',
      confidence: 0.92,
      top3: [RecognitionCandidate(label: 'Anh', confidence: 0.92)],
      qualityHints: [],
    );
  }
}

class _FailingGateway implements RecognitionGateway {
  @override
  Future<RecognitionResult> recognize({
    required String expectedSignId,
    required List<Uint8List> jpegFrames,
  }) {
    throw const RecognitionException('Backend AI không kết nối được.');
  }
}

Widget _app(RecognitionGateway gateway, LearningSession session) => MaterialApp(
  home: Scaffold(
    body: StitchPracticeScreen(
      session: session,
      gateway: gateway,
      enableMedia: false,
      onMenu: () {},
      onNotifications: () {},
      frameCaptureOverride: () async => List.generate(
        12,
        (_) => Uint8List.fromList([0xff, 0xd8, 0xff, 0xd9]),
      ),
    ),
  ),
);

void main() {
  testWidgets('records frames, sends AI request and renders verified result', (
    tester,
  ) async {
    final gateway = _SuccessfulGateway();
    final session = LearningSession();
    await tester.pumpWidget(_app(gateway, session));

    await tester.drag(find.byType(ListView), const Offset(0, -1000));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('camera-main-action')));
    await tester.pumpAndSettle();

    expect(gateway.expectedId, 'anh');
    expect(gateway.frameCount, 12);
    expect(find.byKey(const Key('recognition-result')), findsOneWidget);
    expect(find.text('Đạt'), findsOneWidget);
    expect(find.textContaining('92.0%'), findsOneWidget);
    expect(session.attempts, 1);
    expect(session.isVerified('anh'), isTrue);
  });

  testWidgets('shows backend errors without creating fake progress', (
    tester,
  ) async {
    final session = LearningSession();
    await tester.pumpWidget(_app(_FailingGateway(), session));

    await tester.drag(find.byType(ListView), const Offset(0, -1000));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('camera-main-action')));
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('recognition-error')), findsOneWidget);
    expect(find.text('Backend AI không kết nối được.'), findsOneWidget);
    expect(session.attempts, 0);
    expect(session.learnedCount, 0);
  });

  test('parses the backend recognition contract', () {
    final result = RecognitionResult.fromJson({
      'status': 'success',
      'verified': false,
      'predictedSignId': 'chau',
      'predictedLabel': 'Cháu',
      'confidence': 0.63,
      'top3': [
        {'label': 'Cháu', 'confidence': 0.63},
      ],
      'qualityHints': ['Giữ đủ hai tay trong khung hình'],
    });
    expect(result.predictedLabel, 'Cháu');
    expect(result.verified, isFalse);
    expect(result.top3.single.confidence, 0.63);
    expect(result.qualityHints, hasLength(1));
  });
}
