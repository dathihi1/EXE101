import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:light_sound_mvp/app.dart';

void main() {
  testWidgets('starts with real session data at zero', (tester) async {
    await tester.pumpWidget(const LightSoundApp(enableMedia: false));

    expect(find.text('Light & Sound VSL'), findsOneWidget);
    expect(find.text('0/30 từ đã luyện'), findsOneWidget);
    expect(find.textContaining('0 từ đã được AI xác minh'), findsOneWidget);
    expect(find.text('Người học VSL'), findsNothing);
  });

  testWidgets('navigates through five main tabs', (tester) async {
    await tester.pumpWidget(const LightSoundApp(enableMedia: false));

    await tester.tap(find.text('Bài học'));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('course-list')), findsOneWidget);
    expect(find.text('Gia đình & xưng hô'), findsOneWidget);
    expect(find.text('Anh'), findsNothing);

    await tester.tap(find.byKey(const Key('course-family')));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('course-words-family')), findsOneWidget);
    expect(find.text('Anh'), findsOneWidget);

    await tester.tap(find.byKey(const Key('sign-chau')));
    await tester.pumpAndSettle();
    expect(find.text('Luyện tập với AI'), findsOneWidget);
    expect(find.text('2. Cháu'), findsOneWidget);

    await tester.tap(find.text('Luyện tập'));
    await tester.pumpAndSettle();
    expect(find.text('Luyện tập với AI'), findsOneWidget);
    expect(find.byKey(const Key('trained-word-selector')), findsOneWidget);

    await tester.tap(find.text('Tiến độ'));
    await tester.pumpAndSettle();
    expect(find.text('Tiến độ học tập'), findsOneWidget);

    await tester.tap(find.text('Hồ sơ'));
    await tester.pumpAndSettle();
    expect(find.text('Người học VSL'), findsOneWidget);
    expect(
      find.text('Phiên khách · không có dữ liệu cá nhân mẫu'),
      findsOneWidget,
    );
  });

  testWidgets('opens the navigation menu and notifications safely', (
    tester,
  ) async {
    await tester.pumpWidget(const LightSoundApp(enableMedia: false));

    await tester.tap(find.byKey(const Key('header-menu')).hitTestable());
    await tester.pumpAndSettle();
    final scaffold = tester.state<ScaffoldState>(find.byType(Scaffold));
    expect(scaffold.isDrawerOpen, isTrue);

    scaffold.closeDrawer();
    await tester.pumpAndSettle();
    await tester.tap(
      find.byKey(const Key('header-notifications')).hitTestable(),
    );
    await tester.pumpAndSettle();
    expect(find.text('Bạn chưa có thông báo mới.'), findsOneWidget);
  });
}
