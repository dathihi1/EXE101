import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:light_sound_mvp/app.dart';

Future<void> _setPhoneSize(WidgetTester tester, Size logicalSize) async {
  tester.view.devicePixelRatio = 1;
  tester.view.physicalSize = logicalSize;
  addTearDown(tester.view.resetDevicePixelRatio);
  addTearDown(tester.view.resetPhysicalSize);
}

void main() {
  for (final device in <String, Size>{
    'Android compact 360x800': const Size(360, 800),
    'Phone 390x844': const Size(390, 844),
  }.entries) {
    testWidgets('${device.key}: main navigation has no layout overflow', (
      tester,
    ) async {
      await _setPhoneSize(tester, device.value);
      await tester.pumpWidget(const LightSoundApp(enableMedia: false));
      await tester.pumpAndSettle();

      expect(find.text('Light & Sound VSL'), findsOneWidget);
      expect(tester.takeException(), isNull);

      for (final tab in ['Bài học', 'Luyện tập', 'Tiến độ', 'Hồ sơ']) {
        await tester.tap(find.text(tab).hitTestable());
        await tester.pumpAndSettle();
        expect(tester.takeException(), isNull, reason: 'Lỗi ở tab $tab');
      }
    });
  }

  testWidgets('phone flow: menu, lesson, word and practice', (tester) async {
    await _setPhoneSize(tester, const Size(390, 844));
    await tester.pumpWidget(const LightSoundApp(enableMedia: false));
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const Key('header-menu')).hitTestable());
    await tester.pumpAndSettle();
    final scaffold = tester.state<ScaffoldState>(find.byType(Scaffold));
    expect(scaffold.isDrawerOpen, isTrue);
    expect(tester.takeException(), isNull);

    scaffold.closeDrawer();
    await tester.pumpAndSettle();
    await tester.tap(find.text('Bài học').hitTestable());
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('course-family')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('sign-chau')));
    await tester.pumpAndSettle();

    expect(find.text('Luyện tập với AI'), findsOneWidget);
    expect(find.text('2. Cháu'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
