import 'package:flutter/material.dart';

import '../models/learning_models.dart';
import '../services/recognition_gateway.dart';
import '../theme/app_theme.dart';
import 'stitch_home_screen.dart';
import 'stitch_lessons_screen.dart';
import 'stitch_practice_screen.dart';
import 'stitch_profile_screen.dart';
import 'stitch_progress_screen.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key, this.enableMedia = true, this.gateway});

  final bool enableMedia;
  final RecognitionGateway? gateway;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final LearningSession _session = LearningSession();
  late final RecognitionGateway _gateway;
  int _index = 0;
  String _selectedSignId = 'anh';

  @override
  void initState() {
    super.initState();
    _gateway = widget.gateway ?? BackendRecognitionGateway();
  }

  void _goTo(int index) => setState(() => _index = index);

  void _practiceSign(String signId) {
    setState(() {
      _selectedSignId = signId;
      _index = 2;
    });
  }

  void _openMenu() => _scaffoldKey.currentState?.openDrawer();

  void _showNotifications(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (context) => const SafeArea(
        child: Padding(
          padding: EdgeInsets.fromLTRB(24, 8, 24, 32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.notifications_none_rounded, size: 42),
              SizedBox(height: 12),
              Text(
                'Thông báo',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
              ),
              SizedBox(height: 8),
              Text('Bạn chưa có thông báo mới.'),
            ],
          ),
        ),
      ),
    );
  }

  void _selectFromDrawer(int index) {
    _scaffoldKey.currentState?.closeDrawer();
    _goTo(index);
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      StitchHomeScreen(
        session: _session,
        onNavigate: _goTo,
        onMenu: _openMenu,
        onNotifications: () => _showNotifications(context),
      ),
      StitchLessonsScreen(
        session: _session,
        onPractice: _practiceSign,
        onMenu: _openMenu,
        onNotifications: () => _showNotifications(context),
      ),
      StitchPracticeScreen(
        session: _session,
        gateway: _gateway,
        enableMedia: widget.enableMedia,
        active: _index == 2,
        selectedSignId: _selectedSignId,
        onMenu: _openMenu,
        onNotifications: () => _showNotifications(context),
      ),
      StitchProgressScreen(
        session: _session,
        onMenu: _openMenu,
        onNotifications: () => _showNotifications(context),
      ),
      StitchProfileScreen(
        session: _session,
        onMenu: _openMenu,
        onNotifications: () => _showNotifications(context),
      ),
    ];

    return Scaffold(
      key: _scaffoldKey,
      drawer: NavigationDrawer(
        key: const Key('main-drawer'),
        selectedIndex: _index,
        onDestinationSelected: _selectFromDrawer,
        children: const [
          Padding(
            padding: EdgeInsets.fromLTRB(28, 28, 16, 12),
            child: Text(
              'Light & Sound VSL',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
            ),
          ),
          NavigationDrawerDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home_rounded),
            label: Text('Trang chủ'),
          ),
          NavigationDrawerDestination(
            icon: Icon(Icons.school_outlined),
            selectedIcon: Icon(Icons.school_rounded),
            label: Text('Bài học'),
          ),
          NavigationDrawerDestination(
            icon: Icon(Icons.videocam_outlined),
            selectedIcon: Icon(Icons.videocam_rounded),
            label: Text('Luyện tập'),
          ),
          NavigationDrawerDestination(
            icon: Icon(Icons.bar_chart_outlined),
            selectedIcon: Icon(Icons.bar_chart_rounded),
            label: Text('Tiến độ'),
          ),
          NavigationDrawerDestination(
            icon: Icon(Icons.person_outline_rounded),
            selectedIcon: Icon(Icons.person_rounded),
            label: Text('Hồ sơ'),
          ),
        ],
      ),
      body: SafeArea(
        child: IndexedStack(index: _index, children: pages),
      ),
      bottomNavigationBar: NavigationBar(
        key: const Key('main-navigation'),
        selectedIndex: _index,
        height: 76,
        elevation: 2,
        backgroundColor: Colors.white,
        indicatorColor: AppColors.selected,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        onDestinationSelected: _goTo,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home_rounded),
            label: 'Trang chủ',
          ),
          NavigationDestination(
            icon: Icon(Icons.school_outlined),
            selectedIcon: Icon(Icons.school_rounded),
            label: 'Bài học',
          ),
          NavigationDestination(
            icon: Icon(Icons.videocam_outlined),
            selectedIcon: Icon(Icons.videocam_rounded),
            label: 'Luyện tập',
          ),
          NavigationDestination(
            icon: Icon(Icons.bar_chart_outlined),
            selectedIcon: Icon(Icons.bar_chart_rounded),
            label: 'Tiến độ',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outline_rounded),
            selectedIcon: Icon(Icons.person_rounded),
            label: 'Hồ sơ',
          ),
        ],
      ),
    );
  }
}
