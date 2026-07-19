import 'package:flutter/material.dart';

import '../models/learning_models.dart';
import '../theme/app_theme.dart';

const _tip =
    'Giữ vai và hai bàn tay trong khung hình, thực hiện động tác chậm và rõ ràng.';
SignLesson _sign(String id, String name, IconData icon) =>
    SignLesson(id: id, name: name, tip: _tip, icon: icon);

final trainingCourses = <Course>[
  Course(
    id: 'family',
    title: 'Gia đình & xưng hô',
    description: '8 từ trong bộ dữ liệu đã huấn luyện',
    color: AppColors.violet,
    icon: Icons.family_restroom_rounded,
    lessons: [
      _sign('anh', 'Anh', Icons.person_rounded),
      _sign('chau', 'Cháu', Icons.child_care_rounded),
      _sign('chu', 'Chú', Icons.person_2_rounded),
      _sign('chi', 'Chị', Icons.person_3_rounded),
      _sign('co', 'Cô', Icons.face_3_rounded),
      _sign('cau', 'Cậu', Icons.face_6_rounded),
      _sign('em', 'Em', Icons.emoji_people_rounded),
      _sign('ho-hang', 'Họ hàng', Icons.groups_rounded),
    ],
  ),
  Course(
    id: 'home',
    title: 'Nhà cửa & đồ dùng',
    description: '9 từ trong bộ dữ liệu đã huấn luyện',
    color: AppColors.mint,
    icon: Icons.home_rounded,
    lessons: [
      _sign('cai-ban', 'Cái bàn', Icons.table_restaurant_rounded),
      _sign('cai-chao', 'Cái chảo', Icons.soup_kitchen_rounded),
      _sign('cai-cua', 'Cái cửa', Icons.door_front_door_rounded),
      _sign('cai-den', 'Cái đèn', Icons.lightbulb_rounded),
      _sign('cua-so', 'Cửa sổ', Icons.window_rounded),
      _sign('giuong', 'Giường', Icons.bed_rounded),
      _sign('may-dieu-hoa', 'Máy điều hòa', Icons.air_rounded),
      _sign('noi-com-dien', 'Nồi cơm điện', Icons.rice_bowl_rounded),
      _sign('quat-dung', 'Quạt đứng', Icons.wind_power_rounded),
    ],
  ),
  Course(
    id: 'places',
    title: 'Trường học & nơi chốn',
    description: '7 từ trong bộ dữ liệu đã huấn luyện',
    color: AppColors.amber,
    icon: Icons.school_rounded,
    lessons: [
      _sign('nghe-nghiep', 'Nghề nghiệp', Icons.work_rounded),
      _sign(
        'ngay-nha-giao-viet-nam',
        'Ngày Nhà giáo Việt Nam',
        Icons.event_rounded,
      ),
      _sign('ngan-hang', 'Ngân hàng', Icons.account_balance_rounded),
      _sign('nha-hang', 'Nhà hàng', Icons.restaurant_rounded),
      _sign('nha-tro', 'Nhà trọ', Icons.apartment_rounded),
      _sign('truong-hoc', 'Trường học', Icons.school_rounded),
      _sign('truong-dai-hoc', 'Trường Đại học', Icons.account_balance_rounded),
    ],
  ),
  Course(
    id: 'weather',
    title: 'Thời gian & thời tiết',
    description: '6 từ trong bộ dữ liệu đã huấn luyện',
    color: AppColors.coral,
    icon: Icons.wb_sunny_rounded,
    lessons: [
      _sign('chu-nhat', 'Chủ nhật', Icons.calendar_today_rounded),
      _sign('de', 'Dễ', Icons.thumb_up_alt_rounded),
      _sign('mua-he', 'Mùa hè', Icons.beach_access_rounded),
      _sign('mua-kho', 'Mùa khô', Icons.landscape_rounded),
      _sign('nang', 'Nắng', Icons.wb_sunny_rounded),
      _sign('uot', 'Ướt', Icons.water_drop_rounded),
    ],
  ),
];

final allTrainingSigns = [
  for (final course in trainingCourses)
    for (final lesson in course.lessons) (course: course, lesson: lesson),
];
