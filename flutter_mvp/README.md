# Light & Sound Flutter MVP

Giao diện Flutter của ứng dụng học Ngôn ngữ ký hiệu Việt Nam.

## Tính năng

- Lộ trình 4 chủ đề với 30 ký hiệu và video mẫu H.264.
- Từ điển, bài học, thử thách hằng ngày và hồ sơ tiến độ.
- Camera thật chụp chuỗi frame JPEG và gửi tới Spring Boot API.
- Backend chuyển frame tới dịch vụ AI ONNX và trả kết quả nhận diện/top 3.

## Chạy ứng dụng

Khởi động AI ở cổng `7860` và backend ở cổng `8080` theo README tại thư mục gốc,
sau đó chạy:

```powershell
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8080
```

Android Emulator mặc định dùng `http://10.0.2.2:8080`. Điện thoại thật cần
`API_BASE_URL` trỏ tới địa chỉ LAN hoặc HTTPS công khai của backend.
