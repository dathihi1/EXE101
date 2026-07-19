import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class RecognitionCandidate {
  const RecognitionCandidate({required this.label, required this.confidence});
  final String label;
  final double confidence;

  factory RecognitionCandidate.fromJson(Map<String, dynamic> json) =>
      RecognitionCandidate(
        label: '${json['label'] ?? json['predictedLabel'] ?? ''}',
        confidence: _toDouble(json['confidence']),
      );
}

class RecognitionResult {
  const RecognitionResult({
    required this.status,
    required this.verified,
    required this.predictedSignId,
    required this.predictedLabel,
    required this.confidence,
    required this.top3,
    required this.qualityHints,
  });

  final String status;
  final bool verified;
  final String predictedSignId;
  final String predictedLabel;
  final double confidence;
  final List<RecognitionCandidate> top3;
  final List<String> qualityHints;

  factory RecognitionResult.fromJson(Map<String, dynamic> json) {
    final candidates = json['top3'] is List ? json['top3'] as List : const [];
    final hints = json['qualityHints'] is List
        ? json['qualityHints'] as List
        : const [];
    return RecognitionResult(
      status: '${json['status'] ?? 'unknown'}',
      verified: json['verified'] == true,
      predictedSignId: '${json['predictedSignId'] ?? ''}',
      predictedLabel: '${json['predictedLabel'] ?? ''}',
      confidence: _toDouble(json['confidence']),
      top3: candidates
          .whereType<Map>()
          .map(
            (item) =>
                RecognitionCandidate.fromJson(Map<String, dynamic>.from(item)),
          )
          .toList(growable: false),
      qualityHints: hints.map((item) => '$item').toList(growable: false),
    );
  }
}

abstract interface class RecognitionGateway {
  Future<RecognitionResult> recognize({
    required String expectedSignId,
    required List<Uint8List> jpegFrames,
  });
}

class BackendRecognitionGateway implements RecognitionGateway {
  BackendRecognitionGateway({http.Client? client, String? baseUrl})
    : _client = client ?? http.Client(),
      baseUrl = baseUrl ?? _defaultBaseUrl;

  static String get _defaultBaseUrl {
    const configured = String.fromEnvironment('API_BASE_URL');
    if (configured.isNotEmpty) return configured;
    if (kIsWeb) return 'http://127.0.0.1:8080';
    if (defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8080';
    }
    return 'http://127.0.0.1:8080';
  }

  final http.Client _client;
  final String baseUrl;
  String? _guestToken;

  Future<String> _token() async {
    if (_guestToken != null) return _guestToken!;
    final response = await _client
        .post(Uri.parse('$baseUrl/api/v1/auth/guest'))
        .timeout(const Duration(seconds: 15));
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw RecognitionException(
        'Không thể tạo phiên khách (${response.statusCode}).',
      );
    }
    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    final token = decoded is Map
        ? decoded['accessToken'] ?? decoded['token']
        : null;
    if (token == null) {
      throw const RecognitionException('Backend không trả về token hợp lệ.');
    }
    _guestToken = '$token';
    return _guestToken!;
  }

  @override
  Future<RecognitionResult> recognize({
    required String expectedSignId,
    required List<Uint8List> jpegFrames,
  }) async {
    if (jpegFrames.length < 8 || jpegFrames.length > 32) {
      throw const RecognitionException(
        'Cần từ 8 đến 32 khung hình để nhận diện.',
      );
    }
    final totalBytes = jpegFrames.fold<int>(
      0,
      (total, frame) => total + frame.length,
    );
    if (totalBytes > 2 * 1024 * 1024) {
      throw const RecognitionException(
        'Bản ghi vượt quá 2 MB. Hãy giảm chất lượng camera.',
      );
    }

    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/api/v1/practice/recognize'),
    );
    request.headers['Authorization'] = 'Bearer ${await _token()}';
    request.fields['expectedSignId'] = expectedSignId;
    request.fields['lessonId'] = 'camera-practice';
    for (var index = 0; index < jpegFrames.length; index++) {
      final frame = jpegFrames[index];
      if (frame.length < 3 ||
          frame[0] != 0xff ||
          frame[1] != 0xd8 ||
          frame[2] != 0xff) {
        throw RecognitionException(
          'Khung hình ${index + 1} không phải JPEG hợp lệ.',
        );
      }
      request.files.add(
        http.MultipartFile.fromBytes(
          'frames',
          frame,
          filename: 'frame-$index.jpg',
          contentType: MediaType('image', 'jpeg'),
        ),
      );
    }
    final streamed = await _client
        .send(request)
        .timeout(const Duration(seconds: 45));
    final response = await http.Response.fromStream(streamed);
    Object? decoded;
    try {
      decoded = jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException {
      throw RecognitionException(
        'Backend trả về dữ liệu không hợp lệ (${response.statusCode}).',
      );
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final message = decoded is Map
          ? decoded['message'] ?? decoded['detail']
          : null;
      throw RecognitionException(
        '${message ?? 'Nhận diện thất bại (${response.statusCode}).'}',
      );
    }
    if (decoded is! Map) {
      throw const RecognitionException('Phản hồi AI không đúng định dạng.');
    }
    return RecognitionResult.fromJson(Map<String, dynamic>.from(decoded));
  }
}

class RecognitionException implements Exception {
  const RecognitionException(this.message);
  final String message;
  @override
  String toString() => message;
}

double _toDouble(Object? value) =>
    value is num ? value.toDouble() : double.tryParse('$value') ?? 0;
