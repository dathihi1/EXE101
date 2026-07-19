import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:light_sound_mvp/services/recognition_gateway.dart';

class _RecordingClient extends http.BaseClient {
  http.MultipartRequest? recognitionRequest;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    if (request.url.path.endsWith('/auth/guest')) {
      return _json({'accessToken': 'test-token'});
    }
    if (request is http.MultipartRequest) {
      recognitionRequest = request;
      return _json({
        'status': 'ok',
        'verified': true,
        'predictedSignId': 'anh',
        'predictedLabel': 'Anh',
        'confidence': 0.9,
        'top3': [],
        'qualityHints': [],
      });
    }
    return _json({'message': 'unexpected request'}, statusCode: 400);
  }

  http.StreamedResponse _json(Object value, {int statusCode = 200}) {
    return http.StreamedResponse(
      Stream.value(utf8.encode(jsonEncode(value))),
      statusCode,
      headers: {'content-type': 'application/json'},
    );
  }
}

void main() {
  test('sends every camera frame as image/jpeg multipart data', () async {
    final client = _RecordingClient();
    final gateway = BackendRecognitionGateway(
      client: client,
      baseUrl: 'http://localhost:8080',
    );
    final jpeg = Uint8List.fromList([0xff, 0xd8, 0xff, 0xd9]);

    await gateway.recognize(
      expectedSignId: 'anh',
      jpegFrames: List.generate(12, (_) => jpeg),
    );

    final request = client.recognitionRequest;
    expect(request, isNotNull);
    expect(request!.files, hasLength(12));
    expect(
      request.files.every((file) => file.contentType.mimeType == 'image/jpeg'),
      isTrue,
    );
    expect(request.fields['expectedSignId'], 'anh');
  });

  test(
    'rejects bytes that are not JPEG before calling recognition API',
    () async {
      final client = _RecordingClient();
      final gateway = BackendRecognitionGateway(
        client: client,
        baseUrl: 'http://localhost:8080',
      );

      await expectLater(
        gateway.recognize(
          expectedSignId: 'anh',
          jpegFrames: List.generate(12, (_) => Uint8List.fromList([1, 2, 3])),
        ),
        throwsA(isA<RecognitionException>()),
      );
    },
  );
}
