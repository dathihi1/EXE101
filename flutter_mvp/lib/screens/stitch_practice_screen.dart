import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../data/training_catalog.dart';
import '../data/training_video_assets.dart';
import '../models/learning_models.dart';
import '../services/recognition_gateway.dart';
import '../theme/app_theme.dart';
import '../widgets/stitch_widgets.dart';

typedef FrameCapture = Future<List<Uint8List>> Function();

class StitchPracticeScreen extends StatefulWidget {
  const StitchPracticeScreen({
    super.key,
    required this.session,
    required this.gateway,
    this.enableMedia = true,
    this.active = true,
    this.selectedSignId = 'anh',
    this.frameCaptureOverride,
    required this.onMenu,
    required this.onNotifications,
  });

  final LearningSession session;
  final RecognitionGateway gateway;
  final bool enableMedia;
  final bool active;
  final String selectedSignId;
  final FrameCapture? frameCaptureOverride;
  final VoidCallback onMenu;
  final VoidCallback onNotifications;

  @override
  State<StitchPracticeScreen> createState() => _StitchPracticeScreenState();
}

class _StitchPracticeScreenState extends State<StitchPracticeScreen>
    with WidgetsBindingObserver {
  CameraController? _camera;
  VideoPlayerController? _sample;
  bool _initializingCamera = false;
  bool _processing = false;
  bool _mirror = true;
  String? _error;
  RecognitionResult? _result;
  int _selectedIndex = 0;
  int _capturedFrames = 0;

  SignLesson get _selected => allTrainingSigns[_selectedIndex].lesson;

  @override
  void initState() {
    super.initState();
    _selectedIndex = _indexForSign(widget.selectedSignId);
    WidgetsBinding.instance.addObserver(this);
    if (widget.enableMedia && widget.active) {
      _loadSample();
    }
  }

  @override
  void didUpdateWidget(covariant StitchPracticeScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    final selectedChanged = widget.selectedSignId != oldWidget.selectedSignId;
    if (selectedChanged) {
      _selectedIndex = _indexForSign(widget.selectedSignId);
      _result = null;
      _error = null;
    }
    if (widget.enableMedia &&
        widget.active &&
        (!oldWidget.active || selectedChanged)) {
      _loadSample();
    }
  }

  int _indexForSign(String signId) {
    final index = allTrainingSigns.indexWhere(
      (item) => item.lesson.id == signId,
    );
    return index < 0 ? 0 : index;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _camera?.dispose();
      _camera = null;
    }
  }

  Future<void> _loadSample() async {
    final previous = _sample;
    _sample = null;
    await previous?.dispose();
    final controller = VideoPlayerController.asset(
      trainingVideoAsset(_selected.id),
    );
    try {
      await controller.initialize();
      await controller.setLooping(true);
      await controller.setVolume(0);
      await controller.play();
      if (!mounted) {
        return controller.dispose();
      }
      setState(() => _sample = controller);
    } catch (_) {
      await controller.dispose();
      if (mounted) {
        setState(() => _error = 'Không thể tải video hướng dẫn cho từ này.');
      }
    }
  }

  Future<void> _enableCamera() async {
    if (!widget.enableMedia || _initializingCamera) {
      return;
    }
    setState(() {
      _initializingCamera = true;
      _error = null;
    });
    CameraController? next;
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw CameraException('NoCamera', 'Không tìm thấy camera.');
      }
      final front = cameras.where(
        (camera) => camera.lensDirection == CameraLensDirection.front,
      );
      next = CameraController(
        front.isNotEmpty ? front.first : cameras.first,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await next.initialize();
      await _camera?.dispose();
      if (!mounted) {
        return next.dispose();
      }
      setState(() => _camera = next);
    } on CameraException catch (exception) {
      await next?.dispose();
      if (mounted) {
        setState(
          () => _error = exception.code.contains('AccessDenied')
              ? 'Camera bị từ chối. Hãy cấp quyền camera rồi thử lại.'
              : 'Không thể mở camera: ${exception.description ?? exception.code}',
        );
      }
    } finally {
      if (mounted) {
        setState(() => _initializingCamera = false);
      }
    }
  }

  Future<List<Uint8List>> _captureFrames() async {
    if (widget.frameCaptureOverride != null) {
      return widget.frameCaptureOverride!();
    }
    final camera = _camera;
    if (camera == null || !camera.value.isInitialized) {
      throw const RecognitionException('Camera chưa sẵn sàng.');
    }
    final frames = <Uint8List>[];
    for (var index = 0; index < 12; index++) {
      final file = await camera.takePicture();
      frames.add(await file.readAsBytes());
      if (mounted) setState(() => _capturedFrames = index + 1);
      if (index < 11) {
        await Future<void>.delayed(const Duration(milliseconds: 140));
      }
    }
    return frames;
  }

  Future<void> _recordAndRecognize() async {
    if (_processing) return;
    if (widget.frameCaptureOverride == null &&
        !(_camera?.value.isInitialized ?? false)) {
      await _enableCamera();
      return;
    }
    setState(() {
      _processing = true;
      _capturedFrames = 0;
      _error = null;
      _result = null;
    });
    try {
      final frames = await _captureFrames();
      if (frames.length < 8 || frames.length > 32) {
        throw const RecognitionException(
          'Bản ghi phải có từ 8 đến 32 khung hình.',
        );
      }
      final result = await widget.gateway.recognize(
        expectedSignId: _selected.id,
        jpegFrames: frames,
      );
      widget.session.recordAttempt();
      widget.session.markLearned(_selected.id);
      if (result.verified) widget.session.markVerified(_selected.id);
      if (mounted) setState(() => _result = result);
    } catch (exception) {
      if (mounted) setState(() => _error = exception.toString());
    } finally {
      if (mounted) setState(() => _processing = false);
    }
  }

  void _selectWord(int? index) {
    if (index == null) return;
    setState(() {
      _selectedIndex = index;
      _result = null;
      _error = null;
    });
    if (widget.enableMedia && widget.active) {
      _loadSample();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _camera?.dispose();
    _sample?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cameraReady = _camera?.value.isInitialized ?? false;
    return Column(
      children: [
        StitchHeader(
          title: 'Luyện tập với AI',
          onMenu: widget.onMenu,
          onNotifications: widget.onNotifications,
        ),
        Expanded(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(16, 18, 16, 28),
            children: [
              DropdownButtonFormField<int>(
                key: const Key('trained-word-selector'),
                initialValue: _selectedIndex,
                isExpanded: true,
                decoration: const InputDecoration(
                  labelText: 'Chọn từ đã huấn luyện (30 từ)',
                  prefixIcon: Icon(Icons.sign_language_rounded),
                ),
                items: [
                  for (var index = 0; index < allTrainingSigns.length; index++)
                    DropdownMenuItem(
                      value: index,
                      child: Text(
                        '${index + 1}. ${allTrainingSigns[index].lesson.name}',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                ],
                onChanged: _processing ? null : _selectWord,
              ),
              const SizedBox(height: 16),
              Text(
                'Video mẫu${_mirror ? ' · chế độ gương' : ''}',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              _SampleVideo(
                controller: _sample,
                mirror: _mirror,
                enabled: widget.enableMedia,
              ),
              const SizedBox(height: 16),
              Text(
                'Camera của bạn',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              Container(
                height: 260,
                decoration: BoxDecoration(
                  color: AppColors.surfaceContainer,
                  border: Border.all(
                    color: AppColors.primaryContainer,
                    width: 2,
                  ),
                  borderRadius: BorderRadius.circular(8),
                ),
                clipBehavior: Clip.antiAlias,
                child: cameraReady
                    ? ColoredBox(
                        color: Colors.black,
                        child: Center(
                          child: Transform(
                            alignment: Alignment.center,
                            transform: Matrix4.diagonal3Values(
                              _mirror ? -1 : 1,
                              1,
                              1,
                            ),
                            child: AspectRatio(
                              aspectRatio: _camera!.value.aspectRatio,
                              child: CameraPreview(_camera!),
                            ),
                          ),
                        ),
                      )
                    : _CameraEmptyState(
                        loading: _initializingCamera,
                        enabled: widget.enableMedia,
                        onEnable: _enableCamera,
                      ),
              ),
              const SizedBox(height: 12),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Chế độ gương cho video mẫu và camera'),
                value: _mirror,
                activeTrackColor: AppColors.primaryContainer,
                onChanged: (value) => setState(() => _mirror = value),
              ),
              if (_processing) ...[
                const LinearProgressIndicator(),
                const SizedBox(height: 8),
                Text(
                  _capturedFrames < 12
                      ? 'Đang ghi: $_capturedFrames/12 khung hình'
                      : 'AI đang phân tích...',
                ),
              ],
              if (_error != null) ...[
                const SizedBox(height: 12),
                Text(
                  _error!,
                  key: const Key('recognition-error'),
                  style: const TextStyle(color: Color(0xFFBA1A1A)),
                ),
              ],
              if (_result != null) ...[
                const SizedBox(height: 12),
                _ResultCard(result: _result!),
              ],
              const SizedBox(height: 16),
              FilledButton.icon(
                key: const Key('camera-main-action'),
                onPressed: _processing || _initializingCamera
                    ? null
                    : _recordAndRecognize,
                icon: Icon(
                  cameraReady || widget.frameCaptureOverride != null
                      ? Icons.fiber_manual_record_rounded
                      : Icons.videocam_rounded,
                ),
                label: Text(
                  _processing
                      ? 'Đang xử lý...'
                      : cameraReady || widget.frameCaptureOverride != null
                      ? 'Ghi và gửi AI'
                      : 'Bật camera',
                ),
              ),
              const SizedBox(height: 10),
              Text(
                'Ứng dụng lấy 12 khung hình trong khoảng 2 giây, không lưu video thô trên thiết bị.',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _SampleVideo extends StatelessWidget {
  const _SampleVideo({
    required this.controller,
    required this.mirror,
    required this.enabled,
  });
  final VideoPlayerController? controller;
  final bool mirror;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    final ready = controller?.value.isInitialized ?? false;
    return Container(
      height: 220,
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(8),
      ),
      child: ready
          ? Center(
              child: Transform(
                alignment: Alignment.center,
                transform: Matrix4.diagonal3Values(mirror ? -1 : 1, 1, 1),
                child: AspectRatio(
                  aspectRatio: controller!.value.aspectRatio,
                  child: VideoPlayer(controller!),
                ),
              ),
            )
          : Center(
              child: enabled
                  ? const CircularProgressIndicator()
                  : const Text(
                      'Video mẫu bị tắt trong môi trường kiểm thử',
                      style: TextStyle(color: Colors.white),
                    ),
            ),
    );
  }
}

class _CameraEmptyState extends StatelessWidget {
  const _CameraEmptyState({
    required this.loading,
    required this.enabled,
    required this.onEnable,
  });
  final bool loading;
  final bool enabled;
  final VoidCallback onEnable;

  @override
  Widget build(BuildContext context) => Center(
    child: loading
        ? const CircularProgressIndicator()
        : FilledButton.icon(
            onPressed: enabled ? onEnable : null,
            icon: const Icon(Icons.videocam_rounded),
            label: Text(enabled ? 'Bật camera' : 'Camera tắt khi kiểm thử'),
          ),
  );
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result});
  final RecognitionResult result;

  @override
  Widget build(BuildContext context) {
    final percent = (result.confidence * 100).clamp(0, 100).toStringAsFixed(1);
    return OceanCard(
      key: const Key('recognition-result'),
      color: result.verified
          ? const Color(0xFFE8F7EE)
          : const Color(0xFFFFF4E5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            result.verified ? 'Đạt' : 'Chưa đạt',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 6),
          Text(
            'AI nhận diện: ${result.predictedLabel.isEmpty ? 'Không xác định' : result.predictedLabel} · Độ tin cậy $percent%',
          ),
          if (result.top3.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              'Top dự đoán: ${result.top3.map((item) => '${item.label} ${(item.confidence * 100).toStringAsFixed(0)}%').join(' · ')}',
            ),
          ],
          if (result.qualityHints.isNotEmpty) ...[
            const SizedBox(height: 8),
            for (final hint in result.qualityHints) Text('• $hint'),
          ],
        ],
      ),
    );
  }
}
