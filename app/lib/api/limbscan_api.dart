import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';

class ScanSummary {
  ScanSummary(this.id, this.name, this.stage, this.created, this.summary);

  factory ScanSummary.fromJson(Map<String, dynamic> j) => ScanSummary(
        j['id'] as String,
        (j['name'] ?? j['id']) as String,
        (j['stage'] ?? 'capturing') as String,
        DateTime.fromMillisecondsSinceEpoch(((j['created'] as num?) ?? 0).toInt() * 1000),
        j['summary'] as Map<String, dynamic>?,
      );

  final String id;
  final String name;
  final String stage;
  final DateTime created;
  final Map<String, dynamic>? summary;

  double? get wristGirth => (summary?['wrist_girth_mm'] as num?)?.toDouble();
}

class ScanStatus {
  ScanStatus(this.stage, this.progress, this.error, this.photos, {this.summary, this.measurements});

  final String stage;
  final double progress;
  final String? error;
  final int photos;
  final Map<String, dynamic>? summary;
  final Map<String, dynamic>? measurements;

  double? value(String k) => (summary?[k] as num?)?.toDouble();

  bool get done => stage == 'done';
  bool get failed => stage == 'failed';
}

/// Client for the LimbScan server REST API (the same API the web app uses).
class LimbScanApi {
  LimbScanApi(String baseUrl)
      : _dio = Dio(BaseOptions(baseUrl: baseUrl, connectTimeout: const Duration(seconds: 10),
            receiveTimeout: const Duration(seconds: 60), sendTimeout: const Duration(seconds: 120)));

  final Dio _dio;

  Future<bool> ping() async {
    try {
      final r = await _dio.get('/api/scans');
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<String> createScan(String name) async {
    final r = await _dio.post('/api/scans', data: FormData.fromMap({'name': name}));
    return r.data['id'] as String;
  }

  /// Upload one photo with its AR metadata (pose + intrinsics). Retries on network errors.
  /// A photo is ~0.7 MB: if a request has not finished in 25 s it is hung (tunnel hiccup), not slow. Give up on
  /// that attempt quickly and retry, instead of waiting minutes. The server overwrites a re-sent photo.
  Future<void> uploadFrame(String scanId, File jpeg, Map<String, Object> meta, {int retries = 5}) async {
    for (var attempt = 0;; attempt++) {
      try {
        await _dio.post('/api/scans/$scanId/files',
            data: FormData.fromMap({
              'files': await MultipartFile.fromFile(jpeg.path, filename: jpeg.uri.pathSegments.last),
              'meta': jsonEncode(meta),
            }),
            options: Options(sendTimeout: const Duration(seconds: 25), receiveTimeout: const Duration(seconds: 25)));
        return;
      } on DioException {
        if (attempt >= retries) rethrow;
        await Future<void>.delayed(Duration(milliseconds: 500 * (attempt + 1)));
      }
    }
  }

  /// The phone's log for a scan session (replaces what was sent before for the same [session]).
  Future<void> sendLog(String session, String text) =>
      _dio.post('/api/log', data: FormData.fromMap({'session': session, 'text': text}));

  Future<void> reportCrash(String text) => _dio.post('/api/crash', data: FormData.fromMap({'text': text}));

  /// Process the scan. [arm]: the arm cylinder as finally placed ({a, b, r}, AR world metres): the server cuts every
  /// photo out with it.
  Future<void> start(String scanId, {Map<String, Object>? arm}) => _dio.post('/api/scans/$scanId/start',
      data: arm == null ? null : FormData.fromMap({'arm': jsonEncode(arm)}));

  Future<void> delete(String scanId) => _dio.delete('/api/scans/$scanId');

  Future<List<ScanSummary>> listScans() async {
    final r = await _dio.get('/api/scans');
    return (r.data as List).map((e) => ScanSummary.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<ScanStatus> status(String scanId) async {
    final r = await _dio.get('/api/scans/$scanId');
    final s = r.data['status'] as Map<String, dynamic>;
    return ScanStatus(s['stage'] as String, ((s['progress'] as num?) ?? 0).toDouble(), s['error'] as String?,
        (r.data['n_input'] as num?)?.toInt() ?? 0,
        summary: s['summary'] as Map<String, dynamic>?, measurements: r.data['measurements'] as Map<String, dynamic>?);
  }

  String fileUrl(String scanId, String name) => '${_dio.options.baseUrl}/api/scans/$scanId/file/result/$name';

  String get baseUrl => _dio.options.baseUrl;

  String resultPageUrl(String scanId) => '${_dio.options.baseUrl}/#/scan/$scanId';
}
