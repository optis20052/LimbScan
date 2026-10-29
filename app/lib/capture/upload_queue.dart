import 'dart:async';
import 'dart:collection';
import 'dart:io';

import 'package:limb_capture/limb_capture.dart';

import '../api/limbscan_api.dart';

/// Uploads captured photos in the background while the user keeps scanning (2 in flight, retried).
class UploadQueue {
  UploadQueue(this.api, this.scanId, {this.onChange});

  final LimbScanApi api;
  final String scanId;
  final void Function()? onChange;
  final Queue<CapturedFrame> _queue = Queue();
  final List<CapturedFrame> _failed = [];
  int _inFlight = 0;
  int uploaded = 0;
  int failed = 0;
  int queued = 0;

  void add(CapturedFrame f) {
    _queue.add(f);
    queued++;
    _pump();
  }

  void _pump() {
    while (_inFlight < 2 && _queue.isNotEmpty) {
      final f = _queue.removeFirst();
      _inFlight++;
      api.uploadFrame(scanId, File(f.path), f.toMeta()).then((_) {
        uploaded++;
        File(f.path).delete().ignore(); // keep phone storage small: the server has it now
      }).catchError((Object _) {
        failed++;
        _failed.add(f);
      }).whenComplete(() {
        _inFlight--;
        onChange?.call();
        _pump();
      });
    }
    onChange?.call();
  }

  bool get idle => _inFlight == 0 && _queue.isEmpty;

  /// Put photos whose upload failed (after its own retries) back in the queue.
  void retryFailed() {
    final again = [..._failed];
    _failed.clear();
    failed = 0;
    queued -= again.length;
    again.forEach(add);
  }

  Future<void> drain() async {
    while (!idle) {
      await Future<void>.delayed(const Duration(milliseconds: 250));
    }
  }
}
