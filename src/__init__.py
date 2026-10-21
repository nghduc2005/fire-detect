"""
Hệ thống phát hiện khói / lửa từ video — package chính.

Kiến trúc module hóa:
- `src.video_io`      : đọc / ghi video.
- `src.preprocessing` : resize, blur, chuyển không gian màu.
- `src.color`         : luật màu lửa (RGB + YCbCr) và khói (gần xám, bão hòa thấp).
- `src.motion`        : phát hiện chuyển động (MOG2 / KNN / 3-frame diff) + optical flow.
- `src.filtering`     : hình thái học, trích xuất vùng ứng viên, luật hình học, scoring.
- `src.tracking`      : ghép vùng qua các frame (greedy IoU) + buffer lịch sử.
- `src.temporal`      : flicker (FFT / Wavelet), biến thiên hình dạng, đặc trưng khói.
- `src.detectors`     : BaseDetector interface + ClassicalDetector tích hợp.
- `src.evaluation`    : chỉ số đánh giá (Precision, Recall, F1, FAR, Latency).
- `src.visualization` : vẽ bbox + lưới debug.
"""

__version__ = "0.1.0"
