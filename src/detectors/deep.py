"""(GIAI ĐOẠN MỞ RỘNG) Bộ phát hiện dùng học sâu.

Gợi ý hai hướng, cả hai đều cài đặt `BaseDetector`:

1. `HybridDetector`: dùng `ClassicalDetector` sinh vùng ứng viên, crop từng vùng
   rồi đưa qua CNN nhẹ (MobileNetV3 / EfficientNet-B0) để xác minh fire/smoke/none.
2. `YoloDetector`: YOLOv8 / YOLO11 (thư viện `ultralytics`) huấn luyện trên D-Fire.

Cần bỏ comment phần học sâu trong requirements.txt trước khi cài đặt.
"""
from __future__ import annotations

import numpy as np

from src.structures import Detection

from .base import BaseDetector


class YoloDetector(BaseDetector):
    def __init__(self, weights: str, conf: float = 0.25):
        super().__init__()
        self.weights = weights
        self.conf = conf
        raise NotImplementedError("Giai đoạn mở rộng: cài `ultralytics` và cài đặt lớp này.")

    def process(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError
