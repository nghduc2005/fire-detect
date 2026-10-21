"""
Phát hiện chuyển động.

.. warning::
    Đây là **STUB tạm thời** do TV1 cung cấp. **TV3 thay thế** bằng
    cài đặt MOG2 / KNN / frame_diff thực tế, giữ nguyên interface:

    - :class:`MotionDetector`
        - ``__init__(cfg)``
        - ``update(bundle) -> np.ndarray``  (uint8, mask foreground 0 / 255)
        - ``reset()``

Tham khảo ``configs/default.yaml``:

- ``motion.method`` : ``mog2`` | ``knn`` | ``frame_diff``
- ``motion.history``, ``var_threshold``, ``dist2_threshold``, ...
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

__all__ = ["MotionDetector"]


class MotionDetector:
    """Trừ nền sinh mask chuyển động.

    STUB: trả về mask toàn 255 (coi mọi điểm là chuyển động → không lọc).
    Khi TV3 cài đặt thực, pipeline tự động hoạt động chuẩn.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.method = str(self.cfg.get("method", "mog2"))
        # TODO (TV3): khởi tạo cv2.createBackgroundSubtractorMOG2 / KNN tại đây.

    def update(self, bundle) -> np.ndarray:
        """Nhận :class:`~src.preprocessing.FrameBundle`, trả mask chuyển động uint8."""
        # STUB: coi mọi pixel đều "có chuyển động" để không chặn bước AND mask ở trên.
        # TV3 sẽ thay bằng mask thật từ MOG2/KNN/frame_diff.
        return np.full((bundle.height, bundle.width), 255, dtype=np.uint8)

    def reset(self) -> None:
        """Reset mô hình nền (gọi khi bắt đầu video mới)."""
        # TODO (TV3): khởi tạo lại BackgroundSubtractor.
        return None
