"""Phát hiện chuyển động bằng trừ nền (MOG2 / KNN) và hiệu 3 khung hình.

Thành viên 3 — Triển khai chính thức thay thế stub.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import cv2
import numpy as np

from ..preprocessing import FrameBundle

__all__ = ["MotionDetector"]


class MotionDetector:
    """Bộ phát hiện chuyển động đa phương pháp (MOG2, KNN, Three-Frame Differencing).

    Parameters
    ----------
    cfg : Mapping[str, Any], optional
        Cấu hình từ `motion.*` trong `configs/default.yaml`.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.method = str(self.cfg.get("method", "mog2")).lower()
        self._subtractor: Optional[cv2.BackgroundSubtractor] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev2_gray: Optional[np.ndarray] = None
        self._init_subtractor()

    def _init_subtractor(self) -> None:
        """Khởi tạo thuật toán trừ nền tương ứng với cấu hình."""
        if self.method == "mog2":
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=int(self.cfg.get("history", 300)),
                varThreshold=float(self.cfg.get("var_threshold", 25)),
                detectShadows=bool(self.cfg.get("detect_shadows", False)),
            )
        elif self.method == "knn":
            self._subtractor = cv2.createBackgroundSubtractorKNN(
                history=int(self.cfg.get("history", 300)),
                dist2Threshold=float(self.cfg.get("dist2_threshold", 400)),
                detectShadows=bool(self.cfg.get("detect_shadows", False)),
            )
        else:
            self._subtractor = None

    def update(self, bundle: FrameBundle) -> np.ndarray:
        """Nhận :class:`~src.preprocessing.FrameBundle`, trả về mask chuyển động nhị phân uint8."""
        if self.method in ("mog2", "knn") and self._subtractor is not None:
            learning_rate = float(self.cfg.get("learning_rate", -1))
            fg_mask = self._subtractor.apply(bundle.bgr, learningRate=learning_rate)
            # Nhị phân hóa (loại trừ bóng xám = 127 nếu detectShadows được bật)
            _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
            return fg_mask.astype(np.uint8)

        if self.method == "frame_diff":
            return self._frame_diff(bundle.gray)

        # Fallback an toàn (hoặc method không hợp lệ): trả về full 255 để không chặn pipeline
        return np.full((bundle.height, bundle.width), 255, dtype=np.uint8)

    def _frame_diff(self, gray: np.ndarray) -> np.ndarray:
        """Three-frame differencing: |f(t) - f(t-1)| & |f(t-1) - f(t-2)|."""
        diff_threshold = int(self.cfg.get("diff_threshold", 20))

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return np.full(gray.shape, 255, dtype=np.uint8)

        if self._prev2_gray is None:
            diff = cv2.absdiff(gray, self._prev_gray)
            self._prev2_gray = self._prev_gray.copy()
            self._prev_gray = gray.copy()
            _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            return mask

        # Tính hiệu giữa frame t và t-1, t-1 và t-2
        diff1 = cv2.absdiff(gray, self._prev_gray)
        diff2 = cv2.absdiff(self._prev_gray, self._prev2_gray)
        _, m1 = cv2.threshold(diff1, diff_threshold, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(diff2, diff_threshold, 255, cv2.THRESH_BINARY)

        self._prev2_gray = self._prev_gray.copy()
        self._prev_gray = gray.copy()
        return cv2.bitwise_and(m1, m2)

    def reset(self) -> None:
        """Reset mô hình nền và bộ nhớ đệm khung hình."""
        self._prev_gray = None
        self._prev2_gray = None
        self._init_subtractor()
