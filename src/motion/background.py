"""Phát hiện chuyển động: MOG2 / KNN / hiệu 3 frame.

Đồng thời duy trì một ảnh nền xám (running average) dùng cho đặc trưng khói
(so sánh độ sắc nét của vùng hiện tại với nền).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, cfg: dict):
        self.method = cfg.get("method", "mog2")
        self.learning_rate = float(cfg.get("learning_rate", -1))
        self.diff_threshold = int(cfg.get("diff_threshold", 20))
        self.bg_alpha = float(cfg.get("bg_alpha", 0.02))
        history = int(cfg.get("history", 300))
        shadows = bool(cfg.get("detect_shadows", False))

        if self.method == "mog2":
            self._sub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=float(cfg.get("var_threshold", 25)),
                detectShadows=shadows)
        elif self.method == "knn":
            self._sub = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=float(cfg.get("dist2_threshold", 400)),
                detectShadows=shadows)
        elif self.method == "frame_diff":
            self._sub = None
            self._prev: deque[np.ndarray] = deque(maxlen=2)
        else:
            raise ValueError(f"motion.method không hợp lệ: {self.method}")

        self._bg_gray: np.ndarray | None = None

    def apply(self, frame_bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Trả về mask chuyển động uint8 (0/255)."""
        if self._sub is not None:
            fg = self._sub.apply(frame_bgr, learningRate=self.learning_rate)
            # Bóng (giá trị 127) bị loại, chỉ giữ foreground chắc chắn (255)
            _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        else:
            mask = self._three_frame_diff(gray)

        self._update_background(gray, mask)
        return mask

    def _three_frame_diff(self, gray: np.ndarray) -> np.ndarray:
        """Hiệu 3 frame: pixel chuyển động nếu khác cả frame t-1 và t-2."""
        if len(self._prev) < 2:
            self._prev.append(gray)
            return np.zeros_like(gray)
        d1 = cv2.absdiff(self._prev[0], self._prev[1])
        d2 = cv2.absdiff(self._prev[1], gray)
        self._prev.append(gray)
        _, m1 = cv2.threshold(d1, self.diff_threshold, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(d2, self.diff_threshold, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(m1, m2)

    def _update_background(self, gray: np.ndarray, motion_mask: np.ndarray) -> None:
        """Cập nhật nền xám chỉ tại các pixel KHÔNG chuyển động."""
        if self._bg_gray is None or self._bg_gray.shape != gray.shape:
            self._bg_gray = gray.astype(np.float32)
            return
        cv2.accumulateWeighted(gray, self._bg_gray, self.bg_alpha,
                               mask=cv2.bitwise_not(motion_mask))

    def background_gray(self) -> np.ndarray | None:
        return None if self._bg_gray is None else cv2.convertScaleAbs(self._bg_gray)
