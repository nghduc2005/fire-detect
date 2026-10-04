"""Tiền xử lý: resize, lọc nhiễu, chuyển không gian màu."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessedFrame:
    bgr: np.ndarray      # Frame đã resize (chưa blur) - dùng để hiển thị
    blurred: np.ndarray  # Frame đã Gaussian blur - dùng để phân tích
    hsv: np.ndarray
    ycrcb: np.ndarray    # Thứ tự kênh OpenCV: Y, Cr, Cb
    gray: np.ndarray
    scale: float         # = kích thước gốc / kích thước đã resize

    @property
    def shape(self) -> tuple[int, int]:
        return self.gray.shape[:2]


def resize_keep_ratio(frame: np.ndarray, width: int | None) -> tuple[np.ndarray, float]:
    """Resize về chiều rộng `width`, giữ tỉ lệ. Trả về (ảnh, scale gốc/mới)."""
    h, w = frame.shape[:2]
    if not width or width <= 0 or w == width:
        return frame, 1.0
    scale = w / width
    new_h = max(1, int(round(h / scale)))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA), scale


def preprocess(frame: np.ndarray, cfg: dict) -> PreprocessedFrame:
    resized, scale = resize_keep_ratio(frame, cfg.get("resize_width", 640))
    k = int(cfg.get("blur_ksize", 5))
    if k > 1:
        k |= 1  # Kernel Gaussian phải là số lẻ
        blurred = cv2.GaussianBlur(resized, (k, k), 0)
    else:
        blurred = resized
    return PreprocessedFrame(
        bgr=resized,
        blurred=blurred,
        hsv=cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV),
        ycrcb=cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb),
        gray=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
        scale=scale,
    )
