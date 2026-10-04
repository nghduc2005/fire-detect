"""Hình thái học: làm sạch mask nhị phân."""
from __future__ import annotations

import cv2
import numpy as np


def _kernel(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def clean_mask(mask: np.ndarray, open_ksize: int = 3, close_ksize: int = 7) -> np.ndarray:
    """Opening (xóa điểm nhiễu lẻ) rồi Closing (lấp lỗ, nối mảnh gần nhau)."""
    out = mask
    if open_ksize and open_ksize > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _kernel(open_ksize))
    if close_ksize and close_ksize > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _kernel(close_ksize))
    return out
