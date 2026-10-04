"""Đặc trưng riêng của khói.

- Khói bán trong suốt làm MỜ các cạnh của nền phía sau -> năng lượng tần số cao giảm.
  (B. U. Töreyin et al., "Wavelet based real-time smoke detection in video", EUSIPCO 2005)
- Khói có texture mịn, đồng nhất -> histogram LBP tập trung (entropy thấp).
"""
from __future__ import annotations

import numpy as np
import pywt
from skimage.feature import local_binary_pattern

from src.structures import BBox


def _crop(img: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return img[y:y + h, x:x + w]


def wavelet_hf_energy(gray_patch: np.ndarray) -> float:
    """Năng lượng trung bình các dải chi tiết (LH, HL, HH) của wavelet Haar 2D."""
    if gray_patch.size == 0 or min(gray_patch.shape[:2]) < 2:
        return 0.0
    _, (c_h, c_v, c_d) = pywt.dwt2(gray_patch.astype(np.float32), "haar")
    return float(np.mean(c_h ** 2 + c_v ** 2 + c_d ** 2))


def edge_blur_ratio(gray: np.ndarray, background: np.ndarray | None, bbox: BBox) -> float | None:
    """Tỉ lệ năng lượng tần số cao: vùng hiện tại / cùng vùng trên ảnh nền.
    < 1 nghĩa là vùng đang bị làm mờ (dấu hiệu khói)."""
    if background is None:
        return None
    e_bg = wavelet_hf_energy(_crop(background, bbox))
    if e_bg < 1.0:  # Nền quá phẳng -> không kết luận được
        return None
    return wavelet_hf_energy(_crop(gray, bbox)) / e_bg


def texture_smoothness(gray: np.ndarray, bbox: BBox, points: int = 8, radius: int = 1) -> float:
    """1 - entropy chuẩn hóa của histogram LBP 'uniform'. Càng gần 1 càng mịn."""
    patch = _crop(gray, bbox)
    if patch.size == 0 or min(patch.shape[:2]) < 2 * radius + 1:
        return 0.0
    lbp = local_binary_pattern(patch, points, radius, method="uniform")
    n_bins = points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) / np.log2(n_bins)
    return float(1.0 - entropy)
