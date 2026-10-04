"""Phân tích nhấp nháy (flicker) của lửa theo thời gian.

Lửa thật dao động ngẫu nhiên ~1-13 Hz; vật thể màu đỏ chuyển động cứng
(áo đỏ, xe đỏ) gần như không có thành phần tần số cao này.

Tham khảo: B. U. Töreyin et al., "Computer vision based method for real-time
fire and flame detection", Pattern Recognition Letters, 2006.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
import pywt


class PixelFlickerMap:
    """Đếm số lần mỗi pixel đổi trạng thái (thuộc / không thuộc mask màu lửa)
    trong cửa sổ `window` frame gần nhất. Cập nhật tăng dần O(H*W) mỗi frame.
    """

    def __init__(self, window: int = 25):
        self.window = window
        self._prev: np.ndarray | None = None
        self._transitions: deque[np.ndarray] = deque()
        self._counts: np.ndarray | None = None

    def reset(self) -> None:
        self._prev, self._counts = None, None
        self._transitions.clear()

    def update(self, mask: np.ndarray) -> np.ndarray:
        """Nhận mask uint8 (0/255), trả về bản đồ số lần đổi trạng thái (uint16)."""
        cur = mask > 0
        if self._prev is None or self._prev.shape != cur.shape:
            self.reset()
            self._prev = cur
            self._counts = np.zeros(cur.shape, np.uint16)
            return self._counts

        trans = (cur ^ self._prev).astype(np.uint16)
        self._transitions.append(trans)
        self._counts += trans
        if len(self._transitions) > self.window:
            self._counts -= self._transitions.popleft()
        self._prev = cur
        return self._counts

    def normalized(self) -> np.ndarray:
        """Bản đồ flicker chuẩn hóa về [0, 1]."""
        if self._counts is None:
            return np.zeros((1, 1), np.float32)
        return self._counts.astype(np.float32) / max(1, self.window)


def temporal_flicker_score(series: Sequence[float], fps: float, min_freq: float = 3.0) -> float:
    """Tỉ lệ năng lượng phổ (FFT) ở tần số >= min_freq trên tổng năng lượng (bỏ DC)."""
    x = np.asarray(series, dtype=np.float64)
    if x.size < 8 or x.std() < 1e-6:
        return 0.0
    x = (x - x.mean()) * np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fps)
    total = spec[1:].sum()
    return float(spec[freqs >= min_freq].sum() / total) if total > 0 else 0.0


def wavelet_flicker_score(series: Sequence[float], wavelet: str = "haar") -> float:
    """Tỉ lệ năng lượng dải chi tiết (high-band) của biến đổi wavelet 1 mức."""
    x = np.asarray(series, dtype=np.float64)
    if x.size < 4 or x.std() < 1e-6:
        return 0.0
    c_a, c_d = pywt.dwt(x - x.mean(), wavelet)
    e_a, e_d = float(np.sum(c_a ** 2)), float(np.sum(c_d ** 2))
    return e_d / (e_a + e_d) if (e_a + e_d) > 0 else 0.0


def sign_change_rate(series: Sequence[float]) -> float:
    """Tỉ lệ đổi chiều của đạo hàm (tăng <-> giảm) - đo mức "lắc" của tín hiệu."""
    d = np.sign(np.diff(np.asarray(series, dtype=np.float64)))
    d = d[d != 0]
    if d.size < 2:
        return 0.0
    return float(np.mean(d[1:] != d[:-1]))
