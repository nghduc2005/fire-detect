"""Phân tích đặc trưng nhấp nháy (Flicker) của ngọn lửa trong chuỗi thời gian.

Thành viên 3 — Triển khai chính thức.
Dựa trên cơ sở lý thuyết Töreyin et al. (2006): ngọn lửa thực tế luôn dao động hỗn loạn
trong dải tần số 3 - 13 Hz, khác biệt hoàn toàn với các vật thể màu đỏ chuyển động cứng.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..detectors.base import Region

__all__ = [
    "compute_pixel_flicker",
    "compute_intensity_flicker",
    "compute_fft_flicker",
]


def _extract_intensity_series(history: List[Dict[str, Any]]) -> np.ndarray:
    """Trích xuất chuỗi giá trị độ sáng trung bình (intensity_mean) từ lịch sử vùng."""
    values = [
        float(snap["intensity_mean"])
        for snap in history
        if "intensity_mean" in snap and snap["intensity_mean"] is not None
    ]
    return np.array(values, dtype=np.float32)


def compute_pixel_flicker(
    region: Region,
    bundle: Any,
    flicker_window: int = 25,
) -> float:
    """Đo lường mức độ biến động diện tích và đổi chiều trạng thái pixel qua cửa sổ thời gian.

    Parameters
    ----------
    region : Region
        Vùng ứng viên lửa đang theo dõi.
    bundle : FrameBundle
        Bundle khung hình hiện tại.
    flicker_window : int
        Kích thước cửa sổ trượt khung hình để xét biến động.

    Returns
    -------
    float
        Điểm nhấp nháy pixel chuẩn hóa trong đoạn [0.0, 1.0].
    """
    if not region.history or len(region.history) < 3:
        return 0.0

    recent_history = region.history[-flicker_window:]
    areas = [float(snap.get("area", region.area)) for snap in recent_history]
    if len(areas) < 3:
        return 0.0

    areas_arr = np.array(areas, dtype=np.float32)
    mean_area = float(np.mean(areas_arr))
    if mean_area <= 0:
        return 0.0

    # Đếm số lần đạo hàm bậc nhất của diện tích đổi dấu (chuyển động rung/nhấp nháy)
    diffs = np.diff(areas_arr)
    signs = np.sign(diffs)
    # Loại trừ giá trị 0 không đổi dấu
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0

    sign_changes = int(np.sum(signs[:-1] != signs[1:]))
    max_possible_changes = max(1, len(signs) - 1)
    flicker_ratio = sign_changes / float(max_possible_changes)

    return float(np.clip(flicker_ratio, 0.0, 1.0))


def compute_intensity_flicker(
    region: Region,
    bundle: Any,
    min_history: int = 8,
) -> float:
    """Tính hệ số biến thiên độ sáng (Coefficient of Variation: CV = std / mean).

    Ngọn lửa thật luôn có cường độ sáng dao động liên tục do quá trình đốt cháy khí sinh nhiệt,
    trong khi vật thể rắn màu đỏ (áo, xe, đèn) có độ sáng ổn định khi chuyển động.

    Parameters
    ----------
    region : Region
        Vùng ứng viên lửa.
    bundle : FrameBundle
        Bundle khung hình hiện tại.
    min_history : int
        Số khung hình tối thiểu cần có trong lịch sử để tính độ lệch chuẩn tin cậy.

    Returns
    -------
    float
        Hệ số biến thiên chuẩn hóa trong đoạn [0.0, 1.0].
    """
    if len(region.history) < min_history:
        return 0.0

    intensities = _extract_intensity_series(region.history)
    if len(intensities) < min_history:
        return 0.0

    mean_val = float(np.mean(intensities))
    if mean_val <= 1e-3:
        return 0.0

    std_val = float(np.std(intensities))
    cv = std_val / mean_val

    # Chuẩn hóa: thông thường CV dao động từ 0.0 đến 0.5 đối với lửa
    # Ánh xạ tuyến tính về [0.0, 1.0]
    normalized_score = float(np.clip(cv / 0.5, 0.0, 1.0))
    return normalized_score


def compute_fft_flicker(
    region: Region,
    fps: float = 25.0,
    fire_freq_lo: float = 3.0,
    fire_freq_hi: float = 13.0,
    min_history: int = 16,
) -> float:
    """Phân tích phổ tần số bằng Biến đổi Fourier nhanh (FFT).

    Tính tỉ lệ năng lượng tập trung trong dải tần số đặc trưng của lửa (3 Hz - 13 Hz).

    Parameters
    ----------
    region : Region
        Vùng ứng viên lửa.
    fps : float
        Tốc độ khung hình của video (khung/giây).
    fire_freq_lo : float
        Tần số ngưỡng dưới (Hz).
    fire_freq_hi : float
        Tần số ngưỡng trên (Hz).
    min_history : int
        Số khung hình tối thiểu để thực hiện biến đổi FFT.

    Returns
    -------
    float
        Tỉ lệ năng lượng trong dải tần [3, 13] Hz, khoảng [0.0, 1.0].
    """
    intensities = _extract_intensity_series(region.history)
    if len(intensities) < min_history:
        return 0.0

    # Khử thành phần một chiều (DC component / zero-mean)
    detrended = intensities - np.mean(intensities)
    n = len(detrended)

    # Tính biến đổi Fourier thực
    fft_vals = np.fft.rfft(detrended)
    power_spectrum = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / max(1.0, fps))

    total_power = float(np.sum(power_spectrum))
    if total_power <= 1e-6:
        return 0.0

    # Trích xuất năng lượng trong dải 3 - 13 Hz
    in_band_mask = (freqs >= fire_freq_lo) & (freqs <= fire_freq_hi)
    band_power = float(np.sum(power_spectrum[in_band_mask]))

    ratio = band_power / total_power
    return float(np.clip(ratio, 0.0, 1.0))

