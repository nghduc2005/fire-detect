"""Phân tích đặc trưng theo thời gian: flicker, biến thiên hình dạng,
đặc trưng khói (mờ biên nền, LBP).

- :class:`FireTemporalAnalyzer`  — TV3 (flicker FFT & Wavelet/pixel)
- :class:`SmokeTemporalAnalyzer` — TV4 (mờ biên nền, LBP) + TV3 (hướng chuyển động)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..detectors.base import Region
from .flicker import (
    compute_fft_flicker,
    compute_intensity_flicker,
    compute_pixel_flicker,
)

__all__ = ["FireTemporalAnalyzer", "SmokeTemporalAnalyzer"]


class FireTemporalAnalyzer:
    """Phân tích đặc trưng nhấp nháy (Flicker) và biến động theo thời gian của lửa.

    Thành viên 3 — Triển khai chính thức.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.flicker_window = int(self.cfg.get("flicker_window", 25))
        self.min_history = int(self.cfg.get("min_history", 8))
        self.flicker_min_freq = float(self.cfg.get("flicker_min_freq", 3.0))

    def update(self, region: Region, bundle: Any) -> Dict[str, float]:
        """Cập nhật độ sáng vùng và tính toán các đặc trưng nhấp nháy cho một region.

        Ghi nhận trực tiếp kết quả tính toán vào `region.extras` để `FeatureScorer` sử dụng.
        """
        features: Dict[str, float] = {}

        # 1. Đo lường và lưu trữ cường độ sáng trung bình của ROI vào snapshot cuối
        if region.history:
            x, y, w, h = region.bbox
            h_img, w_img = bundle.height, bundle.width
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)

            if x2 > x1 and y2 > y1:
                roi_gray = bundle.gray[y1:y2, x1:x2]
                region.history[-1]["intensity_mean"] = float(np.mean(roi_gray))
            else:
                region.history[-1]["intensity_mean"] = 0.0

        # 2. Tính các đặc trưng nhấp nháy cốt lõi
        features["pixel_flicker"] = compute_pixel_flicker(
            region=region,
            bundle=bundle,
            flicker_window=self.flicker_window,
        )
        features["intensity_flicker"] = compute_intensity_flicker(
            region=region,
            bundle=bundle,
            min_history=self.min_history,
        )
        features["fft_flicker"] = compute_fft_flicker(
            region=region,
            fire_freq_lo=self.flicker_min_freq,
            min_history=max(12, self.min_history),
        )

        region.extras.update(features)
        return features

    def reset(self) -> None:
        """Reset trạng thái nội bộ nếu có."""
        return None


class SmokeTemporalAnalyzer:
    """STUB cho khói. TV4 cài đặt ``smoke_features.py`` + lấy ``upward_motion``
    từ optical flow của TV3.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg = cfg or {}

    def update(self, region: Region, bundle) -> Dict[str, float]:
        """TODO (TV4 + TV3):
        - edge_blur           (TV4, wavelet 2D)
        - texture_smoothness  (TV4, LBP)
        - upward_motion       (TV3, optical flow)
        - area_growth         (TV4)
        """
        features: Dict[str, float] = {}
        region.extras.update(features)
        return features

    def reset(self) -> None:
        return None
