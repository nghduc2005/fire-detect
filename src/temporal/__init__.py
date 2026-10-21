"""
Phân tích đặc trưng theo thời gian: flicker, biến thiên hình dạng,
đặc trưng khói (mờ biên nền, LBP).

.. warning::
    Đây là **STUB tạm thời** do TV1 cung cấp. **TV3 và TV4 thay thế**:

    - :class:`FireTemporalAnalyzer`  — TV3 (flicker) + TV4 (shape features)
        - ``update(region, bundle) -> dict``  — bổ sung điểm thành phần
    - :class:`SmokeTemporalAnalyzer` — TV4 (mờ biên nền, LBP) + TV3 (hướng chuyển động)
        - ``update(region, bundle) -> dict``

Các file con dự kiến:

- ``flicker.py``        — TV3: FFT + Wavelet 1D cho chuỗi cường độ pixel.
- ``shape_features.py`` — TV4: diện tích variation, Hu moments, boundary roughness.
- ``smoke_features.py`` — TV4: wavelet 2D (mờ biên), LBP (texture smoothness).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..detectors.base import Region

__all__ = ["FireTemporalAnalyzer", "SmokeTemporalAnalyzer"]


class FireTemporalAnalyzer:
    """STUB: không gán điểm temporal nào → score tổng của lửa phụ thuộc
    hoàn toàn vào ``FeatureScorer`` (cũng là stub ở giai đoạn này).
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg = cfg or {}

    def update(self, region: Region, bundle) -> Dict[str, float]:
        """Tính các đặc trưng temporal cho một region và gắn vào ``region.extras``.

        TODO (TV3 + TV4):
        - pixel_flicker, intensity_flicker  (TV3, ``temporal/flicker.py``)
        - area_variation, shape_change, boundary_roughness (TV4, ``shape_features.py``)
        """
        features: Dict[str, float] = {}
        region.extras.update(features)
        return features

    def reset(self) -> None:
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
