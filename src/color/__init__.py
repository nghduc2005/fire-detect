"""
Phân tích màu lửa & khói.

.. warning::
    Đây là **STUB tạm thời** do TV1 cung cấp để pipeline chạy end-to-end
    ngay từ đầu. **TV2 thay thế bằng cài đặt thực** theo đúng chữ ký hàm:

    - :func:`fire_color_mask(bundle, cfg)`  → ``np.ndarray`` uint8 {0, 255}
    - :func:`smoke_color_mask(bundle, cfg)` → ``np.ndarray`` uint8 {0, 255}

    Trong đó ``bundle`` là :class:`~src.preprocessing.FrameBundle`
    (có ``.bgr``, ``.hsv``, ``.ycrcb``, ``.gray``) và ``cfg`` là nhánh con
    của ``configs/default.yaml``.

Luật tham khảo:
- Lửa — RGB (Chen 2004) + YCbCr (Çelik 2009).
- Khói — gần xám (``max |R-G|, |G-B|, |R-B| <= α``), bão hòa thấp.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["fire_color_mask", "smoke_color_mask"]


def _empty_mask(bundle) -> np.ndarray:
    """Mask toàn 0 cùng kích thước frame — hook an toàn trước khi TV2 cài đặt."""
    return np.zeros((bundle.height, bundle.width), dtype=np.uint8)


def fire_color_mask(bundle, cfg: Mapping[str, Any]) -> np.ndarray:
    """Mask ứng viên lửa (uint8, giá trị 0 hoặc 255).

    TODO (TV2): Cài đặt luật RGB (Chen 2004) + YCbCr (Çelik 2009), có thể
    kết hợp AND / OR theo ``cfg["method"]``. Ngưỡng đọc từ:

    - ``cfg["rgb"]["r_threshold"]`` (R_T)
    - ``cfg["rgb"]["s_threshold"]`` (S_T)
    - ``cfg["ycbcr"]["tau"]``       (|Cb - Cr| >= tau)
    """
    return _empty_mask(bundle)


def smoke_color_mask(bundle, cfg: Mapping[str, Any]) -> np.ndarray:
    """Mask ứng viên khói (uint8, giá trị 0 hoặc 255).

    TODO (TV2): Cài đặt luật khói "gần xám + bão hòa thấp". Ngưỡng:

    - ``cfg["max_channel_diff"]`` (α)
    - ``cfg["s_max"]``, ``cfg["v_min"]``, ``cfg["v_max"]``
    """
    return _empty_mask(bundle)
