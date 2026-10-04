"""Phân tích màu khói: khói thường xám/trắng, bão hòa thấp, độ sáng trung bình."""
from __future__ import annotations

import numpy as np

from src.preprocessing import PreprocessedFrame


def smoke_color_mask(pf: PreprocessedFrame, cfg: dict) -> np.ndarray:
    """Mask uint8 (0/255) các pixel có màu giống khói.

    1. max(|R-G|, |G-B|, |R-B|) <= max_channel_diff   (gần xám)
    2. S <= s_max                                     (bão hòa thấp)
    3. v_min <= V <= v_max                            (không quá tối / quá chói)
    """
    bgr = pf.blurred.astype(np.int16)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    max_diff = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    s = pf.hsv[..., 1]
    v = pf.hsv[..., 2]
    cond = (
        (max_diff <= cfg.get("max_channel_diff", 25))
        & (s <= cfg.get("s_max", 60))
        & (v >= cfg.get("v_min", 80))
        & (v <= cfg.get("v_max", 230))
    )
    return cond.astype(np.uint8) * 255
