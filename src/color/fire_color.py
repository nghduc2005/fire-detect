"""Phân tích màu lửa.

Tham khảo:
- T.-H. Chen et al., "An early fire-detection method based on image processing", ICIP 2004.
- T. Celik, H. Demirel, "Fire detection in video sequences using a generic color model",
  Fire Safety Journal, 2009.

Mọi hàm trả về mask uint8 (0 hoặc 255), cùng kích thước ảnh đầu vào.
"""
from __future__ import annotations

import numpy as np

from src.preprocessing import PreprocessedFrame


def fire_mask_rgb(bgr: np.ndarray, hsv: np.ndarray,
                  r_threshold: float = 125, s_threshold: float = 60) -> np.ndarray:
    """Luật RGB + bão hòa (Chen 2004).

    1. R > R_T
    2. R >= G > B
    3. S >= (255 - R) * S_T / R_T
    """
    b = bgr[..., 0].astype(np.int16)
    g = bgr[..., 1].astype(np.int16)
    r = bgr[..., 2].astype(np.int16)
    s = hsv[..., 1].astype(np.float32)
    cond = (
        (r > r_threshold)
        & (r >= g) & (g > b)
        & (s >= (255 - r) * s_threshold / r_threshold)
    )
    return cond.astype(np.uint8) * 255


def fire_mask_ycbcr(ycrcb: np.ndarray, tau: float = 40) -> np.ndarray:
    """Luật YCbCr (Celik & Demirel 2009).

    1. Y > Cb,  Cr > Cb
    2. Y > Y_mean,  Cb < Cb_mean,  Cr > Cr_mean   (so với trung bình toàn ảnh)
    3. |Cb - Cr| >= tau
    """
    y = ycrcb[..., 0].astype(np.float32)
    cr = ycrcb[..., 1].astype(np.float32)
    cb = ycrcb[..., 2].astype(np.float32)
    cond = (
        (y > cb) & (cr > cb)
        & (y > y.mean()) & (cb < cb.mean()) & (cr > cr.mean())
        & (np.abs(cb - cr) >= tau)
    )
    return cond.astype(np.uint8) * 255


def fire_color_mask(pf: PreprocessedFrame, cfg: dict) -> np.ndarray:
    """Kết hợp các luật màu theo `cfg['method']`: rgb | ycbcr | combined | any."""
    method = cfg.get("method", "combined")
    rgb_cfg, ycc_cfg = cfg.get("rgb", {}), cfg.get("ycbcr", {})

    if method == "rgb":
        return fire_mask_rgb(pf.blurred, pf.hsv, **rgb_cfg)
    if method == "ycbcr":
        return fire_mask_ycbcr(pf.ycrcb, **ycc_cfg)

    m_rgb = fire_mask_rgb(pf.blurred, pf.hsv, **rgb_cfg)
    m_ycc = fire_mask_ycbcr(pf.ycrcb, **ycc_cfg)
    if method == "combined":
        return np.bitwise_and(m_rgb, m_ycc)
    if method == "any":
        return np.bitwise_or(m_rgb, m_ycc)
    raise ValueError(f"fire_color.method không hợp lệ: {method}")
