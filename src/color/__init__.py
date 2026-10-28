from __future__ import annotations
from typing import Any, Mapping
import numpy as np

from ..preprocessing import FrameBundle

__all__ = ["fire_color_mask", "smoke_color_mask"]


def _as_mask(condition: np.ndarray) -> np.ndarray:
    return np.where(condition, 255, 0).astype(np.uint8)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def fire_color_mask(bundle: FrameBundle, cfg: Mapping[str, Any]) -> np.ndarray:
    rgb_cfg = _mapping(cfg.get("rgb", {}))
    ycbcr_cfg = _mapping(cfg.get("ycbcr", {}))
    r_threshold = float(rgb_cfg.get("r_threshold", 125))
    s_threshold = float(rgb_cfg.get("s_threshold", 60))
    tau = float(ycbcr_cfg.get("tau", 40))
    if r_threshold <= 0:
        raise ValueError("fire_color.rgb.r_threshold must be greater than zero")
    if s_threshold < 0:
        raise ValueError("fire_color.rgb.s_threshold must not be negative")
    if tau < 0:
        raise ValueError("fire_color.ycbcr.tau must not be negative")

    b, g, r = (bundle.bgr[..., i].astype(np.float32) for i in range(3))
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.divide(
        (max_rgb - min_rgb) * 255.0,
        max_rgb,
        out=np.zeros_like(max_rgb),
        where=max_rgb > 0,
    )
    required_saturation = np.maximum(0.0, (255.0 - r) * s_threshold / r_threshold)
    rgb_rule = (r > r_threshold) & (r >= g) & (g > b) & (saturation >= required_saturation)

    # OpenCV YCrCb channel order is Y, Cr, Cb.
    y = bundle.ycrcb[..., 0].astype(np.float32)
    cr = bundle.ycrcb[..., 1].astype(np.float32)
    cb = bundle.ycrcb[..., 2].astype(np.float32)
    ycbcr_rule = (
        (y > cb)
        & (cr > cb)
        & (y > float(y.mean()))
        & (cr > float(cr.mean()))
        & (cb < float(cb.mean()))
        & (np.abs(cb - cr) >= tau)
    )

    method = str(cfg.get("method", "combined")).lower()
    if method == "rgb":
        selected = rgb_rule
    elif method == "ycbcr":
        selected = ycbcr_rule
    elif method in {"any", "or"}:
        selected = rgb_rule | ycbcr_rule
    elif method in {"combined", "and"}:
        selected = rgb_rule & ycbcr_rule
    else:
        raise ValueError("fire_color.method must be rgb, ycbcr, combined, or any")
    return _as_mask(selected)


def smoke_color_mask(bundle: FrameBundle, cfg: Mapping[str, Any]) -> np.ndarray:
    max_channel_diff = float(cfg.get("max_channel_diff", 25))
    s_max = float(cfg.get("s_max", 60))
    v_min = float(cfg.get("v_min", 80))
    v_max = float(cfg.get("v_max", 230))
    if v_min > v_max:
        raise ValueError("smoke_color.v_min must not exceed smoke_color.v_max")

    b, g, r = (bundle.bgr[..., i].astype(np.int16) for i in range(3))
    channel_difference = np.maximum.reduce((np.abs(r - g), np.abs(g - b), np.abs(r - b)))
    saturation = bundle.hsv[..., 1]
    value = bundle.hsv[..., 2]
    selected = (
        (channel_difference <= max_channel_diff)
        & (saturation <= s_max)
        & (value >= v_min)
        & (value <= v_max)
    )
    return _as_mask(selected)
