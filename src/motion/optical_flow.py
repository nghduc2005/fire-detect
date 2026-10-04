"""Optical flow Farnebäck - dùng để đo hướng chuyển động của vùng khói (thường đi lên)."""
from __future__ import annotations

import cv2
import numpy as np

from src.structures import BBox


class OpticalFlowAnalyzer:
    """Tính flow *lười*: chỉ tính khi có vùng cần phân tích trong frame hiện tại."""

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.params = dict(
            pyr_scale=cfg.get("pyr_scale", 0.5), levels=cfg.get("levels", 3),
            winsize=cfg.get("winsize", 15), iterations=cfg.get("iterations", 3),
            poly_n=cfg.get("poly_n", 5), poly_sigma=cfg.get("poly_sigma", 1.2), flags=0,
        )
        self._prev: np.ndarray | None = None
        self._cur: np.ndarray | None = None
        self._flow: np.ndarray | None = None

    def set_frame(self, gray: np.ndarray) -> None:
        """Gọi MỖI frame để cập nhật cặp (frame trước, frame hiện tại)."""
        self._prev, self._cur = self._cur, gray
        self._flow = None

    def flow(self) -> np.ndarray | None:
        """Flow dày (H, W, 2) giữa frame trước và frame hiện tại."""
        if not self.enabled or self._prev is None or self._cur is None:
            return None
        if self._prev.shape != self._cur.shape:
            return None
        if self._flow is None:
            self._flow = cv2.calcOpticalFlowFarneback(self._prev, self._cur, None, **self.params)
        return self._flow

    def region_motion(self, bbox: BBox) -> dict[str, float] | None:
        """Thống kê chuyển động trong bbox. dy < 0 nghĩa là đi LÊN."""
        flow = self.flow()
        if flow is None:
            return None
        x, y, w, h = bbox
        patch = flow[y:y + h, x:x + w]
        if patch.size == 0:
            return None
        dx, dy = patch[..., 0], patch[..., 1]
        mag = np.hypot(dx, dy)
        moving = mag > 0.2
        return {
            "mean_dx": float(dx.mean()),
            "mean_dy": float(dy.mean()),
            "magnitude": float(mag.mean()),
            "upward_ratio": float((dy[moving] < 0).mean()) if moving.any() else 0.0,
        }
