"""Đặc trưng hình dạng / biến thiên thời gian của vùng lửa — Thành viên 4.

Bổ sung các đặc trưng mà ``scoring.fire`` trong ``configs/default.yaml`` cần:

- ``area_variation``      : hệ số biến thiên diện tích vùng (std / mean) trên
  cửa sổ lịch sử — lửa thật phập phồng liên tục.
- ``shape_change``        : mức thay đổi hình dạng giữa 2 frame liên tiếp,
  đo bằng ``1 - IoU(bbox_t, bbox_{t-1})`` trung bình trên cửa sổ.
- ``boundary_roughness``  : độ gồ ghề biên (perimeter / convex-hull perimeter),
  lấy từ ``Region.extras`` do khâu trích candidates tính sẵn; nếu thiếu thì
  tính trực tiếp từ ``Region.mask``.

Module này chỉ đọc ``region.history`` (do :class:`~src.tracking.RegionTracker`
của TV3 duy trì) nên không giữ trạng thái riêng.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

from ..detectors.base import Region

__all__ = [
    "compute_area_variation",
    "compute_shape_change",
    "compute_boundary_roughness",
    "ShapeTemporalAnalyzer",
]


def _recent_history(region: Region, window: int) -> list[Dict[str, Any]]:
    if not region.history:
        return []
    return list(region.history[-max(2, int(window)) :])


def compute_area_variation(region: Region, window: int = 25) -> float:
    """Hệ số biến thiên diện tích (CV = std / mean) trên cửa sổ gần nhất."""
    history = _recent_history(region, window)
    areas = [float(snap.get("area", region.area)) for snap in history]
    if len(areas) < 3:
        return 0.0
    mean_area = float(np.mean(areas))
    if mean_area <= 1e-6:
        return 0.0
    return float(np.std(areas) / mean_area)


def compute_shape_change(region: Region, window: int = 25) -> float:
    """Trung bình ``1 - IoU`` giữa các bbox liên tiếp trong cửa sổ."""
    history = _recent_history(region, window)
    if len(history) < 2:
        return 0.0
    changes = []
    for prev, curr in zip(history[:-1], history[1:]):
        pb = prev.get("bbox")
        cb = curr.get("bbox")
        if pb is None or cb is None:
            continue
        changes.append(1.0 - _bbox_iou(pb, cb))
    if not changes:
        return 0.0
    return float(np.mean(changes))


def _bbox_iou(a: Any, b: Any) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def compute_boundary_roughness(region: Region) -> float:
    """Độ gồ ghề biên; ưu tiên giá trị đã tính ở ``extract_candidates``."""
    cached = region.extras.get("boundary_roughness")
    if cached is not None:
        return float(cached)
    if region.mask is None or region.mask.size == 0:
        return 1.0
    contours, _ = cv2.findContours(
        region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 1.0
    cnt = max(contours, key=cv2.contourArea)
    perim = cv2.arcLength(cnt, True)
    hull_perim = cv2.arcLength(cv2.convexHull(cnt), True) + 1e-6
    return float(perim / hull_perim)


class ShapeTemporalAnalyzer:
    """Ghi ``area_variation``, ``shape_change``, ``boundary_roughness`` vào extras."""

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        cfg = cfg or {}
        fire_cfg = cfg.get("fire", {})
        if not isinstance(fire_cfg, Mapping):
            fire_cfg = {}
        self.window = int(fire_cfg.get("window", cfg.get("flicker_window", 25)))

    def reset(self) -> None:
        # Stateless: dữ liệu nằm trong region.history của tracker.
        return None

    def update(self, region: Region, bundle: Any = None) -> Dict[str, float]:
        features: Dict[str, float] = {}
        history_len = len(region.history)
        # Chưa đủ lịch sử -> bỏ trống để FeatureScorer dùng điểm trung tính 0.5
        if history_len >= 3:
            features["area_variation"] = round(
                compute_area_variation(region, self.window), 4
            )
        if history_len >= 2:
            features["shape_change"] = round(
                compute_shape_change(region, self.window), 4
            )
        features["boundary_roughness"] = round(
            compute_boundary_roughness(region), 4
        )
        region.extras.update(features)
        return features
