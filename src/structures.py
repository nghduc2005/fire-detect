"""Các cấu trúc dữ liệu dùng chung giữa các module."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BBox = tuple[int, int, int, int]  # x, y, w, h

FIRE = "fire"
SMOKE = "smoke"


@dataclass
class Region:
    """Một vùng ứng viên (connected component) trong một frame."""

    label: str
    bbox: BBox
    area: float
    perimeter: float
    centroid: tuple[float, float]
    contour: np.ndarray
    # Tín hiệu đo được tại frame hiện tại (vd. flicker, độ mờ biên...)
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class Detection:
    """Kết quả phát hiện cuối cùng (toạ độ theo frame GỐC)."""

    bbox: BBox
    label: str
    score: float
    track_id: int | None = None
    features: dict[str, float] = field(default_factory=dict)


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union của hai bbox (x, y, w, h)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
