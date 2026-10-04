"""Đặc trưng hình dạng và biến đổi hình dạng theo thời gian."""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


# ----------------------------- Theo từng frame -----------------------------
def circularity(area: float, perimeter: float) -> float:
    """4*pi*A / P^2: =1 với hình tròn, càng nhỏ biên càng phức tạp."""
    return float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0


def solidity(contour: np.ndarray) -> float:
    """Diện tích contour / diện tích bao lồi."""
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return float(cv2.contourArea(contour) / hull_area) if hull_area > 0 else 0.0


def boundary_roughness(contour: np.ndarray) -> float:
    """Chu vi contour / chu vi bao lồi (>= 1). Lửa có biên răng cưa -> giá trị lớn."""
    hull_perim = cv2.arcLength(cv2.convexHull(contour), True)
    return float(cv2.arcLength(contour, True) / hull_perim) if hull_perim > 0 else 1.0


# ----------------------------- Theo thời gian ------------------------------
def area_variation(areas: Sequence[float]) -> float:
    """Hệ số biến thiên diện tích std/mean. Lửa dao động mạnh, vật cứng ổn định."""
    a = np.asarray(areas, dtype=np.float64)
    return float(a.std() / a.mean()) if a.size >= 2 and a.mean() > 0 else 0.0


def area_growth(areas: Sequence[float]) -> float:
    """Độ dốc hồi quy tuyến tính của diện tích (chuẩn hóa theo trung bình) / frame.
    Khói lan rộng -> dương."""
    a = np.asarray(areas, dtype=np.float64)
    if a.size < 3 or a.mean() <= 0:
        return 0.0
    slope = np.polyfit(np.arange(a.size), a, 1)[0]
    return float(slope / a.mean())


def shape_change(contours: Sequence[np.ndarray]) -> float:
    """Trung bình khoảng cách Hu-moment (cv2.matchShapes) giữa các frame liên tiếp."""
    if len(contours) < 2:
        return 0.0
    dists = [cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
             for c1, c2 in zip(contours[:-1], contours[1:])]
    dists = [d for d in dists if np.isfinite(d)]
    return float(np.mean(dists)) if dists else 0.0


def upward_velocity(centroids: Sequence[tuple[float, float]]) -> float:
    """Vận tốc đi lên trung bình của tâm vùng (px/frame, dương = đi lên)."""
    if len(centroids) < 2:
        return 0.0
    ys = np.asarray([c[1] for c in centroids], dtype=np.float64)
    return float(-np.mean(np.diff(ys)))
