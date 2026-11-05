"""Phân tích trường chuyển động bằng giải thuật Farnebäck Optical Flow.

Thành viên 3 — Triển khai chính thức.
Cung cấp công cụ tính Optical Flow và kiểm tra hướng chuyển động đi lên (upward motion)
phục vụ phân tích đặc trưng khói của Thành viên 4.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import cv2
import numpy as np

from ..detectors.base import Region

__all__ = ["compute_flow", "compute_upward_motion"]


def compute_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    cfg: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Tính trường vector chuyển động dày đặc (Dense Optical Flow) bằng giải thuật Farnebäck.

    Parameters
    ----------
    prev_gray : np.ndarray
        Khung hình xám ở thời điểm t-1 (độ sâu 8-bit).
    curr_gray : np.ndarray
        Khung hình xám ở thời điểm hiện tại t (độ sâu 8-bit).
    cfg : Mapping[str, Any], optional
        Cấu hình tham số Farnebäck từ `configs/default.yaml` (`optical_flow.*`).

    Returns
    -------
    np.ndarray
        Trường vector chuyển động kích thước (H, W, 2), với kênh 0 là dx (ngang)
        và kênh 1 là dy (dọc).
    """
    cfg = cfg or {}
    pyr_scale = float(cfg.get("pyr_scale", 0.5))
    levels = int(cfg.get("levels", 3))
    winsize = int(cfg.get("winsize", 15))
    iterations = int(cfg.get("iterations", 3))
    poly_n = int(cfg.get("poly_n", 5))
    poly_sigma = float(cfg.get("poly_sigma", 1.2))

    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=0,
    )
    return flow


def compute_upward_motion(
    region: Region,
    flow: np.ndarray,
) -> float:
    """Đo lường xu hướng khuếch tán và chuyển động hướng lên trên của vùng ứng viên.

    Trong hệ tọa độ ảnh OpenCV, gốc tọa độ (0, 0) ở góc trên bên trái, trục y hướng xuống.
    Do đó, chuyển động hướng lên trên tương ứng với vector vận tốc có thành phần `dy < 0`.

    Parameters
    ----------
    region : Region
        Vùng ứng viên (đặc biệt là vùng khói) cần đánh giá.
    flow : np.ndarray
        Trường Optical Flow của toàn bộ khung hình, kích thước (H, W, 2).

    Returns
    -------
    float
        Tỉ lệ pixel có xu hướng di chuyển hướng lên trên [0.0, 1.0].
    """
    x, y, w, h = region.bbox
    h_img, w_img = flow.shape[:2]

    # Cắt vùng quan tâm (ROI) hợp lệ nằm trọn trong khung hình
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi_flow = flow[y1:y2, x1:x2]
    dy = roi_flow[..., 1]
    total_pixels = dy.size

    if total_pixels == 0:
        return 0.0

    upward_pixels = int(np.sum(dy < -0.1))  # ngưỡng nhỏ để loại trừ nhiễu rung lắc nhẹ
    return float(upward_pixels) / float(total_pixels)

