"""Theo dõi vùng ứng viên qua các frame (greedy IoU matching).

Thành viên 3 — Triển khai chính thức.
"""

from __future__ import annotations

from .region_tracker import RegionTracker, compute_iou

__all__ = ["RegionTracker", "compute_iou"]
