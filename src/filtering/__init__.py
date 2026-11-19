"""
Module hậu xử lý của Thành viên 4.

Bốn thành phần giao tiếp trực tiếp với :class:`~src.detectors.classical.ClassicalDetector`:

- :func:`morphology_clean`     — Opening & Closing làm sạch mask.
- :func:`extract_candidates`   — Connected components + lọc hình học (diện tích,
  tỷ lệ khung hình, độ đặc) và trích các đặc trưng hình dạng
  (``boundary_roughness``, ``solidity``, ``aspect_ratio``, ``area_ratio``).
- :class:`FeatureScorer`       — Chấm điểm tuyến tính từng đoạn theo
  ``configs/default.yaml`` → ``scoring.*``.
- :class:`PersistenceVoter`    — Lọc bền vững theo cửa sổ thời gian
  (xuất hiện >= ``min_hits`` / ``window`` frame).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import cv2
import numpy as np

from ..detectors.base import Region

__all__ = [
    "morphology_clean",
    "extract_candidates",
    "FeatureScorer",
    "PersistenceVoter",
]


def morphology_clean(
    mask: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 7,
) -> np.ndarray:
    """Opening (khử đốm nhiễu) + Closing (lấp lỗ hổng) với kernel ellipse."""
    if mask is None or mask.size == 0:
        return mask
    cleaned = mask.copy()
    if open_ksize and open_ksize > 1:
        k_open = open_ksize if open_ksize % 2 == 1 else open_ksize + 1
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    if close_ksize and close_ksize > 1:
        k_close = close_ksize if close_ksize % 2 == 1 else close_ksize + 1
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    return cleaned


def extract_candidates(
    mask: np.ndarray,
    label: str,
    cfg: Optional[Mapping[str, Any]] = None,
) -> List[Region]:
    """Trích xuất các vùng liên thông và lọc theo luật hình học.

    Ngoài ``min_area`` / ``max_area_ratio`` / ``min_aspect`` / ``max_aspect``,
    hàm còn tính các đặc trưng hình dạng dùng cho ``FeatureScorer``:

    - ``boundary_roughness`` = perimeter / convex-hull perimeter (lửa mép gồ ghề).
    - ``solidity``           = area / convex-hull area.
    - ``area_ratio``         = area / frame area.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return []
    cfg = cfg or {}
    h_img, w_img = mask.shape[:2]
    total_img_area = float(h_img * w_img)

    min_area = int(cfg.get("min_area", 150))
    max_area_ratio = float(cfg.get("max_area_ratio", 0.6))
    min_aspect = float(cfg.get("min_aspect", 0.1))
    max_aspect = float(cfg.get("max_aspect", 6.0))
    min_solidity = float(cfg.get("min_solidity", 0.0))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    regions: List[Region] = []
    for i in range(1, num_labels):  # Bỏ qua label 0 (background)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or (area / total_img_area) > max_area_ratio:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if h <= 0 or w <= 0:
            continue
        aspect = float(w) / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        # Giữ mask cục bộ theo bbox cho các tầng temporal đo texture/độ mờ.
        roi_mask = ((labels[y : y + h, x : x + w] == i).astype(np.uint8)) * 255

        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        roughness = 1.0
        solidity = 1.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perim = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hull_perim = cv2.arcLength(hull, True) + 1e-6
            hull_area = cv2.contourArea(hull) + 1e-6
            roughness = float(perim / hull_perim)
            solidity = float(np.clip(area / hull_area, 0.0, 1.0))
        if solidity < min_solidity:
            continue

        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        regions.append(
            Region(
                bbox=(x, y, w, h),
                area=area,
                centroid=(cx, cy),
                label=label,
                mask=roi_mask,
                extras={
                    "boundary_roughness": roughness,
                    "solidity": solidity,
                    "aspect_ratio": aspect,
                    "area_ratio": float(area) / total_img_area,
                },
            )
        )
    return regions


class FeatureScorer:
    """Chấm điểm đa đặc trưng theo ``configs/default.yaml`` → ``scoring.*``.

    Mỗi đặc trưng có cấu hình ``[weight, lo, hi]``; điểm thành phần::

        norm = clip((x - lo) / (hi - lo), 0, 1)

    ``hi < lo`` cho phép đảo thang điểm (ví dụ ``edge_blur``: càng mờ hơn nền
    càng giống khói). Đặc trưng chưa có dữ liệu (ví dụ tầng temporal chưa đủ
    lịch sử) được gán điểm trung tính ``0.5``.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, kind: str = "fire") -> None:
        self.cfg = cfg or {}
        self.kind = kind
        self.threshold = float(self.cfg.get("threshold", 0.45))
        self.features_cfg: Mapping[str, Any] = self.cfg.get("features", {})

    def score(self, region: Region) -> Tuple[float, Dict[str, float]]:
        """Trả về ``(score, breakdown)`` — breakdown là điểm từng đặc trưng."""
        if not self.features_cfg:
            return 1.0, {}

        total_score = 0.0
        total_weight = 0.0
        breakdown: Dict[str, float] = {}
        for feat_name, params in self.features_cfg.items():
            if not isinstance(params, (list, tuple)) or len(params) < 3:
                continue
            weight, lo, hi = float(params[0]), float(params[1]), float(params[2])
            raw_val = region.extras.get(feat_name, None)
            if raw_val is None:
                norm_val = 0.5
            else:
                denom = (hi - lo) if abs(hi - lo) > 1e-9 else 1e-9
                norm_val = float(np.clip((float(raw_val) - lo) / denom, 0.0, 1.0))
            breakdown[feat_name] = round(norm_val, 4)
            total_score += weight * norm_val
            total_weight += weight

        final_score = (total_score / total_weight) if total_weight > 0 else 0.0
        return round(float(final_score), 4), breakdown


class PersistenceVoter:
    """Lọc nhiễu nhất thời bằng voting theo cửa sổ trượt.

    Một track được coi là hợp lệ khi có ít nhất ``min_hits`` lần xuất hiện
    trong ``window`` frame gần nhất. Detector gọi :meth:`begin_frame` đầu mỗi
    frame và :meth:`end_frame` với danh sách track đang hoạt động để các frame
    "trượt" được ghi nhận là 0 — nhờ vậy nhiễu ngắn không thể tích đủ điểm.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        cfg = cfg or {}
        self.window = int(cfg.get("window", 15))
        self.min_hits = int(cfg.get("min_hits", 8))
        self._history: Dict[int, Deque[int]] = {}
        self._voted_this_frame: Set[int] = set()

    def reset(self) -> None:
        self._history.clear()
        self._voted_this_frame.clear()

    def begin_frame(self) -> None:
        """Bắt đầu một frame mới."""
        self._voted_this_frame.clear()

    def vote(self, region: Region) -> bool:
        """Ghi nhận 1 hit cho track và trả về True nếu đã đủ bền vững."""
        tid = int(region.track_id)
        if tid < 0:
            # Không có tracker: chỉ cho qua nếu không yêu cầu persistence.
            return self.min_hits <= 1
        if tid not in self._history:
            self._history[tid] = deque(maxlen=self.window)
        self._history[tid].append(1)
        self._voted_this_frame.add(tid)
        return sum(self._history[tid]) >= self.min_hits

    def end_frame(self, active_track_ids: Iterable[int]) -> None:
        """Ghi 0 cho các track hoạt động nhưng không được vote trong frame này."""
        for tid in active_track_ids:
            tid = int(tid)
            if tid in self._voted_this_frame:
                continue
            if tid not in self._history:
                self._history[tid] = deque(maxlen=self.window)
            self._history[tid].append(0)
        self._prune(active_track_ids)

    def _prune(self, active_track_ids: Iterable[int]) -> None:
        """Xoá lịch sử của các track đã bị tracker loại bỏ."""
        alive = set(int(t) for t in active_track_ids)
        for tid in list(self._history.keys()):
            if tid not in alive:
                del self._history[tid]
