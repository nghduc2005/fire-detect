"""
Lọc hình thái học, trích xuất candidates, luật hình học, scoring, persistence.

.. warning::
    Đây là **STUB tạm thời** do TV1 cung cấp. **TV4 thay thế** bằng cài
    đặt thực, giữ chữ ký interface:

    - :func:`morphology_clean(mask, open_ksize, close_ksize)` -> ``np.ndarray``
    - :func:`extract_candidates(mask, label, cfg)` -> ``list[Region]``
    - :class:`FeatureScorer`
    - :class:`PersistenceVoter`
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

import cv2
import numpy as np

from ..detectors.base import Region

__all__ = [
    "morphology_clean",
    "extract_candidates",
    "FeatureScorer",
    "PersistenceVoter",
]


# ---------------------------------------------------------------------- #
# Morphology (TV1 cài tạm — có thể giữ nguyên vì đã đủ chuẩn OpenCV)
# ---------------------------------------------------------------------- #
def morphology_clean(
    mask: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 7,
) -> np.ndarray:
    """Opening (khử đốm) + Closing (lấp lỗ) với kernel vuông elliptic.

    TV4 có thể giữ nguyên hoặc tinh chỉnh.
    """
    if mask is None or mask.size == 0:
        return mask
    out = mask
    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_ksize), int(open_ksize)))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


# ---------------------------------------------------------------------- #
# Trích xuất connected components (TV1 cài tạm — chuẩn OpenCV)
# ---------------------------------------------------------------------- #
def extract_candidates(
    mask: np.ndarray,
    label: str,
    cfg: Optional[Mapping[str, Any]] = None,
) -> List[Region]:
    """Tách mask thành danh sách :class:`Region`.

    Lọc theo ``cfg`` (``min_area``, ``max_area_ratio``, ``min_aspect``,
    ``max_aspect``). TV4 có thể bổ sung các luật hình học khác (solidity,
    circularity, ...).
    """
    if mask is None or mask.size == 0:
        return []
    cfg = cfg or {}
    min_area = int(cfg.get("min_area", 150))
    max_area_ratio = float(cfg.get("max_area_ratio", 0.6))
    min_aspect = float(cfg.get("min_aspect", 0.1))
    max_aspect = float(cfg.get("max_aspect", 6.0))

    frame_area = int(mask.shape[0] * mask.shape[1])
    max_area = max_area_ratio * frame_area

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    regions: List[Region] = []
    # Bỏ qua label 0 (background)
    for i in range(1, num_labels):
        x, y, w, h, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < min_area or area > max_area:
            continue
        if h <= 0:
            continue
        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        regions.append(
            Region(
                bbox=(x, y, w, h),
                area=area,
                centroid=(cx, cy),
                label=label,
            )
        )
    return regions


# ---------------------------------------------------------------------- #
# FeatureScorer — TODO (TV4)
# ---------------------------------------------------------------------- #
class FeatureScorer:
    """Chấm điểm multi-feature theo ``configs/default.yaml`` → ``scoring.*``.

    STUB (TV1): trả 0.5 cố định để pipeline test được pass-through logic.
    **TV4 thay bằng công thức tổ hợp các đặc trưng theo weight.**
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, kind: str = "fire") -> None:
        self.cfg = cfg or {}
        self.kind = kind
        self.threshold = float(self.cfg.get("threshold", 0.45))
        # TODO (TV4): parse self.cfg["features"] → self.weights, self.ranges.

    def score(self, region: Region) -> tuple[float, dict]:
        """Trả về ``(score, breakdown)``.

        STUB: score = 0.0 nên không có vùng nào vượt ngưỡng → không có FP.
        """
        return 0.0, {}


# ---------------------------------------------------------------------- #
# Persistence voting — TODO (TV4)
# ---------------------------------------------------------------------- #
class PersistenceVoter:
    """Đếm số lần một track xuất hiện trong N frame gần nhất.

    Vùng chỉ được coi là hợp lệ khi có ít nhất ``min_hits`` lần trong
    ``window`` frame gần nhất → loại nhiễu ngắn hạn.

    STUB (TV1): luôn trả ``True`` nên pipeline chạy; **TV4 thay thế bằng
    sliding-window voting thực**.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.window = int(self.cfg.get("window", 15))
        self.min_hits = int(self.cfg.get("min_hits", 8))
        # TODO (TV4): self._hits_per_track: dict[int, deque[int]] = {}

    def vote(self, region: Region) -> bool:
        """Trả về True nếu vùng đã đủ bền vững."""
        return True
