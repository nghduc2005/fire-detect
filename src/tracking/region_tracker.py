"""Theo dõi vùng ứng viên qua chuỗi khung hình bằng giải thuật Greedy IoU Matching.

Thành viên 3 — Triển khai chính thức.
"""

from __future__ import annotations

from collections import deque
from itertools import count
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..detectors.base import Region


def compute_iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    """Tính Intersection-over-Union (IoU) giữa 2 bounding box dạng (x, y, w, h).

    Parameters
    ----------
    bbox_a : Tuple[int, int, int, int]
        Bounding box thứ nhất (x, y, w, h).
    bbox_b : Tuple[int, int, int, int]
        Bounding box thứ hai (x, y, w, h).

    Returns
    -------
    float
        Giá trị IoU trong đoạn [0.0, 1.0].
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    # Tọa độ vùng giao nhau
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    return float(inter_area) / float(union_area) if union_area > 0 else 0.0


class RegionTracker:
    """Bộ theo dõi vùng liên thông bằng phương pháp Greedy IoU Matching.

    Duy trì định danh `track_id` ổn định và lưu buffer lịch sử `history` qua thời gian
    cho các tầng phân tích chuỗi thời gian (flicker, biến thiên diện tích).
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.iou_threshold = float(self.cfg.get("iou_threshold", 0.1))
        self.max_missed = int(self.cfg.get("max_missed", 5))
        self.buffer_size = int(self.cfg.get("buffer_size", 30))

        self._tracks: Dict[int, Region] = {}
        self._missed: Dict[int, int] = {}
        self._history: Dict[int, deque[Dict[str, Any]]] = {}
        self._next_id = count(start=1)

    def update(self, regions: List[Region]) -> List[Region]:
        """Cập nhật các regions mới, ghép cặp với tracks cũ và duy trì lịch sử.

        Parameters
        ----------
        regions : List[Region]
            Danh sách vùng ứng viên trích xuất từ khung hình hiện tại.

        Returns
        -------
        List[Region]
            Danh sách vùng ứng viên đã được gán `track_id` và cập nhật `history`.
        """
        track_ids = list(self._tracks.keys())
        matched_tracks: set[int] = set()
        matched_regions: set[int] = set()

        # 1. Tính toán ma trận IoU và ghép cặp tham lam (Greedy Matching)
        if track_ids and regions:
            pairs: List[Tuple[float, int, int]] = []
            for tid in track_ids:
                prev_bbox = self._tracks[tid].bbox
                for ri, region in enumerate(regions):
                    iou = compute_iou(prev_bbox, region.bbox)
                    if iou >= self.iou_threshold:
                        pairs.append((iou, tid, ri))

            # Sắp xếp các cặp theo IoU giảm dần
            pairs.sort(key=lambda item: item[0], reverse=True)

            for iou, tid, ri in pairs:
                if tid not in matched_tracks and ri not in matched_regions:
                    # Ghép cặp thành công
                    regions[ri].track_id = tid
                    matched_tracks.add(tid)
                    matched_regions.add(ri)
                    self._missed[tid] = 0

        # 2. Xử lý từng region: cấp ID mới hoặc cập nhật track cũ
        for ri, region in enumerate(regions):
            if ri in matched_regions:
                tid = region.track_id
            else:
                # Region mới chưa có track tương ứng -> cấp phát track_id mới
                tid = next(self._next_id)
                region.track_id = tid
                self._missed[tid] = 0
                self._history[tid] = deque(maxlen=self.buffer_size)

            # Snapshot thông tin khung hình hiện tại vào lịch sử
            snapshot: Dict[str, Any] = {
                "bbox": region.bbox,
                "area": region.area,
                "centroid": region.centroid,
            }
            if tid not in self._history:
                self._history[tid] = deque(maxlen=self.buffer_size)
            self._history[tid].append(snapshot)
            region.history = list(self._history[tid])

            # Cập nhật thông tin track hiện thời
            self._tracks[tid] = region

        # 3. Tăng bộ đếm missed cho các track không xuất hiện và dọn dẹp track cũ
        stale_tracks: List[int] = []
        for tid in track_ids:
            if tid not in matched_tracks:
                self._missed[tid] = self._missed.get(tid, 0) + 1
                if self._missed[tid] > self.max_missed:
                    stale_tracks.append(tid)

        for tid in stale_tracks:
            self._tracks.pop(tid, None)
            self._missed.pop(tid, None)
            self._history.pop(tid, None)

        return regions

    def reset(self) -> None:
        """Reset toàn bộ trạng thái tracking."""
        self._tracks.clear()
        self._missed.clear()
        self._history.clear()
        self._next_id = count(start=1)

    @property
    def active_ids(self) -> set[int]:
        """Tập track_id hiện còn sống (kể cả vừa mất dấu trong vài frame)."""
        return set(self._tracks.keys())
