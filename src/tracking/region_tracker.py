"""Theo dõi vùng ứng viên qua các frame bằng ghép IoU tham lam (greedy).

Mỗi `Track` giữ lịch sử N frame gần nhất -> nguồn dữ liệu cho phân tích
biến đổi hình dạng theo thời gian.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.structures import BBox, Region, bbox_iou


@dataclass
class Track:
    track_id: int
    label: str
    bbox: BBox
    history: deque[Region]
    hits: deque[bool]          # True nếu được ghép ở frame đó
    missed: int = 0            # Số frame mất dấu liên tiếp
    age: int = 0               # Tổng số frame kể từ khi tạo
    last_score: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def last(self) -> Region:
        return self.history[-1]

    @property
    def is_visible(self) -> bool:
        return self.missed == 0


class RegionTracker:
    def __init__(self, iou_threshold: float = 0.1, max_missed: int = 5, buffer_size: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.buffer_size = buffer_size
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, regions: list[Region]) -> list[Track]:
        """Ghép vùng mới vào track cũ (cùng nhãn). Trả về danh sách track còn sống."""
        matched_tracks: set[int] = set()
        matched_regions: set[int] = set()

        # Tính mọi cặp (IoU, track, region) cùng nhãn, ghép theo IoU giảm dần
        pairs = []
        for ti, t in enumerate(self._tracks):
            for ri, r in enumerate(regions):
                if t.label != r.label:
                    continue
                iou = bbox_iou(t.bbox, r.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, ri))
        for _, ti, ri in sorted(pairs, reverse=True):
            if ti in matched_tracks or ri in matched_regions:
                continue
            matched_tracks.add(ti)
            matched_regions.add(ri)
            t, r = self._tracks[ti], regions[ri]
            t.bbox = r.bbox
            t.history.append(r)
            t.hits.append(True)
            t.missed = 0

        # Track không được ghép
        for ti, t in enumerate(self._tracks):
            t.age += 1
            if ti not in matched_tracks:
                t.missed += 1
                t.hits.append(False)
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        # Vùng mới -> track mới
        for ri, r in enumerate(regions):
            if ri in matched_regions:
                continue
            self._tracks.append(Track(
                track_id=self._next_id, label=r.label, bbox=r.bbox,
                history=deque([r], maxlen=self.buffer_size),
                hits=deque([True], maxlen=self.buffer_size), age=1,
            ))
            self._next_id += 1

        return self.tracks
