"""
Theo dõi vùng ứng viên qua các frame (greedy IoU matching).

.. warning::
    Đây là **STUB tạm thời** do TV1 cung cấp. **TV3 thay thế bằng cài
    đặt thực** ở ``region_tracker.py`` theo chữ ký interface:

    - :class:`RegionTracker`
        - ``__init__(cfg)``
        - ``update(regions) -> list[Region]``
        - ``reset()``

Nhiệm vụ của tracker thực:
1. Ghép bbox mới với track cũ theo IoU (greedy / Hungarian).
2. Gán ``track_id`` ổn định qua các frame.
3. Giữ buffer lịch sử ``buffer_size`` frame cho mỗi track → tầng
   temporal (flicker, biến thiên diện tích) đọc từ ``region.history``.
4. Xóa track mất dấu quá ``max_missed`` frame.
"""

from __future__ import annotations

from itertools import count
from typing import Any, List, Mapping, Optional

from ..detectors.base import Region

__all__ = ["RegionTracker"]


class RegionTracker:
    """STUB: gán ID tăng dần, không ghép qua frame.

    Pipeline vẫn chạy nhưng các đặc trưng thời gian sẽ không có lịch sử
    cho tới khi TV3 cài đặt tracker thật.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.iou_threshold = float(self.cfg.get("iou_threshold", 0.1))
        self.max_missed = int(self.cfg.get("max_missed", 5))
        self.buffer_size = int(self.cfg.get("buffer_size", 30))
        self._next_id = count(start=1)
        # TODO (TV3): self._tracks: dict[int, Region] = {}

    def update(self, regions: List[Region]) -> List[Region]:
        """Nhận regions mới, trả về regions đã gán ``track_id``."""
        for r in regions:
            # STUB: mỗi lần gọi gán ID mới. TV3 thay bằng ghép IoU.
            r.track_id = next(self._next_id)
        return regions

    def reset(self) -> None:
        self._next_id = count(start=1)
