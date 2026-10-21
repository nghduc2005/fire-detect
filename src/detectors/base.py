"""
Interface và các kiểu dữ liệu chia sẻ giữa các module.

Thiết kế dạng *interface* là **điểm mở rộng cho học sâu**:
sau này ``YoloDetector`` hay ``HybridDetector`` chỉ cần kế thừa
:class:`BaseDetector` và cài đặt ``process(frame)``; mọi phần dùng chung
(app Streamlit, CLI, evaluation) đều hoạt động không cần sửa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["Detection", "Region", "BaseDetector"]


# ====================================================================== #
# Kiểu dữ liệu
# ====================================================================== #
@dataclass
class Detection:
    """Một phát hiện cuối cùng xuất ra khỏi detector.

    Attributes
    ----------
    bbox : tuple
        ``(x, y, w, h)`` — bounding box trên hệ tọa độ frame đã resize.
    label : str
        ``"fire"`` hoặc ``"smoke"``.
    score : float
        Điểm tin cậy trong khoảng ``[0.0, 1.0]``.
    track_id : int
        ID của track mà detection này thuộc về (``-1`` nếu không có).
    extras : dict
        Thông tin phụ (các điểm thành phần, feature debug, ...).
    """

    bbox: Tuple[int, int, int, int]
    label: str
    score: float
    track_id: int = -1
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Region:
    """Vùng ứng viên trong pipeline (sau connected-components, trước scoring).

    Dùng chung giữa các module ``filtering``, ``tracking``, ``temporal``.

    Attributes
    ----------
    bbox : tuple
        ``(x, y, w, h)`` trên hệ tọa độ frame đã resize.
    area : int
        Diện tích vùng (số pixel foreground).
    centroid : tuple
        ``(cx, cy)`` — tọa độ trọng tâm (float).
    label : str
        ``"fire"`` hoặc ``"smoke"`` — xác định ngay từ lúc trích xuất
        từ mask màu tương ứng.
    mask : np.ndarray | None
        Mask nhị phân của riêng vùng này trên khung chữ nhật bbox
        (hoặc trên toàn frame, tùy module), hoặc ``None`` nếu không giữ.
    track_id : int
        ID track do :class:`RegionTracker` gán (``-1`` nếu chưa ghép track).
    history : list
        Buffer lịch sử các frame trước — mỗi phần tử là dict chứa
        ``{"bbox", "area", "centroid", "intensity_mean", ...}``.
    extras : dict
        Thông tin phụ do các tầng sau bổ sung (flicker score, LBP, ...).
    """

    bbox: Tuple[int, int, int, int]
    area: int
    centroid: Tuple[float, float]
    label: str
    mask: Optional[np.ndarray] = None
    track_id: int = -1
    history: List[Dict[str, Any]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


# ====================================================================== #
# Interface
# ====================================================================== #
class BaseDetector(ABC):
    """Interface cho mọi detector trong hệ thống.

    Lớp con bắt buộc cài đặt :meth:`process`. Có thể override :meth:`reset`
    nếu detector giữ trạng thái qua các frame (nền MOG2, tracker, buffer).
    """

    @abstractmethod
    def process(self, frame: np.ndarray) -> List[Detection]:
        """Xử lý **một frame BGR nguyên bản** và trả về danh sách phát hiện."""

    def reset(self) -> None:
        """Reset trạng thái nội bộ (gọi khi bắt đầu video mới)."""
        # Mặc định không có gì để reset — detector stateless.
        return None

    # Cho phép gọi detector(frame) ngắn gọn.
    def __call__(self, frame: np.ndarray) -> List[Detection]:
        return self.process(frame)
