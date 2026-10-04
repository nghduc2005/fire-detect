"""Interface chung cho mọi bộ phát hiện (xử lý ảnh truyền thống / học sâu / hybrid).

App, CLI và module đánh giá chỉ phụ thuộc vào interface này, nhờ đó có thể
thay bộ phát hiện mà không sửa phần còn lại.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.structures import Detection


class BaseDetector(ABC):
    #: Ảnh trung gian (mask, heatmap...) của frame gần nhất, phục vụ hiển thị/debug.
    debug: dict[str, np.ndarray]

    def __init__(self) -> None:
        self.debug = {}

    @abstractmethod
    def process(self, frame: np.ndarray) -> list[Detection]:
        """Nhận 1 frame BGR, trả về danh sách phát hiện (toạ độ theo frame gốc)."""

    def reset(self) -> None:
        """Xóa trạng thái (nền, track...) trước khi chạy video mới."""
