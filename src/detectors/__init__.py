"""Bộ phát hiện khói / lửa và interface chung cho mọi detector."""

from __future__ import annotations

from typing import Any

from .base import BaseDetector, Detection, Region

__all__ = ["BaseDetector", "Detection", "Region", "ClassicalDetector"]


def __getattr__(name: str) -> Any:
    """Nạp ``ClassicalDetector`` theo kiểu lazy để tránh vòng import.

    ``classical`` kéo theo ``filtering``/``temporal``/``motion``; nếu import
    ngay ở đây thì việc ``import src.filtering`` hoặc ``import src.temporal``
    trực tiếp sẽ rơi vào vòng lặp import. Chỉ nạp khi thực sự được hỏi tới.
    """
    if name == "ClassicalDetector":
        from .classical import ClassicalDetector

        return ClassicalDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
