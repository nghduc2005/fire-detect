"""Bộ phát hiện khói / lửa và interface chung cho mọi detector."""

from .base import BaseDetector, Detection, Region
from .classical import ClassicalDetector

__all__ = ["BaseDetector", "Detection", "Region", "ClassicalDetector"]
