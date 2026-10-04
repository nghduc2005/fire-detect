"""Ghi frame ra file video."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Tự khởi tạo `cv2.VideoWriter` theo kích thước frame đầu tiên."""

    def __init__(self, path: str | Path, fps: float = 25.0, fourcc: str = "mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, (w, h))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
