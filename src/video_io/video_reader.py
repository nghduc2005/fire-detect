"""Đọc frame từ file video hoặc camera."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Bọc `cv2.VideoCapture`, cho phép duyệt bằng vòng for.

    >>> with VideoReader("data/samples/synthetic_fire.mp4") as reader:
    ...     for idx, frame in reader:
    ...         ...
    """

    def __init__(self, source: str | int | Path, max_frames: int | None = None):
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # "0" -> webcam 0
        self.source = source
        self.max_frames = max_frames
        self.cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not self.cap.isOpened():
            raise IOError(f"Không mở được nguồn video: {source}")

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1 else 25.0

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        idx = 0
        while self.max_frames is None or idx < self.max_frames:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
