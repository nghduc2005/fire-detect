"""
Trình đọc video / camera dạng iterator trả về từng frame BGR.

Nguồn (``source``) được chấp nhận:
- ``int``      : chỉ số camera (0, 1, ...).
- ``str | Path``: đường dẫn tới file video (mp4, avi, mov, ...).
- ``str``      : URL stream RTSP / HTTP mà OpenCV hỗ trợ.

Ví dụ::

    with VideoReader("data/samples/fire.mp4") as reader:
        print(reader.fps, reader.width, reader.height)
        for frame in reader:
            cv2.imshow("preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

Source = Union[int, str, Path]


class VideoReader:
    """Wrapper mỏng quanh ``cv2.VideoCapture`` theo hướng Pythonic."""

    def __init__(self, source: Source) -> None:
        self.source: Source = source

        # OpenCV phân biệt int (camera) và str (file / URL).
        src = source if isinstance(source, int) else str(source)
        self.cap: cv2.VideoCapture = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise IOError(f"Không mở được nguồn video: {source!r}")

    # ------------------------------------------------------------------ #
    # Thuộc tính thiết yếu
    # ------------------------------------------------------------------ #
    @property
    def fps(self) -> float:
        """FPS khai báo trong metadata; mặc định 25.0 nếu không đọc được."""
        f = self.cap.get(cv2.CAP_PROP_FPS)
        return float(f) if f and f > 0 else 25.0

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)``."""
        return self.width, self.height

    @property
    def frame_count(self) -> int:
        """Tổng số frame; trả về ``-1`` nếu không biết (camera, stream)."""
        c = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return c if c > 0 else -1

    @property
    def position(self) -> int:
        """Chỉ số frame hiện tại (0-based)."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def is_opened(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    # ------------------------------------------------------------------ #
    # Đọc frame
    # ------------------------------------------------------------------ #
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Đọc một frame. Trả về ``(ok, frame)`` giống ``cv2.VideoCapture.read``."""
        ok, frame = self.cap.read()
        return bool(ok), frame if ok else None

    def seek(self, frame_index: int) -> None:
        """Nhảy tới frame chỉ định (chỉ hỗ trợ với file video, không phải camera)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        ok, frame = self.read()
        if not ok or frame is None:
            raise StopIteration
        return frame

    # ------------------------------------------------------------------ #
    # Giải phóng tài nguyên
    # ------------------------------------------------------------------ #
    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        # Không để lộ lỗi khi interpreter đang shutdown.
        try:
            self.release()
        except Exception:  # pragma: no cover
            pass
