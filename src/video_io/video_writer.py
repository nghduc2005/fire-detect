"""
Trình ghi video khớp fps / kích thước của frame.

Thiết kế theo kiểu *lazy init*: ``VideoWriter`` chỉ cấp phát writer của OpenCV
khi ``write()`` được gọi lần đầu, nên không cần biết kích thước trước — rất
tiện khi pipeline có thể resize frame theo cấu hình.

Ví dụ::

    with VideoWriter("outputs/result.mp4", fps=reader.fps) as writer:
        for frame in reader:
            writer.write(process(frame))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

__all__ = ["VideoWriter"]


class VideoWriter:
    """Wrapper lazy-init quanh ``cv2.VideoWriter``."""

    def __init__(
        self,
        path: Union[str, Path],
        fps: float = 25.0,
        fourcc: str = "mp4v",
    ) -> None:
        self.path: Path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps: float = float(fps) if fps and fps > 0 else 25.0
        self.fourcc_code: str = fourcc
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)

        self.writer: Optional[cv2.VideoWriter] = None
        self.size: Optional[Tuple[int, int]] = None  # (width, height)
        self._frames_written: int = 0

    # ------------------------------------------------------------------ #
    def _ensure_writer(self, frame: np.ndarray) -> None:
        """Khởi tạo ``cv2.VideoWriter`` theo kích thước của frame đầu tiên."""
        if self.writer is not None:
            return
        if frame is None or frame.size == 0:
            raise ValueError("Không thể khởi tạo writer từ frame rỗng.")

        h, w = frame.shape[:2]
        self.size = (int(w), int(h))
        self.writer = cv2.VideoWriter(
            str(self.path), self._fourcc, self.fps, self.size
        )
        if not self.writer.isOpened():
            raise IOError(
                f"Không mở được VideoWriter cho {self.path} "
                f"(fourcc={self.fourcc_code}, fps={self.fps}, size={self.size})."
            )

    def write(self, frame: np.ndarray) -> None:
        """Ghi một frame BGR."""
        if frame is None:
            raise ValueError("Không được ghi frame None.")
        self._ensure_writer(frame)

        # Khớp kích thước — nếu người gọi lỡ truyền frame khác size, resize về size đầu tiên.
        if self.size is not None and (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)

        self.writer.write(frame)
        self._frames_written += 1

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:  # pragma: no cover
            pass
