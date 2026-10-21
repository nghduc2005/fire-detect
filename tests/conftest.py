"""Fixtures dùng chung cho pytest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def random_frame() -> np.ndarray:
    """Ảnh BGR 480x640x3 ngẫu nhiên (reproducible)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def tiny_frame() -> np.ndarray:
    """Ảnh BGR nhỏ 60x80 — dùng cho test tốc độ."""
    rng = np.random.default_rng(seed=1)
    return rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)


@pytest.fixture
def tmp_video_path(tmp_path: Path, random_frame: np.ndarray) -> Path:
    """Tạo file video ngắn (10 frame) trong tmp_path."""
    import cv2

    path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = random_frame.shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (w, h))
    assert writer.isOpened(), "Không mở được VideoWriter trong fixture (codec?)"
    for i in range(10):
        frame = random_frame.copy()
        # Đánh dấu số frame vào góc để video "có nội dung".
        cv2.putText(frame, f"frame {i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path
