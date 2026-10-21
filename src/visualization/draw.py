"""
Hàm vẽ bounding box, nhãn và lưới mask debug.

Dùng chung cho CLI (``scripts/run_video.py``) và app Streamlit.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import cv2
import numpy as np

from ..detectors.base import Detection

__all__ = [
    "LABEL_COLORS",
    "draw_detections",
    "draw_fps_overlay",
    "mask_to_bgr",
    "make_debug_grid",
]


# Màu BGR dễ nhận biết: lửa đỏ cam, khói xanh dương.
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "fire": (0, 0, 255),      # đỏ
    "smoke": (255, 180, 0),   # xanh dương / cyan
}
_DEFAULT_COLOR = (0, 255, 0)  # xanh lá dự phòng


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    thickness: int = 2,
    show_score: bool = True,
    show_track_id: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Vẽ bbox + nhãn cho mọi detection lên frame.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR; nếu ``inplace=False`` sẽ vẽ lên bản copy.
    detections : Iterable[Detection]
        Danh sách phát hiện.
    thickness : int
        Độ dày viền bbox.
    show_score : bool
        Có hiển thị score trên nhãn hay không.
    show_track_id : bool
        Có hiển thị track_id hay không.
    inplace : bool
        Ghi đè trực tiếp lên frame đầu vào.

    Returns
    -------
    np.ndarray
        Frame đã vẽ.
    """
    out = frame if inplace else frame.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color = LABEL_COLORS.get(det.label.lower(), _DEFAULT_COLOR)
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)

        parts = [det.label.upper()]
        if show_track_id and det.track_id >= 0:
            parts.append(f"#{det.track_id}")
        if show_score:
            parts.append(f"{det.score:.2f}")
        text = " ".join(parts)

        # Nền nhãn
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, int(y) - 4)
        cv2.rectangle(
            out,
            (int(x), y_text - th - base),
            (int(x) + tw + 4, y_text),
            color,
            -1,
        )
        cv2.putText(
            out,
            text,
            (int(x) + 2, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_fps_overlay(
    frame: np.ndarray,
    fps: float,
    detection_count: int = 0,
    inplace: bool = False,
) -> np.ndarray:
    """Overlay góc trái trên: FPS + số detection hiện tại."""
    out = frame if inplace else frame.copy()
    text = f"FPS: {fps:5.1f}  |  det: {detection_count}"
    cv2.putText(
        out,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def mask_to_bgr(mask: np.ndarray, title: Optional[str] = None) -> np.ndarray:
    """Convert mask 1-kênh thành ảnh BGR 3-kênh (tùy chọn gắn tiêu đề góc).

    - Mask ``None`` hoặc rỗng → trả ảnh đen trong 3 kênh.
    """
    if mask is None or mask.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if mask.ndim == 2:
        bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        bgr = mask
    if title:
        cv2.putText(
            bgr,
            title,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return bgr


def make_debug_grid(
    main_frame: np.ndarray,
    masks: Mapping[str, np.ndarray],
    tile_width: int = 320,
) -> np.ndarray:
    """Ghép ảnh gốc + các mask thành lưới 2 hàng để debug trực quan.

    Hàng 1: frame đã vẽ bbox (được tự resize cho khớp cột đầu tiên).
    Hàng 2: các mask trung gian (fire_color, smoke_color, motion, ...).

    Thiết kế đơn giản để hiển thị trong Streamlit / cửa sổ OpenCV.
    """
    if main_frame is None:
        raise ValueError("main_frame là None")

    def _resize_to_width(img: np.ndarray, w: int) -> np.ndarray:
        h = max(1, int(img.shape[0] * w / img.shape[1]))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    tiles = []
    for name, m in masks.items():
        tile = mask_to_bgr(m, title=name)
        tiles.append(_resize_to_width(tile, tile_width))

    if not tiles:
        return main_frame

    # Hàng 2: nối các tile theo chiều ngang.
    row2 = np.hstack(tiles)
    # Hàng 1: resize main_frame cùng chiều rộng với hàng 2.
    row1 = _resize_to_width(main_frame, row2.shape[1])

    # Hai hàng có thể cao khác nhau → đệm thêm viền đen cho khớp.
    h = row1.shape[0] + row2.shape[0]
    canvas = np.zeros((h, row2.shape[1], 3), dtype=np.uint8)
    canvas[: row1.shape[0]] = row1
    canvas[row1.shape[0] :] = row2
    return canvas
